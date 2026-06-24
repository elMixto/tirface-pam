use std::sync::{Arc, Mutex};
use log::{debug, error, info, warn};
use zbus::interface;
use zbus::fdo;
use zbus::message::Header;
use zbus::Connection;

use crate::config::Config;
use crate::camera::{Camera, CameraManager};
use crate::models::{FaceRecognizer, FaceDetector, rustface::RustfaceDetector};
use crate::constants::AUTH_TIMEOUT_SEC;
use crate::db::{EmbeddingDatabase, save_embedding, remove_embedding, list_user_embeddings};

pub struct TirfacePamDbus {
    pub auth_resource: Arc<Mutex<Box<dyn FaceRecognizer + Send>>>,
    pub config: Config,
    pub key: crate::crypto::FaceCrypto,
    pub ir_camera: Camera,
}

async fn get_caller_uid(sender_str: &str, conn: &Connection) -> fdo::Result<u32> {
    let dbus_proxy = fdo::DBusProxy::new(conn)
        .await
        .map_err(|e| fdo::Error::Failed(e.to_string()))?;
    let bus_name = zbus::names::BusName::try_from(sender_str)
        .map_err(|e| fdo::Error::Failed(e.to_string()))?;
    dbus_proxy
        .get_connection_unix_user(bus_name)
        .await
        .map_err(|e| fdo::Error::Failed(e.to_string()))
}

fn uid_for_name(name: &str) -> Option<u32> {
    use std::ffi::CString;
    let name_c = CString::new(name).ok()?;
    unsafe {
        let pwd = libc::getpwnam(name_c.as_ptr());
        if !pwd.is_null() {
            Some((*pwd).pw_uid)
        } else {
            None
        }
    }
}

fn perform_authenticate_user(
    username: &str,
    recognizer: &mut dyn FaceRecognizer,
    detector: &mut dyn FaceDetector,
    key: &crate::crypto::FaceCrypto,
    config: &Config,
    camera_info: &Camera,
) -> bool {
    let t_start = std::time::Instant::now();

    let user_db = EmbeddingDatabase::load_for_user(username, key, config);
    debug!("load_for_user took: {:?}", t_start.elapsed());

    if user_db.is_empty() {
        info!("No signatures found for user: {}", username);
        return false;
    }

    let t_before_cam = std::time::Instant::now();
    let camera = match CameraManager::start(camera_info) {
        Ok(c) => c,
        Err(e) => {
            error!("Camera initialization failed: {}", e);
            return false;
        }
    };
    debug!("Camera init and warmup took: {:?}", t_before_cam.elapsed());

    let start_time = std::time::Instant::now();
    let mut frames_processed = 0;

    while start_time.elapsed().as_secs() < AUTH_TIMEOUT_SEC {
        let buf = match camera.get_latest_frame() {
            Some(b) => b,
            None => {
                std::thread::sleep(std::time::Duration::from_millis(10));
                continue;
            }
        };

        let t_detect_start = std::time::Instant::now();
        let faces = detector
            .detect(&buf, camera.width, camera.height)
            .unwrap_or_default();
        let t_detect = t_detect_start.elapsed();

        let best_face = faces.iter().max_by_key(|f| f.width * f.height);

        if let Some(face) = best_face {
            let t_process_start = std::time::Instant::now();

            if let Ok(emb) = recognizer.get_embedding(&buf, camera.width, camera.height, *face) {
                let t_process = t_process_start.elapsed();
                let mut best_score = 0.0;

                for db_emb in &user_db {
                    let score = recognizer.compare(&emb, db_emb);
                    if score > best_score {
                        best_score = score;
                    }
                }

                debug!(
                    "Frame {} -> detect:{:?}, onnx:{:?}, score: {:.3}",
                    frames_processed, t_detect, t_process, best_score
                );

                if best_score >= config.recognition.threshold {
                    info!("Authentication SUCCESS for {} (Score: {:.3})", username, best_score);
                    info!("Total authentication time: {:?}", t_start.elapsed());
                    return true;
                }
            } else {
                debug!("Frame {} -> detect:{:?}, onnx:FAILED", frames_processed, t_detect);
            }
        }

        frames_processed += 1;
    }

    info!("Authentication FAILED for {} (Timeout)", username);
    info!("Total authentication time (Timeout): {:?}", t_start.elapsed());
    false
}

#[interface(name = "org.freedesktop.TirfacePam1")]
impl TirfacePamDbus {
    async fn verify(
        &self,
        #[zbus(header)] header: Header<'_>,
        #[zbus(connection)] conn: &Connection,
        user: String,
    ) -> fdo::Result<bool> {
        info!("D-Bus Verify requested for user: {}", user);

        // Access Control Verification
        if let Some(sender) = header.sender() {
            let caller_uid = get_caller_uid(sender.as_str(), conn).await?;
            if caller_uid != 0 {
                match uid_for_name(&user) {
                    Some(expected_uid) if caller_uid == expected_uid => {}
                    _ => {
                        warn!("D-Bus Verify AccessDenied: Caller UID {} is not permitted to verify user '{}'", caller_uid, user);
                        return Err(fdo::Error::AccessDenied(format!(
                            "Caller UID {} is not permitted to verify user '{}'", caller_uid, user
                        )));
                    }
                }
            }
        }

        let auth_resource = self.auth_resource.clone();
        let config = self.config.clone();
        let key = self.key.clone();
        let ir_camera = self.ir_camera.clone();
        let user_clone = user.clone();

        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let mut detector = match RustfaceDetector::new(&config.models.detector_path) {
                Ok(d) => d,
                Err(e) => {
                    error!("Failed to load face detector: {}", e);
                    let _ = tx.send(false);
                    return;
                }
            };

            let res = match auth_resource.try_lock() {
                Ok(mut session) => {
                    perform_authenticate_user(
                        &user_clone,
                        &mut **session,
                        &mut detector,
                        &key,
                        &config,
                        &ir_camera,
                    )
                }
                Err(_) => {
                    warn!("Authentication busy. Another request is currently active. Rejecting request for {}", user_clone);
                    false
                }
            };
            let _ = tx.send(res);
        });

        let result = rx.recv().map_err(|e| fdo::Error::Failed(e.to_string()))?;
        Ok(result)
    }

    async fn status(&self) -> fdo::Result<String> {
        let status_json = serde_json::json!({
            "service": "TirfacePam1",
            "active_model": self.config.models.model_name(),
            "threshold": self.config.recognition.threshold,
            "timeout_sec": AUTH_TIMEOUT_SEC,
        });
        Ok(status_json.to_string())
    }

    async fn enroll(
        &self,
        #[zbus(header)] header: Header<'_>,
        #[zbus(connection)] conn: &Connection,
        user: String,
        label: String,
        embedding: Vec<f32>,
    ) -> fdo::Result<bool> {
        info!("D-Bus Enroll requested for user: {}, label: {}", user, label);

        // MUTATION Access Control: ONLY root is permitted to enroll!
        if let Some(sender) = header.sender() {
            let caller_uid = get_caller_uid(sender.as_str(), conn).await?;
            if caller_uid != 0 {
                return Err(fdo::Error::AccessDenied(
                    "Only root (UID 0) is permitted to enroll new faces".to_string()
                ));
            }
        }

        let key = self.key.clone();
        let config = self.config.clone();

        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let res = save_embedding(&user, &label, &config.models.model_name(), &embedding, &key)
                .map(|_| true)
                .unwrap_or(false);
            let _ = tx.send(res);
        });

        rx.recv().map_err(|e| fdo::Error::Failed(e.to_string()))
    }

    async fn remove_model(
        &self,
        #[zbus(header)] header: Header<'_>,
        #[zbus(connection)] conn: &Connection,
        user: String,
        id_or_label: String,
    ) -> fdo::Result<bool> {
        info!("D-Bus RemoveModel requested for user: {}, id/label: {}", user, id_or_label);

        // MUTATION Access Control: ONLY root is permitted!
        if let Some(sender) = header.sender() {
            let caller_uid = get_caller_uid(sender.as_str(), conn).await?;
            if caller_uid != 0 {
                return Err(fdo::Error::AccessDenied(
                    "Only root (UID 0) is permitted to remove face models".to_string()
                ));
            }
        }

        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let res = remove_embedding(&user, &id_or_label).unwrap_or(false);
            let _ = tx.send(res);
        });

        rx.recv().map_err(|e| fdo::Error::Failed(e.to_string()))
    }

    async fn list_models(
        &self,
        #[zbus(header)] header: Header<'_>,
        #[zbus(connection)] conn: &Connection,
        user: String,
    ) -> fdo::Result<String> {
        info!("D-Bus ListModels requested for user: {}", user);

        // MUTATION/ADMIN Access Control: ONLY root is permitted!
        if let Some(sender) = header.sender() {
            let caller_uid = get_caller_uid(sender.as_str(), conn).await?;
            if caller_uid != 0 {
                return Err(fdo::Error::AccessDenied(
                    "Only root (UID 0) is permitted to list face models".to_string()
                ));
            }
        }

        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let res = list_user_embeddings(&user).unwrap_or_default();
            let _ = tx.send(res);
        });

        let models = rx.recv().map_err(|e| fdo::Error::Failed(e.to_string()))?;
        serde_json::to_string(&models).map_err(|e| fdo::Error::Failed(e.to_string()))
    }
}
