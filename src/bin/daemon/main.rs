use log::{error, info};

use pam_tirface_pam::camera::{Camera, CameraType};
use pam_tirface_pam::config::Config;
use pam_tirface_pam::crypto;
use pam_tirface_pam::models;
use pam_tirface_pam::daemon::dbus::TirfacePamDbus;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    info!("Initializing Tirface PAM Daemon (D-Bus System Bus)...");

    let config = Config::load();

    let ir_camera = match Camera::new(&config.camera.ir_device, CameraType::Ir) {
        Ok(c) => c,
        Err(e) => {
            error!("Failed to initialize IR camera: {}", e);
            std::process::exit(1);
        }
    };

    let key = match crypto::FaceCrypto::load_or_create() {
        Ok(k) => k,
        Err(e) => {
            error!("Fatal error with master key: {}", e);
            std::process::exit(1);
        }
    };

    let session = match models::load_recognizer(&config.models) {
        Ok(s) => s,
        Err(e) => {
            error!("Failed to load face recognition model: {}", e);
            std::process::exit(1);
        }
    };

    let auth_resource = std::sync::Arc::new(std::sync::Mutex::new(session));

    let dbus_service = TirfacePamDbus {
        auth_resource,
        config: config.clone(),
        key,
        ir_camera,
    };

    info!("Registering org.freedesktop.TirfacePam1 D-Bus service...");

    let _conn = zbus::connection::Builder::system()?
        .name("org.freedesktop.TirfacePam1")?
        .serve_at("/org/freedesktop/TirfacePam1", dbus_service)?
        .build()
        .await?;

    info!("Tirface PAM Daemon successfully registered on system D-Bus at /org/freedesktop/TirfacePam1!");

    // Hold connection alive
    loop {
        tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
    }
}
