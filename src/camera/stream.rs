use crate::camera::device::{Camera, CameraType};
use crate::camera::ir_emitter::IrEmitter;
use crate::constants::{DARK_FRAME_BRIGHTNESS_THRESHOLD, DARK_FRAME_SAMPLE_SIZE};
use crate::error::FacePamError;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use v4l::buffer::Type;
use v4l::io::traits::CaptureStream;
use v4l::prelude::*;
use v4l::video::Capture;

pub struct CameraManager {
    keep_running: Arc<AtomicBool>,
    latest_frame: Arc<Mutex<Option<Vec<u8>>>>,
    thread_handle: Option<thread::JoinHandle<()>>,
    pub width: u32,
    pub height: u32,
    emitter: Option<IrEmitter>,
}

impl CameraManager {
    pub fn start(camera: &Camera) -> Result<Self, FacePamError> {
        let actual_path = &camera.path;
        let is_dummy = actual_path == "dummy" || actual_path.starts_with("dummy:");

        if is_dummy {
            let keep_running = Arc::new(AtomicBool::new(true));
            let thread_keep_running = keep_running.clone();
            let latest_frame = Arc::new(Mutex::new(None));
            let thread_latest_frame = latest_frame.clone();

            // Determine which set of images to load
            let frames_dir = if actual_path.contains("inf") {
                "tests/data/identities/person_a/inf"
            } else {
                "tests/data/identities/person_a/enroll"
            };

            // Read sorted folder images for the loop
            let mut frames = Vec::new();
            let mut width = 640;
            let mut height = 360;

            if let Ok(entries) = std::fs::read_dir(frames_dir) {
                let mut paths: Vec<_> = entries.filter_map(Result::ok).map(|e| e.path()).collect();
                paths.sort();

                for path in paths {
                    if path.extension().and_then(|s| s.to_str()) == Some("png") {
                        if let Ok(img) = image::open(&path) {
                            let luma = img.into_luma8();
                            width = luma.width();
                            height = luma.height();
                            let raw = luma.into_raw();
                            // Aplicar la pipeline de preprocesamiento de imagen (ecualización)
                            let equalized = crate::camera::equalize_grayscale_image(&raw);
                            frames.push(equalized);
                        }
                    }
                }
            }

            if frames.is_empty() {
                // Crear un frame de respaldo si no hay imágenes
                frames.push(vec![128u8; (width * height) as usize]);
            }

            let thread_handle = thread::spawn(move || {
                let mut idx = 0;
                while thread_keep_running.load(Ordering::Relaxed) {
                    let frame = &frames[idx % frames.len()];
                    if let Ok(mut frame_lock) = thread_latest_frame.lock() {
                        *frame_lock = Some(frame.clone());
                    }
                    idx += 1;
                    // Simular 15 FPS
                    thread::sleep(std::time::Duration::from_millis(66));
                }
            });

            return Ok(Self {
                keep_running,
                latest_frame,
                thread_handle: Some(thread_handle),
                width,
                height,
                emitter: None,
            });
        }

        let emitter = if camera.camera_type == CameraType::Ir {
            let em = IrEmitter::for_device(actual_path);
            if let Some(ref e) = em {
                let _ = e.activate();
                // AGC stabilization: wait 100ms
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            em
        } else {
            None
        };

        let dev = Device::with_path(actual_path).map_err(|e| {
            FacePamError::Camera(format!("Failed to open camera device {}: {}", actual_path, e))
        })?;

        let mut format = dev
            .format()
            .map_err(|e| FacePamError::Camera(format!("Failed to get camera format: {}", e)))?;

        match camera.camera_type {
            CameraType::Ir => {
                format.fourcc = v4l::FourCC::new(b"GREY");
            }
            CameraType::Rgb => {
                format.fourcc = v4l::FourCC::new(b"YUYV");
                format.width = 640;
                format.height = 480;
            }
        }

        // Try to set format
        let _ = dev.set_format(&format);

        let actual_format = dev.format().map_err(|e| {
            FacePamError::Camera(format!("Failed to verify camera format: {}", e))
        })?;

        let mut stream = MmapStream::with_buffers(&dev, Type::VideoCapture, 4).map_err(|e| {
            FacePamError::Camera(format!("Failed to create capture stream: {}", e))
        })?;

        // WARMUP: Discard the first 2 frames for minimal initialization
        for _ in 0..2 {
            let _ = stream.next();
        }

        let keep_running = Arc::new(AtomicBool::new(true));
        let thread_keep_running = keep_running.clone();

        let latest_frame = Arc::new(Mutex::new(None));
        let thread_latest_frame = latest_frame.clone();

        let camera_type = camera.camera_type;

        let thread_handle = thread::spawn(move || {
            while thread_keep_running.load(Ordering::Relaxed) {
                if let Ok((buf, _)) = stream.next() {
                    let mut should_update = true;

                    if camera_type == CameraType::Ir {
                        let mut brightness: u32 = 0;
                        let step = (buf.len() / DARK_FRAME_SAMPLE_SIZE).max(1);

                        for i in (0..buf.len()).step_by(step).take(DARK_FRAME_SAMPLE_SIZE) {
                            brightness += buf[i] as u32;
                        }

                        let avg_brightness = brightness / DARK_FRAME_SAMPLE_SIZE as u32;

                        if avg_brightness <= DARK_FRAME_BRIGHTNESS_THRESHOLD {
                            should_update = false;
                        }
                    }

                    if should_update {
                        if let Ok(mut frame_lock) = thread_latest_frame.lock() {
                            let processed_buf = if camera_type == CameraType::Ir {
                                crate::camera::equalize_grayscale_image(buf)
                            } else {
                                buf.to_vec()
                            };
                            *frame_lock = Some(processed_buf);
                        }
                    }
                }
            }
        });

        Ok(Self {
            keep_running,
            latest_frame,
            thread_handle: Some(thread_handle),
            width: actual_format.width,
            height: actual_format.height,
            emitter,
        })
    }

    pub fn get_latest_frame(&self) -> Option<Vec<u8>> {
        if let Ok(mut frame_lock) = self.latest_frame.lock() {
            frame_lock.take()
        } else {
            None
        }
    }

    pub fn stop(&mut self) {
        self.keep_running.store(false, Ordering::Relaxed);
        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }
        if let Some(ref em) = self.emitter {
            let _ = em.deactivate().ok();
        }
    }
}

impl Drop for CameraManager {
    fn drop(&mut self) {
        self.stop();
    }
}
