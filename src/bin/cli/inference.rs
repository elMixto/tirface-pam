use crate::utils::CameraWidget;
use crossterm::{
    event::{Event, KeyCode, poll, read},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use log::info as log;
use pam_tirface_pam::{
    config::{Backend, Config, RecognizerModel, Runtime},
    crypto,
    db::EmbeddingDatabase,
    models::{
        FaceDetector, rustface::RustfaceDetector,
    },
};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
};
use std::io::stdout;
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::{Duration, Instant};

// --- TEST INFERENCE LOGIC ---
pub fn run_test(config: &Config) -> std::io::Result<()> {
    enable_raw_mode()?;
    let mut stdout_handle = stdout();
    execute!(stdout_handle, EnterAlternateScreen)?;

    let backend = CrosstermBackend::new(stdout_handle);
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;

    let key = match crypto::FaceCrypto::load_or_create() {
        Ok(k) => k,
        Err(e) => {
            disable_raw_mode()?;
            execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
            terminal.show_cursor()?;
            log!("Error initializing Master Key: {}", e);
            return Ok(());
        }
    };

    let mut session = match pam_tirface_pam::models::load_recognizer(&config.models) {
        Ok(s) => s,
        Err(e) => {
            log!("Failed to load face recognition model: {}", e);
            std::process::exit(1);
        }
    };

    let database = EmbeddingDatabase::load_all(&key, config);
    let data_dir_str = pam_tirface_pam::paths::SYSTEM_ENROLL_DIR;
    log!(
        "Database loaded from {}. Signatures found: {}",
        data_dir_str,
        database.len()
    );
    thread::sleep(std::time::Duration::from_secs(2));

    let ir_camera = match pam_tirface_pam::camera::Camera::new(
        &config.camera.ir_device,
        pam_tirface_pam::camera::CameraType::Ir,
    ) {
        Ok(c) => c,
        Err(e) => {
            disable_raw_mode()?;
            execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
            terminal.show_cursor()?;
            log!("Error initializing IR camera: {}", e);
            return Ok(());
        }
    };

    let ir_path = ir_camera.path.clone();

    let camera_manager = match pam_tirface_pam::camera::CameraManager::start(&ir_camera) {
        Ok(cm) => cm,
        Err(e) => {
            disable_raw_mode()?;
            execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
            terminal.show_cursor()?;
            log!(
                "\r\n\x1B[1;31mCritical Error: The IR camera ({}) is busy or could not be started.\x1B[0m\r",
                ir_path
            );
            log!(
                "This usually happens because the authentication daemon is running in the background and using the camera.\r"
            );
            log!("Please stop the daemon before testing the camera by running:\r");
            log!("  \x1B[1;33msudo systemctl stop tirface-pam.service\x1B[0m\r");
            log!("Technical detail: {}\r", e);
            return Ok(());
        }
    };

    let (tx_frame, rx_frame) = mpsc::sync_channel::<Vec<u8>>(1);
    let shared_state: Arc<Mutex<Option<(i32, i32, u32, u32, String, f32, f32)>>> =
        Arc::new(Mutex::new(None));
    let thread_state = shared_state.clone();

    let db_clone = database.clone();
    let width_clone = camera_manager.width;
    let height_clone = camera_manager.height;

    let detector_path = config.models.detector_path.clone();
    let threshold = config.recognition.threshold;

    thread::spawn(move || {
        let mut detector = match RustfaceDetector::new(&detector_path) {
            Ok(d) => d,
            Err(_) => return,
        };

        while let Ok(gray_buf) = rx_frame.recv() {
            let start_time = Instant::now();

            let faces = detector
                .detect(&gray_buf, width_clone, height_clone)
                .unwrap_or_default();
            let best_face = faces.into_iter().max_by_key(|f| f.width * f.height);

            let mut result = None;

            if let Some(face) = best_face {
                let b = face;
                if let Ok(emb) = session.get_embedding(&gray_buf, width_clone, height_clone, b) {
                    let mut best_match = "Unknown".to_string();
                    let mut best_score = 0.0;

                    for (name, db_emb) in &db_clone.records {
                        let score = session.compare(&emb, db_emb);
                        if score > best_score {
                            best_score = score;
                            best_match = name.clone();
                        }
                    }

                    if best_score < threshold {
                        best_match = "Unknown".to_string();
                    }

                    let current_latency_ms = start_time.elapsed().as_secs_f32() * 1000.0;

                    result = Some((
                        b.x,
                        b.y,
                        b.width,
                        b.height,
                        best_match,
                        best_score,
                        current_latency_ms,
                    ));
                }
            } else {
                let current_latency_ms = start_time.elapsed().as_secs_f32() * 1000.0;
                result = Some((
                    0,
                    0,
                    0,
                    0,
                    "Waiting for face...".to_string(),
                    0.0,
                    current_latency_ms,
                ));
            }

            *thread_state.lock().unwrap_or_else(|e| e.into_inner()) = result;
        }
    });

    let mut last_ui_time = Instant::now();
    let mut ui_frames = 0;
    let mut ui_fps = 0.0;

    let mut sticky_name = "Waiting for face...".to_string();
    let mut sticky_score = 0.0;
    let mut sticky_inf_ms = 0.0;
    let mut last_positive_detection = Instant::now();

    loop {
        if poll(Duration::from_millis(5))? {
            match read()? {
                Event::Key(event)
                    if (event.code == KeyCode::Char('q') || event.code == KeyCode::Esc) => {
                        break;
                    }
                Event::Resize(_, _) => {
                    terminal.clear()?;
                }
                _ => {}
            }
        }

        let buf = match camera_manager.get_latest_frame() {
            Some(res) => res,
            None => {
                std::thread::sleep(Duration::from_millis(5));
                continue;
            }
        };

        let mut brightness: u32 = 0;
        let sample_size = 1000;
        let step = buf.len().max(sample_size) / sample_size;
        for i in (0..buf.len()).step_by(step).take(sample_size) {
            brightness += buf[i] as u32;
        }
        let avg_brightness = brightness / sample_size as u32;

        ui_frames += 1;
        if last_ui_time.elapsed().as_secs() >= 1 {
            ui_fps = ui_frames as f32 / last_ui_time.elapsed().as_secs_f32();
            ui_frames = 0;
            last_ui_time = Instant::now();
        }

        let _ = tx_frame.try_send(buf.clone());
        let current_state = shared_state.lock().map(|s| s.clone()).unwrap_or_else(|e| e.into_inner().clone());

        if let Some((_, _, _, _, ref name, score, inf_ms)) = current_state {
            sticky_inf_ms = inf_ms;
            if name != "Waiting for face..." {
                sticky_name = name.clone();
                sticky_score = score;
                last_positive_detection = Instant::now();
            } else if last_positive_detection.elapsed().as_millis() > 1000 {
                sticky_name = "Waiting for face...".to_string();
                sticky_score = score;
            }
        } else if last_positive_detection.elapsed().as_millis() > 1000 {
            sticky_name = "Waiting for face...".to_string();
            sticky_score = 0.0;
        }

        terminal.draw(|f| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(5), Constraint::Min(10)].as_ref())
                .split(f.area());

            let user_color = match sticky_name.as_str() {
                "Waiting for face..." => Color::Gray,
                "Unknown" => Color::Red,
                _ => Color::Green,
            };

            let (model_name, backend_name, device_name) = match config.models.get_recognizer_model() {
                RecognizerModel::MobileFaceNet(backend) => match backend {
                    Backend::Ort => ("mobilefacenet", "ort", "CPU"),
                    Backend::Openvino(device) => (
                        "mobilefacenet",
                        "openvino",
                        match device {
                            Runtime::Cpu => "CPU",
                            Runtime::Gpu => "GPU",
                            Runtime::Npu => "NPU",
                        },
                    ),
                },
                RecognizerModel::ArcFace(backend) => match backend {
                    Backend::Ort => ("arcface", "ort", "CPU"),
                    Backend::Openvino(device) => (
                        "arcface",
                        "openvino",
                        match device {
                            Runtime::Cpu => "CPU",
                            Runtime::Gpu => "GPU",
                            Runtime::Npu => "NPU",
                        },
                    ),
                },
            };

            let info_text = vec![
                Line::from(vec![
                    Span::styled("UI Camera: ", Style::default().fg(Color::Yellow)),
                    Span::raw(format!("{:.1} FPS | ", ui_fps)),
                    Span::styled("AI Inference: ", Style::default().fg(Color::Yellow)),
                    Span::raw(format!("{:.1} ms | ", sticky_inf_ms)),
                    Span::styled("Database: ", Style::default().fg(Color::Yellow)),
                    Span::raw(format!("{} signatures loaded", database.len())),
                ]),
                Line::from(vec![
                    Span::styled("User: ", Style::default().add_modifier(Modifier::BOLD)),
                    Span::styled(
                        sticky_name.clone(),
                        Style::default().fg(user_color).add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(format!(" (Cosine Similarity: {:.2})", sticky_score)),
                ]),
                Line::from(vec![
                    Span::styled("Config: ", Style::default().fg(Color::Yellow)),
                    Span::raw(format!(
                        "IR Camera: {} | Model: {} | Backend: {} | Device: {}",
                        ir_path, model_name, backend_name, device_name
                    )),
                ]),
            ];

            let p_info = Paragraph::new(info_text).block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" PAM Face ID - Inference (q to exit) "),
            );
            f.render_widget(p_info, chunks[0]);

            let cam_chunk = chunks[1];
            let bbox = if let Some((x, y, w, h, _, _, _)) = current_state {
                if w > 0 && h > 0 {
                    Some((x, y, w, h))
                } else {
                    None
                }
            } else {
                None
            };

            let box_color = if let Some((_, _, _, _, ref name, _, _)) = current_state {
                if name == "Unknown" {
                    (255, 0, 0)
                } else {
                    (0, 255, 0)
                }
            } else {
                (128, 128, 128)
            };

            let camera_widget = CameraWidget {
                buf_rgb: None,
                buf_ir: &buf,
                rgb_width: 0,
                rgb_height: 0,
                ir_width: camera_manager.width as usize,
                ir_height: camera_manager.height as usize,
                view_is_ir: true,
                zoom_factor: 1.0,
                current_bbox: bbox,
                box_color,
                thickness: 4,
                blend_box: true,
                title: Some(format!(" Dynamic IR Feed (Luma: {}) ", avg_brightness)),
            };
            f.render_widget(camera_widget, cam_chunk);
        })?;
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    log!("Inference finished.");
    Ok(())
}
