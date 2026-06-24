use crate::utils::{
    CAPTURE_DELAY_MS, FRAMES_PER_POSE, POSES, TARGET_FRAMES,
    CameraWidget,
};
use crossterm::{
    event::{Event, KeyCode, poll, read},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use log::info as log;
use pam_tirface_pam::{
    config::{Backend, Config, RecognizerModel, Runtime},
    crypto,
    db,
    models::{
        FaceDetector, FaceRecognizer, rustface::RustfaceDetector,
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

// --- ENROLL DATA STRUCTURES & ABSTRACTIONS ---

enum EnrollState {
    Idle,
    Active {
        captured_frames: usize,
        last_capture_time: Instant,
        wait_until: Instant,
    },
    Finished,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FaceStatus {
    NoFace,
    TooFar,
    NotCentered,
    Good,
}

impl FaceStatus {
    fn guidance_msg(&self, state: &EnrollState) -> String {
        match self {
            FaceStatus::NoFace => "No face detected".to_string(),
            FaceStatus::TooFar => "Get closer to the camera".to_string(),
            FaceStatus::NotCentered => "Center your face".to_string(),
            FaceStatus::Good => match state {
                EnrollState::Active { captured_frames, wait_until, .. } => {
                    let pose_idx = captured_frames / FRAMES_PER_POSE;
                    if Instant::now() < *wait_until {
                        format!("Prepare to look: {}", POSES[pose_idx])
                    } else {
                        format!("LOOK TOWARDS: {}", POSES[pose_idx])
                    }
                }
                EnrollState::Idle | EnrollState::Finished => {
                    "Face OK. Ready to register.".to_string()
                }
            }
        }
    }

    fn box_color(&self, state: &EnrollState) -> (u8, u8, u8) {
        match self {
            FaceStatus::NoFace => (255, 0, 0),       // Red
            FaceStatus::TooFar => (255, 165, 0),     // Orange
            FaceStatus::NotCentered => (255, 255, 0), // Yellow
            FaceStatus::Good => match state {
                EnrollState::Active { wait_until, .. } => {
                    if Instant::now() < *wait_until {
                        (0, 255, 255) // Cyan (preparing)
                    } else {
                        (0, 255, 0) // Green (active pose)
                    }
                }
                EnrollState::Idle | EnrollState::Finished => {
                    (0, 255, 0) // Green
                }
            }
        }
    }
}

fn evaluate_face(bbox: Option<(i32, i32, u32, u32)>, img_width: i32, img_height: i32) -> FaceStatus {
    let (x, y, w, h) = match bbox {
        Some(b) => b,
        None => return FaceStatus::NoFace,
    };

    let center_x = x + (w / 2) as i32;
    let center_y = y + (h / 2) as i32;
    let img_cx = img_width / 2;
    let img_cy = img_height / 2;

    let is_centered = (center_x - img_cx).abs() < 80 && (center_y - img_cy).abs() < 80;
    let is_large_enough = w > 60; // Ajustado para IR

    if !is_large_enough {
        FaceStatus::TooFar
    } else if !is_centered {
        FaceStatus::NotCentered
    } else {
        FaceStatus::Good
    }
}

struct EnrollSession {
    username: String,
    state: EnrollState,
}

impl EnrollSession {
    fn new(username: &str, _model_name: &str, headless: bool) -> std::io::Result<Self> {
        let state = if headless {
            EnrollState::Active {
                captured_frames: 0,
                last_capture_time: Instant::now(),
                wait_until: Instant::now(),
            }
        } else {
            EnrollState::Idle
        };

        Ok(Self { username: username.to_string(), state })
    }

    fn start_enrollment(&mut self) -> std::io::Result<()> {
        if matches!(self.state, EnrollState::Idle) {
            self.state = EnrollState::Active {
                captured_frames: 0,
                last_capture_time: Instant::now(),
                wait_until: Instant::now(),
            };
        }
        Ok(())
    }

    fn can_capture(&self, face_status: FaceStatus, is_bright: bool) -> bool {
        if let EnrollState::Active { last_capture_time, wait_until, .. } = self.state {
            face_status == FaceStatus::Good
                && is_bright
                && Instant::now() >= wait_until
                && last_capture_time.elapsed().as_millis() > CAPTURE_DELAY_MS
        } else {
            false
        }
    }

    fn save_capture(
        &mut self,
        buf_ir: &[u8],
        width: u32,
        height: u32,
        ir_bbox: pam_tirface_pam::models::BoundingBox,
        session: &mut dyn FaceRecognizer,
        master_key: &pam_tirface_pam::crypto::FaceCrypto,
        config: &Config,
    ) -> Result<(), String> {
        if let EnrollState::Active { captured_frames, last_capture_time, wait_until } = &mut self.state {
            match session.get_embedding(buf_ir, width, height, ir_bbox) {
                Ok(emb) => {
                    let label = format!("frame_{}", captured_frames);
                    let model_name = config.models.model_name();
                    db::save_embedding(
                        &self.username,
                        &label,
                        &model_name,
                        &emb,
                        master_key,
                    ).map_err(|e| e.to_string())?;

                    *captured_frames += 1;
                    *last_capture_time = Instant::now();

                    if *captured_frames >= TARGET_FRAMES {
                        self.state = EnrollState::Finished;
                    } else if *captured_frames % FRAMES_PER_POSE == 0 {
                        *wait_until = Instant::now() + Duration::from_millis(2000);
                    }
                    Ok(())
                }
                Err(e) => Err(format!("{:?}", e)),
            }
        } else {
            Ok(())
        }
    }
}

// --- ENROLL LOGIC ---

fn run_camera_session(
    username: &str,
    session: &mut dyn FaceRecognizer,
    master_key: &pam_tirface_pam::crypto::FaceCrypto,
    config: &Config,
    rgb_camera: &pam_tirface_pam::camera::Camera,
    ir_camera: &pam_tirface_pam::camera::Camera,
    headless: bool,
    terminal: &mut Option<Terminal<CrosstermBackend<std::io::Stdout>>>,
) -> std::io::Result<()> {
    let mut view_is_ir = false;
    let mut zoom_factor = 1.6_f32;

    let rgb_path = rgb_camera.path.clone();
    let ir_path = ir_camera.path.clone();

    let camera_manager_rgb = match pam_tirface_pam::camera::CameraManager::start(rgb_camera) {
        Ok(cm) => cm,
        Err(e) => {
            disable_raw_mode()?;
            log!("Error starting RGB camera {}: {}", rgb_path, e);
            return Ok(());
        }
    };

    let camera_manager_ir = match pam_tirface_pam::camera::CameraManager::start(ir_camera) {
        Ok(cm) => cm,
        Err(e) => {
            disable_raw_mode()?;
            log!(
                "\r\n\x1B[1;31mCritical Error: The IR camera ({}) is busy or could not be started.\x1B[0m\r",
                ir_path
            );
            log!(
                "This usually happens because the authentication daemon is running in the background and using the camera.\r"
            );
            log!("Please stop the daemon before enrolling a user by running:\r");
            log!("  \x1B[1;33msudo systemctl stop tirface-pam.service\x1B[0m\r");
            log!("Technical detail: {}\r", e);
            return Ok(());
        }
    };

    let (tx_frame, rx_frame) = mpsc::sync_channel::<Vec<u8>>(1);
    let shared_bbox: Arc<Mutex<Option<(i32, i32, u32, u32)>>> = Arc::new(Mutex::new(None));
    let detector_path = config.models.detector_path.clone();
    let thread_bbox = shared_bbox.clone();

    let width_clone = camera_manager_ir.width;
    let height_clone = camera_manager_ir.height;

    thread::spawn(move || {
        let mut detector = match RustfaceDetector::new(&detector_path) {
            Ok(d) => d,
            Err(_) => return,
        };

        while let Ok(gray_buf) = rx_frame.recv() {
            let faces = detector
                .detect(&gray_buf, width_clone, height_clone)
                .unwrap_or_default();
            let best_face = faces.into_iter().max_by_key(|f| f.width * f.height);

            let mut locked_bbox = thread_bbox.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(face) = best_face {
                let b = face;
                *locked_bbox = Some((b.x, b.y, b.width, b.height));
            } else {
                *locked_bbox = None;
            }
        }
    });

    let mut enroll_session = EnrollSession::new(username, &config.models.model_name(), headless)?;

    loop {
        if poll(Duration::from_millis(5))? {
            match read()? {
                Event::Key(event) => match event.code {
                    KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                    KeyCode::Char('t') | KeyCode::Tab => {
                        view_is_ir = !view_is_ir;
                        if let Some(t) = terminal.as_mut() {
                            t.clear().ok();
                        }
                    }
                    KeyCode::Char('+') | KeyCode::Up => {
                        zoom_factor = (zoom_factor + 0.1).min(3.0);
                    }
                    KeyCode::Char('-') | KeyCode::Down => {
                        zoom_factor = (zoom_factor - 0.1).max(1.0);
                    }
                    KeyCode::Enter | KeyCode::Char(' ') => {
                        let _ = enroll_session.start_enrollment();
                    }
                    _ => {}
                },
                Event::Resize(_, _) => {
                    if let Some(t) = terminal.as_mut() {
                        t.clear().ok();
                    }
                }
                _ => {}
            }
        }

        let buf_rgb = match camera_manager_rgb.get_latest_frame() {
            Some(res) => res,
            None => {
                std::thread::sleep(Duration::from_millis(5));
                continue;
            }
        };

        let buf_ir = match camera_manager_ir.get_latest_frame() {
            Some(res) => res,
            None => {
                std::thread::sleep(Duration::from_millis(5));
                continue;
            }
        };

        let mut brightness: u32 = 0;
        let sample_size = 1000;
        let step = buf_ir.len().max(sample_size) / sample_size;
        for i in (0..buf_ir.len()).step_by(step).take(sample_size) {
            brightness += buf_ir[i] as u32;
        }
        let avg_brightness = brightness / sample_size as u32;
        let is_bright = avg_brightness > 15;

        if is_bright {
            let _ = tx_frame.try_send(buf_ir.clone());
        }

        let current_bbox = { *shared_bbox.lock().unwrap_or_else(|e| e.into_inner()) };
        let face_status = evaluate_face(current_bbox, camera_manager_ir.width as i32, camera_manager_ir.height as i32);
        let mut guidance_msg = face_status.guidance_msg(&enroll_session.state);
        let mut box_color = face_status.box_color(&enroll_session.state);

        if enroll_session.can_capture(face_status, is_bright) {
            if let Some((x, y, w, h)) = current_bbox {
                let ir_bbox = pam_tirface_pam::models::BoundingBox {
                    x,
                    y,
                    width: w,
                    height: h,
                };

                match enroll_session.save_capture(
                    &buf_ir,
                    camera_manager_ir.width,
                    camera_manager_ir.height,
                    ir_bbox,
                    session,
                    master_key,
                    config,
                ) {
                    Ok(_) => { box_color = (255, 255, 255);}
                    Err(e) => {
                        guidance_msg = format!("AI Error: {}", e);
                        box_color = (255, 0, 0);
                        if headless {
                            log!("HEADLESS ERROR in get_embedding: {}", e);
                        }
                    }
                }
            }
        }

        if let Some(t) = terminal.as_mut() {
            t.draw(|f| {
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Length(7), Constraint::Min(10)].as_ref())
                    .split(f.area());

                let header_color = if view_is_ir { Color::Magenta } else { Color::Cyan };
                let cam_name = if view_is_ir { "IR VIEW (Native Detector)" } else { "RGB VIEW" };

                let (model_name, backend_name, device_name) = match config.models.get_recognizer_model() {
                    RecognizerModel::MobileFaceNet(backend) => {
                        match backend {
                            Backend::Ort => ("mobilefacenet", "ort", "CPU"),
                            Backend::Openvino(device) => ("mobilefacenet", "openvino", match device {
                                Runtime::Cpu => "CPU",
                                Runtime::Gpu => "GPU",
                                Runtime::Npu => "NPU",
                            }),
                        }
                    }
                    RecognizerModel::ArcFace(backend) => {
                        match backend {
                            Backend::Ort => ("arcface", "ort", "CPU"),
                            Backend::Openvino(device) => ("arcface", "openvino", match device {
                                Runtime::Cpu => "CPU",
                                Runtime::Gpu => "GPU",
                                Runtime::Npu => "NPU",
                            }),
                        }
                    }
                };

                let status_line = match &enroll_session.state {
                    EnrollState::Active { captured_frames, .. } => {
                        let progress = if TARGET_FRAMES > 0 {
                            (*captured_frames as f32 / TARGET_FRAMES as f32 * 20.0) as usize
                        } else { 0 };
                        let bar: String = std::iter::repeat_n('█', progress)
                            .chain(std::iter::repeat_n('░', 20 - progress))
                            .collect();

                        Line::from(vec![
                            Span::styled(">> ENROLLMENT IN PROGRESS << ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                            Span::raw(format!("Status: {} | Captures: [{}] {}/{}", guidance_msg, bar, captured_frames, TARGET_FRAMES)),
                        ])
                    }
                    EnrollState::Finished => {
                        Line::from(vec![
                            Span::styled(">> ENROLLMENT COMPLETED SUCCESSFULLY << You can now press [Q] to exit.", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                        ])
                    }
                    EnrollState::Idle => {
                        Line::from(vec![
                            Span::styled(format!("Sensor Status: {} ", guidance_msg), Style::default().fg(Color::Green)),
                            Span::raw("| Press [Space] to start enrollment."),
                        ])
                    }
                };

                let info_text = vec![
                    Line::from(vec![
                        Span::styled(format!("=== PAM Face Enrollment ({}) ===", cam_name), Style::default().fg(header_color).add_modifier(Modifier::BOLD)),
                    ]),
                    Line::from(vec![
                        Span::styled("Controls: ", Style::default().fg(Color::Yellow)),
                        Span::raw(format!("[T] Toggle View | [+/-] Adjust RGB Zoom ({:.1}x) | [Space] Start | [Q] Exit", zoom_factor)),
                    ]),
                    Line::from(vec![
                        Span::styled("Config: ", Style::default().fg(Color::Blue)),
                        Span::raw(format!("IR: {} | RGB: {} | Mod: {} | Exec: {} ({})", ir_path, rgb_path, model_name, backend_name, device_name)),
                    ]),
                    Line::from(vec![]),
                    status_line,
                ];

                let p_info = Paragraph::new(info_text).block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title(" PAM Face ID - Enroll "),
                );
                f.render_widget(p_info, chunks[0]);

                let cam_chunk = chunks[1];
                let camera_widget = CameraWidget {
                    buf_rgb: Some(&buf_rgb),
                    buf_ir: &buf_ir,
                    rgb_width: camera_manager_rgb.width as usize,
                    rgb_height: camera_manager_rgb.height as usize,
                    ir_width: camera_manager_ir.width as usize,
                    ir_height: camera_manager_ir.height as usize,
                    view_is_ir,
                    zoom_factor,
                    current_bbox,
                    box_color,
                    thickness: 2,
                    blend_box: false,
                    title: None,
                };
                f.render_widget(camera_widget, cam_chunk);
            }).ok();
        } else {
            // Headless mode logs
            match enroll_session.state {
                EnrollState::Active { captured_frames, .. } => {
                    log!(
                        "HEADLESS: Enrolling... State: {}, Captured: {}/{}",
                        guidance_msg,
                        captured_frames,
                        TARGET_FRAMES
                    );
                }
                EnrollState::Finished => {
                    log!("HEADLESS: Registration complete. Press Q to quit.");
                }
                EnrollState::Idle => {
                    log!("HEADLESS: State: {}", guidance_msg);
                }
            }
        }
    }
}

pub fn run_enroll(
    config: &Config,
    username_opt: Option<String>,
    headless: bool,
) -> std::io::Result<()> {
    let username = username_opt.unwrap_or_else(|| {
        std::env::var("SUDO_USER").unwrap_or_else(|_| {
            std::env::var("USER").unwrap_or_else(|_| "usuario_desconocido".to_string())
        })
    });

    if unsafe { libc::geteuid() } != 0 {
        log!("WARNING: You are running enroll without administrator privileges (sudo).");
        log!(
            "Temporary local keys will be used and data will be saved in './enroll_data/{}'",
            username
        );
        log!(
            "For a real production registration, press Ctrl+C and run: sudo tirface-pam-cli enroll {}",
            username
        );
        log!("");
        std::thread::sleep(Duration::from_secs(3));
    }

    enable_raw_mode()?;
    print!("\x1B[?25l");

    let mut terminal = if !headless {
        let mut stdout_handle = stdout();
        execute!(stdout_handle, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout_handle);
        let mut t = Terminal::new(backend)?;
        t.clear()?;
        Some(t)
    } else {
        None
    };

    let master_key = match crypto::FaceCrypto::load_or_create() {
        Ok(k) => k,
        Err(e) => {
            disable_raw_mode()?;
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

    let rgb_camera = match pam_tirface_pam::camera::Camera::new(
        &config.camera.rgb_device,
        pam_tirface_pam::camera::CameraType::Rgb,
    ) {
        Ok(c) => c,
        Err(e) => {
            disable_raw_mode()?;
            log!("Error initializing RGB camera: {}", e);
            return Ok(());
        }
    };

    let ir_camera = match pam_tirface_pam::camera::Camera::new(
        &config.camera.ir_device,
        pam_tirface_pam::camera::CameraType::Ir,
    ) {
        Ok(c) => c,
        Err(e) => {
            disable_raw_mode()?;
            log!("Error initializing IR camera: {}", e);
            return Ok(());
        }
    };

    if let Err(e) = run_camera_session(
        &username,
        &mut *session,
        &master_key,
        config,
        &rgb_camera,
        &ir_camera,
        headless,
        &mut terminal,
    ) {
        disable_raw_mode()?;
        log!("\r\nError crítico: {}", e);
    }

    if let Some(mut t) = terminal {
        execute!(t.backend_mut(), LeaveAlternateScreen)?;
        t.show_cursor()?;
    }
    disable_raw_mode()?;
    log!(
        "Enrollment complete. Data stored for '{}'.",
        username
    );
    Ok(())
}
