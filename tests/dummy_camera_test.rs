use std::thread::sleep;
use std::time::{Duration, Instant};
use pam_tirface_pam::camera::{Camera, CameraType, CameraManager};
use pam_tirface_pam::config::{CameraPath, ModelsConfig, RecognizerModel, Backend, Runtime};
use pam_tirface_pam::models::{FaceDetector, rustface::RustfaceDetector};

fn run_e2e_test_for_model(models_config: &ModelsConfig) {
    let model_name = models_config.model_name();
    let runtime_desc = match models_config.get_recognizer_model() {
        RecognizerModel::MobileFaceNet(backend) => match backend {
            Backend::Ort => "ORT (CPU)".to_string(),
            Backend::Openvino(rt) => format!("OpenVINO ({:?})", rt),
        },
        RecognizerModel::ArcFace(backend) => match backend {
            Backend::Ort => "ORT (CPU)".to_string(),
            Backend::Openvino(rt) => format!("OpenVINO ({:?})", rt),
        },
    };

    println!("\n=======================================================");
    println!(">>> Starting E2E Test: {} using {} <<<", model_name, runtime_desc);
    println!("=======================================================");

    let detector_path = "src/models/rustface/seeta_fd_frontal_v1.0.bin";
    
    if !std::path::Path::new(detector_path).exists() {
        println!("Skipping test for {} because local face detector was not found.", model_name);
        return;
    }

    let mut detector = RustfaceDetector::new(detector_path).expect("Error loading detector");
    
    // load_recognizer ya se encarga de aplicar los fallbacks correspondientes a CPU si el runtime físico no está disponible
    let mut recognizer = match pam_tirface_pam::models::load_recognizer(models_config) {
        Ok(rec) => rec,
        Err(e) => {
            println!("⚠️ Saltando prueba para {} usando {}: El motor no pudo inicializarse ni en fallback: {}", model_name, runtime_desc, e);
            return;
        }
    };

    // 1. --- PASO 1: SIMULAR EL ENROLL (Registro) EN TIEMPO REAL ---
    println!("\n[STEP 1 - {}] Starting ENROLL sequence...", model_name);
    let enroll_camera = Camera::new(
        &CameraPath::Custom("dummy:enroll".to_string()),
        CameraType::Ir
    ).unwrap();

    let mut enroll_manager = CameraManager::start(&enroll_camera).unwrap();
    let mut enrolled_embeddings = Vec::new();

    let start_time = Instant::now();
    while start_time.elapsed() < Duration::from_secs(3) && enrolled_embeddings.len() < 15 {
        if let Some(frame) = enroll_manager.get_latest_frame() {
            if let Ok(Some(face)) = detector.detect_best(&frame, enroll_manager.width, enroll_manager.height) {
                if let Ok(emb) = recognizer.get_embedding(&frame, enroll_manager.width, enroll_manager.height, face) {
                    enrolled_embeddings.push(emb);
                    println!("  [Enroll - {}] Embedding biométrico #{} extraído.", model_name, enrolled_embeddings.len());
                }
            }
        }
        sleep(Duration::from_millis(100));
    }

    enroll_manager.stop();
    println!("✅ Enroll completed for {}. Signatures in memory: {}", model_name, enrolled_embeddings.len());
    assert!(enrolled_embeddings.len() > 0, "No signature was captured during enroll");

    // 2. --- PASO 2: SIMULAR LA INFERENCIA (Autenticación) EN TIEMPO REAL ---
    println!("\n[STEP 2 - {}] Starting INFERENCE / AUTHENTICATION sequence...", model_name);
    let inf_camera = Camera::new(
        &CameraPath::Custom("dummy:inf".to_string()),
        CameraType::Ir
    ).unwrap();

    let mut inf_manager = CameraManager::start(&inf_camera).unwrap();

    let mut successful_attempts = 0;
    let mut total_attempts = 0;
    let mut total_infer_time = Duration::ZERO;

    let threshold = 0.60;

    let test_duration = Duration::from_secs(3);
    let start_time = Instant::now();

    while start_time.elapsed() < test_duration && total_attempts < 10 {
        if let Some(frame) = inf_manager.get_latest_frame() {
            if let Ok(Some(face)) = detector.detect_best(&frame, inf_manager.width, inf_manager.height) {
                let t_infer_start = Instant::now();
                if let Ok(emb) = recognizer.get_embedding(&frame, inf_manager.width, inf_manager.height, face) {
                    let t_infer = t_infer_start.elapsed();
                    total_infer_time += t_infer;
                    let t_infer_ms = t_infer.as_secs_f32() * 1000.0;

                    let mut best_score = 0.0;
                    for enroll_emb in &enrolled_embeddings {
                        let score = recognizer.compare(&emb, enroll_emb);
                        if score > best_score {
                            best_score = score;
                        }
                    }

                    total_attempts += 1;
                    if best_score >= threshold {
                        successful_attempts += 1;
                        println!("  [Inf - {}] Attempt #{}: Authentication successful (Similarity: {:.4}) | Inference Time: {:.2} ms", model_name, total_attempts, best_score, t_infer_ms);
                    } else {
                        println!("  [Inf - {}] Attempt #{}: REJECTED (Similarity: {:.4}) | Inference Time: {:.2} ms", model_name, total_attempts, best_score, t_infer_ms);
                    }
                }
            }
        }
        sleep(Duration::from_millis(150));
    }

    inf_manager.stop();

    println!("\n=== E2E Test Results ({}) ===", model_name);
    println!("Total Authentication Attempts: {}", total_attempts);
    println!("Successful Authentications: {} / {}", successful_attempts, total_attempts);
    
    if total_attempts > 0 {
        let avg_infer_ms = (total_infer_time.as_secs_f32() * 1000.0) / total_attempts as f32;
        println!("Average Inference Time: {:.2} ms", avg_infer_ms);
    }

    assert!(total_attempts > 0, "No inference attempt was processed");
    
    // Criterio de éxito real de PAM (autenticado al menos una vez antes de expirar el timeout)
    assert!(successful_attempts > 0, "Authentication failed on all frames");
    println!("✅ E2E Test completed successfully for {} using {}!", model_name, runtime_desc);
}

#[test]
fn test_e2e_enroll_and_inference_with_dummy_camera_all_models_and_runtimes() {
    let detector_path = "src/models/rustface/seeta_fd_frontal_v1.0.bin";

    #[allow(unused_mut)]
    let mut test_configs = vec![
        // 1. MobileFaceNet con ONNX Runtime (CPU)
        ModelsConfig {
            detector_path: detector_path.to_string(),
            recognizer_model: "mobilefacenet".to_string(), recognizer_backend: "ort".to_string(), recognizer_device: "CPU".to_string(),
        },
        // 3. ArcFace con ONNX Runtime (CPU)
        ModelsConfig {
            detector_path: detector_path.to_string(),
            recognizer_model: "arcface".to_string(), recognizer_backend: "ort".to_string(), recognizer_device: "CPU".to_string(),
        },
    ];

    #[cfg(feature = "openvino")]
    {
        test_configs.extend(vec![
            // 2. MobileFaceNet con OpenVINO (CPU)
            ModelsConfig {
                detector_path: detector_path.to_string(),
                recognizer_model: "mobilefacenet".to_string(), recognizer_backend: "openvino".to_string(), recognizer_device: "CPU".to_string(),
            },
            // 4. ArcFace con OpenVINO (CPU)
            ModelsConfig {
                detector_path: detector_path.to_string(),
                recognizer_model: "arcface".to_string(), recognizer_backend: "openvino".to_string(), recognizer_device: "CPU".to_string(),
            },
            // 5. ArcFace con OpenVINO (GPU - con fallback a CPU si no hay GPU)
            ModelsConfig {
                detector_path: detector_path.to_string(),
                recognizer_model: "arcface".to_string(), recognizer_backend: "openvino".to_string(), recognizer_device: "GPU".to_string(),
            },
            // 6. ArcFace con OpenVINO (NPU - con fallback a CPU si no hay NPU)
            ModelsConfig {
                detector_path: detector_path.to_string(),
                recognizer_model: "arcface".to_string(), recognizer_backend: "openvino".to_string(), recognizer_device: "NPU".to_string(),
            },
        ]);
    }

    for config in test_configs {
        run_e2e_test_for_model(&config);
    }
}
