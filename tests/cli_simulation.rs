use image::GenericImageView;
use pam_tirface_pam::config::Config;
use pam_tirface_pam::crypto;
use pam_tirface_pam::models::{FaceDetector, FaceRecognizer, rustface::RustfaceDetector};
use std::fs;
use std::path::Path;

#[test]
fn simulate_cli_enroll_and_inference() {
    let config = Config::load();
    let detector_path = config.models.detector_path.clone();
    let mut detector = RustfaceDetector::new(&detector_path).expect("Failed to load detector");
    let mut session = pam_tirface_pam::models::load_recognizer(&config.models).unwrap();

    // 1. Simulate Enroll
    let enroll_dir = "tests/data/identities/person_a/enroll";
    let db_dir = "target/test_db/person_a";
    fs::create_dir_all(db_dir).unwrap();

    let entries = fs::read_dir(enroll_dir).expect("Failed to read enroll dir");
    let mut captured_frames = 0;

    for entry in entries {
        let path = entry.unwrap().path();
        if path.extension().unwrap_or_default() != "png" {
            continue;
        }

        let img = image::open(&path).unwrap().into_luma8();
        let (width, height) = img.dimensions();
        let buf = img.into_raw();

        let mut brightness: u32 = 0;
        let sample_size = 1000;
        let step = buf.len().max(sample_size) / sample_size;
        for i in (0..buf.len()).step_by(step).take(sample_size) {
            brightness += buf[i] as u32;
        }
        let avg_brightness = brightness / sample_size as u32;
        let is_bright = avg_brightness > 15;

        if is_bright {
            let faces = detector.detect(&buf, width, height).unwrap_or_default();
            let best_face = faces.into_iter().max_by_key(|f| f.width * f.height);

            if let Some(face) = best_face {
                let w = face.width as i32;
                if w > 60 {
                    // is_large_enough
                    if let Ok(emb) = session.get_embedding(&buf, width, height, face) {
                        // Enrolamos
                        let file_path = format!("{}/{}.raw", db_dir, captured_frames);
                        let bytes: &[u8] = bytemuck::cast_slice(&emb);
                        fs::write(&file_path, bytes).unwrap();
                        captured_frames += 1;
                        if captured_frames >= 15 {
                            // TARGET_FRAMES
                            break;
                        }
                    }
                }
            }
        }
    }

    assert!(captured_frames > 0, "No se capturaron frames en el enroll");
    println!(
        "Simulated Enroll completado: {} frames capturados.",
        captured_frames
    );

    // 2. Simulate Inference
    // Load database
    let mut db = Vec::new();
    if let Ok(entries) = std::fs::read_dir(db_dir) {
        for entry in entries {
            let path = entry.unwrap().path();
            if path.extension().unwrap_or_default() == "raw" {
                let raw_bytes = fs::read(&path).unwrap();
                let mut emb = Vec::with_capacity(raw_bytes.len() / 4);
                for chunk in raw_bytes.chunks_exact(4) {
                    let val = f32::from_ne_bytes(chunk.try_into().unwrap());
                    emb.push(val);
                }
                db.push(("person_a".to_string(), emb));
            }
        }
    }

    assert!(!db.is_empty(), "La base de datos virtual está vacía");
    println!("Base de datos virtual cargada con {} firmas.", db.len());

    let inf_dir = "tests/data/identities/person_a/inf";
    let inf_entries = fs::read_dir(inf_dir).expect("Failed to read inf dir");
    let mut inf_successes = 0;
    let mut inf_total = 0;

    for entry in inf_entries {
        let path = entry.unwrap().path();
        if path.extension().unwrap_or_default() != "png" {
            continue;
        }

        let img = image::open(&path).unwrap().into_luma8();
        let (width, height) = img.dimensions();
        let buf = img.into_raw();

        let faces = detector.detect(&buf, width, height).unwrap_or_default();
        let best_face = faces.into_iter().max_by_key(|f| f.width * f.height);

        if let Some(face) = best_face {
            if let Ok(emb) = session.get_embedding(&buf, width, height, face) {
                let mut best_score = 0.0;
                for (_, db_emb) in &db {
                    let score = session.compare(&emb, db_emb);
                    if score > best_score {
                        best_score = score;
                    }
                }

                inf_total += 1;
                if best_score >= config.recognition.threshold {
                    inf_successes += 1;
                } else {
                    println!(
                        "Inference failure: {:?} score: {}",
                        path.file_name().unwrap(),
                        best_score
                    );
                }
            }
        }
    }

    println!(
        "Simulated Inference: {}/{} frames reconocidos con éxito.",
        inf_successes, inf_total
    );
    assert!(inf_total > 0, "No se procesaron frames de inferencia");
    // Verificamos que al menos el 70% pasen
    let accuracy = inf_successes as f32 / inf_total as f32;
    assert!(accuracy >= 0.7, "Precisión demasiado baja: {}", accuracy);

    // Clean up
    fs::remove_dir_all("target/test_db").unwrap();
}
