use std::collections::HashMap;
use std::path::{Path, PathBuf};
use pam_tirface_pam::config::{Config, ModelsConfig, RecognizerModel, Backend};
use pam_tirface_pam::models::{FaceDetector, FaceRecognizer, rustface::RustfaceDetector};

fn load_gray_image(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::ImageReader::open(path).ok()?
        .with_guessed_format().ok()?
        .decode().ok()?
        .into_luma8();
    let (width, height) = img.dimensions();
    let raw = img.into_raw();
    // Usamos la pipeline de ecualización nativa que implementamos
    let equalized = pam_tirface_pam::camera::equalize_grayscale_image(&raw);
    Some((equalized, width, height))
}

#[test]
fn run_biometric_correlation_benchmark() {
    println!("\n=======================================================");
    println!(">>> BIOMETRIC BENCHMARK: CORRELATION EVALUATION <<<");
    println!("=======================================================");

    let detector_path = "src/models/rustface/seeta_fd_frontal_v1.0.bin";
    if !Path::new(detector_path).exists() {
        println!("Skipping test because face detector was not found.");
        return;
    }

    let config = Config::default();
    let mut detector = RustfaceDetector::new(detector_path).expect("Error loading detector");
    let mut recognizer = pam_tirface_pam::models::load_recognizer(&config.models).expect("Error loading recognizer");

    let identities_dir = "tests/data/identities";
    if !Path::new(identities_dir).exists() {
        println!("Saltando benchmark: No existe la carpeta {}", identities_dir);
        return;
    }

    // 1. --- ESCANEO DINÁMICO DE IDENTIDADES EN EL DIRECTORIO ---
    let mut identities = Vec::new();
    if let Ok(entries) = std::fs::read_dir(identities_dir) {
        for entry in entries.filter_map(Result::ok) {
            let path = entry.path();
            if path.is_dir() {
                let name = path.file_name().unwrap().to_string_lossy().to_string();
                if path.join("enroll").exists() && path.join("inf").exists() {
                    identities.push((name, path));
                }
            }
        }
    }

    if identities.is_empty() {
        println!("⚠️ No structured identities found for the benchmark in {}", identities_dir);
        println!("Expected structure for each subject of ThermVision-DB:");
        println!("  {}/<subject>/enroll/ (PNG images for enrollment)", identities_dir);
        println!("  {}/<subject>/inf/    (PNG images for inference)", identities_dir);
        return;
    }

    println!("Detected {} identities for dynamic evaluation:", identities.len());
    for (name, _) in &identities {
        println!("  - {}", name);
    }

    // 2. --- EXTRACCIÓN DE EMBEDDINGS (Enroll e Inferencia) ---
    // Guardaremos: identidad -> lista de embeddings
    let mut enroll_db: HashMap<String, Vec<Vec<f32>>> = HashMap::new();
    let mut inf_db: HashMap<String, Vec<Vec<f32>>> = HashMap::new();

    let mut total_detection_attempts = 0;
    let mut successful_detections = 0;

    for (name, path) in &identities {
        // Load Enroll
        let enroll_path = path.join("enroll");
        if let Ok(files) = std::fs::read_dir(enroll_path) {
            for file_entry in files.filter_map(Result::ok) {
                let img_path = file_entry.path();
                if let Some((buf, w, h)) = load_gray_image(&img_path) {
                    total_detection_attempts += 1;
                    if let Ok(Some(face)) = detector.detect_best(&buf, w, h) {
                        successful_detections += 1;
                        if let Ok(emb) = recognizer.get_embedding(&buf, w, h, face) {
                            enroll_db.entry(name.clone()).or_default().push(emb);
                        }
                    }
                }
            }
        }

        // Load Inference
        let inf_path = path.join("inf");
        if let Ok(files) = std::fs::read_dir(inf_path) {
            for file_entry in files.filter_map(Result::ok) {
                let img_path = file_entry.path();
                if let Some((buf, w, h)) = load_gray_image(&img_path) {
                    total_detection_attempts += 1;
                    if let Ok(Some(face)) = detector.detect_best(&buf, w, h) {
                        successful_detections += 1;
                        if let Ok(emb) = recognizer.get_embedding(&buf, w, h, face) {
                            inf_db.entry(name.clone()).or_default().push(emb);
                        }
                    }
                }
            }
        }
    }

    let detection_rate = (successful_detections as f32 / total_detection_attempts as f32) * 100.0;
    println!("\n=== 1. FACE DETECTION METRICS ===");
    println!("Successful detections: {} / {} ({:.2}% accuracy)", successful_detections, total_detection_attempts, detection_rate);

    // 3. --- ANÁLISIS DE CORRELACIÓN Y MATRIZ DE CONFUSIÓN ---
    println!("\n=== 2. BIOMETRIC CORRELATION MATRIX (Average Similarity) ===");
    
    let threshold = 0.60;
    let mut intra_class_similarities = Vec::new();
    let mut inter_class_similarities = Vec::new();

    let mut false_accepts = 0;
    let mut total_inter_comparisons = 0;

    let mut false_rejects = 0;
    let mut total_intra_comparisons = 0;

    // Print matrix header
    print!("Subject       | ");
    for (col_name, _) in &identities {
        print!("{:12} | ", col_name);
    }
    println!();
    println!("{}", "-".repeat(15 + 15 * identities.len()));

    for (row_name, _) in &identities {
        print!("{:13} | ", row_name);
        
        for (col_name, _) in &identities {
            // Comparar las fotos de inferencia del sujeto de la FILA (row_name)
            // contra las plantillas de registro del sujeto de la COLUMNA (col_name)
            let row_inf_embs = inf_db.get(row_name);
            let col_enroll_embs = enroll_db.get(col_name);

            if let (Some(inf_embs), Some(enroll_embs)) = (row_inf_embs, col_enroll_embs) {
                let mut sum_score = 0.0;
                let mut count = 0;

                for inf_emb in inf_embs {
                    for enroll_emb in enroll_embs {
                        let score = recognizer.compare(inf_emb, enroll_emb);
                        sum_score += score;
                        count += 1;

                        if row_name == col_name {
                            // Intra-clase (misma persona)
                            intra_class_similarities.push(score);
                            total_intra_comparisons += 1;
                            if score < threshold {
                                false_rejects += 1;
                            }
                        } else {
                            // Inter-clase (diferente persona)
                            inter_class_similarities.push(score);
                            total_inter_comparisons += 1;
                            if score >= threshold {
                                false_accepts += 1;
                            }
                        }
                    }
                }

                if count > 0 {
                    let avg = sum_score / count as f32;
                    print!("{:12.4} | ", avg);
                } else {
                    print!("{:12} | ", "N/A");
                }
            } else {
                print!("{:12} | ", "N/A");
            }
        }
        println!();
    }

    // 4. --- REPORTAR MÉTRICAS BIOMÉTRICAS CLAVE (FAR, FRR, EER) ---
    println!("\n=== 3. FINAL BIOMETRIC METRICS ===");
    
    let avg_intra = intra_class_similarities.iter().sum::<f32>() / intra_class_similarities.len() as f32;
    let avg_inter = if !inter_class_similarities.is_empty() {
        inter_class_similarities.iter().sum::<f32>() / inter_class_similarities.len() as f32
    } else { 0.0 };

    let far = if total_inter_comparisons > 0 {
        (false_accepts as f32 / total_inter_comparisons as f32) * 100.0
    } else { 0.0 };

    let frr = if total_intra_comparisons > 0 {
        (false_rejects as f32 / total_intra_comparisons as f32) * 100.0
    } else { 0.0 };

    println!("- Average Intra-Class Similarity (Same person - Desired HIGH): {:.4}", avg_intra);
    println!("- Average Inter-Class Similarity (Different person - Desired LOW): {:.4}", avg_inter);
    println!("- False Acceptance Rate (FAR): {:.2}% (Comparisons: {})", far, total_inter_comparisons);
    println!("- False Rejection Rate (FRR): {:.2}% (Comparisons: {})", frr, total_intra_comparisons);
    
    println!("\n💡 Configuration Solution:");
    println!("  If FAR is > 0%, consider raising the recognition threshold.");
    println!("  If FRR is > 0%, consider lowering the recognition threshold or registering more enroll poses.");
    println!("=======================================================\n");
}
