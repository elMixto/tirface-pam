#![cfg(feature = "openvino")]

use image::ImageReader;
use openvino::DeviceType;
use pam_tirface_pam::models::{
    FaceDetector, FaceRecognizer, Runtime,
    arcface::ArcFace,
    mobilefacenet::MobileFaceNet,
    openvino_rec::{NormalizationType, OpenVINORecognizer},
    rustface::RustfaceDetector,
};
use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant};

fn load_gray_image(path: &str) -> (Vec<u8>, u32, u32) {
    let img = ImageReader::open(path)
        .expect("Failed to open image")
        .with_guessed_format()
        .expect("Failed to guess format")
        .decode()
        .expect("Failed to decode image")
        .into_luma8();
    let (width, height) = img.dimensions();
    let raw = img.into_raw();
    let equalized = pam_tirface_pam::camera::equalize_grayscale_image(&raw);
    (equalized, width, height)
}

#[test]
fn test_models_with_identities_and_performance() {
    let detector_path = "src/models/rustface/seeta_fd_frontal_v1.0.bin";
    let mfn_path = "src/models/mobilefacenet/mobilefacenet.onnx";
    let arcface_path = "src/models/arcface/arcface.onnx";

    if !Path::new(detector_path).exists()
        || !Path::new(mfn_path).exists()
        || !Path::new(arcface_path).exists()
    {
        println!("Skipping identity test because model files are not found in src/models/");
        return;
    }

    let mut detector = RustfaceDetector::new(detector_path).expect("Failed to load detector");
    let mut mfn = MobileFaceNet::new(mfn_path).expect("Failed to load MobileFaceNet");
    let mut arcface = ArcFace::new(arcface_path).expect("Failed to load ArcFace");

    // Instanciar modelos acelerados por OpenVINO
    let mut mfn_openvino_cpu = OpenVINORecognizer::new(
        mfn_path,
        DeviceType::CPU,
        Runtime::Cpu,
        112,
        "input.1",
        "516",
        NormalizationType::MobileFaceNet,
    )
    .expect("Failed to load MFN in OpenVINO CPU");

    let mut arc_openvino_npu = OpenVINORecognizer::new(
        arcface_path,
        DeviceType::NPU,
        Runtime::Npu,
        112,
        "data", 
        "fc1", 
        NormalizationType::ArcFace
    ).unwrap_or_else(|e| {
        println!("Advertencia: NPU no está lista o soportada de inmediato, haciendo fallback a GPU/CPU. Detalle: {}", e);
        OpenVINORecognizer::new(
            arcface_path,
            DeviceType::CPU,
            Runtime::Npu,
            112,
            "data", 
            "fc1", 
            NormalizationType::ArcFace
        ).unwrap()
    });

    println!("\n=== Fase de Enrolamiento (Enroll) ===");
    let identities_dir = "tests/data/identities";

    // HashMaps para guardar los embeddings de cada persona (identidad -> lista de embeddings)
    let mut mfn_db: HashMap<String, Vec<Vec<f32>>> = HashMap::new();
    let mut arc_db: HashMap<String, Vec<Vec<f32>>> = HashMap::new();
    let mut arc_ov_db: HashMap<String, Vec<Vec<f32>>> = HashMap::new();

    if let Ok(entries) = std::fs::read_dir(identities_dir) {
        for entry in entries.filter_map(Result::ok) {
            let identity_path = entry.path();
            if identity_path.is_dir() {
                let identity_name = identity_path
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .to_string();
                let enroll_dir = identity_path.join("enroll");

                if let Ok(enroll_entries) = std::fs::read_dir(&enroll_dir) {
                    for img_entry in enroll_entries.filter_map(Result::ok) {
                        let img_path = img_entry.path();
                        match img_path
                            .extension()
                            .map_or(false, |ext| ext == "png" || ext == "jpg")
                        {
                            true => {
                                let (buf, w, h) = load_gray_image(img_path.to_str().unwrap());

                                // El tiempo de enroll no es tan importante, pero sí la funcionalidad
                                if let Ok(Some(face)) = detector.detect_best(&buf, w, h) {
                                    if let Ok(mfn_emb) = mfn.get_embedding(&buf, w, h, face.clone()) {
                                        mfn_db
                                            .entry(identity_name.clone())
                                            .or_default()
                                            .push(mfn_emb);
                                    }
                                    if let Ok(arc_emb) = arcface.get_embedding(&buf, w, h, face.clone())
                                    {
                                        arc_db
                                            .entry(identity_name.clone())
                                            .or_default()
                                            .push(arc_emb);
                                    }
                                    if let Ok(arc_ov_emb) =
                                        arc_openvino_npu.get_embedding(&buf, w, h, face)
                                    {
                                        arc_ov_db
                                            .entry(identity_name.clone())
                                            .or_default()
                                            .push(arc_ov_emb);
                                    }
                                }
                            }
                            false => (),
                        }
                    }
                }
            }
        }
    }

    if mfn_db.is_empty() {
        println!("No se encontraron imágenes de enrolamiento válidas.");
        return;
    }

    println!("Identidades enroladas: {}", mfn_db.len());
    for (id, embs) in &mfn_db {
        println!(
            "  - {}: {} plantillas MFN, {} plantillas ArcFace",
            id,
            embs.len(),
            arc_db.get(id).unwrap().len()
        );
    }

    println!("\n=== Fase de Evaluación e Inferencia (Inf) ===");

    // Contadores para métricas de tiempo
    let mut total_det_time = Duration::ZERO;
    let mut total_mfn_time = Duration::ZERO;
    let mut total_mfn_ov_cpu_time = Duration::ZERO;
    let mut total_arc_time = Duration::ZERO;
    let mut total_arc_ov_npu_time = Duration::ZERO;
    let mut inf_count = 0;

    // Contadores de precisión (accuracy)
    let mut mfn_correct = 0;
    let mut mfn_ov_correct = 0;
    let mut arc_correct = 0;
    let mut arc_ov_correct = 0;

    if let Ok(entries) = std::fs::read_dir(identities_dir) {
        for entry in entries.filter_map(Result::ok) {
            let identity_path = entry.path();
            if identity_path.is_dir() {
                let true_identity = identity_path
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .to_string();
                let inf_dir = identity_path.join("inf");

                if let Ok(inf_entries) = std::fs::read_dir(&inf_dir) {
                    for img_entry in inf_entries.filter_map(Result::ok) {
                        let img_path = img_entry.path();
                        if img_path
                            .extension()
                            .map_or(false, |ext| ext == "png" || ext == "jpg")
                        {
                            let (buf, w, h) = load_gray_image(img_path.to_str().unwrap());

                            // 1. Detectar (Midiendo tiempo)
                            let det_start = Instant::now();
                            let det_res = detector.detect_best(&buf, w, h);
                            let det_elapsed = det_start.elapsed();

                            if let Ok(Some(face)) = det_res {
                                total_det_time += det_elapsed;
                                inf_count += 1;

                                // 2. Inferir con MobileFaceNet (Midiendo tiempo)
                                let mfn_start = Instant::now();
                                let mfn_emb_res = mfn.get_embedding(&buf, w, h, face.clone());
                                let mfn_elapsed = mfn_start.elapsed();

                                if let Ok(mfn_emb) = mfn_emb_res {
                                    total_mfn_time += mfn_elapsed;

                                    // Buscar mejor match MFN en la BD
                                    let mut best_sim = f32::MIN;
                                    let mut pred_identity = String::new();
                                    for (id, embs) in &mfn_db {
                                        for emb in embs {
                                            let sim = mfn.compare(emb, &mfn_emb);
                                            if sim > best_sim {
                                                best_sim = sim;
                                                pred_identity = id.clone();
                                            }
                                        }
                                    }
                                    if pred_identity == true_identity {
                                        mfn_correct += 1;
                                    }
                                }

                                // 2.1 Inferir con MobileFaceNet (OpenVINO CPU)
                                let mfn_ov_start = Instant::now();
                                let mfn_ov_emb_res =
                                    mfn_openvino_cpu.get_embedding(&buf, w, h, face.clone());
                                let mfn_ov_elapsed = mfn_ov_start.elapsed();

                                if let Ok(mfn_ov_emb) = mfn_ov_emb_res {
                                    total_mfn_ov_cpu_time += mfn_ov_elapsed;

                                    let mut best_sim = f32::MIN;
                                    let mut pred_identity = String::new();
                                    for (id, embs) in &mfn_db {
                                        // Usamos la misma db porque el modelo es el mismo
                                        for emb in embs {
                                            let sim = mfn_openvino_cpu.compare(emb, &mfn_ov_emb);
                                            if sim > best_sim {
                                                best_sim = sim;
                                                pred_identity = id.clone();
                                            }
                                        }
                                    }
                                    if pred_identity == true_identity {
                                        mfn_ov_correct += 1;
                                    }
                                }

                                // 3. Inferir con ArcFace ORT (Midiendo tiempo)
                                let arc_start = Instant::now();
                                let arc_emb_res = arcface.get_embedding(&buf, w, h, face.clone());
                                let arc_elapsed = arc_start.elapsed();

                                if let Ok(arc_emb) = arc_emb_res {
                                    total_arc_time += arc_elapsed;

                                    // Buscar mejor match ArcFace en la BD
                                    let mut best_sim = f32::MIN;
                                    let mut pred_identity = String::new();
                                    for (id, embs) in &arc_db {
                                        for emb in embs {
                                            let sim = arcface.compare(emb, &arc_emb);
                                            if sim > best_sim {
                                                best_sim = sim;
                                                pred_identity = id.clone();
                                            }
                                        }
                                    }
                                    if pred_identity == true_identity {
                                        arc_correct += 1;
                                    }
                                }

                                // 3.1 Inferir con ArcFace OpenVINO NPU (Midiendo tiempo)
                                let arc_ov_start = Instant::now();
                                let arc_ov_emb_res =
                                    arc_openvino_npu.get_embedding(&buf, w, h, face);
                                let arc_ov_elapsed = arc_ov_start.elapsed();

                                if let Ok(arc_ov_emb) = arc_ov_emb_res {
                                    total_arc_ov_npu_time += arc_ov_elapsed;

                                    // Buscar mejor match ArcFace OV en la BD
                                    let mut best_sim = f32::MIN;
                                    let mut pred_identity = String::new();
                                    for (id, embs) in &arc_ov_db {
                                        for emb in embs {
                                            let sim = arc_openvino_npu.compare(emb, &arc_ov_emb);
                                            if sim > best_sim {
                                                best_sim = sim;
                                                pred_identity = id.clone();
                                            }
                                        }
                                    }
                                    if pred_identity == true_identity {
                                        arc_ov_correct += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if inf_count > 0 {
        let avg_det = total_det_time.as_secs_f64() * 1000.0 / (inf_count as f64);
        let avg_mfn = total_mfn_time.as_secs_f64() * 1000.0 / (inf_count as f64);
        let avg_mfn_ov = total_mfn_ov_cpu_time.as_secs_f64() * 1000.0 / (inf_count as f64);
        let avg_arc = total_arc_time.as_secs_f64() * 1000.0 / (inf_count as f64);
        let avg_arc_ov = total_arc_ov_npu_time.as_secs_f64() * 1000.0 / (inf_count as f64);

        let mfn_acc = (mfn_correct as f64 / inf_count as f64) * 100.0;
        let mfn_ov_acc = (mfn_ov_correct as f64 / inf_count as f64) * 100.0;
        let arc_acc = (arc_correct as f64 / inf_count as f64) * 100.0;
        let arc_ov_acc = (arc_ov_correct as f64 / inf_count as f64) * 100.0;

        println!("\n--- Rendimiento y Precisión de Modelos ---");
        println!("Total imágenes inferidas: {}", inf_count);
        println!("| Modelo                 | Framework | Dispositivo | Accuracy | Tiempo (ms) |");
        println!("|------------------------|-----------|-------------|----------|-------------|");
        println!(
            "| Rustface (Detector)    | Nativo    | CPU         | N/A      | {:.2} ms",
            avg_det
        );
        println!(
            "| MobileFaceNet          | ORT       | CPU         | {:.2}%    | {:.2} ms",
            mfn_acc, avg_mfn
        );
        println!(
            "| MobileFaceNet          | OpenVINO  | CPU         | {:.2}%    | {:.2} ms",
            mfn_ov_acc, avg_mfn_ov
        );
        println!(
            "| ArcFace ResNet100      | ORT       | CPU         | {:.2}%    | {:.2} ms",
            arc_acc, avg_arc
        );
        println!(
            "| ArcFace ResNet100      | OpenVINO  | NPU (o fall) | {:.2}%    | {:.2} ms",
            arc_ov_acc, avg_arc_ov
        );
        println!("==========================================\n");
    } else {
        println!("No se encontraron imágenes válidas para inferencia.");
    }
}

#[test]
fn test_face_recognition_pipeline() {
    // 1. Instanciar los modelos
    let detector_path = "src/models/rustface/seeta_fd_frontal_v1.0.bin";
    let mfn_path = "src/models/mobilefacenet/mobilefacenet.onnx";
    let arcface_path = "src/models/arcface/arcface.onnx";

    if !Path::new(detector_path).exists()
        || !Path::new(mfn_path).exists()
        || !Path::new(arcface_path).exists()
    {
        println!("Skipping pipeline test because model files are not found in src/models/");
        return;
    }

    let mut detector = RustfaceDetector::new(detector_path).expect("Failed to load detector");
    let mut mfn = MobileFaceNet::new(mfn_path).expect("Failed to load MobileFaceNet");
    let mut arcface = ArcFace::new(arcface_path).expect("Failed to load ArcFace");

    // 2. Load registration image (enrollment)
    let enroll_path = "tests/data/enroll/enroll_001.png";
    if !Path::new(enroll_path).exists() {
        println!("Skipping pipeline test because enroll image is missing");
        return;
    }
    let (enroll_buf, enroll_w, enroll_h) = load_gray_image(enroll_path);

    // Detect face in the registration image
    let enroll_face = detector
        .detect_best(&enroll_buf, enroll_w, enroll_h)
        .expect("Failed during detection")
        .expect("No face found in enrollment image");

    // Extraer embedding de registro
    let mfn_enroll_emb = mfn
        .get_embedding(&enroll_buf, enroll_w, enroll_h, enroll_face)
        .expect("Failed to get enrollment embedding (MFN)");
    let arc_enroll_emb = arcface
        .get_embedding(&enroll_buf, enroll_w, enroll_h, enroll_face)
        .expect("Failed to get enrollment embedding (ArcFace)");

    // 3. Load evaluation image 1 (Same person)
    let eval1_path = "tests/data/enroll/enroll_001.png"; // Usamos la misma para probar similitud
    let (eval1_buf, eval1_w, eval1_h) = load_gray_image(eval1_path);

    let eval1_face = detector
        .detect_best(&eval1_buf, eval1_w, eval1_h)
        .expect("Failed during detection")
        .expect("No face found in eval1 image");

    let mfn_eval1_emb = mfn
        .get_embedding(&eval1_buf, eval1_w, eval1_h, eval1_face)
        .expect("Failed to get eval1 embedding (MFN)");
    let arc_eval1_emb = arcface
        .get_embedding(&eval1_buf, eval1_w, eval1_h, eval1_face)
        .expect("Failed to get eval1 embedding (ArcFace)");

    // Comparar: Deberían ser exactamente iguales (o extremadamente similares)
    let mfn_sim_same = mfn.compare(&mfn_enroll_emb, &mfn_eval1_emb);
    let arc_sim_same = arcface.compare(&arc_enroll_emb, &arc_eval1_emb);

    println!("Similarity MFN (Same Person): {}", mfn_sim_same);
    println!("Similarity ArcFace (Same Person): {}", arc_sim_same);
    assert!(mfn_sim_same > 0.8, "Same person similarity should be high");
    assert!(arc_sim_same > 0.8, "Same person similarity should be high");

    // 4. Load evaluation image 2 (Different person)
    let eval2_path = "tests/data/enroll/enroll_002.png"; // Usamos una diferente
    let (eval2_buf, eval2_w, eval2_h) = load_gray_image(eval2_path);

    let eval2_face = detector
        .detect_best(&eval2_buf, eval2_w, eval2_h)
        .expect("Failed during detection")
        .expect("No face found in eval2 image");

    let mfn_eval2_emb = mfn
        .get_embedding(&eval2_buf, eval2_w, eval2_h, eval2_face)
        .expect("Failed to get eval2 embedding (MFN)");
    let arc_eval2_emb = arcface
        .get_embedding(&eval2_buf, eval2_w, eval2_h, eval2_face)
        .expect("Failed to get eval2 embedding (ArcFace)");

    // Comparar: Deberían ser different
    let mfn_sim_diff = mfn.compare(&mfn_enroll_emb, &mfn_eval2_emb);
    let arc_sim_diff = arcface.compare(&arc_enroll_emb, &arc_eval2_emb);
    println!("Similarity MFN (Eval2 - Same Person): {}", mfn_sim_diff);
    println!("Similarity ArcFace (Eval2 - Same Person): {}", arc_sim_diff);

    // 5. Load evaluation image 3 (True different person)
    let eval3_path = "/tmp/opencode/messi.jpg";
    if Path::new(eval3_path).exists() {
        let (eval3_buf, eval3_w, eval3_h) = load_gray_image(eval3_path);
        if let Ok(Some(eval3_face)) = detector.detect_best(&eval3_buf, eval3_w, eval3_h) {
            let mfn_eval3_emb = mfn
                .get_embedding(&eval3_buf, eval3_w, eval3_h, eval3_face)
                .unwrap();
            let arc_eval3_emb = arcface
                .get_embedding(&eval3_buf, eval3_w, eval3_h, eval3_face)
                .unwrap();

            let mfn_sim_diff_true = mfn.compare(&mfn_enroll_emb, &mfn_eval3_emb);
            let arc_sim_diff_true = arcface.compare(&arc_enroll_emb, &arc_eval3_emb);

            println!(
                "Similarity MFN (True Different Person - Messi): {}",
                mfn_sim_diff_true
            );
            println!(
                "Similarity ArcFace (True Different Person - Messi): {}",
                arc_sim_diff_true
            );
        }
    }
}

#[test]
fn test_models_with_enroll_and_inf_dataset() {
    let detector_path = "src/models/rustface/seeta_fd_frontal_v1.0.bin";
    let mfn_path = "src/models/mobilefacenet/mobilefacenet.onnx";
    let arcface_path = "src/models/arcface/arcface.onnx";

    if !Path::new(detector_path).exists()
        || !Path::new(mfn_path).exists()
        || !Path::new(arcface_path).exists()
    {
        println!("Skipping dataset test because model files are not found in src/models/");
        return;
    }

    let mut detector = RustfaceDetector::new(detector_path).expect("Failed to load detector");
    let mut mfn = MobileFaceNet::new(mfn_path).expect("Failed to load MobileFaceNet");
    let mut arcface = ArcFace::new(arcface_path).expect("Failed to load ArcFace");

    // Instanciar modelos acelerados por OpenVINO
    let mut mfn_openvino_cpu = OpenVINORecognizer::new(
        mfn_path,
        DeviceType::CPU,
        Runtime::Cpu,
        112,
        "input.1",
        "516",
        NormalizationType::MobileFaceNet,
    )
    .expect("Failed to load MFN in OpenVINO CPU");

    let mut arc_openvino_cpu = OpenVINORecognizer::new(
        arcface_path,
        DeviceType::CPU,
        Runtime::Cpu,
        112,
        "data",
        "fc1",
        NormalizationType::ArcFace,
    )
    .expect("Failed to load ArcFace in OpenVINO CPU");

    let mut arc_openvino_npu = OpenVINORecognizer::new(
        arcface_path,
        DeviceType::NPU,
        Runtime::Npu,
        112,
        "data",
        "fc1",
        NormalizationType::ArcFace,
    )
    .expect("Failed to load ArcFace in OpenVINO NPU");

    println!("\n=== Fase de Enrolamiento (Enroll) ===");
    let enroll_dir = "tests/data/enroll";
    let mut mfn_enroll_embeddings = Vec::new();
    let mut arc_enroll_embeddings = Vec::new();

    if let Ok(entries) = std::fs::read_dir(enroll_dir) {
        for entry in entries.filter_map(Result::ok) {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("png") {
                let file_name = path.file_name().unwrap().to_string_lossy().to_string();
                let (buf, w, h) = load_gray_image(path.to_str().unwrap());

                if let Ok(Some(face)) = detector.detect_best(&buf, w, h) {
                    if let Ok(mfn_emb) = mfn.get_embedding(&buf, w, h, face) {
                        mfn_enroll_embeddings.push((file_name.clone(), mfn_emb));
                    }
                    if let Ok(arc_emb) = arcface.get_embedding(&buf, w, h, face) {
                        arc_enroll_embeddings.push((file_name, arc_emb));
                    }
                }
            }
        }
    }

    if mfn_enroll_embeddings.is_empty() {
        println!("Skipping dataset test because no valid faces were found in tests/data/enroll/");
        return;
    }

    println!(
        "Loaded {} MFN embeddings and {} ArcFace enrollment embeddings.",
        mfn_enroll_embeddings.len(),
        arc_enroll_embeddings.len()
    );

    println!("\n=== Evaluando modelos con dataset inf ===");
    let inf_dir = "tests/data/inf";
    let mut mfn_similarities = Vec::new();
    let mut arc_similarities = Vec::new();

    if let Ok(entries) = std::fs::read_dir(inf_dir) {
        for entry in entries.filter_map(Result::ok) {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("png") {
                let file_name = path.file_name().unwrap().to_string_lossy().to_string();
                let (inf_buf, inf_w, inf_h) = load_gray_image(path.to_str().unwrap());

                if let Ok(Some(inf_face)) = detector.detect_best(&inf_buf, inf_w, inf_h) {
                    // MobileFaceNet Eval
                    if let Ok(inf_emb) = mfn.get_embedding(&inf_buf, inf_w, inf_h, inf_face) {
                        let mut max_sim = f32::MIN;
                        for (_, enroll_emb) in &mfn_enroll_embeddings {
                            let sim = mfn.compare(enroll_emb, &inf_emb);
                            if sim > max_sim {
                                max_sim = sim;
                            }
                        }
                        mfn_similarities.push(max_sim);
                    }

                    // ArcFace Eval
                    if let Ok(inf_emb) = arcface.get_embedding(&inf_buf, inf_w, inf_h, inf_face) {
                        let mut max_sim = f32::MIN;
                        for (_, enroll_emb) in &arc_enroll_embeddings {
                            let sim = arcface.compare(enroll_emb, &inf_emb);
                            if sim > max_sim {
                                max_sim = sim;
                            }
                        }
                        arc_similarities.push(max_sim);
                    }
                }
            }
        }
    }

    if !mfn_similarities.is_empty() {
        let mfn_avg = mfn_similarities.iter().sum::<f32>() / mfn_similarities.len() as f32;
        let arc_avg = arc_similarities.iter().sum::<f32>() / arc_similarities.len() as f32;

        let mfn_min = mfn_similarities.iter().copied().fold(f32::MAX, f32::min);
        let mfn_max = mfn_similarities.iter().copied().fold(f32::MIN, f32::max);

        let arc_min = arc_similarities.iter().copied().fold(f32::MAX, f32::min);
        let arc_max = arc_similarities.iter().copied().fold(f32::MIN, f32::max);

        println!("\n--- Comparativa de Modelos (Dataset Inf vs Enroll) ---");
        println!("| Métrica   | MobileFaceNet | ArcFace ResNet |");
        println!("|-----------|---------------|----------------|");
        println!(
            "| Promedio  | {:.4}        | {:.4}         |",
            mfn_avg, arc_avg
        );
        println!(
            "| Mínimo    | {:.4}        | {:.4}         |",
            mfn_min, arc_min
        );
        println!(
            "| Máximo    | {:.4}        | {:.4}         |",
            mfn_max, arc_max
        );
        println!("====================================================\n");
    } else {
        println!("No se encontraron imágenes en {} para evaluar.", inf_dir);
    }
}
