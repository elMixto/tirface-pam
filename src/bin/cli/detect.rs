use log::info as log;
use pam_tirface_pam::config::{Backend, Config, RecognizerModel, Runtime};
use pam_tirface_pam::models::{self, BoundingBox};
use std::time::Instant;
use v4l::video::Capture;

pub fn run_detect(config: &Config) -> std::io::Result<()> {
    log!("\n🔍 Starting Hardware and Artificial Intelligence Self-Diagnosis...\n");

    log!("=== 1. Physical Silicon Audit ===");
    let mut available_devices: Vec<String> = Vec::new();
    match openvino::Core::new() {
        Ok(core) => {
            log!("✅ Intel OpenVINO library detected and operational.");
            match core.available_devices() {
                Ok(devices) => {
                    log!("Available inference devices in the system:");
                    for dev in &devices {
                        let dev_str = dev.to_string();
                        log!("  - \x1B[1;36m{}\x1B[0m", dev_str);
                        available_devices.push(dev_str);
                    }
                }
                Err(e) => log!("⚠️ Could not list OpenVINO devices: {}", e),
            }
        }
        Err(e) => {
            log!("❌ Critical failure initializing OpenVINO: {}", e);
            log!(
                "Ensure you have the OpenVINO Toolkit libraries installed on your system."
            );
        }
    }

    log!("\n=== 1.5. Camera and Supported Formats Audit ===");
    let nodes = v4l::context::enum_devices();
    if nodes.is_empty() {
        log!("⚠️ No video camera nodes detected in the system (e.g., /dev/video*).");
    } else {
        log!("Searching for available cameras and video formats:");
        for node in nodes {
            let path_str = node.path().to_string_lossy().to_string();
            log!("- Node: \x1B[1;36m{}\x1B[0m", path_str);
            if let Ok(dev) = v4l::Device::with_path(node.path()) {
                if let Ok(caps) = dev.query_caps() {
                    log!("  Card: {}", caps.card);
                    log!("  Driver: {}", caps.driver);
                }
                if let Ok(formats) = dev.enum_formats() {
                    for fmt in formats {
                        let fourcc_str = String::from_utf8_lossy(&fmt.fourcc.repr).to_string();
                        log!("  Format: \x1B[1;33m{}\x1B[0m ({})", fourcc_str, fmt.description);
                        
                        if let Ok(sizes) = dev.enum_framesizes(fmt.fourcc) {
                            let mut size_strs = Vec::new();
                            for size in sizes {
                                match size.size {
                                    v4l::framesize::FrameSizeEnum::Discrete(discrete) => {
                                        size_strs.push(format!("{}x{}", discrete.width, discrete.height));
                                    }
                                    v4l::framesize::FrameSizeEnum::Stepwise(stepwise) => {
                                        size_strs.push(format!("{}x{}..{}x{}", stepwise.min_width, stepwise.min_height, stepwise.max_width, stepwise.max_height));
                                    }
                                }
                            }
                            if !size_strs.is_empty() {
                                log!("    Resolutions: {}", size_strs.join(", "));
                            }
                        }
                    }
                }
            } else {
                log!("  ⚠️ Could not open device.");
            }
        }
    }

    log!("\n=== 2. Active Configuration Validation ===");
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

    log!("Configured model: \x1B[1;33m{}\x1B[0m", model_name);
    log!("Configured backend: \x1B[1;33m{}\x1B[0m", backend_name);
    log!("Configured device: \x1B[1;33m{}\x1B[0m", device_name);

    if backend_name == "ort" {
        if device_name != "CPU" {
            log!("\n\x1B[1;31m⚠️ CONFIGURATION WARNING ⚠️\x1B[0m");
            log!(
                "You have the ONNX Runtime (`ort`) backend configured but requested the device `{}`.",
                device_name
            );
            log!("The current ONNX Runtime compilation only supports CPU execution.");
            log!("The model will execute on the CPU, ignoring this setting.");
            log!(
                "💡 Solution: Change the backend configuration to OpenVINO to use native hardware acceleration."
            );
        } else {
            log!("✅ Consistent ORT configuration (CPU).");
        }
    } else if backend_name == "openvino" {
        if available_devices.contains(&device_name.to_string())
            || (device_name == "GPU" && available_devices.iter().any(|d| d.starts_with("GPU")))
        {
            log!(
                "✅ OPTIMAL: OpenVINO will load the model directly onto your hardware accelerator ({}).",
                device_name
            );
        } else if device_name == "CPU" {
            log!("✅ Consistent OpenVINO configuration (CPU).");
        } else {
            log!("\n\x1B[1;31m⚠️ DEVICE NOT FOUND WARNING ⚠️\x1B[0m");
            log!(
                "You configured OpenVINO to use `{}` but this hardware was not detected in the scan above.",
                device_name
            );
            log!(
                "OpenVINO will fail to start and will likely fall back slowly to CPU."
            );
        }
    }

    log!("\n=== 3. Load and Dry-Run Benchmark (Live Inference) ===");
    log!(
        "Attempting to load model `{}`, loading weights into memory...",
        model_name
    );

    let t_start_load = Instant::now();
    let mut session = match models::load_recognizer(&config.models) {
        Ok(s) => s,
        Err(e) => {
            log!("\n\x1B[1;31m❌ FATAL ERROR INSTANTIATING MODEL ❌\x1B[0m");
            log!("The engine returned the following error:");
            log!("  {}", e);
            return Ok(());
        }
    };
    let t_load = t_start_load.elapsed();
    log!("✅ Model successfully compiled and instantiated on hardware.");
    log!(
        "⏱️ Compilation/startup latency: \x1B[1;32m{:.2} ms\x1B[0m",
        t_load.as_secs_f32() * 1000.0
    );

    log!("\nPreparing warm-up inference (Dry-Run)...");

    // Create a fake buffer for 112x112 grayscale inference
    let w = 112;
    let h = 112;
    let dummy_image = vec![128u8; (w * h) as usize];
    let dummy_bbox = BoundingBox {
        x: 0,
        y: 0,
        width: w,
        height: h,
    };

    let t_start_infer = Instant::now();
    match session.get_embedding(&dummy_image, w, h, dummy_bbox) {
        Ok(emb) => {
            let t_infer = t_start_infer.elapsed();
            log!("✅ Dry-Run inference executed successfully.");
            log!(
                "Feature vector dimension (Embedding): {}",
                emb.len()
            );
            log!(
                "⏱️ Raw AI engine latency on this hardware: \x1B[1;32m{:.2} ms\x1B[0m",
                t_infer.as_secs_f32() * 1000.0
            );

            if t_infer.as_secs_f32() * 1000.0 > 150.0 {
                log!(
                    "\n⚠️ NOTE: Latency above 150 ms typically indicates that the model is running on the CPU instead of the NPU/GPU, or that the CPU is very slow."
                );
            }
        }
        Err(e) => {
            log!("\n\x1B[1;31m❌ ERROR DURING INFERENCE ❌\x1B[0m");
            log!("The model loaded but crashed when trying to process a tensor:");
            log!("  {}", e);
        }
    }

    log!("\n=== Diagnosis Completed ===");

    Ok(())
}
