pub mod arcface;
pub mod mobilefacenet;
#[cfg(feature = "openvino")]
pub mod openvino_rec;
pub mod rustface;
pub mod traits;

use crate::config::{Backend, ModelsConfig, RecognizerModel};
use crate::error::FacePamError;
#[cfg(feature = "openvino")]
use openvino::DeviceType;
pub use traits::{BoundingBox, FaceDetector, FaceRecognizer, Runtime};

pub fn load_recognizer(
    config: &ModelsConfig,
) -> Result<Box<dyn FaceRecognizer + Send>, FacePamError> {
    let recognizer_path = config.get_recognizer_path();

    match config.get_recognizer_model() {
        #[cfg(feature = "openvino")]
        RecognizerModel::MobileFaceNet(Backend::Openvino(device_model)) => {
            let (device, runtime) = match device_model {
                Runtime::Cpu => (DeviceType::CPU, Runtime::Cpu),
                Runtime::Gpu => (DeviceType::GPU, Runtime::Gpu),
                Runtime::Npu => (DeviceType::NPU, Runtime::Npu),
            };
            match openvino_rec::OpenVINORecognizer::new(
                &recognizer_path,
                device,
                runtime,
                112,
                "input.1",
                "516",
                openvino_rec::NormalizationType::MobileFaceNet,
            ) {
                Ok(rec) => Ok(Box::new(rec)),
                Err(e) => {
                    log::warn!(
                        "OpenVINO failed on {:?}: {}. Trying OpenVINO fallback to CPU...",
                        device_model,
                        e
                    );
                    if device_model != Runtime::Cpu {
                        if let Ok(rec) = openvino_rec::OpenVINORecognizer::new(
                            &recognizer_path,
                            DeviceType::CPU,
                            Runtime::Cpu,
                            112,
                            "input.1",
                            "516",
                            openvino_rec::NormalizationType::MobileFaceNet,
                        ) {
                            return Ok(Box::new(rec));
                        }
                    }
                    log::warn!(
                        "OpenVINO fallback to CPU failed (perhaps OpenVINO is not installed). Falling back to ONNX Runtime (CPU)..."
                    );
                    let rec = mobilefacenet::MobileFaceNet::new(&recognizer_path)?;
                    Ok(Box::new(rec))
                }
            }
        }
        RecognizerModel::MobileFaceNet(Backend::Ort) => {
            let rec = mobilefacenet::MobileFaceNet::new(&recognizer_path)?;
            Ok(Box::new(rec))
        }
        #[cfg(feature = "openvino")]
        RecognizerModel::ArcFace(Backend::Openvino(device_model)) => {
            let (device, runtime) = match device_model {
                Runtime::Cpu => (DeviceType::CPU, Runtime::Cpu),
                Runtime::Gpu => (DeviceType::GPU, Runtime::Gpu),
                Runtime::Npu => (DeviceType::NPU, Runtime::Npu),
            };
            match openvino_rec::OpenVINORecognizer::new(
                &recognizer_path,
                device,
                runtime,
                112,
                "data",
                "fc1",
                openvino_rec::NormalizationType::ArcFace,
            ) {
                Ok(rec) => Ok(Box::new(rec)),
                Err(e) => {
                    log::warn!(
                        "OpenVINO failed on {:?}: {}. Trying OpenVINO fallback to CPU...",
                        device_model,
                        e
                    );
                    if device_model != Runtime::Cpu {
                        if let Ok(rec) = openvino_rec::OpenVINORecognizer::new(
                            &recognizer_path,
                            DeviceType::CPU,
                            Runtime::Cpu,
                            112,
                            "data",
                            "fc1",
                            openvino_rec::NormalizationType::ArcFace,
                        ) {
                            return Ok(Box::new(rec));
                        }
                    }
                    log::warn!(
                        "OpenVINO fallback to CPU failed (perhaps OpenVINO is not installed). Falling back to ONNX Runtime (CPU)..."
                    );
                    let rec = arcface::ArcFace::new(&recognizer_path)?;
                    Ok(Box::new(rec))
                }
            }
        }
        RecognizerModel::ArcFace(Backend::Ort) => {
            let rec = arcface::ArcFace::new(&recognizer_path)?;
            Ok(Box::new(rec))
        }
        #[cfg(not(feature = "openvino"))]
        RecognizerModel::MobileFaceNet(Backend::Openvino(_)) | RecognizerModel::ArcFace(Backend::Openvino(_)) => {
            Err(FacePamError::Model("OpenVINO support not compiled in".to_string()))
        }
    }
}
