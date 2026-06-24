pub mod arcface;
pub mod mobilefacenet;
pub mod openvino_rec;
pub mod rustface;
pub mod traits;

use crate::config::{Backend, ModelsConfig, RecognizerModel};
use crate::error::FacePamError;
use openvino::DeviceType;
pub use traits::{BoundingBox, FaceDetector, FaceRecognizer, Runtime};

pub fn load_recognizer(
    config: &ModelsConfig,
) -> Result<Box<dyn FaceRecognizer + Send>, FacePamError> {
    let recognizer_path = config.get_recognizer_path();

    match config.get_recognizer_model() {
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
                    if device_model != Runtime::Cpu {
                        log::warn!(
                            "OpenVINO failed on {:?}, trying fallback to CPU...",
                            device_model
                        );
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
                    Err(e)
                }
            }
        }
        RecognizerModel::MobileFaceNet(Backend::Ort) => {
            let rec = mobilefacenet::MobileFaceNet::new(&recognizer_path)?;
            Ok(Box::new(rec))
        }
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
                    if device_model != Runtime::Cpu {
                        log::warn!(
                            "OpenVINO failed on {:?}, trying fallback to CPU...",
                            device_model
                        );
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
                    Err(e)
                }
            }
        }
        RecognizerModel::ArcFace(Backend::Ort) => {
            let rec = arcface::ArcFace::new(&recognizer_path)?;
            Ok(Box::new(rec))
        }
    }
}
