use serde::Deserialize;
use std::fs;
use std::path::Path;

const SYSTEM_CONFIG_PATH: &str = crate::paths::SYSTEM_CONFIG_PATH;

#[derive(Deserialize, Debug, Clone)]
pub struct Config {
    pub camera: CameraConfig,
    pub recognition: RecognitionConfig,
    pub daemon: DaemonConfig,
    #[serde(default)]
    pub models: ModelsConfig,
}

//CameraPath Stuff

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CameraPath {
    Auto,
    Custom(String),
}

impl<'de> serde::Deserialize<'de> for CameraPath {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        if s.to_lowercase() == "auto" {
            Ok(CameraPath::Auto)
        } else {
            Ok(CameraPath::Custom(s))
        }
    }
}

impl std::fmt::Display for CameraPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CameraPath::Auto => write!(f, "auto"),
            CameraPath::Custom(path) => write!(f, "{}", path),
        }
    }
}

#[derive(Deserialize, Debug, Clone)]
pub struct CameraConfig {
    pub ir_device: CameraPath,
    pub rgb_device: CameraPath,
    pub warmup_ms: u64,
}

#[derive(Deserialize, Debug, Clone)]
pub struct RecognitionConfig {
    pub threshold: f32,
}

#[derive(Deserialize, Debug, Clone)]
pub struct DaemonConfig {
    pub auth_timeout_sec: u64,
    pub secure_enclave: bool,
}

#[derive(Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Runtime {
    Cpu,
    Gpu,
    Npu,
}

#[derive(Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    Ort,
    Openvino(Runtime),
}

#[derive(Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum RecognizerModel {
    MobileFaceNet(Backend),
    ArcFace(Backend),
}

#[derive(Deserialize, Debug, Clone)]
pub struct ModelsConfig {
    #[serde(default = "default_detector_path")]
    pub detector_path: String,

    #[serde(default = "default_recognizer_model")]
    pub recognizer_model: String,

    #[serde(default = "default_recognizer_backend")]
    pub recognizer_backend: String,

    #[serde(default = "default_recognizer_device")]
    pub recognizer_device: String,
}

impl ModelsConfig {
    pub fn get_backend(&self) -> Backend {
        let runtime = match self.recognizer_device.to_uppercase().as_str() {
            "NPU" => Runtime::Npu,
            "GPU" => Runtime::Gpu,
            "CPU" | _ => Runtime::Cpu,
        };

        match self.recognizer_backend.to_lowercase().as_str() {
            "openvino" => Backend::Openvino(runtime),
            "ort" | _ => Backend::Ort,
        }
    }

    pub fn get_recognizer_model(&self) -> RecognizerModel {
        let backend = self.get_backend();
        if self.recognizer_model.to_lowercase().contains("arcface") {
            RecognizerModel::ArcFace(backend)
        } else {
            RecognizerModel::MobileFaceNet(backend)
        }
    }

    pub fn model_name(&self) -> &'static str {
        match self.get_recognizer_model() {
            RecognizerModel::MobileFaceNet(_) => "mobilefacenet",
            RecognizerModel::ArcFace(_) => "arcface",
        }
    }

    pub fn get_recognizer_path(&self) -> String {
        // If it's a valid path directly
        if std::path::Path::new(&self.recognizer_model).exists() {
            return self.recognizer_model.clone();
        }

        let model_name = self.model_name();
        let fallback = format!("src/models/{}/{}.onnx", model_name, model_name);

        if std::path::Path::new(&fallback).exists() {
            fallback
        } else {
            format!("{}/{}.onnx", crate::paths::SYSTEM_MODELS_DIR, model_name)
        }
    }
}

fn default_detector_path() -> String {
    format!("{}/seeta_fd_frontal_v1.0.bin", crate::paths::SYSTEM_MODELS_DIR)
}

fn default_recognizer_model() -> String {
    "mobilefacenet".to_string()
}

fn default_recognizer_backend() -> String {
    "ort".to_string()
}

fn default_recognizer_device() -> String {
    "CPU".to_string()
}

impl Default for ModelsConfig {
    fn default() -> Self {
        ModelsConfig {
            detector_path: default_detector_path(),
            recognizer_model: default_recognizer_model(),
            recognizer_backend: default_recognizer_backend(),
            recognizer_device: default_recognizer_device(),
        }
    }
}


impl Default for Config {
    fn default() -> Self {
        Config {
            camera: CameraConfig {
                ir_device: CameraPath::Auto,
                rgb_device: CameraPath::Auto,
                warmup_ms: 300,
            },
            recognition: RecognitionConfig { threshold: 0.60 },
            daemon: DaemonConfig {
                auth_timeout_sec: 5,
                secure_enclave: false,
            },
            models: ModelsConfig::default(),
        }
    }
}

impl Config {
    pub fn load() -> Self {
        let config_path = SYSTEM_CONFIG_PATH;

        if !Path::new(config_path).exists() {
            return Config::default();
        }

        let content = fs::read_to_string(config_path).unwrap_or_default();
        toml::from_str(&content).unwrap_or_else(|e| {
            eprintln!("Error reading configuration from {}: {}", config_path, e);
            Config::default()
        })
    }
}
