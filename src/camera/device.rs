use crate::config::CameraPath;
use crate::error::FacePamError;
use v4l::context;
use v4l::prelude::*;
use v4l::video::Capture;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraType {
    Rgb,
    Ir,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FormatKind {
    Ir,
    Color,
    Unknown,
}

impl FormatKind {
    fn from_fourcc(fourcc: &[u8; 4]) -> Self {
        match fourcc {
            b"GREY" | b"Y10 " | b"Y12 " | b"Y16 " => Self::Ir,
            b"YUYV" | b"MJPG" | b"RGB3" | b"BGR3" => Self::Color,
            _ => Self::Unknown,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeviceSupport {
    PureColor,
    Hybrid,
    PureIr,
    Unsupported,
}

impl DeviceSupport {
    fn from_device_path(path: &std::path::Path) -> Self {
        let Some(dev) = Device::with_path(path).ok() else {
            return Self::Unsupported;
        };
        let Some(formats) = dev.enum_formats().ok() else {
            return Self::Unsupported;
        };

        let mut supports_ir = false;
        let mut supports_color = false;

        for fmt in formats {
            match FormatKind::from_fourcc(&fmt.fourcc.repr) {
                FormatKind::Ir => supports_ir = true,
                FormatKind::Color => supports_color = true,
                FormatKind::Unknown => {}
            }
        }

        match (supports_color, supports_ir) {
            (true, false) => Self::PureColor,
            (true, true) => Self::Hybrid,
            (false, true) => Self::PureIr,
            (false, false) => Self::Unsupported,
        }
    }

    fn score(self, camera_type: CameraType) -> u32 {
        match (camera_type, self) {
            (CameraType::Rgb, Self::PureColor) => 2,
            (CameraType::Rgb, Self::Hybrid) => 1,
            (CameraType::Ir, Self::PureIr) => 2,
            (CameraType::Ir, Self::Hybrid) => 1,
            _ => 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Camera {
    pub path: String,
    pub camera_type: CameraType,
}

impl Camera {
    pub fn new(config_path: &CameraPath, camera_type: CameraType) -> Result<Self, FacePamError> {
        let path = match config_path {
            CameraPath::Auto => Self::find_camera(camera_type).ok_or_else(|| {
                FacePamError::Camera(format!(
                    "Auto-detection failed: No {} camera found",
                    match camera_type {
                        CameraType::Rgb => "RGB",
                        CameraType::Ir => "IR",
                    }
                ))
            })?,
            CameraPath::Custom(path) => path.clone(),
        };

        Ok(Self { path, camera_type })
    }

    pub fn find_camera(camera_type: CameraType) -> Option<String> {
        context::enum_devices()
            .into_iter()
            .filter_map(|node| {
                let support = DeviceSupport::from_device_path(node.path());
                let score = support.score(camera_type);
                if score > 0 {
                    Some((node.path().to_string_lossy().to_string(), score))
                } else {
                    None
                }
            })
            .max_by_key(|&(_, score)| score)
            .map(|(path, _)| path)
    }
}
