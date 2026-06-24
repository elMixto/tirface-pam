use super::traits::{BoundingBox, FaceDetector};
use crate::error::FacePamError;
use rustface::{Detector, ImageData};

pub struct RustfaceDetector {
    detector: Box<dyn Detector>,
}

impl RustfaceDetector {
    pub fn new(model_path: &str) -> Result<Self, FacePamError> {
        let actual_path = if std::path::Path::new(model_path).exists() {
            model_path
        } else {
            "src/models/rustface/seeta_fd_frontal_v1.0.bin" // Fallback local
        };

        match rustface::create_detector(actual_path) {
            Ok(mut detector) => {
                detector.set_min_face_size(40);
                detector.set_score_thresh(1.0);
                detector.set_pyramid_scale_factor(0.8);
                detector.set_slide_window_step(4, 4);
                Ok(Self { detector })
            }
            Err(e) => Err(FacePamError::Model(format!(
                "Failed to load face detector from {}: {}",
                actual_path, e
            ))),
        }
    }
}

impl FaceDetector for RustfaceDetector {
    fn detect(
        &mut self,
        image: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Vec<BoundingBox>, FacePamError> {
        let img_data = ImageData::new(image, width, height);
        let faces = self.detector.detect(&img_data);

        let mut bboxes = Vec::new();
        for face in faces {
            let bbox = face.bbox();
            bboxes.push(BoundingBox {
                x: bbox.x(),
                y: bbox.y(),
                width: bbox.width(),
                height: bbox.height(),
            });
        }

        Ok(bboxes)
    }
}
