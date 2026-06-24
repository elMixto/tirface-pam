use super::traits::{BoundingBox, FaceRecognizer, Runtime};
use crate::error::FacePamError;
use image::{GrayImage, imageops::FilterType};
use ndarray::Array4;
use ort::{session::Session, value::Value};

pub const INPUT_SIZE: u32 = 112;

pub struct ArcFace {
    session: Session,
    runtime: Runtime,
}

impl ArcFace {
    pub fn new(model_path: &str) -> Result<Self, FacePamError> {
        let actual_path = if std::path::Path::new(model_path).exists() {
            model_path
        } else {"src/models/arcface/arcface.onnx"
};

        let session = Session::builder()
            .map_err(|e| FacePamError::Model(format!("ORT builder error: {}", e)))?
            .commit_from_file(actual_path)
            .map_err(|e| {
                FacePamError::Model(format!(
                    "Failed to load ArcFace from {}: {}",
                    actual_path, e
                ))
            })?;

        Ok(Self {
            session,
            runtime: Runtime::Cpu,
        }) // ORT CPU by default for now
    }
}

impl FaceRecognizer for ArcFace {
    fn get_embedding(
        &mut self,
        image: &[u8],
        width: u32,
        height: u32,
        bbox: BoundingBox,
    ) -> Result<Vec<f32>, FacePamError> {
        let x = bbox.x.max(0) as u32;
        let y = bbox.y.max(0) as u32;
        let w = bbox.width.min(width.saturating_sub(x));
        let h = bbox.height.min(height.saturating_sub(y));

        if w == 0 || h == 0 {
            return Err(FacePamError::Model(
                "Invalid bounding box size".to_string(),
            ));
        }

        let img = GrayImage::from_raw(width, height, image.to_vec()).ok_or_else(|| {
            FacePamError::Model("Failed to create image from raw buffer".to_string())
        })?;

        let cropped = image::imageops::crop_imm(&img, x, y, w, h).to_image();
        let resized =
            image::imageops::resize(&cropped, INPUT_SIZE, INPUT_SIZE, FilterType::Triangle);

        let mut tensor = Array4::<f32>::zeros((1, 3, INPUT_SIZE as usize, INPUT_SIZE as usize));
        for cy in 0..INPUT_SIZE {
            for cx in 0..INPUT_SIZE {
                let pixel = resized.get_pixel(cx, cy)[0] as f32;
                // Standard ArcFace normalization
                let normalized = (pixel - 127.5) / 127.5;
                tensor[[0, 0, cy as usize, cx as usize]] = normalized;
                tensor[[0, 1, cy as usize, cx as usize]] = normalized;
                tensor[[0, 2, cy as usize, cx as usize]] = normalized;
            }
        }

        let val = Value::from_array(tensor)
            .map_err(|e| FacePamError::Model(format!("Failed to create tensor: {}", e)))?;
        let inputs = ort::inputs!["data" => val];
        let outputs = self
            .session
            .run(inputs)
            .map_err(|e| FacePamError::Model(format!("ONNX execution failed: {}", e)))?;

        let (_, data): (_, &[f32]) = outputs["fc1"].try_extract_tensor::<f32>().map_err(|e| {
            FacePamError::Model(format!("Failed to extract output tensor: {}", e))
        })?;

        Ok(data.to_vec())
    }

    fn compare(&self, emb1: &[f32], emb2: &[f32]) -> f32 {
        let mut dot = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;

        for (x, y) in emb1.iter().zip(emb2.iter()) {
            dot += x * y;
            norm_a += x * x;
            norm_b += y * y;
        }

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot / (norm_a.sqrt() * norm_b.sqrt())
    }

    fn get_runtime(&self) -> Runtime {
        self.runtime
    }
}
