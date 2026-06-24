use crate::error::FacePamError;
use crate::models::traits::{BoundingBox, FaceRecognizer, Runtime};
use image::{GrayImage, imageops::FilterType};
use openvino::{Core, DeviceType, ElementType, Shape, Tensor};

pub struct OpenVINORecognizer {
    infer_request: openvino::InferRequest,
    input_shape: Shape,
    runtime: Runtime,
    input_size: u32,
    input_name: String,
    output_name: String,
    normalization_type: NormalizationType,
}

unsafe impl Send for OpenVINORecognizer {}

#[derive(Clone, Copy)]
pub enum NormalizationType {
    MobileFaceNet, // (pixel - 127.5) / 128.0
    ArcFace,       // (pixel - 127.5) / 127.5
}

impl OpenVINORecognizer {
    pub fn new(
        model_path: &str,
        device: DeviceType<'static>,
        runtime_enum: Runtime,
        input_size: u32,
        input_name: &str,
        output_name: &str,
        norm_type: NormalizationType,
    ) -> Result<Self, FacePamError> {
        let mut core = Core::new().map_err(|e| {
            FacePamError::Model(format!("Error inicializando OpenVINO Core: {}", e))
        })?;

        let model_data = std::fs::read(model_path).map_err(|e| {
            FacePamError::Model(format!("No se pudo leer el archivo del modelo: {}", e))
        })?;

        // OpenVINO lee archivos ONNX nativamente desde un buffer
        let model = core
            .read_model_from_buffer(&model_data, None)
            .map_err(|e| {
                FacePamError::Model(format!("Error en read_model de OpenVINO: {}", e))
            })?;

        let device_str = device.to_string();
        let mut compiled_model = core.compile_model(&model, device).map_err(|e| {
            FacePamError::Model(format!(
                "Error compilando modelo para {}: {}",
                device_str, e
            ))
        })?;

        let infer_request = compiled_model
            .create_infer_request()
            .map_err(|e| FacePamError::Model(format!("Error creando infer_request: {}", e)))?;

        let input_shape =
            Shape::new(&[1, 3, input_size as i64, input_size as i64]).map_err(|e| {
                FacePamError::Model(format!("Error definiendo dimensiones de entrada: {}", e))
            })?;

        Ok(Self {
            infer_request,
            input_shape,
            runtime: runtime_enum,
            input_size,
            input_name: input_name.to_string(),
            output_name: output_name.to_string(),
            normalization_type: norm_type,
        })
    }
}

impl FaceRecognizer for OpenVINORecognizer {
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
        let resized = image::imageops::resize(
            &cropped,
            self.input_size,
            self.input_size,
            FilterType::Triangle,
        );

        // Crear el tensor de OpenVINO de tipo f32
        let mut input_tensor = Tensor::new(ElementType::F32, &self.input_shape)
            .map_err(|e| FacePamError::Model(e.to_string()))?;

        let data = input_tensor
            .get_data_mut::<f32>()
            .map_err(|e| FacePamError::Model(e.to_string()))?;

        let size = self.input_size as usize;
        for cy in 0..size {
            for cx in 0..size {
                let pixel = resized.get_pixel(cx as u32, cy as u32)[0] as f32;

                let normalized = match self.normalization_type {
                    NormalizationType::MobileFaceNet => (pixel - 127.5) / 128.0,
                    NormalizationType::ArcFace => (pixel - 127.5) / 127.5,
                };

                let idx_base = cy * size + cx;
                let channel_stride = size * size;
                data[idx_base] = normalized; // Canal R
                data[idx_base + channel_stride] = normalized; // Canal G
                data[idx_base + 2 * channel_stride] = normalized; // Canal B
            }
        }

        self.infer_request
            .set_tensor(&self.input_name, &input_tensor)
            .map_err(|e| FacePamError::Model(e.to_string()))?;

        self.infer_request
            .infer()
            .map_err(|e| FacePamError::Model(e.to_string()))?;

        let output_tensor = self
            .infer_request
            .get_tensor(&self.output_name)
            .map_err(|e| FacePamError::Model(e.to_string()))?;

        let output_data: &[f32] = output_tensor
            .get_data::<f32>()
            .map_err(|e| FacePamError::Model(e.to_string()))?;

        Ok(output_data.to_vec())
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
