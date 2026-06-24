pub use crate::config::Runtime;
use crate::error::FacePamError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundingBox {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

pub trait FaceDetector {
    /// Detects faces in a grayscale image and returns the bounding boxes.
    fn detect(
        &mut self,
        image: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Vec<BoundingBox>, FacePamError>;

    /// Convenience method to get the largest face (best candidate for authentication)
    fn detect_best(
        &mut self,
        image: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Option<BoundingBox>, FacePamError> {
        let faces = self.detect(image, width, height)?;
        Ok(faces.into_iter().max_by_key(|f| f.width * f.height))
    }
}

pub trait FaceRecognizer {
    /// Given a cropped region (bounding box) of an image, extract the feature embedding.
    fn get_embedding(
        &mut self,
        image: &[u8],
        width: u32,
        height: u32,
        bbox: BoundingBox,
    ) -> Result<Vec<f32>, FacePamError>;

    /// Compare two embeddings and return a similarity score.
    /// Typically, higher is more similar (e.g., Cosine Similarity).
    fn compare(&self, emb1: &[f32], emb2: &[f32]) -> f32;

    /// Returns the runtime this recognizer is currently using.
    fn get_runtime(&self) -> Runtime;
}
