pub mod device;
pub mod ir_emitter;
pub mod stream;

pub use device::{Camera, CameraType};
pub use ir_emitter::IrEmitter;
pub use stream::CameraManager;

/// Applies a histogram equalization to a grayscale image buffer to normalize
/// contrast and exposure, making face detection and recognition more robust.
pub fn equalize_grayscale_image(image: &[u8]) -> Vec<u8> {
    if image.is_empty() {
        return Vec::new();
    }
    let mut hist = [0u32; 256];
    let total_pixels = image.len() as f32;

    // 1. Calculate the histogram
    for &pixel in image {
        hist[pixel as usize] += 1;
    }

    // 2. Calculate the Cumulative Distribution Function (CDF)
    let mut cdf = [0f32; 256];
    let mut sum = 0.0;
    for i in 0..256 {
        sum += hist[i] as f32;
        cdf[i] = sum / total_pixels;
    }

    // Find the first non-zero CDF value for normalization
    let mut min_cdf = 0.0;
    for i in 0..256 {
        if cdf[i] > 0.0 {
            min_cdf = cdf[i];
            break;
        }
    }

    // 3. Map pixels to the new equalized scale
    let mut equalized = vec![0u8; image.len()];
    if total_pixels > 1.0 && min_cdf < 1.0 {
        for i in 0..image.len() {
            let val = ((cdf[image[i] as usize] - min_cdf) / (1.0 - min_cdf) * 255.0).round() as u8;
            equalized[i] = val;
        }
    } else {
        for i in 0..image.len() {
            equalized[i] = (cdf[image[i] as usize] * 255.0).round() as u8;
        }
    }

    equalized
}
