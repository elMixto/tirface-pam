// --- COMMON CONSTANTS & HELPERS ---

use ratatui::{
    style::{Color, Style},
    widgets::{Block, Borders, Widget},
};

pub const POSES: [&str; 5] = [
    "FRENTE (Neutral)",
    "ARRIBA (Levanta el mentón)",
    "ABAJO (Baja el mentón)",
    "IZQUIERDA (Gira un poco)",
    "DERECHA (Gira un poco)",
];
pub const FRAMES_PER_POSE: usize = 3;
pub const TARGET_FRAMES: usize = POSES.len() * FRAMES_PER_POSE;
pub const CAPTURE_DELAY_MS: u128 = 400;

#[inline(always)]
pub fn get_gray(buf: &[u8], x: usize, y: usize, width: usize) -> u8 {
    let idx = y * width + x;
    if idx < buf.len() { buf[idx] } else { 0 }
}

#[inline(always)]
pub fn yuyv_to_rgb(y: u8, u: u8, v: u8) -> (u8, u8, u8) {
    let y = y as i32;
    let u = u as i32 - 128;
    let v = v as i32 - 128;
    let r = (y + ((1370705 * v) >> 20)).clamp(0, 255) as u8;
    let g = (y - ((337633 * u) >> 20) - ((698001 * v) >> 20)).clamp(0, 255) as u8;
    let b = (y + ((1732446 * u) >> 20)).clamp(0, 255) as u8;
    (r, g, b)
}

#[inline(always)]
pub fn get_rgb_from_yuyv(buf: &[u8], x: usize, y: usize, width: usize) -> (u8, u8, u8) {
    let macro_pixel_idx = (y * width + x) / 2 * 4;
    if macro_pixel_idx + 3 >= buf.len() {
        return (0, 0, 0);
    }
    let u = buf[macro_pixel_idx + 1];
    let v = buf[macro_pixel_idx + 3];
    let y_val = if x.is_multiple_of(2) {
        buf[macro_pixel_idx]
    } else {
        buf[macro_pixel_idx + 2]
    };
    yuyv_to_rgb(y_val, u, v)
}

// --- RATATUI CAMERA WIDGET ---

pub struct CameraWidget<'a> {
    pub buf_rgb: Option<&'a [u8]>,
    pub buf_ir: &'a [u8],
    pub rgb_width: usize,
    pub rgb_height: usize,
    pub ir_width: usize,
    pub ir_height: usize,
    pub view_is_ir: bool,
    pub zoom_factor: f32,
    pub current_bbox: Option<(i32, i32, u32, u32)>,
    pub box_color: (u8, u8, u8),
    pub thickness: i32,
    pub blend_box: bool,
    pub title: Option<String>,
}

impl<'a> Widget for CameraWidget<'a> {
    fn render(self, area: ratatui::layout::Rect, buf: &mut ratatui::buffer::Buffer) {
        let block = Block::default()
            .borders(Borders::ALL)
            .title(self.title.unwrap_or_default());
        let inner_area = block.inner(area);
        block.render(area, buf);

        let cam_inner_w = inner_area.width as usize;
        let cam_inner_h = inner_area.height as usize;

        if cam_inner_w == 0 || cam_inner_h == 0 {
            return;
        }

        let px_w = cam_inner_w;
        let px_h = cam_inner_h * 2;

        let render_zoom = if self.view_is_ir { 1.0 } else { self.zoom_factor };
        let crop_w = if self.view_is_ir { self.ir_width } else { (self.rgb_width as f32 / render_zoom) as usize };
        let crop_h = if self.view_is_ir { self.ir_height } else { (self.rgb_height as f32 / render_zoom) as usize };
        let offset_x = if self.view_is_ir { 0 } else { (self.rgb_width - crop_w) / 2 };
        let offset_y = if self.view_is_ir { 0 } else { (self.rgb_height - crop_h) / 2 };

        for ty in 0..cam_inner_h {
            for tx in 0..cam_inner_w {
                let top_img_x = offset_x + crop_w.saturating_sub(1).saturating_sub(tx * crop_w / px_w);
                let top_img_y = offset_y + ((ty * 2) * crop_h / px_h);
                let bot_img_x = top_img_x;
                let bot_img_y = offset_y + ((ty * 2 + 1) * crop_h / px_h);

                let mut tr; let mut tg; let mut tb;
                let mut br; let mut bg; let mut bb;

                if self.view_is_ir {
                    tr = get_gray(self.buf_ir, top_img_x, top_img_y, self.ir_width);
                    br = get_gray(self.buf_ir, bot_img_x, bot_img_y, self.ir_width);
                    tg = tr; tb = tr;
                    bg = br; bb = br;
                } else if let Some(buf_rgb) = self.buf_rgb {
                    let t_rgb = get_rgb_from_yuyv(buf_rgb, top_img_x, top_img_y, self.rgb_width);
                    let b_rgb = get_rgb_from_yuyv(buf_rgb, bot_img_x, bot_img_y, self.rgb_width);
                    tr = t_rgb.0; tg = t_rgb.1; tb = t_rgb.2;
                    br = b_rgb.0; bg = b_rgb.1; bb = b_rgb.2;
                } else {
                    tr = 0; tg = 0; tb = 0;
                    br = 0; bg = 0; bb = 0;
                }

                if let Some((x, y, w, h)) = self.current_bbox {
                    let (left, right, top, bottom) = if self.view_is_ir {
                        (x, x + w as i32, y, y + h as i32)
                    } else {
                        let scale_x = self.rgb_width as f32 / self.ir_width as f32;
                        let scale_y = self.rgb_height as f32 / self.ir_height as f32;
                        let left = (x as f32 * scale_x) as i32;
                        let top = (y as f32 * scale_y) as i32;
                        (left, left + (w as f32 * scale_x) as i32, top, top + (h as f32 * scale_y) as i32)
                    };

                    let apply_bbox = |px: usize, py: usize, r: &mut u8, g: &mut u8, b: &mut u8| {
                        let px_i32 = px as i32;
                        let py_i32 = py as i32;
                        if px_i32 >= left && px_i32 <= right && py_i32 >= top && py_i32 <= bottom &&
                            (px_i32 < left + self.thickness || px_i32 > right - self.thickness || py_i32 < top + self.thickness || py_i32 > bottom - self.thickness) {
                            if self.blend_box {
                                *r = ((self.box_color.0 as f32 * 0.8) + (*r as f32 * 0.2)) as u8;
                                *g = ((self.box_color.1 as f32 * 0.8) + (*g as f32 * 0.2)) as u8;
                                *b = ((self.box_color.2 as f32 * 0.8) + (*b as f32 * 0.2)) as u8;
                            } else {
                                *r = self.box_color.0; *g = self.box_color.1; *b = self.box_color.2;
                            }
                        }
                    };

                    apply_bbox(top_img_x, top_img_y, &mut tr, &mut tg, &mut tb);
                    apply_bbox(bot_img_x, bot_img_y, &mut br, &mut bg, &mut bb);
                }

                let cell_x = inner_area.x + tx as u16;
                let cell_y = inner_area.y + ty as u16;
                if cell_x < buf.area.width && cell_y < buf.area.height {
                    let cell = &mut buf[(cell_x, cell_y)];
                    cell.set_char('▀');
                    cell.set_style(Style::default()
                        .fg(Color::Rgb(tr, tg, tb))
                        .bg(Color::Rgb(br, bg, bb)));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hint::black_box;
    use std::time::Instant;

    #[test]
    fn test_yuyv_to_rgb_benchmark() {
        println!("\n=== BENCHMARK: yuyv_to_rgb ===");
        
        let resolutions = [(640, 480), (640, 360)];
        let iterations = 1000;

        for (width, height) in resolutions {
            let num_pixels = width * height;
            let buffer_size = num_pixels * 2;
            
            // Crear un frame de prueba YUYV con un patrón simple
            let mut yuyv_buf = vec![0u8; buffer_size];
            for i in 0..buffer_size {
                yuyv_buf[i] = (i % 256) as u8;
            }

            println!("Resolución: {}x{} ({} píxeles)", width, height, num_pixels);
            println!("Tamaño del buffer YUYV: {} bytes", buffer_size);
            println!("Ejecutando {} iteraciones...", iterations);

            let start = Instant::now();
            
            for _ in 0..iterations {
                for y in 0..height {
                    for x in 0..width {
                        let rgb = get_rgb_from_yuyv(&yuyv_buf, x, y, width);
                        black_box(rgb);
                    }
                }
            }

            let elapsed = start.elapsed();
            let total_ms = elapsed.as_secs_f64() * 1000.0;
            let avg_frame_ms = total_ms / iterations as f64;
            let fps = 1000.0 / avg_frame_ms;
            let mpps = (num_pixels as f64 * iterations as f64) / (elapsed.as_secs_f64() * 1_000_000.0);

            println!("⏱️  Tiempo total: {:.2} ms", total_ms);
            println!("⏱️  Promedio por frame: {:.4} ms", avg_frame_ms);
            println!("🚀 Rendimiento equivalente: {:.2} FPS", fps);
            println!("📈 Ancho de banda de píxeles: {:.2} MP/s (Megapíxeles por segundo)\n", mpps);
        }
    }
}
