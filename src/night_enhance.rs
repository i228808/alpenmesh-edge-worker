use image::{ImageBuffer, Rgb, RgbImage};

/// Path where an exported EnlightenGAN ONNX would be loaded from.
/// TODO: Load this via `ort` when the export lands — until then, tiled CLAHE
/// is the active enhancement path.
pub const ENLIGHTENGAN_MODEL_PATH: &str = "assets/enlighten_gan.onnx";
pub const NIGHT_RATIO_THRESH: f32 = 0.50;

#[derive(Clone, Debug)]
pub struct EnlightenGanSession;

impl EnlightenGanSession {
    pub fn new() -> Self {
        Self
    }

    /// Enhance a dark frame using tiled CLAHE.
    /// TODO: Load assets/enlighten_gan.onnx via ort when the export lands.
    pub fn enhance(&self, rgb_640: &[u8]) -> Vec<u8> {
        let clip: f32 = std::env::var("ALPEN_CLAHE_CLIP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3.0);
        let grid: usize = std::env::var("ALPEN_CLAHE_GRID")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8);
        tiled_clahe(rgb_640, (grid, grid), clip)
    }
}

pub fn dark_ratio(rgb: &[u8]) -> f32 {
    let mut hist = [0u32; 256];
    for chunk in rgb.chunks_exact(3) {
        let y = ((77 * chunk[0] as u32) + (150 * chunk[1] as u32) + (29 * chunk[2] as u32)) >> 8;
        hist[y.min(255) as usize] += 1;
    }
    let total: u32 = hist.iter().sum();
    let dark: u32 = hist[..25].iter().sum();
    dark as f32 / total.max(1) as f32
}

pub fn is_night_histogram(rgb: &[u8]) -> bool {
    dark_ratio(rgb) >= NIGHT_RATIO_THRESH
}

// ---------------------------------------------------------------------------
// Tiled CLAHE — Contrast Limited Adaptive Histogram Equalization
// ---------------------------------------------------------------------------

/// Proper tiled CLAHE implementation with bilinear interpolation between tiles.
///
/// Algorithm:
/// 1. Convert RGB → luminance (BT.601)
/// 2. Divide into `tile_grid` tiles
/// 3. For each tile: compute histogram, clip at `clip_limit` standard deviations
///    above uniform, redistribute clipped counts, build CDF-based LUT
/// 4. For each pixel: bilinear-interpolate between the 4 nearest tile LUTs
/// 5. Apply the equalized luminance as a gain on the original RGB
fn tiled_clahe(rgb_640: &[u8], tile_grid: (usize, usize), clip_limit: f32) -> Vec<u8> {
    const W: usize = 640;
    const H: usize = 640;
    let expected = W * H * 3;
    if rgb_640.len() < expected {
        return rgb_640.to_vec();
    }

    let (tiles_x, tiles_y) = tile_grid;
    let tiles_x = tiles_x.max(1);
    let tiles_y = tiles_y.max(1);
    let tile_w = W / tiles_x;
    let tile_h = H / tiles_y;
    if tile_w == 0 || tile_h == 0 {
        return rgb_640.to_vec();
    }

    // Step 1: compute luminance for every pixel
    let num_pixels = W * H;
    let mut lum = vec![0u8; num_pixels];
    for i in 0..num_pixels {
        let base = i * 3;
        let r = rgb_640[base] as f32;
        let g = rgb_640[base + 1] as f32;
        let b = rgb_640[base + 2] as f32;
        lum[i] = (0.299 * r + 0.587 * g + 0.114 * b).round().clamp(0.0, 255.0) as u8;
    }

    // Step 2: build per-tile LUTs
    let pixels_per_tile = tile_w * tile_h;
    let clip_count = (clip_limit * pixels_per_tile as f32 / 256.0).max(1.0) as u32;

    // LUT storage: tiles_y × tiles_x, each is [u8; 256]
    let mut luts = vec![[0u8; 256]; tiles_y * tiles_x];

    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let x0 = tx * tile_w;
            let y0 = ty * tile_h;

            // Histogram for this tile
            let mut hist = [0u32; 256];
            for dy in 0..tile_h {
                let row = y0 + dy;
                if row >= H {
                    break;
                }
                for dx in 0..tile_w {
                    let col = x0 + dx;
                    if col >= W {
                        break;
                    }
                    hist[lum[row * W + col] as usize] += 1;
                }
            }

            // Clip histogram and redistribute
            let mut excess = 0u32;
            for bin in hist.iter_mut() {
                if *bin > clip_count {
                    excess += *bin - clip_count;
                    *bin = clip_count;
                }
            }
            let incr = excess / 256;
            let remainder = (excess % 256) as usize;
            for (i, bin) in hist.iter_mut().enumerate() {
                *bin += incr;
                if i < remainder {
                    *bin += 1;
                }
            }

            // Build CDF → LUT
            let total: u32 = hist.iter().sum();
            let mut cdf = [0u32; 256];
            cdf[0] = hist[0];
            for i in 1..256 {
                cdf[i] = cdf[i - 1] + hist[i];
            }
            let cdf_min = cdf.iter().copied().find(|&v| v > 0).unwrap_or(0);
            let denom = total.saturating_sub(cdf_min).max(1);
            let lut = &mut luts[ty * tiles_x + tx];
            for i in 0..256 {
                lut[i] =
                    ((cdf[i].saturating_sub(cdf_min) as f64 * 255.0) / denom as f64).round() as u8;
            }
        }
    }

    // Step 3: for each pixel, bilinear interpolate between the 4 nearest tile LUTs
    let mut out = vec![0u8; expected];

    for y in 0..H {
        for x in 0..W {
            let l = lum[y * W + x];

            // Which tile center are we near?
            // Tile centers are at (tx * tile_w + tile_w/2, ty * tile_h + tile_h/2)
            let fx = (x as f32 - tile_w as f32 * 0.5) / tile_w as f32;
            let fy = (y as f32 - tile_h as f32 * 0.5) / tile_h as f32;

            let tx0 = fx.floor().max(0.0) as usize;
            let ty0 = fy.floor().max(0.0) as usize;
            let tx1 = (tx0 + 1).min(tiles_x - 1);
            let ty1 = (ty0 + 1).min(tiles_y - 1);

            let sx = (fx - tx0 as f32).clamp(0.0, 1.0);
            let sy = (fy - ty0 as f32).clamp(0.0, 1.0);

            let v00 = luts[ty0 * tiles_x + tx0][l as usize] as f32;
            let v10 = luts[ty0 * tiles_x + tx1][l as usize] as f32;
            let v01 = luts[ty1 * tiles_x + tx0][l as usize] as f32;
            let v11 = luts[ty1 * tiles_x + tx1][l as usize] as f32;

            let top = v00 + (v10 - v00) * sx;
            let bot = v01 + (v11 - v01) * sx;
            let equalized = top + (bot - top) * sy;

            // Apply as gain to original RGB
            let orig_lum = l.max(1) as f32;
            let gain = equalized / orig_lum;

            let base = (y * W + x) * 3;
            out[base] = (rgb_640[base] as f32 * gain).clamp(0.0, 255.0) as u8;
            out[base + 1] = (rgb_640[base + 1] as f32 * gain).clamp(0.0, 255.0) as u8;
            out[base + 2] = (rgb_640[base + 2] as f32 * gain).clamp(0.0, 255.0) as u8;
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Legacy: original global gamma stretch (kept for A/B comparison)
// ---------------------------------------------------------------------------

#[allow(dead_code)]
pub fn legacy_gamma_enhance(rgb_640: &[u8]) -> Vec<u8> {
    let Some(img) = ImageBuffer::<Rgb<u8>, _>::from_raw(640, 640, rgb_640.to_vec()) else {
        return rgb_640.to_vec();
    };

    let mut out: RgbImage = ImageBuffer::new(640, 640);
    let mut luminances = Vec::with_capacity((640 * 640) as usize);
    for pixel in img.pixels() {
        let y = (0.299 * pixel[0] as f32 + 0.587 * pixel[1] as f32 + 0.114 * pixel[2] as f32)
            .round()
            .clamp(0.0, 255.0) as u8;
        luminances.push(y);
    }

    let mut sorted = luminances.clone();
    sorted.sort_unstable();
    let p5 = sorted[(sorted.len() as f32 * 0.05) as usize] as f32;
    let p95 = sorted[(sorted.len() as f32 * 0.95) as usize] as f32;
    let denom = (p95 - p5).max(24.0);

    for (idx, pixel) in img.pixels().enumerate() {
        let y = luminances[idx] as f32;
        let gain = ((y - p5) / denom).clamp(0.0, 1.0).powf(0.8);
        let boost = 0.75 + gain * 1.10;
        let r = ((pixel[0] as f32).powf(0.92) * boost).clamp(0.0, 255.0) as u8;
        let g = ((pixel[1] as f32).powf(0.92) * boost).clamp(0.0, 255.0) as u8;
        let b = ((pixel[2] as f32).powf(0.92) * boost).clamp(0.0, 255.0) as u8;
        let x = (idx as u32) % 640;
        let y_coord = (idx as u32) / 640;
        out.put_pixel(x, y_coord, Rgb([r, g, b]));
    }

    out.into_raw()
}
