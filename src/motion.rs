// ---------------------------------------------------------------------------
// motion.rs — Per-ROI similarity pre-filter
//
// Provides fast frame hashing for static-scene detection.  When all ROIs in a
// stream show negligible motion, the YOLO pass can be skipped entirely and the
// tracker coasts instead.
// ---------------------------------------------------------------------------

/// Default similarity threshold below which a frame region is considered static.
pub fn skip_threshold() -> f32 {
    std::env::var("ALPEN_SKIP_THRESHOLD")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.02)
}

/// Compute a 64-byte perceptual hash of a 640×640 RGB frame.
///
/// Algorithm: downsample to 8×8 grayscale (average-pooling per block), yielding
/// 64 luminance bytes.  Cheap and deterministic.
pub fn frame_hash_64(rgb: &[u8]) -> [u8; 64] {
    const W: usize = 640;
    const H: usize = 640;
    const BLOCK_W: usize = W / 8; // 80
    const BLOCK_H: usize = H / 8; // 80
    const BLOCK_PIXELS: usize = BLOCK_W * BLOCK_H;

    let mut hash = [0u8; 64];
    if rgb.len() < W * H * 3 {
        return hash;
    }

    for by in 0..8 {
        for bx in 0..8 {
            let mut sum = 0u64;
            let y0 = by * BLOCK_H;
            let x0 = bx * BLOCK_W;
            for dy in 0..BLOCK_H {
                let row = y0 + dy;
                for dx in 0..BLOCK_W {
                    let col = x0 + dx;
                    let base = (row * W + col) * 3;
                    // BT.601 luminance approximation using integer math
                    let lum = (77 * rgb[base] as u64
                        + 150 * rgb[base + 1] as u64
                        + 29 * rgb[base + 2] as u64)
                        >> 8;
                    sum += lum;
                }
            }
            hash[by * 8 + bx] = (sum / BLOCK_PIXELS as u64).min(255) as u8;
        }
    }

    hash
}

/// Normalised mean absolute difference between two 64-byte hashes.
/// Returns a value in [0.0, 1.0].
pub fn hash_delta(a: &[u8; 64], b: &[u8; 64]) -> f32 {
    let total: u32 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs())
        .sum();
    total as f32 / (64.0 * 255.0)
}

/// Compute a hash for a specific ROI region of the frame.
///
/// Finds the axis-aligned bounding box of the polygon, crops that region from the
/// full 640×640 frame, then hashes it in the same 8×8 block scheme.
pub fn roi_hash(rgb: &[u8], _img_w: u32, _img_h: u32, roi_poly: &[[i32; 2]]) -> [u8; 64] {
    const W: usize = 640;
    const H: usize = 640;

    if roi_poly.is_empty() || rgb.len() < W * H * 3 {
        return frame_hash_64(rgb);
    }

    // Axis-aligned bounding box of the ROI polygon
    let mut min_x = i32::MAX;
    let mut min_y = i32::MAX;
    let mut max_x = i32::MIN;
    let mut max_y = i32::MIN;
    for pt in roi_poly {
        min_x = min_x.min(pt[0]);
        min_y = min_y.min(pt[1]);
        max_x = max_x.max(pt[0]);
        max_y = max_y.max(pt[1]);
    }
    let x0 = (min_x.max(0) as usize).min(W - 1);
    let y0 = (min_y.max(0) as usize).min(H - 1);
    let x1 = (max_x.max(0) as usize + 1).min(W);
    let y1 = (max_y.max(0) as usize + 1).min(H);
    let crop_w = x1.saturating_sub(x0).max(1);
    let crop_h = y1.saturating_sub(y0).max(1);

    // Hash the crop in an 8×8 block grid
    let block_w = crop_w.max(8) / 8;
    let block_h = crop_h.max(8) / 8;
    let block_pixels = block_w.max(1) * block_h.max(1);

    let mut hash = [0u8; 64];
    for by in 0..8 {
        for bx in 0..8 {
            let mut sum = 0u64;
            let ry0 = y0 + by * block_h;
            let rx0 = x0 + bx * block_w;
            for dy in 0..block_h {
                let row = ry0 + dy;
                if row >= y1 {
                    break;
                }
                for dx in 0..block_w {
                    let col = rx0 + dx;
                    if col >= x1 {
                        break;
                    }
                    let base = (row * W + col) * 3;
                    if base + 2 < rgb.len() {
                        let lum = (77 * rgb[base] as u64
                            + 150 * rgb[base + 1] as u64
                            + 29 * rgb[base + 2] as u64)
                            >> 8;
                        sum += lum;
                    }
                }
            }
            hash[by * 8 + bx] = (sum / block_pixels as u64).min(255) as u8;
        }
    }

    hash
}

/// Check whether all ROIs are below the motion threshold.
/// Returns `true` if the frame should be skipped (all ROIs static).
///
/// For streams with no ROIs, falls back to a full-frame hash comparison.
pub fn should_skip_frame(
    rgb: &[u8],
    rois: &std::collections::HashMap<String, Vec<[i32; 2]>>,
    prev_roi_hashes: &std::collections::HashMap<String, [u8; 64]>,
    prev_full_hash: Option<&[u8; 64]>,
) -> (bool, std::collections::HashMap<String, [u8; 64]>, [u8; 64]) {
    let threshold = skip_threshold();
    let full_hash = frame_hash_64(rgb);

    if rois.is_empty() {
        // Fallback: full-frame hash
        let skip = if let Some(prev) = prev_full_hash {
            hash_delta(prev, &full_hash) < threshold
        } else {
            false
        };
        return (skip, std::collections::HashMap::new(), full_hash);
    }

    let mut new_hashes = std::collections::HashMap::with_capacity(rois.len());
    let mut all_static = true;

    for (name, poly) in rois {
        let h = roi_hash(rgb, 640, 640, poly);
        if let Some(prev) = prev_roi_hashes.get(name) {
            if hash_delta(prev, &h) >= threshold {
                all_static = false;
            }
        } else {
            // First frame for this ROI — can't skip
            all_static = false;
        }
        new_hashes.insert(name.clone(), h);
    }

    (all_static, new_hashes, full_hash)
}
