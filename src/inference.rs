use serde::Serialize;
use std::collections::HashMap;

pub type RoiMap = HashMap<String, Vec<[i32; 2]>>;

#[derive(Clone, Debug, Serialize)]
pub struct TrafficLightMetric {
    pub color: String,
    pub nearest_roi: String,
    pub confidence: f32,
}

pub fn decode_output(raw_data: &[f32]) -> Vec<[f32; 6]> {
    // Pre-allocate space for results to avoid resizing the Vec in the loop
    let mut results = Vec::with_capacity(50);
    let num_anchors = 8400;

    for col in 0..num_anchors {
        // Directly check car (2) and bus (5) scores ONLY
        // In YOLOv8 [1, 84, 8400], rows 4-83 are class scores.
        // Row 4+2=6 is Car, Row 4+5=9 is Bus.
        let car_score = raw_data[6 * num_anchors + col];
        let bus_score = raw_data[9 * num_anchors + col];

        let max_score = if car_score > bus_score {
            car_score
        } else {
            bus_score
        };

        if max_score > 0.50 {
            // Slightly higher threshold to prune faster
            let best_class = if car_score > bus_score { 2.0 } else { 5.0 };

            let cx = raw_data[0 * num_anchors + col];
            let cy = raw_data[1 * num_anchors + col];
            let w = raw_data[2 * num_anchors + col];
            let h = raw_data[3 * num_anchors + col];

            // Convert cx,cy,w,h to x1,y1,x2,y2 (Still in 640x640 space)
            let x1 = cx - w / 2.0;
            let y1 = cy - h / 2.0;
            let x2 = cx + w / 2.0;
            let y2 = cy + h / 2.0;

            results.push([x1, y1, x2, y2, max_score, best_class]);
        }
    }

    results
}

pub fn compute_roi_vehicle_counts(
    detections: &[[f32; 6]],
    rois: &RoiMap,
    original_w: u32,
    original_h: u32,
) -> HashMap<String, u32> {
    let mut counts: HashMap<String, u32> = rois.keys().map(|k| (k.clone(), 0)).collect();
    if rois.is_empty() {
        return counts;
    }

    let mut roi_keys: Vec<String> = rois.keys().cloned().collect();
    roi_keys.sort();

    for det in detections {
        let cx = ((det[0] + det[2]) * 0.5) * original_w as f32 / 640.0;
        let cy = ((det[1] + det[3]) * 0.5) * original_h as f32 / 640.0;

        for key in &roi_keys {
            let poly = match rois.get(key) {
                Some(p) => p,
                None => continue,
            };
            if point_in_polygon(cx, cy, poly) {
                if let Some(v) = counts.get_mut(key) {
                    *v += 1;
                }
                break;
            }
        }
    }

    counts
}

pub fn compute_congestion_levels(counts: &HashMap<String, u32>) -> HashMap<String, String> {
    counts
        .iter()
        .map(|(k, v)| {
            let level = if *v >= 10 {
                "High"
            } else if *v >= 5 {
                "Medium"
            } else {
                "Low"
            };
            (k.clone(), level.to_string())
        })
        .collect()
}

pub fn detect_traffic_light_state(
    frame: &image::RgbImage,
    rois: &RoiMap,
) -> Vec<TrafficLightMetric> {
    let (w, h) = frame.dimensions();
    if w == 0 || h == 0 {
        return Vec::new();
    }

    let scan_h = ((h as f32) * 0.65) as u32;
    let step: u32 = if w > 1000 { 3 } else { 2 };

    let mut red_count: u32 = 0;
    let mut red_x_sum: f32 = 0.0;
    let mut red_y_sum: f32 = 0.0;

    let mut yellow_count: u32 = 0;
    let mut yellow_x_sum: f32 = 0.0;
    let mut yellow_y_sum: f32 = 0.0;

    let mut green_count: u32 = 0;
    let mut green_x_sum: f32 = 0.0;
    let mut green_y_sum: f32 = 0.0;

    let mut y: u32 = 0;
    while y < scan_h {
        let mut x: u32 = 0;
        while x < w {
            let p = frame.get_pixel(x, y).0;
            let (hue, sat, val) = rgb_to_hsv(p[0], p[1], p[2]);

            if val > 0.32 && sat > 0.35 {
                if !(15.0..=345.0).contains(&hue) {
                    red_count += 1;
                    red_x_sum += x as f32;
                    red_y_sum += y as f32;
                } else if (25.0..=65.0).contains(&hue) {
                    yellow_count += 1;
                    yellow_x_sum += x as f32;
                    yellow_y_sum += y as f32;
                } else if (80.0..=160.0).contains(&hue) {
                    green_count += 1;
                    green_x_sum += x as f32;
                    green_y_sum += y as f32;
                }
            }

            x += step;
        }
        y += step;
    }

    let min_pixels = ((w * scan_h) as f32 * 0.00008).max(10.0) as u32;
    let candidates = [
        ("Red", red_count, red_x_sum, red_y_sum),
        ("Yellow", yellow_count, yellow_x_sum, yellow_y_sum),
        ("Green", green_count, green_x_sum, green_y_sum),
    ];

    let (color, count, x_sum, y_sum) = match candidates.into_iter().max_by_key(|(_, c, _, _)| *c) {
        Some(top) if top.1 >= min_pixels => top,
        _ => return Vec::new(),
    };

    let cx = x_sum / count as f32;
    let cy = y_sum / count as f32;
    let nearest_roi = nearest_roi_label(cx, cy, rois).unwrap_or_else(|| "unknown".to_string());
    let confidence = ((count as f32) / (min_pixels as f32 * 6.0)).clamp(0.3, 0.99);

    vec![TrafficLightMetric {
        color: color.to_string(),
        nearest_roi,
        confidence,
    }]
}

fn nearest_roi_label(x: f32, y: f32, rois: &RoiMap) -> Option<String> {
    if rois.is_empty() {
        return None;
    }

    for (label, poly) in rois {
        if point_in_polygon(x, y, poly) {
            return Some(label.clone());
        }
    }

    let mut best_label: Option<String> = None;
    let mut best_dist = f32::MAX;
    for (label, poly) in rois {
        if poly.is_empty() {
            continue;
        }
        let (cx, cy) = polygon_centroid(poly);
        let dx = cx - x;
        let dy = cy - y;
        let dist = dx * dx + dy * dy;
        if dist < best_dist {
            best_dist = dist;
            best_label = Some(label.clone());
        }
    }

    best_label
}

fn polygon_centroid(poly: &[[i32; 2]]) -> (f32, f32) {
    let mut sx = 0.0;
    let mut sy = 0.0;
    let n = poly.len().max(1) as f32;
    for p in poly {
        sx += p[0] as f32;
        sy += p[1] as f32;
    }
    (sx / n, sy / n)
}

fn point_in_polygon(x: f32, y: f32, poly: &[[i32; 2]]) -> bool {
    if poly.len() < 3 {
        return false;
    }

    let mut inside = false;
    let mut j = poly.len() - 1;
    for i in 0..poly.len() {
        let xi = poly[i][0] as f32;
        let yi = poly[i][1] as f32;
        let xj = poly[j][0] as f32;
        let yj = poly[j][1] as f32;

        let intersects = ((yi > y) != (yj > y))
            && (x < (xj - xi) * (y - yi) / ((yj - yi).abs().max(f32::EPSILON)) + xi);

        if intersects {
            inside = !inside;
        }
        j = i;
    }

    inside
}

fn rgb_to_hsv(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let rf = r as f32 / 255.0;
    let gf = g as f32 / 255.0;
    let bf = b as f32 / 255.0;

    let c_max = rf.max(gf).max(bf);
    let c_min = rf.min(gf).min(bf);
    let delta = c_max - c_min;

    let hue = if delta <= f32::EPSILON {
        0.0
    } else if (c_max - rf).abs() <= f32::EPSILON {
        60.0 * ((gf - bf) / delta).rem_euclid(6.0)
    } else if (c_max - gf).abs() <= f32::EPSILON {
        60.0 * (((bf - rf) / delta) + 2.0)
    } else {
        60.0 * (((rf - gf) / delta) + 4.0)
    };

    let sat = if c_max <= f32::EPSILON {
        0.0
    } else {
        delta / c_max
    };
    (hue, sat, c_max)
}

pub fn apply_nms(mut boxes: Vec<[f32; 6]>, iou_threshold: f32) -> Vec<[f32; 6]> {
    // Sort by score descending
    boxes.sort_by(|a, b| b[4].partial_cmp(&a[4]).unwrap_or(std::cmp::Ordering::Equal));

    let mut kept = Vec::new();
    let mut suppressed = vec![false; boxes.len()];

    for i in 0..boxes.len() {
        if suppressed[i] {
            continue;
        }

        kept.push(boxes[i]);

        for j in (i + 1)..boxes.len() {
            if suppressed[j] {
                continue;
            }

            if iou(&boxes[i], &boxes[j]) > iou_threshold {
                suppressed[j] = true;
            }
        }
    }

    kept
}

fn iou(box_a: &[f32; 6], box_b: &[f32; 6]) -> f32 {
    let x1 = box_a[0].max(box_b[0]);
    let y1 = box_a[1].max(box_b[1]);
    let x2 = box_a[2].min(box_b[2]);
    let y2 = box_a[3].min(box_b[3]);

    let inter_w = (x2 - x1).max(0.0);
    let inter_h = (y2 - y1).max(0.0);
    let intersection = inter_w * inter_h;

    let area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]);
    let area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]);

    if area_a + area_b - intersection <= 0.0 {
        return 0.0;
    }

    intersection / (area_a + area_b - intersection)
}
