use crate::inference;
use image::RgbImage;
use ort::session::Session;
use serde::Serialize;

// ---------------------------------------------------------------------------
// AlpenDetect — YOLOv11-based accident + vehicle detector
//
// The model is a fine-tuned YOLO11 checkpoint (80 MB ONNX) that outputs a
// [1, 14, 8400] tensor (4 bbox coords + 10 class scores per anchor).
//
// Class layout (same as the upstream training labels):
//   0  bike          ← vehicle
//   1  accident      ← accident
//   2  bike_accident ← accident
//   3  bike_crash    ← accident
//   4  car           ← vehicle
//   5  car_accident  ← accident
//   6  car_crash     ← accident
//   7  collision     ← accident
//   8  crash         ← accident
//   9  person        ← vehicle / pedestrian (counted but not alerted)
//
// Test-time augmentation (TTA) strategy — controlled by ALPEN_DETECT_TTA=1
// (on by default).  For each frame we run four passes and merge with NMS:
//   1. Standard 640×640 full-frame pass
//   2. Horizontal-flip pass (catches left-biased accident orientations)
//   3. Left-crop: leftmost 75% of the frame (cols 0..480) scaled to 640×640
//   4. Right-crop: rightmost 75% of the frame (cols 160..640) scaled to 640×640
// Passes 3 and 4 let the model see side events at a larger effective scale.
// ---------------------------------------------------------------------------

pub const ALPEN_DETECT_MODEL_PATH: &str = "assets/alpen_detect.onnx";
pub const ALPEN_DETECT_NUM_CLASSES: usize = 10;
pub const ALPEN_DETECT_NUM_ANCHORS: usize = 8400;
/// Class IDs whose detections trigger an accident alert.
pub const ALPEN_DETECT_ACCIDENT_CLASSES: [usize; 7] = [1, 2, 3, 5, 6, 7, 8];
/// Class IDs counted as vehicles for ROI congestion.
pub const ALPEN_DETECT_VEHICLE_CLASSES: [usize; 2] = [0, 4];
/// Pedestrian/person class — counted but not alerted.
pub const ALPEN_DETECT_PERSON_CLASS: usize = 9;
/// Minimum confidence for an accident class to fire an alert.
pub const ALPEN_DETECT_ACCIDENT_CONF: f32 = 0.85;
/// Minimum confidence for a vehicle/person class to appear in ROI counts.
pub const ALPEN_DETECT_VEHICLE_CONF: f32 = 0.35;
/// IoU threshold for NMS post-processing.
pub const ALPEN_DETECT_NMS_IOU: f32 = 0.45;

// Width of each side crop in pixels (out of 640). 480 = 75%.
const SIDE_CROP_W: usize = 480;

#[derive(Clone, Debug, Serialize)]
pub struct AccidentReport {
    pub detected_at: f64,
    pub involved_ids: Vec<i32>,
    pub iou: f32,
    pub pred_iou: f32,
    pub speed_1: f32,
    pub speed_2: f32,
    pub accel_1: f32,
    pub accel_2: f32,
    pub location_bbox: [f32; 4],
    pub reasons: Vec<String>,
    pub confidence: String,
    /// Track ID from the IoU tracker (populated on the live stream path).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub track_id: Option<u32>,
    /// Accumulated EMA confidence for this track (populated on the live stream path).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accumulated_conf: Option<f32>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct DetectionSummary {
    pub detections: Vec<[f32; 6]>,
    pub accident_reports: Vec<AccidentReport>,
    pub vehicle_detections: Vec<[f32; 6]>,
    pub used_night_enhancement: bool,
    pub dark_ratio: f32,
    pub max_accident_confidence: f32,
}

pub fn confidence_threshold_for_class(class_id: usize) -> f32 {
    if ALPEN_DETECT_ACCIDENT_CLASSES.contains(&class_id) {
        ALPEN_DETECT_ACCIDENT_CONF
    } else if ALPEN_DETECT_VEHICLE_CLASSES.contains(&class_id)
        || class_id == ALPEN_DETECT_PERSON_CLASS
    {
        ALPEN_DETECT_VEHICLE_CONF
    } else {
        1.01
    }
}

pub fn preprocess_rgb(rgb: &[u8]) -> ndarray::ArrayD<f32> {
    ndarray::Array::from_shape_vec((640, 640, 3), rgb.to_vec())
        .expect("Failed to reshape 640x640 RGB frame")
        .mapv(|v| v as f32 / 255.0)
        .permuted_axes([2, 0, 1])
        .insert_axis(ndarray::Axis(0))
        .into_dyn()
}

// ---------------------------------------------------------------------------
// Core inference helpers
// ---------------------------------------------------------------------------

/// Run a single ONNX pass, returning decoded + confidence-filtered boxes
/// (no NMS yet — callers that aggregate multi-pass results do NMS once at end).
fn infer_raw(session: &mut Session, rgb: &[u8]) -> anyhow::Result<Vec<[f32; 6]>> {
    let input = preprocess_rgb(rgb);
    let input_tensor = ort::value::Value::from_array(input)?;
    let mut io_binding = session.create_binding()?;
    io_binding.bind_input("images", &input_tensor)?;
    io_binding.bind_output_to_device("output0", session.allocator().memory_info())?;
    let outputs = session.run_binding(&io_binding)?;
    let raw = outputs["output0"].try_extract_tensor::<f32>()?.1.to_vec();
    Ok(inference::decode_output(&raw))
}

/// Standard single-pass inference (decoded + NMS). Used when TTA is disabled.
pub fn run_session(session: &mut Session, rgb: &[u8]) -> anyhow::Result<Vec<[f32; 6]>> {
    let boxes = infer_raw(session, rgb)?;
    Ok(inference::apply_nms(boxes, ALPEN_DETECT_NMS_IOU))
}

// ---------------------------------------------------------------------------
// TTA helpers
// ---------------------------------------------------------------------------

/// Mirror a 640×640 RGB frame horizontally (in-place copy).
fn flip_horizontal_rgb(rgb: &[u8]) -> Vec<u8> {
    const W: usize = 640;
    const H: usize = 640;
    let mut out = vec![0u8; W * H * 3];
    for y in 0..H {
        for x in 0..W {
            let src = (y * W + x) * 3;
            let dst = (y * W + (W - 1 - x)) * 3;
            out[dst] = rgb[src];
            out[dst + 1] = rgb[src + 1];
            out[dst + 2] = rgb[src + 2];
        }
    }
    out
}

/// Crop a horizontal strip of width `crop_w` starting at column `x_start` from a
/// 640×640 RGB frame and scale it back to 640×640 using bilinear interpolation.
/// Returns the scaled frame and the parameters needed to map boxes back.
fn crop_and_scale(rgb: &[u8], x_start: usize, crop_w: usize) -> Vec<u8> {
    const FULL: usize = 640;
    let mut out = vec![0u8; FULL * FULL * 3];
    let scale = crop_w as f32 / FULL as f32; // < 1.0 means we're zooming in

    for dy in 0..FULL {
        for dx in 0..FULL {
            // Source coordinate in the original 640×640 frame
            let src_xf = x_start as f32 + dx as f32 * scale;
            let src_x0 = src_xf.floor() as usize;
            let src_x1 = (src_x0 + 1).min(FULL - 1);
            let tx = src_xf - src_x0 as f32;

            let src_y = dy; // no vertical crop

            let p00 = pixel(rgb, src_x0, src_y);
            let p10 = pixel(rgb, src_x1, src_y);

            let dst = (dy * FULL + dx) * 3;
            for c in 0..3 {
                out[dst + c] = lerp(p00[c], p10[c], tx);
            }
        }
    }
    out
}

#[inline]
fn pixel(rgb: &[u8], x: usize, y: usize) -> [u8; 3] {
    let i = (y * 640 + x) * 3;
    [rgb[i], rgb[i + 1], rgb[i + 2]]
}

#[inline]
fn lerp(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 + (b as f32 - a as f32) * t).round() as u8
}

/// Map a box from a side-crop's 640×640 coordinate space back to the
/// original 640×640 frame coordinate space.
///
/// crop boxes x' ∈ [0,640] → original x = x_start + x' * (crop_w / 640.0)
fn remap_boxes_to_original(
    boxes: Vec<[f32; 6]>,
    x_start: usize,
    crop_w: usize,
) -> Vec<[f32; 6]> {
    let scale = crop_w as f32 / 640.0;
    boxes
        .into_iter()
        .map(|d| {
            let x1 = x_start as f32 + d[0] * scale;
            let x2 = x_start as f32 + d[2] * scale;
            [x1.clamp(0.0, 640.0), d[1], x2.clamp(0.0, 640.0), d[3], d[4], d[5]]
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Public multi-pass entry point
// ---------------------------------------------------------------------------

/// Returns `true` when test-time augmentation is enabled.
/// Controlled by the `ALPEN_DETECT_TTA` env var (default: on).
pub fn tta_enabled() -> bool {
    !matches!(
        std::env::var("ALPEN_DETECT_TTA")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "0" | "false" | "off" | "no"
    )
}

/// Multi-pass inference with TTA (flip + two side crops) followed by a single NMS.
/// Falls back to a plain single pass when `ALPEN_DETECT_TTA=0`.
pub fn run_session_augmented(session: &mut Session, rgb: &[u8]) -> anyhow::Result<Vec<[f32; 6]>> {
    if !tta_enabled() {
        return run_session(session, rgb);
    }

    // Pass 1 — standard full-frame
    let mut all = infer_raw(session, rgb)?;

    // Pass 2 — horizontal flip; mirror x back to original space
    let flipped = flip_horizontal_rgb(rgb);
    let flip_boxes = infer_raw(session, &flipped)?;
    all.extend(flip_boxes.into_iter().map(|d| {
        [640.0 - d[2], d[1], 640.0 - d[0], d[3], d[4], d[5]]
    }));

    // Pass 3 — left 75% crop (cols 0..480) zoomed to 640×640
    let left_crop = crop_and_scale(rgb, 0, SIDE_CROP_W);
    let left_boxes = remap_boxes_to_original(infer_raw(session, &left_crop)?, 0, SIDE_CROP_W);
    all.extend(left_boxes);

    // Pass 4 — right 75% crop (cols 160..640) zoomed to 640×640
    let right_start = 640 - SIDE_CROP_W;
    let right_crop = crop_and_scale(rgb, right_start, SIDE_CROP_W);
    let right_boxes =
        remap_boxes_to_original(infer_raw(session, &right_crop)?, right_start, SIDE_CROP_W);
    all.extend(right_boxes);

    Ok(inference::apply_nms(all, ALPEN_DETECT_NMS_IOU))
}

// ---------------------------------------------------------------------------
// Summarise detections into the high-level output struct
// ---------------------------------------------------------------------------

pub fn summarize_detections(
    detections: Vec<[f32; 6]>,
    detected_at: f64,
    used_night_enhancement: bool,
    dark_ratio: f32,
) -> DetectionSummary {
    let mut accident_reports = Vec::new();
    let mut vehicle_detections = Vec::new();
    let mut max_accident_confidence = 0.0f32;

    for det in &detections {
        let class_id = det[5] as usize;
        if ALPEN_DETECT_VEHICLE_CLASSES.contains(&class_id) {
            vehicle_detections.push(*det);
        }

        if !ALPEN_DETECT_ACCIDENT_CLASSES.contains(&class_id)
            || det[4] < ALPEN_DETECT_ACCIDENT_CONF
        {
            continue;
        }

        max_accident_confidence = max_accident_confidence.max(det[4]);
        accident_reports.push(AccidentReport {
            detected_at,
            involved_ids: vec![class_id as i32],
            iou: 0.0,
            pred_iou: 0.0,
            speed_1: 0.0,
            speed_2: 0.0,
            accel_1: 0.0,
            accel_2: 0.0,
            location_bbox: [det[0], det[1], det[2], det[3]],
            reasons: vec![
                format!("AlpenDetect accident class {}", class_id),
                format!("confidence {:.3}", det[4]),
                if used_night_enhancement {
                    format!("night enhancement active (dark_ratio={:.3})", dark_ratio)
                } else {
                    format!("day/original frame path (dark_ratio={:.3})", dark_ratio)
                },
            ],
            confidence: confidence_bucket(det[4]).to_string(),
            track_id: None,
            accumulated_conf: None,
        });
    }

    DetectionSummary {
        detections,
        accident_reports,
        vehicle_detections,
        used_night_enhancement,
        dark_ratio,
        max_accident_confidence,
    }
}

// ---------------------------------------------------------------------------
// Visualisation helper
// ---------------------------------------------------------------------------

pub fn draw_detection_boxes(frame: &mut RgbImage, boxes: &[[f32; 6]], color: image::Rgb<u8>) {
    let (w, h) = frame.dimensions();
    if w == 0 || h == 0 {
        return;
    }

    for det in boxes {
        let x1 = det[0].floor().clamp(0.0, (w - 1) as f32) as u32;
        let y1 = det[1].floor().clamp(0.0, (h - 1) as f32) as u32;
        let x2 = det[2].ceil().clamp(0.0, (w - 1) as f32) as u32;
        let y2 = det[3].ceil().clamp(0.0, (h - 1) as f32) as u32;
        if x2 <= x1 || y2 <= y1 {
            continue;
        }

        for x in x1..=x2 {
            frame.put_pixel(x, y1, color);
            frame.put_pixel(x, y2, color);
        }
        for y in y1..=y2 {
            frame.put_pixel(x1, y, color);
            frame.put_pixel(x2, y, color);
        }
    }
}

fn confidence_bucket(score: f32) -> &'static str {
    if score >= 0.95 {
        "very_high"
    } else if score >= 0.90 {
        "high"
    } else {
        "medium"
    }
}
