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

        let max_score = if car_score > bus_score { car_score } else { bus_score };
        
        if max_score > 0.50 { // Slightly higher threshold to prune faster
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

pub fn apply_nms(mut boxes: Vec<[f32; 6]>, iou_threshold: f32) -> Vec<[f32; 6]> {
    // Sort by score descending
    boxes.sort_by(|a, b| b[4].partial_cmp(&a[4]).unwrap_or(std::cmp::Ordering::Equal));
    
    let mut kept = Vec::new();
    let mut suppressed = vec![false; boxes.len()];
    
    for i in 0..boxes.len() {
        if suppressed[i] { continue; }
        
        kept.push(boxes[i]);
        
        for j in (i + 1)..boxes.len() {
            if suppressed[j] { continue; }
            
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
    
    if area_a + area_b - intersection <= 0.0 { return 0.0; }
    
    intersection / (area_a + area_b - intersection)
}
