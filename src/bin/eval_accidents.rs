use std::ffi::OsStr;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use alpenmesh_worker::{accident_debug, alpen_detect, motion, state::WorkerState, tracker::{Tracker, TrackEvent}};
use std::collections::HashMap;
use anyhow::{Context, Result};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "eval_accidents")]
struct Args {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    negatives: Option<PathBuf>,
    #[arg(long, default_value_t = 10)]
    fps: u32,
    #[arg(long)]
    max_videos: Option<usize>,
    #[arg(long, default_value = "eval_runs")]
    out: PathBuf,
    /// Enable extended metrics columns (tracked vehicles, dwell, stalled, skip reason)
    #[arg(long, default_value_t = false)]
    emit_metrics: bool,
    /// When emit_metrics is set, also engage the motion pre-filter (skips YOLO on static frames)
    #[arg(long, default_value_t = false)]
    skip_motion: bool,
}

#[derive(Default, Debug, Clone)]
struct PerVideo {
    label: &'static str,
    video_stem: String,
    duration_sec: f64,
    frames_processed: u64,
    mean_detect_ms: f64,
    nirikshan_day_fires: u64,
    nirikshan_day_first_fire_sec: Option<f64>,
    nirikshan_night_enhanced_fires: u64,
    nirikshan_night_enhanced_first_fire_sec: Option<f64>,
    max_day_confidence: f32,
    max_night_confidence: f32,
    was_night_by_histogram: bool,
    used_enhancement: bool,
    // -- Phase 1 extended metrics --
    total_tracked_vehicles: u64,
    mean_dwell_sec: f64,
    stalled_events: u64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if std::env::var("ACCIDENT_DEBUG").is_err() {
        unsafe {
            std::env::set_var("ACCIDENT_DEBUG", "1");
        }
    }

    let run_dir = args.out.join(run_stamp());
    let frames_root = run_dir.join("frames");
    fs::create_dir_all(&frames_root).context("create eval frames dir")?;
    let mut results = csv::Writer::from_path(run_dir.join("results.csv"))?;
    let mut timeline = csv::Writer::from_path(run_dir.join("timeline.csv"))?;

    {
        let mut results_header = vec![
            "label",
            "video",
            "duration_sec",
            "frames_processed",
            "mean_detect_ms",
            "night_histogram_video",
            "used_enhancement",
            "nirikshan_day_fires",
            "nirikshan_day_first_fire_sec",
            "nirikshan_day_max_conf",
            "nirikshan_night_enhanced_fires",
            "nirikshan_night_enhanced_first_fire_sec",
            "nirikshan_night_enhanced_max_conf",
        ];
        if args.emit_metrics {
            results_header.extend_from_slice(&[
                "negatives_tested",
                "false_positives",
                "total_tracked_vehicles",
                "mean_dwell_sec",
                "stalled_events",
            ]);
        }
        results.write_record(&results_header)?;
    }
    {
        let mut timeline_header = vec![
            "label",
            "video",
            "frame_idx",
            "sec",
            "path",
            "dark_ratio",
            "fire_count",
            "max_accident_conf",
            "used_night_enhancement",
        ];
        if args.emit_metrics {
            timeline_header.extend_from_slice(&[
                "skip_reason",
                "dwell_time_sec",
                "stalled_flag",
                "tracked_vehicle_count",
            ]);
        }
        timeline.write_record(&timeline_header)?;
    }

    let state = WorkerState::new();
    let mut videos = Vec::new();
    videos.extend(list_videos(&args.input, "pos")?);
    if let Some(neg) = &args.negatives {
        videos.extend(list_videos(neg, "neg")?);
    }
    if let Some(max) = args.max_videos {
        videos.truncate(max);
    }
    if videos.is_empty() {
        anyhow::bail!("No videos found");
    }

    let emit_metrics = args.emit_metrics;
    let skip_motion = args.skip_motion;
    let mut all = Vec::new();
    for (path, label) in videos {
        let metrics = eval_one(&state, &path, label, args.fps, &frames_root, &mut timeline, emit_metrics, skip_motion)?;
        let mut row = vec![
            metrics.label.to_string(),
            metrics.video_stem.clone(),
            format!("{:.3}", metrics.duration_sec),
            metrics.frames_processed.to_string(),
            format!("{:.3}", metrics.mean_detect_ms),
            metrics.was_night_by_histogram.to_string(),
            metrics.used_enhancement.to_string(),
            metrics.nirikshan_day_fires.to_string(),
            opt_f64(metrics.nirikshan_day_first_fire_sec),
            format!("{:.3}", metrics.max_day_confidence),
            metrics.nirikshan_night_enhanced_fires.to_string(),
            opt_f64(metrics.nirikshan_night_enhanced_first_fire_sec),
            format!("{:.3}", metrics.max_night_confidence),
        ];
        if emit_metrics {
            row.extend_from_slice(&[
                if label == "neg" { "1" } else { "0" }.to_string(),
                if label == "neg" && metrics.nirikshan_day_fires > 0 { "1" } else { "0" }.to_string(),
                metrics.total_tracked_vehicles.to_string(),
                format!("{:.3}", metrics.mean_dwell_sec),
                metrics.stalled_events.to_string(),
            ]);
        }
        results.write_record(&row)?;
        all.push(metrics);
    }

    let pos_total = all.iter().filter(|m| m.label == "pos").count() as f64;
    let neg_total = all.iter().filter(|m| m.label == "neg").count() as f64;
    let day_hits = all
        .iter()
        .filter(|m| m.label == "pos" && m.nirikshan_day_fires > 0)
        .count() as f64;
    let night_hits = all
        .iter()
        .filter(|m| m.label == "pos" && m.nirikshan_night_enhanced_fires > 0)
        .count() as f64;
    let neg_day_fp = all
        .iter()
        .filter(|m| m.label == "neg" && m.nirikshan_day_fires > 0)
        .count() as f64;
    let neg_night_fp = all
        .iter()
        .filter(|m| m.label == "neg" && m.nirikshan_night_enhanced_fires > 0)
        .count() as f64;

    let mut summary = format!(
        "# AlpenDetect Evaluation\n\n- positives: {}\n- negatives: {}\n- day recall: {:.2}%\n- night-enhanced recall: {:.2}%\n- day FP video rate: {:.2}%\n- night-enhanced FP video rate: {:.2}%\n",
        pos_total as usize,
        neg_total as usize,
        pct(day_hits, pos_total),
        pct(night_hits, pos_total),
        pct(neg_day_fp, neg_total),
        pct(neg_night_fp, neg_total),
    );
    if emit_metrics {
        let total_tracked: u64 = all.iter().map(|m| m.total_tracked_vehicles).sum();
        let mean_dwell = {
            let videos_with_dwell: Vec<f64> = all.iter().filter(|m| m.mean_dwell_sec > 0.0).map(|m| m.mean_dwell_sec).collect();
            if videos_with_dwell.is_empty() { 0.0 } else { videos_with_dwell.iter().sum::<f64>() / videos_with_dwell.len() as f64 }
        };
        let total_stalled: u64 = all.iter().map(|m| m.stalled_events).sum();
        summary.push_str(&format!(
            "\n## Tracker Metrics\n\n- total_tracked_vehicles: {}\n- mean_dwell_sec: {:.3}\n- stalled_events: {}\n",
            total_tracked, mean_dwell, total_stalled,
        ));
    }
    fs::write(run_dir.join("summary.md"), summary)?;
    println!("Wrote evaluation to {}", run_dir.display());
    Ok(())
}

fn eval_one(
    state: &WorkerState,
    video_path: &Path,
    label: &'static str,
    fps: u32,
    frames_root: &Path,
    timeline: &mut csv::Writer<std::fs::File>,
    emit_metrics: bool,
    skip_motion: bool,
) -> Result<PerVideo> {
    let mut child = spawn_ffmpeg(video_path, fps)?;
    let stdout = child.stdout.as_mut().context("ffmpeg stdout missing")?;
    let mut frame = vec![0u8; 640 * 640 * 3];
    let mut idx = 0u64;
    let mut total_ms = 0.0f64;
    let stem = video_path
        .file_stem()
        .and_then(OsStr::to_str)
        .unwrap_or("video")
        .to_string();
    let frame_dir = frames_root.join(&stem);
    fs::create_dir_all(&frame_dir)?;

    let mut metrics = PerVideo {
        label,
        video_stem: stem.clone(),
        ..PerVideo::default()
    };

    let mut tracker = Tracker::new();
    let mut total_entered: u64 = 0;
    let mut total_stalled: u64 = 0;
    let mut exited_dwell_times: Vec<f32> = Vec::new();
    let mut detection_count: u64 = 0;

    let empty_rois: HashMap<String, Vec<[i32; 2]>> = HashMap::new();
    let mut prev_roi_hashes: HashMap<String, [u8; 64]> = HashMap::new();
    let mut prev_full_hash: Option<[u8; 64]> = None;

    loop {
        if let Err(err) = stdout.read_exact(&mut frame) {
            if err.kind() == std::io::ErrorKind::UnexpectedEof {
                break;
            }
            return Err(err).context("read raw frame");
        }

        idx += 1;
        let sec = idx as f64 / fps.max(1) as f64;
        let dark_ratio = alpenmesh_worker::night_enhance::dark_ratio(&frame);

        if skip_motion && emit_metrics {
            let (skip, new_roi_hashes, new_full_hash) = motion::should_skip_frame(
                &frame,
                &empty_rois,
                &prev_roi_hashes,
                prev_full_hash.as_ref(),
            );
            prev_roi_hashes = new_roi_hashes;
            prev_full_hash = Some(new_full_hash);

            if skip {
                let coast_events = tracker.coast(idx, sec);
                for ev in &coast_events {
                    match ev {
                        TrackEvent::Entered { .. } => total_entered += 1,
                        TrackEvent::Exited { dwell_sec, .. } => exited_dwell_times.push(*dwell_sec),
                        TrackEvent::Stalled { .. } => total_stalled += 1,
                    }
                }
                metrics.frames_processed += 1;
                metrics.duration_sec = sec;

                if emit_metrics {
                    let mut tl_row = vec![
                        label.to_string(),
                        stem.clone(),
                        idx.to_string(),
                        format!("{sec:.3}"),
                        "day".to_string(),
                        format!("{dark_ratio:.3}"),
                        "0".to_string(),
                        "0.000".to_string(),
                        "false".to_string(),
                    ];
                    tl_row.extend_from_slice(&[
                        "low_motion".to_string(),
                        "0.0".to_string(),
                        "0".to_string(),
                        tracker.active_track_count().to_string(),
                    ]);
                    timeline.write_record(&tl_row)?;
                }
                continue;
            }
        }

        let start = Instant::now();
        let day_dets = run_detector(state, &frame);
        let day_summary = alpen_detect::summarize_detections(day_dets, sec, false, dark_ratio);
        let final_summary = if dark_ratio >= alpenmesh_worker::night_enhance::NIGHT_RATIO_THRESH {
            let enhanced = state.night_enhancer.enhance(&frame);
            let mut merged = day_summary.detections.clone();
            merged.extend(run_detector(state, &enhanced));
            merged = alpenmesh_worker::inference::apply_nms(
                merged,
                alpen_detect::ALPEN_DETECT_NMS_IOU,
            );
            alpen_detect::summarize_detections(merged, sec, true, dark_ratio)
        } else {
            day_summary.clone()
        };
        total_ms += start.elapsed().as_secs_f64() * 1000.0;
        detection_count += 1;

        let events = tracker.update(&final_summary.detections, idx, sec);
        let active_count = tracker.active_track_count();
        let mean_dwell_this_frame = {
            let active = tracker.active_tracks();
            if active.is_empty() {
                0.0
            } else {
                active.values().map(|t| sec - t.entered_at_sec).sum::<f64>() / active.len() as f64
            }
        };
        let has_stalled = events.iter().any(|e| matches!(e, TrackEvent::Stalled { .. }));

        for ev in &events {
            match ev {
                TrackEvent::Entered { .. } => total_entered += 1,
                TrackEvent::Exited { dwell_sec, .. } => exited_dwell_times.push(*dwell_sec),
                TrackEvent::Stalled { .. } => total_stalled += 1,
            }
        }

        metrics.frames_processed += 1;
        metrics.duration_sec = sec;
        metrics.max_day_confidence = metrics
            .max_day_confidence
            .max(day_summary.max_accident_confidence);
        metrics.max_night_confidence = metrics
            .max_night_confidence
            .max(final_summary.max_accident_confidence);
        metrics.was_night_by_histogram |=
            dark_ratio >= alpenmesh_worker::night_enhance::NIGHT_RATIO_THRESH;
        metrics.used_enhancement |= final_summary.used_night_enhancement;

        if !day_summary.accident_reports.is_empty() {
            metrics.nirikshan_day_fires += 1;
            if metrics.nirikshan_day_first_fire_sec.is_none() {
                metrics.nirikshan_day_first_fire_sec = Some(sec);
            }
        }
        if !final_summary.accident_reports.is_empty() {
            metrics.nirikshan_night_enhanced_fires += 1;
            if metrics.nirikshan_night_enhanced_first_fire_sec.is_none() {
                metrics.nirikshan_night_enhanced_first_fire_sec = Some(sec);
            }

            if let Some(frame_img) = image::RgbImage::from_raw(640, 640, frame.clone()) {
                let context = serde_json::json!({
                    "video": video_path.to_string_lossy(),
                    "frame_idx": idx,
                    "sec": sec,
                    "dark_ratio": dark_ratio,
                    "used_night_enhancement": final_summary.used_night_enhancement,
                });
                for report in &final_summary.accident_reports {
                    accident_debug::save_accident_snapshot(&stem, &frame_img, report, &context);
                }
                if idx % fps.max(1) as u64 == 0 {
                    save_frame_preview(&frame_dir, idx, &frame_img, &final_summary)?;
                }
            }
        } else if idx % fps.max(1) as u64 == 0 {
            if let Some(frame_img) = image::RgbImage::from_raw(640, 640, frame.clone()) {
                save_frame_preview(&frame_dir, idx, &frame_img, &final_summary)?;
            }
        }

        {
            let mut tl_row = vec![
                label.to_string(),
                stem.clone(),
                idx.to_string(),
                format!("{sec:.3}"),
                if final_summary.used_night_enhancement {
                    "night_enhanced".to_string()
                } else {
                    "day".to_string()
                },
                format!("{dark_ratio:.3}"),
                final_summary.accident_reports.len().to_string(),
                format!("{:.3}", final_summary.max_accident_confidence),
                final_summary.used_night_enhancement.to_string(),
            ];
            if emit_metrics {
                tl_row.extend_from_slice(&[
                    String::new(),
                    format!("{:.3}", mean_dwell_this_frame),
                    if has_stalled { "1" } else { "0" }.to_string(),
                    active_count.to_string(),
                ]);
            }
            timeline.write_record(&tl_row)?;
        }
    }

    let _ = child.kill();
    let _ = child.wait();
    metrics.mean_detect_ms = total_ms / detection_count.max(1) as f64;
    metrics.total_tracked_vehicles = total_entered;
    metrics.stalled_events = total_stalled;
    metrics.mean_dwell_sec = if exited_dwell_times.is_empty() {
        0.0
    } else {
        exited_dwell_times.iter().map(|d| *d as f64).sum::<f64>() / exited_dwell_times.len() as f64
    };
    Ok(metrics)
}

fn run_detector(state: &WorkerState, rgb_640: &[u8]) -> Vec<[f32; 6]> {
    let model_arc = loop {
        if let Some(s) = state.pool.pop() {
            break s;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    };
    let detections = {
        let mut model = model_arc.lock().unwrap();
        alpen_detect::run_session_augmented(&mut model, rgb_640).unwrap_or_default()
    };
    let _ = state.pool.push(model_arc);
    detections
}

fn spawn_ffmpeg(video_path: &Path, fps: u32) -> Result<Child> {
    Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            video_path.to_string_lossy().as_ref(),
            "-vf",
            "scale=640:640:force_original_aspect_ratio=decrease,pad=640:640:(ow-iw)/2:(oh-ih)/2:color=black",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-r",
            &fps.to_string(),
            "-an",
            "-",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .context("spawn ffmpeg")
}

fn list_videos(root: &Path, label: &'static str) -> Result<Vec<(PathBuf, &'static str)>> {
    let mut out = Vec::new();
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        let ext = path
            .extension()
            .and_then(OsStr::to_str)
            .unwrap_or_default()
            .to_ascii_lowercase();
        if ["mov", "mp4", "avi", "mkv"].contains(&ext.as_str()) {
            out.push((path, label));
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

fn save_frame_preview(
    frame_dir: &Path,
    frame_idx: u64,
    frame_img: &image::RgbImage,
    summary: &alpen_detect::DetectionSummary,
) -> Result<()> {
    let mut vis = frame_img.clone();
    alpen_detect::draw_detection_boxes(
        &mut vis,
        &summary.vehicle_detections,
        image::Rgb([0, 255, 0]),
    );
    alpen_detect::draw_detection_boxes(&mut vis, &summary.detections, image::Rgb([255, 64, 64]));
    vis.save(frame_dir.join(format!("{frame_idx:06}.jpg")))?;
    Ok(())
}

fn run_stamp() -> String {
    let ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    format!("alpendetect_{ms}")
}

fn opt_f64(v: Option<f64>) -> String {
    v.map(|v| format!("{v:.3}")).unwrap_or_default()
}

fn pct(a: f64, b: f64) -> f64 {
    if b <= 0.0 {
        0.0
    } else {
        (a * 100.0) / b
    }
}
