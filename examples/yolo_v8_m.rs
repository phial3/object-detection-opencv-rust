use std::time::Instant;

use od_opencv::{DnnBackend, DnnTarget, Model};
use anyhow::Result;

use opencv::{
    core::{Point, Rect, Scalar, Size, Vector, Mat},
    imgcodecs::imread,
    imgcodecs::imwrite,
    imgproc::{self, FILLED, FONT_HERSHEY_SIMPLEX, LINE_4},
    prelude::MatTraitConst,
};

fn main() {
    // Print OpenCV version
    let cv_version = opencv::core::get_version_string().unwrap();
    println!("OpenCV version: {}", cv_version);

    let classes_labels: Vec<&str> = vec!["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"];

    let net_width = 640;
    let net_height = 640;

    let mut model = Model::opencv("pretrained/yolov8m.onnx", (net_width, net_height), DnnBackend::Cuda, DnnTarget::Cuda).unwrap();
    let mut frame = imread("images/dog.jpg", 1).unwrap();
    let start = Instant::now();
    let (bboxes, class_ids, confidences) = model.forward(&frame, 0.25, 0.4).unwrap();
    println!("Inference time: {:?}", start.elapsed());

    // draw
    draw_detections(&mut frame, &bboxes, &class_ids, &classes_labels, &confidences).unwrap();

    imwrite("images/dog_yolov8_m.jpg", &frame, &Vector::new()).unwrap();
}

/// 把所有检测画到图片上（先画框再画标签）
pub fn draw_detections(
    frame: &mut Mat,
    bboxes: &[Rect],
    class_ids: &[usize],
    class_labels: &[&str],
    confidences: &[f32],
) -> Result<()> {
    // 基于图像大小自动选择字体尺度（可根据需求调整）
    let max_dim = (frame.cols().max(frame.rows()) as f64).max(1.0);
    let font_scale = (max_dim / 1000.0).max(0.4);
    let thickness = 1.max((font_scale * 1.5) as i32);

    for (i, bbox) in bboxes.iter().enumerate() {
        let class_id = class_ids[i];
        let label_text = if let Some(name) = class_labels.get(class_id) {
            format!("{} {:.5}", name, confidences.get(i).copied().unwrap_or(0.0))
        } else {
            format!("id:{} {:.5}", class_id, confidences.get(i).copied().unwrap_or(0.0))
        };

        println!("Class: {}", class_labels[class_id]);
        println!("\tBounding box: {:?}", bbox);
        println!("\tConfidences: {}", confidences[i]);

        let color = class_color_by_id(class_id);

        // 1) 画框（使用 class color）
        draw_bbox(frame, *bbox, color, thickness)?;

        // 2) 画标签（在 bbox 上方或内部）
        draw_label(frame, *bbox, &label_text, color, font_scale, thickness)?;
    }

    Ok(())
}

/// 画标签（类别名 + 置信度），带背景色。位置尝试放在 bbox 之上，不够则放在 bbox 内部顶部。
fn draw_label(
    frame: &mut Mat,
    bbox: Rect,
    label: &str,
    class_color: Scalar,
    font_scale: f64,
    thickness: i32,
) -> Result<()> {
    // 文本样式
    let font_face = FONT_HERSHEY_SIMPLEX;
    let mut baseline: i32 = 0;
    let text_size: Size = imgproc::get_text_size(label, font_face, font_scale, thickness, &mut baseline)?;

    let padding = (4.0 * font_scale).ceil() as i32;
    let text_w = text_size.width + 2 * padding;
    let text_h = text_size.height + 2 * padding;

    // 目标图像边界
    let img_w = frame.cols();
    let img_h = frame.rows();

    // 尝试放在 bbox 顶部外部
    let mut x = bbox.x;
    if x < 0 { x = 0; }
    let mut y = bbox.y - text_h; // top-left y of background
    if y < 0 {
        // 放不下则放在 bbox 内部顶部
        y = bbox.y + padding;
    }

    // 保证不超右边界
    if x + text_w > img_w {
        x = (img_w - text_w).max(0);
    }
    // 保证不超下边界
    if y + text_h > img_h {
        y = (img_h - text_h).max(0);
    }

    // 背景矩形与文本起始点（opencv 的 put_text 以 baseline 底线为基准）
    let bg_rect = Rect::new(x, y, text_w.max(1), text_h.max(1));
    imgproc::rectangle(frame, bg_rect, class_color, FILLED, LINE_4, 0)?;

    // 根据背景亮度选文字颜色（亮背景 -> 黑字，暗背景 -> 白字）
    let brightness = perceived_brightness_from_bgr(&class_color);
    let text_color = if brightness > 128.0 {
        Scalar::from((0.0, 0.0, 0.0)) // 黑
    } else {
        Scalar::from((255.0, 255.0, 255.0)) // 白
    };

    // 文本原点（底线位置）
    let text_org = Point::new(x + padding, y + padding + text_size.height - baseline);

    imgproc::put_text(frame, label, text_org, font_face, font_scale, text_color, thickness, LINE_4, false)?;
    Ok(())
}

/// 计算给定 BGR 颜色的感知亮度（0..255），用于选择文字黑/白
fn perceived_brightness_from_bgr(color: &Scalar) -> f64 {
    // color: (B, G, R)
    let b = color[0] as f64;
    let g = color[1] as f64;
    let r = color[2] as f64;
    // 使用常用加权公式（RGB 顺序）
    0.299 * r + 0.587 * g + 0.114 * b
}


/// 画单个检测框（只负责矩形框，不画文字）
fn draw_bbox(frame: &mut Mat, bbox: Rect, color: Scalar, thickness: i32) -> Result<()> {
    Ok(imgproc::rectangle(frame, bbox, color, thickness, LINE_4, 0)?)
}

/// 返回按类别 id 映射的 BGR 颜色
fn class_color_by_id(id: usize) -> Scalar {
    // BGR color palette (choose as you like)
    let palette = [
        Scalar::from((0.0, 255.0, 0.0)),   // green
        Scalar::from((0.0, 0.0, 255.0)),   // red
        Scalar::from((255.0, 0.0, 0.0)),   // blue
        Scalar::from((0.0, 255.0, 255.0)), // yellow
        Scalar::from((255.0, 0.0, 255.0)), // magenta
        Scalar::from((255.0, 255.0, 0.0)), // cyan
        Scalar::from((0.0, 128.0, 255.0)), // orange-ish
        Scalar::from((128.0, 0.0, 128.0)), // purple
        Scalar::from((0.0, 165.0, 255.0)), // teal/orange variant
        Scalar::from((42.0, 42.0, 165.0)), // brown-ish
    ];
    palette[id % palette.len()]
}
