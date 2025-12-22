#![allow(unused_attributes)]

use std::process::Command;

pub struct ModelConfig {
    pub version: &'static str,
    pub size: &'static str,
    pub url: &'static str,
    pub filename: &'static str,
    pub needs_simplify: bool,
}

pub const MODELS: &[(&str, ModelConfig)] = &[
    // YOLOv8 系列
    (
        "v8_n",
        ModelConfig {
            version: "v8",
            size: "n",
            url: "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
            filename: "yolov8n",
            needs_simplify: false,
        },
    ),
    (
        "v8_s",
        ModelConfig {
            version: "v8",
            size: "s",
            url: "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt",
            filename: "yolov8s",
            needs_simplify: false,
        },
    ),
    (
        "v8_m",
        ModelConfig {
            version: "v8",
            size: "m",
            url: "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt",
            filename: "yolov8m",
            needs_simplify: false,
        },
    ),
    (
        "v8_l",
        ModelConfig {
            version: "v8",
            size: "l",
            url: "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt",
            filename: "yolov8l",
            needs_simplify: false,
        },
    ),
    (
        "v8_x",
        ModelConfig {
            version: "v8",
            size: "x",
            url: "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt",
            filename: "yolov8x",
            needs_simplify: false,
        },
    ),
    // YOLOv9 系列
    (
        "v9_t",
        ModelConfig {
            version: "v9",
            size: "t",
            url: "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov9t.pt",
            filename: "yolov9t",
            needs_simplify: true,
        },
    ),
    (
        "v9_s",
        ModelConfig {
            version: "v9",
            size: "s",
            url: "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov9s.pt",
            filename: "yolov9s",
            needs_simplify: true,
        },
    ),
    (
        "v9_m",
        ModelConfig {
            version: "v9",
            size: "m",
            url: "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov9m.pt",
            filename: "yolov9m",
            needs_simplify: true,
        },
    ),
    (
        "v9_c",
        ModelConfig {
            version: "v9",
            size: "c",
            url: "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov9c.pt",
            filename: "yolov9c",
            needs_simplify: true,
        },
    ),
    (
        "v9_e",
        ModelConfig {
            version: "v9",
            size: "e",
            url: "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov9e.pt",
            filename: "yolov9e",
            needs_simplify: true,
        },
    ),
    // YOLO11 系列
    (
        "v11_n",
        ModelConfig {
            version: "v11",
            size: "n",
            url: "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
            filename: "yolo11n",
            needs_simplify: true,
        },
    ),
    (
        "v11_s",
        ModelConfig {
            version: "v11",
            size: "s",
            url: "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt",
            filename: "yolo11s",
            needs_simplify: true,
        },
    ),
    (
        "v11_m",
        ModelConfig {
            version: "v11",
            size: "m",
            url: "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt",
            filename: "yolo11m",
            needs_simplify: true,
        },
    ),
    (
        "v11_l",
        ModelConfig {
            version: "v11",
            size: "l",
            url: "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt",
            filename: "yolo11l",
            needs_simplify: true,
        },
    ),
    (
        "v11_x",
        ModelConfig {
            version: "v11",
            size: "x",
            url: "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt",
            filename: "yolo11x",
            needs_simplify: true,
        },
    ),
];

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // 跳过第一个参数（程序名）
    let args = &args[1..];

    if args.is_empty() {
        show_help();
        return;
    }

    let model_name = &args[0];

    let config = match MODELS.iter().find(|&&(name, _)| name == model_name) {
        Some((_, config)) => config,
        None => {
            eprintln!("Model '{}' not found", model_name);
            return;
        }
    };

    println!("==========================================");
    println!("Downloading model: {}", model_name);
    println!("==========================================");
    println!("URL: {}", config.url);
    println!("Filename: {}", config.filename);
    println!("");

    // 创建目录
    std::fs::create_dir_all("pretrained").expect("Failed to create pretrained directory");

    let output_path = format!("pretrained/{}.pt", config.filename);
    println!("Downloading to: {}", output_path);
    println!("");

    // 检查下载工具
    let download_tool = if is_command_available("curl") {
        "curl"
    } else if is_command_available("wget") {
        "wget"
    } else {
        eprintln!("Error: Neither curl nor wget found. Please install one of them.");
        std::process::exit(1);
    };

    // 检查 Python 环境
    check_python_environment();

    // 下载文件
    let download_success = if download_tool == "curl" {
        Command::new("curl")
            .args(&["-L", config.url, "-o", &output_path, "--progress-bar"])
            .status()
            .expect("Failed to execute curl")
            .success()
    } else {
        // wget 使用更简单的输出
        Command::new("wget")
            .args(&[config.url, "-O", &output_path])
            .status()
            .expect("Failed to execute wget")
            .success()
    };

    if !download_success {
        eprintln!("Error: Failed to download model");
        std::process::exit(1);
    }

    println!("✓ Download completed");
    println!("");
    println!("Exporting to ONNX format...");

    // 导出为 ONNX
    let export_success = if config.needs_simplify {
        Command::new("python3")
            .args(&["-c", &format!(
                "from ultralytics import YOLO; model = YOLO('{}'); model.export(format='onnx', imgsz=640, simplify=True, opset=12); print('Export completed with simplify=True')",
                output_path
            )])
            .status()
            .expect("Failed to execute python3")
            .success()
    } else {
        Command::new("python3")
            .args(&["-c", &format!(
                "from ultralytics import YOLO; model = YOLO('{}'); model.export(format='onnx', opset=12); print('Export completed')",
                output_path
            )])
            .status()
            .expect("Failed to execute python3")
            .success()
    };

    if export_success {
        println!("");
        println!(
            "✓ Model '{}' successfully downloaded and exported!",
            model_name
        );
        println!("");
        println!("Files created:");
        println!("  - {}", output_path);
        println!("  - pretrained/{}.onnx", config.filename);
    } else {
        eprintln!("Error: Failed to export model");
        std::process::exit(1);
    }
}

fn show_help() {
    println!("YOLO Model Downloader");
    println!("=====================");
    println!("");
    println!("Available models:");
    println!("  YOLOv8: v8_n, v8_s, v8_m, v8_l, v8_x");
    println!("  YOLOv9: v9_t, v9_s, v9_m, v9_c, v9_e");
    println!("  YOLO11: v11_n, v11_s, v11_m, v11_l, v11_x");
    println!("");
    println!("Examples:");
    println!("  download_model v8_n    # Download YOLOv8 nano");
    println!("  download_model v9_m    # Download YOLOv9 medium");
    println!("  download_model v11_x   # Download YOLO11 extra large");
    println!("");
}

fn is_command_available(command: &str) -> bool {
    Command::new("which")
        .arg(command)
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn check_python_environment() {
    let check_result = Command::new("python3")
        .args(&["-c", "import ultralytics"])
        .output();

    match check_result {
        Ok(output) if !output.status.success() => {
            eprintln!("Error: 'ultralytics' Python package is not installed.");
            eprintln!("Install with: pip install ultralytics");

            println!("Do you want to continue anyway? [y/N]");

            use std::io::{self, Write};
            io::stdout().flush().unwrap();

            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();

            if !input.trim().to_lowercase().starts_with('y') {
                std::process::exit(1);
            }
        }
        Err(_) => {
            eprintln!("Warning: Could not check Python environment");
        }
        _ => {}
    }
}
