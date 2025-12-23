#![allow(unused_attributes)]

use std::io::{self, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};

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
    if std::path::Path::new(&output_path).exists() {
        println!("{} already exists.", output_path);
    } else {
        download_pt_file(config, &output_path)
    }

    // 检查 Python 环境, 导出为 ONNX
    let ret = export_yolo_to_onnx(&output_path, "", config.needs_simplify, true);
    match ret {
        Ok(_) => (),
        Err(e) => {
            eprintln!("{}", e);
        }
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

/// Get the actual python3 executable path (sys.executable).
fn get_python_executable() -> io::Result<String> {
    let out = Command::new("python3")
        .args(&["-c", "import sys; print(sys.executable)"])
        .output()?;
    if !out.status.success() {
        Err(io::Error::new(
            io::ErrorKind::Other,
            format!("python3 call failed: {}", String::from_utf8_lossy(&out.stderr)),
        ))
    } else {
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    }
}

/// Check whether `module` can be imported by the given python executable.
fn python_has_module(python_exe: &str, module: &str) -> io::Result<bool> {
    let code = format!(
        "import importlib.util; print('INSTALLED' if importlib.util.find_spec('{m}') else 'MISSING')",
        m = module
    );
    let out = Command::new(python_exe).args(&["-c", &code]).output()?;
    if !out.status.success() {
        Ok(false) // treat failures as missing; caller can show stderr if needed
    } else {
        let s = String::from_utf8_lossy(&out.stdout);
        Ok(s.lines().next().map(|l| l.trim() == "INSTALLED").unwrap_or(false))
    }
}

/// Run `python -m pip install ...` and stream output; return success.
fn pip_install(python_exe: &str, pkgs: &[&str]) -> io::Result<bool> {
    let mut args = vec!["-m", "pip", "install"];
    for p in pkgs {
        args.push(p);
    }
    println!("Running: {} {}", python_exe, args.join(" "));
    let child = Command::new(python_exe)
        .args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let output = child.wait_with_output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !stdout.is_empty() {
        println!("{}", stdout);
    }
    if !stderr.is_empty() {
        eprintln!("{}", stderr);
    }
    if output.status.success() {
        // Installed successfully in the provided python environment
        return Ok(true);
    }

    // If pip failed and indicates an externally-managed environment (Homebrew / PEP 668),
    // create a venv and try installing into the venv.
    let stderr_lower = stderr.to_ascii_lowercase();
    if stderr_lower.contains("externally-managed-environment") || stderr_lower.contains("pep 668") {
        eprintln!("Detected externally-managed environment; creating a virtual environment and retrying installation there.");

        // use default .venv if exists
        let mut venv_dir = PathBuf::from(format!(".venv"));
        if venv_dir.exists() {
            println!("{} already exists.", venv_dir.display());
        } else {
            // Construct a unique venv directory under current working dir (e.g., .venv_autocreate_<ts>)
            // let ts = SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
            // venv_dir = PathBuf::from(format!(".venv_autocreate_{}", ts));
            venv_dir = PathBuf::from(format!(".venv_autocreate"));
            // Ensure directory does not already exist (very unlikely)
            // if venv_dir.exists() {
            //    // add pid suffix if collision
            //     let pid = std::process::id();
            //     venv_dir = PathBuf::from(format!(".venv_autocreate_{}_{}", ts, pid));
            // }
        }

        // Create the venv: python_exe -m venv <venv_dir>
        let venv_str = venv_dir.to_string_lossy().to_string();
        println!("Creating virtual environment at: {}", venv_str);
        let status = Command::new(python_exe)
            .args(&["-m", "venv", &venv_str])
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status()?;

        if !status.success() {
            eprintln!("Failed to create virtual environment at {}", venv_str);
            return Ok(false);
        }

        // Determine path to the venv's python executable
        #[cfg(windows)]
        let venv_python = venv_dir.join("Scripts").join("python.exe");
        #[cfg(not(windows))]
        let venv_python = venv_dir.join("bin").join("python");

        let venv_python_str = venv_python.to_string_lossy().to_string();

        // Upgrade pip/setuptools/wheel in the venv first
        println!("Upgrading pip/setuptools/wheel in venv: {}", venv_python_str);
        let upgrade_status = Command::new(&venv_python_str)
            .args(&["-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?
            .wait_with_output()?;
        let up_stdout = String::from_utf8_lossy(&upgrade_status.stdout);
        let up_stderr = String::from_utf8_lossy(&upgrade_status.stderr);
        if !up_stdout.is_empty() {
            println!("{}", up_stdout);
        }
        if !up_stderr.is_empty() {
            eprintln!("{}", up_stderr);
        }
        if !upgrade_status.status.success() {
            eprintln!("Failed to upgrade pip in venv (continuing to try installing packages, but this may fail).");
        }

        // Install requested packages into the venv
        let mut venv_args = vec!["-m", "pip", "install"];
        for p in pkgs {
            venv_args.push(p);
        }
        println!("Installing packages into venv: {} {}", venv_python_str, venv_args.join(" "));
        let venv_child = Command::new(&venv_python_str)
            .args(&venv_args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        let venv_output = venv_child.wait_with_output()?;
        let venv_out = String::from_utf8_lossy(&venv_output.stdout);
        let venv_err = String::from_utf8_lossy(&venv_output.stderr);
        if !venv_out.is_empty() {
            println!("{}", venv_out);
        }
        if !venv_err.is_empty() {
            eprintln!("{}", venv_err);
        }

        if venv_output.status.success() {
            println!("Packages installed successfully into virtual environment.");
            println!("To use this environment for subsequent python calls, run:");
            #[cfg(windows)]
            {
                println!("  {}\\Scripts\\activate (or use {})", venv_str, venv_python_str);
            }
            #[cfg(not(windows))]
            {
                println!("  source {}/bin/activate", venv_str);
                println!("or call the venv python directly: {}", venv_python_str);
            }
            return Ok(true);
        } else {
            eprintln!("Failed to install packages into the virtual environment.");
            // Optionally remove the venv directory on failure to avoid clutter:
            let _ = std::fs::remove_dir_all(&venv_dir);
            return Ok(false);
        }
    }

    Ok(false)
}

/// Run the ultralytics export (via python -c "...") and return (success, stdout, stderr).
fn run_ultralytics_export(
    python_exe: &str,
    pt_path: &str,
    opset: u32,
    needs_simplify: bool,
) -> io::Result<(bool, String, String)> {
    let export_snippet = if needs_simplify {
        format!(
            "from ultralytics import YOLO; m=YOLO('{}'); m.export(format='onnx', imgsz=640, opset={}, dynamic=True, simplify=True, verbose=True); print('EXPORT_OK')",
            pt_path, opset
        )
    } else {
        format!(
            "from ultralytics import YOLO; m=YOLO('{}'); m.export(format='onnx', opset={}); print('EXPORT_OK')",
            pt_path, opset
        )
    };

    let out = Command::new(python_exe)
        .args(&["-c", &export_snippet])
        .output()?;

    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    Ok((out.status.success(), stdout, stderr))
}

/// Interactive yes/no prompt. Default false on empty input.
fn prompt_yes_no(prompt: &str) -> bool {
    print!("{} [y/N]: ", prompt);
    io::stdout().flush().ok();
    let mut input = String::new();
    if io::stdin().read_line(&mut input).is_ok() {
        let t = input.trim().to_lowercase();
        return t.starts_with('y');
    }
    false
}

/// High level wrapper to export a YOLO .pt to ONNX using the same python3 interpreter.
/// - `pt_path`: path to the .pt file (used when constructing YOLO(...))
/// - `onnx_expected_name`: expected created filename (for user messages)
/// - `needs_simplify`: whether to call export(..., simplify=True)
/// - `try_auto_install`: if true, offers to install missing packages into detected python
pub fn export_yolo_to_onnx(
    pt_path: &str,
    onnx_expected_name: &str,
    needs_simplify: bool,
    try_auto_install: bool,
) -> Result<(), String> {
    // 1) find python executable
    let python_exe = get_python_executable().map_err(|e| format!("Failed to call python3: {}", e))?;
    println!("Using python executable: {}", python_exe);

    // 2) check required modules
    let mut missing = Vec::new();
    for module in ["ultralytics", "torch", "onnx"].iter().cloned() {
        match python_has_module(&python_exe, module) {
            Ok(true) => println!("{}: found", module),
            Ok(false) => {
                eprintln!("{}: NOT found", module);
                missing.push(module);
            }
            Err(e) => {
                eprintln!("{} check failed: {}", module, e);
                missing.push(module);
            }
        }
    }

    // 3) if missing, optionally try to install
    if !missing.is_empty() {
        eprintln!("Missing Python packages: {:?}", missing);
        eprintln!("Recommended install command (uses the same python):");
        eprintln!("  {} -m pip install {}", python_exe, missing.join(" "));
        if try_auto_install && prompt_yes_no("Attempt automatic installation now?") {
            // try install all at once; note: torch on macOS/arm may need special handling
            if missing.contains(&"torch") {
                println!("Note: installing 'torch' via pip may fail on macOS (Apple Silicon). If pip install fails, follow https://pytorch.org/get-started/locally/ or use conda/miniforge.");
            }
            let pkgs: Vec<&str> = missing.iter().map(|s| *s).collect();
            let ok = pip_install(&python_exe, &pkgs).map_err(|e| format!("pip spawn failed: {}", e))?;
            if !ok {
                return Err("pip install reported failure. Please install the packages manually and retry.".into());
            }
        } else {
            return Err("Missing required Python packages; aborting export.".into());
        }
    }

    // 4) run the export command and capture stdout/stderr
    println!("Exporting to ONNX format...");
    let (success, stdout, stderr) =
        run_ultralytics_export(&python_exe, pt_path, 12, needs_simplify).map_err(|e| format!("Failed to run export command: {}", e))?;

    if !stdout.is_empty() {
        println!("python stdout:\n{}", stdout);
    }
    if !stderr.is_empty() {
        eprintln!("python stderr:\n{}", stderr);
    }

    if success && stdout.contains("EXPORT_OK") {
        println!();
        println!("✓ Model '{}' successfully exported!", pt_path);
        println!();
        println!("Files created (expected):");
        println!("  - {}", onnx_expected_name);
        Ok(())
    } else {
        Err(format!(
            "Export failed. See python stderr for details. (success={}, stdout_contains_EXPORT_OK={})",
            success,
            stdout.contains("EXPORT_OK")
        ))
    }
}

fn download_pt_file(config: &ModelConfig, output_path: &str) {
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
}
