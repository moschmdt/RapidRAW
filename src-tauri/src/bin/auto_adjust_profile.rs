// Standalone binary for profiling the auto-adjust pipeline.
// Processes all images in src-tauri/test-images/ once and exits.
//
// ─────────────────────────────────────────────────────────────
//  BUILD
// ─────────────────────────────────────────────────────────────
//
//    cargo build --profile profiling --bin auto_adjust_profile
//    → binary lands at target/profiling/auto_adjust_profile
//
// ─────────────────────────────────────────────────────────────
//  TOOL USAGE
// ─────────────────────────────────────────────────────────────
//
// 1. SAMPLY  (CPU — sampling profiler, opens Firefox Profiler UI)
//    cargo build --profile profiling --bin auto_adjust_profile
//    samply record ./target/profiling/auto_adjust_profile
//    → automatically opens https://profiler.firefox.com with the trace
//
// 2. FLAMEGRAPH  (CPU — which functions are hot, SVG output)
//    cargo flamegraph --profile profiling --bin auto_adjust_profile
//    → writes flamegraph.svg in the current directory
//
// 3. HYPERFINE  (wall-clock — how long the whole pipeline takes)
//    cargo build --profile profiling --bin auto_adjust_profile
//    hyperfine --warmup 3 './target/profiling/auto_adjust_profile'
//    → prints min/max/mean/stddev across runs
//
// ─────────────────────────────────────────────────────────────

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;

use rapidraw_lib::image_loader::load_base_image_from_bytes;
use rapidraw_lib::image_processing::{auto_results_to_json, perform_auto_analysis};

const TEST_IMAGES_DIR: &str = "test-images";

fn print_usage(prog: &str) {
    eprintln!("Usage: {} [--threads <N>]", prog);
    eprintln!("  --threads <N>   Number of rayon worker threads (default: logical CPU count)");
}

fn main() {
    // ── Parse CLI args ──────────────────────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    let mut thread_count: Option<usize> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--threads" => {
                i += 1;
                match args.get(i).and_then(|v| v.parse::<usize>().ok()) {
                    Some(n) if n > 0 => thread_count = Some(n),
                    _ => {
                        eprintln!("Error: --threads requires a positive integer");
                        print_usage(&args[0]);
                        std::process::exit(1);
                    }
                }
            }
            "--help" | "-h" => {
                print_usage(&args[0]);
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                print_usage(&args[0]);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    // ── Build rayon thread pool ─────────────────────────────────────────────
    let mut builder = rayon::ThreadPoolBuilder::new();
    if let Some(n) = thread_count {
        builder = builder.num_threads(n);
    }
    let pool = builder.build().expect("Failed to build rayon thread pool");
    let effective_threads = pool.current_num_threads();
    println!(
        "Using {} rayon thread(s){}",
        effective_threads,
        if thread_count.is_none() { " (default)" } else { "" }
    );

    // ── Discover images ─────────────────────────────────────────────────────
    let dir = Path::new(TEST_IMAGES_DIR);
    if !dir.exists() {
        eprintln!(
            "No '{}' directory found. Create it and add images before profiling.",
            TEST_IMAGES_DIR
        );
        std::process::exit(1);
    }

    let entries: Vec<_> = std::fs::read_dir(dir)
        .expect("Failed to read test-images dir")
        .flatten()
        .filter(|e| {
            let p = e.path();
            p.is_file() && p.file_name().map_or(false, |n| n != ".DS_Store")
        })
        .collect();

    if entries.is_empty() {
        eprintln!("No files in '{}'. Add JPEG, PNG, or RAW files.", TEST_IMAGES_DIR);
        std::process::exit(1);
    }

    let total = entries.len();
    let processed = Arc::new(AtomicUsize::new(0));

    // ── Process in parallel ─────────────────────────────────────────────────
    pool.install(|| {
        entries.par_iter().for_each(|entry| {
            let path = entry.path();
            let name = path.file_name().unwrap().to_string_lossy().into_owned();

            let bytes = match std::fs::read(&path) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("  skip {}: {}", name, e);
                    return;
                }
            };

            let image = match load_base_image_from_bytes(
                &bytes,
                &name,
                true, // fast demosaic: perform_auto_analysis downscales to 1024px anyway
                2.5_f32,
                "none".to_string(),
                None::<(Arc<AtomicUsize>, usize)>,
            ) {
                Ok(img) => img,
                Err(e) => {
                    eprintln!("  skip {}: {}", name, e);
                    return;
                }
            };

            let results = perform_auto_analysis(&image);
            let json = auto_results_to_json(&results);

            // Write <image-name>.auto.json next to the image for easy diffing.
            let out_path = path.with_extension("auto.json");
            match std::fs::write(&out_path, serde_json::to_string_pretty(&json).unwrap()) {
                Ok(_) => {},
                Err(e) => eprintln!("  warn: could not write {}: {}", out_path.display(), e),
            }

            let n = processed.fetch_add(1, Ordering::Relaxed) + 1;
            println!("[{}/{}] {} → {}", n, total, name, out_path.display());
        });
    });

    println!("Done. Processed {}/{} images.", processed.load(Ordering::Relaxed), total);
}
