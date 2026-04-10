use std::sync::{Arc, RwLock};
use std::thread;
use std::time::{Duration, Instant};

use ndarray::{Array1, Array2, Axis};

use crate::correlation::predict;

pub struct BenchResult {
    pub label: String,
    pub duration: Duration,
    pub accuracy: f32,
    pub predictions: Vec<usize>,
    pub num_threads: usize, // 1 for serial/rayon-managed, N for manual thread methods
}

/// Compute classification accuracy.
pub fn accuracy(predictions: &[usize], labels: &[u8], classes: &[u8]) -> f32 {
    let correct = predictions
        .iter()
        .zip(labels.iter())
        .filter(|(&pred, &true_label)| classes[pred] == true_label)
        .count();
    correct as f32 / labels.len() as f32
}

/// Parallel classification using std::thread with Arc (shared ownership).
pub fn classify_threaded(
    test_images: &Array2<f32>,
    templates: &Array2<f32>,
    num_threads: usize,
) -> Vec<usize> {
    let templates = Arc::new(templates.clone());
    let chunk_size = (test_images.nrows() + num_threads - 1) / num_threads;

    let rows: Vec<Array1<f32>> = test_images
        .axis_iter(Axis(0))
        .map(|r| r.to_owned())
        .collect();

    let handles: Vec<_> = rows
        .chunks(chunk_size)
        .map(|chunk| {
            let chunk = chunk.to_vec();
            let templates = Arc::clone(&templates);
            thread::spawn(move || {
                chunk
                    .iter()
                    .map(|row| predict(row, &templates))
                    .collect::<Vec<usize>>()
            })
        })
        .collect();

    handles
        .into_iter()
        .flat_map(|h| h.join().expect("Thread panicked"))
        .collect()
}

/// Parallel classification using std::thread with Arc<RwLock<>> shared state.
pub fn classify_threaded_rwlock(
    test_images: &Array2<f32>,
    templates: &Array2<f32>,
    num_threads: usize,
) -> Vec<usize> {
    let templates: Arc<RwLock<Array2<f32>>> = Arc::new(RwLock::new(templates.clone()));
    let chunk_size = (test_images.nrows() + num_threads - 1) / num_threads;

    let rows: Vec<Array1<f32>> = test_images
        .axis_iter(Axis(0))
        .map(|r| r.to_owned())
        .collect();

    let handles: Vec<_> = rows
        .chunks(chunk_size)
        .map(|chunk| {
            let chunk = chunk.to_vec();
            let templates = Arc::clone(&templates);
            thread::spawn(move || {
                let templates = templates.read().expect("RwLock poisoned");
                chunk
                    .iter()
                    .map(|row| predict(row, &templates))
                    .collect::<Vec<usize>>()
            })
        })
        .collect();

    handles
        .into_iter()
        .flat_map(|h| h.join().expect("Thread panicked"))
        .collect()
}

/// Run a classification function, time it, compute accuracy, return BenchResult.
pub fn run<F>(
    label: &str,
    num_threads: usize,
    f: F,
    true_labels: &[u8],
    classes: &[u8],
) -> BenchResult
where
    F: FnOnce() -> Vec<usize>,
{
    let start = Instant::now();
    let predictions = f();
    let duration = start.elapsed();
    let acc = accuracy(&predictions, true_labels, classes);
    BenchResult {
        label: label.to_string(),
        duration,
        accuracy: acc,
        predictions,
        num_threads,
    }
}

/// Print benchmark results table with accuracy, throughput, speedup, and efficiency.
pub fn print_results(results: &[BenchResult], num_images: usize) {
    let serial_secs = results[0].duration.as_secs_f64();

    println!("\n{:-<90}", "");
    println!(
        "{:<35} {:>10} {:>10} {:>14} {:>10} {:>10}",
        "Method", "Time(ms)", "Accuracy", "Throughput", "Speedup", "Efficiency"
    );
    println!(
        "{:<35} {:>10} {:>10} {:>14} {:>10} {:>10}",
        "", "", "", "(img/sec)", "", "(per thread)"
    );
    println!("{:-<90}", "");

    for r in results {
        let secs = r.duration.as_secs_f64();
        let throughput = num_images as f64 / secs;
        let speedup = serial_secs / secs;
        let efficiency = speedup / r.num_threads as f64;

        println!(
            "{:<35} {:>10.2} {:>9.2}% {:>14.0} {:>10.2} {:>10.2}",
            r.label,
            r.duration.as_millis(),
            r.accuracy * 100.0,
            throughput,
            speedup,
            efficiency,
        );
    }

    println!("{:-<90}", "");
}

/// Print a confusion matrix for a given set of predictions.
/// Rows = true class, Columns = predicted class.
pub fn print_confusion_matrix(
    predictions: &[usize],
    true_labels: &[u8],
    classes: &[u8],
    class_names: &[&str],
) {
    let n = classes.len();
    let mut matrix = vec![vec![0usize; n]; n];

    for (&pred, &true_label) in predictions.iter().zip(true_labels.iter()) {
        if let Some(true_idx) = classes.iter().position(|&c| c == true_label) {
            matrix[true_idx][pred] += 1;
        }
    }

    println!("\nConfusion Matrix (rows=true, cols=predicted):");
    println!();

    print!("{:<12}", "");
    for name in class_names {
        print!("{:>10}", name);
    }
    println!();
    println!("{:-<62}", "");

    for (i, row) in matrix.iter().enumerate() {
        print!("{:<12}", class_names[i]);
        for &val in row {
            print!("{:>10}", val);
        }
        let total: usize = row.iter().sum();
        let correct = row[i];
        let pct = if total > 0 {
            correct as f32 / total as f32 * 100.0
        } else {
            0.0
        };
        println!("  | {:.1}%", pct);
    }

    println!();
    println!("Per-class precision (predicted as X, how often correct):");
    for j in 0..n {
        let col_total: usize = matrix.iter().map(|row| row[j]).sum();
        let correct = matrix[j][j];
        let precision = if col_total > 0 {
            correct as f32 / col_total as f32 * 100.0
        } else {
            0.0
        };
        println!("  {:<12}: {:.1}%", class_names[j], precision);
    }
}