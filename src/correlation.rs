use ndarray::{Array1, Array2, Axis};
use rayon::prelude::*;

/// Zero-mean normalize a single image vector.
/// Subtracts the mean pixel value so NCC compares contrast patterns
/// rather than raw brightness.
pub fn normalize(image: &Array1<f32>) -> Array1<f32> {
    let mean = image.mean().unwrap_or(0.0);
    image - mean
}

/// Normalized Cross-Correlation between two 1D vectors.
/// Returns a score in [-1.0, 1.0]; higher = more similar.
pub fn ncc(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let dot = a.dot(b);
    let norm_a = a.dot(a).sqrt();
    let norm_b = b.dot(b).sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Build `templates_per_class` zero-mean normalized mean templates per class.
/// Each class's training images are split into equal chunks; each chunk is averaged
/// into one template. Returns Array2 of shape (num_classes * templates_per_class, IMAGE_SIZE).
pub fn build_templates(
    images: &Array2<f32>,
    labels: &[u8],
    classes: &[u8],
    templates_per_class: usize,
) -> Array2<f32> {
    println!("Building {} templates per class from training data...", templates_per_class);

    let image_size = images.ncols();
    let total_templates = classes.len() * templates_per_class;
    let mut templates = Array2::<f32>::zeros((total_templates, image_size));

    for (class_idx, &class) in classes.iter().enumerate() {
        let class_indices: Vec<usize> = labels
            .iter()
            .enumerate()
            .filter(|(_, &l)| l == class)
            .map(|(i, _)| i)
            .collect();

        let n = class_indices.len();
        let chunk_size = (n + templates_per_class - 1) / templates_per_class;

        for chunk_idx in 0..templates_per_class {
            let start = chunk_idx * chunk_size;
            let end = (start + chunk_size).min(n);
            let chunk = &class_indices[start..end];
            let chunk_n = chunk.len() as f32;

            let template_row = class_idx * templates_per_class + chunk_idx;

            // Average the chunk
            for &i in chunk {
                let row = images.row(i);
                for (j, &val) in row.iter().enumerate() {
                    templates[[template_row, j]] += val / chunk_n;
                }
            }

            // Zero-mean normalize
            let mean = templates.row(template_row).mean().unwrap_or(0.0);
            for j in 0..image_size {
                templates[[template_row, j]] -= mean;
            }
        }
    }

    println!(
        "Templates built: {} classes x {} = {} total templates.",
        classes.len(),
        templates_per_class,
        total_templates
    );
    templates
}

/// Predict the template index for a single image using zero-mean NCC.
/// Returns the template index (0..total_templates); divide by templates_per_class
/// to get the class index.
pub fn predict(image: &Array1<f32>, templates: &Array2<f32>) -> usize {
    let normalized = normalize(image);
    templates
        .axis_iter(Axis(0))
        .enumerate()
        .map(|(i, template)| (i, ncc(&normalized, &template.to_owned())))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap()
        .0
}

/// Serial classification — single thread baseline.
pub fn classify_serial(test_images: &Array2<f32>, templates: &Array2<f32>) -> Vec<usize> {
    test_images
        .axis_iter(Axis(0))
        .map(|row| predict(&row.to_owned(), templates))
        .collect()
}

/// Parallel classification using Rayon parallel iterators.
pub fn classify_rayon(test_images: &Array2<f32>, templates: &Array2<f32>) -> Vec<usize> {
    let rows: Vec<Array1<f32>> = test_images
        .axis_iter(Axis(0))
        .map(|r| r.to_owned())
        .collect();

    rows.par_iter()
        .map(|row| predict(row, templates))
        .collect()
}

/// Save template images as upscaled PNGs (requires `image` crate).
pub fn save_templates(templates: &Array2<f32>, class_names: &[&str], templates_per_class: usize) {
    use image::{imageops, ImageBuffer, Rgb};

    std::fs::create_dir_all("templates").expect("Could not create templates dir");

    for (i, template) in templates.axis_iter(Axis(0)).enumerate() {
        // Shift back to [0, 1] range for visualization (template is zero-mean)
        let min = template.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = template.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = (max - min).max(1e-6);

        let mut img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(32, 32);
        for y in 0..32u32 {
            for x in 0..32u32 {
                let idx = (y * 32 + x) as usize;
                let r = ((template[idx] - min) / range * 255.0).clamp(0.0, 255.0) as u8;
                let g = ((template[1024 + idx] - min) / range * 255.0).clamp(0.0, 255.0) as u8;
                let b = ((template[2048 + idx] - min) / range * 255.0).clamp(0.0, 255.0) as u8;
                img.put_pixel(x, y, Rgb([r, g, b]));
            }
        }

        // Upscale to 256x256 for visibility
        let big = imageops::resize(&img, 256, 256, imageops::FilterType::Nearest);
        let class_name = class_names[i / templates_per_class];
        let path = format!("templates/{}_{}.png", class_name, i % templates_per_class);
        big.save(&path).expect("Failed to save template image");
        println!("  Saved: {}", path);
    }
}