use ndarray::Array2;
use std::fs::File;
use std::io::{BufReader, Read};

const IMAGE_SIZE: usize = 3072;
const BATCH_SIZE: usize = 10_000;

const CLASSES: [u8; 5] = [0, 3, 4, 6, 8]; // airplane, cat, deer, frog, ship
const CLASS_NAMES: [&str; 5] = ["airplane", "cat", "deer", "frog", "ship"];

fn load_batch(path: &str) -> (Array2<f32>, Vec<u8>) {
    let file = File::open(path).expect("Could not open file");
    let mut reader = BufReader::new(file);
    let mut buf = vec![0u8; (IMAGE_SIZE + 1) * BATCH_SIZE];
    reader.read_exact(&mut buf).expect("Failed to read batch");

    let mut labels = Vec::with_capacity(BATCH_SIZE);
    let mut images = vec![0f32; IMAGE_SIZE * BATCH_SIZE];

    for i in 0..BATCH_SIZE {
        let offset = i * (IMAGE_SIZE + 1);
        labels.push(buf[offset]);
        for j in 0..IMAGE_SIZE {
            images[i * IMAGE_SIZE + j] = buf[offset + 1 + j] as f32 / 255.0;
        }
    }

    (
        Array2::from_shape_vec((BATCH_SIZE, IMAGE_SIZE), images).unwrap(),
        labels,
    )
}

fn filter_classes(images: Array2<f32>, labels: Vec<u8>) -> (Array2<f32>, Vec<u8>) {
    let indices: Vec<usize> = labels
        .iter()
        .enumerate()
        .filter(|(_, &l)| CLASSES.contains(&l))
        .map(|(i, _)| i)
        .collect();

    let filtered_images: Vec<f32> = indices
        .iter()
        .flat_map(|&i| images.row(i).to_vec())
        .collect();

    let filtered_labels: Vec<u8> = indices.iter().map(|&i| labels[i]).collect();

    let n = indices.len();
    (
        Array2::from_shape_vec((n, IMAGE_SIZE), filtered_images).unwrap(),
        filtered_labels,
    )
}

fn main() {
    let batch_paths = [
        "data/data_batch_1.bin",
        "data/data_batch_2.bin",
        "data/data_batch_3.bin",
        "data/data_batch_4.bin",
        "data/data_batch_5.bin",
    ];

    let mut all_images: Vec<f32> = Vec::new();
    let mut all_labels: Vec<u8> = Vec::new();

    for path in &batch_paths {
        let (images, labels) = load_batch(path);
        let (filtered_images, filtered_labels) = filter_classes(images, labels);
        all_images.extend(filtered_images.into_raw_vec());
        all_labels.extend(filtered_labels);
    }

    let n_train = all_labels.len();
    let train_images = Array2::from_shape_vec((n_train, IMAGE_SIZE), all_images).unwrap();

    let (test_images, test_labels_raw) = load_batch("data/test_batch.bin");
    let (test_images, _test_labels) = filter_classes(test_images, test_labels_raw);

    println!("Classes: {:?}", CLASS_NAMES);
    println!("Training images: {}", train_images.nrows());
    println!("Test images: {}", test_images.nrows());
}