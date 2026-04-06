use ndarray::Array2;
use std::fs::File;
use std::io::{BufReader, Read};

const IMAGE_SIZE: usize = 3072; // 32x32x3
const BATCH_SIZE: usize = 10_000;

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

fn main() {
    let batch_paths = [
        "cifar-10-batches-bin/data_batch_1.bin",
        "cifar-10-batches-bin/data_batch_2.bin",
        "cifar-10-batches-bin/data_batch_3.bin",
        "cifar-10-batches-bin/data_batch_4.bin",
        "cifar-10-batches-bin/data_batch_5.bin",
    ];

    let mut all_labels: Vec<u8> = Vec::new();

    for path in &batch_paths {
        let (_, labels) = load_batch(path);
        all_labels.extend(labels);
    }

    let (_, test_labels) = load_batch("cifar-10-batches-bin/test_batch.bin");

    println!("Training images: {}", all_labels.len());
    println!("Test images: {}", test_labels.len());
}