# Template Matching & Cross Correaltion on the CIFAR-10 dataset

### Genesis Bautista & Jessica McIlree

## Description
- This project aims to classify on 8 of the 10 presented image classes in the CIFAR-10 dataset, demonstrating how template matching via Normalized Cross-Correlation (NCC) can be parallelized using Rust's concurrency features.

  Each test image is compared against 8 class templates across four different classification methods:
    - sequential classification - Images are compared to each class template one at a time on a single thread, serving as the baseline for benchmarking.
    - Rayon -  Work is distributed across threads using Rayon's parallel iterators, which automatically split images into chunks based on available CPU cores.
    - std::thread + Arc - Templates are wrapped in an Arc so all threads share a single copy of the data. Threads are spawned manually, one per chunk, and run independently before their results are joined.
    - std::thread + Arc<RwLock<>> - Templates are wrapped in Arc<RwLock<>>, allowing multiple threads to hold and read locks on the same memory. This demonstrates true shared-state concurrency.
Normalized Cross-Correlation (NCC) measures the similarity between a test image and each class template by computing their dot product normalized by their magnitudes. The class whose template produces the highest NCC score is assigned as the predicted label for that image.

## Evaluation Metrics

The model is evaluated by precision and recal, it produces a confusion matrix displaying the correclty predicted images over th eoverall false positives, and precision score is displayed as a percentage to the right.
Recall is calculated by finding the overall true positives (correctly predectied) over the total number of that classes images in the dataset.

Accuracy is measured by dividing the correclty matched images by the total test images and is displayed for each parallelization method.

Additional displayed metrics include:
  -Throughput is images classified per second for each method.
  -Speedup is parallel method time divided by serial time, showing how much faster parallelization is.
  -Efficiency is speedup divided by number of threads, showing how effectively each thread is being utilized.
These help show the computing power used by each method per specific device.

## Prerequisites
To run this projecty you must have rustc version at least 1.73.0, we reccomend using 1.93.1 or the latest version available.

**Dependencies:**
```toml
[package]
name = "image_classification"
version = "0.1.0"
edition = "2021"

[dependencies]
ndarray = "0.15"
rayon = "1.10"
image = "0.25"
```

We reccomend having at least 8 CPU cores to run this project, as there are 8 image classes and for parallelization you want at least 8 threads that won't have to wait on eachother
This project was run on devices with 16GB of RAM

## Setup Instructions

1. Download the CIFAR-10 Dataset at this link: https://www.cs.toronto.edu/~kriz/cifar.html
   - Be sure to download the **binary version (162 MB)**

2. Clone the repo and type `cargo init` in the terminal to initialize the project as a Cargo project.

3. Unzip the dataset and move the `cifar-10-batches-bin` folder to your directory, rename it to `data`.

4. Be sure you have the packages and dependencies listed above in your `Cargo.toml` file.

5. Run the following in the terminal to build and run the project:
```bash
cargo run --release
```

---

```
image-classification/
├── Cargo.toml
├── Cargo.lock
├── README.md
├── src/
│   ├── main.rs
│   ├── benchmark.rs
│   └── correlation.rs
├── data/
│   └── (batch .bin files)
├── templates/
│   └── ... (images generated after compiling)
└── target/
    └── ... (debug & release)
```

 # This project was made with Claude by Anthropic

