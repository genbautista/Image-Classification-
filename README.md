### Template Matching & Cross Correaltion on the CIFAR-10 dataset

## Genesis Bautista & Jessica McIlree

# Description
- This project aims to classify on 8 of the 10 presented image classes in the CIFAR-10 dataset, it is aimed to explain how this process can be parallelized

  The project classifies the images by comparing to the 8 class templates made, under 4 different methods
    - sequential classification - every image is compared one at a time to 1 class template, using only 1 thread
    - Rayon - the process is iterated by image chunks based on CPU core count, multithreaded
    - std::thread + Arc - shares template ownership accross threads, pointing to the same data, spawsn threads based on core count amnd rund independantely
    - std::thread + Arc<RwLock<>> - shares reading data accorss threads, ownership is determined by which thread is writing
 
Cross correlation is used to compute the NCC dot product of each image to be classified and the template image, the higher the correlation, the more likely that image is a part of that class

# Evaluation Metrics

The model is evaluated by precision and recal, it produces a confusion matrix displaying the correclty predicted images over th eoverall false positives, and precision score is displayed as a percentage to the right
recall is calculated by finding the overall true positives (correctly predectied) over the total number of that classes images in the dataset

Accuracy is measured by dividing the correclty matched images by the total test images and is displayed for each parallelization method.

## Prerequisites
To run this projecty you must have rustc version at least 1.49.0, we reccomend using 1.93.1
dependancies:
ndarray = "0.15"
rayon = "1.10"
image = "0.25"

packages:
name = "image_classification"
version = "0.1.0"
edition = "2021"

We reccomend having at least 8 CPU cores to run this project, as there are 8 image classes and for parallelization you want at least 8 threads that won't have to wait on eachother
This project was run on devices with 16GB of RAM

### Setup Instructions:

First download the CIFAR-10 Dataset at this link: <https://www.cs.toronto.edu/~kriz/cifar.html>
 - be sure to download to binary version (162 MB)

-next clone the repo and type cargo  init in the terminal to initialize the project as a cargo project
-unzip the dataset and move the cifar-10-batches-bin folder to your directory, rename in data
-be sure you have the packages and dependancies listed as above in the Cargo.TOML file
-run cargo --release in the terminal to run the project

Your project directory should look like
image-classification-/
├── Cargo.toml
├── Cargo.lock
├── README.md
├── src/
│ ├── main.rs
│ ├── benchmark.rs
│ ├── correlation.rs (or similar)
├── data/
│ └── (batch.bin files)
├── templates/
│ └── ... (images generated after compiling)
├──target/
 └── ... (debug & release)

 # This project was made with Claude by Anthropic

