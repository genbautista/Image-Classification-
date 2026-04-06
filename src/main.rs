use cifar_ten::*;

fn main() {
    let CifarResult(train_data, train_labels, test_data, test_labels) = Cifar10::default()
        .download_and_extract(true)
        .encode_one_hot(true)
        .build()
        .unwrap();

    println!("Dataset loaded! Training images: {}", train_data.len() / 3072);
}