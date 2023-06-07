use tch::Tensor;
use romarin::transistor_model;

fn main() {
    // let t = Tensor::from_slice(&[3, 1, 4, 1, 5]);
    // let t = t * 2;
    // t.print();

    let err = transistor_model::simple_net::run();
    println!("{:?}", err);
}