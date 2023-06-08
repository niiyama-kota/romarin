use tch::Tensor;
use romarin::transistor_model::PiNN;

fn main() {
    // let t = Tensor::from_slice(&[3, 1, 4, 1, 5, 6]);
    // let t = t.view([2, 3, 1]);
    // t.tanh().print();

    let _ = PiNN::run();
}