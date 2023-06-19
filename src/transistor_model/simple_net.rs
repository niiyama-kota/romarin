use anyhow::Result;
use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Kind, Tensor};

use crate::{loader, transistor_model::simple_net};

const NEURON_NUM: i64 = 500;

fn new(vs: &nn::Path) -> impl Module {
    nn::seq()
        .add(nn::linear(
            vs / "input layer",
            2,
            NEURON_NUM,
            Default::default(),
        ))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(
            vs / "layer1",
            NEURON_NUM,
            NEURON_NUM,
            Default::default(),
        ))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(
            vs / "output layer",
            NEURON_NUM,
            1,
            Default::default(),
        ))
}

pub fn run() -> Result<()> {
    let mut dataset = loader::read_csv("data/integral_train.csv".to_string()).unwrap();
    dataset.min_max_scaling();
    let x = Tensor::stack(
        &[
            Tensor::from_slice(dataset.VDS.as_slice()),
            Tensor::from_slice(dataset.VGS.as_slice()),
        ],
        1,
    )
    .to_kind(Kind::Float);
    let y = Tensor::from_slice(dataset.IDS.as_slice())
        .to_kind(Kind::Float);
    let vs = nn::VarStore::new(Device::Cpu);
    let net = simple_net::new(&vs.root());
    let mut opt = nn::RmsProp::default().build(&vs, 1e-3)?;
    let mut losses = Vec::<Tensor>::new();
    for epoch in 1..=100 {
        opt.zero_grad();
        let loss = (net.forward(&x) - &y)
            .pow_tensor_scalar(2)
            .mean(Kind::Float);
        loss.backward();
        loss.print();
        opt.step();
        losses.push(loss);
        println!("epoch {}", epoch);
    }

    println!(
        "initial loss: {:?} \n last loss: {:?}",
        losses.first(),
        losses.last()
    );

    Ok(())
}

#[test]
fn test_simple_net() {
    use crate::transistor_model::simple_net;
    let _ = simple_net::run();
}
