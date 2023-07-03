use anyhow::Result;
use plotters::prelude::*;
use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use tch::nn::ModuleT;
use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

use crate::loader;

const NEURON_NUM: i64 = 500;

#[derive(Debug)]
struct SimpleNet {
    // layers: nn::SequentialT,
    input_layer: nn::Linear,
    hidden_layer: nn::Linear,
    output_layer: nn::Linear,
}
impl SimpleNet {
    fn new(vs: &nn::Path) -> Self {
        let _input = nn::linear(vs / "input layer", 2, NEURON_NUM, Default::default());
        let _hidden = nn::linear(
            vs / "hidden layer",
            NEURON_NUM,
            NEURON_NUM,
            Default::default(),
        );
        let _output = nn::linear(vs / "output layer", NEURON_NUM, 1, Default::default());
        Self {
            input_layer: _input,
            hidden_layer: _hidden,
            output_layer: _output,
        }
    }

    fn rmse(&self, xs: &tch::Tensor, ys: &tch::Tensor) -> tch::Tensor {
        (ys - self.forward_t(xs, false))
            .square()
            .mean(Kind::Float)
            .sqrt()
    }
}

impl ModuleT for SimpleNet {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let y = self.input_layer.forward_t(&xs, train);
        let y = self.hidden_layer.forward_t(&y, train).relu();
        let y = self.output_layer.forward_t(&y, train);

        return y;
    }
}

pub fn run() -> Result<()> {
    let mut dataset = loader::read_csv("data/SCT2080KE_ID-VDS-VGS_train.csv".to_string()).unwrap();
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
        .to_kind(Kind::Float)
        .reshape([-1, 1]);

    let mut test_dataset = loader::read_csv("data/SCT2080KE_ID-VDS-VGS.csv".to_string()).unwrap();
    test_dataset.min_max_scaling();
    let x_test = Tensor::stack(
        &[
            Tensor::from_slice(test_dataset.VDS.as_slice()),
            Tensor::from_slice(test_dataset.VGS.as_slice()),
        ],
        1,
    )
    .to_kind(Kind::Float);
    let y_test = Tensor::from_slice(test_dataset.IDS.as_slice())
        .to_kind(Kind::Float)
        .reshape([-1, 1]);

    let vs = nn::VarStore::new(Device::Cpu);
    let net = SimpleNet::new(&vs.root());
    let mut opt = nn::AdamW::default().build(&vs, 1e-5)?;
    let mut losses = Vec::<f64>::new();
    for epoch in 1..=5000 {
        opt.zero_grad();
        // let loss = (net.forward_t(&x, true) - &y)
        //     .pow_tensor_scalar(2)
        //     .mean(Kind::Float);
        let loss = net.forward_t(&x, true).mse_loss(&y, tch::Reduction::Mean);
        loss.print();
        opt.backward_step(&loss);
        losses.push(loss.double_value(&[]));
        println!("epoch {}", epoch);
    }

    println!(
        "initial loss: {:?} \n last loss: {:?}",
        losses.first(),
        losses.last()
    );

    let rmse = net.rmse(&x_test, &y_test);
    println!("test rmse: {:?}", rmse);
    let y_pred = net.forward_t(&x_test, false);

    let mut vgs_test: Vec<f32> = vec![0.0; x_test.transpose(0, 1).get(0).numel()];
    let mut vds_test: Vec<f32> = vec![0.0; x_test.transpose(0, 1).get(1).numel()];
    let mut ids_test: Vec<f32> = vec![0.0; y_test.numel()];
    let mut ids_pred: Vec<f32> = vec![0.0; y_pred.numel()];
    x_test
        .transpose(0, 1)
        .get(0)
        .copy_data(&mut vgs_test, x_test.transpose(0, 1).get(0).numel());
    x_test
        .transpose(0, 1)
        .get(1)
        .copy_data(&mut vds_test, x_test.transpose(0, 1).get(1).numel());
    y_test.copy_data(&mut ids_test, y_test.numel());
    y_pred.copy_data(&mut ids_pred, y_pred.numel());

    let data_output_path = Path::new("./data");
    create_dir_all(&data_output_path)?;
    let mut w = BufWriter::new(File::create(data_output_path.join("test_data.csv"))?);

    writeln!(w, "VGS,VDS,IDS,IDS_PRED")?;
    for (vgs, (vds, (ids_t, ids_p))) in vgs_test.into_iter().zip(
        vds_test
            .into_iter()
            .zip(ids_test.into_iter().zip(ids_pred.into_iter())),
    ) {
        writeln!(w, "{},{},{},{}", vgs, vds, ids_t, ids_p)?;
    }

    let root = BitMapBackend::new("plots/3d_scatter.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .caption("Test Samples Scatter", ("sans-serif", 40))
        .build_cartesian_3d(0.0..1.0, 0.0..1.0, 0.0..1.0)
        .unwrap();
    chart.configure_axes().draw().unwrap();
    // chart
    //     .draw_series(PointSeries::new(
    //         vds_test
    //             .iter()
    //             .zip(vgs_test.iter())
    //             .zip(ids_test.iter())
    //             .map(|((vds, vgs), ids)| (*vds, *vgs, *ids)),
    //         2,
    //         &RED,
    //     ))
    //     .unwrap();

    Ok(())
}

#[test]
fn test_simple_net() {
    use crate::transistor_model::SimpleNet;
    let _ = SimpleNet::run();
}

#[test]
fn test_convert_to_vec() {
    // Tensorを作成
    let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .reshape(&[2, 3])
        .to_kind(Kind::Float);
    let original_vec: Vec<f32> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].to_vec();
    let num_tensor = tensor.numel();
    let mut vec: Vec<f32> = vec![0.; num_tensor];
    tensor.copy_data(&mut vec, num_tensor);

    assert_eq!(original_vec, vec);
}
