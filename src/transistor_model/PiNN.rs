use crate::loader;
use anyhow::Result;
use plotters::prelude::*;
use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use tch::{
    nn::Module,
    nn::OptimizerConfig,
    nn::{self, LinearConfig},
    Device, Kind, Tensor,
};

const VD_NEURON_NUM: i64 = 2;
const VG_NEURON_NUM: i64 = 3;

#[derive(Debug)]
struct Vd_subnet {
    vd_sub_layer0: nn::Linear,
    vd_sub_layer1: nn::Linear,
}

impl Vd_subnet {
    fn new(vs: &nn::Path) -> Vd_subnet {
        let vd_sub_layer0 = nn::linear(
            vs / "vd_subnet_layer0",
            1,
            VD_NEURON_NUM,
            Default::default(),
        );
        let vd_sub_layer1 = nn::linear(
            vs / "vd_subnet_layer1",
            VD_NEURON_NUM,
            1,
            Default::default(),
        );

        Vd_subnet {
            vd_sub_layer0,
            vd_sub_layer1,
        }
    }
}

#[derive(Debug)]
struct Vg_subnet {
    vg_sub_layer0: nn::Linear,
    vg_sub_layer1: nn::Linear,
}

impl Vg_subnet {
    fn new(vs: &nn::Path) -> Vg_subnet {
        let vg_sub_layer0 = nn::linear(
            vs / "vg_subnet_layer0",
            1,
            VG_NEURON_NUM,
            LinearConfig {
                ws_init: nn::init::DEFAULT_KAIMING_UNIFORM,
                bs_init: None,
                bias: false,
            },
        );
        let vg_sub_layer1 = nn::linear(
            vs / "vg_subnet_layer1",
            VG_NEURON_NUM,
            1,
            LinearConfig {
                ws_init: nn::init::DEFAULT_KAIMING_UNIFORM,
                bs_init: None,
                bias: false,
            },
        );

        Vg_subnet {
            vg_sub_layer0,
            vg_sub_layer1,
        }
    }
}

#[derive(Debug)]
pub struct PiNN {
    vd_subnet: Vd_subnet,
    vg_subnet: Vg_subnet,
    vd2vg_layer0: nn::Linear,
    vd2vg_layer1: nn::Linear,
}

impl PiNN {
    fn new(vs: &nn::Path) -> PiNN {
        let vd_subnet = Vd_subnet::new(vs);
        let vg_subnet = Vg_subnet::new(vs);
        let vd2vg_layer0 = nn::linear(
            vs / "vd2vg_layer0",
            VD_NEURON_NUM,
            VG_NEURON_NUM,
            LinearConfig {
                ws_init: nn::init::DEFAULT_KAIMING_UNIFORM,
                bs_init: None,
                bias: false,
            },
        );
        let vd2vg_layer1 = nn::linear(vs / "vd2vg_layer1", 1, 1, Default::default());

        PiNN {
            vd_subnet,
            vg_subnet,
            vd2vg_layer0,
            vd2vg_layer1,
        }
    }

    fn rmse(&self, xs: &tch::Tensor, ys: &tch::Tensor) -> tch::Tensor {
        (ys - self.forward(xs)).square().mean(Kind::Float).sqrt()
    }
}

impl Module for PiNN {
    fn forward(&self, xs: &tch::Tensor) -> tch::Tensor {
        let vd = xs.select(1, 0).view([-1, 1]);
        let vg = xs.select(1, 1).view([-1, 1]);
        let t = vd.apply(&self.vd_subnet.vd_sub_layer0).tanh();
        let s = (t.apply(&self.vd2vg_layer0) + vg.apply(&self.vg_subnet.vg_sub_layer0)).sigmoid();
        let t = t.apply(&self.vd_subnet.vd_sub_layer1).tanh();
        let s = (t.apply(&self.vd2vg_layer1) + s.apply(&self.vg_subnet.vg_sub_layer1)).sigmoid();

        &t * &s
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
    let net = PiNN::new(&vs.root());
    let mut opt = nn::AdamW::default().build(&vs, 1e-3)?;
    let mut losses = Vec::<Tensor>::new();
    for epoch in 1..=50000 {
        let loss = (net.forward(&x) - &y).square().mean(Kind::Float);
        loss.print();
        opt.backward_step(&loss);
        losses.push(loss);
        println!("epoch {}", epoch);
    }

    println!(
        "initial loss: {:?} \n last loss: {:?}",
        losses.first(),
        losses.last()
    );

    let rmse = net.rmse(&x_test, &y_test);
    println!("test rmse: {:?}", rmse);

    let y_pred = net.forward(&x_test);

    let mut vds_test: Vec<f32> = vec![0.0; x_test.transpose(0, 1).get(0).numel()];
    let mut vgs_test: Vec<f32> = vec![0.0; x_test.transpose(0, 1).get(1).numel()];
    let mut ids_test: Vec<f32> = vec![0.0; y_test.numel()];
    let mut ids_pred: Vec<f32> = vec![0.0; y_pred.numel()];
    x_test
        .transpose(0, 1)
        .get(0)
        .copy_data(&mut vds_test, x_test.transpose(0, 1).get(0).numel());
    x_test
        .transpose(0, 1)
        .get(1)
        .copy_data(&mut vgs_test, x_test.transpose(0, 1).get(1).numel());
    y_test.copy_data(&mut ids_test, y_test.numel());
    y_pred.copy_data(&mut ids_pred, y_pred.numel());

    let data_output_path = Path::new("./data");
    create_dir_all(&data_output_path)?;
    let mut w = BufWriter::new(File::create(data_output_path.join("test_data.csv"))?);

    writeln!(w, "VGS,VDS,IDS,IDS_PRED")?;
    for (&vgs, (&vds, (&ids_t, &ids_p))) in vgs_test
        .iter()
        .zip(vds_test.iter().zip(ids_test.iter().zip(ids_pred.iter())))
    {
        writeln!(w, "{},{},{},{}", vgs, vds, ids_t, ids_p)?;
    }

    let root = BitMapBackend::new("plots/3d_scatter_PiNN.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0f32..1.1f32, 0f32..1.1f32)?;
    chart
        .configure_mesh()
        .x_desc("Vds")
        .y_desc("Ids")
        .axis_desc_style(("sans-serif", 15))
        .draw()?;
    chart
        .draw_series(
            vds_test
                .iter()
                .zip(ids_test.iter())
                .map(|(vds, ids)| Circle::new((*vds, *ids), 2, RED.filled())),
        )
        .unwrap();
    chart
        .draw_series(
            vds_test
                .iter()
                .zip(ids_pred.iter())
                .map(|(vds, ids)| Circle::new((*vds, *ids), 2, BLUE.filled())),
        )
        .unwrap();

    Ok(())
}

#[test]
fn test_pinn() {
    use crate::transistor_model::PiNN;
    let _ = PiNN::run();
}
