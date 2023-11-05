use crate::loader::{self, min_max_scaling, DataSet};
use anyhow::Result;
use plotters::prelude::*;
use std::collections::HashMap;
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
    let dataset = loader::read_csv("data/SCT2080KE_ID-VDS-VGS_train.csv".to_string()).unwrap();
    let dataset = min_max_scaling(&dataset);
    let x = Tensor::stack(
        &[
            Tensor::from_slice(dataset.get("Vds").unwrap().as_slice()),
            Tensor::from_slice(dataset.get("Vgs").unwrap().as_slice()),
        ],
        1,
    )
    .to_kind(Kind::Float);
    let y = Tensor::from_slice(dataset.get("Ids").unwrap().as_slice())
        .to_kind(Kind::Float)
        .reshape([-1, 1]);

    let test_dataset = loader::read_csv("data/SCT2080KE_ID-VDS-VGS.csv".to_string()).unwrap();
    let test_dataset = min_max_scaling(&test_dataset);
    let x_test = Tensor::stack(
        &[
            Tensor::from_slice(test_dataset.get("Vds").unwrap().as_slice()),
            Tensor::from_slice(test_dataset.get("Vgs").unwrap().as_slice()),
        ],
        1,
    )
    .to_kind(Kind::Float);
    let y_test = Tensor::from_slice(test_dataset.get("Ids").unwrap().as_slice())
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

#[test]
fn test_pinn_by_graph() {
    use crate::components::graph::Graph;
    use crate::components::utils::Activations;
    use crate::components::{edge::*, node::*};

    let dataset = loader::read_csv("data/SCT2080KE_ID-VDS-VGS_train.csv".to_string()).unwrap();
    let dataset = min_max_scaling(&dataset);
    // let x = Tensor::stack(
    //     &[
    //         Tensor::from_slice(dataset.get("Vds").unwrap().as_slice()),
    //         Tensor::from_slice(dataset.get("Vgs").unwrap().as_slice()),
    //     ],
    //     1,
    // )
    // .to_kind(Kind::Float)
    // .reshape([-1, 2]);
    let mut xs = HashMap::new();
    xs.insert(
        "vd_input",
        Tensor::from_slice(dataset.get("Vds").unwrap().as_slice()),
    );
    xs.insert(
        "vg_input",
        Tensor::from_slice(dataset.get("Vgs").unwrap().as_slice()),
    );
    let y = Tensor::from_slice(dataset.get("Ids").unwrap().as_slice())
        .to_kind(Kind::Float)
        .reshape([-1, 1]);

    // let test_dataset = loader::read_csv("data/SCT2080KE_ID-VDS-VGS.csv".to_string()).unwrap();
    // let test_dataset = min_max_scaling(&test_dataset);
    // let x_test = Tensor::stack(
    //     &[
    //         Tensor::from_slice(test_dataset.get("Vds").unwrap().as_slice()),
    //         Tensor::from_slice(test_dataset.get("Vgs").unwrap().as_slice()),
    //     ],
    //     1,
    // )
    // .to_kind(Kind::Float);
    // let y_test = Tensor::from_slice(test_dataset.get("Ids").unwrap().as_slice())
    //     .to_kind(Kind::Float)
    //     .reshape([-1, 1]);

    let mut pinn = Graph::new();
    let input_vd = NodeType::Input(InputNode::new(1, Activations::Id, "vd_input", &["V(b_ds)"]));
    let input_vg = NodeType::Input(InputNode::new(1, Activations::Id, "vg_input", &["V(b_gs)"]));
    let vd_sub1 = NodeType::Hidden(HiddenNode::new(2, Activations::Tanh));
    let vd_sub2 = NodeType::Hidden(HiddenNode::new(1, Activations::Tanh));
    let vg_sub1 = NodeType::Hidden(HiddenNode::new(3, Activations::Tanh));
    let vg_sub2 = NodeType::Hidden(HiddenNode::new(1, Activations::Tanh));
    let output = NodeType::Output(OutputNode::new(1, Activations::Id));
    let id2vd1 = nn::linear(pinn.vs.root(), 1, 2, Default::default());
    let vd12vd2 = nn::linear(pinn.vs.root(), 2, 1, Default::default());
    let vd22o = nn::linear(pinn.vs.root(), 1, 1, Default::default());
    let ig2vg1 = nn::linear(pinn.vs.root(), 1, 3, Default::default());
    let vg12vg2 = nn::linear(pinn.vs.root(), 3, 1, Default::default());
    let vg22o = nn::linear(pinn.vs.root(), 1, 1, Default::default());
    let vd12vg1 = nn::linear(pinn.vs.root(), 2, 3, Default::default());
    let vd22vg2 = nn::linear(pinn.vs.root(), 1, 1, Default::default());

    pinn.add_edge(Linear::new(input_vd, vd_sub1, id2vd1));
    pinn.add_edge(Linear::new(input_vg, vg_sub1, ig2vg1));
    pinn.add_edge(Linear::new(vd_sub1, vg_sub1, vd12vg1));
    pinn.add_edge(Linear::new(vd_sub1, vd_sub2, vd12vd2));
    pinn.add_edge(Linear::new(vg_sub1, vg_sub2, vg12vg2));
    pinn.add_edge(Linear::new(vd_sub2, vg_sub2, vd22vg2));
    pinn.add_edge(Linear::new(vd_sub2, output, vd22o));
    pinn.add_edge(Linear::new(vg_sub2, output, vg22o));

    let _ = pinn.train(&xs, &y, 10000, 1e-3);
    println!("{}", pinn.gen_verilog("// INPUT", "//OUTPUT"));
}

#[test]
fn test_split_tensor() {
    let dataset = loader::read_csv("data/SCT2080KE_ID-VDS-VGS_train.csv".to_string()).unwrap();
    let dataset = min_max_scaling(&dataset);
    let x = Tensor::stack(
        &[
            Tensor::from_slice(dataset.get("Vds").unwrap().as_slice()),
            Tensor::from_slice(dataset.get("Vgs").unwrap().as_slice()),
        ],
        1,
    )
    .to_kind(Kind::Float)
    .reshape([-1, 2]);
    let vgs_vds = x.split_sizes([1, 1], 1);
    x.print();
    vgs_vds[1].print();
}
