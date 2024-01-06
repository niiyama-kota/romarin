#[test]
fn test_pinn_by_graph() {
    use romarin::components::graph::Graph;
    use romarin::components::utils::Activations;
    use romarin::components::{edge::*, node::*};
    use romarin::loader;
    use std::io::Write;
    use std::{
        collections::HashMap,
        fs::{create_dir_all, File},
        io::BufWriter,
        path::Path,
    };
    use tch::{
        nn::{self, LinearConfig},
        Kind, Tensor,
    };

    let dataset = loader::read_csv("data/SCT2080KE_ID-VDS-VGS.csv".to_string()).unwrap();

    let mut xs = HashMap::new();
    xs.insert(
        "vd_input".to_owned(),
        Tensor::from_slice(dataset.vds.as_slice())
            .to_kind(Kind::Float)
            .reshape([-1, 1]),
    );
    xs.insert(
        "vg_input".to_owned(),
        Tensor::from_slice(dataset.vgs.as_slice())
            .to_kind(Kind::Float)
            .reshape([-1, 1]),
    );
    let mut y = HashMap::new();
    y.insert(
        "ids_output".to_owned(),
        Tensor::from_slice(dataset.ids.as_slice())
            .to_kind(Kind::Float)
            .reshape([-1, 1]),
    );

    let vs = nn::VarStore::new(tch::Device::Cpu);
    let mut pinn = Graph::new();
    let input_vd = NodeType::Input(InputNode::new(
        1,
        Activations::Id,
        AccFn::Sum,
        "vd_input",
        &["V(b_ds)"],
    ));
    let input_vg = NodeType::Input(InputNode::new(
        1,
        Activations::Id,
        AccFn::Sum,
        "vg_input",
        &["V(b_gs)"],
    ));
    let vd_sub1 = NodeType::Hidden(HiddenNode::new(
        20,
        Activations::Tanh,
        AccFn::Sum,
        "vd_sub1",
    ));
    let vd_sub2 = NodeType::Hidden(HiddenNode::new(1, Activations::Tanh, AccFn::Sum, "vd_sub2"));
    let vg_sub1 = NodeType::Hidden(HiddenNode::new(
        30,
        Activations::Sigmoid,
        AccFn::Sum,
        "vg_sub1",
    ));
    let vg_sub2 = NodeType::Hidden(HiddenNode::new(
        1,
        Activations::Sigmoid,
        AccFn::Sum,
        "vg_sub2",
    ));
    let output = NodeType::Output(OutputNode::new(
        1,
        Activations::Id,
        AccFn::Prod,
        "ids_output",
        &["I(b_ds)"],
    ));
    let id2vd1 = nn::linear(
        vs.root(),
        1,
        20,
        LinearConfig {
            ws_init: nn::init::DEFAULT_KAIMING_UNIFORM,
            bs_init: None,
            bias: false,
        },
    );
    let vd12vd2 = nn::linear(
        vs.root(),
        20,
        1,
        LinearConfig {
            ws_init: nn::init::DEFAULT_KAIMING_UNIFORM,
            bs_init: None,
            bias: false,
        },
    );
    let vd22o = nn::linear(
        vs.root(),
        1,
        1,
        LinearConfig {
            ws_init: nn::init::DEFAULT_KAIMING_UNIFORM,
            bs_init: None,
            bias: false,
        },
    );
    let ig2vg1 = nn::linear(
        vs.root(),
        1,
        30,
        LinearConfig {
            ws_init: nn::init::DEFAULT_KAIMING_UNIFORM,
            bs_init: None,
            bias: false,
        },
    );
    let vg12vg2 = nn::linear(
        vs.root(),
        30,
        1,
        LinearConfig {
            ws_init: nn::init::DEFAULT_KAIMING_UNIFORM,
            bs_init: None,
            bias: false,
        },
    );
    let vg22o = nn::linear(
        vs.root(),
        1,
        1,
        LinearConfig {
            ws_init: nn::init::DEFAULT_KAIMING_UNIFORM,
            bs_init: None,
            bias: false,
        },
    );
    let vd12vg1 = nn::linear(
        vs.root(),
        20,
        30,
        LinearConfig {
            ws_init: nn::init::DEFAULT_KAIMING_UNIFORM,
            bs_init: None,
            bias: false,
        },
    );
    let vd22vg2 = nn::linear(
        vs.root(),
        1,
        1,
        LinearConfig {
            ws_init: nn::init::DEFAULT_KAIMING_UNIFORM,
            bs_init: None,
            bias: false,
        },
    );

    pinn.add_edge(Linear::new(input_vd, vd_sub1, id2vd1));
    pinn.add_edge(Linear::new(input_vg, vg_sub1, ig2vg1));
    pinn.add_edge(Linear::new(vd_sub1, vg_sub1, vd12vg1));
    pinn.add_edge(Linear::new(vd_sub1, vd_sub2, vd12vd2));
    pinn.add_edge(Linear::new(vg_sub1, vg_sub2, vg12vg2));
    pinn.add_edge(Linear::new(vd_sub2, vg_sub2, vd22vg2));
    pinn.add_edge(Linear::new(vd_sub2, output, vd22o));
    pinn.add_edge(Linear::new(vg_sub2, output, vg22o));

    // let _ = pinn.train(&xs, &y, 10000, 1e-3);
    let va_code = pinn.gen_verilog();
    // println!("{}", va_code);

    let data_output_path = Path::new("./data");
    let _ = create_dir_all(&data_output_path);
    let mut w =
        BufWriter::new(File::create(data_output_path.join("auto_generated_model.va")).unwrap());

    let _ = writeln!(w, "{}", va_code);

    let mut w = BufWriter::new(File::create(data_output_path.join("test_data.csv")).unwrap());
    let output = pinn
        .forward(&xs)
        .get("ids_output")
        .unwrap()
        .copy()
        .reshape([-1, 1]);
    let mut ids_pred: Vec<f32> = vec![0.0; output.numel()];
    output.copy_data(&mut ids_pred, output.numel());

    let _ = writeln!(w, "VGS,VDS,IDS,IDS_PRED");
    for (&vgs, (&vds, (&ids_t, &ids_p))) in dataset.vgs.iter().zip(
        dataset
            .vds
            .iter()
            .zip(dataset.ids.iter().zip(ids_pred.iter())),
    ) {
        let _ = writeln!(w, "{},{},{},{}", vgs, vds, ids_t, ids_p);
    }
}
