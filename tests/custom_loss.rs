use tch::{
    nn::{self, OptimizerConfig},
    Device, Kind, Tensor,
};

#[test]
fn test_custom_loss() {
    let x = Tensor::from_slice2(&[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        .reshape([-1, 5])
        .to_kind(Kind::Float)
        .set_requires_grad(true);
    // y = sum(xi^2)
    let y = Tensor::from_slice2(&[[2, 4, 6, 8, 10], [12, 14, 16, 18, 20]])
        .reshape([-1, 5])
        .to_kind(Kind::Float)
        .set_requires_grad(true);
    let vs = nn::VarStore::new(Device::Cpu);
    let k = vs
        .root()
        .f_var("coef", &[], nn::Init::Uniform { lo: 0f64, up: 1f64 })
        .unwrap();
    println!("{:?}", vs.variables());

    let mut opt = nn::AdamW::default().build(&vs, 5e-4).unwrap();
    for _ in 0..=10000 {
        opt.zero_grad();
        let out = &k * &x;
        let loss = ((&out - &y) * (&out - &y)).mean(Some(Kind::Float));
        loss.backward();
        opt.step();
        loss.print();
        k.grad().print();
    }
    (&k * &x).print();
    x.print();
    y.print();
    println!("");
}

#[test]
fn test_pinn_with_monotonous_restrict() {
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

    let dataset = loader::read_csv("data/SCT2080KE_ID-VDS-VGS_train.csv".to_string()).unwrap();

    let mut xs = HashMap::new();
    xs.insert(
        "vd_input".to_owned(),
        Tensor::from_slice(dataset.vds.as_slice())
            .to_kind(Kind::Float)
            .reshape([-1, 1])
            .set_requires_grad(true),
    );
    xs.insert(
        "vg_input".to_owned(),
        Tensor::from_slice(dataset.vgs.as_slice())
            .to_kind(Kind::Float)
            .reshape([-1, 1])
            .set_requires_grad(true),
    );
    let mut y = HashMap::new();
    y.insert(
        "ids_output".to_owned(),
        Tensor::from_slice(dataset.ids.as_slice())
            .to_kind(Kind::Float)
            .reshape([-1, 1]),
    );

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
        pinn.vs.root(),
        1,
        20,
        LinearConfig {
            ws_init: nn::init::DEFAULT_KAIMING_UNIFORM,
            bs_init: None,
            bias: false,
        },
    );
    let vd12vd2 = nn::linear(
        pinn.vs.root(),
        20,
        1,
        LinearConfig {
            ws_init: nn::init::DEFAULT_KAIMING_UNIFORM,
            bs_init: None,
            bias: false,
        },
    );
    let vd22o = nn::linear(
        pinn.vs.root(),
        1,
        1,
        LinearConfig {
            ws_init: nn::init::DEFAULT_KAIMING_UNIFORM,
            bs_init: None,
            bias: false,
        },
    );
    let ig2vg1 = nn::linear(
        pinn.vs.root(),
        1,
        30,
        LinearConfig {
            ws_init: nn::init::DEFAULT_KAIMING_UNIFORM,
            bs_init: None,
            bias: false,
        },
    );
    let vg12vg2 = nn::linear(
        pinn.vs.root(),
        30,
        1,
        LinearConfig {
            ws_init: nn::init::DEFAULT_KAIMING_UNIFORM,
            bs_init: None,
            bias: false,
        },
    );
    let vg22o = nn::linear(
        pinn.vs.root(),
        1,
        1,
        LinearConfig {
            ws_init: nn::init::DEFAULT_KAIMING_UNIFORM,
            bs_init: None,
            bias: false,
        },
    );
    let vd12vg1 = nn::linear(
        pinn.vs.root(),
        20,
        30,
        LinearConfig {
            ws_init: nn::init::DEFAULT_KAIMING_UNIFORM,
            bs_init: None,
            bias: false,
        },
    );
    let vd22vg2 = nn::linear(
        pinn.vs.root(),
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

    let lr = 1e-3;
    let epoch = 10000;
    let mut opt = nn::AdamW::default().build(&pinn.vs, lr).unwrap();

    for _epoch in 1..=epoch {
        println!("epoch {}", _epoch);
        // calculate loss1
        let output_mp = pinn.forward(&xs);
        opt.zero_grad();
        let loss1 = output_mp
            .get("ids_output")
            .unwrap()
            .mse_loss(y.get("ids_output").unwrap(), tch::Reduction::Mean);
        println!("loss1: {}", loss1.double_value(&[]));
        opt.backward_step(&loss1);

        // calculate loss2
        xs.insert("vd_input".to_owned(), xs.get("vd_input").unwrap().detach());
        let vd = xs.get("vd_input").unwrap().set_requires_grad(true);
        opt.zero_grad();
        let output_mp = pinn.forward(&xs);
        let ids = output_mp.get("ids_output").unwrap();
        ids.mean(Some(Kind::Float)).backward();
        let loss2: Tensor = vd.grad() * vd.tanh();
        let loss2 = loss2.mean(Some(Kind::Float));
        println!("loss2: {}", loss2.double_value(&[]));
        opt.backward_step(&loss2);
    }

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
