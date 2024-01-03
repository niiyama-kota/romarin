#[test]
fn test_pinn_embedding_sp_model() {
    use romarin::components::graph::Graph;
    use romarin::components::utils::Activations;
    use romarin::components::{edge::*, node::*};
    use romarin::loader;
    use romarin::transistor_model::physics::surface_potential::SurfacePotentialModel;
    use std::io::Write;
    use std::{
        collections::HashMap,
        fs::{create_dir_all, File},
        io::BufWriter,
        path::Path,
    };
    use tch::{
        nn::{self, LinearConfig, OptimizerConfig},
        Kind, Tensor,
    };

    let dataset = loader::read_csv("data/SCT2450KE_train.csv".to_string()).unwrap();

    let mut xs = HashMap::new();
    xs.insert(
        "vd_input".to_owned(),
        Tensor::from_slice(&dataset.vds.as_slice())
            .to_kind(Kind::Float)
            .reshape([-1, 1])
            .set_requires_grad(true),
    );
    xs.insert(
        "vg_input".to_owned(),
        Tensor::from_slice(&dataset.vgs.as_slice())
            .to_kind(Kind::Float)
            .reshape([-1, 1])
            .set_requires_grad(true),
    );
    let mut y = HashMap::new();
    y.insert(
        "ids_output".to_owned(),
        Tensor::from_slice(&dataset.ids.as_slice())
            .to_kind(Kind::Float)
            .reshape([-1, 1]),
    );

    let vg = (0..20).map(|x| x as f32 * 1.0);
    let vg = Tensor::from_slice(
        &vg.flat_map(|n| std::iter::repeat(n).take(600))
            .collect::<Vec<_>>()
            .as_slice(),
    )
    .to_kind(Kind::Float)
    .reshape([-1, 1]);
    let vd = (0..600 * 20).map(|x| (x % 600) as f32 * 1.0);
    let vd = Tensor::from_slice(&vd.collect::<Vec<_>>().as_slice())
        .to_kind(Kind::Float)
        .reshape([-1, 1]);
    let mut mesh = HashMap::new();
    mesh.insert("vd_input".to_owned(), vd.copy());
    mesh.insert("vg_input".to_owned(), vg.copy());

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

    let lr = 1e-2;
    let epoch = 5000;
    let mut opt = nn::AdamW::default().build(&pinn.vs, lr).unwrap();

    // let sp_model = SurfacePotentialModel::new(
    //     105899.84475147062,
    //     5e-8,
    //     5.074958142963157e16,
    //     0.002616791838610094,
    //     -0.2694651491610863,
    //     0.009159716686281865,
    //     1.4849531177386044,
    //     12.836176866857711,
    //     0.03204362781211176,
    // );

    for _epoch in 1..=epoch {
        println!("epoch {}", _epoch);
        // calculate loss1
        opt.zero_grad();
        let output_mp = pinn.forward(&xs);
        let loss1 = output_mp
            .get("ids_output")
            .unwrap()
            .mse_loss(y.get("ids_output").unwrap(), tch::Reduction::Mean);
        println!("loss1: {}", loss1.double_value(&[]));

        // calculate loss2
        // let output_mp = pinn.forward(&mesh);
        // let ids = output_mp
        //     .get("ids_output")
        //     .unwrap()
        //     .reshape(&[-1, 1])
        //     .to_kind(Kind::Float);
        // let loss2 = sp_model
        //     .tfun(
        //         &Tensor::cat(&[vg.copy(), vd.copy()], 1)
        //             .to_kind(Kind::Float)
        //             .reshape(&[-1, 2]),
        //     )
        //     .reshape(&[-1, 1])
        //     .to_kind(Kind::Float)
        //     .mse_loss(&ids, tch::Reduction::Mean);
        // println!("loss2: {}", loss2.double_value(&[]));

        // let loss = loss1 + (1e3 / (_epoch as f32 + 1.0) * loss2);
        // println!("loss: {}", loss.double_value(&[]));
        opt.backward_step(&loss1);
    }

    let va_code = pinn.gen_verilog();
    // println!("{}", va_code);

    let data_output_path = Path::new("./plot");
    let _ = create_dir_all(&data_output_path);
    let mut w =
        BufWriter::new(File::create(data_output_path.join("auto_generated_model.va")).unwrap());

    let _ = writeln!(w, "{}", va_code);

    let mut w = BufWriter::new(
        File::create(data_output_path.join("embedding_sp_model_prediction.csv")).unwrap(),
    );
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
