#[test]
fn test_transfer_learning() {
    use romarin::components::graph::Graph;
    use romarin::components::utils::Activations;
    use romarin::components::{edge::*, node::*};
    use romarin::loader;
    use romarin::transistor_model::physics::level1::Level1;
    use std::io::Write;
    use std::{
        collections::HashMap,
        fs::{create_dir_all, File},
        io::BufWriter,
        path::Path,
    };
    use tch::{
        nn::{self, linear, LinearConfig, OptimizerConfig},
        Kind, Tensor,
    };

    // load datasets
    let domain_dataset = loader::read_csv("data/SCT2450KE_train.csv".to_string()).unwrap();
    let target_dataset =
        loader::read_csv("data/SCT2080KE_ID-VDS-VGS_train.csv".to_string()).unwrap();

    // define scaling factor
    let domain_ids_max = domain_dataset
        .ids
        .iter()
        .fold(f32::MIN, |m, x| f32::max(m, *x)) as f64;
    let target_ids_max = target_dataset
        .ids
        .iter()
        .fold(f32::MIN, |m, x| f32::max(m, *x)) as f64;
    let mut threshod_model = Level1::new(0.83, 0.022, 5.99);
    threshod_model.simulated_anealing(&domain_dataset, 1.0, 100000);
    let domain_vth = threshod_model.params().2 as f64;
    threshod_model.simulated_anealing(&target_dataset, 1.0, 100000);
    let target_vth = threshod_model.params().2 as f64;

    // processing input and output data
    let mut xs = HashMap::new();
    let scaled_vg: Tensor = Tensor::from_slice(&domain_dataset.vgs) / (domain_vth);
    let scaled_vd: Tensor = Tensor::from_slice(&domain_dataset.vds) / (domain_vth);
    let scaled_id: Tensor = Tensor::from_slice(&domain_dataset.ids) / (domain_ids_max);
    xs.insert(
        "input".to_owned(),
        Tensor::stack(&[scaled_vg, scaled_vd], 1)
            .to_kind(Kind::Float)
            .reshape([-1, 2]),
    );
    let mut y = HashMap::new();
    y.insert(
        "output".to_owned(),
        scaled_id.to_kind(Kind::Float).reshape([-1, 1]),
    );

    // define nn architecture
    let vs = nn::VarStore::new(tch::Device::Cpu);
    let mut pinn = Graph::new();
    let input = NodeType::Input(InputNode::new(
        2,
        Activations::Id,
        AccFn::Sum,
        "input",
        &["V(b_gs)", "V(b_ds)"],
    ));
    let output = NodeType::Output(OutputNode::new(
        1,
        Activations::Id,
        AccFn::Sum,
        "output",
        &["I(b_ds)"],
    ));
    let h1 = NodeType::Hidden(HiddenNode::new(100, Activations::ReLU, AccFn::Sum, "h1"));
    let h2 = NodeType::Hidden(HiddenNode::new(100, Activations::ReLU, AccFn::Sum, "h2"));

    let l1 = Linear::new(
        input,
        h1,
        linear(
            vs.root(),
            input.size() as i64,
            h1.size() as i64,
            LinearConfig::default(),
        ),
    );
    let l2 = Linear::new(
        h1,
        h2,
        linear(
            vs.root(),
            h1.size() as i64,
            h2.size() as i64,
            LinearConfig::default(),
        ),
    );
    let l3 = Linear::new(
        h2,
        output,
        linear(
            vs.root(),
            h2.size() as i64,
            output.size() as i64,
            LinearConfig::default(),
        ),
    );

    pinn.add_edge(l1);
    pinn.add_edge(l2);
    pinn.add_edge(l3);

    // run training
    let lr = 1e-3;
    let epoch = 10000;
    let mut opt = nn::AdamW::default().build(&vs, lr).unwrap();

    for _epoch in 1..=epoch {
        println!("epoch {}", _epoch);
        opt.zero_grad();
        // calculate loss1
        let output_mp = pinn.forward(&xs);
        let loss = output_mp
            .get("output")
            .unwrap()
            .mse_loss(y.get("output").unwrap(), tch::Reduction::Mean);

        println!("loss: {}", loss.double_value(&[]));
        opt.backward_step(&loss);
    }

    // set relearning config
    pinn.grad(&[1,2], false);

    // prepare target dataset
    let mut xs = HashMap::new();
    let scaled_vg: Tensor = Tensor::from_slice(&target_dataset.vgs) / (target_vth);
    let scaled_vd: Tensor = Tensor::from_slice(&target_dataset.vds) / (target_vth);
    let scaled_id: Tensor = Tensor::from_slice(&target_dataset.ids) / (target_ids_max);
    xs.insert(
        "input".to_owned(),
        Tensor::stack(&[scaled_vg, scaled_vd], 1)
            .to_kind(Kind::Float)
            .reshape([-1, 2]),
    );
    let mut y = HashMap::new();
    y.insert(
        "output".to_owned(),
        scaled_id.to_kind(Kind::Float).reshape([-1, 1]),
    );

    // run fine tuning
    let lr = 1e-3;
    let epoch = 10000;
    let mut opt = nn::AdamW::default().build(&vs, lr).unwrap();

    for _epoch in 1..=epoch {
        println!("epoch {}", _epoch);
        opt.zero_grad();
        // calculate loss1
        let output_mp = pinn.forward(&xs);
        let loss = output_mp
            .get("output")
            .unwrap()
            .mse_loss(y.get("output").unwrap(), tch::Reduction::Mean);

        println!("loss: {}", loss.double_value(&[]));
        opt.backward_step(&loss);
    }

    // output verilog-A code
    let va_code = pinn.gen_verilog();

    let data_output_path = Path::new("./sim");
    let _ = create_dir_all(&data_output_path);
    let mut w =
        BufWriter::new(File::create(data_output_path.join("auto_generated_model.va")).unwrap());

    let _ = writeln!(w, "{}", va_code);

    // output model prediction for plot
    let data_output_path = Path::new("./plot");
    let mut w = BufWriter::new(
        File::create(data_output_path.join("transfer_learning_model_prediction.csv")).unwrap(),
    );
    // let mut xs = HashMap::new();
    // xs.insert(
    //     "input".to_owned(),
    //     Tensor::stack(
    //         &[
    //             Tensor::from_slice(
    //                 domain_dataset
    //                     .vgs
    //                     .into_iter()
    //                     .map(|x| x as f64 * target_vth / domain_vth)
    //                     .collect::<Vec<_>>()
    //                     .as_slice(),
    //             ),
    //             Tensor::from_slice(
    //                 domain_dataset
    //                     .vds
    //                     .into_iter()
    //                     .map(|x| x as f64 * target_vth / domain_vth)
    //                     .collect::<Vec<_>>()
    //                     .as_slice(),
    //             ),
    //         ],
    //         1,
    //     )
    //     .to_kind(Kind::Float)
    //     .reshape([-1, 2]),
    // );
    let output = pinn
        .forward(&xs)
        .get("output")
        .unwrap()
        .copy()
        .reshape([-1, 1]);
    let mut ids_pred: Vec<f32> = vec![0.0; output.numel()];
    output.copy_data(&mut ids_pred, output.numel());

    let _ = writeln!(w, "VGS,VDS,IDS,IDS_PRED");
    for (vgs, (vds, (ids, ids_p))) in target_dataset.vgs.into_iter().zip(
        target_dataset.vds.into_iter().zip(
            target_dataset
                .ids
                .clone()
                .into_iter()
                .zip(ids_pred.into_iter()),
        ),
    ) {
        let vgs = vgs as f64;
        let vds = vds as f64;
        let ids = ids as f64;
        let ids_p = ids_p as f64 * target_ids_max;
        let _ = writeln!(w, "{},{},{},{}", vgs, vds, ids, ids_p as f64);
    }

    println!("DEBUG: dvth = {}", domain_vth);
    println!("DEBUG: tvth = {}", target_vth);
}
