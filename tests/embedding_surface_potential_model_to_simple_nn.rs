#[test]
fn test_pinn_embedding_sp_model_to_simple_nn() {
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
        nn::{self, linear, LinearConfig, OptimizerConfig},
        Kind, Tensor,
    };

    // load datasets
    let dataset = loader::read_csv("data/SCT2450KE_train.csv".to_string()).unwrap();
    let ref_dataset =
        loader::read_csv("data/SCT2450KE_surface_potential_reference_data.csv".to_string())
            .unwrap();

    // processing input and output data
    let mut xs = HashMap::new();
    let vg = Tensor::from_slice(&dataset.vgs);
    let vd = Tensor::from_slice(&dataset.vds);
    let id = Tensor::from_slice(&dataset.ids);
    xs.insert(
        "vgs".to_owned(),
        Tensor::stack(&[vg], 1)
            .to_kind(Kind::Float)
            .reshape([-1, 1]),
    );
    xs.insert(
        "vds".to_owned(),
        Tensor::stack(&[vd], 1)
            .to_kind(Kind::Float)
            .reshape([-1, 1]),
    );
    let mut y = HashMap::new();
    y.insert("ids".to_owned(), id.to_kind(Kind::Float).reshape([-1, 1]));
    let mut xs_ref = HashMap::new();
    xs_ref.insert(
        "vgs".to_owned(),
        Tensor::stack(
            &[
                // Tensor::from_slice(&scaled_ref_dataset.get("VGS").unwrap().as_slice()),
                // Tensor::from_slice(&scaled_ref_dataset.get("VDS").unwrap().as_slice()),
                Tensor::from_slice(&ref_dataset.vgs.as_slice()),
            ],
            1,
        )
        .reshape([-1, 1]),
    );
    xs_ref.insert(
        "vds".to_owned(),
        Tensor::stack(
            &[
                // Tensor::from_slice(&scaled_ref_dataset.get("VGS").unwrap().as_slice()),
                // Tensor::from_slice(&scaled_ref_dataset.get("VDS").unwrap().as_slice()),
                Tensor::from_slice(&ref_dataset.vds.as_slice()),
            ],
            1,
        )
        .reshape([-1, 1]),
    );
    let mut y_ref = HashMap::new();
    y_ref.insert(
        "ids".to_owned(),
        // Tensor::from_slice(&scaled_ref_dataset.get("IDS").unwrap().as_slice())
        Tensor::from_slice(&ref_dataset.ids.as_slice())
            .to_kind(Kind::Float)
            .reshape([-1, 1]),
    );

    // define nn architecture
    let vs = nn::VarStore::new(tch::Device::Cpu);
    let mut pinn = Graph::new();
    let input_vgs = NodeType::Input(InputNode::new(
        1,
        Activations::Scale(20.0),
        AccFn::Sum,
        "vgs",
        &["V(b_gs)"],
    ));
    let input_vds = NodeType::Input(InputNode::new(
        1,
        Activations::Scale(600.0),
        AccFn::Sum,
        "vds",
        &["V(b_ds)"],
    ));
    let ids = NodeType::Output(OutputNode::new(
        1,
        Activations::Scale(1.0 / 40.0),
        AccFn::Sum,
        "ids",
        &["I(b_ds)"],
    ));
    let h1 = NodeType::Hidden(HiddenNode::new(100, Activations::ReLU, AccFn::Sum, "h1"));
    let h2 = NodeType::Hidden(HiddenNode::new(100, Activations::ReLU, AccFn::Sum, "h2"));

    let vgs2h1 = Linear::new(
        input_vgs,
        h1,
        linear(
            vs.root(),
            input_vgs.size() as i64,
            h1.size() as i64,
            LinearConfig::default(),
        ),
    );
    let vds2h1 = Linear::new(
        input_vds,
        h1,
        linear(
            vs.root(),
            input_vds.size() as i64,
            h1.size() as i64,
            LinearConfig::default(),
        ),
    );
    let h12h2 = Linear::new(
        h1,
        h2,
        linear(
            vs.root(),
            h1.size() as i64,
            h2.size() as i64,
            LinearConfig::default(),
        ),
    );
    let h22ids = Linear::new(
        h2,
        ids,
        linear(
            vs.root(),
            h2.size() as i64,
            ids.size() as i64,
            LinearConfig::default(),
        ),
    );

    pinn.add_edge(vgs2h1);
    pinn.add_edge(vds2h1);
    pinn.add_edge(h12h2);
    pinn.add_edge(h22ids);

    // run training
    let lr = 1e-2;
    let epoch = 5000;
    let mut opt = nn::AdamW::default().build(&vs, lr).unwrap();
    
    for _epoch in 1..=epoch {
        println!("epoch {}", _epoch);
        opt.zero_grad();
        // calculate loss1
        let output_mp = pinn.forward(&xs);
        let loss1 = output_mp
            .get("ids")
            .unwrap()
            .mse_loss(y.get("ids").unwrap(), tch::Reduction::Mean);
        println!("loss1: {}", loss1.double_value(&[]));

        // calculate loss2
        let ref_output_mp = pinn.forward(&xs_ref);
        let loss2 = ref_output_mp
            .get("ids")
            .unwrap()
            .mse_loss(y_ref.get("ids").unwrap(), tch::Reduction::Mean);
        println!("loss2: {}", loss2.double_value(&[]));

        // let loss = loss1 + (1e1 / (_epoch as f32 + 1.0) * loss2);
        let loss: Tensor = 0.8 * loss1 + 0.2 * loss2;
        println!("loss: {}", loss.double_value(&[]));
        opt.backward_step(&loss);
    }

    // output verilog-A code
    let va_code = pinn.gen_verilog();
    // println!("{}", va_code);

    let data_output_path = Path::new("./sim");
    let _ = create_dir_all(&data_output_path);
    let mut w =
        BufWriter::new(File::create(data_output_path.join("auto_generated_model.va")).unwrap());

    let _ = writeln!(w, "{}", va_code);

    // output model prediction for plot
    let data_output_path = Path::new("./plot");
    let _ = create_dir_all(&data_output_path);
    let mut w = BufWriter::new(
        File::create(data_output_path.join("embedding_sp_model_prediction.csv")).unwrap(),
    );
    let output = pinn
        .forward(&xs_ref)
        .get("ids")
        .unwrap()
        .copy()
        .reshape([-1, 1]);
    let mut ids_pred: Vec<f32> = vec![0.0; output.numel()];
    output.copy_data(&mut ids_pred, output.numel());

    let _ = writeln!(w, "VGS,VDS,IDS_PRED");
    // for (&vgs, (&vds, &ids_p)) in scaled_ref_dataset.get("VGS").unwrap().iter().zip(
    //     scaled_ref_dataset
    //         .get("VDS")
    //         .unwrap()
    //         .iter()
    //         .zip(ids_pred.iter()),
    // ) {
    //     // let maximums = &scaled_ref_dataset.maximums;
    //     // let minimums = &scaled_ref_dataset.minimums;
    //     let vgs = vgs * (maximums.get("VGS").unwrap() - minimums.get("VGS").unwrap())
    //         + minimums.get("VGS").unwrap();
    //     let vds = vds * (maximums.get("VDS").unwrap() - minimums.get("VDS").unwrap())
    //         + minimums.get("VDS").unwrap();
    //     let ids_p = ids_p * (maximums.get("IDS").unwrap() - minimums.get("IDS").unwrap())
    //         + minimums.get("IDS").unwrap();
    //     let _ = writeln!(w, "{},{},{}", vgs, vds, ids_p);
    // }
    for (vgs, (vds, ids_p)) in ref_dataset
        .vgs
        .into_iter()
        .zip(ref_dataset.vds.into_iter().zip(ids_pred.iter()))
    {
        let _ = writeln!(w, "{},{},{}", vgs, vds, ids_p);
    }
}
