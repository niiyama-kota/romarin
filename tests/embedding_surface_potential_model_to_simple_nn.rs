#[test]
fn test_pinn_embedding_sp_model_to_simple_nn() {
    use romarin::components::graph::Graph;
    use romarin::components::utils::Activations;
    use romarin::components::{edge::*, node::*};
    use romarin::loader;
    use romarin::loader::{min_max_scaling, DataSet};
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
    let scaled_ref_dataset = min_max_scaling(&ref_dataset);
    // define min max scale factor
    let ids_min = *scaled_ref_dataset.minimums.get("IDS").unwrap() as f64;
    let ids_max = *scaled_ref_dataset.maximums.get("IDS").unwrap() as f64;
    let vds_min = *scaled_ref_dataset.minimums.get("VDS").unwrap() as f64;
    let vds_max = *scaled_ref_dataset.maximums.get("VDS").unwrap() as f64;
    let vgs_min = *scaled_ref_dataset.minimums.get("VGS").unwrap() as f64;
    let vgs_max = *scaled_ref_dataset.maximums.get("VGS").unwrap() as f64;

    // processing input and output data
    let mut xs = HashMap::new();
    let scaled_vg: Tensor = (Tensor::from_slice(&dataset.vgs) - vgs_min) / (vgs_max - vgs_min);
    let scaled_vd: Tensor = (Tensor::from_slice(&dataset.vds) - vds_min) / (vds_max - vds_min);
    let scaled_id: Tensor = (Tensor::from_slice(&dataset.ids) - ids_min) / (ids_max - ids_min);
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
    let mut xs_ref = HashMap::new();
    xs_ref.insert(
        "input".to_owned(),
        Tensor::stack(
            &[
                Tensor::from_slice(&scaled_ref_dataset.get("VGS").unwrap().as_slice()),
                Tensor::from_slice(&scaled_ref_dataset.get("VDS").unwrap().as_slice()),
            ],
            1,
        ),
    );
    let mut y_ref = HashMap::new();
    y_ref.insert(
        "output".to_owned(),
        Tensor::from_slice(&scaled_ref_dataset.get("IDS").unwrap().as_slice())
            .to_kind(Kind::Float)
            .reshape([-1, 1]),
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
        opt.zero_grad();
        // calculate loss1
        let output_mp = pinn.forward(&xs);
        let loss1 = output_mp
            .get("output")
            .unwrap()
            .mse_loss(y.get("output").unwrap(), tch::Reduction::Mean);
        println!("loss1: {}", loss1.double_value(&[]));

        // calculate loss2
        let ref_output_mp = pinn.forward(&xs_ref);
        let loss2 = ref_output_mp
            .get("output")
            .unwrap()
            .mse_loss(y_ref.get("output").unwrap(), tch::Reduction::Mean);
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
    let mut w = BufWriter::new(
        File::create(data_output_path.join("embedding_sp_model_prediction.csv")).unwrap(),
    );
    let output = pinn
        .forward(&xs_ref)
        .get("output")
        .unwrap()
        .copy()
        .reshape([-1, 1]);
    let mut ids_pred: Vec<f32> = vec![0.0; output.numel()];
    output.copy_data(&mut ids_pred, output.numel());

    let _ = writeln!(w, "VGS,VDS,IDS_PRED");
    for (&vgs, (&vds, &ids_p)) in scaled_ref_dataset.get("VGS").unwrap().iter().zip(
        scaled_ref_dataset
            .get("VDS")
            .unwrap()
            .iter()
            .zip(ids_pred.iter()),
    ) {
        let maximums = &scaled_ref_dataset.maximums;
        let minimums = &scaled_ref_dataset.minimums;
        let vgs = vgs * (maximums.get("VGS").unwrap() - minimums.get("VGS").unwrap())
            + minimums.get("VGS").unwrap();
        let vds = vds * (maximums.get("VDS").unwrap() - minimums.get("VDS").unwrap())
            + minimums.get("VDS").unwrap();
        let ids_p = ids_p * (maximums.get("IDS").unwrap() - minimums.get("IDS").unwrap())
            + minimums.get("IDS").unwrap();
        let _ = writeln!(w, "{},{},{}", vgs, vds, ids_p);
    }
}
