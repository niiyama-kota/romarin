// use anyhow::Result;
// use plotters::prelude::*;
// use std::collections::HashMap;
// use std::fs::{create_dir_all, File};
// use std::io::{BufWriter, Write};
// use std::path::Path;
// use tch::nn::ModuleT;
// use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

// use crate::components::utils::{
//     self, array_init, declare_activation, declare_matrix_add, declare_matrix_mul, declare_tensor,
//     mosfet_template,
// };
// use crate::loader::{self, min_max_scaling, DataSet};

// const NEURON_NUM: i64 = 100;
// const EPOCH: i64 = 10000;

// #[derive(Debug)]
// struct SimpleNet {
//     input_layer: nn::Linear,
//     hidden_layer: nn::Linear,
//     output_layer: nn::Linear,
// }
// impl SimpleNet {
//     fn new(vs: &nn::Path) -> Self {
//         let _input = nn::linear(vs / "input layer", 2, NEURON_NUM, Default::default());
//         let _hidden = nn::linear(
//             vs / "hidden layer",
//             NEURON_NUM,
//             NEURON_NUM,
//             Default::default(),
//         );
//         let _output = nn::linear(vs / "output layer", NEURON_NUM, 1, Default::default());
//         Self {
//             input_layer: _input,
//             hidden_layer: _hidden,
//             output_layer: _output,
//         }
//     }

//     fn rmse(&self, xs: &tch::Tensor, ys: &tch::Tensor) -> tch::Tensor {
//         (ys - self.forward_t(xs, false))
//             .square()
//             .mean(Kind::Float)
//             .sqrt()
//     }

//     pub fn export_params(
//         &self,
//         minimums: HashMap<String, f32>,
//         maximums: HashMap<String, f32>,
//     ) -> Result<()> {
//         let data_output_path = Path::new("./data");
//         create_dir_all(&data_output_path)?;
//         let mut w = BufWriter::new(File::create(data_output_path.join("model_parameters.va"))?);

//         writeln!(w, "{}", "`include \"disciplines.vams\"")?;

//         let declare_relu = declare_activation(utils::Activations::ReLU);
//         // `define relu(xs, x_dim)\\\n\tfor (i = 0; i < x_dim; i = i + 1) begin\\\n\t\txs[i] = (xs[i] + abs(xs[i])) / 2;\\\n\tend\n
//         writeln!(w, "{}", declare_relu)?;
//         writeln!(w, "{}", declare_matrix_mul())?;
//         writeln!(w, "{}", declare_matrix_add())?;

//         let mut header = "".to_owned();
//         header += "\tinteger i, j, k;\n\t";
//         // header += "\treal Vgs, Vds, Vgd;\n";

//         let l1 = &self.input_layer;
//         let ws = &l1.ws;
//         let declare_w1 = declare_tensor(ws, "W1", Some(2));
//         header += &declare_w1.replace("\n", "\n\t");
//         if let Some(bs) = &l1.bs {
//             let declare_b1 = declare_tensor(bs, "B1", Some(1));
//             header += &declare_b1.replace("\n", "\n\t");
//         }

//         let l2 = &self.hidden_layer;
//         let ws = &l2.ws;
//         let declare_w2 = declare_tensor(ws, "W2", Some(NEURON_NUM as usize));
//         header += &declare_w2.replace("\n", "\n\t");
//         if let Some(bs) = &l2.bs {
//             let declare_b2 = declare_tensor(bs, "B2", Some(1));
//             header += &declare_b2.replace("\n", "\n\t");
//         }

//         let l3 = &self.output_layer;
//         let ws = &l3.ws;
//         let declare_w3 = declare_tensor(ws, "W3", Some(NEURON_NUM as usize));
//         header += &declare_w3.replace("\n", "\n\t");
//         if let Some(bs) = &l3.bs {
//             let declare_b3 = declare_tensor(bs, "B3", Some(1));
//             header += &declare_b3.replace("\n", "\n\t");
//         }

//         header += &format!("\n\treal inputs[0:1] = {};\n", array_init(2, 0.0));
//         header += &format!(
//             "\treal X1[0:({})-1] = {};\n",
//             NEURON_NUM,
//             array_init(NEURON_NUM as usize, 0.0)
//         );
//         header += &format!(
//             "\treal X2[0:({})-1] = {};\n",
//             NEURON_NUM,
//             array_init(NEURON_NUM as usize, 0.0)
//         );
//         header += "\treal X3[0:0] = {0};\n";

//         let mut content = "".to_owned();
//         content += &format!(
//             "\tinputs[0] = (V(b_ds) - ({}))/(({}) - ({}));\n\tinputs[1] = (V(b_gs) - ({}))/(({}) - ({}));\n",
//             minimums.get("Vds").unwrap(),
//             maximums.get("Vds").unwrap(),
//             minimums.get("Vds").unwrap(),
//             minimums.get("Vgs").unwrap(),
//             maximums.get("Vgs").unwrap(),
//             minimums.get("Vgs").unwrap(),
//         );
//         content += &format!("\t`relu(inputs, ({}));\n", 2);
//         content += &format!("\t`MATMUL(W1, inputs, X1, ({}), 1, 2);\n", NEURON_NUM);
//         content += &format!("\t`MATADD(X1, B1, ({}), 1);\n", NEURON_NUM);
//         content += &format!("\t`relu(X1, ({}));\n", NEURON_NUM);
//         content += &format!(
//             "\t`MATMUL(W2, X1, X2, ({}), 1, ({}));\n",
//             NEURON_NUM, NEURON_NUM
//         );
//         content += &format!("\t`MATADD(X2, B2, ({}), 1);\n", NEURON_NUM);
//         content += &format!("\t`relu(X2, ({}));\n", NEURON_NUM);
//         content += &format!("\t`MATMUL(W3, X2, X3, 1, 1, ({}));\n", NEURON_NUM);
//         content += "\t`MATADD(X3, B3, 1, 1);\n";
//         content += &format!(
//             "\tI(b_ds) <+ ({}) + X3[0] * (({}) - ({}));\n",
//             minimums.get("Ids").unwrap(),
//             maximums.get("Ids").unwrap(),
//             minimums.get("Ids").unwrap()
//         );

//         writeln!(w, "{}", mosfet_template(&header, &content))?;
//         Ok(())
//     }
// }

// impl ModuleT for SimpleNet {
//     fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
//         let y = self.input_layer.forward_t(&xs, train);
//         let y = self.hidden_layer.forward_t(&y, train).relu();
//         let y = self.output_layer.forward_t(&y, train).relu();

//         return y;
//     }
// }

// pub fn run() -> Result<()> {
//     let dataset = loader::read_csv("data/SCT2080KE_ID-VDS-VGS_train.csv".to_string()).unwrap();
//     let dataset = min_max_scaling(&dataset);
//     let x = Tensor::stack(
//         &[
//             Tensor::from_slice(dataset.get("Vds").unwrap().as_slice()),
//             Tensor::from_slice(dataset.get("Vgs").unwrap().as_slice()),
//         ],
//         1,
//     )
//     .to_kind(Kind::Float);
//     let y = Tensor::from_slice(dataset.get("Ids").unwrap().as_slice())
//         .to_kind(Kind::Float)
//         .reshape([-1, 1]);

//     let test_dataset = loader::read_csv("data/SCT2080KE_ID-VDS-VGS.csv".to_string()).unwrap();
//     let test_dataset = min_max_scaling(&test_dataset);
//     let x_test = Tensor::stack(
//         &[
//             Tensor::from_slice(test_dataset.get("Vds").unwrap().as_slice()),
//             Tensor::from_slice(test_dataset.get("Vgs").unwrap().as_slice()),
//         ],
//         1,
//     )
//     .to_kind(Kind::Float);
//     let y_test = Tensor::from_slice(test_dataset.get("Ids").unwrap().as_slice())
//         .to_kind(Kind::Float)
//         .reshape([-1, 1]);

//     let vs = nn::VarStore::new(Device::Cpu);
//     let net = SimpleNet::new(&vs.root());
//     let mut opt = nn::AdamW::default().build(&vs, 1e-5)?;
//     let mut losses = Vec::<f64>::new();
//     for epoch in 1..=EPOCH {
//         opt.zero_grad();
//         let loss = net.forward_t(&x, true).mse_loss(&y, tch::Reduction::Mean);
//         loss.print();
//         opt.backward_step(&loss);
//         losses.push(loss.double_value(&[]));
//         println!("epoch {}", epoch);
//     }

//     println!(
//         "initial loss: {:?} \n last loss: {:?}",
//         losses.first(),
//         losses.last()
//     );

//     let rmse = net.rmse(&x_test, &y_test);
//     println!("test rmse: {:?}", rmse);
//     let y_pred = net.forward_t(&x_test, false);

//     let mut vds_test: Vec<f32> = vec![0.0; x_test.transpose(0, 1).get(0).numel()];
//     let mut vgs_test: Vec<f32> = vec![0.0; x_test.transpose(0, 1).get(1).numel()];
//     let mut ids_test: Vec<f32> = vec![0.0; y_test.numel()];
//     let mut ids_pred: Vec<f32> = vec![0.0; y_pred.numel()];
//     x_test
//         .transpose(0, 1)
//         .get(0)
//         .copy_data(&mut vds_test, x_test.transpose(0, 1).get(0).numel());
//     x_test
//         .transpose(0, 1)
//         .get(1)
//         .copy_data(&mut vgs_test, x_test.transpose(0, 1).get(1).numel());
//     y_test.copy_data(&mut ids_test, y_test.numel());
//     y_pred.copy_data(&mut ids_pred, y_pred.numel());

//     let data_output_path = Path::new("./data");
//     create_dir_all(&data_output_path)?;
//     let mut w = BufWriter::new(File::create(data_output_path.join("test_data.csv"))?);

//     writeln!(w, "VGS,VDS,IDS,IDS_PRED")?;
//     for (&vgs, (&vds, (&ids_t, &ids_p))) in vgs_test
//         .iter()
//         .zip(vds_test.iter().zip(ids_test.iter().zip(ids_pred.iter())))
//     {
//         writeln!(w, "{},{},{},{}", vgs, vds, ids_t, ids_p)?;
//     }

//     net.export_params(dataset.minimums, dataset.maximums)?;

//     let root = BitMapBackend::new("plots/3d_scatter.png", (640, 480)).into_drawing_area();
//     root.fill(&WHITE).unwrap();
//     let mut chart = ChartBuilder::on(&root)
//         .x_label_area_size(40)
//         .y_label_area_size(40)
//         .build_cartesian_2d(0f32..1.1f32, 0f32..1.1f32)?;
//     chart
//         .configure_mesh()
//         .x_desc("Vds")
//         .y_desc("Ids")
//         .axis_desc_style(("sans-serif", 15))
//         .draw()?;
//     chart
//         .draw_series(
//             vds_test
//                 .iter()
//                 .zip(ids_test.iter())
//                 .map(|(vds, ids)| Circle::new((*vds, *ids), 2, RED.filled())),
//         )
//         .unwrap();
//     chart
//         .draw_series(
//             vds_test
//                 .iter()
//                 .zip(ids_pred.iter())
//                 .map(|(vds, ids)| Circle::new((*vds, *ids), 2, BLUE.filled())),
//         )
//         .unwrap();

//     Ok(())
// }

// #[test]
// fn test_simple_net() {
//     use crate::transistor_model::SimpleNet;
//     let _ = SimpleNet::run();
// }

// #[test]
// fn test_convert_to_vec() {
//     // Tensorを作成
//     let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
//         .reshape(&[2, 3])
//         .to_kind(Kind::Float);
//     let original_vec: Vec<f32> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].to_vec();
//     let num_tensor = tensor.numel();
//     let mut vec: Vec<f32> = vec![0.; num_tensor];
//     tensor.copy_data(&mut vec, num_tensor);

//     assert_eq!(original_vec, vec);
// }
