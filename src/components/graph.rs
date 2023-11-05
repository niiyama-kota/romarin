use std::collections::HashMap;

use crate::components::node::*;
use crate::components::utils::{
    array_init, declare_linear, declare_matrix_add, declare_matrix_mul,
};
use anyhow::Result;
use tch::{
    nn::{self, Module, OptimizerConfig, VarStore},
    Device, Tensor,
};

use super::edge::{Edge, Linear};

#[derive(Debug)]
pub struct Graph {
    pub vs: VarStore,
    edge_list: Vec<Linear>,
}

impl Graph {
    pub fn new() -> Self {
        Graph {
            vs: nn::VarStore::new(Device::Cpu),
            edge_list: vec![],
        }
    }

    pub fn train(
        &self,
        xs: &HashMap<&str, Tensor>,
        y: &HashMap<&str, Tensor>,
        epoch: usize,
        lr: f64,
    ) -> Result<()> {
        let mut opt = nn::AdamW::default().build(&self.vs, lr)?;
        let mut losses = Vec::<f64>::new();

        for _epoch in 1..=epoch {
            opt.zero_grad();
            let loss = self
                .forward(xs)
                .get("")
                .unwrap()
                .mse_loss(y.get("").unwrap(), tch::Reduction::Mean);
            loss.print();
            opt.backward_step(&loss);
            losses.push(loss.double_value(&[]));
            println!("epoch {}", _epoch);
        }

        Ok(())
    }

    pub fn add_edge(&mut self, e: Linear) {
        self.edge_list.push(e);
    }

    pub fn forward(&self, inputs: &HashMap<&str, Tensor>) -> HashMap<&str, Tensor> {
        let mut mp = HashMap::<NodeType, Tensor>::new();

        for e in self.edge_list.iter() {
            let from = &e.from();
            let to = &e.to();
            let t = &e.get_fun();
            let v = match mp.get(from) {
                Some(v) => v.copy(),
                None => match from {
                    NodeType::Input(n) => inputs.get(n.name()).unwrap().copy(),
                    NodeType::Hidden(_) => {
                        assert!(false, "Node Type error");
                        // FIXME
                        panic!()
                    }
                    NodeType::Output(_) => {
                        assert!(false, "Node Type error");
                        // FIXME
                        panic!()
                    }
                },
            };

            let term = t.forward(&from.forward(&v));
            if let Some(acc) = mp.get(to) {
                let acc = acc.copy() * term;
                mp.insert(*to, acc);
            } else {
                mp.insert(*to, term);
            }
        }

        let output_node = self.edge_list.last().unwrap().to();
        return output_node.forward(mp.get(&output_node).unwrap());
    }

    pub fn gen_verilog(&self, input: &str, output: &str) -> String {
        let mut var: usize = 0;
        let mut fresh = || {
            var += 1;
            return var - 1;
        };
        let mut node_variables = HashMap::<NodeType, usize>::new();
        // assume no multiple edges
        let mut edge_variables = HashMap::<(NodeType, NodeType), usize>::new();

        let mut header = "`include \"disciplines.vams\"\n\n".to_owned();
        header += &declare_matrix_mul();
        header += "\n";
        header += &declare_matrix_add();
        header += "\n";
        header += "module mosfet(term_G, term_D, term_S);\n\tinout term_G, term_D, term_S;\n\telectrical term_G, term_D, term_S;\n\tbranch (term_G, term_S) b_gs;\n\tbranch (term_G, term_D) b_gd;\n\tbranch (term_D, term_S) b_ds;\n\n\tinteger i, j, k;\n\treal tmp = 0.0;\n\n";

        let mut content = "\t".to_owned();

        for e in self.edge_list.iter() {
            let evar = fresh();
            content += &e.export_params(&format!("l{}", evar));
            edge_variables.insert((e.from(), e.to()), evar);
        }
        for e in self.edge_list.iter() {
            let from = e.from();
            if let Some(_) = node_variables.get(&from) {
                // FIXME: we should return some Error
            } else {
                let var = fresh();
                content += &from.export_init(&format!("n{var}"));
                node_variables.insert(from, var);
            }

            let to = e.to();
            if let Some(_) = node_variables.get(&to) {
            } else {
                let var = fresh();
                content += &to.export_init(&format!("n{var}"));
                node_variables.insert(to, var);
            }
        }

        for e in self.edge_list.iter() {
            let from = e.from();
            let to = e.to();
            let edge = (from, to);
            let &from_var = node_variables.get(&from).unwrap();
            let &to_var = node_variables.get(&to).unwrap();
            let &e_var = edge_variables.get(&edge).unwrap();
            content += &from.export_forward(&from_var.to_string());
            content += &e.export_forward(&format!("l{e_var}"));
            // content += &format!(
            //     "`MATMUL(l{e_var}_ws, n{from_var}, n{to_var}, {}, 1, {});\n",
            //     to.size(),
            //     from.size()
            // );
            // content += &format!("`MATADD(n{to_var}, l{e_var}_bs, {}, 1);\n", to.size());
        }

        let last_node = self.edge_list.last().unwrap().to();
        content +=
            &last_node.export_forward(&(*node_variables.get(&last_node).unwrap().to_string()));
        content += &format!(
            "{output}n{}[0];\n",
            *node_variables.get(&last_node).unwrap()
        );

        let footer = "\nend //end analog block\nendmodule\n";

        return format!("{}{}{}", header, content.replace("\n", "\n\t"), footer);
    }
}

#[test]
fn test_hermite_prod() {
    let x1 = &[1.0, 2.0, 4.0, 8.0];
    let x2 = &[1.0, 0.5, 0.25, 0.125];
    let y = &[1.0, 1.0, 1.0, 1.0];
    let x1 = Tensor::from_slice(x1);
    let x2 = Tensor::from_slice(x2);
    let y = Tensor::from_slice(y);
    assert_eq!(y, x1 * x2);
}

#[test]
fn test_add_edge() {
    use crate::components::utils::Activations;
    use tch::Kind;

    let mut g = Graph::new();
    let dummy_input = &["Dummy"];
    g.add_edge(Linear::new(
        NodeType::Input(InputNode::new(2, Activations::Id, "Vs", dummy_input)),
        NodeType::Hidden(HiddenNode::new(5, Activations::ReLU)),
        nn::linear(g.vs.root(), 2, 5, Default::default()),
    ));

    g.edge_list.first().unwrap().get_fun().ws.print();
    g.edge_list
        .first()
        .unwrap()
        .get_fun()
        .bs
        .as_ref()
        .unwrap()
        .print();

    let input = Tensor::stack(
        &[
            Tensor::from_slice(&[0.1, 0.2, 0.3, 0.4]),
            Tensor::from_slice(&[0.5, 1.0, 1.5, 2.0]),
        ],
        1,
    )
    .to_kind(Kind::Float);

    let mut inputs = HashMap::<&str, Tensor>::new();
    inputs.insert("Vs", input);
    g.forward(&inputs).print();
}

// #[test]
// fn test_train() {
//     use tch::Kind;

//     let mut g = Graph::new();
//     let n1 = Node::new(2, Activations::Id);
//     let n2 = Node::new(5, Activations::ReLU);
//     let n3 = Node::new(1, Activations::ReLU);
//     let l1 = nn::linear(g.vs.root(), 2, 5, Default::default());
//     let l2 = nn::linear(g.vs.root(), 5, 1, Default::default());
//     g.add_edge(Edge {
//         from: n1,
//         to: n2,
//         trans: l1,
//     });
//     g.add_edge(Edge {
//         from: n2,
//         to: n3,
//         trans: l2,
//     });

//     // y = (x1 + 2*x2) / 10
//     let x1 = &[0.1, 0.2, 0.3, 0.4];
//     let x2 = &[0.5, 1.0, 1.5, 2.0];
//     let xs = Tensor::stack(&[Tensor::from_slice(x1), Tensor::from_slice(x2)], 1)
//         .to_kind(Kind::Float)
//         .reshape([-1, 2]);
//     let y: Vec<f32> = x1
//         .iter()
//         .zip(x2)
//         .map(|(&x1, &x2)| (x1 + 2.0 * x2) / 10.0)
//         .collect();
//     let y = Tensor::stack(&[Tensor::from_slice(y.as_slice())], 1)
//         .to_kind(Kind::Float)
//         .reshape([-1, 1]);

//     let _ = g.train(&xs, &y, 10000, 1e-3);

//     println!(
//         "{}",
//         g.gen_verilog(
//             "analog begin\n/// input ///\n",
//             "/// output ///\nI(b_ds) <+ "
//         )
//     );
// }
