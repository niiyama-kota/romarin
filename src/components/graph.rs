use std::collections::{HashMap, HashSet};

use crate::components::node::*;
use crate::components::utils::{declare_matrix_mul, declare_matrix_mul_add};
use anyhow::Result;
use tch::Kind;
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
        xs: &HashMap<String, Tensor>,
        y: &HashMap<String, Tensor>,
        epoch: usize,
        lr: f64,
    ) -> Result<()> {
        let mut opt = nn::AdamW::default().build(&self.vs, lr)?;

        for _epoch in 1..=epoch {
            println!("epoch {}", _epoch);
            opt.zero_grad();
            let output_mp = self.forward(xs);
            for output_name in y.keys() {
                let loss = output_mp
                    .get(output_name)
                    .unwrap()
                    .mse_loss(y.get(output_name).unwrap(), tch::Reduction::Mean);
                println!("loss for {output_name}:");
                loss.print();
                opt.backward_step(&loss);
            }
        }

        Ok(())
    }

    pub fn add_edge(&mut self, e: Linear) {
        self.edge_list.push(e);
    }

    pub fn forward(&self, inputs: &HashMap<String, Tensor>) -> HashMap<String, Tensor> {
        let mut hidden_mp = HashMap::<NodeType, Tensor>::new();
        let mut output_mp = HashMap::<NodeType, Tensor>::new();

        for e in self.edge_list.iter() {
            let from = &e.from();
            let to = &e.to();
            let t = &e.get_fun();
            let v = match from {
                NodeType::Input(n) => inputs.get(n.name()).unwrap(),
                NodeType::Hidden(_) => {
                    match hidden_mp.get(from) {
                        Some(tensor) => tensor,
                        None => {
                            // FIXME: should return errors
                            panic!()
                        }
                    }
                }
                NodeType::Output(_) => {
                    // FIXME: should return errors
                    panic!()
                }
            };
            // apply activation function at from Node
            let v = from.forward(v).to_kind(Kind::Float);
            // apply transform function of edges
            let v = t.forward(&v).to_kind(Kind::Float);
            match to {
                NodeType::Input(_) => {
                    // FIXME: should return errors
                    panic!()
                }
                NodeType::Hidden(_) => {
                    match hidden_mp.get(to) {
                        Some(tensor) => {
                            assert_eq!(tensor.size(), v.size());
                            match to.get_acc() {
                                AccFn::Sum => {
                                    hidden_mp.insert(*to, tensor + v);
                                }
                                AccFn::Prod => {
                                    hidden_mp.insert(*to, tensor * v);
                                }
                                AccFn::Max => {
                                    assert!(false, "not implemented now");
                                    // NOTE: we may be able to use pooling layer.
                                    // hidden_mp.insert(*to, tensor.max_other(&v));
                                }
                                AccFn::Min => {
                                    assert!(false, "not implemented now");
                                    // NOTE: we may be able to use pooling layer.
                                    // hidden_mp.insert(*to, tensor.min_other(&v));
                                }
                            }
                        }
                        None => {
                            hidden_mp.insert(*to, v);
                        }
                    }
                }
                NodeType::Output(_) => {
                    match output_mp.get(to) {
                        Some(tensor) => {
                            assert_eq!(tensor.size(), v.size());
                            match to.get_acc() {
                                AccFn::Sum => {
                                    output_mp.insert(*to, tensor + v);
                                }
                                AccFn::Prod => {
                                    output_mp.insert(*to, tensor * v);
                                }
                                AccFn::Max => {
                                    assert!(false, "not implemented now");
                                    // output_mp.insert(*to, tensor.max_other(&v));
                                }
                                AccFn::Min => {
                                    assert!(false, "not implemented now");
                                    // output_mp.insert(*to, tensor.min_other(&v));
                                }
                            }
                        }
                        None => {
                            output_mp.insert(*to, v);
                        }
                    }
                }
            }
        }

        let mut ret = HashMap::<String, Tensor>::new();
        for k in output_mp.keys() {
            match k {
                NodeType::Input(_) => {
                    // FIXME: should return errors
                    panic!()
                }
                NodeType::Hidden(_) => {
                    // FIXME: should return errors
                    panic!()
                }
                NodeType::Output(n) => {
                    ret.insert(n.name().to_string(), n.forward(output_mp.get(k).unwrap()));
                }
            }
        }

        return ret;
    }

    pub fn gen_verilog(&self) -> String {
        let mut var: usize = 0;
        let mut fresh = || {
            var += 1;
            return var - 1;
        };
        let mut node_variables = HashMap::<NodeType, usize>::new();
        // assume no multiple edges
        let mut edge_variables = HashMap::<(NodeType, NodeType), usize>::new();

        let mut header = "`include \"disciplines.vams\"\n\n".to_owned();
        header += &declare_matrix_mul(AccFn::Sum);
        header += "\n";
        header += &declare_matrix_mul(AccFn::Prod);
        header += "\n";
        header += &declare_matrix_mul(AccFn::Max);
        header += "\n";
        header += &declare_matrix_mul(AccFn::Min);
        header += "\n";
        header += &declare_matrix_mul_add(AccFn::Sum);
        header += "\n";
        header += &declare_matrix_mul_add(AccFn::Prod);
        header += "\n";
        header += &declare_matrix_mul_add(AccFn::Max);
        header += "\n";
        header += &declare_matrix_mul_add(AccFn::Min);
        header += "\n";
        header += "module mosfet(term_G, term_D, term_S);\n\tinout term_G, term_D, term_S;\n\telectrical term_G, term_D, term_S;\n\tbranch (term_G, term_S) b_gs;\n\tbranch (term_G, term_D) b_gd;\n\tbranch (term_D, term_S) b_ds;\n\n\tinteger i, j, k;\n\treal tmp = 0.0;\n\n";

        let mut content = "\t".to_owned();

        // initialize variables for Edge transformation
        for e in self.edge_list.iter() {
            let evar = fresh();
            content += &e.export_params(&format!("l{}", evar));
            edge_variables.insert((e.from(), e.to()), evar);
        }
        // initialize variables for storing tensor at Node
        for e in self.edge_list.iter() {
            let from = e.from();
            if let Some(_) = node_variables.get(&from) {
                // already initialized
            } else {
                let var = fresh();
                match from {
                    NodeType::Input(n) => {
                        content += &n.export_init(n.name());
                    }
                    NodeType::Hidden(n) => {
                        content += &n.export_init(&format!("{}", n.name()));
                    }
                    NodeType::Output(_) => {
                        // FIXME: should return errors (OutputNode cannot be from Node)
                        panic!()
                    }
                }
                node_variables.insert(from, var);
            }

            let to = e.to();
            if let Some(_) = node_variables.get(&to) {
                // already initialized
            } else {
                let var = fresh();
                match to {
                    NodeType::Input(_) => {
                        // FIXME: should return errors(InputNode cannot be to Node)
                        panic!()
                    }
                    NodeType::Hidden(n) => {
                        content += &n.export_init(n.name());
                    }
                    NodeType::Output(n) => {
                        content += &n.export_init(n.name());
                    }
                }
                node_variables.insert(to, var);
            }
        }

        // implement forwarding process
        let mut node_activation = HashSet::<NodeType>::new();
        content += "analog begin\n";
        for e in self.edge_list.iter() {
            let from = e.from();
            let to = e.to();
            let edge = (from, to);
            let &e_var = edge_variables.get(&edge).unwrap();
            // if already activated from Node values, this operation should be removeds
            match from {
                NodeType::Input(n) => {
                    content += &n.export_input(n.name());
                }
                _ => {}
            }
            if !node_activation.contains(&from) {
                content += &from.export_forward();
                node_activation.insert(from);
            } else {
                content += &format!("// already activated this node: {}\n", from.name());
            }
            content += &e.export_forward(&format!("{e_var}"));
        }

        // implement assign output process
        let mut node_output = HashSet::<NodeType>::new();
        for e in self.edge_list.iter() {
            let to = e.to();
            match to {
                NodeType::Output(n) => {
                    // FIXME!: if already assign output, should do nothing
                    if !node_output.contains(&to) {
                        content += &n.export_forward();
                        content += &n.export_output();
                        node_output.insert(to);
                    } else {
                        content +=
                            &format!("// already assign output for this node: {}\n", to.name());
                    }
                }
                _ => {
                    // Do Nothing
                }
            }
        }

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
    use crate::components::node::AccFn;
    use crate::components::utils::Activations;
    use tch::Kind;

    let mut g = Graph::new();
    g.add_edge(Linear::new(
        NodeType::Input(InputNode::new(
            2,
            Activations::Id,
            AccFn::Sum,
            "Vs",
            &["Dummy"],
        )),
        NodeType::Hidden(HiddenNode::new(5, Activations::ReLU, AccFn::Sum, "1")),
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

    let mut inputs = HashMap::<String, Tensor>::new();
    inputs.insert("Vs".to_owned(), input);
    for outs in g.forward(&inputs) {
        println!("{}:", outs.0);
        outs.1.print();
    }
}
