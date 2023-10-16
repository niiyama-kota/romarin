use std::{collections::HashMap, hash::Hash};

use crate::transpiler::utils::{
    array_init, declare_linear, declare_matrix_add, declare_matrix_mul, Activations,
};
use anyhow::Result;
use tch::{
    nn::{self, Module, OptimizerConfig},
    Device, Kind, Tensor,
};

#[derive(Eq, PartialEq, Hash, Clone, Copy, Debug)]
pub struct Node {
    size: usize,
    act: Activations,
}

impl Node {
    fn new(_size: usize, _act: Activations) -> Self {
        Node {
            size: _size,
            act: _act,
        }
    }

    fn gen_verilog(&self, var: usize) -> String {
        let mut ret = "".to_owned();
        match self.act {
            Activations::Id => {
                ret += &format!("// applying Id to n{var}\n");
            }
            Activations::Sigmoid => {
                ret += &format!("for(integer i = 0; i < {}; i = i+1) begin\n\tn{var}[i] = n{var}[i] = 1 / (1 + exp(-n{var}[i]));\nend\n", self.size);
            }
            Activations::Tanh => {
                ret += &format!(
                    "for(integer i = 0; i < {}; i = i+1) begin\n\tn{var}[i] = tanh(n{var}[i]);\nend\n",
                    self.size
                );
            }
            Activations::ReLU => {
                ret += &format!("for(integer i = 0; i < {}; i = i+1) begin\n\tn{var}[i] = (n{var}[i] + abs(n{var}[i])) / 2;\nend\n", self.size);
            }
        }

        return ret;
    }

    fn size(&self) -> usize {
        return self.size;
    }
}

#[derive(Debug)]
pub struct Edge {
    from: Node,
    to: Node,
    trans: nn::Linear,
}

impl Edge {
    fn gen_verilog(&self) -> String {
        let mut ret = declare_linear(&self.trans, "sample");
        return ret;
    }
}

#[derive(Debug)]
pub struct Graph {
    edge_list: Vec<Edge>,
}

impl Graph {
    fn new() -> Self {
        Graph { edge_list: vec![] }
    }

    fn train(&self, vs: nn::VarStore, xs: &Tensor, y: &Tensor, epoch: usize) -> Result<()> {
        let mut opt = nn::AdamW::default().build(&vs, 1e-5)?;
        let mut losses = Vec::<f64>::new();

        for _epoch in 1..=epoch {
            opt.zero_grad();
            let loss = self.forward(xs).mse_loss(y, tch::Reduction::Mean);
            loss.print();
            opt.backward_step(&loss);
            losses.push(loss.double_value(&[]));
            println!("epoch {}", _epoch);
        }

        Ok(())
    }

    fn add_edge(&mut self, e: Edge) {
        self.edge_list.push(e);
    }

    fn gen_verilog(&self, input: &str, output: &str) -> String {
        let mut var: usize = 0;
        let mut fresh = || {
            var += 1;
            return var - 1;
        };
        let mut node_variables = HashMap::<Node, usize>::new();
        // assume no multiple edges
        let mut edge_variables = HashMap::<(Node, Node), usize>::new();

        let mut header = "".to_owned();
        header += &declare_matrix_mul();
        header += &declare_matrix_add();
        header += "`include \"disciplines.vams\"\n\nmodule mosfet(term_G, term_D, term_S);\n\tinout term_G, term_D, term_S;\n\telectrical term_G, termD, term_S;\n\n";

        let mut content = "\t".to_owned();

        for e in self.edge_list.iter() {
            let evar = fresh();
            content += &declare_linear(&e.trans, &format!("l{}", evar));
            edge_variables.insert((e.from, e.to), evar);
        }
        for e in self.edge_list.iter() {
            let from = e.from;
            if let Some(_) = node_variables.get(&from) {
            } else {
                let var = fresh();
                content += &format!("real n{}[0:{}] = {}", var, from.size, array_init(from.size));
                node_variables.insert(from, var);
            }

            let to = e.to;
            if let Some(_) = node_variables.get(&to) {
            } else {
                let var = fresh();
                content += &format!("real n{}[0:{}] = {}", var, to.size, array_init(to.size));
                node_variables.insert(to, var);
            }
        }

        // content += &format!("real n0[0:1] = {{V(b_DS), V(b_GS)}}");
        content += input;
        for e in self.edge_list.iter() {
            let from = e.from;
            let to = e.to;
            let edge = (from, to);
            let &from_var = node_variables.get(&from).unwrap();
            let &to_var = node_variables.get(&to).unwrap();
            let &e_var = edge_variables.get(&edge).unwrap();
            content += &from.gen_verilog(from_var);
            content += &format!(
                "`MATMULL(l{e_var}_ws, n{from_var}, n{to_var}, {}, 1, {});\n",
                from.size, to.size
            );
            content += &format!("`MATADD(n{to_var}, l{e_var}_bs, {}, 1);\n", to.size);
        }

        let footer = "\nendmodule\n";

        return format!("{}{}{}", header, content.replace("\n", "\n\t"), footer);
    }
}

impl Module for Graph {
    fn forward(&self, inputs: &Tensor) -> Tensor {
        let root = self.edge_list.first().unwrap().from;
        let mut mp = HashMap::<Node, Tensor>::new();
        mp.insert(root, inputs.copy());

        for e in self.edge_list.iter() {
            let from = &e.from;
            let to = &e.to;
            let t = &e.trans;
            let v = mp.get(from).unwrap();

            if let Some(acc) = mp.get(to) {
                let term = match from.act {
                    Activations::Id => t.forward(v),
                    Activations::Sigmoid => t.forward(&v.sigmoid()),
                    Activations::Tanh => t.forward(&v.tanh()),
                    Activations::ReLU => t.forward(&v.relu()),
                };
                let acc = acc.copy() + term;
                mp.insert(*to, acc);
            } else {
                mp.insert(
                    *to,
                    match from.act {
                        Activations::Id => t.forward(v),
                        Activations::Sigmoid => t.forward(&v.sigmoid()),
                        Activations::Tanh => t.forward(&v.tanh()),
                        Activations::ReLU => t.forward(&v.relu()),
                    },
                );
            }
        }

        let output_node = self.edge_list.last().unwrap().to;
        match output_node.act {
            Activations::Id => return mp.get(&output_node).unwrap().copy(),
            Activations::Sigmoid => return mp.get(&output_node).unwrap().sigmoid(),
            Activations::Tanh => return mp.get(&output_node).unwrap().tanh(),
            Activations::ReLU => return mp.get(&output_node).unwrap().relu(),
        };
    }
}

#[test]
fn test_add_edge() {
    let vs = nn::VarStore::new(Device::Cpu);

    let mut g = Graph::new();
    g.add_edge(Edge {
        from: Node::new(2, Activations::Id),
        to: Node::new(5, Activations::ReLU),
        trans: nn::linear(vs.root(), 2, 5, Default::default()),
    });

    g.edge_list.first().unwrap().trans.ws.print();
    g.edge_list
        .first()
        .unwrap()
        .trans
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
    g.forward(&input).print();
}

#[test]
fn test_train() {
    let vs = nn::VarStore::new(Device::Cpu);

    let mut g = Graph::new();
    let n1 = Node::new(2, Activations::Id);
    let n2 = Node::new(5, Activations::ReLU);
    let n3 = Node::new(1, Activations::ReLU);
    let l1 = nn::linear(vs.root(), 2, 5, Default::default());
    let l2 = nn::linear(vs.root(), 5, 1, Default::default());
    g.add_edge(Edge {
        from: n1,
        to: n2,
        trans: l1,
    });
    g.add_edge(Edge {
        from: n2,
        to: n3,
        trans: l2,
    });

    // y = (x1 + 2*x2) / 10
    let x1 = &[0.1, 0.2, 0.3, 0.4];
    let x2 = &[0.5, 1.0, 1.5, 2.0];
    let xs = Tensor::stack(&[Tensor::from_slice(x1), Tensor::from_slice(x2)], 1)
        .to_kind(Kind::Float)
        .reshape([-1, 2]);
    let y: Vec<f32> = x1
        .iter()
        .zip(x2)
        .map(|(&x1, &x2)| (x1 + 2.0 * x2) / 10.0)
        .collect();
    let y = Tensor::stack(&[Tensor::from_slice(y.as_slice())], 1)
        .to_kind(Kind::Float)
        .reshape([-1, 1]);

    let _ = g.train(vs, &xs, &y, 10000);

    println!("{}", g.gen_verilog("// input\n", "// output\n"));
}
