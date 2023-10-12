use anyhow::Result;

use crate::transpiler::utils::Activations;
use tch::{
    nn::{self, Module, OptimizerConfig},
    Tensor, Device,
};

#[derive(Clone, Copy, Debug)]
pub struct node {
    inputs: usize,
    outputs: usize,
    act: Activations,
}

#[derive(Debug)]
pub struct edge {
    from: node,
    to: node,
    trans: nn::Linear,
}

#[derive(Debug)]
pub struct graph {
    edge_list: Vec<edge>,
}

impl graph {
    fn train(&self, epoch: usize, x: Tensor, y: Tensor) -> Result<()> {
        let vs = nn::VarStore::new(Device::Cpu);
        let mut opt = nn::AdamW::default().build(&vs, 1e-5)?;
        let mut losses = Vec::<f64>::new();
        for _epoch in 1..=epoch {
            opt.zero_grad();
            let loss = self.forward(&x).mse_loss(&y, tch::Reduction::Mean);
            // loss.print();
            opt.backward_step(&loss);
            losses.push(loss.double_value(&[]));
            println!("epoch: {}", _epoch);
        }

        Ok(())
    }

    fn add_edge(&mut self, e: edge) {
        self.edge_list.push(e);
    }
}

impl Module for graph {
    fn forward(&self, inputs: &Tensor) -> Tensor {
        let mut v: Tensor = inputs.copy();
        if let Some(e) = self.edge_list.first() {
            match e.from.act {
                Activations::Sigmoid => &v.sigmoid(),
                Activations::Tanh => &v.tanh(),
                Activations::ReLU => &v.relu(),
            }
        } else {
            // meaning empty graph
            panic!();
        };

        for (i, e) in self.edge_list.iter().enumerate() {
            let t = &e.trans;
            // this forwarding apply transform W * v + B
            let v = t.forward(&v);
            let v = match e.to.act {
                Activations::Sigmoid => v.sigmoid(),
                Activations::Tanh => v.tanh(),
                Activations::ReLU => v.relu(),
            };
        }

        return v;
    }
}

#[test]
fn test_graph_function() {
    let mut g = graph {edge_list: Vec::<edge>::new()};
}
