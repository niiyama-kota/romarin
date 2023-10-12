use crate::transpiler::utils::Activations;
use tch::{
    nn::{self, Module},
    Tensor,
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
    fn train(&self) {}

    fn add_edge(&mut self, e: edge) {
        todo!()
    }
}

impl Module for graph {
    fn forward(&self, inputs: Vec<f32>) -> Vec<f32> {
        let mut variables = Vec::<Vec<f32>>::new();
        variables.push(inputs.to_vec());
        for (i, e) in self.edge_list.iter().enumerate() {
            let t = &e.trans;
            let v: &Vec<f32> = &variables[i];
            // this forwarding apply transform W * v + B
            let tmp = t.forward(&Tensor::from_slice(v.as_slice()));
            let mut v = vec![0.0f32; tmp.numel()];
            tmp.copy_data(&mut v, tmp.numel());
            let v: Vec<f32> = v
                .iter()
                .map(|x| match e.from.act {
                    Activations::Sigmoid => todo!(),
                    Activations::Tanh => todo!(),
                    Activations::ReLU => todo!(),
                })
                .collect();

            variables.push(v);
        }

        return variables.last().unwrap().clone();
    }
}
