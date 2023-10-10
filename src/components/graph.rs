use crate::transpiler::utils::Activations;

pub struct node {
    inputs: usize,
    outputs: usize,
    act: Activations,
}

pub struct edge {
    from: node,
    to: node,
    weight: Vec<Vec<f32>>,
}
