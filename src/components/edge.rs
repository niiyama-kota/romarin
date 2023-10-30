use tch::nn::{self, Module};

use super::{node::NodeType, utils::declare_linear};

pub trait Edge: Module {
    fn export_params(&self, id: &str) -> String;
    fn export_forward(&self, id: &str) -> String;
    fn from(&self) -> NodeType;
    fn to(&self) -> NodeType;
    fn get_fun(&self) -> &nn::Linear;
}

#[derive(Debug)]
pub struct Linear {
    from: NodeType,
    to: NodeType,
    trans: nn::Linear,
}

impl Linear {
    pub fn new(from: NodeType, to: NodeType, trans: nn::Linear) -> Self {
        Self {
            from: from,
            to: to,
            trans: trans,
        }
    }
}

impl Module for Linear {
    fn forward(&self, xs: &tch::Tensor) -> tch::Tensor {
        return self.forward(xs);
    }
}

impl Edge for Linear {
    fn export_forward(&self, id: &str) -> String {
        todo!()
    }
    fn export_params(&self, id: &str) -> String {
        return declare_linear(&self.trans, id);
    }

    fn from(&self) -> NodeType {
        return self.from;
    }

    fn to(&self) -> NodeType {
        return self.to;
    }

    fn get_fun(&self) -> &nn::Linear {
        return &self.trans;
    }
}
