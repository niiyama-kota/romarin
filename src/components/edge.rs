use tch::nn::{self, Module};

use super::{
    node::{Node, NodeType},
    utils::declare_linear,
};

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
        return self.trans.forward(xs);
    }
}

impl Edge for Linear {
    fn export_forward(&self, id: &str) -> String {
        let mut ret = format!(
            "`MATMUL({}, {}, l{}_ws, {}, {}, {});\n",
            self.from.name(),
            self.to.name(),
            id,
            self.to.size(),
            "1",
            self.from.size()
        );
        match self.trans.bs {
            Some(_) => {
                ret += &format!(
                    "`MATADD(n{}, l{}_bs, {}, 1);\n",
                    self.to.name(),
                    id,
                    self.to.size()
                );
            }
            None => ret += "// no bias are set\n",
        }

        return ret;
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
