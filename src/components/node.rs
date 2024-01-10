use std::hash::Hash;
use tch::{nn::Module, Tensor};

use super::utils::{array_init, Activations};

pub trait Node: Module + PartialEq + Eq + Hash + Clone + Copy {
    fn size(&self) -> usize;
    fn export_init(&self, id: &str) -> String;
    fn export_forward(&self) -> String;
    fn get_fun(&self) -> Activations;
    fn get_acc(&self) -> AccFn;
    fn name(&self) -> &str;
}

// use Enum Wrapper Pattern
#[derive(Eq, PartialEq, Hash, Clone, Copy, Debug)]
pub enum NodeType {
    Input(InputNode),
    Hidden(HiddenNode),
    Output(OutputNode),
}

#[derive(Eq, PartialEq, Hash, Clone, Copy, Debug)]
pub enum AccFn {
    Sum,
    Prod,
    Max,
    Min,
}

#[derive(Eq, PartialEq, Hash, Clone, Copy, Debug)]
pub struct InputNode {
    size: usize,
    act: Activations,
    acc: AccFn,
    name: &'static str,
    verilog_inputs: &'static [&'static str],
}

#[derive(Eq, PartialEq, Hash, Clone, Copy, Debug)]
pub struct HiddenNode {
    size: usize,
    act: Activations,
    acc: AccFn,
    name: &'static str,
}

#[derive(Eq, PartialEq, Hash, Clone, Copy, Debug)]
pub struct OutputNode {
    size: usize,
    act: Activations,
    acc: AccFn,
    name: &'static str,
    verilog_outputs: &'static [&'static str],
}

impl AccFn {
    pub fn to_string(&self) -> String {
        match self {
            AccFn::Sum => "SUM".to_owned(),
            AccFn::Prod => "PROD".to_owned(),
            AccFn::Max => "MAX".to_owned(),
            AccFn::Min => "MIN".to_owned(),
        }
    }
}

impl Module for NodeType {
    fn forward(&self, xs: &Tensor) -> Tensor {
        match self {
            NodeType::Input(n) => n.forward(xs),
            NodeType::Hidden(n) => n.forward(xs),
            NodeType::Output(n) => n.forward(xs),
        }
    }
}

impl Node for NodeType {
    fn size(&self) -> usize {
        match self {
            NodeType::Input(n) => n.size(),
            NodeType::Hidden(n) => n.size(),
            NodeType::Output(n) => n.size(),
        }
    }

    fn export_init(&self, id: &str) -> String {
        match self {
            NodeType::Input(n) => n.export_init(id),
            NodeType::Hidden(n) => n.export_init(id),
            NodeType::Output(n) => n.export_init(id),
        }
    }

    fn export_forward(&self) -> String {
        match self {
            NodeType::Input(n) => n.export_forward(),
            NodeType::Hidden(n) => n.export_forward(),
            NodeType::Output(n) => n.export_forward(),
        }
    }

    fn get_fun(&self) -> Activations {
        match self {
            NodeType::Input(n) => n.get_fun(),
            NodeType::Hidden(n) => n.get_fun(),
            NodeType::Output(n) => n.get_fun(),
        }
    }

    fn get_acc(&self) -> AccFn {
        match self {
            NodeType::Input(n) => n.get_acc(),
            NodeType::Hidden(n) => n.get_acc(),
            NodeType::Output(n) => n.get_acc(),
        }
    }

    fn name(&self) -> &str {
        match self {
            NodeType::Input(n) => n.name(),
            NodeType::Hidden(n) => n.name(),
            NodeType::Output(n) => n.name(),
        }
    }
}

impl InputNode {
    pub fn new(
        _size: usize,
        _act: Activations,
        _acc: AccFn,
        _name: &'static str,
        _verilog_inputs: &'static [&str],
    ) -> Self {
        InputNode {
            size: _size,
            act: _act,
            acc: _acc,
            name: _name,
            verilog_inputs: _verilog_inputs,
        }
    }

    pub fn export_input(&self, input_var: &str) -> String {
        let mut ret = "".to_owned();
        for (idx, input) in self.verilog_inputs().iter().enumerate() {
            ret += &format!("{}[{}] = ({});\n", input_var, idx, input);
        }

        return ret;
    }

    pub fn name(&self) -> &str {
        return self.name;
    }

    pub fn verilog_inputs(&self) -> &[&str] {
        return self.verilog_inputs;
    }
}

impl Module for InputNode {
    fn forward(&self, xs: &Tensor) -> Tensor {
        match self.act {
            Activations::Id => xs.copy(),
            Activations::Scale(factor) => xs / factor as f64,
            Activations::Sigmoid => xs.sigmoid(),
            Activations::Tanh => xs.tanh(),
            Activations::ReLU => xs.relu(),
            Activations::LeakyReLU => xs.leaky_relu(),
        }
    }
}

impl Node for InputNode {
    fn size(&self) -> usize {
        return self.size;
    }

    fn export_init(&self, id: &str) -> String {
        let mut ret = "".to_owned();
        ret += &format!("real {}[0:{}] = ", id, self.size() - 1,);
        // initialize identity of node's op
        match self.get_acc() {
            AccFn::Sum => {
                ret += &array_init(self.size(), 0.0);
            }
            AccFn::Prod => {
                ret += &array_init(self.size(), 1.0);
            }
            AccFn::Max => {
                ret += &array_init(self.size(), f32::MIN);
            }
            AccFn::Min => {
                ret += &array_init(self.size(), f32::MAX);
            }
        }
        ret += ";\n";

        return ret;
    }

    fn export_forward(&self) -> String {
        let ret = self.act.export_apply(self.name(), self.size());

        return ret;
    }

    fn get_fun(&self) -> Activations {
        return self.act;
    }

    fn get_acc(&self) -> AccFn {
        return self.acc;
    }

    fn name(&self) -> &str {
        return self.name;
    }
}

impl HiddenNode {
    pub fn new(_size: usize, _act: Activations, _acc: AccFn, _name: &'static str) -> Self {
        HiddenNode {
            size: _size,
            act: _act,
            acc: _acc,
            name: _name,
        }
    }
}

impl Module for HiddenNode {
    fn forward(&self, xs: &Tensor) -> Tensor {
        match self.act {
            Activations::Id => xs.copy(),
            Activations::Scale(factor) => xs / factor as f64,
            Activations::Sigmoid => xs.sigmoid(),
            Activations::Tanh => xs.tanh(),
            Activations::ReLU => xs.relu(),
            Activations::LeakyReLU => xs.leaky_relu(),
        }
    }
}

impl Node for HiddenNode {
    fn size(&self) -> usize {
        return self.size;
    }

    fn export_init(&self, id: &str) -> String {
        let mut ret = "".to_owned();
        ret += &format!("real {}[0:{}] = ", id, self.size() - 1,);
        // initialize identity of node's op
        match self.get_acc() {
            AccFn::Sum => {
                ret += &array_init(self.size(), 0.0);
            }
            AccFn::Prod => {
                ret += &array_init(self.size(), 1.0);
            }
            AccFn::Max => {
                ret += &array_init(self.size(), f32::MIN);
            }
            AccFn::Min => {
                ret += &array_init(self.size(), f32::MAX);
            }
        }
        ret += ";\n";

        return ret;
    }

    fn export_forward(&self) -> String {
        let ret = self.act.export_apply(self.name(), self.size());

        return ret;
    }

    fn get_fun(&self) -> Activations {
        return self.act;
    }

    fn get_acc(&self) -> AccFn {
        return self.acc;
    }

    fn name(&self) -> &str {
        return self.name;
    }
}

impl OutputNode {
    pub fn new(
        _size: usize,
        _act: Activations,
        _acc: AccFn,
        _name: &'static str,
        _verilog_outputs: &'static [&str],
    ) -> Self {
        OutputNode {
            size: _size,
            act: _act,
            acc: _acc,
            name: _name,
            verilog_outputs: _verilog_outputs,
        }
    }

    pub fn export_output(&self) -> String {
        let mut ret = "".to_owned();
        for (i, out) in self.verilog_outputs().iter().enumerate() {
            ret += &format!("{} <+ {}[{}];\n", out, self.name(), i);
        }

        return ret;
    }

    pub fn name(&self) -> &str {
        return self.name;
    }

    pub fn verilog_outputs(&self) -> &[&str] {
        return self.verilog_outputs;
    }
}

impl Module for OutputNode {
    fn forward(&self, xs: &Tensor) -> Tensor {
        match self.act {
            Activations::Id => xs.copy(),
            Activations::Scale(factor) => xs / factor as f64,
            Activations::Sigmoid => xs.sigmoid(),
            Activations::Tanh => xs.tanh(),
            Activations::ReLU => xs.relu(),
            Activations::LeakyReLU => xs.leaky_relu(),
        }
    }
}

impl Node for OutputNode {
    fn size(&self) -> usize {
        return self.size;
    }

    fn export_init(&self, id: &str) -> String {
        let mut ret = "".to_owned();
        ret += &format!("real {}[0:{}] = ", id, self.size() - 1,);
        // initialize identity of node's op
        match self.get_acc() {
            AccFn::Sum => {
                ret += &array_init(self.size(), 0.0);
            }
            AccFn::Prod => {
                ret += &array_init(self.size(), 1.0);
            }
            AccFn::Max => {
                ret += &array_init(self.size(), f32::MIN);
            }
            AccFn::Min => {
                ret += &array_init(self.size(), f32::MAX);
            }
        }
        ret += ";\n";

        return ret;
    }

    fn export_forward(&self) -> String {
        let ret = self.act.export_apply(self.name(), self.size());

        return ret;
    }

    fn get_fun(&self) -> Activations {
        return self.act;
    }

    fn get_acc(&self) -> AccFn {
        return self.acc;
    }

    fn name(&self) -> &str {
        return self.name();
    }
}

#[test]
fn test_double_val() {
    let t = Tensor::from_slice(&[5]);
    println!("{:?}", t.size());
}
