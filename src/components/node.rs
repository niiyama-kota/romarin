use std::hash::Hash;
use tch::{nn::Module, Tensor};

use super::utils::{array_init, Activations};

pub trait Node: Module + Eq + PartialEq + Hash + Clone + Copy {
    fn size(&self) -> usize;
    fn export_init(&self, id: &str) -> String;
    fn export_forward(&self) -> String;
    fn get_fun(&self) -> Activations;
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
pub struct InputNode {
    size: usize,
    act: Activations,
    name: &'static str,
    verilog_inputs: &'static [&'static str],
}

#[derive(Eq, PartialEq, Hash, Clone, Copy, Debug)]
pub struct HiddenNode {
    size: usize,
    act: Activations,
    name: &'static str,
}

#[derive(Eq, PartialEq, Hash, Clone, Copy, Debug)]
pub struct OutputNode {
    size: usize,
    act: Activations,
    name: &'static str,
    verilog_outputs: &'static [&'static str],
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
        _name: &'static str,
        _verilog_inputs: &'static [&str],
    ) -> Self {
        InputNode {
            size: _size,
            act: _act,
            name: _name,
            verilog_inputs: _verilog_inputs,
        }
    }

    pub fn export_input(&self, input_var: Vec<&str>) -> String {
        assert_eq!(self.size(), input_var.len());
        return format!(
            "real {}[0:{}] = {{{}}};\n",
            self.name(),
            self.size() - 1,
            &input_var
                .into_iter()
                .fold("".to_owned(), |acc, x| -> String {
                    if acc == "" {
                        acc + x
                    } else {
                        acc + ", " + x
                    }
                })
        );
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
            Activations::Sigmoid => xs.sigmoid(),
            Activations::Tanh => xs.tanh(),
            Activations::ReLU => xs.relu(),
        }
    }
}

impl Node for InputNode {
    fn size(&self) -> usize {
        return self.size;
    }

    fn export_init(&self, id: &str) -> String {
        let mut ret = "".to_owned();
        ret += &format!(
            "real n{}[0:{}] = {};\n",
            id,
            self.size(),
            array_init(self.size(), 1.0) // initialize identity of node's op
        );

        return ret;
    }

    fn export_forward(&self) -> String {
        let ret = self.act.export_apply(self.name(), self.size());

        return ret;
    }

    fn get_fun(&self) -> Activations {
        return self.act;
    }

    fn name(&self) -> &str {
        return self.name;
    }
}

impl HiddenNode {
    pub fn new(_size: usize, _act: Activations, _name: &'static str) -> Self {
        HiddenNode {
            size: _size,
            act: _act,
            name: _name,
        }
    }
}

impl Module for HiddenNode {
    fn forward(&self, xs: &Tensor) -> Tensor {
        match self.act {
            Activations::Id => xs.copy(),
            Activations::Sigmoid => xs.sigmoid(),
            Activations::Tanh => xs.tanh(),
            Activations::ReLU => xs.relu(),
        }
    }
}

impl Node for HiddenNode {
    fn size(&self) -> usize {
        return self.size;
    }

    fn export_init(&self, id: &str) -> String {
        let mut ret = "".to_owned();
        ret += &format!(
            "real {}[0:{}] = {};\n",
            id,
            self.size(),
            array_init(self.size(), 1.0) // initialize identity of node's op
        );

        return ret;
    }

    fn export_forward(&self) -> String {
        let ret = self.act.export_apply(self.name(), self.size());

        return ret;
    }

    fn get_fun(&self) -> Activations {
        return self.act;
    }

    fn name(&self) -> &str {
        return self.name;
    }
}

impl OutputNode {
    pub fn new(
        _size: usize,
        _act: Activations,
        _name: &'static str,
        _verilog_outputs: &'static [&str],
    ) -> Self {
        OutputNode {
            size: _size,
            act: _act,
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
            Activations::Sigmoid => xs.sigmoid(),
            Activations::Tanh => xs.tanh(),
            Activations::ReLU => xs.relu(),
        }
    }
}

impl Node for OutputNode {
    fn size(&self) -> usize {
        return self.size;
    }

    fn export_init(&self, id: &str) -> String {
        let mut ret = "".to_owned();
        ret += &format!(
            "real {}[0:{}] = {};\n",
            id,
            self.size(),
            array_init(self.size(), 1.0) // initialize identity of node's op
        );

        return ret;
    }

    fn export_forward(&self) -> String {
        let ret = self.act.export_apply(self.name(), self.size());

        return ret;
    }

    fn get_fun(&self) -> Activations {
        return self.act;
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
