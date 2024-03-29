use std::hash::Hash;
use tch::{nn, Tensor};

use super::node::AccFn;

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum Activations {
    Id,
    Scale(f32),
    Sigmoid,
    Tanh,
    ReLU,
    LeakyReLU,
}

impl Eq for Activations {}
impl Hash for Activations {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
    }
}

impl Activations {
    pub fn export_apply(&self, id: &str, size: usize) -> String {
        let mut ret = "".to_owned();
        match self {
            Activations::Id => {
                ret += &format!("/// applying Id to {id} ///\n");
            }
            Activations::Scale(factor) => {
                ret += &format!(
                    "for(i = 0; i < {}; i = i+1) begin\n\t{id}[i] = {id}[i] / ({});\nend // end for\n",
                    size, factor
                );
            }
            Activations::Sigmoid => {
                ret += &format!("for(i = 0; i < {}; i = i+1) begin\n\t{id}[i] = 1 / (1 + exp(-{id}[i]));\nend // end for\n", size);
            }
            Activations::Tanh => {
                ret += &format!(
                    "for(i = 0; i < {}; i = i+1) begin\n\t{id}[i] = tanh({id}[i]);\nend // end for\n",
                    size
                );
            }
            Activations::ReLU => {
                ret += &format!("for(i = 0; i < {}; i = i+1) begin\n\t{id}[i] = ({id}[i] + abs({id}[i])) / 2;\nend// end for\n", size);
            }
            Activations::LeakyReLU => {
                ret += &format!("for(i = 0; i < {}; i = i+1) begin\n\tif({id}[i] < 0) begin\n\t\t{id}[i] = 0.1*{id}[i];\n\tend // end if\nend// end for\n", size);
            }
        }

        return ret;
    }
}

pub fn tensor2varray(ts: &Tensor, tab: usize) -> String {
    let mut ret = "".to_owned();
    if ts.size().len() == 0 {
        ret += &format!("{}", ts.double_value(&[]));
    } else {
        if tab > 0 {
            ret += "\n";
        }
        ret += &"\t".repeat(tab);
        ret += "{";
        for i in 0..ts.size()[0] {
            ret += &tensor2varray(&ts.get(i), tab + 1);
            ret += ", ";
        }
        if tab == 0 {
            ret += "\n";
        }
        ret += "}";
    }

    return ret;
}

pub fn declare_tensor(ts: &Tensor, alias: &str) -> String {
    let dims = ts.size();
    let mut ret: String = format!("real {}", alias);
    for dim in dims {
        ret += &format!("[0:{}-1]", dim);
    }
    ret += " = ";
    ret += &tensor2varray(ts, Default::default());
    ret += ";\n";

    return ret;
}

pub fn declare_linear(linear: &nn::Linear, alias: &str) -> String {
    let mut ret = "".to_owned();

    ret += &declare_tensor(&linear.ws, &format!("{}{}", alias, "_ws"));
    if let Some(bs) = &linear.bs {
        ret += &declare_tensor(bs, &format!("{}{}", alias, "_bs"));
    }

    return ret;
}

pub fn declare_matrix_mul(acc_kind: AccFn) -> String {
    let mut ret = "".to_owned();
    ret += &format!("`ifndef MATMUL{}\\\n", (acc_kind as AccFn).to_string());
    // H = Wx + b
    ret += &format!(
        "`define MATMUL{}(W, x, H, x_dim, H_dim)\\\n",
        (acc_kind as AccFn).to_string()
    );
    ret += "\tfor (i = 0; i < H_dim; i = i + 1) begin\\\n";
    ret += "\t\ttmp = 0.0;\\\n";
    ret += "\t\tfor (j = 0; j < x_dim; j = j + 1) begin\\\n";
    ret += "\t\t\ttmp = tmp + W[i][j]*x[j];\\\n";
    ret += "\t\tend\\\n";
    match acc_kind {
        AccFn::Sum => {
            ret += "\t\tH[i] = H[i] + tmp;\\\n";
        }
        AccFn::Prod => {
            ret += "\t\tH[i] = H[i] * tmp;\\\n";
        }
        AccFn::Max => {
            ret += "\t\tH[i] = max(H[i], tmp);\\\n";
        }
        AccFn::Min => {
            ret += "\t\tH[i] = min(H[i], tmp);\\\n";
        }
    }
    ret += "\tend\\\n";

    return ret;
}

pub fn declare_matrix_mul_add(acc_kind: AccFn) -> String {
    let mut ret = "".to_owned();
    ret += &format!("`ifndef MATMULADD{}\\\n", (acc_kind as AccFn).to_string());
    // H = Wx + b
    ret += &format!(
        "`define MATMULADD{}(W, x, B, H, x_dim, H_dim)\\\n",
        (acc_kind as AccFn).to_string()
    );
    ret += "\tfor (i = 0; i < H_dim; i = i + 1) begin\\\n";
    ret += "\t\ttmp = 0.0;\\\n";
    ret += "\t\tfor (j = 0; j < x_dim; j = j + 1) begin\\\n";
    ret += "\t\t\ttmp = tmp + W[i][j]*x[j];\\\n";
    ret += "\t\tend\\\n";
    ret += "\t\ttmp = tmp + B[i];\\\n";
    match acc_kind {
        AccFn::Sum => {
            ret += "\t\t\tH[i] = H[i] + tmp;\\\n";
        }
        AccFn::Prod => {
            ret += "\t\t\tH[i] = H[i] * tmp;\\\n";
        }
        AccFn::Max => {
            ret += "\t\t\tH[i] = max(H[i], tmp);\\\n";
        }
        AccFn::Min => {
            ret += "\t\t\tH[i] = min(H[i], tmp);\\\n";
        }
    }
    ret += "\tend\\\n";

    return ret;
}

pub fn mosfet_template(header: &str, analog_behavior: &str) -> String {
    let mut ret = "".to_owned();
    ret += "module mosfet(term_G, term_D, term_S);\n";
    ret += "\tinout term_G, term_D, term_S;\n";
    ret += "\telectrical term_G, term_D, term_S;\n";
    ret += "\n";
    ret += "\tbranch (term_G, term_S) b_gs;\n";
    ret += "\tbranch (term_G, term_D) b_gd;\n";
    ret += "\tbranch (term_D, term_S) b_ds;\n";

    ret += header;

    ret += "// define analog behavior\n";
    ret += "\tanalog begin\n";
    ret += analog_behavior;
    ret += "\tend // analog block\n";

    ret += "endmodule // mosfet\n";

    return ret;
}

pub fn array_init(size: usize, id: f32) -> String {
    let mut ret = "".to_owned();
    ret += "{";
    for _ in 0..size {
        ret += &format!("{id}, ");
    }
    ret += "}";
    ret = ret.replace(", }", "}");

    return ret;
}

#[test]
fn test_declare_tensor() {
    use tch::Kind;

    let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .reshape(&[2, 3])
        .to_kind(Kind::Float);
    let expected: String =
        "real sample_tensor[0:2-1][0:3-1] = {\n\t{1, 2, 3, }, \n\t{4, 5, 6, }, \n};\n".to_owned();

    assert_eq!(declare_tensor(&tensor, "sample_tensor",), expected);
}

#[test]
fn test_array_init() {
    let expected = "{0, 0, 0}";
    assert_eq!(array_init(3, 0.0), expected);
}
