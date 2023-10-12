use tch::Tensor;

#[derive(Clone, Copy, Debug)]
pub enum Activations {
    Sigmoid,
    Tanh,
    ReLU,
}

pub fn declare_tensor(ts: &Tensor, alias: &str, break_line_num: Option<usize>) -> String {
    let mut ret: String = "".to_owned();
    ret += &format!("real {}[0:{}] = {{\\\n", alias, ts.numel() - 1);
    let mut raw_ts: Vec<f32> = vec![0.0; ts.numel()];
    ts.copy_data(&mut raw_ts, ts.numel());
    for (i, raw_t) in raw_ts.into_iter().enumerate() {
        ret += &format!("{raw_t},");
        if (i + 1) % break_line_num.unwrap_or(1) == 0 {
            ret += "\\\n";
        }
    }
    ret += "};\n";

    return ret;
}

pub fn declare_activation(activation_kind: Activations) -> String {
    let mut ret = "".to_owned();
    match activation_kind {
        Activations::Sigmoid => ret += "`define sigmoid(xs, x_dim)\\\n",
        Activations::Tanh => ret += "`define tanh(xs, x_dim)\\\n",
        Activations::ReLU => ret += "`define relu(xs, x_dim)\\\n",
    }
    ret += "\tfor (i = 0; i < x_dim; i = i + 1) begin\\\n";
    match activation_kind {
        Activations::Sigmoid => ret += "\t\txs[i] = 1 / (1 + exp(-xs[i]));\\\n",
        Activations::Tanh => ret += "\t\txs[i] = tanh(xs[i]);\\\n",
        Activations::ReLU => ret += "\t\txs[i] = (xs[i] + abs(xs[i])) / 2;\\\n",
    }
    ret += "\tend\n";

    return ret;
}

pub fn declare_matrix_mul() -> String {
    let mut ret = "".to_owned();
    ret += "`ifndef MATMUL\\\n";
    ret += "`define MATMUL(A, B, C, C_dim1, C_dim2, K)\\\n";
    ret += "\tfor (i = 0; i < C_dim1; i = i + 1) begin\\\n";
    ret += "\t\tfor (j = 0; j < C_dim2; j = j + 1) begin\\\n";
    ret += "\t\t\tfor (k = 0; k < K; k = k + 1) begin\\\n";
    ret += "\t\t\t\tC[i*C_dim2 + j] = C[i*C_dim2 + j] + A[i*K + k]*B[k*C_dim2 + j];\\\n";
    ret += "\t\t\tend\\\n";
    ret += "\t\tend\\\n";
    ret += "\tend\\\n";

    return ret;
}

pub fn declare_matrix_add() -> String {
    let mut ret = "".to_owned();
    ret += "`ifndef MATADD\\\n";
    ret += "`define MATADD(A, B, dim1, dim2)\\\n";
    ret += "\tfor (i = 0; i < dim1; i = i + 1) begin\\\n";
    ret += "\t\tfor (j = 0; j < dim2; j = j + 1) begin\\\n";
    ret += "\t\t\tA[i*dim2 + j] = A[i*dim2 + j] + B[i*dim2 + j];\\\n";
    ret += "\t\tend\\\n";
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
    // ret += "\tVgs = V(b_gs);\n";
    // ret += "\tVds = V(b_ds);\n";
    // ret += "\tVgd = V(b_gd);\n";
    ret += analog_behavior;
    ret += "\tend // analog block\n";

    ret += "endmodule // mosfet\n";

    return ret;
}

pub fn array_init(size: usize) -> String {
    let mut ret = "".to_owned();
    ret += "{";
    for _ in 0..size {
        ret += "0, ";
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
    let expected: String = "`define sample_tensor {\\\n1,2,3,\\\n4,5,6,\\\n}\n".to_owned();

    assert_eq!(
        declare_tensor(
            &tensor,
            "sample_tensor",
            Some(tensor.size2().unwrap().1 as usize)
        ),
        expected
    );
}

#[test]
fn test_declare_activation() {
    let expected_sigmoid: String = "`define sigmoid(xs, x_dim)\\\n\tfor (i = 0; i < x_dim; i = i + 1) begin\\\n\t\txs[i] = 1 / (1 + exp(-xs[i]));\\\n\tend\n".to_owned();
    let expected_tanh: String = "`define tanh(xs, x_dim)\\\n\tfor (i = 0; i < x_dim; i = i + 1) begin\\\n\t\txs[i] = tanh(xs[i]);\\\n\tend\n".to_owned();
    let expected_relu: String = "`define relu(xs, x_dim)\\\n\tfor (i = 0; i < x_dim; i = i + 1) begin\\\n\t\txs[i] = (xs[i] + abs(xs[i])) / 2;\\\n\tend\n".to_owned();

    assert_eq!(declare_activation(Activations::Sigmoid), expected_sigmoid);
    assert_eq!(declare_activation(Activations::Tanh), expected_tanh);
    assert_eq!(declare_activation(Activations::ReLU), expected_relu);
}

#[test]
fn test_declare_matrix_mul() {
    let expected: String = "`ifndef MATMUL\\\n`define MATMUL(A, B, C, C_dim1, C_dim2, K)\\\n\tfor (i = 0; i < C_dim1; i = i + 1) begin\\\n\t\tfor (j = 0; j < C_dim2; j = j + 1) begin\\\n\t\t\tfor (k = 0; k < K; k = k + 1) begin\\\n\t\t\t\tC[i*C_dim2 + j] = C[i*C_dim2 + j] + A[i*K + k]*B[k*C_dim2 + j];\\\n\t\t\tend\\\n\t\tend\\\n\tend\\\n".to_owned();

    assert_eq!(declare_matrix_mul(), expected);
}

#[test]
fn test_declare_matrix_add() {
    let expected: String = "`ifndef MATADD\\\n`define MATADD(A, B, dim1, dim2)\\\n\tfor (i = 0; i < dim1; i = i + 1) begin\\\n\t\tfor (j = 0; j < dim2; j = j + 1) begin\\\n\t\t\tA[i*dim2 + j] = A[i*dim2 + j] + B[i*dim2 + j];\\\n\t\tend\\\n\tend\\\n".to_owned();

    assert_eq!(declare_matrix_add(), expected);
}

#[test]
fn test_array_init() {
    let expected = "{0, 0, 0}";
    assert_eq!(array_init(3), expected);
}
