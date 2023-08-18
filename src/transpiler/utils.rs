use tch::Tensor;

pub enum Activations {
    Sigmoid,
    Tanh,
    ReLU,
}

pub fn declare_tensor(ts: &Tensor, alias: &str) -> String {
    let mut ret: String = "".to_owned();
    ret += &format!("`define {} {{\\\n", alias);
    let mut raw_ts: Vec<f32> = vec![0.0; ts.numel()];
    ts.copy_data(&mut raw_ts, ts.numel());
    for (i, raw_t) in raw_ts.into_iter().enumerate() {
        ret += &format!("{raw_t},");
        if (i + 1) % (ts.size2().unwrap().1 as usize) == 0 {
            ret += "\\\n";
        }
    }
    ret += &format!("}}\n");

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
        Activations::Sigmoid => ret += &format!("\t\txs[i] = 1 / (1 + exp(-xs[i]));\\\n"),
        Activations::Tanh => ret += "\t\txs[i] = tanh(xs[i]);\\\n",
        Activations::ReLU => ret += "\t\txs[i] = (xs[i] + abs(xs[i])) / 2;\\\n",
    }
    ret += "\tend\n";

    return ret;
}

#[test]
fn test_declare_tensor() {
    use tch::Kind;

    let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .reshape(&[2, 3])
        .to_kind(Kind::Float);
    let expected: String = "`define sample_tensor {\\\n1,2,3,\\\n4,5,6,\\\n}\n".to_owned();

    assert_eq!(declare_tensor(&tensor, "sample_tensor"), expected);
}

#[test]
fn test_declare_activation() {
    let expected_sigmoid: String = "`define sigmoid(xs, x_dim)\\\n\tfor (i = 0; i < x_dim; i = i + 1) begin\\\n\t\txs[i] = 1 / (1 + exp(-xs[i]));\\\n\tend\n".to_owned();
    let expected_tanh: String = "`define tanh(xs, x_dim)\\\n\tfor (i = 0; i < x_dim; i = i + 1) begin\\\n\t\txs[i] = tanh(xs[i]);\\\n\tend\n".to_owned();
    let expected_relu: String = "`define relu(xs, x_dim)\\\n\tfor (i = 0; i < x_dim; i = i + 1) begin\\\n\t\txs[i] = (xs[i] + abs(xs[i])) / 2;\\\n\tend\n".to_owned();

    println!("{}", expected_tanh);
    assert_eq!(declare_activation(Activations::Sigmoid), expected_sigmoid);
    assert_eq!(declare_activation(Activations::Tanh), expected_tanh);
    assert_eq!(declare_activation(Activations::ReLU), expected_relu);
}
