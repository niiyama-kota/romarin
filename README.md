# romarin

romarinはニューラルネットワークによるトランジスタモデリングを援用するためのライブラリである．
ニューラルネットワークを計算グラフとして定義し，tch-rsというPyTorchのRust版バインディングを呼び出して訓練を行う．
訓練済みモデルを`Verilog-A`コードとして出力することで回路シミュレーションへの組み込みを可能とする．

## 頂点
romarinにおける頂点は`Trait`として定義される．
```
pub trait Node: Module + PartialEq + Eq + Hash + Clone + Copy {
    fn size(&self) -> usize;
    fn export_init(&self, id: &str) -> String;
    fn export_forward(&self) -> String;
    fn get_fun(&self) -> Activations;
    fn get_acc(&self) -> AccFn;
    fn name(&self) -> &str;
}
```

`
Node := Input(InputNode::new(<size>, <activation>, <acc>, <name>, <v_input>)) | Hidden(HiddenNode::new(<size>, <activation>, <acc>, <name>)) | Output(OutputNode::new(<size>, <activation>, <acc>, <name>,<v_output>))
`

`activation := Id | Scale(f32) | Sigmoid | Tanh | ReLU | LeakyReLU`

`acc := Sum | Prod | Min | Max`

## 辺
romarinにおける辺は`Trait`として定義される．
```
pub trait Edge: Module {
    fn export_params(&self, id: &str) -> String;
    fn export_forward(&self, id: &str) -> String;
    fn from(&self) -> NodeType;
    fn to(&self) -> NodeType;
    fn get_fun(&self) -> &nn::Linear;
}
```

辺の宣言
`
Edge := <EdgeType>::new(<Node>, <Node>, <Trans>);
`

`EdgeType := Linear`

`Trans := tch::nn::Linear`

現在は全結合層のみを辺として想定している．内部実装ではtchの全結合層が用いられる．

## 計算グラフ
romarinにおける計算グラフ`Graph`は`struct`として定義される．
この`struct`は内部に`edge_list`という辺を格納するデータ構造をもつ．

## 利用例
簡単な二層の全結合層のみからなるニューラルネットワークを定義する例を示す．

```
use tch::nn::{self, LinearConfig, OptimizerConfig};


let mut vs = nn::VarStore::new(tch::Device::CPU);
let mut net = Graph::new();
let input: NodeType = NodeType::Input(InputNode::new(
  2,
  Id,
  Sum,
  "input",
  &["V(b_gs)", "V(b_ds)"]
);
let hidden: NodeType = NodeType::Hidden(HiddenNode::new(
  100,
  ReLU,
  Sum,
  "hidden"
);
let output = : NodeType = NodeType::Output(OutputNode::new(
  1,
  Id,
  Sum,
  "output",
  &["I(b_ds)"]
);

let l1 = Linear::new(input, hidden, nn::Linear(vs.root(), 2, 100, LinearConfig::default());
let l2 = Linear::new(hidden, output, nn::Linear(vs.root(), 100, 1, LinearConfig::default());

net.add_edge(l1);
net.add_edge(l2);
```

定義したネットワークは`forward(&HashMap::<String, Tensor>) -> HashMap<String, Tensor>`により推論が行われる．
引数の`HashMap`には`InputNode`の名前をkeyとして，その`Tensor`をvalueとするエントリが含まれている．出力の`HashMap`には`OutputNode`の名前をkeyとして，その`Tensor`をvalueとするエントリが含まれている．

上で定義したネットワークには測定データなどから入力として，`HashMap { "input": Tensor([[vg, vd], ...., ]) }`が与えられて，`HashMap{ "output": Tensor([[id], ...., ]) }`が得られる．
この出力と測定データの実測値との誤差を損失関数として逆伝搬することでモデルの訓練が可能である．

```
// define input and output tensor
let measured: Tensor = ...;
let input_mp: HashMap<String, Tensor> = ....;
let mut opt: Optimizer = ....;
// training
for epoch in 1..= EPOCH {
  let pred = net.forward(&input_mp).get("input").unwrap();
  let loss = pred.mse_loss(measured, tch::Reduction::Mean);
  opt.backward_step(&loss);
}

// writeout  verilog-A code
println!("{}", net.gen_verilog());
```
