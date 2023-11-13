use tch::{
    nn::{self, LinearConfig, Module, OptimizerConfig},
    Device, Kind, Tensor,
};

#[test]
fn test_custom_loss() {
    let vs = nn::VarStore::new(Device::Cpu);
    let x = Tensor::from_slice2(&[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        .reshape([-1, 5])
        .to_kind(Kind::Float)
        .set_requires_grad(true);
    // y = sum(xi^2)
    let y = Tensor::from_slice2(&[[1 + 4 + 9 + 16 + 25], [36 + 49 + 64 + 81 + 100]])
        .reshape([-1, 1])
        .to_kind(Kind::Float)
        .set_requires_grad(true);
    let lin1 = nn::linear(&vs.root(), 5, 100, LinearConfig::default());
    let lin2 = nn::linear(&vs.root(), 100, 1, LinearConfig::default());

    let epoch = 10000;
    let lr = 1e-2;
    let mut opt = nn::AdamW::default().build(&vs, lr).unwrap();

    for _epoch in 1..=epoch {
        println!("epoch {}", _epoch);
        opt.zero_grad();
        let t = lin1.forward(&x).sigmoid();
        let t = lin2.forward(&t).relu();
        let loss = t.mse_loss(&y, tch::Reduction::Mean);
        println!("loss: {}", loss.double_value(&[]));
        loss.backward();
        {
            println!("x grad: ");
            x.grad().print();
            println!("");
        }
        opt.step();
        // opt.backward_step(&loss);
    }
}
