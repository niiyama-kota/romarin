use tch::{
    nn::{self, OptimizerConfig},
    Device, Kind, Tensor,
};

#[test]
fn test_custom_loss() {
    let x = Tensor::from_slice2(&[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        .reshape([-1, 5])
        .to_kind(Kind::Float)
        .set_requires_grad(true);
    // y = sum(xi^2)
    let y = Tensor::from_slice2(&[[2, 4, 6, 8, 10], [12, 14, 16, 18, 20]])
        .reshape([-1, 5])
        .to_kind(Kind::Float)
        .set_requires_grad(true);
    let vs = nn::VarStore::new(Device::Cpu);
    let k = vs
        .root()
        .f_var("coef", &[], nn::Init::Uniform { lo: 0f64, up: 1f64 })
        .unwrap();
    println!("{:?}", vs.variables());

    let mut opt = nn::AdamW::default().build(&vs, 5e-4).unwrap();
    for _ in 0..=10000 {
        opt.zero_grad();
        let out = &k * &x;
        let loss = ((&out - &y) * (&out - &y)).mean(Some(Kind::Float));
        loss.backward();
        opt.step();
        loss.print();
        k.grad().print();
    }
    (&k * &x).print();
    x.print();
    y.print();
    println!("");
}

#[test]
fn test_simple_grad() {
    let x = Tensor::ones(&[2, 2], (Kind::Float, Device::Cpu)).set_requires_grad(true);
    let y = &x + 2;
    y.print();
    let z = &y * &y * 3;
    let out = z.mean(Some(Kind::Float));
    z.print();
    out.print();
    out.backward();
    x.grad().print();
}
