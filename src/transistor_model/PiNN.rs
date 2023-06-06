use anyhow::Result;
use tch::{
    nn,
    nn::OptimizerConfig,
    nn::{Func, Module},
    Device, Tensor,
};

const VD_NEURON_NUM: i64 = 3;
const VG_NEURON_NUM: i64 = 2;

#[derive(Debug)]
pub struct PiNN_net<'a> {
    // vd_sub_layer0: nn::Linear,
    // vd_sub_layer1: nn::Linear,
    // vg_sub_layer0: nn::Linear,
    // vg_sub_layer1: nn::Linear,
    layer0: Func<'a>,
    layer1: Func<'a>,
}

impl<'a> PiNN_net<'a> {
    fn new(vs: &nn::Path) -> PiNN_net<'a> {
        // stride -- padding -- dilation
        let vd_sub_layer0 = nn::linear(vs / "vd_subnet_layer0", 1, VD_NEURON_NUM, Default::default());
        let vd_sub_layer1 = nn::linear(vs / "vd_subnet_layer1", 1, VD_NEURON_NUM, Default::default());
        let vg_sub_layer0 = nn::linear(vs / "vg_subnet_layer0", 1, VG_NEURON_NUM, Default::default());
        let vg_sub_layer1 = nn::linear(vs / "vg_subnet_layer1", 1, VG_NEURON_NUM, Default::default());
        let layer0 = nn::func::<'a>(|tensor| {
            let v = tensor.view([2]);
            let vd_out = v.get(0).apply(&vd_sub_layer0).tanh();
            let vg_out = v.get(1).apply(&vg_sub_layer0).sigmoid();
            let v = Tensor::cat(&[vd_out, vd_out + vg_out], 0);
            v
        });
        let layer1 = nn::func(|tensor| {
            let v = tensor.view([2]);
            let vd_out = v.get(0).apply(&vd_sub_layer1).tanh();
            let vg_out = v.get(1).apply(&vg_sub_layer1).sigmoid();
            let v = Tensor::cat(&[vd_out, vd_out + vg_out], 0);
            v
        });
        PiNN_net {
            layer0,
            layer1,
        }
    }
}

impl<'a> Module for PiNN_net<'a> {
    fn forward(&self, xs: &tch::Tensor) -> tch::Tensor {}
}

pub fn run() -> Result<()> {
    let m = tch::vision::mnist::load_dir("data")?;
    let vs = nn::VarStore::new(Device::Cpu);
    let net = net(&vs.root());
    let mut opt = nn::AdamW::default().build(&vs, 1e-3)?;
    for epoch in 1..200 {
        let loss = net
            .forward(&m.train_images)
            .cross_entropy_for_logits(&m.train_labels);
        opt.backward_step(&loss);
        let test_accuracy = net
            .forward(&m.test_images)
            .accuracy_for_logits(&m.test_labels);
        println!(
            "epoch: {:4} train loss: {:8.5} test acc: {:5.2}%",
            epoch,
            f64::from(&loss),
            100. * f64::from(&test_accuracy),
        );
    }
    Ok(())
}
