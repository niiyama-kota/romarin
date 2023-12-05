use tch::nn::Module;
use tch::Tensor;

pub struct Threshold {
    vth: f32,
    delta: f32,
    k: f32,
    clm: f32,
    md: f32,
    mdv: f32,
    rd: f32,
}

impl Threshold {
    pub fn new(vth: f32, delta: f32, k: f32, clm: f32, md: f32, mdv: f32, rd: f32) -> Self {
        Threshold {
            vth: vth,
            delta: delta,
            k: k,
            clm: clm,
            md: md,
            mdv: mdv,
            rd: rd,
        }
    }

    pub fn model<'a>(&'a self) -> impl Fn(f32, f32) -> f32 + 'a {
        |vgs: f32, vds: f32| -> f32 {
            let vp = vgs - self.vth;
            let vds_mod = vds / (1.0 + (vds / vp).powf(self.delta)).powf(1.0 / self.delta);
            let idd = vp * vds_mod - 0.5 * vds_mod * vds_mod;
            let ids_on =
                (self.k * (1.0 + self.clm * vds) * idd) / (1.0 + self.md * (vgs - self.mdv));

            let ids = if vgs >= self.vth { ids_on } else { 0.0 };

            ids
        }
    }

    pub fn tfun(&self, vgs_vds: &Tensor) -> Tensor {
        assert_eq!(vgs_vds.size2().unwrap().1, 2);

        let mut vec: Vec<f32> = Vec::new();
        for i in 0..vgs_vds.size2().unwrap().0 {
            let vgs = vgs_vds.double_value(&[i, 0]);
            let vds = vgs_vds.double_value(&[i, 1]);

            vec.push(self.model()(vgs as f32, vds as f32));
        }

        Tensor::from_slice(&vec)
    }
}

#[test]
fn test_tfun() {
    let model = Threshold::new(
        5.965096659982493,
        1.8418581815787844,
        0.8412370200276217,
        0.020643794780914274,
        0.042401849118418024,
        14.845124194937101,
        0.0,
    );

    let vgs_vds = Tensor::from_slice2(&[
        &[0.0, 0.0],
        &[0.0, 5.0],
        &[0.0, 10.0],
        &[3.0, 0.0],
        &[3.0, 5.0],
        &[3.0, 10.0],
        &[6.0, 0.0],
        &[6.0, 5.0],
        &[6.0, 10.0],
    ]);

    let ids = model.tfun(&vgs_vds);

    ids.print();
}
