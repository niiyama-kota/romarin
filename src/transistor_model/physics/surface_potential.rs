// use super::constant;
use crate::loader::IV_measurements;
use rand::distributions::Uniform;
use rand::Rng;
use tch::Tensor;

#[derive(Debug, Copy, Clone, Hash)]
pub enum Mostype {
    Pmos,
    Nmos,
}

#[derive(Clone, Debug)]
pub struct SurfacePotential {
    kp: f32,
    lambda: f32,
    vth: f32,
}

impl SurfacePotential {
    pub fn new(kp: f32, lambda: f32, vth: f32) -> Self {
        Self {
            kp: kp,
            lambda: lambda,
            vth: vth,
        }
    }

    pub fn model<'a>(&'a self) -> impl Fn(f32, f32) -> f32 + 'a {
        |vgs: f32, vds: f32| -> f32 {
            let ids: f32 = if vgs <= self.vth {
                0.0
            } else if vds < vgs - self.vth {
                self.kp * (1.0 + self.lambda * vds) * ((vgs - self.vth) * vds - 0.5 * vds * vds)
            } else {
                0.5 * self.kp * (1.0 + self.lambda * vds) * (vgs - self.vth) * (vgs - self.vth)
            };

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

    pub fn make_grid(&self, vgs_grid: Vec<f32>, vds_grid: Vec<f32>) -> Vec<Vec<f32>> {
        let mut ret = vec![Vec::<f32>::new(), Vec::<f32>::new(), Vec::<f32>::new()];
        for vgs in &vgs_grid {
            for vds in &vds_grid {
                let ids = self.model()(*vgs, *vds);
                ret[0].push(*vgs);
                ret[1].push(*vds);
                ret[2].push(ids);
            }
        }

        ret
    }

    fn set_param(&mut self, params: (f32, f32, f32)) {
        self.kp = params.0;
        self.lambda = params.1;
        self.vth = params.2;
    }

    fn params(&self) -> (f32, f32, f32) {
        (self.kp, self.lambda, self.vth)
    }

    pub fn simulated_anealing(
        &mut self,
        data: IV_measurements,
        start_temp: f32,
        end_temp: f32,
        epoch: usize,
    ) {
        let vgs = data.vgs;
        let vds = data.vds;
        let ids = data.ids;

        let mut rng = rand::thread_rng();
        let uni = Uniform::new_inclusive(-1.0, 1.0);

        let mut best_param = (self.kp, self.lambda, self.vth);
        let param_sensitivity = (1.0f32, 0.01f32, 1.0f32);
        let objective = |model: &dyn Fn(f32, f32) -> f32,
                         vgs: &Vec<f32>,
                         vds: &Vec<f32>,
                         ids: &Vec<f32>|
         -> f32 {
            let datanum: f32 = vgs.len() as f32;
            vgs.iter()
                .zip(vds.iter().zip(ids.iter()))
                .fold(0.0, |acc, (vg, (vd, id))| {
                    let id_pred = model(*vg, *vd);
                    acc + (id_pred - id) * (id_pred - id)
                })
                / datanum
        };
        for e in 0..epoch {
            let temp = start_temp + (end_temp - start_temp) * (e as f32 / epoch as f32);
            let rate = f32::exp(-1.0 / temp);

            let pre_param = self.params();
            let pre_score = objective(&self.model(), &vgs, &vds, &ids);

            // 遷移関数
            let new_kp = pre_param.0 + rng.sample(uni) * param_sensitivity.0 * rate;
            let new_lambda = pre_param.1 + rng.sample(uni) * param_sensitivity.1 * rate;
            let new_vth = pre_param.2 + rng.sample(uni) * param_sensitivity.2 * rate;
            let new_param = (new_kp, new_lambda, new_vth);
            self.set_param(new_param);

            let new_score = objective(&self.model(), &vgs, &vds, &ids);

            if new_score > pre_score {
                best_param = new_param;
            }

            let prob = f32::exp((pre_score - new_score) / temp);

            // f32型: [0, 1)の一様分布からサンプル
            if prob <= rng.gen() {
                self.set_param(pre_param);
            }
        }

        self.set_param(best_param);
        // println!("Score: {:?}", objective(&self.model(), &vgs, &vds, &ids));
    }
}
