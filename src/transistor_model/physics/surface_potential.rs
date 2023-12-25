use super::constant;
use crate::loader::IV_measurements;
use rand::distributions::Uniform;
use rand::Rng;
use tch::Tensor;

#[derive(Clone, Debug)]
pub struct SurfacePotentialModel {
    wmu_l: f32, // ゲート幅 * 移動度 / ゲート長
    cox: f32,   // 酸化被膜の単位面積当たりの容量
    beta: f32,  // q/kT thermal voltageの逆数
    vfb: f32,   // フラットバンド電圧
    nsub: f32,  // 基板不純物濃度
    vbi: f32,   // built-in voltage
}

impl SurfacePotentialModel {
    pub fn new(wmu_l: f32, cox: f32, /*beta: f32,*/ vfb: f32, nsub: f32, vbi: f32) -> Self {
        Self {
            wmu_l: wmu_l,
            cox: cox,
            beta: 1.0 / constant::PHI_T,
            vfb: vfb,
            nsub: nsub,
            vbi: vbi,
        }
    }

    fn implicit_charge_equation(
        &self,
        phi_s: f32,
        phi_f: f32,
        vgs: f32,
        vds: f32,
        vbs: f32,
    ) -> f32 {
        // 熱平衡時の多数キャリアと小数キャリアの密度比
        let np0_nn0: f32 = f32::exp(-(constant::Q * self.vbi) / (constant::K * constant::T));
        self.cox * (vgs - self.vfb - phi_s)
            - f32::sqrt(2.0 * constant::ESI * self.nsub / self.beta)
                / f32::sqrt(
                    f32::exp(-self.beta * (phi_s - vbs)) + self.beta * (phi_s - vbs) - 1.0
                        + np0_nn0
                            * (f32::exp(self.beta * (phi_s - phi_f))
                                - f32::exp(self.beta * (vbs - phi_f))),
                )
    }

    pub fn calc_drain_potential(&self, vgs: f32, vds: f32, vbs: f32) -> f32 {
        let mut l = -1e5f32;
        let mut r = 1e5f32;
        let eps = 1e-9;
        while r - l > eps {
            let m = (l + r) / 2.0;
            if self.implicit_charge_equation(m, vds, vgs, vds, vbs) < 0.0 {
                l = m;
            } else {
                r = m;
            }
        }

        (l + r) / 2.0
    }

    pub fn calc_source_potential(&self, vgs: f32, vds: f32, vbs: f32) -> f32 {
        let mut l = -1e10f32;
        let mut r = 1e10f32;
        let eps = 1e-3;
        while r - l > eps {
            let m = (l + r) / 2.0;
            if self.implicit_charge_equation(m, 0.0, vgs, vds, vbs) < 0.0 {
                l = m;
            } else {
                r = m;
            }
        }

        (l + r) / 2.0
    }

    pub fn model<'a>(&'a self) -> impl Fn(f32, f32, f32) -> f32 + 'a {
        |vgs: f32, vds: f32, vbs: f32| -> f32 {
            let phi_s0 = self.calc_source_potential(vgs, vds, vbs);
            let phi_sl = self.calc_drain_potential(vgs, vds, vbs);
            let idd = self.cox * (self.beta * (vgs - self.vfb + 1.0) * (phi_sl - phi_s0))
                - self.beta / 2.0 * self.cox * (phi_sl * phi_sl - phi_s0 * phi_s0)
                - 2.0 / 3.0
                    * f32::sqrt(2.0 * constant::ESI * constant::Q * self.nsub / self.beta)
                    * (f32::powf(self.beta * (phi_sl - vbs) - 1.0, 1.5)
                        - (self.beta * (phi_s0 - vbs) - 1.0))
                + f32::sqrt(2.0 * constant::ESI * constant::Q * self.nsub / self.beta)
                    * (f32::sqrt(self.beta * (phi_sl - vbs) - 1.0)
                        - (self.beta * (phi_s0 - vbs) - 1.0));
            let ids: f32 = self.wmu_l * idd / self.beta;

            ids
        }
    }

    pub fn tfun(&self, vgs_vds: &Tensor) -> Tensor {
        assert_eq!(vgs_vds.size2().unwrap().1, 2);

        let mut vec: Vec<f32> = Vec::new();
        for i in 0..vgs_vds.size2().unwrap().0 {
            let vgs = vgs_vds.double_value(&[i, 0]);
            let vds = vgs_vds.double_value(&[i, 1]);

            vec.push(self.model()(vgs as f32, vds as f32, 0.0));
        }

        Tensor::from_slice(&vec)
    }

    pub fn make_grid(&self, vgs_grid: Vec<f32>, vds_grid: Vec<f32>) -> Vec<Vec<f32>> {
        let mut ret = vec![Vec::<f32>::new(), Vec::<f32>::new(), Vec::<f32>::new()];
        for vgs in &vgs_grid {
            for vds in &vds_grid {
                let ids = self.model()(*vgs, *vds, 0.0);
                ret[0].push(*vgs);
                ret[1].push(*vds);
                ret[2].push(ids);
            }
        }

        ret
    }

    fn set_param(&mut self, params: (f32, f32, f32, f32, f32, f32)) {
        self.wmu_l = params.0; // ゲート幅 * 移動度 / ゲート長
        self.cox = params.1; // 酸化被膜の単位面積当たりの容量
        self.beta = params.2; // q/kT thermal voltageの逆数
        self.vfb = params.3; // フラットバンド電圧
        self.nsub = params.4; // 基板不純物濃度
        self.vbi = params.5; // built-in voltage
    }

    fn params(&self) -> (f32, f32, f32, f32, f32, f32) {
        (
            self.wmu_l, self.cox, self.beta, self.vfb, self.nsub, self.vbi,
        )
    }

    pub fn simulated_anealing(&mut self, data: IV_measurements, start_temp: f32, epoch: usize) {
        let vgs = data.vgs;
        let vds = data.vds;
        let ids = data.ids;

        let mut rng = rand::thread_rng();
        let uni = Uniform::new_inclusive(-1.0, 1.0);

        let mut best_param = (
            self.wmu_l, self.cox, self.beta, self.vfb, self.nsub, self.vbi,
        );
        let objective = |model: &dyn Fn(f32, f32, f32) -> f32,
                         vgs: &Vec<f32>,
                         vds: &Vec<f32>,
                         ids: &Vec<f32>|
         -> f32 {
            let datanum: f32 = vgs.len() as f32;
            vgs.iter()
                .zip(vds.iter().zip(ids.iter()))
                .fold(0.0, |acc, (vg, (vd, id))| {
                    let id_pred = model(*vg, *vd, 0.0);
                    acc + (id_pred - id) * (id_pred - id)
                })
                / datanum
        };
        let mut best_score = objective(&self.model(), &vgs, &vds, &ids);
        for e in 0..epoch {
            println!("DEBUG: epoch={}", e);
            println!("MODEL: {:?}", self);
            let temp = start_temp
                * f32::powi(1.0 - (e as f32 / epoch as f32), 1)
                * f32::powi(-(e as f32 / epoch as f32), 4);
            let rate = f32::exp(-1.0 / temp);

            let pre_param = self.params();
            let pre_score = objective(&self.model(), &vgs, &vds, &ids);

            // 遷移関数
            let select_param: f32 = rng.gen();
            // betaは動かさない．
            let (select_wmu_l, select_cox, _select_beta, select_vfb, select_nsub, select_vbi) =
                if select_param < 1.0 / 5.0 {
                    (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                } else if select_param < 2.0 / 5.0 {
                    (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
                } else if select_param < 3.0 / 5.0 {
                    (0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
                } else if select_param < 4.0 / 5.0 {
                    (0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
                } else {
                    (0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
                };
            let diff = rng.sample(uni);
            let new_wmu_l = pre_param.0 + select_wmu_l * diff * rate;
            let new_cox = pre_param.1 + select_cox * diff * rate;
            let new_vfb = pre_param.3 + select_vfb * diff * rate;
            let new_nsub = pre_param.4 + select_nsub * diff * rate;
            let new_vbi = pre_param.5 + select_vbi * diff * rate;
            let new_param = (new_wmu_l, new_cox, self.beta, new_vfb, new_nsub, new_vbi);
            self.set_param(new_param);

            let new_score = objective(&self.model(), &vgs, &vds, &ids);

            if new_score < best_score {
                best_score = new_score;
                best_param = new_param;
            }

            let prob = f32::exp((1.0 / new_score - 1.0 / pre_score) / temp);

            // f32型: [0, 1)の一様分布からサンプル
            if prob <= rng.gen() {
                self.set_param(pre_param);
            }
        }

        self.set_param(best_param);
        println!("Score: {:?}", objective(&self.model(), &vgs, &vds, &ids));
    }
}

#[test]
fn test_calc_potential() {
    // use crate::loader;
    // let dataset = loader::read_csv("data/25_train.csv".to_string()).unwrap();
    // let vgs = dataset.vgs;
    // let vds = dataset.vds;
    // let ids = dataset.ids;
    let model = SurfacePotentialModel {
        wmu_l: 1.5474586,
        cox: 0.32980648,
        beta: 1.0 / constant::PHI_T,
        vfb: 1.2253355,
        nsub: 1.5272179,
        vbi: 1.3043324,
    };
    let phi_s0 = model.calc_source_potential(5.0, 10.0, 0.0);
    println!("surface potential(source) = {}", phi_s0);
}

#[test]
fn test_sa() {
    use crate::loader;

    let dataset = loader::read_csv("data/25_train.csv".to_string()).unwrap();
    let mut model = SurfacePotentialModel {
        wmu_l: 1.5474586,
        cox: 0.32980648,
        beta: 1.0 / constant::PHI_T,
        vfb: 1.2253355,
        nsub: 1.5272179,
        vbi: 1.3043324,
    };
    // let mut model = Level1 { kp: 0.61027044, lambda: 0.037695386, vth: 5.5387435 };
    // Score: 0.58780473

    model.simulated_anealing(dataset, 100.0, 1000);

    println!("{:?}", model);
}

#[test]
fn test_make_grid() {
    use std::fs::File;
    use std::io::BufWriter;
    use std::io::Write;
    use std::path::Path;

    let model = SurfacePotentialModel {
        wmu_l: 1.5474586,
        cox: 0.32980648,
        beta: 1.0 / constant::PHI_T,
        vfb: 1.2253355,
        nsub: 1.5272179,
        vbi: 1.3043324,
    };
    // let model = Level1::new(0.83, 0.022, 5.99);
    let grid = model.make_grid(
        (0..20)
            .step_by(2)
            .into_iter()
            .map(|x| x as f32 / 1.0)
            .collect::<Vec<_>>(),
        (0..200)
            .step_by(4)
            .into_iter()
            .map(|x| x as f32 / 10.0)
            .collect::<Vec<_>>(),
    );
    let data_output_path = Path::new("./data");
    let mut w = BufWriter::new(
        File::create(data_output_path.join("surface_potential_reference_data.csv")).unwrap(),
    );
    let _ = writeln!(w, "VGS,VDS,IDS,IDS_PRED");
    for (vgs, (vds, ids)) in grid[0]
        .clone()
        .into_iter()
        .zip(grid[1].clone().into_iter().zip(grid[2].clone().into_iter()))
    {
        let _ = writeln!(w, "{},{},{},{}", vgs, vds, ids, ids);
    }
}
