use super::constant;
use crate::loader::IV_measurements;
use ::std::ops::Index;
use ::std::ops::IndexMut;
use rand::distributions::Uniform;
use rand::Rng;
use tch::Tensor;

#[derive(Clone, Debug)]
pub struct SurfacePotentialModel {
    tox: f32,    // 酸化被膜の厚さ
    vfbc: f32,   // p型基板領域のフラットバンド電圧
    na: f32,     // アクセプター濃度
    rs: f32,     // ソースの寄生容量
    delta: f32,  // スムージングパラメータ
    k: f32,      // スケーリングファクター(Idsの係数みたいなやつ)
    lambda: f32, // チャネル長変調
    theta: f32,  // 移動度減衰?(Mobility Degletion)
    rd: f32,     // ドレインの寄生容量
}

impl Index<usize> for SurfacePotentialModel {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < 9);
        if index == 0 {
            &self.tox
        } else if index == 1 {
            &self.vfbc
        } else if index == 2 {
            &self.na
        } else if index == 3 {
            &self.rs
        } else if index == 4 {
            &self.delta
        } else if index == 5 {
            &self.k
        } else if index == 6 {
            &self.lambda
        } else if index == 7 {
            &self.theta
        } else if index == 8 {
            &self.rd
        } else {
            panic!()
        }
    }
}
impl IndexMut<usize> for SurfacePotentialModel {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        assert!(index < 9);
        if index == 0 {
            &mut self.tox
        } else if index == 1 {
            &mut self.vfbc
        } else if index == 2 {
            &mut self.na
        } else if index == 3 {
            &mut self.rs
        } else if index == 4 {
            &mut self.delta
        } else if index == 5 {
            &mut self.k
        } else if index == 6 {
            &mut self.lambda
        } else if index == 7 {
            &mut self.theta
        } else if index == 8 {
            &mut self.rd
        } else {
            panic!()
        }
    }
}

impl SurfacePotentialModel {
    const T: f32 = 300.0;
    const PHI_T: f32 = constant::K * Self::T / constant::Q; // thermal voltage[V]
    const ESIC: f32 = 9.7 * constant::E0;

    pub fn new(
        tox: f32,    // 酸化被膜の厚さ
        vfbc: f32,   // p型基板領域のフラットバンド電圧
        na: f32,     // アクセプター濃度
        rs: f32,     // ソースの寄生容量
        delta: f32,  // スムージングパラメータ
        k: f32,      // スケーリングファクター(Idsの係数みたいなやつ)
        lambda: f32, // チャネル長変調
        theta: f32,  // 移動度減衰?(Mobility Degletion)
        rd: f32,     // ドレインの寄生容量
    ) -> Self {
        Self {
            tox: tox,       // 酸化被膜の厚さ
            vfbc: vfbc,     // p型基板領域のフラットバンド電圧
            na: na,         // アクセプター濃度
            rs: rs,         // ソースの寄生容量
            delta: delta,   // スムージングパラメータ
            k: k,           // スケーリングファクター(Idsの係数みたいなやつ)
            lambda: lambda, // チャネル長変調
            theta: theta,   // 移動度減衰?(Mobility Degletion)
            rd: rd,         // ドレインの寄生容量
        }
    }

    fn fermi_level(&self) -> f32 {
        2.0 * constant::K * Self::T / constant::Q * f32::log10(self.na / constant::NI)
    }

    pub fn calc_potential(&self, vgs: f32, vds: f32, vbs: f32) -> (f32, f32) {
        // initial values
        let mut phi_s0 =
            2.0 * constant::K * Self::T / constant::Q * f32::log10(self.na / constant::NI);
        let mut phi_sl = phi_s0 + vds;
        // let mut phi_s0 = 1.0;
        // let mut phi_sl = 1.0;

        let eq = |phi_s: f32, phi_f: f32| -> f32 {
            let eyy = 0.0; //gradual channel 近似?
            let cox = constant::EOX / self.tox;
            let delta_vg_dash = |phi_s: f32| -> f32 {
                constant::ESI / cox
                    * eyy
                    * f32::sqrt(2.0 * constant::ESI / (constant::Q * self.na) * (phi_s - vbs - 1.0))
            };
            let vg_dash = |phi_s: f32| -> f32 { vgs + delta_vg_dash(phi_s) - self.vfbc };

            cox * (vg_dash(phi_s) - phi_s)
                - (f32::sqrt(2.0 * constant::Q * Self::ESIC * self.na)
                    * f32::sqrt(
                        Self::PHI_T * f32::exp(-phi_s / Self::PHI_T) + phi_s - Self::PHI_T
                            + f32::exp(-(2.0 * self.fermi_level() + phi_f) / Self::PHI_T)
                                * (Self::PHI_T * f32::exp(phi_s / Self::PHI_T)
                                    - phi_s
                                    - Self::PHI_T),
                    ))
        };

        let eps = 1e-9;
        while eq(phi_s0, 0.0) < eps {
            let dif_phi = 1e-12;
            let dif_eq = eq(phi_s0 + dif_phi, 0.0) - eq(phi_s0, 0.0);
            phi_s0 = phi_s0 - eq(phi_s0, 0.0) / dif_eq;
        }

        while eq(phi_sl, vds) < eps {
            let dif_phi = 1e-12;
            let dif_eq = eq(phi_s0 + dif_phi, vds) - eq(phi_s0, vds);
            phi_sl = phi_sl - eq(phi_s0, vds) / dif_eq;
        }

        assert!(!phi_s0.is_nan());
        assert!(!phi_sl.is_nan());

        (phi_s0, phi_sl)
    }

    pub fn model<'a>(&'a self) -> impl Fn(f32, f32, f32) -> f32 + 'a {
        |vgs: f32, vds: f32, vbs: f32| -> f32 {
            let cox = constant::EOX / self.tox;
            let (phi_ss, phi_sd) = self.calc_potential(vgs, vds, vbs);
            let idd = cox * (vgs - self.vfbc + Self::PHI_T) * (phi_sd - phi_ss)
                - 0.5 * cox * (phi_sd * phi_sd - phi_ss * phi_ss)
                - 2.0 / 3.0
                    * Self::PHI_T
                    * f32::sqrt(2.0 * Self::ESIC * constant::K * Self::T * self.na)
                    * (f32::powf(phi_sd / Self::PHI_T - 1.0, 1.5)
                        - f32::powf(phi_ss / Self::PHI_T - 1.0, 1.5))
                + Self::PHI_T
                    * f32::sqrt(2.0 * Self::ESIC * constant::K * Self::T * self.na)
                    * (f32::powf(phi_sd / Self::PHI_T - 1.0, 0.5)
                        - f32::powf(phi_ss / Self::PHI_T - 1.0, 0.5));

            assert!(!idd.is_nan());

            1.0 / (1.0 + self.theta * vgs) * (1.0 + self.lambda * vds) * self.k * idd
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

    fn set_param(&mut self, params: (f32, f32, f32, f32, f32, f32, f32, f32, f32)) {
        self.tox = params.0; // 酸化被膜の厚さ
        self.vfbc = params.1; // p型基板領域のフラットバンド電圧
        self.na = params.2; // アクセプター濃度
        self.rs = params.3; // ソースの寄生容量
        self.delta = params.4; // スムージングパラメータ
        self.k = params.5; // スケーリングファクター(Idsの係数みたいなやつ)
        self.lambda = params.6; // チャネル長変調
        self.theta = params.7; // 移動度減衰?(Mobility Degletion)
        self.rd = params.8; // ドレインの寄生容量
    }

    fn params(&self) -> (f32, f32, f32, f32, f32, f32, f32, f32, f32) {
        (
            self.tox,
            self.vfbc,
            self.na,
            self.rs,
            self.delta,
            self.k,
            self.lambda,
            self.theta,
            self.rd,
        )
    }

    pub fn simulated_anealing(&mut self, data: IV_measurements, start_temp: f32, epoch: usize) {
        let vgs = data.vgs;
        let vds = data.vds;
        let ids = data.ids;

        let mut rng = rand::thread_rng();
        let uni = Uniform::new_inclusive(-1.0, 1.0);

        let mut best_param = (
            self.tox,
            self.vfbc,
            self.na,
            self.rs,
            self.delta,
            self.k,
            self.lambda,
            self.theta,
            self.rd,
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
            let temp = start_temp
                * f32::powi(1.0 - (e as f32 / epoch as f32), 1)
                * f32::exp2(-((e as f32 + 0.5) / epoch as f32) * 4.0);
            let rate = f32::exp(-1.0 / temp);

            let pre_score = objective(&self.model(), &vgs, &vds, &ids);

            // 遷移関数
            let select_param: usize = rng.sample(Uniform::new(0, 9));
            // let param_sensitivity = vec![1e-8, 1e-8, 1e10, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
            let diff = rng.sample(uni);
            let pre_param = self[select_param];
            self[select_param] = self[select_param] + diff * rate; /* * param_sensitivity[select_param]; */

            let new_score = objective(&self.model(), &vgs, &vds, &ids);

            if new_score < best_score {
                best_score = new_score;
                best_param = self.params();
            }

            let prob = f32::exp((1.0 / new_score - 1.0 / pre_score) / temp);

            // f32型: [0, 1)の一様分布からサンプル
            if prob <= rng.gen() {
                self[select_param] = pre_param;
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
        tox: 1.6328718e-7,
        vfbc: -8.462384e-7,
        na: 439611800000.0,
        rs: -113.43237,
        delta: 41.88081,
        k: 47.76392,
        lambda: -0.0010460697,
        theta: -56.88513,
        rd: -2.7292771,
    };
    let (phi_s0, phi_sl) = model.calc_potential(5.0, 100.0, 0.0);
    println!("surface potential(source) = {}", phi_s0);
    println!("surface potential(drain) = {}", phi_sl);
}

#[test]
fn test_sa() {
    use crate::loader;

    let dataset = loader::read_csv("data/integral.csv".to_string()).unwrap();
    let mut model = SurfacePotentialModel {
        tox: 1.6328718e-7,
        vfbc: -8.462384e-7,
        na: 439611800000.0,
        rs: -113.43237,
        delta: 41.88081,
        k: 48.12051,
        lambda: -0.0010460697,
        theta: -56.88513,
        rd: -2.7292771,
    };
    // Score: 1.8893663

    model.simulated_anealing(dataset, 100000.0, 1000000);

    println!("{:?}", model);
}

#[test]
fn test_make_grid() {
    use std::fs::File;
    use std::io::BufWriter;
    use std::io::Write;
    use std::path::Path;

    let model = SurfacePotentialModel {
        tox: 1.6328718e-7,
        vfbc: -8.462384e-7,
        na: 1e5,
        rs: -113.43237,
        delta: 41.88081,
        k: 48.12051,
        lambda: -0.0010460697,
        theta: -56.88513,
        rd: -2.7292771,
    };
    // let model = Level1::new(0.83, 0.022, 5.99);
    let grid = model.make_grid(
        (0..20)
            .step_by(2)
            .into_iter()
            .map(|x| x as f32 / 1.0)
            .collect::<Vec<_>>(),
        (0..6000)
            .step_by(10)
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

#[test]
fn test_objective() {
    use crate::loader;

    let dataset = loader::read_csv("data/integral.csv".to_string()).unwrap();

    let model = SurfacePotentialModel {
        tox: 1.6328718e-7,
        vfbc: -8.462384e-7,
        na: 439611800000.0,
        rs: -113.43237,
        delta: 41.88081,
        k: 48.12051,
        lambda: -0.0010460697,
        theta: -56.88513,
        rd: -2.7292771,
    };

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

    println!(
        "Score: {}",
        objective(&model.model(), &dataset.vgs, &dataset.vds, &dataset.ids)
    );
}
