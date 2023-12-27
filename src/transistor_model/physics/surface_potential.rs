use crate::loader::IV_measurements;
use ::std::ops::Index;
use ::std::ops::IndexMut;
use rand::distributions::Uniform;
use rand::seq::SliceRandom;
use rand::Rng;
use rand_distr::Distribution;
use rand_distr::Normal;
use tch::Tensor;

#[derive(Clone, Debug)]
pub struct SurfacePotentialModel {
    scale: f64,  // スケーリングファクター(Idsの係数みたいなやつ)
    tox: f64,    // 酸化被膜の厚さ
    na: f64,     // アクセプター濃度
    lambda: f64, // チャネル長変調
    vfbc: f64,   // p型基板領域のフラットバンド電圧
    theta: f64,  // 移動度減衰?(Mobility Degletion)
    delta: f64,  // スムージングパラメータ
    alpha: f64,  // ?
    rd: f64,     // ドレインの寄生容量
}

impl Index<usize> for SurfacePotentialModel {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < 9);
        if index == 0 {
            &self.scale
        } else if index == 1 {
            &self.tox
        } else if index == 2 {
            &self.na
        } else if index == 3 {
            &self.lambda
        } else if index == 4 {
            &self.vfbc
        } else if index == 5 {
            &self.theta
        } else if index == 6 {
            &self.delta
        } else if index == 7 {
            &self.alpha
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
            &mut self.scale
        } else if index == 1 {
            &mut self.tox
        } else if index == 2 {
            &mut self.na
        } else if index == 3 {
            &mut self.lambda
        } else if index == 4 {
            &mut self.vfbc
        } else if index == 5 {
            &mut self.theta
        } else if index == 6 {
            &mut self.delta
        } else if index == 7 {
            &mut self.alpha
        } else if index == 8 {
            &mut self.rd
        } else {
            panic!()
        }
    }
}

impl SurfacePotentialModel {
    const Q: f64 = 1.60210e-19; // 電気素量[C]
    const KB: f64 = 1.38054e-23; // ボルツマン定数[J/K]
    const T: f64 = 300.0; // 絶対温度[K}]
    const PHI_T: f64 = Self::KB * Self::T / Self::Q; // thermal voltage[V]
    const EPS0_M2: f64 = 8.854187817 * 1e-12; //真空中の誘電率[F/m]
    const EPS0_CM2: f64 = Self::EPS0_M2 * 1e-2; //真空中の誘電率[F/cm]
    const EPS_OX_CM2: f64 = 3.9 * Self::EPS0_CM2;
    const EPS_SIC_CM2: f64 = 9.7 * Self::EPS0_CM2; //SiCの誘電率[F/cm]
    const NI_SIC_CM2: f64 = 9.7e-9; // intrinsic carrier density of SiC[個/cm^3]

    pub fn new(
        scale: f64,  // スケーリングファクター(Idsの係数みたいなやつ)
        tox: f64,    // 酸化被膜の厚さ
        na: f64,     // アクセプター濃度
        lambda: f64, // チャネル長変調
        vfbc: f64,   // p型基板領域のフラットバンド電圧
        theta: f64,  // 移動度減衰?(Mobility Degletion)
        delta: f64,  // スムージングパラメータ
        alpha: f64,  // ?
        rd: f64,     // ドレインの寄生容量
    ) -> Self {
        Self {
            scale: scale,   // スケーリングファクター(Idsの係数みたいなやつ)
            tox: tox,       // 酸化被膜の厚さ
            na: na,         // アクセプター濃度
            lambda: lambda, // チャネル長変調
            vfbc: vfbc,     // p型基板領域のフラットバンド電圧
            theta: theta,   // 移動度減衰?(Mobility Degletion)
            delta: delta,   // スムージングパラメータ
            alpha: alpha,   // ?
            rd: rd,         // ドレインの寄生容量
        }
    }

    // PSP版の表面電位計算法?
    fn psi_calc(&self, vgs: f64, vcb: f64, two_phi_f: f64, g: f64) -> f64 {
        let xg = (vgs - self.vfbc) / Self::PHI_T;
        let xi = 1.0 + g / f64::sqrt(2.0);

        let x = if f64::abs(xg / xi) < 1e-7 {
            xg / xi
        } else {
            let x23 = (two_phi_f + vcb) / (2.0 * Self::PHI_T);
            let xg23 = g * f64::sqrt(x23 - 1.0);
            let del_n = f64::exp(-(two_phi_f + vcb) / Self::PHI_T);
            let gsq = g * g;
            if xg < 0.0 {
                let yg = -xg;
                let z = 1.25 * yg / xi;
                let eta = 0.5 * (z + 10.0 - f64::sqrt((z - 6.0) * (z - 6.0) + 64.0));
                let a = (yg - eta) * (yg - eta) + gsq * (eta + 1.0);
                let c = 2.0 * (yg - eta) - gsq;
                let tau = -eta + f64::ln(a / gsq);
                let u = (a + c) * (a + c) / tau + 0.5 * c * c - a;
                let y0 = eta + a * (a + c) / (u + c * (a + c) * (c * c / 3.0 - a) / u);
                let del_0 = f64::exp(y0);
                let del_1 = 1.0 / del_0;
                let p = 2.0 * (yg - y0) + gsq * (del_0 - 1.0 + del_n * (1.0 - del_1));
                let q =
                    (yg - y0) * (yg - y0) + gsq * (y0 - del_0 + 1.0 + del_n * (1.0 - del_1 - y0));

                -y0 - 2.0 * q
                    / (p + f64::sqrt(p * p - 2.0 * q * (2.0 - gsq * (del_0 + del_n * del_1))))
            } else {
                let xbar = xg * (1.0 + xg * (xi * x23 - xg23) / (xg23 * xg23)) / xi;
                let x0 = if xg < xg23 {
                    let ebar = f64::exp(-xbar);
                    let omega = 1.0 - ebar - del_n * ((1.0 / ebar) - ebar - 2.0 + xbar);

                    xg + 0.5 * gsq - (g) * f64::sqrt(xg + 0.25 * gsq - omega)
                } else {
                    let xsub = xg + 0.5 * gsq - (g) * f64::sqrt(xg + 0.25 * gsq - 1.0);
                    let xn = (two_phi_f + vcb) / (Self::PHI_T);
                    let b = xn + 3.0;
                    let eta = 0.5 * (xsub + b - f64::sqrt((xsub - b) * (xsub - b) + 5.0));
                    let a = (xg - eta) * (xg - eta) - gsq * (eta - 1.0);
                    let c = 2.0 * (xg - eta) + gsq;
                    let tau = xn - eta + f64::ln(a / gsq);
                    let u = (a + c) * (a + c) / tau + 0.5 * c * c - a;

                    eta + (a * (a + c)) / (u + c * (a + c) * (c * c / 3.0 - a) / u)
                };

                let del_0 = f64::exp(x0);
                let del_1 = 1.0 / del_0;
                let p = 2.0 * (xg - x0) + gsq * (1.0 - del_1 + del_n * (del_0 + del_1 - 2.0));
                let q = (xg - x0) * (xg - x0)
                    - gsq * (x0 + del_1 - 1.0 + del_n * (del_0 - del_1 - 2.0 * xbar));

                x0 + 2.0 * q
                    / (p + f64::sqrt(p * p - 2.0 * q * (2.0 - gsq * (del_1 + del_n * del_0))))
            }
        };

        Self::PHI_T * x
    }

    fn idd_calc(&self, vgs: f64, /*vds: f64,*/ phi_s0: f64, phi_sl: f64) -> f64 {
        let tox_cm = self.tox * 1e2;
        let cox_cm = Self::EPS_OX_CM2 / tox_cm;
        let p_cm2 = self.na;

        let gamma = f64::sqrt(2.0 * Self::EPS_SIC_CM2 * Self::KB * Self::T * p_cm2);

        if phi_s0 >= 0.0 && phi_sl >= 0.0 {
            let idd = cox_cm * (vgs - self.vfbc + Self::PHI_T) * (phi_sl - phi_s0);
            let idd = idd - 0.5 * cox_cm * (phi_sl * phi_sl - phi_s0 * phi_s0);
            let idd = idd
                - (2.0 / 3.0)
                    * Self::PHI_T
                    * gamma
                    * (f64::powf(phi_sl / Self::PHI_T - 1.0, 1.5)
                        - f64::powf(phi_s0 / Self::PHI_T - 1.0, 1.5));
            let idd = idd
                + Self::PHI_T
                    * gamma
                    * (f64::powf(phi_sl / Self::PHI_T - 1.0, 0.5)
                        - f64::powf(phi_s0 / Self::PHI_T - 1.0, 0.5));

            let idd = idd / Self::PHI_T;

            idd
        } else {
            1e-12
        }
    }

    pub fn model<'a>(&'a self) -> impl Fn(f64, f64, f64) -> f64 + 'a {
        |vgs: f64, vds: f64, _vbs: f64| -> f64 {
            let p_cm2 = self.na;
            let tox_cm = self.tox * 1e2;
            let cox_cm = Self::EPS_OX_CM2 / tox_cm;

            let vds = if vds <= 0.0 { 1e-6 } else { vds };
            let vgs_eff = vgs;
            let vds_eff = vds / f64::powf(1.0 + f64::powf(vds / vgs, self.delta), 1.0 / self.delta);

            let xxx = f64::sqrt(2.0 * Self::Q * Self::EPS_SIC_CM2 * p_cm2) / cox_cm;
            let ggg = xxx / f64::sqrt(Self::PHI_T);
            let two_phi_f = 2.0 * Self::PHI_T * f64::ln(p_cm2 / Self::NI_SIC_CM2);
            let phi_s0 = self.psi_calc(vgs, 0.0, two_phi_f, ggg);
            let phi_sl = self.psi_calc(vgs, vds_eff, two_phi_f, ggg);

            let _gamma = f64::sqrt(2.0 * Self::EPS_SIC_CM2 * Self::KB * Self::T * p_cm2);
            let idd = self.idd_calc(vgs_eff, phi_s0, phi_sl);

            let ids_tmp = if vgs > self.alpha {
                idd * self.scale * (1.0 + self.lambda * vds)
                    / (1.0 + self.theta * (vgs - self.alpha))
            } else {
                idd * self.scale * (1.0 + self.lambda * vds)
            };

            let ids = ids_tmp / (1.0 + (ids_tmp * self.rd) / vds);
            if ids < 0.0 {
                1e-12
            } else {
                ids
            }
        }
    }

    pub fn tfun(&self, vgs_vds: &Tensor) -> Tensor {
        assert_eq!(vgs_vds.size2().unwrap().1, 2);

        let mut vec: Vec<f64> = Vec::new();
        for i in 0..vgs_vds.size2().unwrap().0 {
            let vgs = vgs_vds.double_value(&[i, 0]);
            let vds = vgs_vds.double_value(&[i, 1]);

            vec.push(self.model()(vgs as f64, vds as f64, 0.0));
        }

        Tensor::from_slice(&vec)
    }

    pub fn make_grid(&self, vgs_grid: Vec<f64>, vds_grid: Vec<f64>) -> Vec<Vec<f64>> {
        let mut ret = vec![Vec::<f64>::new(), Vec::<f64>::new(), Vec::<f64>::new()];
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

    fn set_param(&mut self, params: (f64, f64, f64, f64, f64, f64, f64, f64, f64)) {
        self.scale = params.0; // スケーリングファクター(Idsの係数みたいなやつ)
        self.tox = params.1; // 酸化被膜の厚さ
        self.na = params.2; // アクセプター濃度
        self.lambda = params.3; // チャネル長変調
        self.vfbc = params.4; // p型基板領域のフラットバンド電圧
        self.theta = params.5; // 移動度減衰?(Mobility Degletion)
        self.delta = params.6; // スムージングパラメータ
        self.alpha = params.7; // ?
        self.rd = params.8; // ドレインの寄生容量
    }

    fn params(&self) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64) {
        (
            self.scale,
            self.tox,
            self.na,
            self.lambda,
            self.vfbc,
            self.theta,
            self.delta,
            self.alpha,
            self.rd,
        )
    }

    pub fn simulated_anealing(&mut self, data: IV_measurements, start_temp: f64, epoch: usize) {
        let vgs = data.vgs.into_iter().map(|x| x as f64).collect::<Vec<f64>>();
        let vds = data.vds.into_iter().map(|x| x as f64).collect::<Vec<f64>>();
        let ids = data.ids.into_iter().map(|x| x as f64).collect::<Vec<f64>>();

        let mut rng = rand::thread_rng();

        let mut best_param = (
            self.scale,
            self.tox,
            self.na,
            self.lambda,
            self.vfbc,
            self.theta,
            self.delta,
            self.alpha,
            self.rd,
        );
        let objective = |model: &dyn Fn(f64, f64, f64) -> f64,
                         vgs: &Vec<f64>,
                         vds: &Vec<f64>,
                         ids: &Vec<f64>|
         -> f64 {
            let datanum: f64 = vgs.len() as f64;
            vgs.iter()
                .zip(vds.iter().zip(ids.iter()))
                .fold(0.0, |acc, (vg, (vd, id))| {
                    let id_pred = model(*vg, *vd, 0.0);
                    acc + f64::sqrt((id_pred - id) * (id_pred - id))
                })
                / datanum
        };
        let mut best_score = objective(&self.model(), &vgs, &vds, &ids);
        // 遷移関数
        let param_step_dist = vec![
            Normal::new(0.0, f64::abs(self[0]) / 10.0).unwrap(),
            Normal::new(0.0, f64::abs(self[1]) / 10.0).unwrap(),
            Normal::new(0.0, f64::abs(self[2]) / 10.0).unwrap(),
            Normal::new(0.0, f64::abs(self[3]) / 10.0).unwrap(),
            Normal::new(0.0, f64::abs(self[4]) / 10.0).unwrap(),
            Normal::new(0.0, f64::abs(self[5]) / 10.0).unwrap(),
            Normal::new(0.0, f64::abs(self[6]) / 10.0).unwrap(),
            Normal::new(0.0, f64::abs(self[7]) / 10.0).unwrap(),
            Normal::new(0.0, f64::abs(self[8]) / 10.0).unwrap(),
            // Uniform::new_inclusive(-self[0] * 1e-4, self[0] * 1e-4),
            // Uniform::new_inclusive(-self[1] * 1e-3, self[1] * 1e-3),
            // Uniform::new_inclusive(-self[2] * 1e-4, self[2] * 1e-4),
            // Uniform::new_inclusive(-self[3] * 1e-3, self[3] * 1e-3),
            // Uniform::new_inclusive(-f64::abs(self[4]) * 1e-3, f64::abs(self[4]) * 1e-3),
            // Uniform::new_inclusive(-self[5] * 1e-3, self[5] * 1e-3),
            // Uniform::new_inclusive(-self[6] * 1e-3, self[6] * 1e-3),
            // Uniform::new_inclusive(-self[7] * 1e-3, self[7] * 1e-3),
            // Uniform::new_inclusive(-self[8] * 1e-2, self[8] * 1e-2),
        ];
        // パラメータの範囲
        let taboo = vec![
            (self[0] / 2.0, self[0] * 3.0),
            (5e-8, 5e-8),
            (1e16, 2e17),
            (self[3] / 100.0, self[3] * 100.0),
            (-7.0, 2.0),
            (self[5] / 100.0, 1.0),
            (0.4, 10.0),
            (1.0, 20.0),
            (1e-3, 100e-3),
        ];
        for e in 0..epoch {
            println!("DEBUG: {}", e);
            let temp = start_temp
                * f64::powi(1.0 - (e as f64 / epoch as f64), 3)
                * f64::exp2(-((e as f64 + 0.5) / epoch as f64) * 4.0);
            let rate = f64::exp(-1.0 / temp);

            let pre_score = objective(&self.model(), &vgs, &vds, &ids);

            let mut order = vec![0, 1, 2, 3, 4, 5, 6, 7, 8];
            order.shuffle(&mut rng);
            for p in order {
                let pre_param = self[p];
                let diff = param_step_dist[p].sample(&mut rng);
                self[p] = self[p] + diff * rate;

                let new_score = objective(&self.model(), &vgs, &vds, &ids);

                if taboo[p].0 <= self[p] && self[p] <= taboo[p].1 {
                    if new_score < best_score {
                        best_score = new_score;
                        best_param = self.params();
                    }

                    let prob = f64::exp((1.0 / new_score - 1.0 / pre_score) / temp);

                    // f64型: [0, 1)の一様分布からサンプル
                    if prob <= rng.gen() {
                        self[p] = pre_param;
                    }
                } else {
                    self[p] = pre_param;
                }
            }
            println!("Score: {:?}", objective(&self.model(), &vgs, &vds, &ids));
        }

        self.set_param(best_param);
        println!("Score: {:?}", objective(&self.model(), &vgs, &vds, &ids));
    }
}

#[test]
fn test_sa() {
    use crate::loader;

    let dataset = loader::read_csv("data/integral_train.csv".to_string()).unwrap();
    // let mut model = SurfacePotentialModel {
    //     scale: 171455.1768651102,
    //     tox: 5e-8,
    //     na: 1.2531591504771211e17,
    //     lambda: 0.019574827818568466,
    //     vfbc: -1.5850173370816367,
    //     theta: 0.004944312000295522,
    //     delta: 0.5289369284239606,
    //     alpha: 16.568182413277864,
    //     rd: 0.07734977009721601,
    // };
    let mut model = SurfacePotentialModel {
        scale: 1e5,
        tox: 5e-8,
        na: 1e17,
        lambda: 0.02,
        vfbc: -1.5850173370816367,
        theta: 0.0,
        delta: 0.5,
        alpha: 10.,
        rd: 0.1,
    };

    model.simulated_anealing(dataset, 1e5, 10000);

    println!("{:?}", model);
}

#[test]
fn test_surfacepotential_calc() {
    let model = SurfacePotentialModel {
        scale: 171455.1768651102,
        tox: 5e-8,
        na: 1.2531591504771211e17,
        lambda: 0.019574827818568466,
        vfbc: -1.5850173370816367,
        theta: 0.004944312000295522,
        delta: 0.5289369284239606,
        alpha: 16.568182413277864,
        rd: 0.07734977009721601,
    };
    // for _vgs in 0..100 {
    for _vds in 0..20 {
        let vgs = 100.0 as f64 / 10.0;
        let vds = _vds as f64 / 1.0;

        let xxx = f64::sqrt(
            2.0 * SurfacePotentialModel::Q * SurfacePotentialModel::EPS_SIC_CM2 * model.na
                / SurfacePotentialModel::EPS_OX_CM2
                / (model.tox * 1e2),
        );
        let ggg = xxx / SurfacePotentialModel::PHI_T;
        let phi = model.psi_calc(vgs, vds, 2.0 * 1.38054e-23 * 300.0 / 1.60210e-19, ggg);
        println!("vgs: {}, vds: {}, phi: {}", vgs, vds, phi);
    }
    // }
}

#[test]
fn test_make_grid() {
    use std::fs::File;
    use std::io::BufWriter;
    use std::io::Write;
    use std::path::Path;

    // Score: 0.1607046045577904
    let model = SurfacePotentialModel { scale: 171324.3635846557, tox: 5e-8, na: 1.2534396079784723e17, lambda: 0.019669796078470445, vfbc: -1.5301638574847534, theta: 0.005303409021004406, delta: 0.537279227992155, alpha: 15.468242796282272, rd: 0.07671305067487988 };
    // let model = Level1::new(0.83, 0.022, 5.99);
    let grid = model.make_grid(
        (0..20)
            .step_by(2)
            .into_iter()
            .map(|x| x as f64 / 1.0)
            .collect::<Vec<_>>(),
        (0..300)
            .step_by(10)
            .into_iter()
            .map(|x| x as f64 / 10.0)
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
fn test_surface_potential() {
    let mosfet = SurfacePotentialModel {
        scale: 6704999.096087718,
        tox: 5e-8,
        na: 1.2483487166088576e17,
        lambda: 1e2,
        vfbc: -1.564917913861022,
        theta: 0.006559247583163868,
        delta: 0.5357,
        alpha: -29.455399,
        rd: 0.1,
    };

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
        &[12.0, 31.6],
        &[12.0, 300.0],
    ]);

    let ids = mosfet.tfun(&vgs_vds);
    ids.print();
}
