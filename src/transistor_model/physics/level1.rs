use super::constant;
use tch::Tensor;

#[derive(Debug, Copy, Clone, Hash)]
pub enum Mostype {
    Pmos,
    Nmos,
}

#[derive(Clone, Debug)]
pub struct Level1 {
    // Mosfet Type
    ty: Mostype,
    // ゲート幅[um]
    w: f32,
    // ゲート長[um]
    l: f32,
    // 酸化皮膜の厚み
    tox: f32,
    // n型不純物濃度
    ng: f32,
    // p型不純物濃度
    np: f32,
    // キャリアの移動度
    mu: f32,
}

impl Level1 {
    pub fn new(norp: Mostype, w: f32, l: f32, tox: f32, ng: f32, np: f32, mu: f32) -> Self {
        Self {
            ty: norp,
            w: w,
            l: l,
            tox: tox,
            ng: ng,
            np: np,
            mu: mu,
        }
    }

    pub fn model<'a>(&'a self) -> impl Fn(f32, f32) -> f32 + 'a {
        |vgs: f32, vds: f32| -> f32 {
            let cox = constant::EOX / self.tox; // シリコン酸化膜の単位面積あたりのキャパシタンス
            println!("DEBUG COX: {}", cox);
            let phi_g = -constant::PHI_T * f32::log10(self.ng / constant::NI); // ゲートポリシリコンのフェルミポテンシャル
            let phi_f = constant::PHI_T * f32::log10(self.np / constant::NI); // シリコン表面のフェルミポテンシャル
            let phi_ms = phi_g - phi_f; //拡散電位
            let vfb = phi_ms - constant::QSS / cox;
            let vth = vfb + 2.0 * phi_f;
            let ids: f32 = -if vgs <= vth {
                0.0
            } else if vds < vgs - vth {
                self.w / self.l * cox * self.mu * ((vgs - vth) * vds - 0.5 * vds * vds)
            } else {
                self.w / self.l * cox * self.mu * ((vgs - vth) * (vgs - vth) - 0.5 * vds * vds)
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

    fn kp(&self) -> f32 {
        match self.ty {
            Mostype::Pmos => 8.632e-6,
            Mostype::Nmos => 2.0718e-5,
        }
    }
}

#[test]
fn test_level1() {
    let mosfet = Level1::new(
        Mostype::Nmos,
        20.0,
        0.25,
        1e-7,
        1.0 * 1e20,
        2.5 * 1e17,
        150.0,
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
        &[12.0, 31.6],
    ]);

    let ids = mosfet.tfun(&vgs_vds);
    ids.print();
}
