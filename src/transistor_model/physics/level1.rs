#[derive(Clone, Debug)]
pub struct Level1 {
    // 厚み
    w: f32,
    // 幅
    l: f32,
    // 酸化皮膜の厚み
    tox: f32,
    // n型不純物濃度
    ng: f32,
    // p型不純物濃度
    np: f32,
}

impl Level1 {
    pub fn new(w: f32, l: f32, tox: f32, ng: f32, np: f32) -> Self {
        Self {
            w: w,
            l: l,
            tox: tox,
            ng: ng,
            np: np,
        }
    }

    pub fn model<'a>(&'a self) -> impl Fn(f32, f32) -> f32 + 'a {
        |vgs: f32, vds:f32| -> f32 {
            let ids = if vds < vgs - vth {

            } else {
                
            }

            ids
        }
    }
}
