pub(crate) const NI: f32 = 1.5 * 1e10; // 300Kでの真正シリコンの自由電子数[個/cm^3]
pub(crate) const K: f32 = 1.38 * 1e-23; // ボルツマン定数[J/K]
pub(crate) const T: f32 = 300.0; //絶対温度[K]
pub(crate) const Q: f32 = 1.6 * 1e-19; // 素電荷[C]
pub(crate) const PHI_T: f32 = K * T / Q; // thermal voltage[V]
// pub(crate) const E0: f32 = 8.854 * 1e-14; // 真空の誘電率[F/cm]
pub(crate) const ESI: f32 = 1.0359e-12; // シリコンの誘電率[F/cm]
// pub(crate) const EOX: f32 = 3.9 * E0; // シリコン酸化膜の誘電率[F/cm]
// pub(crate) const NSS: f32 = 3.0 * 1e10; // 単位面積当たりのゲート酸化膜中とシリコン基板との海面におけるトラップによる正電荷の個数[個/cm^2]
// pub(crate) const QSS: f32 = NSS * Q; // 単位面積当たりのゲート酸化膜中とシリコン基板との海面におけるトラップによる正電荷
