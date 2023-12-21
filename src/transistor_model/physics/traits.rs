pub trait mosfet {
    fn fermi_potential() -> f32;
    fn length() -> f32;
    fn width() -> f32;
    fn flat_band_voltage() -> f32;
}

pub trait Param {
    fn val(&self) -> f32;
    fn fix() -> bool;
    fn default() -> f32;
}
