pub trait mosfet {
    fn fermi_potential() -> f32;
    fn length() -> f32;
    fn width() -> f32;
    fn flat_band_voltage() -> f32;
}
