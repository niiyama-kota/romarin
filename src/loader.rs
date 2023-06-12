use anyhow::Result;
use csv;
use serde::Deserialize;
use std::error::Error;
use std::fs;
use tch::{
    nn,
    nn::OptimizerConfig,
    nn::{Func, Module},
    Device, Tensor,
};

pub trait DataSet<T> {
    fn into_tensor(self: &Self) -> Tensor;
}

#[derive(Debug, Deserialize)]
struct IV_measurement {
    VGS: f64,
    IDS: f64,
    VDS: f64,
}

#[derive(Debug)]
pub struct IV_measurements {
    pub VGS: Vec<f64>,
    pub IDS: Vec<f64>,
    pub VDS: Vec<f64>,
}

impl DataSet<f64> for IV_measurement {
    fn into_tensor(self: &Self) -> Tensor {
        Tensor::cat(
            &[
                std::convert::Into::<Tensor>::into(self.VDS),
                std::convert::Into::<Tensor>::into(self.VGS),
                self.IDS.into(),
            ],
            0,
        )
    }
}

pub fn read_csv(file_path: String) -> Result<IV_measurements, Box<dyn Error>> {
    let mut vgs = Vec::<f64>::new();
    let mut ids = Vec::<f64>::new();
    let mut vds = Vec::<f64>::new();

    let csv_text = fs::read_to_string(file_path)?;
    let mut rdr = csv::Reader::from_reader(csv_text.as_bytes());
    for result in rdr.records() {
        let record: IV_measurement = result?.deserialize(None)?;
        vgs.push(record.VGS);
        vds.push(record.VDS);
        ids.push(record.IDS);
    }

    Ok(IV_measurements { VGS: vgs, IDS: ids, VDS: vds })
}

#[test]
fn test() {
    let iv_measurement = read_csv("./data/integral.csv".to_string()).unwrap();
    println!("{:?}", iv_measurement)
}
