use anyhow::Result;
use csv;
use serde::Deserialize;
use std::error::Error;
use std::fs;
use tch::Tensor;

pub trait DataSet<T> {
    fn into_tensor(self: &Self) -> Tensor;
}

#[derive(Debug, Deserialize)]
struct IV_measurement {
    VGS: f32,
    IDS: f32,
    VDS: f32,
}

#[derive(Debug)]
pub struct IV_measurements {
    pub VGS: Vec<f32>,
    pub IDS: Vec<f32>,
    pub VDS: Vec<f32>,
}

impl IV_measurements {
    pub fn min_max_scaling(&mut self) {
        let minimum = self.IDS.iter().fold(f32::MAX, |m, x| f32::min(m, *x));
        let maximum = self.IDS.iter().fold(f32::MIN, |m, x| f32::max(m, *x));
        self.IDS = self
            .IDS
            .iter()
            .map(|x| (*x - minimum) / (maximum - minimum))
            .collect();

        let minimum = self.VDS.iter().fold(f32::MAX, |m, x| f32::min(m, *x));
        let maximum = self.VDS.iter().fold(f32::MIN, |m, x| f32::max(m, *x));
        self.VDS = self
            .VDS
            .iter()
            .map(|x| (*x - minimum) / (maximum - minimum))
            .collect();

        let minimum = self.VGS.iter().fold(f32::MAX, |m, x| f32::min(m, *x));
        let maximum = self.VGS.iter().fold(f32::MIN, |m, x| f32::max(m, *x));
        self.VGS = self
            .VGS
            .iter()
            .map(|x| (*x - minimum) / (maximum - minimum))
            .collect();
    }
}

impl DataSet<f32> for IV_measurement {
    fn into_tensor(self: &Self) -> Tensor {
        Tensor::cat(
            &[
                std::convert::Into::<Tensor>::into(self.VGS),
                std::convert::Into::<Tensor>::into(self.VDS),
                self.IDS.into(),
            ],
            0,
        )
    }
}

pub fn read_csv(file_path: String) -> Result<IV_measurements, Box<dyn Error>> {
    let mut vgs = Vec::<f32>::new();
    let mut ids = Vec::<f32>::new();
    let mut vds = Vec::<f32>::new();

    let csv_text = fs::read_to_string(file_path)?;
    let mut rdr = csv::Reader::from_reader(csv_text.as_bytes());
    for result in rdr.records() {
        let record: IV_measurement = result?.deserialize(None)?;
        vgs.push(record.VGS);
        vds.push(record.VDS);
        ids.push(record.IDS);
    }

    Ok(IV_measurements {
        VGS: vgs,
        IDS: ids,
        VDS: vds,
    })
}

#[test]
fn test() {
    let iv_measurement = read_csv("./data/25_train.csv".to_string()).unwrap();
    println!("{:?}", iv_measurement);

    Tensor::from_slice(iv_measurement.IDS.as_slice()).print();
}
