use anyhow::Result;
use csv;
use std::collections::HashMap;
use std::error::Error;
use std::fs;

pub trait DataSet {
    // fn into_tensor(self: &Self) -> Tensor;
    fn get(&self, name: &str) -> Option<&Vec<f32>>;
    // fn get_column_name(&self) -> Vec<String>;
}

#[derive(Debug)]
#[allow(non_camel_case_types)]
pub struct IV_measurements {
    pub vgs: Vec<f32>,
    pub ids: Vec<f32>,
    pub vds: Vec<f32>,
}

#[derive(Debug)]
pub struct ScaledMeasurements {
    pub minimums: HashMap<String, f32>,
    pub maximums: HashMap<String, f32>,
    pub measurements: HashMap<String, Vec<f32>>,
}

impl DataSet for ScaledMeasurements {
    fn get(&self, name: &str) -> Option<&Vec<f32>> {
        return self.measurements.get(name);
    }
}

pub fn min_max_scaling(measurements: &IV_measurements) -> ScaledMeasurements {
    let mut minimums = HashMap::<String, f32>::new();
    let mut maximums = HashMap::<String, f32>::new();
    minimums.insert(
        "IDS".to_owned(),
        measurements
            .ids
            .iter()
            .fold(f32::MAX, |m, x| f32::min(m, *x)),
    );
    maximums.insert(
        "IDS".to_owned(),
        measurements
            .ids
            .iter()
            .fold(f32::MIN, |m, x| f32::max(m, *x)),
    );
    minimums.insert(
        "VDS".to_owned(),
        measurements
            .vds
            .iter()
            .fold(f32::MAX, |m, x| f32::min(m, *x)),
    );
    maximums.insert(
        "VDS".to_owned(),
        measurements
            .vds
            .iter()
            .fold(f32::MIN, |m, x| f32::max(m, *x)),
    );
    minimums.insert(
        "VGS".to_owned(),
        measurements
            .vgs
            .iter()
            .fold(f32::MAX, |m, x| f32::min(m, *x)),
    );
    maximums.insert(
        "VGS".to_owned(),
        measurements
            .vgs
            .iter()
            .fold(f32::MIN, |m, x| f32::max(m, *x)),
    );

    let scaled_ids = measurements
        .ids
        .iter()
        .map(|x| {
            (*x - minimums.get("IDS").unwrap_or(&0.))
                / (maximums.get("IDS").unwrap_or(&1.) - minimums.get("IDS").unwrap_or(&0.))
        })
        .collect();
    let scaled_vds = measurements
        .vds
        .iter()
        .map(|x| {
            (*x - minimums.get("VDS").unwrap_or(&0.))
                / (maximums.get("VDS").unwrap_or(&1.) - minimums.get("VDS").unwrap_or(&0.))
        })
        .collect();
    let scaled_vgs = measurements
        .vgs
        .iter()
        .map(|x| {
            (*x - minimums.get("VGS").unwrap_or(&0.))
                / (maximums.get("VGS").unwrap_or(&1.) - minimums.get("VGS").unwrap_or(&0.))
        })
        .collect();

    let mut measurements = HashMap::<String, Vec<f32>>::new();
    measurements.insert("VGS".to_owned(), scaled_vgs);
    measurements.insert("VDS".to_owned(), scaled_vds);
    measurements.insert("IDS".to_owned(), scaled_ids);

    ScaledMeasurements {
        minimums: minimums,
        maximums: maximums,
        measurements: measurements,
    }
}

// (vgs,vds,ids)の順
pub fn scaling(
    measurements: &IV_measurements,
    max: (f32, f32, f32),
    min: (f32, f32, f32),
) -> ScaledMeasurements {
    let mut minimums = HashMap::<String, f32>::new();
    let mut maximums = HashMap::<String, f32>::new();
    minimums.insert("IDS".to_owned(), min.2);
    maximums.insert("IDS".to_owned(), max.2);
    minimums.insert("VDS".to_owned(), min.1);
    maximums.insert("VDS".to_owned(), max.1);
    minimums.insert("VGS".to_owned(), min.0);
    maximums.insert("VGS".to_owned(), max.0);

    let scaled_ids = measurements
        .ids
        .iter()
        .map(|x| {
            (*x - minimums.get("IDS").unwrap_or(&0.))
                / (maximums.get("IDS").unwrap_or(&1.) - minimums.get("IDS").unwrap_or(&0.))
        })
        .collect();
    let scaled_vds = measurements
        .vds
        .iter()
        .map(|x| {
            (*x - minimums.get("VDS").unwrap_or(&0.))
                / (maximums.get("VDS").unwrap_or(&1.) - minimums.get("VDS").unwrap_or(&0.))
        })
        .collect();
    let scaled_vgs = measurements
        .vgs
        .iter()
        .map(|x| {
            (*x - minimums.get("VGS").unwrap_or(&0.))
                / (maximums.get("VGS").unwrap_or(&1.) - minimums.get("VGS").unwrap_or(&0.))
        })
        .collect();

    let mut measurements = HashMap::<String, Vec<f32>>::new();
    measurements.insert("VGS".to_owned(), scaled_vgs);
    measurements.insert("VDS".to_owned(), scaled_vds);
    measurements.insert("IDS".to_owned(), scaled_ids);

    ScaledMeasurements {
        minimums: minimums,
        maximums: maximums,
        measurements: measurements,
    }
}

pub fn read_csv(file_path: String) -> Result<IV_measurements, Box<dyn Error>> {
    let mut vgs = Vec::<f32>::new();
    let mut ids = Vec::<f32>::new();
    let mut vds = Vec::<f32>::new();

    let csv_text = fs::read_to_string(file_path)?;
    let mut rdr = csv::Reader::from_reader(csv_text.as_bytes());
    for result in rdr.records() {
        let record: (f32, f32, f32) = result?.deserialize(None)?;
        vgs.push(record.0);
        vds.push(record.2);
        ids.push(record.1);
    }

    Ok(IV_measurements {
        vgs: vgs,
        ids: ids,
        vds: vds,
    })
}

#[test]
fn test_load_data() {
    use tch::Tensor;
    let iv_measurement = read_csv("./data/integral_train.csv".to_string()).unwrap();
    println!("{:?}", iv_measurement);

    Tensor::from_slice(iv_measurement.ids.as_slice()).print();
}
