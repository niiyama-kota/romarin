use tch::nn;

#[derive(PartialEq, Debug)]
pub enum Node {
    ID(String),
    INT(i32),
    FLOAT(f32),
}
