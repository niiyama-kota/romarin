#[derive(PartialEq, Debug)]
pub enum Token {
    ID(String),
    INT(String),
    FLOAT(String),
}
