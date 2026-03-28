use std::io;

#[derive(Debug, thiserror::Error)]
pub enum ImodError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    #[error("invalid MRC header: {0}")]
    InvalidHeader(String),

    #[error("unsupported MRC mode: {0}")]
    UnsupportedMode(i32),

    #[error("invalid data: {0}")]
    InvalidData(String),

    #[error("parse error: {0}")]
    Parse(String),
}
