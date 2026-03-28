use std::io;

/// Unified error type used across all IMOD crates.
#[derive(Debug, thiserror::Error)]
pub enum ImodError {
    /// An underlying I/O error (file not found, permission denied, etc.).
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// The MRC file header failed validation (bad magic, corrupt fields).
    #[error("invalid MRC header: {0}")]
    InvalidHeader(String),

    /// The MRC data mode value is not recognised.
    #[error("unsupported MRC mode: {0}")]
    UnsupportedMode(i32),

    /// Generic invalid-data error (unexpected values, truncated files, etc.).
    #[error("invalid data: {0}")]
    InvalidData(String),

    /// A text parsing error (bad number format, malformed input).
    #[error("parse error: {0}")]
    Parse(String),
}
