//! Runtime selection for the experimental Rust-native image and FFT backends.
//!
//! The IMOD-compatible implementation remains the process default.  These
//! selectors are deliberately small and explicit: a requested experimental
//! backend is never silently redirected to the parity implementation.

use std::sync::OnceLock;

/// Runtime choice for libtiff-backed TIFF operations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TiffBackend {
    /// Existing source-compatible libtiff boundary.
    Parity,
    /// Experimental Rust `tiff` crate implementation.
    Rust,
}

/// Runtime choice for mrc2tif JPEG/PNG encoding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Mrc2TifEncoder {
    /// Retained spelling for callers that requested the former default.  It
    /// now selects the Rust encoder because Qt is no longer linked.
    Parity,
    /// Experimental Rust image encoder.
    Rust,
}

/// Runtime choice for translated FFT execution boundaries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FftBackend {
    /// Existing translated IMOD FFT implementation.
    Parity,
    /// Experimental RustFFT implementation.
    Rustfft,
}

fn value(name: &str) -> Result<Option<String>, String> {
    match std::env::var(name) {
        Ok(value) => Ok(Some(value.trim().to_ascii_lowercase())),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(format!(
            "ERROR: Rust-native backend - {name} must contain ASCII backend name"
        )),
    }
}

/// Parses `IMOD_RS_TIFF_BACKEND`; unset selects the parity boundary.
pub fn tiff_backend() -> Result<TiffBackend, String> {
    static BACKEND: OnceLock<Result<TiffBackend, String>> = OnceLock::new();
    BACKEND
        .get_or_init(|| match value("IMOD_RS_TIFF_BACKEND") {
            Ok(value) => match value.as_deref() {
                None | Some("parity") => Ok(TiffBackend::Parity),
                Some("rust") => Ok(TiffBackend::Rust),
                Some(_) => Err(
                    "ERROR: Rust-native backend - IMOD_RS_TIFF_BACKEND must be parity or rust"
                        .into(),
                ),
            },
            Err(error) => Err(error),
        })
        .clone()
}

/// Parses `IMOD_RS_MRC2TIF_ENCODER`; unset selects the Rust encoder.
pub fn mrc2tif_encoder() -> Result<Mrc2TifEncoder, String> {
    static BACKEND: OnceLock<Result<Mrc2TifEncoder, String>> = OnceLock::new();
    BACKEND
        .get_or_init(|| match value("IMOD_RS_MRC2TIF_ENCODER") {
            Ok(value) => match value.as_deref() {
                None => Ok(Mrc2TifEncoder::Rust),
                Some("parity") => Ok(Mrc2TifEncoder::Rust),
                Some("rust") => Ok(Mrc2TifEncoder::Rust),
                Some(_) => Err(
                    "ERROR: Rust-native backend - IMOD_RS_MRC2TIF_ENCODER must be parity or rust"
                        .into(),
                ),
            },
            Err(error) => Err(error),
        })
        .clone()
}

/// Parses `IMOD_RS_FFT_BACKEND`; unset selects the translated IMOD FFT path.
pub fn fft_backend() -> Result<FftBackend, String> {
    static BACKEND: OnceLock<Result<FftBackend, String>> = OnceLock::new();
    BACKEND
        .get_or_init(|| match value("IMOD_RS_FFT_BACKEND") {
            Ok(value) => match value.as_deref() {
                None | Some("parity") => Ok(FftBackend::Parity),
                Some("rustfft") => Ok(FftBackend::Rustfft),
                Some(_) => Err(
                    "ERROR: Rust-native backend - IMOD_RS_FFT_BACKEND must be parity or rustfft"
                        .into(),
                ),
            },
            Err(error) => Err(error),
        })
        .clone()
}

#[cfg(test)]
mod tests {
    use super::{
        FftBackend, Mrc2TifEncoder, TiffBackend, fft_backend, mrc2tif_encoder, tiff_backend,
    };

    #[test]
    fn unset_backend_variables_select_the_current_defaults() {
        // The test process does not alter environment state because libtest
        // runs in one process; normal test invocations explicitly clear these
        // variables in their child commands instead.
        if std::env::var_os("IMOD_RS_TIFF_BACKEND").is_none() {
            assert_eq!(tiff_backend(), Ok(TiffBackend::Parity));
        }
        if std::env::var_os("IMOD_RS_MRC2TIF_ENCODER").is_none() {
            assert_eq!(mrc2tif_encoder(), Ok(Mrc2TifEncoder::Rust));
        }
        if std::env::var_os("IMOD_RS_FFT_BACKEND").is_none() {
            assert_eq!(fft_backend(), Ok(FftBackend::Parity));
        }
    }
}
