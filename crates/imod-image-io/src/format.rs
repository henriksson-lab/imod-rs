use std::path::Path;

use imod_core::ImodError;
use imod_slice::Slice;

/// Information about an image file.
#[derive(Debug, Clone)]
pub struct ImageInfo {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub pixel_size: [f32; 3],
    pub mode_name: String,
    pub min: f32,
    pub max: f32,
    pub mean: f32,
}

/// Trait for image file backends (MRC, TIFF, etc.).
pub trait ImageFile {
    /// Get image dimensions and metadata.
    fn info(&self) -> &ImageInfo;

    /// Read a single Z slice as f32.
    fn read_slice(&mut self, z: usize) -> Result<Slice, ImodError>;

    /// Read the full stack as a vector of slices.
    fn read_all(&mut self) -> Result<Vec<Slice>, ImodError> {
        let nz = self.info().nz;
        let mut slices = Vec::with_capacity(nz);
        for z in 0..nz {
            slices.push(self.read_slice(z)?);
        }
        Ok(slices)
    }
}

/// Detect image format from file extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    Mrc,
    Tiff,
    Jpeg,
    Hdf5,
    Unknown,
}

impl ImageFormat {
    pub fn from_path(path: impl AsRef<Path>) -> Self {
        match path
            .as_ref()
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase())
            .as_deref()
        {
            Some("mrc") | Some("st") | Some("ali") | Some("rec") | Some("map") | Some("preali") => {
                Self::Mrc
            }
            Some("tif") | Some("tiff") => Self::Tiff,
            Some("jpg") | Some("jpeg") => Self::Jpeg,
            Some("hdf") | Some("h5") | Some("hdf5") => Self::Hdf5,
            _ => Self::Unknown,
        }
    }
}

/// Open an image file, auto-detecting the format.
/// Currently only MRC is supported.
pub fn open_image(path: impl AsRef<Path>) -> Result<Box<dyn ImageFile>, ImodError> {
    let format = ImageFormat::from_path(path.as_ref());
    match format {
        ImageFormat::Mrc | ImageFormat::Unknown => {
            // Try MRC for unknown formats too (IMOD convention)
            Ok(Box::new(crate::mrc_backend::MrcImageFile::open(path)?))
        }
        _ => Err(ImodError::InvalidData(format!(
            "unsupported image format: {:?}",
            format
        ))),
    }
}
