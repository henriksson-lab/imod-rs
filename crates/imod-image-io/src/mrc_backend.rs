use std::path::Path;

use imod_core::ImodError;
use imod_mrc::MrcReader;
use imod_slice::Slice;

use crate::format::{ImageFile, ImageInfo};

/// MRC file backend for the ImageFile trait.
pub struct MrcImageFile {
    reader: MrcReader,
    info: ImageInfo,
}

impl MrcImageFile {
    /// Open an MRC file and prepare it for slice-by-slice reading.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, ImodError> {
        let reader = MrcReader::open(path)?;
        let h = reader.header();
        let mode_name = match h.data_mode() {
            Some(m) => format!("{:?}", m),
            None => format!("unknown({})", h.mode),
        };
        let info = ImageInfo {
            nx: h.nx as usize,
            ny: h.ny as usize,
            nz: h.nz as usize,
            pixel_size: [h.pixel_size_x(), h.pixel_size_y(), h.pixel_size_z()],
            mode_name,
            min: h.amin,
            max: h.amax,
            mean: h.amean,
        };
        Ok(Self { reader, info })
    }
}

impl ImageFile for MrcImageFile {
    fn info(&self) -> &ImageInfo {
        &self.info
    }

    fn read_slice(&mut self, z: usize) -> Result<Slice, ImodError> {
        let data = self.reader.read_slice_f32(z)?;
        Ok(Slice::from_data(self.info.nx, self.info.ny, data))
    }
}
