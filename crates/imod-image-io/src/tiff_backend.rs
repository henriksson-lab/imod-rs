use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use imod_core::ImodError;
use imod_slice::Slice;
use tiff::decoder::{Decoder, DecodingResult};
use tiff::ColorType;

use crate::format::{ImageFile, ImageInfo};

/// TIFF file backend for the ImageFile trait.
///
/// Supports grayscale TIFF files with 8-bit, 16-bit, and 32-bit float samples.
/// Multi-page TIFFs are treated as Z stacks (one page per slice).
pub struct TiffImageFile {
    path: std::path::PathBuf,
    info: ImageInfo,
}

/// Count pages in a TIFF file by iterating through all IFDs.
fn count_pages(reader: &mut Decoder<BufReader<File>>) -> usize {
    let mut count = 1;
    while reader.next_image().is_ok() {
        count += 1;
    }
    count
}

/// Describe the sample type for ImageInfo.mode_name.
fn mode_name_for(color_type: ColorType) -> String {
    match color_type {
        ColorType::Gray(8) => "Gray8".to_string(),
        ColorType::Gray(16) => "Gray16".to_string(),
        ColorType::Gray(32) => "Gray32f".to_string(),
        ColorType::Gray(bits) => format!("Gray{bits}"),
        other => format!("{other:?}"),
    }
}

impl TiffImageFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, ImodError> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        let buf = BufReader::new(file);
        let mut decoder = Decoder::new(buf)
            .map_err(|e| ImodError::InvalidData(format!("TIFF decode error: {e}")))?;

        let (nx, ny) = decoder.dimensions()
            .map_err(|e| ImodError::InvalidData(format!("TIFF dimensions: {e}")))?;
        let color_type = decoder.colortype()
            .map_err(|e| ImodError::InvalidData(format!("TIFF color type: {e}")))?;

        let mode_name = mode_name_for(color_type);

        // Read the first page to get min/max/mean statistics.
        let first_data = decode_page_f32(&mut decoder)?;

        let mut min = f32::MAX;
        let mut max = f32::MIN;
        let mut sum: f64 = 0.0;
        for &v in &first_data {
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
            sum += v as f64;
        }
        let mean = if first_data.is_empty() {
            0.0
        } else {
            (sum / first_data.len() as f64) as f32
        };

        // Count remaining pages.
        let nz = count_pages(&mut decoder);

        let info = ImageInfo {
            nx: nx as usize,
            ny: ny as usize,
            nz,
            pixel_size: [1.0, 1.0, 1.0],
            mode_name,
            min,
            max,
            mean,
        };

        Ok(Self { path, info })
    }
}

/// Decode the current page of a TIFF decoder into f32 pixel values.
fn decode_page_f32(decoder: &mut Decoder<BufReader<File>>) -> Result<Vec<f32>, ImodError> {
    let result = decoder
        .read_image()
        .map_err(|e| ImodError::InvalidData(format!("TIFF read error: {e}")))?;
    match result {
        DecodingResult::U8(buf) => Ok(buf.into_iter().map(|v| v as f32).collect()),
        DecodingResult::U16(buf) => Ok(buf.into_iter().map(|v| v as f32).collect()),
        DecodingResult::F32(buf) => Ok(buf),
        DecodingResult::U32(buf) => Ok(buf.into_iter().map(|v| v as f32).collect()),
        DecodingResult::U64(buf) => Ok(buf.into_iter().map(|v| v as f32).collect()),
        DecodingResult::F64(buf) => Ok(buf.into_iter().map(|v| v as f32).collect()),
        DecodingResult::I8(buf) => Ok(buf.into_iter().map(|v| v as f32).collect()),
        DecodingResult::I16(buf) => Ok(buf.into_iter().map(|v| v as f32).collect()),
        DecodingResult::I32(buf) => Ok(buf.into_iter().map(|v| v as f32).collect()),
        DecodingResult::I64(buf) => Ok(buf.into_iter().map(|v| v as f32).collect()),
    }
}

impl ImageFile for TiffImageFile {
    fn info(&self) -> &ImageInfo {
        &self.info
    }

    fn read_slice(&mut self, z: usize) -> Result<Slice, ImodError> {
        if z >= self.info.nz {
            return Err(ImodError::InvalidData(format!(
                "slice index {z} out of range (nz={})",
                self.info.nz
            )));
        }

        // Re-open the file and seek to the requested page.
        let file = File::open(&self.path)?;
        let buf = BufReader::new(file);
        let mut decoder = Decoder::new(buf)
            .map_err(|e| ImodError::InvalidData(format!("TIFF decode error: {e}")))?;

        for _ in 0..z {
            decoder
                .next_image()
                .map_err(|e| ImodError::InvalidData(format!("TIFF seek page: {e}")))?;
        }

        let data = decode_page_f32(&mut decoder)?;
        Ok(Slice::from_data(self.info.nx, self.info.ny, data))
    }
}
