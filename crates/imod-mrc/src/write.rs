use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

use binrw::BinWrite;

use imod_core::ImodError;

use crate::MrcHeader;

/// Writer for MRC image files.
pub struct MrcWriter {
    writer: BufWriter<File>,
    header: MrcHeader,
    slices_written: usize,
}

impl MrcWriter {
    /// Create a new MRC file with the given header.
    pub fn create(path: impl AsRef<Path>, header: MrcHeader) -> Result<Self, ImodError> {
        let file = File::create(path.as_ref())?;
        let mut writer = BufWriter::new(file);

        // Write header (will be rewritten on close to update statistics)
        header
            .write_le(&mut writer)
            .map_err(|e| ImodError::InvalidData(e.to_string()))?;

        // Write extended header (zeros) if next > 0
        let ext_size = header.next.max(0) as usize;
        if ext_size > 0 {
            let zeros = vec![0u8; ext_size];
            writer.write_all(&zeros)?;
        }

        Ok(Self {
            writer,
            header,
            slices_written: 0,
        })
    }

    /// Write a raw slice (bytes must match the expected slice size).
    pub fn write_slice_raw(&mut self, data: &[u8]) -> Result<(), ImodError> {
        let expected = self.header.slice_size_bytes();
        if data.len() != expected {
            return Err(ImodError::InvalidData(format!(
                "slice data size {} doesn't match expected {expected}",
                data.len()
            )));
        }
        self.writer.write_all(data)?;
        self.slices_written += 1;
        Ok(())
    }

    /// Write a slice from f32 data, converting to the file's native mode.
    pub fn write_slice_f32(&mut self, data: &[f32]) -> Result<(), ImodError> {
        let mode = self.header.data_mode().ok_or(ImodError::UnsupportedMode(self.header.mode))?;
        let nx = self.header.nx as usize;
        let ny = self.header.ny as usize;
        let npix = nx * ny;

        match mode {
            imod_core::MrcMode::Byte => {
                if data.len() != npix {
                    return Err(ImodError::InvalidData("data length mismatch".into()));
                }
                let bytes: Vec<u8> = data.iter().map(|&v| v.clamp(0.0, 255.0) as u8).collect();
                self.write_slice_raw(&bytes)
            }
            imod_core::MrcMode::Short => {
                if data.len() != npix {
                    return Err(ImodError::InvalidData("data length mismatch".into()));
                }
                let mut buf = Vec::with_capacity(npix * 2);
                for &v in data {
                    buf.extend_from_slice(&(v as i16).to_le_bytes());
                }
                self.write_slice_raw(&buf)
            }
            imod_core::MrcMode::UShort => {
                if data.len() != npix {
                    return Err(ImodError::InvalidData("data length mismatch".into()));
                }
                let mut buf = Vec::with_capacity(npix * 2);
                for &v in data {
                    buf.extend_from_slice(&(v.clamp(0.0, 65535.0) as u16).to_le_bytes());
                }
                self.write_slice_raw(&buf)
            }
            imod_core::MrcMode::Float => {
                if data.len() != npix {
                    return Err(ImodError::InvalidData("data length mismatch".into()));
                }
                let mut buf = Vec::with_capacity(npix * 4);
                for &v in data {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                self.write_slice_raw(&buf)
            }
            _ => Err(ImodError::UnsupportedMode(self.header.mode)),
        }
    }

    /// Finalize the file: update header statistics and rewrite the header.
    pub fn finish(mut self, amin: f32, amax: f32, amean: f32) -> Result<(), ImodError> {
        self.header.amin = amin;
        self.header.amax = amax;
        self.header.amean = amean;

        // Seek back to start and rewrite header
        self.writer.seek(SeekFrom::Start(0))?;
        self.header
            .write_le(&mut self.writer)
            .map_err(|e| ImodError::InvalidData(e.to_string()))?;
        self.writer.flush()?;
        Ok(())
    }
}
