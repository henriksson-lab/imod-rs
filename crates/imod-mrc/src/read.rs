use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use binrw::BinRead;

use imod_core::ImodError;

use crate::MrcHeader;

/// Reader for MRC image files.
///
/// Automatically detects byte-swapped (big-endian) files and converts data
/// to native endianness when reading.
pub struct MrcReader {
    reader: BufReader<File>,
    header: MrcHeader,
    ext_header: Vec<u8>,
    swapped: bool,
}

impl MrcReader {
    /// Open an MRC file and read its header.
    ///
    /// The reader first tries little-endian. If the header looks swapped
    /// (machine stamp is big-endian, or nx/ny/nz are unreasonable), it
    /// re-reads as big-endian and byte-swaps the header fields.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, ImodError> {
        let file = File::open(path.as_ref())?;
        let mut reader = BufReader::new(file);

        // Try little-endian first
        let header = MrcHeader::read_le(&mut reader)
            .map_err(|e| ImodError::InvalidHeader(e.to_string()))?;

        let swapped = header.is_swapped() || !Self::dimensions_reasonable(&header);

        let header = if swapped {
            // Re-read as big-endian
            reader.seek(SeekFrom::Start(0))?;
            let h = MrcHeader::read_be(&mut reader)
                .map_err(|e| ImodError::InvalidHeader(e.to_string()))?;
            // Validate after swap
            if !Self::dimensions_reasonable(&h) {
                return Err(ImodError::InvalidHeader(
                    "unreasonable dimensions even after byte-swap".into(),
                ));
            }
            h
        } else {
            header
        };

        // Read extended header if present
        let ext_size = header.next.max(0) as usize;
        let mut ext_header = vec![0u8; ext_size];
        if ext_size > 0 {
            reader.read_exact(&mut ext_header)?;
        }

        Ok(Self {
            reader,
            header,
            ext_header,
            swapped,
        })
    }

    /// Whether the file was detected as byte-swapped.
    pub fn is_swapped(&self) -> bool {
        self.swapped
    }

    pub fn header(&self) -> &MrcHeader {
        &self.header
    }

    pub fn ext_header(&self) -> &[u8] {
        &self.ext_header
    }

    /// Read raw bytes for a single Z slice.
    pub fn read_slice_raw(&mut self, z: usize) -> Result<Vec<u8>, ImodError> {
        let nz = self.header.nz as usize;
        if z >= nz {
            return Err(ImodError::InvalidData(format!(
                "slice index {z} out of range (nz={nz})"
            )));
        }
        let slice_bytes = self.header.slice_size_bytes();
        let offset = self.header.data_offset() + (z as u64 * slice_bytes as u64);
        self.reader.seek(SeekFrom::Start(offset))?;

        let mut buf = vec![0u8; slice_bytes];
        self.reader.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Read a Z slice as f32 values (converts from the file's native mode).
    /// Handles byte-swapped files automatically.
    pub fn read_slice_f32(&mut self, z: usize) -> Result<Vec<f32>, ImodError> {
        let raw = self.read_slice_raw(z)?;
        let mode = self
            .header
            .data_mode()
            .ok_or(ImodError::UnsupportedMode(self.header.mode))?;
        let nx = self.header.nx as usize;
        let ny = self.header.ny as usize;
        let npix = nx * ny;

        let data = match mode {
            imod_core::MrcMode::Byte => raw.iter().map(|&b| b as f32).collect(),
            imod_core::MrcMode::Short => {
                let conv = if self.swapped {
                    i16::from_be_bytes as fn([u8; 2]) -> i16
                } else {
                    i16::from_le_bytes
                };
                raw.chunks_exact(2)
                    .map(|c| conv([c[0], c[1]]) as f32)
                    .collect()
            }
            imod_core::MrcMode::UShort => {
                let conv = if self.swapped {
                    u16::from_be_bytes as fn([u8; 2]) -> u16
                } else {
                    u16::from_le_bytes
                };
                raw.chunks_exact(2)
                    .map(|c| conv([c[0], c[1]]) as f32)
                    .collect()
            }
            imod_core::MrcMode::Float => {
                let conv = if self.swapped {
                    f32::from_be_bytes as fn([u8; 4]) -> f32
                } else {
                    f32::from_le_bytes
                };
                raw.chunks_exact(4)
                    .map(|c| conv([c[0], c[1], c[2], c[3]]))
                    .collect()
            }
            imod_core::MrcMode::Rgb => {
                // Byte data, no swapping needed
                raw.iter().map(|&b| b as f32).collect()
            }
            imod_core::MrcMode::FourBit => {
                let mut out = Vec::with_capacity(npix);
                for row in 0..ny {
                    let row_bytes = (nx + 1) / 2;
                    let row_start = row * row_bytes;
                    for col in 0..nx {
                        let byte = raw[row_start + col / 2];
                        let val = if col % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                        out.push(val as f32);
                    }
                }
                out
            }
            _ => {
                return Err(ImodError::UnsupportedMode(self.header.mode));
            }
        };
        Ok(data)
    }

    /// Read the entire image stack as f32.
    pub fn read_all_f32(&mut self) -> Result<Vec<f32>, ImodError> {
        let nz = self.header.nz as usize;
        let mut all = Vec::new();
        for z in 0..nz {
            let slice = self.read_slice_f32(z)?;
            all.extend_from_slice(&slice);
        }
        Ok(all)
    }

    fn dimensions_reasonable(h: &MrcHeader) -> bool {
        h.nx > 0
            && h.ny > 0
            && h.nz > 0
            && h.nx < 100_000
            && h.ny < 100_000
            && h.nz < 1_000_000
    }
}
