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
            imod_core::MrcMode::ComplexShort => {
                // Return magnitude sqrt(re^2 + im^2)
                let conv = if self.swapped {
                    i16::from_be_bytes as fn([u8; 2]) -> i16
                } else {
                    i16::from_le_bytes
                };
                raw.chunks_exact(4)
                    .map(|c| {
                        let re = conv([c[0], c[1]]) as f32;
                        let im = conv([c[2], c[3]]) as f32;
                        (re * re + im * im).sqrt()
                    })
                    .collect()
            }
            imod_core::MrcMode::ComplexFloat => {
                // Return magnitude sqrt(re^2 + im^2)
                let conv = if self.swapped {
                    f32::from_be_bytes as fn([u8; 4]) -> f32
                } else {
                    f32::from_le_bytes
                };
                raw.chunks_exact(8)
                    .map(|c| {
                        let re = conv([c[0], c[1], c[2], c[3]]);
                        let im = conv([c[4], c[5], c[6], c[7]]);
                        (re * re + im * im).sqrt()
                    })
                    .collect()
            }
            imod_core::MrcMode::HalfFloat => {
                let conv16 = if self.swapped {
                    u16::from_be_bytes as fn([u8; 2]) -> u16
                } else {
                    u16::from_le_bytes
                };
                raw.chunks_exact(2)
                    .map(|c| {
                        let bits = conv16([c[0], c[1]]);
                        f16_to_f32(bits)
                    })
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

    /// Read a rectangular subarea of a Z slice as f32.
    ///
    /// The subarea is defined by its top-left corner `(x0, y0)` and its
    /// `width` x `height` extent.  Pixels are returned in row-major order.
    pub fn read_subarea_f32(
        &mut self,
        z: usize,
        x0: usize,
        y0: usize,
        width: usize,
        height: usize,
    ) -> Result<Vec<f32>, ImodError> {
        let nx = self.header.nx as usize;
        let ny = self.header.ny as usize;
        let nz = self.header.nz as usize;
        if z >= nz {
            return Err(ImodError::InvalidData(format!(
                "slice index {z} out of range (nz={nz})"
            )));
        }
        if x0 + width > nx || y0 + height > ny {
            return Err(ImodError::InvalidData(format!(
                "subarea ({x0},{y0})+({width},{height}) exceeds image ({nx},{ny})"
            )));
        }

        let mode = self
            .header
            .data_mode()
            .ok_or(ImodError::UnsupportedMode(self.header.mode))?;
        let bpp = mode.bytes_per_pixel();
        let data_start = self.header.data_offset() + (z as u64 * self.header.slice_size_bytes() as u64);

        let mut out = Vec::with_capacity(width * height);
        let mut row_buf = vec![0u8; width * bpp];

        for row in y0..(y0 + height) {
            let row_offset = data_start + (row * nx * bpp) as u64 + (x0 * bpp) as u64;
            self.reader.seek(SeekFrom::Start(row_offset))?;
            self.reader.read_exact(&mut row_buf)?;
            let row_f32 = self.convert_raw_to_f32(&row_buf, mode, width)?;
            out.extend_from_slice(&row_f32);
        }
        Ok(out)
    }

    /// Read a Z slice as complex pairs (for modes 3/4).
    ///
    /// Returns a vector of `(real, imaginary)` pairs, one per pixel.
    pub fn read_slice_complex(&mut self, z: usize) -> Result<Vec<(f32, f32)>, ImodError> {
        let raw = self.read_slice_raw(z)?;
        let mode = self
            .header
            .data_mode()
            .ok_or(ImodError::UnsupportedMode(self.header.mode))?;

        match mode {
            imod_core::MrcMode::ComplexShort => {
                let conv = if self.swapped {
                    i16::from_be_bytes as fn([u8; 2]) -> i16
                } else {
                    i16::from_le_bytes
                };
                Ok(raw
                    .chunks_exact(4)
                    .map(|c| {
                        let re = conv([c[0], c[1]]) as f32;
                        let im = conv([c[2], c[3]]) as f32;
                        (re, im)
                    })
                    .collect())
            }
            imod_core::MrcMode::ComplexFloat => {
                let conv = if self.swapped {
                    f32::from_be_bytes as fn([u8; 4]) -> f32
                } else {
                    f32::from_le_bytes
                };
                Ok(raw
                    .chunks_exact(8)
                    .map(|c| {
                        let re = conv([c[0], c[1], c[2], c[3]]);
                        let im = conv([c[4], c[5], c[6], c[7]]);
                        (re, im)
                    })
                    .collect())
            }
            _ => Err(ImodError::UnsupportedMode(self.header.mode)),
        }
    }

    /// Read a Y row across all Z slices. Returns `nx * nz` values in
    /// slice-major order (all X values for z=0, then z=1, etc.).
    pub fn read_y_slice_f32(&mut self, y: usize) -> Result<Vec<f32>, ImodError> {
        let nx = self.header.nx as usize;
        let ny = self.header.ny as usize;
        let nz = self.header.nz as usize;
        if y >= ny {
            return Err(ImodError::InvalidData(format!(
                "y index {y} out of range (ny={ny})"
            )));
        }

        let mode = self
            .header
            .data_mode()
            .ok_or(ImodError::UnsupportedMode(self.header.mode))?;
        let bpp = mode.bytes_per_pixel();
        let slice_bytes = self.header.slice_size_bytes() as u64;
        let data_start = self.header.data_offset();
        let row_bytes = nx * bpp;

        let mut out = Vec::with_capacity(nx * nz);
        let mut row_buf = vec![0u8; row_bytes];

        for z in 0..nz {
            let offset = data_start + z as u64 * slice_bytes + (y * row_bytes) as u64;
            self.reader.seek(SeekFrom::Start(offset))?;
            self.reader.read_exact(&mut row_buf)?;
            let row_f32 = self.convert_raw_to_f32(&row_buf, mode, nx)?;
            out.extend_from_slice(&row_f32);
        }
        Ok(out)
    }

    /// Convert raw bytes to f32 values for a given mode and pixel count.
    fn convert_raw_to_f32(
        &self,
        raw: &[u8],
        mode: imod_core::MrcMode,
        _npix: usize,
    ) -> Result<Vec<f32>, ImodError> {
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
            imod_core::MrcMode::ComplexFloat => {
                let conv = if self.swapped {
                    f32::from_be_bytes as fn([u8; 4]) -> f32
                } else {
                    f32::from_le_bytes
                };
                raw.chunks_exact(8)
                    .map(|c| {
                        let re = conv([c[0], c[1], c[2], c[3]]);
                        let im = conv([c[4], c[5], c[6], c[7]]);
                        (re * re + im * im).sqrt()
                    })
                    .collect()
            }
            imod_core::MrcMode::ComplexShort => {
                let conv = if self.swapped {
                    i16::from_be_bytes as fn([u8; 2]) -> i16
                } else {
                    i16::from_le_bytes
                };
                raw.chunks_exact(4)
                    .map(|c| {
                        let re = conv([c[0], c[1]]) as f32;
                        let im = conv([c[2], c[3]]) as f32;
                        (re * re + im * im).sqrt()
                    })
                    .collect()
            }
            imod_core::MrcMode::HalfFloat => {
                let conv16 = if self.swapped {
                    u16::from_be_bytes as fn([u8; 2]) -> u16
                } else {
                    u16::from_le_bytes
                };
                raw.chunks_exact(2)
                    .map(|c| {
                        let bits = conv16([c[0], c[1]]);
                        f16_to_f32(bits)
                    })
                    .collect()
            }
            imod_core::MrcMode::Rgb => raw.iter().map(|&b| b as f32).collect(),
            imod_core::MrcMode::FourBit => {
                return Err(ImodError::UnsupportedMode(self.header.mode));
            }
        };
        Ok(data)
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

/// Convert an IEEE 754 half-precision (f16) bit pattern to f32.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exponent = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x3FF) as u32;

    if exponent == 0 {
        if mantissa == 0 {
            // Signed zero
            f32::from_bits(sign << 31)
        } else {
            // Subnormal: normalize
            let mut m = mantissa;
            let mut e = 0i32;
            while (m & 0x400) == 0 {
                m <<= 1;
                e += 1;
            }
            let exp = (127 - 15 + 1 - e) as u32;
            let frac = (m & 0x3FF) << 13;
            f32::from_bits((sign << 31) | (exp << 23) | frac)
        }
    } else if exponent == 31 {
        // Inf or NaN
        let frac = mantissa << 13;
        f32::from_bits((sign << 31) | (0xFF << 23) | frac)
    } else {
        // Normal number
        let exp = (exponent as i32 - 15 + 127) as u32;
        let frac = mantissa << 13;
        f32::from_bits((sign << 31) | (exp << 23) | frac)
    }
}
