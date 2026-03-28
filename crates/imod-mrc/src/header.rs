use binrw::{BinRead, BinWrite};
use imod_core::{ExtHeaderType, MrcMode};

/// MRC 2014 file header (1024 bytes on disk).
///
/// Field layout matches the C `MrcHeader` struct in IMOD's mrcfiles.h,
/// covering only the on-disk portion (bytes 0–1023).
#[derive(Debug, Clone, BinRead, BinWrite)]
#[brw(little)]
pub struct MrcHeader {
    // Words 1-3: dimensions
    pub nx: i32,
    pub ny: i32,
    pub nz: i32,

    // Word 4: data mode
    pub mode: i32,

    // Words 5-7: start indices (unsupported in IMOD, usually 0)
    pub nxstart: i32,
    pub nystart: i32,
    pub nzstart: i32,

    // Words 8-10: grid size
    pub mx: i32,
    pub my: i32,
    pub mz: i32,

    // Words 11-13: cell dimensions in Angstroms
    pub xlen: f32,
    pub ylen: f32,
    pub zlen: f32,

    // Words 14-16: cell angles (usually 90)
    pub alpha: f32,
    pub beta: f32,
    pub gamma: f32,

    // Words 17-19: axis mapping (1=x, 2=y, 3=z)
    pub mapc: i32,
    pub mapr: i32,
    pub maps: i32,

    // Words 20-22: density statistics
    pub amin: f32,
    pub amax: f32,
    pub amean: f32,

    // Word 23: space group (4 bytes, was ispg+nsymbt before 2012)
    pub ispg: i32,

    // Word 24: extended header size in bytes (nsymbt in MRC standard)
    pub next: i32,

    // Word 25: creator id (2 bytes) + 6 bytes blank
    pub creatid: i16,
    pub blank: [u8; 6],

    // Extended header type (4 bytes, e.g. "SERI", "FEI1")
    pub ext_type: [u8; 4],

    // MRC 2014 version number
    pub nversion: i32,

    // 16 bytes blank
    pub blank2: [u8; 16],

    // IMOD-specific short fields
    pub nint: i16,
    pub nreal: i16,
    pub sub: i16,
    pub zfac: i16,

    // Additional min/max for second and third channel
    pub min2: f32,
    pub max2: f32,
    pub min3: f32,
    pub max3: f32,

    // IMOD stamp
    pub imod_stamp: i32,
    pub imod_flags: i32,

    // HVEM extra data
    pub idtype: i16,
    pub lens: i16,
    pub nd1: i16,
    pub nd2: i16,
    pub vd1: i16,
    pub vd2: i16,

    // Tilt angles: [0..2] = original, [3..5] = current
    pub tilt_angles: [f32; 6],

    // MRC 2000 standard origin
    pub xorg: f32,
    pub yorg: f32,
    pub zorg: f32,

    // MAP and machine stamp
    pub cmap: [u8; 4],
    pub stamp: [u8; 4],

    // RMS deviation of density
    pub rms: f32,

    // Labels
    pub nlabl: i32,
    pub labels: [[u8; 80]; 10],
}

impl MrcHeader {
    /// Total size of the header on disk.
    pub const SIZE: usize = 1024;

    /// The IMOD stamp value (little-endian encoding of bytes 0x88, 0x44, 0, 0x25)
    pub const IMOD_STAMP: i32 = 0x2500_4488_u32 as i32;

    /// Parse the data mode.
    pub fn data_mode(&self) -> Option<MrcMode> {
        MrcMode::from_i32(self.mode)
    }

    /// Parse the extended header type.
    pub fn ext_header_type(&self) -> ExtHeaderType {
        ExtHeaderType::from_bytes(&self.ext_type)
    }

    /// Whether this file was written byte-swapped relative to native endianness.
    pub fn is_swapped(&self) -> bool {
        // MRC 2014: stamp bytes should be 0x44 0x44 0x00 0x00 for little-endian
        // or 0x11 0x11 0x00 0x00 for big-endian
        self.stamp[0] == 0x11 && self.stamp[1] == 0x11
    }

    /// Whether this file has the IMOD stamp.
    pub fn is_imod(&self) -> bool {
        self.imod_stamp == Self::IMOD_STAMP
    }

    /// Whether this is an old-style (pre-2014) MRC header.
    ///
    /// Old-style headers lack the "MAP " signature at bytes 208-211 and have
    /// nversion == 0. These files use a different origin field layout.
    pub fn is_old_style(&self) -> bool {
        self.nversion == 0 && &self.cmap != b"MAP "
    }

    /// Pixel size in X (Angstroms per pixel).
    pub fn pixel_size_x(&self) -> f32 {
        if self.mx > 0 { self.xlen / self.mx as f32 } else { 1.0 }
    }

    /// Pixel size in Y.
    pub fn pixel_size_y(&self) -> f32 {
        if self.my > 0 { self.ylen / self.my as f32 } else { 1.0 }
    }

    /// Pixel size in Z.
    pub fn pixel_size_z(&self) -> f32 {
        if self.mz > 0 { self.zlen / self.mz as f32 } else { 1.0 }
    }

    /// Size of the data section for one slice, in bytes.
    pub fn slice_size_bytes(&self) -> usize {
        let mode = self.data_mode().unwrap_or(MrcMode::Byte);
        let nx = self.nx as usize;
        let ny = self.ny as usize;
        match mode {
            MrcMode::FourBit => (nx + 1) / 2 * ny,
            _ => nx * ny * mode.bytes_per_pixel(),
        }
    }

    /// Offset in the file where image data begins.
    pub fn data_offset(&self) -> u64 {
        Self::SIZE as u64 + self.next.max(0) as u64
    }

    /// Create a new header with sensible defaults.
    pub fn new(nx: i32, ny: i32, nz: i32, mode: MrcMode) -> Self {
        let xlen = nx as f32;
        let ylen = ny as f32;
        let zlen = nz as f32;
        Self {
            nx,
            ny,
            nz,
            mode: mode as i32,
            nxstart: 0,
            nystart: 0,
            nzstart: 0,
            mx: nx,
            my: ny,
            mz: nz,
            xlen,
            ylen,
            zlen,
            alpha: 90.0,
            beta: 90.0,
            gamma: 90.0,
            mapc: 1,
            mapr: 2,
            maps: 3,
            amin: 0.0,
            amax: 0.0,
            amean: 0.0,
            ispg: 0,
            next: 0,
            creatid: 0,
            blank: [0; 6],
            ext_type: [0; 4],
            nversion: 20140,
            blank2: [0; 16],
            nint: 0,
            nreal: 0,
            sub: 0,
            zfac: 0,
            min2: 0.0,
            max2: 0.0,
            min3: 0.0,
            max3: 0.0,
            imod_stamp: Self::IMOD_STAMP,
            imod_flags: 0,
            idtype: 0,
            lens: 0,
            nd1: 0,
            nd2: 0,
            vd1: 0,
            vd2: 0,
            tilt_angles: [0.0; 6],
            xorg: 0.0,
            yorg: 0.0,
            zorg: 0.0,
            cmap: *b"MAP ",
            stamp: [0x44, 0x44, 0x00, 0x00], // little-endian
            rms: -1.0,
            nlabl: 0,
            labels: [[0u8; 80]; 10],
        }
    }

    /// Add a label string. Returns false if all 10 label slots are full.
    pub fn add_label(&mut self, text: &str) -> bool {
        let idx = self.nlabl as usize;
        if idx >= 10 {
            return false;
        }
        let bytes = text.as_bytes();
        let len = bytes.len().min(80);
        self.labels[idx][..len].copy_from_slice(&bytes[..len]);
        // Pad with spaces (matching IMOD behavior)
        for b in &mut self.labels[idx][len..] {
            *b = b' ';
        }
        self.nlabl += 1;
        true
    }

    /// Get label at index as a string, trimming trailing spaces/nulls.
    pub fn label(&self, idx: usize) -> Option<&str> {
        if idx >= self.nlabl as usize {
            return None;
        }
        let raw = &self.labels[idx];
        let end = raw.iter().rposition(|&b| b != b' ' && b != 0).map_or(0, |i| i + 1);
        std::str::from_utf8(&raw[..end]).ok()
    }
}
