//! Translation of `IMOD/raptor/mainClasses/ioMRCVol.h` and `ioMRCVol.cpp`:
//! RAPTOR's own MRC reader, which reads the 1024-byte header as the raw
//! little-endian `struct header` and the whole volume into memory.  It does
//! not go through IMOD's `libiimod` and does not honour byte swapping, the
//! MRC 2014 conventions or any mode other than 0, 1, 2 and 6 -- that is the
//! source's behaviour and it is kept.
//!
//! Of the header only the fields the reached code reads are kept (`nx`,
//! `ny`, `nz`, `mode`, `next`).  The members that only the unreached
//! `writeMRCfile`, `createHeader`, `readMRCheader`, `get2DPatch`, the
//! `double` overload of `readMRCSlice` and the header accessors other than
//! `headerGetNx/Ny/Nz` use are recorded in `DEAD_CODE.md`.

use crate::imod::cxx_stream::cout;
use crate::imod::libcfshr::b3dutil::exit;
use std::io::{Read, Seek, SeekFrom};

/// The volume data, `void* MRCvol` as the four element types
/// `readMRCfile` allocates.
#[derive(Debug, Default)]
enum MrcVol {
    #[default]
    None,
    Mode0(Vec<u8>),
    Mode1(Vec<i16>),
    Mode2(Vec<f32>),
    Mode6(Vec<u16>),
}

/// The members of `ioMRC::header` the reached code reads.
#[derive(Debug, Default, Clone, Copy)]
struct Header {
    nx: i32,
    ny: i32,
    nz: i32,
    mode: i32,
    next: i32,
}

/// `class ioMRC`.
#[derive(Debug, Default)]
pub struct IoMrc {
    mrc_vol: MrcVol,
    size_vol: i64,
    mrc_header: Header,
}

impl IoMrc {
    /// `ioMRC()` (`ioMRCVol.h:21`).
    pub fn new() -> IoMrc {
        IoMrc::default()
    }

    /// `ioMRC::clear()` (`ioMRCVol.h:30`).
    pub fn clear(&mut self) {
        self.mrc_vol = MrcVol::None;
    }

    /// `ioMRC::headerGetNx()`.
    pub fn header_get_nx(&self) -> i32 {
        self.mrc_header.nx
    }

    /// `ioMRC::headerGetNy()`.
    pub fn header_get_ny(&self) -> i32 {
        self.mrc_header.ny
    }

    /// `ioMRC::headerGetNz()`.
    pub fn header_get_nz(&self) -> i32 {
        self.mrc_header.nz
    }

    /// `ioMRC::getType()` (`ioMRCVol.h:156`): bytes per voxel.
    pub fn get_type(&self) -> i32 {
        match self.mrc_header.mode {
            0 => 1,
            1 => 2,
            2 => 4,
            3 => 4,
            4 => 8,
            6 => 2,
            _ => {
                cout("Wrong mode read from the header\n");
                exit(-1);
            }
        }
    }

    /// `ioMRC::readMRCfile(string fileIn)` (`ioMRCVol.cpp:64`).  The
    /// `sizeof(header) != 1024` architecture check cannot fail and the
    /// `sizeVol > LONG_MAX` check cannot be true, so neither is kept.
    pub fn read_mrc_file(&mut self, file_in: &str) -> bool {
        let mut input = match std::fs::File::open(file_in) {
            Ok(file) => file,
            Err(_) => {
                cout("Error opening file\n");
                self.mrc_vol = MrcVol::None;
                return false;
            }
        };
        // `in.read((char*)&MRCheader, sizeof(header))`: a short file leaves
        // the rest of the header as it was.
        let mut raw = [0u8; 1024];
        let mut got = 0usize;
        while got < raw.len() {
            match input.read(&mut raw[got..]) {
                Ok(0) | Err(_) => break,
                Ok(n) => got += n,
            }
        }
        let field = |off: usize, old: i32| -> i32 {
            if off + 4 <= got {
                i32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]])
            } else {
                old
            }
        };
        self.mrc_header.nx = field(0, self.mrc_header.nx);
        self.mrc_header.ny = field(4, self.mrc_header.ny);
        self.mrc_header.nz = field(8, self.mrc_header.nz);
        self.mrc_header.mode = field(12, self.mrc_header.mode);
        self.mrc_header.next = field(92, self.mrc_header.next);
        self.mrc_vol = MrcVol::None;
        self.size_vol =
            self.mrc_header.nx as i64 * self.mrc_header.ny as i64 * self.mrc_header.nz as i64;
        // `calloc(sizeVol, getType())`; a failed allocation (NULL) is the
        // source's "Problem allocating memory" exit.
        let n = self.size_vol.max(0) as usize;
        let allocated = match self.mrc_header.mode {
            0 => zeroed(n).map(MrcVol::Mode0),
            1 => zeroed(n).map(MrcVol::Mode1),
            2 => zeroed(n).map(MrcVol::Mode2),
            6 => zeroed(n).map(MrcVol::Mode6),
            mode => {
                cout(&format!(
                    "Error reading volume.Mode {mode}for MRC not supported\n"
                ));
                exit(-1);
            }
        };
        let Some(vol) = allocated else {
            cout("Problem allocating memory for all the volume at once\n");
            exit(-1);
        };
        self.mrc_vol = vol;
        // Move the pointer of the binary file to the start of the volume
        // (some datasets use extra head bytes).
        let _ = input.seek(SeekFrom::Start(
            (1024i64 + self.mrc_header.next as i64) as u64,
        ));
        // `in.read((char*)MRCvol, sizeVol*getType())`: the file is raw
        // little-endian, and a short file leaves the rest of the calloc'd
        // volume zero -- including the high bytes of an element it ends in
        // the middle of.  Decoded a chunk at a time so the volume is not held
        // twice.
        let elem = self.get_type() as usize;
        let total = n * elem;
        let mut chunk = vec![0u8; 1 << 20];
        let mut done = 0usize;
        while done < total {
            let want = chunk.len().min(total - done);
            let got = match input.read(&mut chunk[..want]) {
                Ok(0) | Err(_) => break,
                Ok(got) => got,
            };
            let bytes = &chunk[..got];
            match &mut self.mrc_vol {
                MrcVol::Mode0(v) => v[done..done + got].copy_from_slice(bytes),
                MrcVol::Mode1(v) => {
                    for (k, &byte) in bytes.iter().enumerate() {
                        let at = done + k;
                        let x = &mut v[at / 2];
                        *x = (*x as u16 | (byte as u16) << (8 * (at % 2))) as i16;
                    }
                }
                MrcVol::Mode2(v) => {
                    for (k, &byte) in bytes.iter().enumerate() {
                        let at = done + k;
                        let x = &mut v[at / 4];
                        *x = f32::from_bits(x.to_bits() | (byte as u32) << (8 * (at % 4)));
                    }
                }
                MrcVol::Mode6(v) => {
                    for (k, &byte) in bytes.iter().enumerate() {
                        let at = done + k;
                        v[at / 2] |= (byte as u16) << (8 * (at % 2));
                    }
                }
                MrcVol::None => {}
            }
            done += got;
        }
        true
    }

    /// `ioMRC::readMRCSlice(int numSlice, float* slice)`
    /// (`ioMRCVol.cpp:228`).  The source rejects only `numSlice >= nz`; a
    /// negative slice number (which `main` can pass for the synthetic
    /// template of a stack of fewer than 11 views, `zerotilt` plus a random
    /// offset of up to 5) read before the volume.  It takes the same error
    /// exit here (`BUGS.md`, RAPTOR).
    pub fn read_mrc_slice(&self, num_slice: i32, slice: &mut [f32]) {
        if num_slice >= self.mrc_header.nz || num_slice < 0 {
            cout("Error reading slice. Numslice is greater than number of slices\n");
            exit(-1);
        }
        let mut pos = num_slice as i64 * self.mrc_header.nx as i64 * self.mrc_header.ny as i64;
        let mut count = 0usize;
        let n = self.mrc_header.nx as i64 * self.mrc_header.ny as i64;
        macro_rules! copy_slice {
            ($v:expr) => {
                for _ in 0..n {
                    slice[count] = $v[pos as usize] as f32;
                    count += 1;
                    pos += 1;
                }
            };
        }
        match &self.mrc_vol {
            MrcVol::Mode0(v) => copy_slice!(v),
            MrcVol::Mode1(v) => copy_slice!(v),
            MrcVol::Mode2(v) => copy_slice!(v),
            MrcVol::Mode6(v) => copy_slice!(v),
            MrcVol::None => {
                cout(&format!(
                    "Error reading slice. MRC mode {} not recognized\n",
                    self.mrc_header.mode
                ));
                exit(-1);
            }
        }
    }

    /// `ioMRC::readMRCpatch(int numSlice, int x0, int y0, int sizeX, int
    /// sizeY, float* patch)` (`ioMRCVol.cpp:297`): the patch centred at
    /// (x0, y0), stored transposed (x outer, y inner).
    pub fn read_mrc_patch(
        &self,
        num_slice: i32,
        x0: i32,
        y0: i32,
        size_x: i32,
        size_y: i32,
        patch: &mut [f32],
    ) -> bool {
        if num_slice >= self.mrc_header.nz {
            cout(&format!(
                "Error reading slice {num_slice}. Numslice is greater than number of slices\n"
            ));
            exit(-1);
        }
        let pos_ini = num_slice as i64 * self.mrc_header.nx as i64 * self.mrc_header.ny as i64;
        let mut count = 0usize;
        let x_min = (x0 - size_x).max(0);
        let x_max = self.mrc_header.nx.min(x0 + size_x);
        let y_min = (y0 - size_y).max(0);
        let y_max = self.mrc_header.ny.min(y0 + size_y);
        if x_min == 0 || x_max == self.mrc_header.nx || y_min == 0 || y_max == self.mrc_header.ny {
            return false;
        }
        let nx = self.mrc_header.nx as i64;
        macro_rules! copy_patch {
            ($v:expr) => {
                for x in x_min..x_max {
                    let mut pos = pos_ini + (y_min as i64 * nx) + x as i64;
                    for _y in y_min..y_max {
                        patch[count] = $v[pos as usize] as f32;
                        count += 1;
                        // x,y are transposed when acquiring the patch
                        pos += nx;
                    }
                }
            };
        }
        match &self.mrc_vol {
            MrcVol::Mode0(v) => copy_patch!(v),
            MrcVol::Mode1(v) => copy_patch!(v),
            MrcVol::Mode2(v) => copy_patch!(v),
            MrcVol::Mode6(v) => copy_patch!(v),
            MrcVol::None => {
                cout(&format!(
                    "Error reading slice. MRC mode {} not recognized\n",
                    self.mrc_header.mode
                ));
                exit(-1);
            }
        }
        true
    }
}

/// `calloc(n, size)`: `None` where the allocation fails.
fn zeroed<T: Default + Clone>(n: usize) -> Option<Vec<T>> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).ok()?;
    v.resize(n, T::default());
    Some(v)
}
