//! Safe translation of `IMOD/raptor/mainClasses/ioMRCVol.{h,cpp}`.

use std::fs;
use std::path::Path;

/// The source's `void *MRCvol`, made owned and type-safe.
#[derive(Clone, Debug, PartialEq)]
pub enum MrcVolume {
    Byte(Vec<u8>),
    Short(Vec<i16>),
    Float(Vec<f32>),
    ComplexShort(Vec<[i16; 2]>),
    ComplexFloat(Vec<[f32; 2]>),
    Ushort(Vec<u16>),
}

impl MrcVolume {
    fn mode(&self) -> i32 {
        match self {
            Self::Byte(_) => 0,
            Self::Short(_) => 1,
            Self::Float(_) => 2,
            Self::ComplexShort(_) => 3,
            Self::ComplexFloat(_) => 4,
            Self::Ushort(_) => 6,
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Byte(values) => values.len(),
            Self::Short(values) => values.len(),
            Self::Float(values) => values.len(),
            Self::ComplexShort(values) => values.len(),
            Self::ComplexFloat(values) => values.len(),
            Self::Ushort(values) => values.len(),
        }
    }

    fn bytes(&self) -> Vec<u8> {
        match self {
            Self::Byte(values) => values.clone(),
            Self::Short(values) => values
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect(),
            Self::Float(values) => values
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect(),
            Self::ComplexShort(values) => values
                .iter()
                .flat_map(|value| {
                    value[0]
                        .to_ne_bytes()
                        .into_iter()
                        .chain(value[1].to_ne_bytes())
                })
                .collect(),
            Self::ComplexFloat(values) => values
                .iter()
                .flat_map(|value| {
                    value[0]
                        .to_ne_bytes()
                        .into_iter()
                        .chain(value[1].to_ne_bytes())
                })
                .collect(),
            Self::Ushort(values) => values
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect(),
        }
    }

    fn value_f64(&self, position: usize) -> f64 {
        match self {
            Self::Byte(values) => values[position] as f64,
            Self::Short(values) => values[position] as f64,
            Self::Float(values) => values[position] as f64,
            Self::Ushort(values) => values[position] as f64,
            Self::ComplexShort(_) | Self::ComplexFloat(_) => {
                panic!("MRC mode has no scalar value representation")
            }
        }
    }
}

/// The exact 1024-byte MRC header used by RAPTOR's historical `ioMRC`.
#[derive(Clone, Debug, PartialEq)]
pub struct IoMrcHeader {
    pub nx: i32,
    pub ny: i32,
    pub nz: i32,
    pub mode: i32,
    pub nxstart: i32,
    pub nystart: i32,
    pub nzstart: i32,
    pub mx: i32,
    pub my: i32,
    pub mz: i32,
    pub cell_dim: [f32; 3],
    pub cell_ang: [f32; 3],
    pub mapc: i32,
    pub mapr: i32,
    pub maps: i32,
    pub dmin: f32,
    pub dmax: f32,
    pub dmean: f32,
    pub ispg: i16,
    pub nsymbt: i16,
    pub next: i32,
    pub creatid: i16,
    pub extra: [u8; 30],
    pub nint: i16,
    pub nreal: i16,
    pub extra2: [u8; 28],
    pub idtype: i16,
    pub lens: i16,
    pub nd1: i16,
    pub nd2: i16,
    pub vd1: i16,
    pub vd2: i16,
    pub tilt_angles: [f32; 6],
    pub origin: [f32; 3],
    pub cmap: [u8; 4],
    pub stamp: [u8; 4],
    pub rms: f32,
    pub nlbl: i32,
    pub label: [u8; 800],
}

impl Default for IoMrcHeader {
    fn default() -> Self {
        Self {
            nx: 0,
            ny: 0,
            nz: 0,
            mode: 0,
            nxstart: 0,
            nystart: 0,
            nzstart: 0,
            mx: 0,
            my: 0,
            mz: 0,
            cell_dim: [0.0; 3],
            cell_ang: [0.0; 3],
            mapc: 0,
            mapr: 0,
            maps: 0,
            dmin: 0.0,
            dmax: 0.0,
            dmean: 0.0,
            ispg: 0,
            nsymbt: 0,
            next: 0,
            creatid: 0,
            extra: [0; 30],
            nint: 0,
            nreal: 0,
            extra2: [0; 28],
            idtype: 0,
            lens: 0,
            nd1: 0,
            nd2: 0,
            vd1: 0,
            vd2: 0,
            tilt_angles: [0.0; 6],
            origin: [0.0; 3],
            cmap: [0; 4],
            stamp: [0; 4],
            rms: 0.0,
            nlbl: 0,
            label: [0; 800],
        }
    }
}

impl IoMrcHeader {
    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 1024 {
            return None;
        }
        let integer = |offset| i32::from_ne_bytes(bytes[offset..offset + 4].try_into().unwrap());
        let short = |offset| i16::from_ne_bytes(bytes[offset..offset + 2].try_into().unwrap());
        let float = |offset| f32::from_ne_bytes(bytes[offset..offset + 4].try_into().unwrap());
        let mut header = Self {
            nx: integer(0),
            ny: integer(4),
            nz: integer(8),
            mode: integer(12),
            nxstart: integer(16),
            nystart: integer(20),
            nzstart: integer(24),
            mx: integer(28),
            my: integer(32),
            mz: integer(36),
            cell_dim: [float(40), float(44), float(48)],
            cell_ang: [float(52), float(56), float(60)],
            mapc: integer(64),
            mapr: integer(68),
            maps: integer(72),
            dmin: float(76),
            dmax: float(80),
            dmean: float(84),
            ispg: short(88),
            nsymbt: short(90),
            next: integer(92),
            creatid: short(96),
            nint: short(128),
            nreal: short(130),
            idtype: short(160),
            lens: short(162),
            nd1: short(164),
            nd2: short(166),
            vd1: short(168),
            vd2: short(170),
            tilt_angles: [
                float(172),
                float(176),
                float(180),
                float(184),
                float(188),
                float(192),
            ],
            origin: [float(196), float(200), float(204)],
            cmap: bytes[208..212].try_into().unwrap(),
            stamp: bytes[212..216].try_into().unwrap(),
            rms: float(216),
            nlbl: integer(220),
            ..Self::default()
        };
        header.extra.copy_from_slice(&bytes[98..128]);
        header.extra2.copy_from_slice(&bytes[132..160]);
        header.label.copy_from_slice(&bytes[224..1024]);
        Some(header)
    }

    fn bytes(&self) -> [u8; 1024] {
        let mut bytes = [0; 1024];
        macro_rules! integer {
            ($offset:expr, $value:expr) => {
                bytes[$offset..$offset + 4].copy_from_slice(&$value.to_ne_bytes())
            };
        }
        macro_rules! short {
            ($offset:expr, $value:expr) => {
                bytes[$offset..$offset + 2].copy_from_slice(&$value.to_ne_bytes())
            };
        }
        macro_rules! float {
            ($offset:expr, $value:expr) => {
                bytes[$offset..$offset + 4].copy_from_slice(&$value.to_ne_bytes())
            };
        }
        integer!(0, self.nx);
        integer!(4, self.ny);
        integer!(8, self.nz);
        integer!(12, self.mode);
        integer!(16, self.nxstart);
        integer!(20, self.nystart);
        integer!(24, self.nzstart);
        integer!(28, self.mx);
        integer!(32, self.my);
        integer!(36, self.mz);
        for index in 0..3 {
            float!(40 + 4 * index, self.cell_dim[index]);
            float!(52 + 4 * index, self.cell_ang[index]);
        }
        integer!(64, self.mapc);
        integer!(68, self.mapr);
        integer!(72, self.maps);
        float!(76, self.dmin);
        float!(80, self.dmax);
        float!(84, self.dmean);
        short!(88, self.ispg);
        short!(90, self.nsymbt);
        integer!(92, self.next);
        short!(96, self.creatid);
        bytes[98..128].copy_from_slice(&self.extra);
        short!(128, self.nint);
        short!(130, self.nreal);
        bytes[132..160].copy_from_slice(&self.extra2);
        short!(160, self.idtype);
        short!(162, self.lens);
        short!(164, self.nd1);
        short!(166, self.nd2);
        short!(168, self.vd1);
        short!(170, self.vd2);
        for index in 0..6 {
            float!(172 + 4 * index, self.tilt_angles[index]);
        }
        for index in 0..3 {
            float!(196 + 4 * index, self.origin[index]);
        }
        bytes[208..212].copy_from_slice(&self.cmap);
        bytes[212..216].copy_from_slice(&self.stamp);
        float!(216, self.rms);
        integer!(220, self.nlbl);
        bytes[224..1024].copy_from_slice(&self.label);
        bytes
    }
}

/// C++ `ioMRC`, with owned volume storage and explicit disk encoding.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct IoMrc {
    pub volume: Option<MrcVolume>,
    pub size_vol: usize,
    pub header: IoMrcHeader,
}

impl IoMrc {
    /// C++ `ioMRC(void *, int, int, int, int)`.
    pub fn new(volume: MrcVolume, nx: i32, ny: i32, nz: i32, mode: i32) -> Self {
        let size_vol = nx.max(0) as usize * ny.max(0) as usize * nz.max(0) as usize;
        assert_eq!(
            volume.len(),
            size_vol,
            "MRC volume dimensions disagree with payload"
        );
        assert_eq!(volume.mode(), mode, "MRC mode disagrees with payload type");
        Self {
            volume: Some(volume),
            size_vol,
            header: IoMrcHeader {
                nx,
                ny,
                nz,
                mode,
                ..IoMrcHeader::default()
            },
        }
    }

    /// C++ `clear`; dropping the vector replaces `free`.
    pub fn clear(&mut self) {
        self.volume = None;
        self.size_vol = 0;
    }

    /// C++ `writeMRCfile`.
    pub fn write_mrc_file(&mut self, path: impl AsRef<Path>, create_header: bool) -> bool {
        if create_header {
            self.create_header();
        }
        let Some(volume) = &self.volume else {
            return false;
        };
        if volume.len() != self.size_vol || volume.mode() != self.header.mode {
            return false;
        }
        let mut output = Vec::with_capacity(1024 + volume.bytes().len());
        output.extend_from_slice(&self.header.bytes());
        output.extend_from_slice(&volume.bytes());
        fs::write(path, output).is_ok()
    }

    /// C++ `readMRCfile`.  Like the source, only scalar modes 0, 1, 2, and 6 load.
    pub fn read_mrc_file(&mut self, path: impl AsRef<Path>) -> bool {
        let Ok(input) = fs::read(path) else {
            self.clear();
            return false;
        };
        let Some(header) = IoMrcHeader::from_bytes(&input) else {
            return false;
        };
        let Some(size_vol) = (header.nx >= 0 && header.ny >= 0 && header.nz >= 0)
            .then(|| header.nx as usize)
            .and_then(|n| n.checked_mul(header.ny as usize))
            .and_then(|n| n.checked_mul(header.nz as usize))
        else {
            return false;
        };
        let data_start = match 1024usize.checked_add(header.next.max(0) as usize) {
            Some(value) => value,
            None => return false,
        };
        let bytes_per_value = match header.mode {
            0 => 1,
            1 | 6 => 2,
            2 => 4,
            _ => return false,
        };
        let Some(data_bytes) = size_vol.checked_mul(bytes_per_value) else {
            return false;
        };
        let Some(data_end) = data_start.checked_add(data_bytes) else {
            return false;
        };
        if data_end > input.len() {
            return false;
        }
        let data = &input[data_start..data_end];
        self.volume = Some(match header.mode {
            0 => MrcVolume::Byte(data.to_vec()),
            1 => MrcVolume::Short(
                data.chunks_exact(2)
                    .map(|value| i16::from_ne_bytes(value.try_into().unwrap()))
                    .collect(),
            ),
            2 => MrcVolume::Float(
                data.chunks_exact(4)
                    .map(|value| f32::from_ne_bytes(value.try_into().unwrap()))
                    .collect(),
            ),
            6 => MrcVolume::Ushort(
                data.chunks_exact(2)
                    .map(|value| u16::from_ne_bytes(value.try_into().unwrap()))
                    .collect(),
            ),
            _ => unreachable!(),
        });
        self.size_vol = size_vol;
        self.header = header;
        true
    }

    /// C++ `readMRCheader`.
    pub fn read_mrc_header(&mut self, path: impl AsRef<Path>) -> bool {
        match fs::read(path)
            .ok()
            .and_then(|input| IoMrcHeader::from_bytes(&input))
        {
            Some(header) => {
                self.header = header;
                true
            }
            None => {
                self.volume = None;
                false
            }
        }
    }

    /// C++ `createHeader`.
    pub fn create_header(&mut self) {
        let mut minimum = self.read_mrc_value_double(0);
        let mut maximum = minimum;
        let mut mean = 0.0;
        for index in 0..self.size_vol {
            let value = self.read_mrc_value_double(index);
            minimum = minimum.min(value);
            maximum = maximum.max(value);
            mean += value;
        }
        self.header.nxstart = 0;
        self.header.nystart = 0;
        self.header.nzstart = 0;
        self.header.mx = 0;
        self.header.my = 0;
        self.header.mz = 0;
        self.header.cell_dim = [0.0; 3];
        self.header.cell_ang = [90.0; 3];
        self.header.mapc = 1;
        self.header.mapr = 2;
        self.header.maps = 3;
        self.header.dmin = minimum as f32;
        self.header.dmax = maximum as f32;
        self.header.dmean = (mean / self.size_vol as f64) as f32;
        self.header.ispg = 0;
        self.header.nsymbt = 0;
        self.header.next = 0;
        self.header.creatid = 1000;
        self.header.extra = [b'0'; 30];
        self.header.nint = 0;
        self.header.nreal = 0;
        self.header.extra2 = [b'0'; 28];
        self.header.idtype = 0;
        self.header.lens = 0;
        self.header.nd1 = 0;
        self.header.nd2 = 0;
        self.header.vd1 = 0;
        self.header.vd2 = 0;
        self.header.tilt_angles = [0.0; 6];
        self.header.origin = [0.0; 3];
        self.header.cmap = *b"MAP ";
        self.header.stamp = [0x44, b'0', b'0', b'0'];
        self.header.rms = 0.0;
        self.header.nlbl = 0;
        self.header.label = [b' '; 800];
    }

    /// C++ `readMRCValueDouble(long int)`.
    pub fn read_mrc_value_double(&self, position: usize) -> f64 {
        self.volume
            .as_ref()
            .expect("MRC volume is empty")
            .value_f64(position)
    }
    /// C++ `readMRCValueFloat(long int)`.
    pub fn read_mrc_value_float(&self, position: usize) -> f32 {
        self.read_mrc_value_double(position) as f32
    }
    /// C++ coordinate overload of `readMRCValueDouble`.
    pub fn read_mrc_value_double_at(&self, x: i32, y: i32, z: i32) -> f64 {
        self.read_mrc_value_double((x + self.header.nx * (y + z * self.header.ny)) as usize)
    }
    /// C++ coordinate overload of `readMRCValueFloat`.
    pub fn read_mrc_value_float_at(&self, x: i32, y: i32, z: i32) -> f32 {
        self.read_mrc_value_float((x + self.header.nx * (y + z * self.header.ny)) as usize)
    }

    /// C++ `readMRCSlice(int, float *)`.
    pub fn read_mrc_slice_float(&self, slice: i32, output: &mut [f32]) -> bool {
        if slice < 0
            || slice >= self.header.nz
            || output.len() < self.header.nx as usize * self.header.ny as usize
        {
            return false;
        }
        let start = slice as usize * self.header.nx as usize * self.header.ny as usize;
        for (index, value) in output
            .iter_mut()
            .take(self.header.nx as usize * self.header.ny as usize)
            .enumerate()
        {
            *value = self.read_mrc_value_float(start + index);
        }
        true
    }

    /// C++ `readMRCSlice(int, double *)`.
    pub fn read_mrc_slice_double(&self, slice: i32, output: &mut [f64]) -> bool {
        if slice < 0
            || slice >= self.header.nz
            || output.len() < self.header.nx as usize * self.header.ny as usize
        {
            return false;
        }
        let start = slice as usize * self.header.nx as usize * self.header.ny as usize;
        for (index, value) in output
            .iter_mut()
            .take(self.header.nx as usize * self.header.ny as usize)
            .enumerate()
        {
            *value = self.read_mrc_value_double(start + index);
        }
        true
    }

    /// C++ `readMRCpatch`; preserves its transposed x/y traversal.
    pub fn read_mrc_patch(
        &self,
        slice: i32,
        x0: i32,
        y0: i32,
        size_x: i32,
        size_y: i32,
        output: &mut [f32],
    ) -> bool {
        if slice < 0 || slice >= self.header.nz {
            return false;
        }
        let x_min = (x0 - size_x).max(0);
        let x_max = self.header.nx.min(x0 + size_x);
        let y_min = (y0 - size_y).max(0);
        let y_max = self.header.ny.min(y0 + size_y);
        if x_min == 0
            || x_max == self.header.nx
            || y_min == 0
            || y_max == self.header.ny
            || output.len() < ((x_max - x_min) * (y_max - y_min)) as usize
        {
            return false;
        }
        let base = slice as usize * self.header.nx as usize * self.header.ny as usize;
        let mut count = 0;
        for x in x_min..x_max {
            for y in y_min..y_max {
                output[count] = self
                    .read_mrc_value_float(base + y as usize * self.header.nx as usize + x as usize);
                count += 1;
            }
        }
        true
    }

    /// C++ `get2DPatch`, including its one-pixel interpolation margin and ordering.
    pub fn get_2d_patch(
        &self,
        p0e123: &[f64; 9],
        patch_size_x: i32,
        patch_size_y: i32,
        output: &mut [f32],
    ) -> bool {
        if output.len() < (patch_size_x * patch_size_y) as usize {
            return false;
        }
        let cx = patch_size_x as f32 / 2.0;
        let cy = patch_size_y as f32 / 2.0;
        for (u, v) in [
            (-cx as f64, -cy as f64),
            (patch_size_x as f64 - 1.0 - cx as f64, -cy as f64),
            (
                patch_size_x as f64 - 1.0 - cx as f64,
                patch_size_y as f64 - 1.0 - cy as f64,
            ),
            (-cx as f64, patch_size_y as f64 - 1.0 - cy as f64),
        ] {
            for axis in 0..3 {
                let coordinate =
                    (p0e123[axis] + u * p0e123[axis + 3] + v * p0e123[axis + 6]).ceil() as f32;
                let bound = [self.header.nx, self.header.ny, self.header.nz][axis] as f32;
                if coordinate > bound - 1.0 || coordinate - 1.0 < 0.0 {
                    return false;
                }
            }
        }
        let mut xini =
            (p0e123[0] + (-1.0 - cx as f64) * p0e123[3] + (-1.0 - cy as f64) * p0e123[6]) as f32;
        let mut yini =
            (p0e123[1] + (-1.0 - cx as f64) * p0e123[4] + (-1.0 - cy as f64) * p0e123[7]) as f32;
        let mut zini =
            (p0e123[2] + (-1.0 - cx as f64) * p0e123[5] + (-1.0 - cy as f64) * p0e123[8]) as f32;
        let mut count = 0;
        for ii in 0..patch_size_x {
            xini = (xini as f64 + p0e123[3]) as f32;
            yini = (yini as f64 + p0e123[4]) as f32;
            zini = (zini as f64 + p0e123[5]) as f32;
            let mut xi = xini;
            let mut yi = yini;
            let mut zi = zini;
            for jj in 0..patch_size_y {
                xi = (xi as f64 + p0e123[6]) as f32;
                yi = (yi as f64 + p0e123[7]) as f32;
                zi = (zi as f64 + p0e123[8]) as f32;
                let fx = xi.floor() as usize;
                let fy = yi.floor() as usize;
                let fz = zi.floor() as usize;
                let wx = xi - fx as f32;
                let wy = yi - fy as f32;
                let wz = zi - fz as f32;
                let sample = |dx, dy, dz| {
                    self.read_mrc_value_float_at(
                        (fx + dx) as i32,
                        (fy + dy) as i32,
                        (fz + dz) as i32,
                    )
                };
                output[count] = wx * wy * wz * sample(1, 1, 1)
                    + wx * wy * (1.0 - wz) * sample(1, 1, 0)
                    + wx * (1.0 - wy) * wz * sample(1, 0, 1)
                    + wx * (1.0 - wy) * (1.0 - wz) * sample(1, 0, 0)
                    + (1.0 - wx) * wy * wz * sample(0, 1, 1)
                    + (1.0 - wx) * wy * (1.0 - wz) * sample(0, 1, 0)
                    + (1.0 - wx) * (1.0 - wy) * wz * sample(0, 0, 1)
                    + (1.0 - wx) * (1.0 - wy) * (1.0 - wz) * sample(0, 0, 0);
                count += 1;
            }
        }
        true
    }

    pub fn header_get_pixel_spacing(&self) -> [f32; 3] {
        [
            self.header.cell_dim[0] / self.header.mx as f32,
            self.header.cell_dim[1] / self.header.my as f32,
            self.header.cell_dim[2] / self.header.mz as f32,
        ]
    }
    pub fn get_type(&self) -> usize {
        match self.header.mode {
            0 => 1,
            1 | 6 => 2,
            2 | 3 => 4,
            4 => 8,
            mode => panic!("wrong MRC mode {mode}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{IoMrc, MrcVolume};

    #[test]
    fn writes_source_compatible_header_and_owned_float_data() {
        let path = std::env::temp_dir().join(format!("io-mrc-vol-{}.mrc", std::process::id()));
        let mut volume = IoMrc::new(MrcVolume::Float(vec![1.0, 2.0, 3.0, 4.0]), 2, 2, 1, 2);
        assert!(volume.write_mrc_file(&path, true));
        let raw = std::fs::read(&path).unwrap();
        assert_eq!(raw.len(), 1024 + 16);
        assert_eq!(i32::from_ne_bytes(raw[0..4].try_into().unwrap()), 2);
        assert_eq!(&raw[208..212], b"MAP ");
        assert_eq!(f32::from_ne_bytes(raw[76..80].try_into().unwrap()), 1.0);
        assert_eq!(f32::from_ne_bytes(raw[80..84].try_into().unwrap()), 4.0);
        let mut read = IoMrc::default();
        assert!(read.read_mrc_file(&path));
        assert_eq!(read.volume, volume.volume);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn patch_and_trilinear_order_follow_source() {
        let volume = IoMrc::new(MrcVolume::Byte((0..125).collect()), 5, 5, 5, 0);
        let mut patch = [0.0; 4];
        assert!(volume.read_mrc_patch(1, 2, 2, 1, 1, &mut patch));
        assert_eq!(patch, [31.0, 36.0, 32.0, 37.0]);
        let mut plane = [0.0; 1];
        assert!(volume.get_2d_patch(
            &[1.5, 1.5, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1,
            1,
            &mut plane
        ));
        assert_eq!(plane, [46.5]);
    }
}
