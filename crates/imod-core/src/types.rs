/// MRC data modes, matching the C defines in mrcfiles.h
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum MrcMode {
    /// 8-bit unsigned integer (mode 0).
    Byte = 0,
    /// 16-bit signed integer (mode 1).
    Short = 1,
    /// 32-bit IEEE float (mode 2).
    Float = 2,
    /// Complex pair of 16-bit signed integers (mode 3).
    ComplexShort = 3,
    /// Complex pair of 32-bit floats (mode 4).
    ComplexFloat = 4,
    /// 16-bit unsigned integer (mode 6).
    UShort = 6,
    /// 16-bit IEEE half-precision float (mode 12).
    HalfFloat = 12,
    /// 3-channel 8-bit RGB (mode 16).
    Rgb = 16,
    /// 4-bit unsigned integer, two pixels per byte (mode 101).
    FourBit = 101,
}

impl MrcMode {
    /// Convert an `i32` mode value from the MRC header to the corresponding enum variant.
    pub fn from_i32(value: i32) -> Option<Self> {
        match value {
            0 => Some(Self::Byte),
            1 => Some(Self::Short),
            2 => Some(Self::Float),
            3 => Some(Self::ComplexShort),
            4 => Some(Self::ComplexFloat),
            6 => Some(Self::UShort),
            12 => Some(Self::HalfFloat),
            16 => Some(Self::Rgb),
            101 => Some(Self::FourBit),
            _ => None,
        }
    }

    /// Bytes per pixel for this mode (for complex types, bytes per complex element)
    pub fn bytes_per_pixel(&self) -> usize {
        match self {
            Self::Byte => 1,
            Self::Short => 2,
            Self::Float => 4,
            Self::ComplexShort => 4,
            Self::ComplexFloat => 8,
            Self::UShort => 2,
            Self::HalfFloat => 2,
            Self::Rgb => 3,
            Self::FourBit => 1, // 2 pixels per byte, handled specially
        }
    }

    /// Number of channels (components per pixel)
    pub fn channels(&self) -> usize {
        match self {
            Self::Rgb => 3,
            Self::ComplexShort | Self::ComplexFloat => 2,
            _ => 1,
        }
    }
}

/// Extended header types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtHeaderType {
    /// No extended header present (all-zero bytes).
    None,
    /// SerialEM extended header format.
    Seri,
    /// Agard-style extended header format.
    Agar,
    /// FEI/ThermoFisher extended header format (FEI1 or FEI2).
    Fei,
    /// Unrecognised extended header type.
    Unknown,
}

impl ExtHeaderType {
    /// Parse the 4-byte extended header type field from the MRC header.
    pub fn from_bytes(bytes: &[u8; 4]) -> Self {
        match bytes {
            b"SERI" => Self::Seri,
            b"AGAR" => Self::Agar,
            b"FEI1" | b"FEI2" => Self::Fei,
            [0, 0, 0, 0] => Self::None,
            _ => Self::Unknown,
        }
    }

    /// Encode this variant back to the 4-byte on-disk representation.
    pub fn to_bytes(&self) -> [u8; 4] {
        match self {
            Self::None => [0; 4],
            Self::Seri => *b"SERI",
            Self::Agar => *b"AGAR",
            Self::Fei => *b"FEI1",
            Self::Unknown => [0; 4],
        }
    }
}

/// IMOD units for pixel size
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelUnit {
    /// No unit / raw pixel coordinates.
    Pixel,
    /// Meters.
    Meter,
    /// Centimeters.
    Cm,
    /// Millimeters.
    Mm,
    /// Micrometers.
    Um,
    /// Nanometers.
    Nm,
    /// Angstroms (0.1 nm).
    Angstrom,
    /// Picometers.
    Pm,
    /// Kilometers.
    Kilo,
}

impl PixelUnit {
    /// Convert the integer unit code stored in the MRC header to a `PixelUnit`.
    pub fn from_i32(value: i32) -> Self {
        match value {
            0 => Self::Pixel,
            1 => Self::Meter,
            -2 => Self::Cm,
            -3 => Self::Mm,
            -6 => Self::Um,
            -9 => Self::Nm,
            -10 => Self::Angstrom,
            -12 => Self::Pm,
            3 => Self::Kilo,
            _ => Self::Pixel,
        }
    }
}

/// 3D point (matches Ipoint in IMOD)
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Point3f {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}
