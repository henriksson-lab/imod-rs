use imod_core::ImodError;

/// Per-section metadata extracted from an MRC extended header.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SectionMetadata {
    /// Tilt angle in degrees (if available).
    pub tilt_angle: Option<f32>,
    /// Defocus in micrometers (if available).
    pub defocus: Option<f32>,
    /// Exposure dose for this section (if available).
    pub dose: Option<f32>,
    /// X-axis stage position (if available).
    pub x_stage: Option<f32>,
    /// Y-axis stage position (if available).
    pub y_stage: Option<f32>,
    /// Magnification (if available).
    pub magnification: Option<f32>,
    /// Pixel size in Angstroms (if available from ext header).
    pub pixel_size: Option<f32>,
    /// Exposure time in seconds (if available).
    pub exposure_time: Option<f32>,
}

/// Parse SERI-type extended header data into per-section metadata.
///
/// The SERI format stores 32 bytes per section with the following layout:
///   offset  0: f32 tilt_angle
///   offset  4: f32 defocus (micrometers)
///   offset  8: f32 dose
///   offset 12: f32 x_stage
///   offset 16: f32 y_stage
///   offset 20: f32 magnification
///   offset 24-31: reserved
pub fn parse_seri_extended_header(
    ext_data: &[u8],
    nz: usize,
) -> Result<Vec<SectionMetadata>, ImodError> {
    const SERI_SECTION_SIZE: usize = 32;

    if ext_data.len() < nz * SERI_SECTION_SIZE {
        return Err(ImodError::InvalidData(format!(
            "SERI extended header too small: {} bytes for {} sections (need {})",
            ext_data.len(),
            nz,
            nz * SERI_SECTION_SIZE
        )));
    }

    let mut sections = Vec::with_capacity(nz);
    for i in 0..nz {
        let base = i * SERI_SECTION_SIZE;
        let s = &ext_data[base..base + SERI_SECTION_SIZE];

        let tilt_angle = f32::from_le_bytes([s[0], s[1], s[2], s[3]]);
        let defocus = f32::from_le_bytes([s[4], s[5], s[6], s[7]]);
        let dose = f32::from_le_bytes([s[8], s[9], s[10], s[11]]);
        let x_stage = f32::from_le_bytes([s[12], s[13], s[14], s[15]]);
        let y_stage = f32::from_le_bytes([s[16], s[17], s[18], s[19]]);
        let magnification = f32::from_le_bytes([s[20], s[21], s[22], s[23]]);

        sections.push(SectionMetadata {
            tilt_angle: Some(tilt_angle),
            defocus: Some(defocus),
            dose: Some(dose),
            x_stage: Some(x_stage),
            y_stage: Some(y_stage),
            magnification: Some(magnification),
            pixel_size: None,
            exposure_time: None,
        });
    }
    Ok(sections)
}

/// Parse FEI-type extended header data into per-section metadata.
///
/// The FEI extended header stores 128 bytes per section with the layout:
///   offset   0: f64 tilt_angle (degrees)
///   offset   8: f64 defocus (meters, converted to micrometers)
///   offset  16: f64 x_stage (meters)
///   offset  24: f64 y_stage (meters)
///   offset  32: f64 dose
///   offset  40: f64 magnification
///   offset  48: f64 pixel_size (meters, converted to Angstroms)
///   offset  56: f64 reserved
///   offset  64: f64 reserved
///   offset  72: f64 reserved
///   offset  80: f64 exposure_time (seconds)
///   offset 88-127: reserved / additional FEI metadata
pub fn parse_fei_extended_header(
    ext_data: &[u8],
    nz: usize,
) -> Result<Vec<SectionMetadata>, ImodError> {
    const FEI_SECTION_SIZE: usize = 128;

    if ext_data.len() < nz * FEI_SECTION_SIZE {
        return Err(ImodError::InvalidData(format!(
            "FEI extended header too small: {} bytes for {} sections (need {})",
            ext_data.len(),
            nz,
            nz * FEI_SECTION_SIZE
        )));
    }

    let mut sections = Vec::with_capacity(nz);
    for i in 0..nz {
        let base = i * FEI_SECTION_SIZE;
        let s = &ext_data[base..base + FEI_SECTION_SIZE];

        let read_f64 = |offset: usize| -> f64 {
            f64::from_le_bytes([
                s[offset],
                s[offset + 1],
                s[offset + 2],
                s[offset + 3],
                s[offset + 4],
                s[offset + 5],
                s[offset + 6],
                s[offset + 7],
            ])
        };

        let tilt_angle = read_f64(0) as f32;
        // FEI stores defocus in meters; convert to micrometers
        let defocus = (read_f64(8) * 1e6) as f32;
        let x_stage = read_f64(16) as f32;
        let y_stage = read_f64(24) as f32;
        let dose = read_f64(32) as f32;
        let magnification = read_f64(40) as f32;
        // FEI stores pixel size in meters; convert to Angstroms
        let pixel_size = (read_f64(48) * 1e10) as f32;
        // Exposure time in seconds (offset 80)
        let exposure_time = read_f64(80) as f32;

        sections.push(SectionMetadata {
            tilt_angle: Some(tilt_angle),
            defocus: Some(defocus),
            dose: Some(dose),
            x_stage: Some(x_stage),
            y_stage: Some(y_stage),
            magnification: Some(magnification),
            pixel_size: if pixel_size > 0.0 {
                Some(pixel_size)
            } else {
                None
            },
            exposure_time: if exposure_time > 0.0 {
                Some(exposure_time)
            } else {
                None
            },
        });
    }
    Ok(sections)
}
