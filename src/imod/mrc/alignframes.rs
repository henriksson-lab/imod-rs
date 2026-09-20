//! Bottom-up owned portions of `alignframes.{h,cpp}`.
//!
//! High-level image-unit, TIFF/EER, and GPU orchestration remains dependent on
//! unfinished frame I/O/framealign layers.  This module intentionally contains
//! only source methods whose inputs are already owned Rust values.

use super::framealign::FrameAlign;
use crate::imod::libcfshr::autodoc::adoc_open_image_metadata;
use crate::imod::libcfshr::b3dutil::ImodFile;
use crate::imod::libcfshr::rotateflip::{RotateFlipData, rotate_flip_image};
use crate::imod::libcfshr::samplemeansd::{sample_mean_only, type_for_sample_mean};
use crate::imod::libiimod::iimage::OwnedImageStack;
use crate::imod::libiimod::mrcfiles::{
    MRC_MODE_FLOAT, MRC_MODE_SHORT, MRC_MODE_USHORT, MrcHeader, mrc_head_new, mrc_head_read,
    mrc_head_write, mrc_write_slice,
};
use std::io::Write;

#[derive(Clone, Debug, PartialEq)]
pub struct AliFrame {
    pub parallel_read: bool,
    pub wall_read: f64,
    pub partial_thresh: [f32; 2],
    pub debug: i32,
    pub use_gpu: i32,
    pub gpu_flags: i32,
    pub gpu_mem_limit: f32,
    pub memory_limit: f32,
    pub test_mode: i32,
    pub trunc_limit: f32,
    pub group_size: i32,
    pub use_block_group: bool,
    pub dose_file_type: i32,
    pub max_frame_doses: i32,
    pub total_dose: f32,
    pub dose_accumulates: i32,
    pub default_byte_scale: f32,
    pub rotation_flip: i32,
    pub sum_rotation_flip: i32,
    pub num_out_files: i32,
    pub initial_dose: f32,
    pub dose_scaling: f32,
}

/// Owned non-metadata result of `processDoseWeightingOptions`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DoseWeightingOptions {
    pub dose_scaling: f32,
    pub reweight_ones: Option<Vec<f32>>,
    pub fixed_frame_doses: Option<String>,
    pub total_doses: Vec<f32>,
    pub prior_doses: Vec<f32>,
    pub frame_dose_lines: Vec<String>,
}

/// Parsed fields consumed by `AliFrame::getAnglesAndTitlesFromMdoc`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MdocTiltTitleInput {
    pub tilt_angles: Vec<f32>,
    pub frame_ts_start_end: Vec<Option<(i32, i32)>>,
    pub pixel_spacing: Option<f32>,
    pub image_size: Option<(i32, i32)>,
    pub titles: Vec<String>,
    pub global_title: Option<String>,
    pub first_rotation_angle: Option<f32>,
    pub frame_set_rotation_angle: Option<f32>,
}

/// Owned result of `getAnglesAndTitlesFromMdoc` after metadata I/O.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MdocTiltTitleResult {
    pub tilt_records: TiltAngleRecords,
    pub pixel_spacing: Option<f32>,
    pub image_size: Option<(i32, i32)>,
    pub titles: Vec<String>,
    pub axis_angle: Option<f32>,
    pub are_fei_frames: bool,
}

/// Parsed dose values returned by the metadata subsystem for each Mdoc section.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MdocDoseResult {
    pub dose_from_mdoc: Vec<f32>,
    pub prior_from_mdoc: Vec<f32>,
    pub iz_piece: Vec<i32>,
    pub frame_dose_lines: Vec<String>,
}

/// File names discovered by `checkTitlesForRefNames`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReferenceNamesFromTitles {
    pub gain_name: Option<std::path::PathBuf>,
    pub defect_name: Option<std::path::PathBuf>,
}

/// Result retained by `AliFrame::openMdocFile` for subsequent autodoc access.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OpenMdocFile {
    pub autodoc_index: i32,
    pub montage: i32,
    pub number_of_sections: i32,
    pub autodoc_type: i32,
}

/// Inputs to the source's post-parse dose reconciliation.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct UnifiedDoseInput {
    pub number_of_files: usize,
    pub dose_file_type: i32,
    pub fixed_frame_doses: Option<String>,
    pub fixed_total_dose: Option<f32>,
    pub frame_dose_lines: Vec<String>,
    pub mdoc_doses: Option<MdocDoseResult>,
    pub dose_accumulates: i32,
    pub maximum_frames: usize,
}

/// `unifyDoseInformation` output used by each downstream alignment set.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct UnifiedDoseInformation {
    pub dose_file_type: i32,
    pub total_doses: Vec<f32>,
    pub prior_doses: Vec<f32>,
    pub frame_dose_lines: Vec<String>,
}

/// Owned image references consumed by `getGainDarkDefects` after their file
/// readers have completed.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GainDarkDefectsInput {
    pub image_dimensions: (usize, usize),
    pub gain: Option<OwnedImageStack>,
    pub dark: Option<OwnedImageStack>,
    pub frames_are_eer: bool,
    pub gain_is_tiff: bool,
}

/// Validated gain/dark images for the FrameAlign boundary.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GainDarkDefects {
    pub gain: Option<Vec<u8>>,
    pub gain_dimensions: Option<(usize, usize)>,
    pub dark: Option<Vec<u8>>,
    pub super_resolution_factor: usize,
}

/// Backend-reported inputs to `assessGpuNeeds` after GPU discovery and
/// FrameAlign allocation estimation.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GpuNeedsInput {
    pub gpu_requested: bool,
    pub gpu_available_memory: f32,
    pub gpu_memory_limit: f32,
    pub sum_memory_need: f32,
    pub alignment_memory_need: f32,
    pub sum_pad_size: f32,
    pub multiple_outputs: bool,
    pub test_mode: bool,
    pub getting_frc: bool,
}

/// Source GPU flags and whether FRC even/odd output remains feasible.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GpuNeeds {
    pub flags: u32,
    pub usable_memory: f32,
    pub getting_frc: bool,
}

/// `AliFrame()`: construct the native command state with its source defaults.
pub fn ali_frame() -> AliFrame {
    AliFrame::default()
}

/// Typed replacement for the input/output pointer pairs in
/// `AliFrame::addToSumBuffer`.
#[derive(Debug, PartialEq)]
pub enum AliFrameSumBuffer {
    Short(Vec<i16>),
    UShort(Vec<u16>),
    Float(Vec<f32>),
}

#[derive(Clone, Copy, Debug)]
pub enum AliFrameInputPixels<'a> {
    Byte(&'a [u8]),
    Short(&'a [i16]),
    UShort(&'a [u16]),
    Float(&'a [f32]),
}
impl Default for AliFrame {
    fn default() -> Self {
        Self {
            parallel_read: false,
            wall_read: 0.,
            partial_thresh: [0.; 2],
            debug: 0,
            use_gpu: -1,
            gpu_flags: 0,
            gpu_mem_limit: 0.,
            memory_limit: 12.,
            test_mode: 0,
            trunc_limit: 0.,
            group_size: 1,
            use_block_group: false,
            dose_file_type: -1,
            max_frame_doses: 0,
            total_dose: 0.,
            dose_accumulates: -1,
            default_byte_scale: 30.,
            rotation_flip: 0,
            sum_rotation_flip: -10,
            num_out_files: 1,
            initial_dose: 0.,
            dose_scaling: 1.,
        }
    }
}
impl AliFrame {
    /// `AliFrame::assessGpuNeeds`: select GPU summing/alignment based on
    /// translated FrameAlign byte estimates and backend-reported capacity.
    pub fn assess_gpu_needs(input: GpuNeedsInput) -> GpuNeeds {
        use super::framealign::{GPU_FOR_ALIGNING, GPU_FOR_SUMMING};
        const GPU_DO_EVEN_ODD: u32 = 1 << 1;
        if !input.gpu_requested || input.gpu_available_memory <= 0. {
            return GpuNeeds::default();
        }
        let usable = if input.gpu_memory_limit > 0. {
            input.gpu_memory_limit * 1024. * 1024. * 1024.
        } else if input.gpu_memory_limit < 0. {
            -input.gpu_available_memory * input.gpu_memory_limit
        } else {
            input.gpu_available_memory * 0.85
        };
        let mut result = GpuNeeds {
            usable_memory: usable,
            getting_frc: input.getting_frc,
            ..Default::default()
        };
        let mut used = 0.;
        if !input.test_mode && input.sum_memory_need <= usable {
            result.flags |= GPU_FOR_SUMMING;
            used = input.sum_memory_need
                + if input.multiple_outputs {
                    input.sum_pad_size
                } else {
                    0.
                };
        }
        if input.alignment_memory_need + used <= usable {
            result.flags |= GPU_FOR_ALIGNING;
        }
        if result.flags & GPU_FOR_SUMMING != 0 && result.getting_frc {
            if used + input.sum_pad_size <= usable {
                result.flags |= GPU_DO_EVEN_ODD;
            } else {
                result.getting_frc = false;
            }
        }
        result
    }

    /// `AliFrame::getGainDarkDefects`: validate and extract the first owned
    /// gain/dark reference sections after MRC/TIFF reading and defect parsing.
    pub fn get_gain_dark_defects(input: GainDarkDefectsInput) -> Result<GainDarkDefects, String> {
        let (nx, ny) = input.image_dimensions;
        let mut result = GainDarkDefects::default();
        if let Some(gain) = input.gain {
            if gain.mode != MRC_MODE_FLOAT {
                return Err("gain reference must be floating point".into());
            }
            let x_factor = nx
                .checked_div(gain.nx)
                .ok_or("gain reference width is zero")?;
            let y_factor = ny
                .checked_div(gain.ny)
                .ok_or("gain reference height is zero")?;
            let exact =
                x_factor == y_factor && gain.nx * x_factor == nx && gain.ny * y_factor == ny;
            if (input.frames_are_eer || input.gain_is_tiff)
                && (!exact || !matches!(x_factor, 1 | 2 | 4))
            {
                return Err(
                    "image size must match the gain reference or be 2x/4x super-resolution".into(),
                );
            }
            if (gain.nx < nx || gain.ny < ny) && !(input.frames_are_eer || input.gain_is_tiff) {
                return Err("gain reference is smaller than input image".into());
            }
            result.super_resolution_factor = x_factor;
            result.gain_dimensions = Some((gain.nx, gain.ny));
            result.gain = Some(
                gain.frames
                    .into_iter()
                    .next()
                    .ok_or("gain reference has no sections")?,
            );
        }
        if let Some(dark) = input.dark {
            if !matches!(dark.mode, MRC_MODE_SHORT | MRC_MODE_USHORT) {
                return Err("dark reference must be signed or unsigned short".into());
            }
            if (dark.nx, dark.ny) != (nx, ny) {
                return Err("dark reference is not the same size as the image".into());
            }
            result.dark = Some(
                dark.frames
                    .into_iter()
                    .next()
                    .ok_or("dark reference has no sections")?,
            );
        }
        Ok(result)
    }

    /// `AliFrame::unifyDoseInformation`, reconciled after file/Mdoc parsing.
    pub fn unify_dose_information(
        input: &UnifiedDoseInput,
    ) -> Result<UnifiedDoseInformation, String> {
        let mut result = UnifiedDoseInformation {
            dose_file_type: input.dose_file_type,
            ..Default::default()
        };
        if let Some(total) = input.fixed_total_dose.filter(|total| *total > 0.) {
            result.total_doses = vec![total; input.number_of_files];
            result.prior_doses = vec![0.; input.number_of_files];
        } else if let Some(line) = &input.fixed_frame_doses {
            let (_, total) = Self::expand_frame_doses_numbers(line, input.maximum_frames)?;
            result.dose_file_type = 5;
            result.frame_dose_lines = vec![line.clone(); input.number_of_files];
            result.total_doses = vec![total; input.number_of_files];
            result.prior_doses = vec![0.; input.number_of_files];
        } else if input.dose_file_type == 4 {
            let mdoc = input
                .mdoc_doses
                .as_ref()
                .ok_or("mdoc dose file requires parsed metadata doses")?;
            if mdoc.dose_from_mdoc.len() < input.number_of_files {
                return Err("mdoc has fewer dose entries than aligned files".into());
            }
            result.total_doses = mdoc.dose_from_mdoc[..input.number_of_files].to_vec();
            result.prior_doses = if input.dose_accumulates > 0 {
                mdoc.prior_from_mdoc[..input.number_of_files].to_vec()
            } else {
                vec![0.; input.number_of_files]
            };
            result.frame_dose_lines = mdoc.frame_dose_lines.clone();
        } else if input.dose_file_type > 0 {
            if input.frame_dose_lines.len() < input.number_of_files {
                return Err("dose file has fewer lines than aligned files".into());
            }
            result.frame_dose_lines = input.frame_dose_lines[..input.number_of_files].to_vec();
            for line in &result.frame_dose_lines {
                if input.dose_file_type > 4 {
                    let (_, total) = Self::expand_frame_doses_numbers(line, input.maximum_frames)?;
                    result.total_doses.push(total);
                    result.prior_doses.push(0.);
                }
            }
        }
        if result.total_doses.len() == input.number_of_files
            && input.dose_accumulates > 0
            && result.prior_doses.iter().all(|dose| *dose == 0.)
        {
            let mut prior = 0.;
            for (index, dose) in result.total_doses.iter().enumerate() {
                result.prior_doses[index] = prior;
                prior += dose;
            }
        }
        Ok(result)
    }

    /// `AliFrame::openMdocFile`, using the translated image-metadata autodoc
    /// reader and retaining its index and image-stack description.
    pub fn open_mdoc_file(path: impl AsRef<std::path::Path>) -> Result<OpenMdocFile, String> {
        let path = path.as_ref();
        let mut montage = 0;
        let mut number_of_sections = 0;
        let mut autodoc_type = 0;
        let index = adoc_open_image_metadata(
            path.to_string_lossy().as_bytes(),
            0,
            &mut montage,
            &mut number_of_sections,
            &mut autodoc_type,
        );
        match index {
            -2 => Err(format!("metadata file {} does not exist", path.display())),
            -3 => Err(format!(
                "metadata file {} has no image-stack information",
                path.display()
            )),
            value if value < 0 => Err(format!("cannot open or read mdoc file {}", path.display())),
            autodoc_index => Ok(OpenMdocFile {
                autodoc_index,
                montage,
                number_of_sections,
                autodoc_type,
            }),
        }
    }

    /// `AliFrame::readOneFrame`: return the requested complete section after
    /// its MRC/TIFF reader boundary has produced an owned image stack.
    pub fn read_one_frame(stack: &OwnedImageStack, section: usize) -> Result<Vec<u8>, String> {
        stack
            .frames
            .get(section)
            .cloned()
            .ok_or_else(|| format!("frame section {section} is out of range"))
    }

    /// `AliFrame::checkTitlesForRefNames`: locate gain and defect references
    /// declared in frame labels, after the image reader supplies EER metadata.
    pub fn check_titles_for_ref_names(
        frame_file: impl AsRef<std::path::Path>,
        titles: &[String],
        eer_gain_reference: Option<&str>,
    ) -> Result<ReferenceNamesFromTitles, String> {
        let frame_file = frame_file.as_ref();
        if let Some(gain) = eer_gain_reference {
            let name = gain
                .rsplit(['/', '\\'])
                .next()
                .filter(|name| !name.is_empty())
                .ok_or("EER metadata gain reference has no file name")?;
            return Ok(ReferenceNamesFromTitles {
                gain_name: Some(std::path::PathBuf::from(name)),
                ..Default::default()
            });
        }
        let directory = frame_file
            .parent()
            .unwrap_or_else(|| std::path::Path::new(""));
        let mut result = ReferenceNamesFromTitles::default();
        for title in titles {
            let title = title.trim();
            let lower = title.to_ascii_lowercase();
            let gain = (lower.contains("ref"))
                && [".mrc", ".dm4", ".tif"]
                    .iter()
                    .any(|extension| lower.ends_with(extension));
            let defect = lower.contains("defect") && lower.ends_with(".txt");
            if (gain && result.gain_name.is_none()) || (defect && result.defect_name.is_none()) {
                let candidate = directory.join(title);
                if candidate.is_file() {
                    if gain {
                        result.gain_name = Some(candidate);
                    } else {
                        result.defect_name = Some(candidate);
                    }
                }
            }
        }
        Ok(result)
    }

    /// `AliFrame::getDosesFromMdoc`: allocate the source result vectors and
    /// receive already-parsed metadata doses from the autodoc boundary.
    pub fn get_doses_from_mdoc(
        number_of_sections: usize,
        number_of_input_files: usize,
        doses: &[f32],
        priors: &[f32],
        iz_piece: &[i32],
        frame_dose_lines: &[String],
    ) -> Result<MdocDoseResult, String> {
        if doses.len() != number_of_sections
            || priors.len() != number_of_sections
            || iz_piece.len() != number_of_sections
        {
            return Err("mdoc dose metadata does not match its section count".into());
        }
        if frame_dose_lines.len() > number_of_input_files {
            return Err("mdoc frame-dose lines exceed input files".into());
        }
        Ok(MdocDoseResult {
            dose_from_mdoc: doses.to_vec(),
            prior_from_mdoc: priors.to_vec(),
            iz_piece: iz_piece.to_vec(),
            frame_dose_lines: frame_dose_lines.to_vec(),
        })
    }

    /// `AliFrame::getAnglesAndTitlesFromMdoc`, with autodoc lookup replaced by
    /// the parsed fields supplied in `MdocTiltTitleInput`.
    pub fn get_angles_and_titles_from_mdoc(
        input: &MdocTiltTitleInput,
        read_angles: bool,
        doing_frame_ts: bool,
        axis_angle: Option<f32>,
        angles_only: bool,
    ) -> Result<MdocTiltTitleResult, String> {
        let mut result = MdocTiltTitleResult {
            pixel_spacing: input.pixel_spacing,
            image_size: input.image_size,
            ..Default::default()
        };
        if read_angles {
            if doing_frame_ts && input.frame_ts_start_end.len() > input.tilt_angles.len() {
                return Err("mdoc saved-frame fields exceed tilt-angle sections".into());
            }
            result.tilt_records.angles = input.tilt_angles.clone();
            if doing_frame_ts {
                for item in input.tilt_angles.iter().enumerate() {
                    let pair = input.frame_ts_start_end.get(item.0).copied().flatten();
                    result
                        .tilt_records
                        .relative_starts
                        .push(pair.map_or(-1, |v| v.0));
                    result
                        .tilt_records
                        .relative_ends
                        .push(pair.map_or(-1, |v| v.1));
                    result.tilt_records.relative_frame_starts_found |= pair.is_some();
                }
            }
        }
        if angles_only {
            return Ok(result);
        }
        let mut axis = axis_angle;
        let mut got_axis = false;
        let titles = if input.titles.is_empty() {
            input.global_title.iter().cloned().collect()
        } else {
            input.titles.clone()
        };
        for mut title in titles {
            if title.contains("TiltAxisAngle") {
                result.are_fei_frames = true;
                if axis.is_none() {
                    let title_angle = title.split_once('=').and_then(|(_, value)| {
                        let value = value.trim_start();
                        value.split([' ', ',']).next()?.parse::<f32>().ok()
                    });
                    if let (Some(title_angle), Some(rotation)) =
                        (title_angle, input.first_rotation_angle)
                    {
                        let corrected = if (-(rotation + 90.) - title_angle).abs() < 0.11 {
                            Some(rotation)
                        } else if ((rotation - 90.) - title_angle).abs() < 0.11 {
                            Some(title_angle)
                        } else if (-(rotation - 90.) - title_angle).abs() < 0.11 {
                            Some(-title_angle)
                        } else {
                            None
                        };
                        if let Some(value) = corrected {
                            axis = Some(value);
                            got_axis = true;
                            let suffix = title
                                .split_once('=')
                                .map(|(_, value)| {
                                    value.trim_start().trim_start_matches(|c: char| {
                                        c.is_ascii_digit()
                                            || matches!(c, '.' | '-' | '+' | 'e' | 'E')
                                    })
                                })
                                .unwrap_or("");
                            title = format!("  Tilt axis angle = {value:.2}{suffix}");
                        }
                    }
                }
            } else if title.contains("Tilt axis angle") && !title.starts_with(' ') && axis.is_none()
            {
                got_axis = true;
                title = format!("    {title}");
            }
            result.titles.push(title);
        }
        if !got_axis && axis.is_none() {
            axis = input.frame_set_rotation_angle.map(|value| value - 90.);
        }
        result.axis_angle = axis;
        Ok(result)
    }

    /// `processDoseWeightingOptions`: process command values and non-mdoc dose
    /// files.  Mdoc lookup remains a separate owned metadata boundary.
    pub fn process_dose_weighting_options(
        fixed_frame_doses: Option<&str>,
        voltage: i32,
        dose_scaling: f32,
        normalize: bool,
        dose_file_type: i32,
        dose_file_text: Option<&str>,
    ) -> Result<DoseWeightingOptions, String> {
        let mut options = DoseWeightingOptions {
            dose_scaling,
            reweight_ones: normalize.then(|| vec![1.; 9000]),
            fixed_frame_doses: fixed_frame_doses.map(ToOwned::to_owned),
            ..Default::default()
        };
        match voltage {
            300 => {}
            200 => options.dose_scaling *= 0.8,
            _ => return Err("voltage must be either 200 or 300".into()),
        }
        if dose_file_type <= 0 {
            return Ok(options);
        }
        if dose_file_type == 4 {
            return Err("mdoc dose metadata requires the metadata boundary".into());
        }
        let text = dose_file_text.ok_or("a dose weighting file is required")?;
        for line in text.lines().filter(|line| !line.trim().is_empty()) {
            if dose_file_type == 1 {
                options
                    .total_doses
                    .push(line.trim().parse().map_err(|_| "invalid total dose")?);
                options.prior_doses.push(0.);
            } else if dose_file_type > 4 {
                options.frame_dose_lines.push(line.to_owned());
            } else {
                let values = line
                    .split_whitespace()
                    .map(str::parse::<f32>)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|_| "invalid prior and dose values")?;
                if values.len() < 2 {
                    return Err("dose line needs prior and dose values".into());
                }
                options.prior_doses.push(values[0]);
                options.total_doses.push(if dose_file_type == 3 {
                    values[1] - values[0]
                } else {
                    values[1]
                });
            }
        }
        Ok(options)
    }

    /// `AliFrame::analyzeExtraHeader`, operating on the already-read float
    /// extended header instead of a borrowed C `FILE *` and scratch buffer.
    pub fn analyze_extra_header(
        header: &MrcHeader,
        extended: &[f32],
        frame_range: Option<(usize, usize)>,
        axis_pix_only: bool,
    ) -> Result<ExtendedHeaderAnalysis, String> {
        let fields = usize::try_from(header.nint.max(0)).unwrap_or(0)
            + usize::try_from(header.nreal.max(0)).unwrap_or(0);
        let sections = usize::try_from(header.nz.max(0)).unwrap_or(0);
        if fields == 0 || sections == 0 {
            return Ok(ExtendedHeaderAnalysis::default());
        }
        if extended.len() < fields.saturating_mul(sections) {
            return Err("extended header is shorter than its MRC dimensions".into());
        }
        let mut result = ExtendedHeaderAnalysis {
            axis_angle: -999.,
            ..Default::default()
        };
        if header.nreal >= 12 {
            let base = header.nint.max(0) as usize;
            let mut pixel = extended[base + 11];
            if !(0.05..100_000.).contains(&pixel) {
                pixel *= 1.0e10;
            }
            if (0.05..100_000.).contains(&pixel) {
                let mut axis = extended[base + 10];
                if (-360. ..=360.).contains(&axis) {
                    if axis < -180. {
                        axis += 360.;
                    }
                    if axis > 180. {
                        axis -= 360.;
                    }
                    result.pixel_size = pixel;
                    result.axis_angle = axis;
                    result.source_status = 1;
                }
            }
        }
        if axis_pix_only {
            return Ok(result);
        }
        let (start, end) = frame_range.unwrap_or((0, sections - 1));
        if start > end || end >= sections {
            return Err("extended-header frame range is invalid".into());
        }
        let tilt_at = |section: usize| extended[header.nint.max(0) as usize + section * fields];
        let mut last_size = 0_usize;
        let mut equal_sizes = 0_usize;
        let mut inserted_double = false;
        for section in start..=end {
            let tilt = tilt_at(section);
            if !(-180. ..=180.).contains(&tilt) {
                return Ok(result);
            }
            if result
                .tilts
                .last()
                .is_none_or(|last| (tilt - *last).abs() > 0.01)
            {
                if let Some(&previous_start) = result.set_starts.last() {
                    let mut size = section - previous_start;
                    if equal_sizes > 5 && size == 2 * last_size && !inserted_double {
                        result.set_starts.push(section - last_size);
                        result.tilts.push(tilt);
                        size = last_size;
                        inserted_double = true;
                    }
                    if size == last_size {
                        equal_sizes += 1;
                    } else {
                        equal_sizes = 1;
                        last_size = size;
                    }
                }
                result.set_starts.push(section);
                result.tilts.push(tilt);
            }
        }
        result.set_starts.push(end + 1);
        result.set_sizes = result
            .set_starts
            .windows(2)
            .map(|pair| pair[1] - pair[0])
            .collect();
        result.min_set_size = result.set_sizes.iter().copied().min().unwrap_or(0);
        result.max_set_size = result.set_sizes.iter().copied().max().unwrap_or(0);
        result.source_status += 2;
        Ok(result)
    }

    /// `rotateFlipGainReference()`: apply the source rotation/flip convention
    /// to an owned floating-point gain image and return its resulting shape.
    pub fn rotate_flip_gain_reference(
        reference: &mut Vec<f32>,
        nx: &mut usize,
        ny: &mut usize,
        rotation_flip: i32,
    ) -> Result<(), String> {
        if *nx == 0 || *ny == 0 || reference.len() != nx.saturating_mul(*ny) {
            return Err("gain reference dimensions do not match its pixels".into());
        }
        let input_nx = i32::try_from(*nx).map_err(|_| "gain reference width is too large")?;
        let input_ny = i32::try_from(*ny).map_err(|_| "gain reference height is too large")?;
        let mut output = vec![0.0; reference.len()];
        let (mut output_nx, mut output_ny) = (0, 0);
        if rotate_flip_image(
            RotateFlipData::Float {
                array: reference,
                brray: &mut output,
            },
            input_nx,
            input_ny,
            rotation_flip,
            0,
            0,
            0,
            &mut output_nx,
            &mut output_ny,
            0,
        ) != 0
        {
            return Err(format!("inappropriate rotation/flip value {rotation_flip}"));
        }
        *nx = output_nx as usize;
        *ny = output_ny as usize;
        *reference = output;
        Ok(())
    }

    /// `AliFrame::defectFileToString`: read nonempty defect-file records into
    /// the newline-delimited protocol form consumed by the alignment backend.
    pub fn defect_file_to_string(path: impl AsRef<std::path::Path>) -> Result<String, String> {
        let path = path.as_ref();
        let contents = std::fs::read_to_string(path)
            .map_err(|error| format!("failed to open defect file {}: {error}", path.display()))?;
        let mut output = String::new();
        for line in contents.lines() {
            if !line.is_empty() {
                output.push_str(line);
                output.push('\n');
            }
        }
        Ok(output)
    }

    /// `AliFrame::addToSumBuffer` (`alignframes.cpp:2886`).  Source selects
    /// the accumulator type from the MRC input mode; matching enum variants
    /// make invalid raw casts impossible and retain the native wrapping
    /// integer addition behavior.
    pub fn add_to_sum_buffer(
        input: AliFrameInputPixels<'_>,
        sum: &mut AliFrameSumBuffer,
    ) -> Result<(), String> {
        match (input, sum) {
            (AliFrameInputPixels::Byte(input), AliFrameSumBuffer::Short(sum)) => {
                if input.len() != sum.len() {
                    return Err("input and sum sizes differ".into());
                }
                for (source, target) in input.iter().zip(sum) {
                    *target = target.wrapping_add(*source as i16);
                }
            }
            (AliFrameInputPixels::Short(input), AliFrameSumBuffer::Short(sum)) => {
                if input.len() != sum.len() {
                    return Err("input and sum sizes differ".into());
                }
                for (source, target) in input.iter().zip(sum) {
                    *target = target.wrapping_add(*source);
                }
            }
            (AliFrameInputPixels::UShort(input), AliFrameSumBuffer::UShort(sum)) => {
                if input.len() != sum.len() {
                    return Err("input and sum sizes differ".into());
                }
                for (source, target) in input.iter().zip(sum) {
                    *target = target.wrapping_add(*source);
                }
            }
            (AliFrameInputPixels::Float(input), AliFrameSumBuffer::Float(sum)) => {
                if input.len() != sum.len() {
                    return Err("input and sum sizes differ".into());
                }
                for (source, target) in input.iter().zip(sum) {
                    *target += *source;
                }
            }
            _ => return Err("input mode does not match sum buffer mode".into()),
        }
        Ok(())
    }

    /// Owned translation of `AliFrame::openAndReadHeader`: unlike the C
    /// `FILE *` route, the returned handle and header remain coupled in Rust
    /// ownership and close automatically when the handle is dropped.
    pub fn open_and_read_header(
        filename: impl AsRef<std::path::Path>,
        description: &str,
        test_mode: bool,
    ) -> Result<(ImodFile, MrcHeader), String> {
        let path = filename.as_ref();
        let mut file = ImodFile::open(path, "rb").ok_or_else(|| {
            format!(
                "cannot open {description} file {}{}",
                path.display(),
                if test_mode {
                    "; do not specify an output file when not making sums"
                } else {
                    ""
                }
            )
        })?;
        let mut header = MrcHeader::default();
        if mrc_head_read(&mut file, &mut header) != 0 {
            return Err(format!(
                "cannot read header of {description} file {}",
                path.display()
            ));
        }
        Ok((file, header))
    }

    /// Owned-file equivalent of source `AliFrame::getNextFilename` for the
    /// command forms already supported by the MRC route: repeated input
    /// options, positional input names, or a list file.  Mdoc-derived frame
    /// names require the unfinished autodoc orchestration and are intentionally
    /// not fabricated here.
    pub fn get_next_filename(
        index: usize,
        option_inputs: &[std::path::PathBuf],
        positional_inputs: &[std::path::PathBuf],
        input_list: Option<&std::path::Path>,
    ) -> Result<Option<std::path::PathBuf>, String> {
        if let Some(list_path) = input_list {
            let text = std::fs::read_to_string(list_path).map_err(|error| {
                format!(
                    "cannot read input-file list {}: {error}",
                    list_path.display()
                )
            })?;
            return Ok(text
                .lines()
                .filter(|line| !line.trim().is_empty())
                .nth(index)
                .map(|line| std::path::PathBuf::from(line.trim())));
        }
        Ok(option_inputs.get(index).cloned().or_else(|| {
            positional_inputs
                .get(index.saturating_sub(option_inputs.len()))
                .cloned()
        }))
    }

    /// Source `checkInputFile`, performed on an owned stack before alignment
    /// creates any work buffers.
    pub fn check_input_file(
        filename: impl AsRef<std::path::Path>,
        stack: &OwnedImageStack,
        expected_size: Option<(usize, usize)>,
        combine: bool,
    ) -> Result<(), String> {
        if !matches!(stack.mode, 0 | 1 | 2 | 6) {
            return Err(format!(
                "file mode for {} is {}; only real MRC modes are supported",
                filename.as_ref().display(),
                stack.mode
            ));
        }
        if let Some((nx, ny)) = expected_size {
            if (stack.nx, stack.ny) != (nx, ny) {
                return Err(format!(
                    "file {} has dimensions {} x {}; expected {} x {}",
                    filename.as_ref().display(),
                    stack.nx,
                    stack.ny,
                    nx,
                    ny
                ));
            }
        }
        if combine && stack.frame_count() > 1 {
            return Err(format!(
                "file {} has more than one section and cannot be combined",
                filename.as_ref().display()
            ));
        }
        Ok(())
    }

    /// Owned translation of `readAnalyzeSavedFrameList`: parse SEMCCD saved
    /// frame numbers and identify contiguous frame sets separated by the
    /// source's gap/negative-number rules.
    pub fn read_analyze_saved_frame_list(
        list_path: impl AsRef<std::path::Path>,
        input_file_count: usize,
        break_set_size: usize,
        max_gap_within_set: i32,
    ) -> Result<SavedFrameSets, String> {
        if input_file_count != 1 {
            return Err("saved-frame lists require exactly one input file".into());
        }
        if break_set_size != 0 {
            return Err("saved-frame lists cannot be used with fixed frame-set breaking".into());
        }
        let saved_frames = std::fs::read_to_string(list_path.as_ref())
            .map_err(|error| {
                format!(
                    "cannot read saved frame list {}: {error}",
                    list_path.as_ref().display()
                )
            })?
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| {
                line.trim()
                    .parse::<i32>()
                    .map_err(|_| format!("invalid saved frame number {line}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if saved_frames.len() < 10 {
            return Err(format!(
                "there are only {} numbers in saved frame list {}",
                saved_frames.len(),
                list_path.as_ref().display()
            ));
        }
        let mut starts = Vec::new();
        let mut counts = Vec::new();
        let mut in_set = false;
        let mut single_sets_ok = false;
        let mut last_kept = -1;
        let mut start = 0;
        for (index, &saved) in saved_frames.iter().enumerate() {
            if saved < 0
                || (last_kept < 0 && saved >= 0)
                || (in_set && saved > last_kept + max_gap_within_set + 1)
            {
                if in_set {
                    starts.push(start);
                    counts.push(index - start);
                    in_set = false;
                }
                if saved >= 0 {
                    in_set = true;
                    start = index;
                } else if index + 1 < saved_frames.len() {
                    single_sets_ok = true;
                }
            }
            if saved >= 0 {
                last_kept = saved;
            }
        }
        if !single_sets_ok {
            single_sets_ok = saved_frames.len() < 4 * counts.len();
        }
        if in_set && (saved_frames.len() - start > 1 || single_sets_ok) {
            starts.push(start);
            counts.push(saved_frames.len() - start);
        }
        if counts.first() == Some(&1) && !single_sets_ok {
            starts.remove(0);
            counts.remove(0);
        }
        if starts.is_empty() {
            return Err("saved frame list contains no usable frame sets".into());
        }
        Ok(SavedFrameSets {
            saved_frames,
            starts,
            counts,
        })
    }

    /// Source `BreakFramesIntoSets` setup in the main processing loop.  Frame
    /// numbers are zero-based here, matching the MRC section indices used by
    /// the owned reader.
    pub fn break_frames_into_sets(
        frame_count: usize,
        break_set_size: usize,
        start_frame: Option<usize>,
        end_frame: Option<usize>,
    ) -> Result<SavedFrameSets, String> {
        if break_set_size < 2 {
            return Err("frame-set break size must be at least 2".into());
        }
        let first = start_frame.unwrap_or(0);
        let last = end_frame.unwrap_or(frame_count.saturating_sub(1));
        if first >= frame_count || last < first || last >= frame_count {
            return Err("selected frame range is outside the input stack".into());
        }
        let usable = last + 1 - first;
        let set_count = usable / break_set_size;
        if set_count == 0 {
            return Err("selected frame range is shorter than the frame-set size".into());
        }
        let mut starts = Vec::with_capacity(set_count);
        let mut counts = Vec::with_capacity(set_count);
        for set in 0..set_count {
            let set_start = first + set * usable / set_count;
            let set_end = first + (set + 1) * usable / set_count;
            starts.push(set_start);
            counts.push(set_end - set_start);
        }
        Ok(SavedFrameSets {
            saved_frames: (first..=last).map(|frame| frame as i32).collect(),
            starts,
            counts,
        })
    }

    /// Owned translation of `readTiltAngleFile`.  A missing first/second
    /// relative frame integer stays `-1`, just as in the source arrays.
    pub fn read_tilt_angle_file(
        path: impl AsRef<std::path::Path>,
    ) -> Result<TiltAngleRecords, String> {
        let text = std::fs::read_to_string(path.as_ref()).map_err(|error| {
            format!(
                "cannot read tilt-angle file {}: {error}",
                path.as_ref().display()
            )
        })?;
        let mut angles = Vec::new();
        let mut relative_starts = Vec::new();
        let mut relative_ends = Vec::new();
        let mut relative_frame_starts_found = false;
        for line in text.lines().filter(|line| !line.trim().is_empty()) {
            let mut fields = line.split_whitespace();
            let angle = fields
                .next()
                .ok_or_else(|| "tilt-angle line is empty".to_owned())?
                .parse::<f32>()
                .map_err(|_| format!("invalid tilt angle {line}"))?;
            let start = fields.next().map_or(Ok(-1), |value| {
                value
                    .parse::<i32>()
                    .map_err(|_| format!("invalid relative start frame {value}"))
            })?;
            let end = if let Some(value) = fields.next() {
                relative_frame_starts_found = true;
                value
                    .parse::<i32>()
                    .map_err(|_| format!("invalid relative end frame {value}"))?
            } else {
                -1
            };
            angles.push(angle);
            relative_starts.push(start);
            relative_ends.push(end);
        }
        Ok(TiltAngleRecords {
            angles,
            relative_starts,
            relative_ends,
            relative_frame_starts_found,
        })
    }

    /// Source `addAxisAngleTitle`, updating the next permitted fixed-width MRC
    /// label without a C string buffer.
    pub fn add_axis_angle_title(header: &mut MrcHeader, axis_angle: f32) {
        let label_index = (header.nlabl.max(0) as usize).min(8);
        let text = format!("    Tilt axis angle = {axis_angle:.1}");
        let mut label = [b' '; 80];
        let bytes = text.as_bytes();
        let count = bytes.len().min(label.len());
        label[..count].copy_from_slice(&bytes[..count]);
        header.labels[label_index] = label;
        header.nlabl = (label_index + 1) as i32;
    }

    /// Source `handleTooManyTiltAngles` overlap test, retaining only tilt
    /// records that cover at least one saved-frame set when relative frame
    /// limits are available.
    pub fn handle_too_many_tilt_angles(
        records: &mut TiltAngleRecords,
        saved_sets: &SavedFrameSets,
        expected_sets: usize,
    ) -> Result<usize, String> {
        if records.angles.len() != records.relative_starts.len()
            || records.angles.len() != records.relative_ends.len()
        {
            return Err("tilt-angle records have mismatched field counts".into());
        }
        if saved_sets.starts.len() != saved_sets.counts.len() || saved_sets.starts.is_empty() {
            return Err("saved-frame sets are invalid".into());
        }
        let mut removed = 0;
        if records.angles.len() > expected_sets && records.relative_frame_starts_found {
            let base = *saved_sets
                .saved_frames
                .get(saved_sets.starts[0])
                .ok_or_else(|| "first saved-frame set is outside saved-frame list".to_owned())?;
            for index in (0..records.angles.len()).rev() {
                let (start, end) = (records.relative_starts[index], records.relative_ends[index]);
                if start < 0 || end < 0 {
                    continue;
                }
                let overlaps =
                    saved_sets
                        .starts
                        .iter()
                        .zip(&saved_sets.counts)
                        .any(|(&set_start, &count)| {
                            saved_sets
                                .saved_frames
                                .get(set_start)
                                .is_some_and(|&saved_start| {
                                    let relative_start = saved_start - base;
                                    !(relative_start > end
                                        || relative_start + (count as i32) < start)
                                })
                        });
                if !overlaps {
                    records.angles.remove(index);
                    records.relative_starts.remove(index);
                    records.relative_ends.remove(index);
                    removed += 1;
                }
            }
        }
        if records.angles.len() < expected_sets {
            return Err(format!(
                "there are only {} tilt angles for {} frame sets",
                records.angles.len(),
                expected_sets
            ));
        }
        Ok(removed)
    }

    /// Source `analyzeForPartialFrames` over owned MRC sections.  It samples
    /// first, middle, and last frames, then removes underexposed endpoints.
    pub fn analyze_for_partial_frames(
        &self,
        stack: &OwnedImageStack,
        start: usize,
        end: usize,
    ) -> Result<PartialFrameSelection, String> {
        if start > end || end >= stack.frame_count() {
            return Err("partial-frame range is outside input MRC sections".into());
        }
        let mut selected_start = start;
        let mut selected_end = end;
        let mut dropped = Vec::new();
        let count = end + 1 - start;
        if (self.partial_thresh[0] <= 0. && self.partial_thresh[1] <= 0.) || count < 3 {
            return Ok(PartialFrameSelection {
                start,
                end,
                dropped,
            });
        }
        let data_type = type_for_sample_mean(stack.mode);
        if data_type < 0 {
            return Err("MRC mode cannot be sampled for partial-frame analysis".into());
        }
        let trim = stack.nx.min(stack.ny) / 20;
        let (nx_use, ny_use) = (stack.nx - 2 * trim, stack.ny - 2 * trim);
        let sample = 40_000usize.min(nx_use * ny_use) as f32 / (nx_use * ny_use) as f32;
        let mut means = [0.; 3];
        for (slot, frame_index) in [start, start + count / 2, end].into_iter().enumerate() {
            let frame = &stack.frames[frame_index];
            let row_bytes = frame.len() / stack.ny;
            let rows: Vec<&[u8]> = frame.chunks_exact(row_bytes).collect();
            if sample_mean_only(
                Some(&rows),
                data_type,
                stack.nx as i32,
                stack.ny as i32,
                sample,
                trim as i32,
                trim as i32,
                nx_use as i32,
                ny_use as i32,
                Some(&mut means[slot]),
            ) != 0
            {
                return Err("cannot compute sampled mean for partial-frame analysis".into());
            }
        }
        if self.partial_thresh[0] > 0. && means[0] < self.partial_thresh[0] * means[1] {
            dropped.push(selected_start);
            selected_start += 1;
        }
        if self.partial_thresh[1] > 0.
            && means[2] < self.partial_thresh[1] * means[1]
            && selected_end - selected_start > 1
        {
            dropped.push(selected_end);
            selected_end -= 1;
        }
        Ok(PartialFrameSelection {
            start: selected_start,
            end: selected_end,
            dropped,
        })
    }

    /// Apply source partial-frame selection to one owned MRC movie before the
    /// normal frame-set output path.  The retained endpoints stay inclusive.
    pub fn align_mrc_partial_frames_to(
        &self,
        input: impl AsRef<std::path::Path>,
        output: impl AsRef<std::path::Path>,
        start: usize,
        end: usize,
        bin_sum: usize,
        bin_align: usize,
        max_shift: usize,
        gain: Option<&[f32]>,
        dark: Option<&[u8]>,
        dose_per_frame: Option<f32>,
        critical_dose: f32,
    ) -> Result<(AliFrameResult, PartialFrameSelection), String> {
        let stack = OwnedImageStack::open_mrc(input.as_ref())?;
        let selection = self.analyze_for_partial_frames(&stack, start, end)?;
        let sets = SavedFrameSets {
            saved_frames: (selection.start..=selection.end)
                .map(|index| index as i32)
                .collect(),
            starts: vec![selection.start],
            counts: vec![selection.end + 1 - selection.start],
        };
        let mut results = self.align_mrc_frame_sets_to(
            input,
            output,
            &sets,
            bin_sum,
            bin_align,
            max_shift,
            gain,
            dark,
            dose_per_frame,
            critical_dose,
        )?;
        Ok((results.remove(0), selection))
    }

    /// Source unweighted-output branch for one owned input movie.
    pub fn align_mrc_file_with_unweighted_to(
        &self,
        input: impl AsRef<std::path::Path>,
        weighted_output: impl AsRef<std::path::Path>,
        unweighted_output: impl AsRef<std::path::Path>,
        bin_sum: usize,
        bin_align: usize,
        max_shift: usize,
    ) -> Result<AliFrameResult, String> {
        let stack = OwnedImageStack::open_mrc(input.as_ref())?;
        let result = self.align_mrc_file_to(
            input,
            weighted_output,
            bin_sum,
            bin_align,
            max_shift,
            None,
            None,
            None,
            0.,
            None,
            None,
        )?;
        let mut header = MrcHeader::default();
        if mrc_head_new(
            &mut header,
            stack.nx as i32,
            stack.ny as i32,
            1,
            MRC_MODE_FLOAT,
        ) != 0
        {
            return Err("cannot initialize unweighted MRC header".into());
        }
        let mut file = ImodFile::open(unweighted_output.as_ref(), "wb").ok_or_else(|| {
            format!(
                "cannot open unweighted MRC output {}",
                unweighted_output.as_ref().display()
            )
        })?;
        if mrc_head_write(&mut file, &mut header) != 0 {
            return Err("cannot write unweighted MRC header".into());
        }
        let bytes: Vec<u8> = result
            .unweighted_sum
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect();
        if mrc_write_slice(&bytes, &mut file, &mut header, 0, b'z') != 0 {
            return Err("cannot write unweighted MRC sum".into());
        }
        if let Some((&first, rest)) = result.unweighted_sum.split_first() {
            header.amin = rest.iter().fold(first, |low, value| low.min(*value));
            header.amax = rest.iter().fold(first, |high, value| high.max(*value));
            header.amean =
                result.unweighted_sum.iter().sum::<f32>() / result.unweighted_sum.len() as f32;
        }
        if mrc_head_write(&mut file, &mut header) != 0 {
            return Err("cannot finalize unweighted MRC header".into());
        }
        Ok(result)
    }

    /// Source sum-scaling option for an owned float MRC output.
    pub fn scale_sum_output(result: &mut AliFrameResult, scale: f32) {
        for value in &mut result.weighted_sum {
            *value *= scale;
        }
        for value in &mut result.unweighted_sum {
            *value *= scale;
        }
    }

    /// Write an owned aligned MRC sum and its source tilt-axis title.
    pub fn align_mrc_file_with_axis_to(
        &self,
        input: impl AsRef<std::path::Path>,
        output: impl AsRef<std::path::Path>,
        axis_angle: f32,
        bin_sum: usize,
        bin_align: usize,
        max_shift: usize,
    ) -> Result<AliFrameResult, String> {
        let result = self.align_mrc_file_to(
            input, &output, bin_sum, bin_align, max_shift, None, None, None, 0., None, None,
        )?;
        let mut file = ImodFile::open(output.as_ref(), "r+")
            .ok_or_else(|| format!("cannot reopen output MRC {}", output.as_ref().display()))?;
        let mut header = MrcHeader::default();
        if mrc_head_read(&mut file, &mut header) != 0 {
            return Err("cannot read output MRC header".into());
        }
        Self::add_axis_angle_title(&mut header, axis_angle);
        if mrc_head_write(&mut file, &mut header) != 0 {
            return Err("cannot write output MRC axis title".into());
        }
        Ok(result)
    }

    pub fn extract_file_tail(filename: &str) -> String {
        filename
            .rsplit(['/', '\\'])
            .next()
            .unwrap_or(filename)
            .into()
    }
    pub fn adjust_title_binning(title: &str, relative_binning: f32) -> Option<String> {
        let marker = "binning =";
        let at = title.find(marker)?;
        if at == 0 {
            return None;
        }
        let start = at
            + marker.len()
            + title[at + marker.len()..]
                .chars()
                .take_while(|character| *character == ' ')
                .count();
        let end = start + title[start..].find(' ').unwrap_or(title.len() - start);
        let stack: f32 = title[start..end].parse().ok()?;
        let adjusted = stack * relative_binning;
        if adjusted < 0.46
            || (adjusted > 0.54 && adjusted < 0.96)
            || (adjusted > 0.96 && (adjusted.round() - adjusted).abs() > 0.04)
        {
            return None;
        }
        let replacement = if adjusted < 0.54 {
            format!("{adjusted:.1}")
        } else {
            adjusted.round().to_string()
        };
        let mut result = format!("{}{}{}", &title[..start], replacement, &title[end..]);
        if !result.starts_with(' ') {
            result.insert_str(0, "    ");
        }
        Some(result)
    }
    pub fn expand_frame_doses_numbers(
        line: &str,
        maximum_frames: usize,
    ) -> Result<(Vec<f32>, f32), String> {
        let values = line
            .split_whitespace()
            .map(|word| {
                word.parse::<f32>()
                    .map_err(|_| format!("invalid frame dose value {word}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if values.len() % 2 != 0 {
            return Err("odd number of frame doses and counts".into());
        }
        let mut doses = Vec::new();
        let mut total = 0.;
        for pair in values.chunks_exact(2) {
            let count = pair[1].round();
            if (count - pair[1]).abs() > 1.0e-3 || count < 0. {
                return Err("frame dose count is not a nonnegative integer".into());
            }
            if doses.len() + count as usize > maximum_frames {
                return Err("frame numbers exceed maximum expected frames".into());
            }
            total += count * pair[0];
            doses.extend(std::iter::repeat_n(pair[0], count as usize));
        }
        Ok((doses, total))
    }

    /// Owned CPU segment of `AliFrame::main`: preprocess every fetched frame,
    /// derive the selected all-vs-all trajectory, then make weighted and
    /// unweighted sums through active `FrameAlign` APIs.
    pub fn align_image_stack(
        &self,
        stack: &OwnedImageStack,
        bin_sum: usize,
        bin_align: usize,
        max_shift: usize,
        gain: Option<&[f32]>,
        dark: Option<&[u8]>,
        dose_per_frame: Option<f32>,
        critical_dose: f32,
    ) -> Result<AliFrameResult, String> {
        if stack.frames.is_empty() {
            return Err("no input frames to align".into());
        }
        let mut aligner = FrameAlign::default();
        aligner.initialize(bin_sum, bin_align, 0.02, 0.1, stack.nx, stack.ny, max_shift)?;
        for frame in &stack.frames {
            aligner.next_frame(frame, stack.mode, dark, gain, self.trunc_limit)?;
        }
        if self.group_size > 1 {
            aligner.find_all_vs_all_alignment()?;
        }
        let unweighted_sum = aligner.get_unweighted_sum()?;
        let weights = dose_per_frame.map(|dose| aligner.setup_dose_weighting(dose, critical_dose));
        let weighted_sum = aligner.finish_align_and_sum(weights.as_deref())?;
        let (x_shifts, y_shifts) = aligner.get_all_frame_shifts()?;
        Ok(AliFrameResult {
            weighted_sum,
            unweighted_sum,
            x_shifts,
            y_shifts,
        })
    }

    /// File-input portion of `AliFrame::main` for ordinary MRC movie stacks.
    ///
    /// `OwnedImageStack` reads and owns every source section before alignment,
    /// so this active route never opens an `ImodImageFile` or transfers a raw
    /// caller-owned pixel pointer into `FrameAlign`.
    pub fn align_mrc_file(
        &self,
        path: impl AsRef<std::path::Path>,
        bin_sum: usize,
        bin_align: usize,
        max_shift: usize,
        gain: Option<&[f32]>,
        dark: Option<&[u8]>,
        dose_per_frame: Option<f32>,
        critical_dose: f32,
    ) -> Result<AliFrameResult, String> {
        let stack = OwnedImageStack::open_mrc(path)?;
        self.align_image_stack(
            &stack,
            bin_sum,
            bin_align,
            max_shift,
            gain,
            dark,
            dose_per_frame,
            critical_dose,
        )
    }

    /// End-to-end ordinary-MRC path: read a movie stack, align it, and write
    /// the weighted summed image as a one-section float MRC file.
    ///
    /// This is the owned equivalent of the source's output-header setup and
    /// `mrcWriteZFloat` block.  Multi-input output stacks, extended headers,
    /// and metadata-sidecar edits remain outside this already-supported
    /// single-stack route.
    pub fn align_mrc_file_to(
        &self,
        input: impl AsRef<std::path::Path>,
        output: impl AsRef<std::path::Path>,
        bin_sum: usize,
        bin_align: usize,
        max_shift: usize,
        gain: Option<&[f32]>,
        dark: Option<&[u8]>,
        dose_per_frame: Option<f32>,
        critical_dose: f32,
        transform_output: Option<&std::path::Path>,
        plottable_shift_output: Option<&std::path::Path>,
    ) -> Result<AliFrameResult, String> {
        let stack = OwnedImageStack::open_mrc(input)?;
        let result = self.align_image_stack(
            &stack,
            bin_sum,
            bin_align,
            max_shift,
            gain,
            dark,
            dose_per_frame,
            critical_dose,
        )?;
        let mut header = MrcHeader::default();
        if mrc_head_new(
            &mut header,
            i32::try_from(stack.nx).map_err(|_| "output width exceeds MRC range")?,
            i32::try_from(stack.ny).map_err(|_| "output height exceeds MRC range")?,
            1,
            MRC_MODE_FLOAT,
        ) != 0
        {
            return Err("cannot initialize MRC output header".into());
        }
        let mut output_file = ImodFile::open(output.as_ref(), "wb")
            .ok_or_else(|| format!("cannot open output MRC file {}", output.as_ref().display()))?;
        if mrc_head_write(&mut output_file, &mut header) != 0 {
            return Err("cannot write MRC output header".into());
        }
        let mut bytes = Vec::with_capacity(result.weighted_sum.len() * std::mem::size_of::<f32>());
        for value in &result.weighted_sum {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
        if mrc_write_slice(&bytes, &mut output_file, &mut header, 0, b'z') != 0 {
            return Err("cannot write aligned MRC sum".into());
        }
        if let Some((&first, rest)) = result.weighted_sum.split_first() {
            header.amin = rest
                .iter()
                .fold(first, |minimum, value| minimum.min(*value));
            header.amax = rest
                .iter()
                .fold(first, |maximum, value| maximum.max(*value));
            header.amean =
                result.weighted_sum.iter().sum::<f32>() / result.weighted_sum.len() as f32;
        }
        if mrc_head_write(&mut output_file, &mut header) != 0 {
            return Err("cannot finalize MRC output header".into());
        }
        if let Some(path) = transform_output {
            let file = std::fs::File::create(path).map_err(|error| {
                format!("cannot open transform file {}: {error}", path.display())
            })?;
            let mut writer = std::io::BufWriter::new(file);
            for (&x_shift, &y_shift) in result.x_shifts.iter().zip(&result.y_shifts) {
                writeln!(
                    writer,
                    " 1.00000    0.00000    0.00000   1.00000  {x_shift:8.3} {y_shift:8.3}"
                )
                .map_err(|error| {
                    format!("cannot write transform file {}: {error}", path.display())
                })?;
            }
        }
        if let Some(path) = plottable_shift_output {
            let file = std::fs::File::create(path).map_err(|error| {
                format!(
                    "cannot open plottable-shift file {}: {error}",
                    path.display()
                )
            })?;
            let mut writer = std::io::BufWriter::new(file);
            for (&x_shift, &y_shift) in result.x_shifts.iter().zip(&result.y_shifts) {
                writeln!(writer, "{:3}  {x_shift:.3}  {y_shift:.3}", 10).map_err(|error| {
                    format!(
                        "cannot write plottable-shift file {}: {error}",
                        path.display()
                    )
                })?;
            }
        }
        Ok(result)
    }

    /// Process the selected input movies in source order, writing one aligned
    /// sum per input as a Z section of one owned float MRC output stack.
    pub fn align_mrc_files_to(
        &self,
        inputs: &[std::path::PathBuf],
        output: impl AsRef<std::path::Path>,
        bin_sum: usize,
        bin_align: usize,
        max_shift: usize,
        gain: Option<&[f32]>,
        dark: Option<&[u8]>,
        dose_per_frame: Option<f32>,
        critical_dose: f32,
    ) -> Result<Vec<AliFrameResult>, String> {
        let first_path = inputs
            .first()
            .ok_or_else(|| "no input MRC files to align".to_owned())?;
        let first = OwnedImageStack::open_mrc(first_path)?;
        let (nx, ny) = (first.nx, first.ny);
        Self::check_input_file(first_path, &first, None, false)?;
        let mut header = MrcHeader::default();
        if mrc_head_new(
            &mut header,
            i32::try_from(nx).map_err(|_| "output width exceeds MRC range")?,
            i32::try_from(ny).map_err(|_| "output height exceeds MRC range")?,
            i32::try_from(inputs.len()).map_err(|_| "output section count exceeds MRC range")?,
            MRC_MODE_FLOAT,
        ) != 0
        {
            return Err("cannot initialize MRC output header".into());
        }
        let mut file = ImodFile::open(output.as_ref(), "wb")
            .ok_or_else(|| format!("cannot open output MRC file {}", output.as_ref().display()))?;
        if mrc_head_write(&mut file, &mut header) != 0 {
            return Err("cannot write MRC output header".into());
        }
        let mut results = Vec::with_capacity(inputs.len());
        let mut sum = 0f64;
        for (section, path) in inputs.iter().enumerate() {
            let stack = if section == 0 {
                first.clone()
            } else {
                OwnedImageStack::open_mrc(path)?
            };
            Self::check_input_file(path, &stack, Some((nx, ny)), false)?;
            let result = self.align_image_stack(
                &stack,
                bin_sum,
                bin_align,
                max_shift,
                gain,
                dark,
                dose_per_frame,
                critical_dose,
            )?;
            let bytes: Vec<u8> = result
                .weighted_sum
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect();
            if mrc_write_slice(&bytes, &mut file, &mut header, section as i32, b'z') != 0 {
                return Err(format!("cannot write aligned MRC output section {section}"));
            }
            let (&first_value, _) = result
                .weighted_sum
                .split_first()
                .ok_or_else(|| "alignment produced an empty summed section".to_owned())?;
            let (minimum, maximum) = result.weighted_sum[1..]
                .iter()
                .fold((first_value, first_value), |(minimum, maximum), &value| {
                    (minimum.min(value), maximum.max(value))
                });
            header.amin = if section == 0 {
                minimum
            } else {
                header.amin.min(minimum)
            };
            header.amax = if section == 0 {
                maximum
            } else {
                header.amax.max(maximum)
            };
            sum += result
                .weighted_sum
                .iter()
                .map(|&value| value as f64)
                .sum::<f64>();
            results.push(result);
        }
        header.amean = (sum / (nx * ny * inputs.len()) as f64) as f32;
        if mrc_head_write(&mut file, &mut header) != 0 {
            return Err("cannot finalize MRC output header".into());
        }
        Ok(results)
    }

    /// Source frame-set processing branch: align each selected group from one
    /// movie independently and write its summed image as one output Z section.
    pub fn align_mrc_frame_sets_to(
        &self,
        input: impl AsRef<std::path::Path>,
        output: impl AsRef<std::path::Path>,
        sets: &SavedFrameSets,
        bin_sum: usize,
        bin_align: usize,
        max_shift: usize,
        gain: Option<&[f32]>,
        dark: Option<&[u8]>,
        dose_per_frame: Option<f32>,
        critical_dose: f32,
    ) -> Result<Vec<AliFrameResult>, String> {
        if sets.starts.len() != sets.counts.len() || sets.starts.is_empty() {
            return Err("frame-set starts and counts are invalid".into());
        }
        let stack = OwnedImageStack::open_mrc(input.as_ref())?;
        Self::check_input_file(input.as_ref(), &stack, None, false)?;
        let mut header = MrcHeader::default();
        if mrc_head_new(
            &mut header,
            i32::try_from(stack.nx).map_err(|_| "output width exceeds MRC range")?,
            i32::try_from(stack.ny).map_err(|_| "output height exceeds MRC range")?,
            i32::try_from(sets.starts.len())
                .map_err(|_| "output section count exceeds MRC range")?,
            MRC_MODE_FLOAT,
        ) != 0
        {
            return Err("cannot initialize MRC output header".into());
        }
        let mut file = ImodFile::open(output.as_ref(), "wb")
            .ok_or_else(|| format!("cannot open output MRC file {}", output.as_ref().display()))?;
        if mrc_head_write(&mut file, &mut header) != 0 {
            return Err("cannot write MRC output header".into());
        }
        let mut results = Vec::with_capacity(sets.starts.len());
        let mut total = 0f64;
        for (section, (&start, &count)) in sets.starts.iter().zip(&sets.counts).enumerate() {
            let end = start
                .checked_add(count)
                .ok_or_else(|| "frame-set range overflows".to_owned())?;
            let frames = stack
                .frames
                .get(start..end)
                .ok_or_else(|| format!("frame-set {section} is outside input MRC section range"))?;
            let group =
                OwnedImageStack::from_raw_frames(stack.nx, stack.ny, stack.mode, frames.to_vec())?;
            let result = self.align_image_stack(
                &group,
                bin_sum,
                bin_align,
                max_shift,
                gain,
                dark,
                dose_per_frame,
                critical_dose,
            )?;
            let bytes: Vec<u8> = result
                .weighted_sum
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect();
            if mrc_write_slice(&bytes, &mut file, &mut header, section as i32, b'z') != 0 {
                return Err(format!("cannot write aligned frame-set section {section}"));
            }
            let (&first, _) = result
                .weighted_sum
                .split_first()
                .ok_or_else(|| "alignment produced an empty summed section".to_owned())?;
            let (minimum, maximum) = result.weighted_sum[1..]
                .iter()
                .fold((first, first), |(minimum, maximum), &value| {
                    (minimum.min(value), maximum.max(value))
                });
            header.amin = if section == 0 {
                minimum
            } else {
                header.amin.min(minimum)
            };
            header.amax = if section == 0 {
                maximum
            } else {
                header.amax.max(maximum)
            };
            total += result
                .weighted_sum
                .iter()
                .map(|&value| value as f64)
                .sum::<f64>();
            results.push(result);
        }
        header.amean = (total / (stack.nx * stack.ny * sets.starts.len()) as f64) as f32;
        if mrc_head_write(&mut file, &mut header) != 0 {
            return Err("cannot finalize MRC output header".into());
        }
        Ok(results)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AliFrameResult {
    pub weighted_sum: Vec<f32>,
    pub unweighted_sum: Vec<f32>,
    pub x_shifts: Vec<f32>,
    pub y_shifts: Vec<f32>,
}

/// Owned result of `analyzeExtraHeader`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExtendedHeaderAnalysis {
    pub source_status: i32,
    pub axis_angle: f32,
    pub pixel_size: f32,
    pub tilts: Vec<f32>,
    pub set_starts: Vec<usize>,
    pub set_sizes: Vec<usize>,
    pub min_set_size: usize,
    pub max_set_size: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SavedFrameSets {
    pub saved_frames: Vec<i32>,
    pub starts: Vec<usize>,
    pub counts: Vec<usize>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TiltAngleRecords {
    pub angles: Vec<f32>,
    pub relative_starts: Vec<i32>,
    pub relative_ends: Vec<i32>,
    pub relative_frame_starts_found: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartialFrameSelection {
    pub start: usize,
    pub end: usize,
    pub dropped: Vec<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mdoc_tilts_saved_frames_and_fei_axis_title_are_transferred() {
        let result = AliFrame::get_angles_and_titles_from_mdoc(
            &MdocTiltTitleInput {
                tilt_angles: vec![-30., 30.],
                frame_ts_start_end: vec![Some((2, 7)), None],
                pixel_spacing: Some(1.25),
                image_size: Some((4096, 4096)),
                titles: vec!["TiltAxisAngle = -100.00 deg".into()],
                first_rotation_angle: Some(10.),
                ..Default::default()
            },
            true,
            true,
            None,
            false,
        )
        .unwrap();
        assert_eq!(result.tilt_records.angles, vec![-30., 30.]);
        assert_eq!(result.tilt_records.relative_starts, vec![2, -1]);
        assert_eq!(result.axis_angle, Some(10.));
        assert_eq!(result.titles, vec!["  Tilt axis angle = 10.00 deg"]);
        assert!(result.are_fei_frames);
    }

    #[test]
    fn mdoc_doses_keep_section_order_and_reject_bad_metadata_sizes() {
        let lines = vec!["1.5 3".to_owned()];
        let result =
            AliFrame::get_doses_from_mdoc(2, 2, &[1.5, 2.], &[0., 1.5], &[4, 9], &lines).unwrap();
        assert_eq!(result.dose_from_mdoc, vec![1.5, 2.]);
        assert_eq!(result.prior_from_mdoc, vec![0., 1.5]);
        assert_eq!(result.iz_piece, vec![4, 9]);
        assert!(AliFrame::get_doses_from_mdoc(2, 1, &[1.], &[0.], &[0], &[]).is_err());
    }

    #[test]
    fn title_reference_discovery_keeps_existing_files_and_eer_basename() {
        let directory =
            std::env::temp_dir().join(format!("imod-title-refs-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("gainRef.mrc"), []).unwrap();
        std::fs::write(directory.join("camera-defect.txt"), []).unwrap();
        let references = AliFrame::check_titles_for_ref_names(
            directory.join("frames.mrc"),
            &[" gainRef.mrc ".into(), "camera-defect.txt".into()],
            None,
        )
        .unwrap();
        assert_eq!(references.gain_name, Some(directory.join("gainRef.mrc")));
        assert_eq!(
            references.defect_name,
            Some(directory.join("camera-defect.txt"))
        );
        assert_eq!(
            AliFrame::check_titles_for_ref_names("frames.eer", &[], Some("C:\\refs\\eer.tif"))
                .unwrap()
                .gain_name,
            Some(std::path::PathBuf::from("eer.tif"))
        );
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn owned_frame_reader_selects_one_section_and_checks_bounds() {
        let stack =
            OwnedImageStack::from_raw_frames(2, 1, 0, vec![vec![1, 2], vec![3, 4]]).unwrap();
        assert_eq!(AliFrame::read_one_frame(&stack, 1).unwrap(), vec![3, 4]);
        assert!(AliFrame::read_one_frame(&stack, 2).is_err());
    }

    #[test]
    fn mdoc_open_reports_the_source_missing_file_category() {
        let path = std::env::temp_dir().join(format!("imod-no-mdoc-{}", std::process::id()));
        assert!(
            AliFrame::open_mdoc_file(path)
                .unwrap_err()
                .contains("does not exist")
        );
    }

    #[test]
    fn unified_doses_expand_fixed_frames_and_accumulate_priors() {
        let result = AliFrame::unify_dose_information(&UnifiedDoseInput {
            number_of_files: 3,
            fixed_frame_doses: Some("1.5 2 3 1".into()),
            dose_accumulates: 1,
            maximum_frames: 8,
            ..Default::default()
        })
        .unwrap();
        assert_eq!(result.dose_file_type, 5);
        assert_eq!(result.total_doses, vec![6., 6., 6.]);
        assert_eq!(result.prior_doses, vec![0., 6., 12.]);
        assert_eq!(result.frame_dose_lines.len(), 3);
    }

    #[test]
    fn gain_dark_validation_preserves_source_modes_sizes_and_super_resolution() {
        let gain =
            OwnedImageStack::from_raw_frames(2, 2, MRC_MODE_FLOAT, vec![vec![0; 16]]).unwrap();
        let dark =
            OwnedImageStack::from_raw_frames(4, 4, MRC_MODE_USHORT, vec![vec![0; 32]]).unwrap();
        let result = AliFrame::get_gain_dark_defects(GainDarkDefectsInput {
            image_dimensions: (4, 4),
            gain: Some(gain),
            dark: Some(dark),
            frames_are_eer: true,
            ..Default::default()
        })
        .unwrap();
        assert_eq!(result.super_resolution_factor, 2);
        assert_eq!(result.gain_dimensions, Some((2, 2)));
        assert!(result.dark.is_some());
        let bad =
            OwnedImageStack::from_raw_frames(3, 2, MRC_MODE_FLOAT, vec![vec![0; 24]]).unwrap();
        assert!(
            AliFrame::get_gain_dark_defects(GainDarkDefectsInput {
                image_dimensions: (4, 4),
                gain: Some(bad),
                frames_are_eer: true,
                ..Default::default()
            })
            .is_err()
        );
    }

    #[test]
    fn gpu_planner_keeps_sum_alignment_and_drops_only_unaffordable_frc() {
        let result = AliFrame::assess_gpu_needs(GpuNeedsInput {
            gpu_requested: true,
            gpu_available_memory: 1_000.,
            sum_memory_need: 600.,
            alignment_memory_need: 200.,
            sum_pad_size: 300.,
            getting_frc: true,
            ..Default::default()
        });
        assert_eq!(result.flags & 1, 1);
        assert_eq!(result.flags & (1 << 2), 1 << 2);
        assert!(!result.getting_frc);
    }

    #[test]
    fn source_constructor_and_defect_protocol_are_owned() {
        assert_eq!(ali_frame().memory_limit, 12.);
        let path = std::env::temp_dir().join(format!("imod-defects-{}", std::process::id()));
        std::fs::write(&path, "# header\n\n1 2 3\n").unwrap();
        assert_eq!(
            AliFrame::defect_file_to_string(&path).unwrap(),
            "# header\n1 2 3\n"
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn dose_option_processing_preserves_voltage_and_text_file_rules() {
        let options =
            AliFrame::process_dose_weighting_options(None, 200, 2., true, 3, Some("1 4\n2 6\n"))
                .unwrap();
        assert_eq!(options.dose_scaling, 1.6);
        assert_eq!(options.total_doses, vec![3., 4.]);
        assert_eq!(options.prior_doses, vec![1., 2.]);
        assert_eq!(options.reweight_ones.as_ref().unwrap().len(), 9000);
    }

    #[test]
    fn gain_reference_rotation_uses_source_float_transform() {
        let (mut nx, mut ny) = (2, 3);
        let mut gain = vec![1., 2., 3., 4., 5., 6.];
        AliFrame::rotate_flip_gain_reference(&mut gain, &mut nx, &mut ny, 0).unwrap();
        assert_eq!((nx, ny), (2, 3));
        assert_eq!(gain, vec![1., 2., 3., 4., 5., 6.]);
    }

    #[test]
    fn extended_header_analysis_normalizes_axis_and_groups_tilts() {
        let header = MrcHeader {
            nz: 3,
            nint: 1,
            nreal: 12,
            ..Default::default()
        };
        let mut extended = vec![0.; 39];
        for (section, tilt) in [10., 10., 20.].into_iter().enumerate() {
            extended[section * 13 + 1] = tilt;
        }
        extended[11] = -190.;
        extended[12] = 2.5;
        let result = AliFrame::analyze_extra_header(&header, &extended, None, false).unwrap();
        assert_eq!(result.source_status, 3);
        assert_eq!(result.axis_angle, 170.);
        assert_eq!(result.pixel_size, 2.5);
        assert_eq!(result.tilts, vec![10., 20.]);
        assert_eq!(result.set_sizes, vec![2, 1]);
    }

    #[test]
    fn source_filename_title_and_dose_helpers() {
        assert_eq!(AliFrame::extract_file_tail("a\\b/c.mrc"), "c.mrc");
        assert_eq!(
            AliFrame::adjust_title_binning("title binning = 2 rest", 0.5).unwrap(),
            "    title binning = 1 rest"
        );
        assert_eq!(
            AliFrame::expand_frame_doses_numbers("1.5 2 3 1", 4).unwrap(),
            (vec![1.5, 1.5, 3.], 6.)
        );
    }

    #[test]
    fn source_sum_buffer_accumulates_each_real_input_mode() {
        let mut short = AliFrameSumBuffer::Short(vec![1, 2]);
        AliFrame::add_to_sum_buffer(AliFrameInputPixels::Byte(&[3, 4]), &mut short).unwrap();
        AliFrame::add_to_sum_buffer(AliFrameInputPixels::Short(&[-1, 2]), &mut short).unwrap();
        assert_eq!(short, AliFrameSumBuffer::Short(vec![3, 8]));
        let mut float = AliFrameSumBuffer::Float(vec![1., 2.]);
        AliFrame::add_to_sum_buffer(AliFrameInputPixels::Float(&[0.5, -1.]), &mut float).unwrap();
        assert_eq!(float, AliFrameSumBuffer::Float(vec![1.5, 1.]));
    }

    #[test]
    fn source_filename_selection_prefers_list_then_option_then_positional_inputs() {
        let list = std::env::temp_dir().join(format!(
            "imod-alignframes-list-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::write(&list, "\nfirst.mrc\n\nsecond.mrc\n").unwrap();
        assert_eq!(
            AliFrame::get_next_filename(1, &[], &[], Some(&list)).unwrap(),
            Some("second.mrc".into())
        );
        let _ = std::fs::remove_file(list);
        assert_eq!(
            AliFrame::get_next_filename(
                1,
                &["option.mrc".into()],
                &["positional.mrc".into()],
                None,
            )
            .unwrap(),
            Some("positional.mrc".into())
        );
    }

    #[test]
    fn source_header_open_reports_the_test_mode_context() {
        let missing =
            std::env::temp_dir().join(format!("imod-alignframes-missing-{}", std::process::id()));
        let error = match AliFrame::open_and_read_header(&missing, "input", true) {
            Ok(_) => panic!("opening a missing header unexpectedly succeeded"),
            Err(error) => error,
        };
        assert!(error.contains("cannot open input file"));
        assert!(error.contains("do not specify an output file"));
    }

    #[test]
    fn saved_frame_list_groups_source_gap_separated_sets() {
        let list = std::env::temp_dir().join(format!(
            "imod-alignframes-saved-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::write(&list, "0\n1\n2\n10\n11\n12\n20\n21\n22\n23\n").unwrap();
        let sets = AliFrame::read_analyze_saved_frame_list(&list, 1, 0, 0).unwrap();
        let _ = std::fs::remove_file(list);
        assert_eq!(sets.starts, vec![0, 3, 6]);
        assert_eq!(sets.counts, vec![3, 3, 4]);
    }

    #[test]
    fn frame_set_breaking_balances_the_source_selected_range() {
        let sets = AliFrame::break_frames_into_sets(12, 3, Some(1), Some(10)).unwrap();
        assert_eq!(sets.starts, vec![1, 4, 7]);
        assert_eq!(sets.counts, vec![3, 3, 4]);
    }

    #[test]
    fn tilt_angle_file_retains_optional_relative_frame_limits() {
        let path = std::env::temp_dir().join(format!(
            "imod-alignframes-tilts-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::write(&path, "-60\n0 2\n60 4 8\n").unwrap();
        let records = AliFrame::read_tilt_angle_file(&path).unwrap();
        let _ = std::fs::remove_file(path);
        assert_eq!(records.angles, vec![-60., 0., 60.]);
        assert_eq!(records.relative_starts, vec![-1, 2, 4]);
        assert_eq!(records.relative_ends, vec![-1, -1, 8]);
        assert!(records.relative_frame_starts_found);
    }

    #[test]
    fn axis_angle_title_uses_source_label_slot_and_fixed_width_padding() {
        let mut header = MrcHeader::default();
        header.nlabl = 8;
        AliFrame::add_axis_angle_title(&mut header, -12.25);
        assert_eq!(header.nlabl, 9);
        assert_eq!(
            std::str::from_utf8(&header.labels[8][..27]).unwrap(),
            "    Tilt axis angle = -12.2"
        );
        assert!(header.labels[8][27..].iter().all(|&byte| byte == b' '));
    }

    #[test]
    fn tilt_overlap_trimming_removes_angles_without_a_saved_frame_set() {
        let sets = SavedFrameSets {
            saved_frames: vec![100, 101, 102, 110, 111, 112],
            starts: vec![0, 3],
            counts: vec![3, 3],
        };
        let mut records = TiltAngleRecords {
            angles: vec![-10., 0., 10.],
            relative_starts: vec![0, 10, 30],
            relative_ends: vec![2, 12, 31],
            relative_frame_starts_found: true,
        };
        assert_eq!(
            AliFrame::handle_too_many_tilt_angles(&mut records, &sets, 2).unwrap(),
            1
        );
        assert_eq!(records.angles, vec![-10., 0.]);
    }

    #[test]
    fn partial_frame_sampling_drops_dim_endpoints_in_source_order() {
        let ali = AliFrame {
            partial_thresh: [0.5, 0.5],
            ..AliFrame::default()
        };
        let stack =
            OwnedImageStack::from_raw_frames(4, 4, 0, vec![vec![1; 16], vec![10; 16], vec![1; 16]])
                .unwrap();
        assert_eq!(
            ali.analyze_for_partial_frames(&stack, 0, 2).unwrap(),
            PartialFrameSelection {
                start: 1,
                // `analyzeForPartialFrames` retains two frames after dropping
                // the first endpoint: its native guard is `end - start > 1`.
                end: 2,
                dropped: vec![0]
            }
        );
    }

    #[test]
    fn real_mrc_partial_selection_preserves_the_middle_section() {
        let base = std::env::temp_dir().join(format!("imod-partial-{}", std::process::id()));
        let input = base.with_extension("in.mrc");
        let output = base.with_extension("out.mrc");
        let weighted = base.with_extension("weighted.mrc");
        let unweighted = base.with_extension("unweighted.mrc");
        let axis_output = base.with_extension("axis.mrc");
        {
            let mut file = ImodFile::open(&input, "wb").unwrap();
            let mut header = MrcHeader::default();
            assert_eq!(mrc_head_new(&mut header, 4, 4, 3, 0), 0);
            header.bytes_signed = 0;
            assert_eq!(mrc_head_write(&mut file, &mut header), 0);
            for (section, value) in [1_u8, 10, 1].into_iter().enumerate() {
                assert_eq!(
                    mrc_write_slice(
                        &vec![value; 16],
                        &mut file,
                        &mut header,
                        section as i32,
                        b'z'
                    ),
                    0
                );
            }
        }
        let ali = AliFrame {
            partial_thresh: [0.5, 0.5],
            ..AliFrame::default()
        };
        let (result, selection) = ali
            .align_mrc_partial_frames_to(&input, &output, 0, 2, 1, 1, 1, None, None, None, 0.)
            .unwrap();
        assert_eq!(
            selection,
            PartialFrameSelection {
                start: 1,
                end: 2,
                dropped: vec![0]
            }
        );
        // The source keeps sections 1 and 2, so the aligned sum is their mean.
        assert_eq!(result.weighted_sum, vec![5.5; 16]);
        assert_eq!(OwnedImageStack::open_mrc(&output).unwrap().frame_count(), 1);
        AliFrame::default()
            .align_mrc_file_with_unweighted_to(&input, &weighted, &unweighted, 1, 1, 1)
            .unwrap();
        let weighted_stack = OwnedImageStack::open_mrc(&weighted).unwrap();
        let unweighted_stack = OwnedImageStack::open_mrc(&unweighted).unwrap();
        assert_eq!(weighted_stack.frames, unweighted_stack.frames);
        let mut weighted_file = ImodFile::open(&weighted, "rb").unwrap();
        let mut unweighted_file = ImodFile::open(&unweighted, "rb").unwrap();
        let mut weighted_header = MrcHeader::default();
        let mut unweighted_header = MrcHeader::default();
        assert_eq!(mrc_head_read(&mut weighted_file, &mut weighted_header), 0);
        assert_eq!(
            mrc_head_read(&mut unweighted_file, &mut unweighted_header),
            0
        );
        assert_eq!(
            (
                weighted_header.amin,
                weighted_header.amax,
                weighted_header.amean
            ),
            (
                unweighted_header.amin,
                unweighted_header.amax,
                unweighted_header.amean
            )
        );
        AliFrame::default()
            .align_mrc_file_with_axis_to(&input, &axis_output, 37.5, 1, 1, 1)
            .unwrap();
        let mut axis_file = ImodFile::open(&axis_output, "rb").unwrap();
        let mut axis_header = MrcHeader::default();
        assert_eq!(mrc_head_read(&mut axis_file, &mut axis_header), 0);
        assert_eq!(axis_header.nlabl, 1);
        assert_eq!(
            std::str::from_utf8(&axis_header.labels[0][..26]).unwrap(),
            "    Tilt axis angle = 37.5"
        );
        let _ = std::fs::remove_file(input);
        let _ = std::fs::remove_file(output);
        let _ = std::fs::remove_file(weighted);
        let _ = std::fs::remove_file(unweighted);
        let _ = std::fs::remove_file(axis_output);
    }
    #[test]
    fn sum_scaling_applies_to_both_source_output_buffers() {
        let mut result = AliFrameResult {
            weighted_sum: vec![1., 2.],
            unweighted_sum: vec![3., 4.],
            x_shifts: vec![],
            y_shifts: vec![],
        };
        AliFrame::scale_sum_output(&mut result, 2.);
        assert_eq!(
            (result.weighted_sum, result.unweighted_sum),
            (vec![2., 4.], vec![6., 8.])
        );
    }
    #[test]
    fn owned_orchestration_uses_active_framealign() {
        let ali = AliFrame::default();
        let stack =
            OwnedImageStack::from_raw_frames(2, 2, 0, vec![vec![1, 2, 3, 4], vec![1, 2, 3, 4]])
                .unwrap();
        let result = ali
            .align_image_stack(&stack, 1, 1, 1, None, None, Some(1.), 2.)
            .unwrap();
        assert_eq!(result.unweighted_sum, vec![1., 2., 3., 4.]);
        assert_eq!(result.x_shifts, vec![0., 0.]);
    }

    #[test]
    fn mrc_entry_reports_a_read_error_without_a_legacy_image_handle() {
        assert!(
            AliFrame::default()
                .align_mrc_file("does-not-exist.mrc", 1, 1, 1, None, None, None, 0.)
                .is_err()
        );
    }

    #[test]
    fn owned_mrc_route_writes_a_real_sum_and_source_transform_layout() {
        let base = std::env::temp_dir().join(format!(
            "imod-alignframes-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let input = base.with_extension("input.mrc");
        let input_two = base.with_extension("input-two.mrc");
        let output = base.with_extension("output.mrc");
        let batch_output = base.with_extension("batch.mrc");
        let grouped_output = base.with_extension("grouped.mrc");
        let transforms = base.with_extension("xf");
        let shifts = base.with_extension("shifts");
        {
            let mut file = ImodFile::open(&input, "wb").unwrap();
            let mut header = MrcHeader::default();
            assert_eq!(mrc_head_new(&mut header, 2, 2, 2, 0), 0);
            header.bytes_signed = 0;
            assert_eq!(mrc_head_write(&mut file, &mut header), 0);
            assert_eq!(
                mrc_write_slice(&[1, 2, 3, 4], &mut file, &mut header, 0, b'z'),
                0
            );
            assert_eq!(
                mrc_write_slice(&[1, 2, 3, 4], &mut file, &mut header, 1, b'z'),
                0
            );
        }
        {
            let mut file = ImodFile::open(&input_two, "wb").unwrap();
            let mut header = MrcHeader::default();
            assert_eq!(mrc_head_new(&mut header, 2, 2, 2, 0), 0);
            header.bytes_signed = 0;
            assert_eq!(mrc_head_write(&mut file, &mut header), 0);
            assert_eq!(
                mrc_write_slice(&[5, 6, 7, 8], &mut file, &mut header, 0, b'z'),
                0
            );
            assert_eq!(
                mrc_write_slice(&[5, 6, 7, 8], &mut file, &mut header, 1, b'z'),
                0
            );
        }
        let result = AliFrame::default()
            .align_mrc_file_to(
                &input,
                &output,
                1,
                1,
                1,
                None,
                None,
                None,
                0.,
                Some(&transforms),
                Some(&shifts),
            )
            .unwrap();
        let written = OwnedImageStack::open_mrc(&output).unwrap();
        let pixels: Vec<f32> = written.frames[0]
            .chunks_exact(4)
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
            .collect();
        assert_eq!(result.weighted_sum, vec![1., 2., 3., 4.]);
        assert_eq!(pixels, result.weighted_sum);
        assert_eq!(
            std::fs::read_to_string(&transforms).unwrap(),
            " 1.00000    0.00000    0.00000   1.00000     0.000    0.000\n".repeat(2)
        );
        assert_eq!(
            std::fs::read_to_string(&shifts).unwrap(),
            " 10  0.000  0.000\n".repeat(2)
        );
        let batch = AliFrame::default()
            .align_mrc_files_to(
                &[input.clone(), input_two.clone()],
                &batch_output,
                1,
                1,
                1,
                None,
                None,
                None,
                0.,
            )
            .unwrap();
        let batch_stack = OwnedImageStack::open_mrc(&batch_output).unwrap();
        assert_eq!(batch.len(), 2);
        assert_eq!(batch_stack.frame_count(), 2);
        assert_eq!(batch_stack.mode, MRC_MODE_FLOAT);
        let section_values = |section: usize| {
            batch_stack.frames[section]
                .chunks_exact(4)
                .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        assert_eq!(section_values(0), vec![1., 2., 3., 4.]);
        assert_eq!(section_values(1), vec![5., 6., 7., 8.]);
        let mut output_file = ImodFile::open(&batch_output, "rb").unwrap();
        let mut output_header = MrcHeader::default();
        assert_eq!(mrc_head_read(&mut output_file, &mut output_header), 0);
        assert_eq!(
            (output_header.amin, output_header.amax, output_header.amean),
            (1., 8., 4.5)
        );
        let grouped = AliFrame::default()
            .align_mrc_frame_sets_to(
                &input,
                &grouped_output,
                &AliFrame::break_frames_into_sets(2, 2, None, None).unwrap(),
                1,
                1,
                1,
                None,
                None,
                None,
                0.,
            )
            .unwrap();
        assert_eq!(grouped.len(), 1);
        assert_eq!(
            OwnedImageStack::open_mrc(&grouped_output)
                .unwrap()
                .frame_count(),
            1
        );
        let _ = std::fs::remove_file(input);
        let _ = std::fs::remove_file(input_two);
        let _ = std::fs::remove_file(output);
        let _ = std::fs::remove_file(batch_output);
        let _ = std::fs::remove_file(grouped_output);
        let _ = std::fs::remove_file(transforms);
        let _ = std::fs::remove_file(shifts);
    }
}
