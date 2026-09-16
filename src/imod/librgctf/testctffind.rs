//! Executable support from `IMOD/librgctf/testctffind.cpp`.
//!
//! The source's final fitting call is `ctffind(...)`, implemented in the
//! separate, not-yet-translated `ctffind.cpp` unit.  Everything here is the
//! executable logic that does not depend on that missing algorithm or its
//! callback writer: source defaults, box sizing, section validation, and the
//! precise result reports.

#[derive(Clone, Debug, PartialEq)]
pub struct TestCtffindParameters {
    pub acceleration_voltage: f32,
    pub spherical_aberration: f32,
    pub box_size: i32,
    pub minimum_resolution: f32,
    pub maximum_resolution: f32,
    pub minimum_defocus: f32,
    pub maximum_defocus: f32,
    pub defocus_search_step: f32,
    pub astigmatism_tolerance: f32,
    pub additional_phase_shift_search_step: f32,
    pub known_astigmatism: f32,
    pub known_astigmatism_angle: f32,
    pub minimum_additional_phase_shift: f32,
    pub maximum_additional_phase_shift: f32,
    pub noisy_input_image: bool,
    pub slower_search: bool,
    pub astigmatism_is_known: bool,
    pub find_additional_phase_shift: bool,
    pub compute_extra_stats: bool,
    pub pixel_size_of_input_image: f32,
}

impl Default for TestCtffindParameters {
    fn default() -> Self {
        Self {
            acceleration_voltage: 200.,
            spherical_aberration: 2.,
            box_size: 256,
            minimum_resolution: 50.,
            maximum_resolution: 10.,
            minimum_defocus: 5_000.,
            maximum_defocus: 80_000.,
            defocus_search_step: 500.,
            astigmatism_tolerance: -100.,
            additional_phase_shift_search_step: 0.1,
            known_astigmatism: 0.,
            known_astigmatism_angle: 0.,
            minimum_additional_phase_shift: 0.,
            maximum_additional_phase_shift: 0.,
            noisy_input_image: false,
            slower_search: false,
            astigmatism_is_known: false,
            find_additional_phase_shift: false,
            compute_extra_stats: false,
            pixel_size_of_input_image: 0.,
        }
    }
}

pub fn even_box_size(box_size: i32) -> i32 {
    2 * ((box_size + 1) / 2)
}

/// Source `useBox` calculation.  `nice_frame` is supplied by the existing
/// numerical module before this point, so this preserves the later source
/// resampling branch exactly.
pub fn resampled_box_size(box_size: i32, resolution: f32, pixel_size: f32) -> i32 {
    if resolution > pixel_size * 2. {
        2 * ((1. + 0.5 * box_size as f32 * resolution / pixel_size).round() as i32 / 2)
    } else {
        box_size
    }
}

pub fn validate_section_range(start: i32, end: i32, depth: i32) -> Result<(), String> {
    if start < 0 || end >= depth || start > end {
        Err("Section range is out of range or out of order".into())
    } else {
        Ok(())
    }
}

pub fn format_ctffind_results(
    results: [f32; 7],
    multiple_sections: bool,
    find_phase: bool,
    extra_stats: bool,
) -> String {
    if multiple_sections {
        return format!(
            "{:.0}  {:.1}  {:.2}  {:.4}  {:.4}  {:.1}  {:.1}\n",
            results[0], results[1], results[2], results[3], results[4], results[5], results[6]
        );
    }
    let mut report = format!(
        "Defocus 1 {:.1}  2 {:.1}  (astig. {:.1}) angle {:.2}",
        results[0],
        results[1],
        results[0] - results[1],
        results[2]
    );
    if find_phase {
        report.push_str(&format!(
            "    Phase shift {:.4} rad ({:.2} deg)",
            results[3],
            results[3] / (std::f32::consts::PI / 180.)
        ));
    }
    report.push_str(&format!("\nScore {:.4}\n", results[4]));
    if extra_stats {
        report.push_str(&format!("Thon rings well fit to {:.1}\n", results[5]));
        if results[6] != 0. {
            report.push_str(&format!("CTF aliasing detected at {:.1}\n", results[6]));
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_parameter_and_report_workflow_is_deterministic() {
        assert_eq!(even_box_size(255), 256);
        assert_eq!(resampled_box_size(256, 2.8, 1.), 358);
        assert!(validate_section_range(0, 1, 2).is_ok());
        assert!(validate_section_range(1, 0, 2).is_err());
        assert_eq!(
            format_ctffind_results([10_000., 9_000., 45., 0.5, 0.8, 4., 0.], false, true, true),
            "Defocus 1 10000.0  2 9000.0  (astig. 1000.0) angle 45.00    Phase shift 0.5000 rad (28.65 deg)\nScore 0.8000\nThon rings well fit to 4.0\n"
        );
    }
}
