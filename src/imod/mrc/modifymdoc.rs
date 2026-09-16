//! Translation of `IMOD/mrc/modifymdoc.cpp`.

use crate::imod::libcfshr::autodoc::*;
use crate::imod::libcfshr::extraheader::{get_metadata_items, get_metadata_weighting_doses};
use crate::imod::libcfshr::parse_params::{
    pip_done, pip_get_float, pip_get_in_out_file, pip_get_integer, pip_read_or_parse_options,
};

/// C `main` in `modifymdoc.cpp`.
pub fn modifymdoc(arguments: &[String]) -> i32 {
    let options: [&[u8]; 8] = [
        b"input:InputFile:FN:",
        b"output:OutputFile:FN:",
        b"order:OrderToProduce:I:",
        b"dose:ElectronDosePerImage:F:",
        b"binning:BinningToScaleTo:F:",
        b"pixel:PixelSpacingToSet:F:",
        b"param:ParameterFile:PF:",
        b"help:usage:B:",
    ];
    let argv = arguments
        .iter()
        .map(|arg| arg.as_bytes().to_vec())
        .collect::<Vec<_>>();
    let mut opt_args = 0;
    let mut non_opt_args = 0;
    pip_read_or_parse_options(
        argv.len() as i32,
        &argv,
        &options,
        8,
        argv.first().map_or(b"modifymdoc", Vec::as_slice),
        3,
        1,
        1,
        &mut opt_args,
        &mut non_opt_args,
        None,
    );
    let mut input = Vec::new();
    let mut output = Vec::new();
    if pip_get_in_out_file(b"InputFile", 0, &mut input) != 0
        || pip_get_in_out_file(b"OutputFile", 1, &mut output) != 0
    {
        pip_done();
        return 1;
    }
    let mut reorder = 0_i32;
    let mut dose = 0.;
    let mut binning = 0.;
    let mut pixel = 0.;
    let have_dose = pip_get_float(b"ElectronDosePerImage", &mut dose) == 0;
    let have_bin = pip_get_float(b"BinningToScaleTo", &mut binning) == 0;
    let have_pixel = pip_get_float(b"PixelSpacingToSet", &mut pixel) == 0;
    let _ = pip_get_integer(b"OrderToProduce", &mut reorder);
    pip_done();
    if (have_dose && dose <= 0.)
        || (have_pixel && pixel <= 0.)
        || (have_bin
            && !((binning - 0.5).abs() < 0.001
                || (binning > 0.51 && (binning.round() - binning).abs() < 0.001)))
    {
        return 1;
    }
    let (mut montage, mut sections, mut section_type) = (0, 0, 0);
    let adoc = adoc_open_image_metadata(&input, 0, &mut montage, &mut sections, &mut section_type);
    if adoc < 0 || montage != 0 || section_type != 1 || sections < 2 {
        adoc_done();
        return 1;
    }
    let mut order = (0..sections).collect::<Vec<_>>();
    let mut tilts = vec![0.; sections as usize];
    let mut unused = tilts.clone();
    let (mut values, mut found) = (0, 0);
    if get_metadata_items(
        adoc,
        section_type,
        sections,
        1,
        &mut tilts,
        &mut unused,
        &mut values,
        &mut found,
        &order,
    ) != 0
        || found < sections
    {
        adoc_done();
        return 1;
    }
    if have_dose {
        let mut priors = vec![0.; sections as usize];
        let mut section_doses = priors.clone();
        for section in 0..sections {
            if adoc_set_float(ADOC_ZVALUE_NAME, section, b"ExposureDose", dose) != 0 {
                adoc_done();
                return 1;
            }
            let mut prior = 0.;
            if adoc_get_float(ADOC_ZVALUE_NAME, section, b"PriorRecordDose", &mut prior) == 0
                && adoc_delete_key_value(ADOC_ZVALUE_NAME, section, b"PriorRecordDose") != 0
            {
                adoc_done();
                return 1;
            }
        }
        if get_metadata_weighting_doses(
            adoc,
            section_type,
            sections,
            &order,
            0,
            &mut priors,
            &mut section_doses,
        ) != 0
        {
            adoc_done();
            return 1;
        }
        for (section, prior) in priors.into_iter().enumerate() {
            if adoc_set_float(ADOC_ZVALUE_NAME, section as i32, b"PriorRecordDose", prior) != 0 {
                adoc_done();
                return 1;
            }
        }
    }
    if have_bin || have_pixel {
        let mut old_bin = 1.;
        let mut old_pixel = 0.;
        let new_bin = if have_bin { binning } else { 1. };
        if have_bin && adoc_get_float(ADOC_ZVALUE_NAME, 0, b"Binning", &mut old_bin) < 0 {
            adoc_done();
            return 1;
        }
        let scale = new_bin / old_bin;
        let mut new_pixel = if have_pixel { pixel } else { 0. };
        if !have_pixel {
            let ret = adoc_get_float(ADOC_GLOBAL_NAME, 0, b"PixelSpacing", &mut old_pixel);
            if ret < 0
                || (ret > 0
                    && adoc_get_float(ADOC_ZVALUE_NAME, 0, b"PixelSpacing", &mut old_pixel) < 0)
            {
                adoc_done();
                return 1;
            }
            if old_pixel != 0. {
                new_pixel = old_pixel * scale;
            }
        }
        if have_bin {
            let (mut nx, mut ny) = (0, 0);
            let ret = adoc_get_two_integers(ADOC_GLOBAL_NAME, 0, b"ImageSize", &mut nx, &mut ny);
            if ret < 0 {
                adoc_done();
                return 1;
            }
            if ret == 0
                && adoc_set_two_integers(
                    ADOC_GLOBAL_NAME,
                    0,
                    b"ImageSize",
                    (nx as f32 / scale).round() as i32,
                    (ny as f32 / scale).round() as i32,
                ) != 0
            {
                adoc_done();
                return 1;
            }
        }
        if (old_pixel != 0. || new_pixel != 0.)
            && adoc_set_float(ADOC_GLOBAL_NAME, 0, b"PixelSpacing", new_pixel) != 0
        {
            adoc_done();
            return 1;
        }
        for section in 0..sections {
            if (have_bin && adoc_set_float(ADOC_ZVALUE_NAME, section, b"Binning", new_bin) != 0)
                || ((old_pixel != 0. || new_pixel != 0.)
                    && adoc_set_float(ADOC_ZVALUE_NAME, section, b"PixelSpacing", new_pixel) != 0)
            {
                adoc_done();
                return 1;
            }
        }
    }
    if reorder != 0 {
        order.sort_by(|a, b| tilts[*a as usize].total_cmp(&tilts[*b as usize]));
        if reorder < 0 {
            order.reverse();
        }
        for (name, section) in order.iter().enumerate() {
            if adoc_change_section_name(ADOC_ZVALUE_NAME, *section, name.to_string().as_bytes())
                != 0
            {
                adoc_done();
                return 1;
            }
        }
        if adoc_order_write_by_value(Some(ADOC_ZVALUE_NAME)) != 0 {
            adoc_done();
            return 1;
        }
    }
    let status = adoc_write(&output);
    adoc_done();
    if status < 0 {
        return 1;
    }
    println!("Wrote new mdoc file {}", String::from_utf8_lossy(&output));
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    static SEQUENCE: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn parameter_file_reorders_real_mdoc_sections_and_scales_metadata() {
        let number = SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "imod-rs-modifymdoc-{}-{number}",
            std::process::id()
        ));
        let input = root.with_extension("in.mdoc");
        let output = root.with_extension("out.mdoc");
        let params = root.with_extension("com");
        std::fs::write(&input, "ImageFile = source.mrc\nPixelSpacing = 2.0\nImageSize = 100 80\n\n[ZValue = 0]\nTiltAngle = 10\nBinning = 1\n\n[ZValue = 1]\nTiltAngle = -10\nBinning = 1\n").unwrap();
        std::fs::write(
            &params,
            format!(
                "InputFile = {}\nOutputFile = {}\nOrderToProduce = 1\nBinningToScaleTo = 2\n",
                input.display(),
                output.display()
            ),
        )
        .unwrap();
        assert_eq!(
            modifymdoc(&[
                "modifymdoc".into(),
                "-param".into(),
                params.display().to_string()
            ]),
            0
        );
        let result = std::fs::read_to_string(&output).unwrap();
        assert!(result.contains("ImageSize = 50 40"));
        assert!(result.contains("PixelSpacing = 4"));
        assert!(result.find("[ZValue = 0]").unwrap() < result.find("TiltAngle = -10").unwrap());
        let _ = std::fs::remove_file(input);
        let _ = std::fs::remove_file(output);
        let _ = std::fs::remove_file(params);
    }
}
