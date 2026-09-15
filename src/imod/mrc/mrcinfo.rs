//! Translation of `IMOD/mrc/mrcinfo.c`.

use std::fmt::Write as _;

use crate::imod::libcfshr::b3dutil::ImodFile;
use crate::imod::libiimod::mrcfiles::{
    MRC_MODE_BYTE, MRC_MODE_COMPLEX_FLOAT, MRC_MODE_COMPLEX_SHORT, MRC_MODE_FLOAT, MRC_MODE_RGB,
    MRC_MODE_SHORT, MRC_NLABELS, MrcHeader, mrc_head_read,
};

/// Render the source's header-information report for one already-read MRC
/// header.  Keeping report construction separate from file opening makes the
/// fixed MRC vocabulary testable without a stdout-capture boundary.
pub fn mrcinfo_report(filename: &str, header: &MrcHeader) -> String {
    let mode = match header.mode {
        MRC_MODE_BYTE => "mode = Byte\n",
        MRC_MODE_SHORT => "mode = Short\n",
        MRC_MODE_FLOAT => "mode = Float\n",
        MRC_MODE_COMPLEX_SHORT => "mode = Complex Short\n",
        MRC_MODE_COMPLEX_FLOAT => "mode = Complex Float\n",
        MRC_MODE_RGB => "mode = rgb byte\n",
        _ => "mode is unknown.\n",
    };
    let mut report = format!("MRCinfo: Info on image file {filename}\n{mode}");
    let _ = writeln!(
        report,
        "Image size    =  ( {}, {}, {})",
        header.nx, header.ny, header.nz
    );
    let _ = writeln!(report, "minimum value = {}", header.amin);
    let _ = writeln!(report, "maximum value = {}", header.amax);
    let _ = writeln!(report, "mean value    = {}", header.amean);
    let _ = writeln!(
        report,
        "Start reading image at ( {}, {}, {}).",
        header.nxstart, header.nystart, header.nzstart
    );
    let _ = writeln!(
        report,
        "Read length   =  ( {}, {}, {}).",
        header.mx, header.my, header.mz
    );
    let _ = writeln!(
        report,
        "Size of voxel is ( {} x {} x {} ) um.",
        header.xlen, header.ylen, header.zlen
    );
    let _ = writeln!(
        report,
        "Rotation      =  ( {}, {}, {}).",
        header.alpha, header.beta, header.gamma
    );
    let _ = writeln!(
        report,
        "Col, rows, sections  = axis ({} , {}, {})",
        header.mapc, header.mapr, header.maps
    );
    let _ = writeln!(
        report,
        "Angles ({}, {}, {}) --> ({}, {}, {})",
        header.tiltangles[0],
        header.tiltangles[1],
        header.tiltangles[2],
        header.tiltangles[3],
        header.tiltangles[4],
        header.tiltangles[5]
    );
    if header.ispg != 0 {
        let _ = writeln!(report, "ispg =\t\t{}", header.ispg);
    }
    if header.idtype != 0 {
        let _ = writeln!(report, "idtype =\t{}", header.idtype);
        let _ = writeln!(report, "nd1 =\t\t{}", header.nd1);
        let _ = writeln!(report, "nd2 =\t\t{}", header.nd2);
        let _ = writeln!(report, "vd1 =\t\t{}", header.vd1 as f32 / 100.0);
        let _ = writeln!(report, "vd2 =\t\t{}", header.vd2 as f32 / 100.0);
    }
    let _ = writeln!(
        report,
        "orgin = ( {}, {}, {})",
        header.xorg, header.yorg, header.zorg
    );
    if header.nlabl > MRC_NLABELS as i32 {
        report.push_str("There are to many labels.\n\n");
        return report;
    }
    let _ = writeln!(report, "Thare are {} labels.\n", header.nlabl);
    for label in header.labels.iter().take(header.nlabl.max(0) as usize) {
        let end = label
            .iter()
            .position(|byte| *byte == 0)
            .unwrap_or(label.len());
        report.push_str(&String::from_utf8_lossy(&label[..end]));
    }
    report
}

/// C `main` in `mrcinfo.c`.
pub fn mrcinfo(arguments: &[String]) -> i32 {
    let progname = arguments.first().map_or("mrcinfo", String::as_str);
    eprintln!("{progname} version 1.0 Copyright (C)1994 Boulder Laboratory for");
    eprintln!("3-Dimensional Fine Structure, University of Colorado.");
    let Some(filename) = arguments.get(1) else {
        eprintln!("Usage: {progname} <image file>");
        return 3;
    };
    let Some(mut input) = ImodFile::open(filename, "rb") else {
        eprintln!("Error opening {filename}.");
        return 3;
    };
    let mut header = MrcHeader::default();
    let error = mrc_head_read(&mut input, &mut header);
    if error < 0 {
        eprintln!("Can't Read Input File Header.");
        return 3;
    }
    if error > 0 {
        eprintln!("Warning: File is not in MRC format.\n");
    }
    print!("{}", mrcinfo_report(filename, &header));
    0
}

#[cfg(test)]
mod tests {
    use super::{mrcinfo, mrcinfo_report};
    use crate::imod::libiimod::mrcfiles::{MRC_MODE_FLOAT, MrcHeader, mrc_head_new};

    #[test]
    fn generated_mrc_header_has_the_source_report_vocabulary() {
        let mut header = MrcHeader::default();
        assert_eq!(mrc_head_new(&mut header, 12, 8, 3, MRC_MODE_FLOAT), 0);
        header.amin = -2.5;
        header.amax = 7.25;
        header.amean = 1.5;
        header.labels[0][..5].copy_from_slice(b"test ");
        header.nlabl = 1;
        let report = mrcinfo_report("generated.mrc", &header);
        assert!(report.starts_with("MRCinfo: Info on image file generated.mrc\nmode = Float\n"));
        assert!(report.contains("Image size    =  ( 12, 8, 3)\n"));
        assert!(report.contains("minimum value = -2.5\n"));
        assert!(report.contains("Thare are 1 labels.\n\n"));
        assert!(report.ends_with("test "));
    }

    #[test]
    fn missing_filename_has_the_source_usage_status() {
        assert_eq!(mrcinfo(&["mrcinfo".into()]), 3);
    }
}
