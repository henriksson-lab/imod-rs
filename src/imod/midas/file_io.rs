//! Translation of the ordinary transform-file and tilt-angle paths in
//! `IMOD/midas/file_io.cpp`.  MRC image loading remains on the translated
//! `libiimod` API; Qt dialogs are not used for file I/O here.

use super::midas::{MidasTransform, MidasView, XTYPE_MONT};
use super::transforms::{rotate_transform, stretch_transform, tramat_copy, tramat_idmat};
use crate::imod::libcfshr::reduce_by_binning::{SLICE_MODE_BYTE, reduce_by_binning};
use crate::imod::libiimod::mrcfiles::{LoadInfo, MrcHeader, mrc_head_read, mrc_init_li};
use crate::imod::libiimod::mrcsec::mrc_read_z_byte;
use std::fs;
use std::path::Path;

/// C `load_transforms` (`file_io.cpp:178`) linear XF/XG portion.
pub fn load_transforms(view: &mut MidasView, filename: &Path) -> Result<i32, String> {
    let text = fs::read_to_string(filename)
        .map_err(|error| format!("Couldn't open {}: {error}", filename.display()))?;
    let mut matrices = Vec::new();
    for line in text.lines() {
        let values: Vec<f32> = line
            .split_whitespace()
            .map(str::parse)
            .collect::<Result<_, _>>()
            .map_err(|_| format!("Bad transform line in {}", filename.display()))?;
        if values.is_empty() {
            continue;
        }
        if values.len() != 6 {
            return Err(format!(
                "Unsupported MIDAS transform record in {}",
                filename.display()
            ));
        }
        let mut matrix = [0.; 9];
        tramat_idmat(&mut matrix);
        matrix[0] = values[0];
        matrix[3] = values[1];
        matrix[1] = values[2];
        matrix[4] = values[3];
        matrix[6] = values[4] / view.binning.max(1) as f32;
        matrix[7] = values[5] / view.binning.max(1) as f32;
        matrices.push(MidasTransform {
            black: 0,
            white: 0,
            mat: matrix,
        });
    }
    if matrices.is_empty() {
        return Err(format!("No transforms found in {}", filename.display()));
    }
    view.zsize = view.zsize.max(matrices.len() as i32);
    view.tr = matrices;
    Ok(0)
}

/// C `write_transforms` (`file_io.cpp:448`) linear XF/XG portion.  Montage
/// edge and warp storage require their complete source closures.
pub fn write_transforms(view: &MidasView, filename: &Path) -> Result<i32, String> {
    if view.xtype == XTYPE_MONT {
        return Err("MIDAS montage-edge transform output needs the montage graph closure".into());
    }
    if view.cur_warp_file >= 0 {
        return Err("MIDAS warp-file output needs the libwarp storage closure".into());
    }
    let number = if view.num_chunks != 0 {
        view.num_chunks as usize
    } else {
        view.zsize.max(0) as usize
    };
    let mut text = String::new();
    for index in 0..number {
        let transform = view
            .tr
            .get(index)
            .ok_or_else(|| "Transform count is shorter than image count".to_owned())?;
        let mut matrix = transform.mat;
        if view.cos_stretch != 0 {
            stretch_transform(view, &mut matrix, index, 1);
        }
        if view.rot_mode != 0 {
            rotate_transform(&mut matrix, -view.global_rot);
        }
        text.push_str(&format!(
            "{:12.7}{:12.7}{:12.7}{:12.7}{:12.3}{:12.3}\n",
            matrix[0],
            matrix[3],
            matrix[1],
            matrix[4],
            matrix[6] * view.binning.max(1) as f32,
            matrix[7] * view.binning.max(1) as f32
        ));
    }
    fs::write(filename, text)
        .map_err(|error| format!("Couldn't open {}: {error}", filename.display()))?;
    Ok(0)
}

/// C `load_angles` (`file_io.cpp:544`).
pub fn load_angles(view: &mut MidasView) -> Result<(), String> {
    let filename = view
        .tiltname
        .as_deref()
        .ok_or_else(|| "No tilt angle file specified".to_owned())?;
    let text = fs::read_to_string(filename)
        .map_err(|error| format!("Error opening or reading tilt angle file {filename}: {error}"))?;
    let mut angles: Vec<f32> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.trim().parse())
        .collect::<Result<_, _>>()
        .map_err(|_| format!("Invalid tilt angle in {filename}"))?;
    if angles.is_empty() {
        return Err(format!("No tilt angles found in the file {filename}"));
    }
    let last = *angles.last().unwrap();
    angles.resize(view.zsize.max(0) as usize, last);
    view.tilt_angles = angles;
    Ok(())
}

/// C `load_image` (`file_io.cpp:28`).
pub fn load_image(view: &mut MidasView, filename: &Path) -> Result<i32, String> {
    let Some(mut fp) =
        crate::imod::libcfshr::b3dutil::ImodFile::open(&filename.to_string_lossy(), "rb")
    else {
        return Err(format!("Couldn't open {}", filename.display()));
    };
    let mut header = MrcHeader::default();
    header.fp = Some(fp.clone());
    if mrc_head_read(&mut fp, &mut header) != 0 {
        return Err(format!("Error reading header from {}", filename.display()));
    }
    let (mut smin, mut smax) = (header.amin, header.amax);
    if view.sminin != 0. || view.smaxin != 0. {
        smin = view.sminin;
        smax = view.smaxin;
    }
    let mut li = LoadInfo::default();
    mrc_init_li(Some(&mut li), None);
    li.xmin = 0;
    li.xmax = header.nx - 1;
    li.ymin = 0;
    li.ymax = header.ny - 1;
    if smin == smax {
        smax = smin + 1.;
    }
    li.slope = 255. / (smax - smin);
    li.offset = -smin * li.slope;
    view.li = Some(li);
    view.hin = Some(header);
    Ok(0)
}
/// C `load_refimage` (`file_io.cpp:66`).
pub fn load_refimage(view: &mut MidasView, filename: &Path) -> Result<i32, String> {
    let Some(mut fp) =
        crate::imod::libcfshr::b3dutil::ImodFile::open(&filename.to_string_lossy(), "rb")
    else {
        return Err(format!(
            "Error opening reference image {}",
            filename.display()
        ));
    };
    let mut header = MrcHeader::default();
    header.fp = Some(fp.clone());
    if mrc_head_read(&mut fp, &mut header) != 0 {
        return Err(format!(
            "Error reading header of reference image {}",
            filename.display()
        ));
    }
    view.refzsize = header.nz;
    if header.nx != view.xsize || header.ny != view.ysize {
        return Err(format!(
            "Error: size of reference image in {} does not match size of images being aligned.",
            filename.display()
        ));
    }
    if view.xsec < 0 {
        view.xsec = 0;
    }
    if view.xsec >= header.nz {
        view.xsec = header.nz - 1;
    }
    let mut li = view
        .li
        .take()
        .ok_or_else(|| "Primary image must be loaded before reference image".to_owned())?;
    let mut smax = header.amax;
    if header.amin == smax {
        smax = header.amin + 1.;
    }
    li.slope = 255. / (smax - header.amin);
    li.offset = -header.amin * li.slope;
    view.ref_mean = header.amean * li.slope + li.offset;
    let mut data = vec![0; (view.xsize * view.ysize).max(0) as usize];
    let sec = view.xsec;
    let result = midas_read_z_byte(view, &mut header, &mut li, &mut data, sec);
    view.li = Some(li);
    drop(fp);
    result.map(|_| {
        view.ref_data = data;
        view.ref_present = true;
        0
    })
}
/// C `midasReadZByte` (`file_io.cpp:110`).
pub fn midas_read_z_byte(
    view: &mut MidasView,
    header: &mut MrcHeader,
    li: &mut LoadInfo,
    data: &mut [u8],
    sec: i32,
) -> Result<i32, String> {
    if view.binning == 1 {
        if mrc_read_z_byte(header, li, data, sec) != 0 {
            return Err("Error reading MRC byte section".into());
        }
    } else {
        view.unbinned_buf
            .resize((header.nx * header.ny).max(0) as usize, 0);
        if mrc_read_z_byte(header, li, &mut view.unbinned_buf, sec) != 0 {
            return Err("Error reading MRC byte section".into());
        }
        let (mut nx, mut ny) = (0, 0);
        if {
            reduce_by_binning(
                &view.unbinned_buf,
                SLICE_MODE_BYTE,
                header.nx,
                header.ny,
                view.binning,
                data,
                1,
                &mut nx,
                &mut ny,
            )
        } != 0
        {
            return Err("Error reducing MIDAS byte section by binning".into());
        }
    }
    Ok(0)
}
/// C `save_view` boundary.
pub fn save_view(_view: &MidasView, filename: &Path) -> Result<i32, String> {
    Err(format!(
        "MIDAS contrast rendering closure is not yet translated for {}",
        filename.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::libiimod::mrcfiles::{
        MRC_MODE_BYTE, mrc_head_new, mrc_head_write, mrc_write_slice,
    };
    use crate::imod::midas::midas::new_view;
    #[test]
    fn xf_roundtrip_source_columns() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("imod-rs-midas-{}-xf", std::process::id()));
        let mut view = new_view();
        view.zsize = 1;
        view.binning = 2;
        view.tr = vec![MidasTransform {
            black: 0,
            white: 0,
            mat: [1., 2., 0., 3., 4., 0., 5., 6., 1.],
        }];
        write_transforms(&view, &path).unwrap();
        let mut restored = new_view();
        restored.binning = 2;
        load_transforms(&mut restored, &path).unwrap();
        assert_eq!(restored.tr[0].mat, view.tr[0].mat);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn image_header_and_load_info_own_a_real_mrc_file() {
        let path =
            std::env::temp_dir().join(format!("imod-rs-midas-image-{}.mrc", std::process::id()));
        let mut fp =
            crate::imod::libcfshr::b3dutil::ImodFile::open(&path.to_string_lossy(), "wb").unwrap();
        let mut header = MrcHeader::default();
        mrc_head_new(&mut header, 2, 2, 1, MRC_MODE_BYTE);
        header.amin = 0.;
        header.amax = 3.;
        header.amean = 1.5;
        assert_eq!(mrc_head_write(&mut fp, &mut header), 0);
        assert_eq!(
            mrc_write_slice(&[0_u8, 1, 2, 3], &mut fp, &mut header, 0, b'z'),
            0
        );
        drop(fp);
        let mut view = new_view();
        view.binning = 1;
        assert_eq!(load_image(&mut view, &path), Ok(0));
        assert_eq!(view.hin.as_ref().unwrap().nx, 2);
        assert_eq!(view.li.as_ref().unwrap().slope, 85.);
        view.xsize = 2;
        view.ysize = 2;
        view.xsec = 9;
        assert_eq!(load_refimage(&mut view, &path), Ok(0));
        assert_eq!(view.xsec, 0);
        assert_eq!(view.ref_data, vec![0, 85, 170, 255]);
        assert_eq!(view.ref_mean, 127.5);
        let mut bytes = [0; 4];
        let mut header = view.hin.take().unwrap();
        let mut li = view.li.take().unwrap();
        assert_eq!(
            midas_read_z_byte(&mut view, &mut header, &mut li, &mut bytes, 0),
            Ok(0)
        );
        view.hin = Some(header);
        view.li = Some(li);
        assert_eq!(bytes, [0, 85, 170, 255]);
        drop(view.hin.take());
        let _ = fs::remove_file(path);
    }
}
