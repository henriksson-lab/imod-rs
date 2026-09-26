//! `IMOD/imodutil/imodtrans.c`: translate, scale and rotate imod model files.

use std::io::Write;

use crate::imod::clip::clip::{ScanArg, atoi, sscanf};
use crate::imod::libcfshr::b3dutil::{
    ImodFile, fgetline, imod_backup_file, imod_copyright, imod_prog_name, imod_version,
};
use crate::imod::libcfshr::parse_params::{exit_error, setExitPrefix};
use crate::imod::libiimod::iimage::{ii_fclose, ii_fopen};
use crate::imod::libiimod::mrcfiles::{MrcHeader, mrc_head_read};
use crate::imod::libimod::imat::{
    Axis3, Imat, imod_mat_new, imod_mat_rot, imod_mat_scale, imod_mat_trans,
};
use crate::imod::libimod::imodel::{
    IMODF_FLIPYZ, IMODF_OTRANS_ORIGIN, IMODF_TILTOK, Imod, Ipoint, Iref_image, imod_default,
    imod_flip_yz, imod_rot90x, imod_set_ref_image, imod_trans_from_ref_image, imod_trans_model3d,
};
use crate::imod::libimod::imodel_files::{imod_read_file, imod_write};

/// Original: `usage` (`imodtrans.c:27`, static).
fn usage(progname: &str) -> ! {
    imod_version(Some(progname));
    imod_copyright();
    let mut out = ImodFile::Stdout;
    let _ = out
        .write_all(format!("Usage: {progname} [options] <input file> <output file>\n").as_bytes());
    let _ = out.write_all(b"Options:\n");
    let _ = out.write_all(b"\t-tx #\tTranslate in X by #\n");
    let _ = out.write_all(b"\t-ty #\tTranslate in Y by #\n");
    let _ = out.write_all(b"\t-tz #\tTranslate in Z by #\n");
    let _ = out.write_all(b"\t-sx #\tScale in X by #\n");
    let _ = out.write_all(b"\t-sy #\tScale in Y by #\n");
    let _ = out.write_all(b"\t-sz #\tScale in Z by #\n");
    let _ = out.write_all(b"\t-rx #\tRotate around X axis by #\n");
    let _ = out.write_all(b"\t-ry #\tRotate around Y axis by #\n");
    let _ = out.write_all(b"\t-rz #\tRotate around Z axis by #\n");
    let _ = out.write_all(b"\t-2 file\tApply 2D transformations from file, one per section\n");
    let _ = out.write_all(b"\t-l #\tLine number of single 2D transformation to apply (from 0)\n");
    let _ = out.write_all(b"\t-3 file\tApply 3D transformation from file\n");
    let _ = out.write_all(b"\t-S #\tScale dx/dy[/dz] values in 2D/3D transformations by #\n");
    let _ = out.write_all(b"\t-n #,#,#\tSet X,Y,Z size of transformed volume\n");
    let _ = out.write_all(b"\t-z\tTransform Z-scaled coordinates using model's Z-scale\n");
    let _ =
        out.write_all(b"\t-f\tTransform flipped instead of native coordinates if Y-Z flipped\n");
    let _ = out.write_all(b"\t-i file\tTransform to match given image file coordinate system\n");
    let _ = out.write_all(
        b"\t-I file\tFirst set image coordinate information from the given image file\n",
    );
    let _ = out.write_all(b"\t-Y\tFlip model in Y and Z (without toggling flipped flag)\n");
    let _ = out.write_all(b"\t-T\tToggle flag that model is flipped in Y and Z\n");
    let _ = out.write_all(b"\t-R #\tRotate model around X (1 forward, 0 backward\n");

    crate::imod::libcfshr::b3dutil::exit(3)
}

/// Original: `main` (`imodtrans.c:65`).
///
/// Deviation note: where the source reads `argv[++i]` past the last argument
/// it gets the terminating NULL and, for every option but `-2`/`-3`/`-l`/`-R`,
/// dereferences it (`sscanf`/`iiFOpen` on NULL crash natively).  Here a
/// missing value reads as the empty string, which assigns nothing, and the
/// run then ends in the source's own "two non-option arguments" usage exit.
pub fn imodtrans() {
    let argv: Vec<String> = crate::imod::libcfshr::b3dutil::program_args();
    let argc = argv.len();
    let mut model = Imod::default();
    let mut filename: Option<String> = None;
    let mut i: usize;
    let mut mode = 0;
    let mut use_zscale = 0;
    let mut doflip = 1;
    let mut transopt = 0;
    let mut rot_scaleopt = 0;
    let mut to_image = 0;
    let mut from_image = 0;
    let mut one_line = -1;
    let mut toggle_flip = 0;
    let mut flip_model = 0;
    let mut rot_model = 0;
    let mut rot_like3dmod = 0;
    let mut zscale: f32 = 1.;
    let mut trans_scale: f32 = 1.;
    let mut rx: f32 = 0.;
    let mut ry: f32 = 0.;
    let mut rz: f32 = 0.;
    let mut transx: f32 = 0.;
    let mut transy: f32 = 0.;
    let mut transz: f32 = 0.;
    let mut multi_trans = 0;
    let mut new_nx = 0;
    let mut new_ny = 0;
    let mut new_nz = 0;
    let progname = imod_prog_name(argv.first().map(String::as_str).unwrap_or(""));
    let mut new_cen = Ipoint::default();
    let mut tmp_pt = Ipoint::default();
    let mut mat = imod_mat_new(3).unwrap();
    let mut norm_mat = imod_mat_new(3).unwrap();
    let mut use_ref = Iref_image::default();
    let mut hdata = MrcHeader::default();
    let mut hdata_first = MrcHeader::default();
    let unit_pt = Ipoint {
        x: 1.,
        y: 1.,
        z: 1.,
    };
    let prefix = format!("ERROR: {progname} - ");
    setExitPrefix(prefix.as_bytes());

    if argc < 3 {
        usage(&progname);
    }

    i = 1;
    while i < argc {
        let arg = argv[i].as_bytes();
        if arg.first() == Some(&b'-') {
            match arg.get(1).copied().unwrap_or(0) {
                b'2' => {
                    i += 1;
                    filename = argv.get(i).cloned();
                    mode += 2;
                }

                b'3' => {
                    i += 1;
                    filename = argv.get(i).cloned();
                    mode += 3;
                }

                b'S' => {
                    i += 1;
                    sscanf(
                        argv.get(i).map(String::as_str).unwrap_or(""),
                        "%f",
                        &mut [ScanArg::Flt(&mut trans_scale)],
                    );
                }

                b'i' | b'I' => {
                    let which = arg[1];
                    if which == b'i' {
                        to_image = 1;
                    } else {
                        from_image = 1;
                    }
                    i += 1;
                    let name = argv.get(i).map(String::as_str).unwrap_or("");
                    let Some(mut fin) = ii_fopen(name.as_bytes(), "rb") else {
                        exit_error(format!("Couldn't open {name}").as_bytes())
                    };

                    let target = if which == b'i' {
                        &mut hdata
                    } else {
                        &mut hdata_first
                    };
                    if mrc_head_read(&mut fin, target) != 0 {
                        exit_error(format!("Reading header from {name}").as_bytes());
                    }
                    ii_fclose(&mut fin);
                }

                b'z' => use_zscale = 1,

                b'f' => doflip = 0,

                b'Y' => flip_model = 1,

                b'R' => {
                    rot_model = 1;
                    i += 1;
                    rot_like3dmod = atoi(argv.get(i).map(String::as_str).unwrap_or(""));
                }

                b'T' => toggle_flip = 1,

                b'n' => {
                    i += 1;
                    sscanf(
                        argv.get(i).map(String::as_str).unwrap_or(""),
                        "%d%*c%d%*c%d",
                        &mut [
                            ScanArg::Int(&mut new_nx),
                            ScanArg::Int(&mut new_ny),
                            ScanArg::Int(&mut new_nz),
                        ],
                    );
                }

                b'l' => {
                    i += 1;
                    one_line = atoi(argv.get(i).map(String::as_str).unwrap_or(""));
                }

                b't' => {
                    /* Translations */
                    tmp_pt.x = 0.;
                    tmp_pt.y = 0.;
                    tmp_pt.z = 0.;
                    transopt = 1;
                    let axis = arg.get(2).copied().unwrap_or(0);
                    let (component, embedded) = match axis {
                        b'x' => (&mut tmp_pt.x, "-tx%f"),
                        b'y' => (&mut tmp_pt.y, "-ty%f"),
                        b'z' => (&mut tmp_pt.z, "-tz%f"),
                        _ => exit_error(format!("Invalid option {}", argv[i]).as_bytes()),
                    };
                    if arg.get(3).copied().unwrap_or(0) != 0x00 {
                        sscanf(&argv[i], embedded, &mut [ScanArg::Flt(component)]);
                    } else {
                        i += 1;
                        sscanf(
                            argv.get(i).map(String::as_str).unwrap_or(""),
                            "%f",
                            &mut [ScanArg::Flt(component)],
                        );
                    }
                    match axis {
                        b'x' => {
                            if transx != 0. {
                                multi_trans = 1;
                            }
                            transx = tmp_pt.x;
                        }
                        b'y' => {
                            if transy != 0. {
                                multi_trans = 1;
                            }
                            transy = tmp_pt.y;
                        }
                        _ => {
                            if transz != 0. {
                                multi_trans = 1;
                            }
                            transz = tmp_pt.z;
                        }
                    }
                    imod_mat_trans(&mut mat, &tmp_pt);
                }

                b's' => {
                    /* Scaling */
                    tmp_pt.x = 1.;
                    tmp_pt.y = 1.;
                    tmp_pt.z = 1.;
                    transopt = 1;
                    rot_scaleopt = 1;
                    let (component, embedded) = match arg.get(2).copied().unwrap_or(0) {
                        b'x' => (&mut tmp_pt.x, "-sx%f"),
                        b'y' => (&mut tmp_pt.y, "-sy%f"),
                        b'z' => (&mut tmp_pt.z, "-sz%f"),
                        _ => exit_error(format!("Invalid option {}", argv[i]).as_bytes()),
                    };
                    if arg.get(3).copied().unwrap_or(0) != 0x00 {
                        sscanf(&argv[i], embedded, &mut [ScanArg::Flt(component)]);
                    } else {
                        i += 1;
                        sscanf(
                            argv.get(i).map(String::as_str).unwrap_or(""),
                            "%f",
                            &mut [ScanArg::Flt(component)],
                        );
                    }
                    let _ = imod_mat_scale(&mut mat, &tmp_pt);
                    tmp_pt.x = (1. / tmp_pt.x as f64) as f32;
                    tmp_pt.y = (1. / tmp_pt.y as f64) as f32;
                    tmp_pt.z = (1. / tmp_pt.z as f64) as f32;
                    let _ = imod_mat_scale(&mut norm_mat, &tmp_pt);
                }

                b'r' => {
                    /* Rotations */
                    transopt = 1;
                    rot_scaleopt = 1;
                    let (angle, embedded, rot_axis) = match arg.get(2).copied().unwrap_or(0) {
                        b'x' => (&mut rx, "-rx%f", Axis3::X),
                        b'y' => (&mut ry, "-ry%f", Axis3::Y),
                        b'z' => (&mut rz, "-rz%f", Axis3::Z),
                        _ => exit_error(format!("Invalid option {}", argv[i]).as_bytes()),
                    };
                    if arg.get(3).copied().unwrap_or(0) != 0x00 {
                        sscanf(&argv[i], embedded, &mut [ScanArg::Flt(angle)]);
                    } else {
                        i += 1;
                        sscanf(
                            argv.get(i).map(String::as_str).unwrap_or(""),
                            "%f",
                            &mut [ScanArg::Flt(angle)],
                        );
                    }
                    let _ = imod_mat_rot(&mut mat, *angle as f64, rot_axis);
                    let _ = imod_mat_rot(&mut norm_mat, *angle as f64, rot_axis);
                }

                _ => exit_error(format!("Invalid option {}", argv[i]).as_bytes()),
            }
        } else {
            break;
        }
        i += 1;
    }

    if mode != 0 && mode / 2 != 1 {
        exit_error(b"You cannot enter both -2 and -3");
    }
    if rot_model != 0 && (flip_model != 0 || toggle_flip != 0) {
        // BUGS.md: the source names a nonexistent `-F`; the flag is `-Y`.
        exit_error(b"You cannot enter -R with either -Y or -T");
    }

    if mode != 0 && rot_scaleopt != 0 {
        exit_error(b"You cannot enter -r or -s options with -2 and -3");
    }
    if transopt != 0 && mode == 2 && one_line < 0 && transz != 0. {
        exit_error(b"You cannot enter -tz option with -2 unless transforming with one line");
    }
    if mode != 0 && multi_trans != 0 {
        exit_error(b"You cannot enter -t options more than once with -2 and -3");
    }

    if i + 2 != argc {
        let _ = ImodFile::Stdout.write_all(
            format!("ERROR: {progname} - Command line should end with two non-option arguments\n")
                .as_bytes(),
        );
        usage(&progname);
    }

    let Some(mut fin) = ImodFile::open(&argv[i], "rb") else {
        exit_error(format!("Opening input file {}", argv[i]).as_bytes())
    };

    imod_default(&mut model);
    if imod_read_file(&mut model, &mut fin).is_err() {
        exit_error(format!("Reading imod model ({})", argv[i]).as_bytes());
    }
    drop(fin);

    if imod_backup_file(&argv[i + 1]) != 0 {
        exit_error(b"Couldn't create backup file");
    }

    let Some(mut fout) = ImodFile::open(&argv[i + 1], "wb") else {
        exit_error(format!("Opening output file {}", argv[i + 1]).as_bytes())
    };

    /* Change image reference information */
    if from_image != 0 && imod_set_ref_image(&mut model, &hdata_first) != 0 {
        exit_error(b"Allocating a IrefImage structure");
    }

    /* Do flipping operations first */
    if flip_model != 0 {
        imod_flip_yz(&mut model);
    }
    if toggle_flip != 0 {
        if model.flags & IMODF_FLIPYZ != 0 {
            model.flags &= !IMODF_FLIPYZ;
        } else {
            model.flags |= IMODF_FLIPYZ;
        }
    }

    /* Do rotation */
    if rot_model != 0 {
        /* First unflip the model if necessary, then rotate */
        if model.flags & IMODF_FLIPYZ != 0 {
            imod_flip_yz(&mut model);
            model.flags &= !IMODF_FLIPYZ;
        }
        imod_rot90x(&mut model, if rot_like3dmod != 0 { 0 } else { 1 });

        /* Adjust the origin and rotation.  zmax and ymax are already swapped so old ymax
        is now zmax, old zmax is ymax */
        let zmax = model.zmax;
        let ymax = model.ymax;
        if let Some(mod_refp) = model.ref_image.as_mut() {
            if rot_like3dmod != 0 {
                let temp = mod_refp.ctrans.z;
                mod_refp.ctrans.z = zmax as f32 * mod_refp.cscale.y - mod_refp.ctrans.y;
                mod_refp.ctrans.y = temp;
                mod_refp.crot.x = (mod_refp.crot.x as f64 - 90.) as f32;
            } else {
                let temp = mod_refp.ctrans.y;
                mod_refp.ctrans.y = ymax as f32 * mod_refp.cscale.z - mod_refp.ctrans.z;
                mod_refp.ctrans.z = temp;
                mod_refp.crot.x = (mod_refp.crot.x as f64 + 90.) as f32;
            }
            let temp = mod_refp.cscale.z;
            mod_refp.cscale.z = mod_refp.cscale.y;
            mod_refp.cscale.y = temp;
        }
    }

    /* Do scaling to image reference next */
    if to_image != 0 {
        /* Took out test and error if the existing ref did not have IMODF_OTRANS_ORIGIN set;
        this seems not to matter in the transform, and synthesized models can get the right
        scale and not set this flag */

        /* get the target transformation */
        use_ref.ctrans.x = hdata.xorg;
        use_ref.ctrans.y = hdata.yorg;
        use_ref.ctrans.z = hdata.zorg;
        use_ref.crot.x = hdata.tiltangles[3];
        use_ref.crot.y = hdata.tiltangles[4];
        use_ref.crot.z = hdata.tiltangles[5];
        use_ref.cscale = unit_pt;
        if hdata.xlen != 0. && hdata.mx != 0 {
            use_ref.cscale.x = hdata.xlen / hdata.mx as f32;
        }
        if hdata.ylen != 0. && hdata.my != 0 {
            use_ref.cscale.y = hdata.ylen / hdata.my as f32;
        }
        if hdata.zlen != 0. && hdata.mz != 0 {
            use_ref.cscale.z = hdata.zlen / hdata.mz as f32;
        }
        model.flags |= IMODF_TILTOK;
        if let Some(mod_refp) = model.ref_image {
            /* If there is a refImage, scale the data as when reading into 3dmod */
            use_ref.otrans = mod_refp.ctrans;
            use_ref.orot = mod_refp.crot;
            use_ref.oscale = mod_refp.cscale;
            imod_trans_from_ref_image(&mut model, &use_ref, unit_pt);
        } else {
            /* If there is no refImage, do not scale data but assign all the information
            just as if reading into 3dmod */
            /* The source `malloc`s the structure and the copy below leaves its
            `orot` and `oscale` as `useRef`'s uninitialised stack contents
            (`imodtrans.c:93`, `:404`); here they are `Iref_image::default()`'s. */
            model.ref_image = Some(Iref_image::default());
        }
        let mod_refp = model.ref_image.as_mut().unwrap();
        *mod_refp = use_ref;
        mod_refp.otrans = use_ref.ctrans;
        model.flags |= IMODF_OTRANS_ORIGIN | IMODF_TILTOK;

        /* Adjust the maxes for this change */
        model.xmax = hdata.nx;
        model.ymax = hdata.ny;
        model.zmax = hdata.nz;
    }

    /* Use zscale if user indicated it */
    if use_zscale != 0 && model.zscale > 0. {
        zscale = model.zscale;
    }

    /* Do flipping if model flipped and user did not turn it off */
    if model.flags & IMODF_FLIPYZ == 0 {
        doflip = 0;
    }

    /* Warning every time is too noxious!
    if (!newNx)
      printf("Assuming transformed images have same size as original "
              "images\n (use -n option to specify size of transformed "
              "images)\n");
    */
    new_cen.x = (if new_nx != 0 { new_nx } else { model.xmax }) as f32 * 0.5f32;
    new_cen.y = (if new_ny != 0 { new_ny } else { model.ymax }) as f32 * 0.5f32;
    new_cen.z = (if new_nz != 0 { new_nz } else { model.zmax }) as f32 * 0.5f32;

    if let Some(filename) = filename.as_deref() {
        if filetrans(
            filename,
            &mut model,
            mode,
            one_line,
            new_cen,
            zscale,
            doflip,
            trans_scale,
            transx,
            transy,
            transz,
        ) != 0
        {
            exit_error(b"Transforming model.");
        }
    } else if transopt != 0 {
        imod_trans_model3d(
            &mut model,
            &mut mat,
            Some(&mut norm_mat),
            new_cen,
            zscale,
            doflip,
        );
    }

    model.xmax = if new_nx != 0 { new_nx } else { model.xmax };
    model.ymax = if new_ny != 0 { new_ny } else { model.ymax };
    model.zmax = if new_nz != 0 { new_nz } else { model.zmax };
    let _ = imod_write(&model, &mut fout);
    drop(fout);
    crate::imod::libcfshr::b3dutil::exit(0);
}

/// Original: `filetrans` (`imodtrans.c:458`, static).
///
/// Transform by 2D or 3D transforms from a file
///  filename  has the name of the file with transforms
///  model     is the model
///  mode      is 2 for 2D, 3 for 3D transforms
///  oneLine   is line number for 2D applied to whole model
///  newCen    has the new center coordinates of the volume
///  zscale    is the z scale factor, or 1 not to use any
///  doflip    indicates transform native form of flipped data
///
/// Deviation note: `float mat[6]` is uninitialised in the source, so a first
/// 2D line with fewer than six numbers leaves the rest as stack contents;
/// here they start at 0.  Later lines keep the previous line's values, as in
/// the source, because the array lives outside the loop.
#[allow(clippy::too_many_arguments)]
fn filetrans(
    filename: &str,
    model: &mut Imod,
    mode: i32,
    one_line: i32,
    new_cen: Ipoint,
    zscale: f32,
    doflip: i32,
    trans_scale: f32,
    transx: f32,
    transy: f32,
    transz: f32,
) -> i32 {
    let mut line = [0u8; 80];
    let mut mat = [0f32; 6];
    let mut k = 0;
    let mut nread: i32;
    let mut mat3d: Imat = imod_mat_new(3).unwrap();

    let fin = ImodFile::open(filename, "r");

    let Some(mut fin) = fin else {
        let _ = ImodFile::Stdout.write_all(format!("ERROR: Couldn't open {filename}\n").as_bytes());
        return -1;
    };

    if mode == 2 {
        while fgetline(&mut fin, &mut line, 80) >= 0 {
            let end = line.iter().position(|&c| c == 0).unwrap_or(line.len());
            let text = String::from_utf8_lossy(&line[..end]);
            {
                let [m0, m1, m2, m3, m4, m5] = &mut mat;
                nread = sscanf(
                    &text,
                    "%f %f %f %f %f %f",
                    &mut [
                        ScanArg::Flt(m0),
                        ScanArg::Flt(m1),
                        ScanArg::Flt(m2),
                        ScanArg::Flt(m3),
                        ScanArg::Flt(m4),
                        ScanArg::Flt(m5),
                    ],
                );
            }
            // `sscanf` returns EOF for a line with no conversion input; the
            // translated scanner's -1 is that same value.
            if nread <= 0 {
                continue;
            }
            mat[4] = mat[4] * trans_scale + transx;
            mat[5] = mat[5] * trans_scale + transy;
            if one_line < 0 {
                trans_model_slice(model, &mat, k, new_cen);
            } else if k == one_line {
                mat3d.data[0] = mat[0];
                mat3d.data[4] = mat[1];
                // BUGS.md: `imodtrans.c:492,495` add `transx`/`transy` a
                // second time; they are already in `mat[4]`/`mat[5]` above.
                // Fixed in translation: the translation is applied once.
                mat3d.data[12] = mat[4];
                mat3d.data[1] = mat[2];
                mat3d.data[5] = mat[3];
                mat3d.data[13] = mat[5];
                mat3d.data[8] = 0.;
                mat3d.data[9] = 0.;
                mat3d.data[10] = 1.;
                mat3d.data[2] = 0.;
                mat3d.data[6] = 0.;
                mat3d.data[14] = transz;
                mat3d.data[15] = 1.;
                imod_trans_model3d(model, &mut mat3d, None, new_cen, zscale, doflip);
                return 0;
            }
            k += 1;
        }

        if one_line >= 0 {
            let _ = ImodFile::Stdout.write_all(
                format!("ERROR: End of file or error before line {one_line} of {filename}\n")
                    .as_bytes(),
            );
            return -1;
        }
    } else {
        k = 0;
        while k < 3 {
            if fgetline(&mut fin, &mut line, 80) <= 0 {
                let _ = ImodFile::Stdout
                    .write_all(format!("ERROR: Reading line {k} from  {filename}\n").as_bytes());
                return -1;
            }
            let end = line.iter().position(|&c| c == 0).unwrap_or(line.len());
            let text = String::from_utf8_lossy(&line[..end]);
            let ku = k as usize;
            let mut a = mat3d.data[ku];
            let mut b = mat3d.data[ku + 4];
            let mut c = mat3d.data[ku + 8];
            let mut d = mat3d.data[ku + 12];
            sscanf(
                &text,
                "%f %f %f %f",
                &mut [
                    ScanArg::Flt(&mut a),
                    ScanArg::Flt(&mut b),
                    ScanArg::Flt(&mut c),
                    ScanArg::Flt(&mut d),
                ],
            );
            mat3d.data[ku] = a;
            mat3d.data[ku + 4] = b;
            mat3d.data[ku + 8] = c;
            mat3d.data[ku + 12] = d;
            mat3d.data[ku + 12] *= trans_scale;
            k += 1;
        }
        mat3d.data[12] += transx;
        mat3d.data[13] += transy;
        mat3d.data[14] += transz;
        imod_trans_model3d(model, &mut mat3d, None, new_cen, zscale, doflip);
    }
    0
}

/// Original: `trans_model_slice` (`imodtrans.c:539`, static).
///
/// Translate one slice of the model
///  model  is the model
///  mat    is a 6-element matrix with transformation
///  slice  is the slice number
///  newCen has the new X and Y center coordinates
fn trans_model_slice(model: &mut Imod, mat: &[f32; 6], slice: i32, new_cen: Ipoint) -> i32 {
    let mut zval: i32;
    let mut x: f32;
    let mut y: f32;

    let xcen = model.xmax as f32 * 0.5f32;
    let ycen = model.ymax as f32 * 0.5f32;

    for ob in 0..model.obj.len() {
        let obj = &mut model.obj[ob];
        for co in 0..obj.cont.len() {
            let cont = &mut obj.cont[co];
            for pt in 0..cont.pts.len() {
                zval = (cont.pts[pt].z + 0.5f32) as i32;
                if zval == slice {
                    x = cont.pts[pt].x;
                    y = cont.pts[pt].y;
                    cont.pts[pt].x = mat[0] * (x - xcen) + mat[1] * (y - ycen) + mat[4] + new_cen.x;
                    cont.pts[pt].y = mat[2] * (x - xcen) + mat[3] * (y - ycen) + mat[5] + new_cen.y;
                }
            }
        }
    }
    0
}
