//! Translation of `IMOD/flib/subrs/imsubs/irdhdr.f90`.
#![allow(dead_code, unused_variables)]

use crate::imod::libiimod::unit_fileio::{iiu_file_info, iiuretbrief_, iiuretprint_};
use crate::imod::libiimod::unit_header::{
    iiu_ret_axis_map, iiu_ret_basic_head, iiu_ret_cell, iiu_ret_data_type, iiu_ret_delta,
    iiu_ret_imod_flags, iiu_ret_labels, iiu_ret_mrc_version, iiu_ret_num_extended, iiu_ret_origin,
    iiu_ret_rms, iiu_ret_size, iiu_ret_space_group, iiu_ret_tilt, iiu_ret_tilt_orig,
};

/// Original `irdhdr` (`irdhdr.f90:15`).
pub unsafe fn irdhdr(
    iunit: i32,
    nxyz: *mut i32,
    mxyz: *mut i32,
    imode: *mut i32,
    dmin: *mut f32,
    dmax: *mut f32,
    dmean: *mut f32,
) {
    unsafe {
        let if_brief = iiuretbrief_();
        let do_print = iiuretprint_() > 0;
        let do_extra = if_brief == 0 && do_print;
        iiu_ret_basic_head(iunit, nxyz, mxyz, imode, dmin, dmax, dmean);
        let mut imod_flags = 0;
        let mut is_imod = 0;
        let mut file_size = 0;
        let mut file_type = 0;
        let mut iflags = 0;
        let mut nversion = 0;
        let mut ispg = 0;
        iiu_ret_imod_flags(iunit, &mut imod_flags, &mut is_imod);
        iiu_file_info(iunit, &mut file_size, &mut file_type, &mut iflags);
        iiu_ret_mrc_version(iunit, &mut nversion);
        iiu_ret_space_group(iunit, &mut ispg);
        if do_extra && iflags & (1 << 2) != 0 {
            println!("\n                    This file has an old-style MRC header.");
        }
        if do_extra && iflags & (1 << 4) != 0 {
            println!("\n         This file has invalid axis indices, adjusting them to 1,2,3");
        }
        if do_extra && iflags & (1 << 3) != 0 {
            println!("\n  Assuming that the header entry for 128 integers per section is an error");
        }
        if do_extra && iflags & (1 << 5) != 0 {
            if file_type == 1 {
                println!("\n  This file is actually a 4-bit TIFF file");
            } else {
                println!("\n  This file actually has 4-bit values and is MRC mode 101");
            }
        }
        if do_extra && iflags & (1 << 6) != 0 {
            println!(
                "\n  This file actually has 4-bit values packed in bytes, with{:4} bytes in X",
                *nxyz / 2
            );
        }
        if do_extra && iflags & (1 << 7) != 0 {
            println!("\n         This MRC file is apparently inverted in Y");
        }
        let mut nxyzst = [0_i32; 3];
        iiu_ret_size(iunit, nxyz, mxyz, nxyzst.as_mut_ptr());
        let mut idtype = 0;
        let mut lensnum = 0;
        let mut nd1 = 0;
        let mut nd2 = 0;
        let mut vd1 = 0.0;
        let mut vd2 = 0.0;
        let mut delta = [0.0_f32; 3];
        let mut cell = [0.0_f32; 6];
        let mut mapcrs = [0_i32; 3];
        let mut xorig = 0.0;
        let mut yorig = 0.0;
        let mut zorig = 0.0;
        let mut labels = [[0_u8; 81]; 10];
        let mut num_labels = 0;
        let mut tilt = [0.0_f32; 3];
        let mut tilt_orig = [0.0_f32; 3];
        let mut num_extra = 0;
        let mut rms = 0.0;
        iiu_ret_data_type(
            iunit,
            &mut idtype,
            &mut lensnum,
            &mut nd1,
            &mut nd2,
            &mut vd1,
            &mut vd2,
        );
        iiu_ret_delta(iunit, delta.as_mut_ptr());
        iiu_ret_cell(iunit, cell.as_mut_ptr());
        iiu_ret_axis_map(iunit, mapcrs.as_mut_ptr());
        iiu_ret_origin(iunit, &mut xorig, &mut yorig, &mut zorig);
        iiu_ret_labels(iunit, labels.as_mut_ptr().cast(), &mut num_labels);
        iiu_ret_tilt(iunit, tilt.as_mut_ptr());
        iiu_ret_tilt_orig(iunit, tilt_orig.as_mut_ptr());
        iiu_ret_num_extended(iunit, &mut num_extra);
        iiu_ret_rms(iunit, &mut rms);
        if ispg == 401 && *nxyz.add(2) / *mxyz.add(2) > 1 {
            println!(
                "\n  This file is an MRC volume stack with{:7} subvolumes of{:4} x{:4} x{:4}",
                *nxyz.add(2) / *mxyz.add(2),
                *nxyz,
                *nxyz.add(1),
                *mxyz.add(2)
            );
        }
        let mut use_mode = *imode;
        if iflags & (1 << 8) != 0 {
            use_mode = 12;
        }
        if do_extra {
            let origin_text = if (nversion > 0 || (is_imod > 0 && imod_flags & 4 != 0))
                && (xorig != 0.0 || yorig != 0.0 || zorig != 0.0)
            {
                "(inverted_in_file_)"
            } else {
                "..................."
            };
            let min_max_text = if *dmin > *dmax {
                "...(undetermined)..."
            } else {
                "..................."
            };
            let mean_text = if *dmean < (*dmin).min(*dmax) {
                "...(undetermined)..."
            } else {
                "..................."
            };
            let mode_label = match use_mode {
                0 if iflags & (1 << 1) != 0 || imod_flags & 1 != 0 => "(bytes - signed in file)",
                0 => "(byte)",
                1 => "(16-bit integer)",
                2 => "(32-bit float)",
                3 => "(complex integer)",
                4 => "(complex)",
                6 => "(unsigned 16-bit integer)",
                12 => "(16-bit float)",
                16 => "RGB color",
                _ => "(unknown)",
            };
            println!(
                "\n Number of columns, rows, sections .....{:8}{:8}{:8}",
                *nxyz,
                *nxyz.add(1),
                *nxyz.add(2)
            );
            println!(
                " Map mode ..............................{:5}   {}",
                use_mode, mode_label
            );
            println!(
                " Start cols, rows, sects, grid x,y,z ...{:5}{:6}{:6} {:7}{:7}{:7}",
                nxyzst[0],
                nxyzst[1],
                nxyzst[2],
                *mxyz,
                *mxyz.add(1),
                *mxyz.add(2)
            );
            println!(
                " Pixel spacing (Angstroms).............. {:11.4}{:11.4}{:11.4}",
                delta[0], delta[1], delta[2]
            );
            println!(
                " Cell angles ...........................{:9.3}{:9.3}{:9.3}",
                cell[3], cell[4], cell[5]
            );
            let lxyz = [' ', 'X', 'Y', 'Z'];
            println!(
                " Fast, medium, slow axes ...............    {}    {}    {}",
                if (1..=3).contains(&mapcrs[0]) {
                    lxyz[mapcrs[0] as usize]
                } else {
                    ' '
                },
                if (1..=3).contains(&mapcrs[1]) {
                    lxyz[mapcrs[1] as usize]
                } else {
                    ' '
                },
                if (1..=3).contains(&mapcrs[2]) {
                    lxyz[mapcrs[2] as usize]
                } else {
                    ' '
                }
            );
            println!(
                " Origin on x,y,z ..{}.. {:12.4}{:12.4}{:12.4}",
                origin_text, xorig, yorig, zorig
            );
            let min_label = if *imode == 0 && (iflags & 2 != 0 || imod_flags & 1 != 0) {
                format!(" ({:13.5} in file)", *dmin - 128.0)
            } else {
                String::new()
            };
            let max_label = if *imode == 0 && (iflags & 2 != 0 || imod_flags & 1 != 0) {
                format!(" ({:13.5} in file)", *dmax - 128.0)
            } else {
                String::new()
            };
            let mean_label = if *imode == 0 && (iflags & 2 != 0 || imod_flags & 1 != 0) {
                format!(" ({:13.5} in file)", *dmean - 128.0)
            } else {
                String::new()
            };
            println!(
                " Minimum density ..{}..{:13.5}{}",
                min_max_text, *dmin, min_label
            );
            println!(
                " Maximum density ..{}..{:13.5}{}",
                min_max_text, *dmax, max_label
            );
            println!(
                " Mean density .....{}..{:13.5}{}",
                mean_text, *dmean, mean_label
            );
            if rms > 0.0 || (rms == 0.0 && (nversion > 0 || (is_imod != 0 && imod_flags & 8 != 0)))
            {
                println!(" RMS deviation from mean................{:13.5}", rms);
            }
            println!(
                " tilt angles (original,current) ........{:6.1}{:6.1}{:6.1}{:6.1}{:6.1}{:6.1}",
                tilt_orig[0], tilt_orig[1], tilt_orig[2], tilt[0], tilt[1], tilt[2]
            );
            println!(
                " Space group,# extra bytes,idtype,lens .{:9}{:9}{:9}{:9}\n",
                ispg, num_extra, idtype, lensnum
            );
            // FORMAT 1020: `1x,i5,' Titles :' / 10(19a4,a3/)`.
            println!(" {:5} Titles :", num_labels);
            for label in labels.iter().take(num_labels.clamp(0, 10) as usize) {
                println!("{}", String::from_utf8_lossy(&label[..79]));
            }
        }
        if do_print && if_brief > 0 {
            println!(
                " Dimensions:{:7}{:7}{:7}   Pixel size:{:11.4}{:11.4}{:11.4}",
                *nxyz,
                *nxyz.add(1),
                *nxyz.add(2),
                delta[0],
                delta[1],
                delta[2]
            );
            println!(
                " Mode:{:3}               Min, max, mean:{:13.5}{:13.5}{:13.5}",
                use_mode, *dmin, *dmax, *dmean
            );
            if num_labels > 0 {
                println!("{}", String::from_utf8_lossy(&labels[0][..79]));
            }
            if num_labels > 1 {
                println!(
                    "{}",
                    String::from_utf8_lossy(&labels[(num_labels - 1).min(9) as usize][..79])
                );
            }
            if if_brief < 2 {
                println!();
            }
        }
        if do_print && idtype > 0 && if_brief <= 0 {
            let lxyz = [' ', 'X', 'Y', 'Z'];
            if idtype == 1 {
                println!(
                    "     TILT data set, axis= {} delta,start angle= {:8.2}{:8.2}\n",
                    if (1..=3).contains(&nd1) {
                        lxyz[nd1 as usize]
                    } else {
                        ' '
                    },
                    vd1,
                    vd2
                );
            } else if idtype == 2 {
                println!(
                    " SERIAL STEREO data set, axis= {} left angle= {:8.2} right angle= {:8.2}\n",
                    if (1..=3).contains(&nd1) {
                        lxyz[nd1 as usize]
                    } else {
                        ' '
                    },
                    vd1,
                    vd2
                );
            } else if idtype == 3 {
                println!(
                    "     AVERAGED data set, Navg,Noffset   =  {:6}{:6}\n",
                    nd1, nd2
                );
            } else if idtype == 4 {
                println!(
                    "     AVG STEREO data set, Navg,Noffset=  {:3}{:3} L,R angles= {:8.2}{:8.2}\n",
                    nd1, nd2, vd1, vd2
                );
            }
        }
    }
}
