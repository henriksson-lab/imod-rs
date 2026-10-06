//! Translation of `IMOD/flib/model/repackseed.f90`.
//!
//! REPACKSEED is a companion program for the script Transferfid.  It is used
//! to give the user a list of how points correspond between the original
//! fiducial model and the new seed model, and to repack the seed model to
//! eliminate empty contours.
//!
//! The main program maps to [`repackseed`]; the `fortmodel` module arrays are
//! the [`FortModel`] that `readw_or_imod` fills.  All entries come from
//! standard input.  A list-directed or `(a)` read with no `END=`/`ERR=` that
//! fails is the gfortran runtime error, status 2.

use crate::imod::flib::subrs::compat::gfortran_rt::format_f;
use crate::imod::flib::subrs::hvem::dopen::dopen;
use crate::imod::flib::subrs::hvem::frefor::{ListItem, ListReadError, list_read};
use crate::imod::flib::subrs::hvem::int_iwrite::int_iwrite;
use crate::imod::flib::subrs::hvem::objtocont::objtocont;
use crate::imod::flib::subrs::hvem::parse_input_params::{exit_error, memory_error, set_exit_prefix};
use crate::imod::flib::subrs::model::fortmodel::FortModel;
use crate::imod::flib::subrs::model::readw_or_imod::readw_or_imod;
use crate::imod::flib::subrs::model::scale_model::scale_model;
use crate::imod::flib::subrs::model::write_wmod::write_wmod;
use crate::imod::libcfshr::b3dutil::exit;
use crate::imod::libcfshr::writelist::wrlist;
use crate::imod::libimod::imodel_fwrap::{deleteimodcont, getimodscales};
use std::io::{BufRead, BufReader, Write};

/// Original program `repackseed` (`repackseed.f90:8`).
pub fn repackseed() {
    let mut fm = FortModel::default();
    let abtext = ['A', 'B'];
    let (mut iz_orig, mut iz_trans, mut if_bto_a) = (0_i32, 0_i32, 0_i32);
    let (mut x_im_scale, mut y_im_scale, mut z_im_scale) = (0.0_f32, 0.0_f32, 0.0_f32);
    let mut stdin = std::io::stdin().lock();
    let runtime_abort = |err: ListReadError| -> ! {
        let _ = std::io::stdout().flush();
        match err {
            ListReadError::End => eprintln!("Fortran runtime error: End of file"),
            ListReadError::Error => eprintln!("Fortran runtime error: Bad value during read"),
        }
        exit(2);
    };
    // `read(5, 101) name`, `101 format(a)`, into a `character*320`
    let read_name = |stdin: &mut std::io::StdinLock| -> String {
        let mut line = Vec::new();
        match stdin.read_until(b'\n', &mut line) {
            Ok(0) | Err(_) => runtime_abort(ListReadError::End),
            Ok(_) => {}
        }
        if line.last() == Some(&b'\n') {
            line.pop();
        }
        line.truncate(320);
        String::from_utf8_lossy(&line).trim_end_matches(' ').to_owned()
    };
    let prompt = |text: &str| {
        print!(" {text}");
        let _ = std::io::stdout().flush();
    };
    //
    fm.fm_mod_size_type = 2;
    set_exit_prefix("ERROR: REPACKSEED - ");
    prompt("Name of fiducial file from first axis: ");
    let fid_name = read_name(&mut stdin);
    println!(" Enter name of file with X/Y/Z coordinates, or Return if none available");
    let xyz_name = read_name(&mut stdin);
    prompt("Name of input file with new seed model: ");
    let seed_name = read_name(&mut stdin);
    prompt("Name of output file for packed model: ");
    let out_file = read_name(&mut stdin);
    prompt("Name of output file for matching coordinates, or Return for none: ");
    let match_name = read_name(&mut stdin);
    prompt("Section numbers in original and second series, 1 for B to A: ");
    if let Err(err) = list_read(
        &mut stdin,
        &mut [
            ListItem::Integer(&mut iz_orig),
            ListItem::Integer(&mut iz_trans),
            ListItem::Integer(&mut if_bto_a),
        ],
    ) {
        runtime_abort(err);
    }
    let mut indab = 1_usize;
    if if_bto_a != 0 {
        indab = 2;
    }
    //
    // read original file
    //
    if !readw_or_imod(&fid_name, &mut fm) {
        exit_error("Reading original fiducial file");
    }
    scale_model(0, &mut fm);
    let n = fm.max_obj_num.max(0) as usize;
    let mut ifid_in_a = vec![0_i32; n];
    let mut map_list = vec![0_i32; n];
    let mut iobj_to_delete = vec![0_i32; n];
    let mut ixyz_cont = vec![0_i32; n];
    let mut ixyz_point = vec![0_i32; n];
    let mut ixyz_obj = vec![0_i32; n];
    let mut imod_obj = vec![0_i32; n];
    let mut imod_cont = vec![0_i32; n];
    let mut x_orig = vec![0.0_f32; n];
    let mut y_orig = vec![0.0_f32; n];
    memory_error(0, "arrays for object data");
    //
    // read xyz file if any
    //
    let mut nxyz = 0_usize;
    if !xyz_name.is_empty() {
        let mut unit1 = BufReader::new(dopen(1, &xyz_name, "ro", "f"));
        // 10 read(1,*,end = 20) ixyzPoint, dum, dum, dum, ixyzObj, ixyzCont
        loop {
            let (mut point, mut obj, mut cont) = (0_i32, 0_i32, 0_i32);
            let (mut d1, mut d2, mut d3) = (0.0_f32, 0.0_f32, 0.0_f32);
            match list_read(
                &mut unit1,
                &mut [
                    ListItem::Integer(&mut point),
                    ListItem::Real(&mut d1),
                    ListItem::Real(&mut d2),
                    ListItem::Real(&mut d3),
                    ListItem::Integer(&mut obj),
                    ListItem::Integer(&mut cont),
                ],
            ) {
                Ok(()) => {
                    ixyz_point[nxyz] = point;
                    ixyz_obj[nxyz] = obj;
                    ixyz_cont[nxyz] = cont;
                    nxyz += 1;
                }
                // 20 continue
                Err(ListReadError::End) => break,
                Err(err) => runtime_abort(err),
            }
        }
    }
    //
    // Keep track of original object numbers
    //
    let mut num_original = 0_i32;
    let max_obj_orig = fm.max_mod_obj;
    for iobj in 1..=fm.max_mod_obj as usize {
        let k = iobj - 1;
        objtocont(iobj as i32, &fm.obj_color, &mut imod_obj[k], &mut imod_cont[k]);
        x_orig[k] = 0.;
        y_orig[k] = 0.;
        let ibase = fm.ibase_obj[k];
        for ipt in 1..=fm.npt_in_obj[k] {
            let ip = fm.object[(ibase + ipt - 1) as usize];
            let p = fm.p_coord[(ip - 1) as usize];
            if (p[2] - iz_orig as f32).round() as i32 == 0 {
                x_orig[k] = p[0];
                y_orig[k] = p[1];
            }
        }

        if nxyz == 0 {
            //
            // if no xyz file available, have to assume that every contour with
            // at least one point will be a fiducial
            //
            if fm.npt_in_obj[k] > 0 {
                num_original += 1;
                ifid_in_a[k] = num_original;
            } else {
                ifid_in_a[k] = 0;
            }
        } else {
            //
            // look for object in xyz list, take number if found
            //
            ifid_in_a[k] = 0;
            for i in 0..nxyz {
                if ixyz_obj[i] == imod_obj[k] && ixyz_cont[i] == imod_cont[k] {
                    ifid_in_a[k] = ixyz_point[i];
                }
            }
        }
    }
    //
    // read new seed model
    //
    if !readw_or_imod(&seed_name, &mut fm) {
        exit_error("Reading new seed file");
    }
    scale_model(0, &mut fm);
    let mut unit1: Option<std::fs::File> = None;
    if !match_name.is_empty() {
        let mut file = dopen(1, &match_name, "new", "f");
        let ierr = getimodscales(&mut x_im_scale, &mut y_im_scale, &mut z_im_scale);
        if ierr == 0 {
            let _ = writeln!(
                file,
                "{:>6}{:>6}{:>6}{}",
                iz_orig,
                iz_trans,
                if_bto_a,
                format_f(x_im_scale as f64, 12, 3)
            );
        } else {
            let _ = writeln!(file, "{:>6}{:>6}{:>6}", iz_orig, iz_trans, if_bto_a);
        }
        unit1 = Some(file);
    }
    //
    // pack objects down and accumulate map list
    //
    let mut nmap = 0_usize;
    let mut num_delete = 0_usize;
    for iobj in 1..=fm.max_mod_obj as usize {
        let k = iobj - 1;
        if fm.npt_in_obj[k] > 0 {
            if !match_name.is_empty()
                && x_orig[k] != 0.
                && y_orig[k] != 0.
                && iobj as i32 <= max_obj_orig
            {
                let ip = fm.object[fm.ibase_obj[k] as usize];
                let p = fm.p_coord[(ip - 1) as usize];
                // `107 format(4f12.3)`
                if let Some(file) = unit1.as_mut() {
                    let _ = writeln!(
                        file,
                        "{}{}{}{}",
                        format_f(x_orig[k] as f64, 12, 3),
                        format_f(y_orig[k] as f64, 12, 3),
                        format_f(p[0] as f64, 12, 3),
                        format_f(p[1] as f64, 12, 3)
                    );
                }
            }
            nmap += 1;
            map_list[nmap - 1] = ifid_in_a[k];
            fm.npt_in_obj[nmap - 1] = fm.npt_in_obj[k];
            fm.ibase_obj[nmap - 1] = fm.ibase_obj[k];
            fm.obj_color[nmap - 1] = fm.obj_color[k];
        } else {
            num_delete += 1;
            iobj_to_delete[num_delete - 1] = iobj as i32;
        }
    }

    for iobj in nmap + 1..=fm.max_mod_obj as usize {
        fm.npt_in_obj[iobj - 1] = 0;
    }
    fm.max_mod_obj = nmap as i32;

    // Delete contours in inverse order
    for ip in (1..=num_delete).rev() {
        let k = (iobj_to_delete[ip - 1] - 1) as usize;
        if deleteimodcont(imod_obj[k], imod_cont[k]) != 0 {
            exit_error("Deleting contour from IMOD model");
        }
    }

    //
    // output the model and the map list
    //
    scale_model(1, &mut fm);
    write_wmod(&out_file, &mut fm);
    if !match_name.is_empty() {
        // `108 format(//,' The correspondence between points is as follows',/)`
        print!("\n\n The correspondence between points is as follows\n\n");
        drop(unit1.take());
    } else {
        // `102 format(//,' Make the following entries ...',/, ' fiducials ...',/)`
        print!(
            "\n\n Make the following entries to Setupcombine or Solvematch to describe how \n fiducials correspond between the first and second axes\n\n"
        );
    }
    let mut out_buf = [b' '; 320];
    let mut nmap_len = 0_i32;
    int_iwrite(&mut out_buf, nmap as i32, &mut nmap_len);
    let nmap_text = String::from_utf8_lossy(&out_buf[..nmap_len.max(0) as usize]).into_owned();
    let nmap_i32 = nmap as i32;

    if if_bto_a == 0 {
        // `109 format(' Points in ',a,': ',$)`
        print!(" Points in {}: ", abtext[indab - 1]);
        let _ = std::io::stdout().flush();
        wrlist(&map_list, &nmap_i32);
        // `110 format(' Points in ',a,': 1-',a)`
        println!(" Points in {}: 1-{}", abtext[2 - indab], nmap_text);
    } else {
        println!(" Points in {}: 1-{}", abtext[2 - indab], nmap_text);
        print!(" Points in {}: ", abtext[indab - 1]);
        let _ = std::io::stdout().flush();
        wrlist(&map_list, &nmap_i32);
    }

    if !match_name.is_empty() {
        print!(
            "\n Solvematch will be able to match up points using data in {}\n regardless of how well points track or whether you add or delete points\n",
            match_name.trim_end_matches(' ')
        );
    } else {
        if nxyz == 0 {
            print!(
                "\n These lists may be wrong if they include a contour in {} that will not\n correspond to a fiducial point (e.g., if it has only one point)\n",
                abtext[indab - 1]
            );
        }
        print!(
            "\n These lists may be thrown off if you delete a contour in either set\n and will be thrown off if one of the seed points fails to track\n"
        );
    }
    let _ = std::io::stdout().flush();
    exit(0);
}
