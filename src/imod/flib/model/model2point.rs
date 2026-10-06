//! Translation of `IMOD/flib/model/model2point.f90`.
//!
//! MODEL2POINT will convert an IMOD or WIMP model file to a list of points
//! with integer or floating point coordinates.  Each point will be written on
//! a separate line, and the coordinates may be written either with no
//! information about which object or contour the point came from, or with
//! contour numbers, or with object numbers.
//!
//! The main program maps to [`model2point`]; the `fortmodel` module arrays are
//! the [`FortModel`] that `readw_or_imod` fills.  The output unit is built as
//! a byte string record by record: a `$` format leaves the record open, so
//! the next `write` continues it, and an advancing `write` ends it.

use crate::imod::flib::image::densmatch::densmatch_g_edit;
use crate::imod::flib::subrs::compat::gfortran_rt::format_f;
use crate::imod::flib::subrs::hvem::dopen::dopen;
use crate::imod::flib::subrs::hvem::objtocont::objtocont;
use crate::imod::flib::subrs::hvem::parse_input_params::{
    exit_error, memory_error, pip_get_in_out_file, pip_get_logical, pip_read_or_parse_options,
};
use crate::imod::flib::subrs::model::fortmodel::FortModel;
use crate::imod::flib::subrs::model::readw_or_imod::{get_model_object_range, readw_or_imod};
use crate::imod::flib::subrs::model::scale_model::scale_model;
use crate::imod::libcfshr::b3dutil::{exit, imodgetenv};
use crate::imod::libcfshr::parse_params::{pip_get_boolean, pip_get_float, pip_get_integer};
use crate::imod::libimod::imodel_fwrap::{
    getcontpointsizes, getcontvalue, getimodmesh, getimodobjsize, getimodobjtimes,
    getpointvalue, getscatsize, imodpartialmode,
};
use std::io::Write;

/// `parameter (numOptions = 15)` (`model2point.f90:33`).
const MODEL2POINT_NUM_OPTIONS: i32 = 15;
/// Fallback PIP table, the `options(1)` string (`model2point.f90:35-41`).
const MODEL2POINT_OPTIONS: &str = "input:InputFile:FN:@output:OutputFile:FN:@float:FloatingPoint:B:@\
integer:IntegerOutput:B:@scaled:ScaledCoordinates:B:@\
object:ObjectAndContour:B:@contour:Contour:B:@zero:NumberedFromZero:B:@\
zcoord:ZCoordinatesFromZero:B:@sizes:PointSizes:B:@times:TimesForContours:B:@\
values:ValuesInLastColumn:I:@fill:FillValue:F:@mesh:MeshVertexCoordinates:I:@\
help:usage:B:";

/// Original program `model2point` (`model2point.f90:13`).
pub fn model2point() {
    let mut fm = FortModel::default();
    let mut modelfile = String::new();
    let mut pointfile = String::new();
    let mut print_obj = false;
    let mut print_cont = false;
    let mut floating = true;
    let mut scaled = false;
    let mut do_point_sizes = false;
    let mut do_times = false;
    let mut num_offset = 0_i32;
    let mut fill_val = 0.0_f32;
    let mut if_values = 0_i32;
    let mut if_mesh = 0_i32;
    let mut z_from_zero = false;
    let mut z_offset = 0.0_f32;
    let mut lim_vert = 0_i32;
    let mut lim_index = 0_i32;
    let mut lim_time = 0_i32;
    let mut vertices: Vec<f32>;
    let mut mesh_inds: Vec<i32>;
    let mut point_sizes: Vec<f32> = Vec::new();
    let mut i_cont_times: Vec<i32> = Vec::new();
    let mut scat_size = 0_i32;
    let mut num_sizes = 0_i32;
    let (mut mod_obj, mut mod_cont) = (0_i32, 0_i32);
    let mut size = 0.0_f32;
    let mut gen_val: f32;
    let (mut num_opt_arg, mut num_non_opt_arg) = (0_i32, 0_i32);
    // The output unit: whole records, plus the open (`$`) one.
    let mut out = String::new();
    // `103 format(i6,$)`, `109 format(f12.2,$)`
    let fmt_i = |value: i32, w: usize| -> String {
        let text = format!("{:>w$}", value);
        if text.len() > w { "*".repeat(w) } else { text }
    };
    // nint
    let nint = |value: f32| -> i32 { value.round() as i32 };
    //
    // Process environment variable for output type
    {
        let mut value = [b' '; 320];
        if imodgetenv(b"MODEL2POINT_INTEGERS", &mut value) == 0 {
            floating = false;
        }
    }
    //
    // Pip startup: set error, parse options, check help, set flag if used
    //
    pip_read_or_parse_options(
        &[MODEL2POINT_OPTIONS],
        MODEL2POINT_NUM_OPTIONS,
        "model2point",
        "ERROR: MODEL2POINT - ",
        false,
        2,
        1,
        1,
        &mut num_opt_arg,
        &mut num_non_opt_arg,
    );
    if pip_get_in_out_file("InputFile", 1, " ", &mut modelfile, 320) != 0 {
        exit_error("No input file specified");
    }
    if pip_get_in_out_file("OutputFile", 2, " ", &mut pointfile, 320) != 0 {
        exit_error("No output file specified");
    }
    //
    // read model in
    //
    imodpartialmode(1);
    fm.fm_max_obj_loaded = 1;
    fm.fm_boost_read_in_by = 1.05;
    let exist = readw_or_imod(modelfile.trim_end_matches(' '), &mut fm);
    if !exist {
        exit_error("Reading model file");
    }
    let nobj_tot = getimodobjsize();
    //
    let mut unit1 = dopen(1, &pointfile, "new", "f");
    let _ = pip_get_logical("ObjectAndContour", &mut print_obj);
    let _ = pip_get_logical("Contour", &mut print_cont);
    let mut integers = floating;
    let ierr = pip_get_logical("FloatingPoint", &mut floating);
    if pip_get_logical("IntegerOutput", &mut integers) == 0 {
        if ierr == 0 {
            exit_error("You cannot enter both -float and -integers");
        }
        floating = !integers;
    } else if ierr == 0 && integers && floating {
        println!(" The -float entry is no longer needed");
    }
    let _ = pip_get_logical("ScaledCoordinates", &mut scaled);
    let _ = pip_get_boolean(b"NumberedFromZero", &mut num_offset);
    let _ = pip_get_integer(b"ValuesInLastColumn", &mut if_values);
    let _ = pip_get_float(b"FillValue", &mut fill_val);
    let _ = pip_get_integer(b"MeshVertexCoordinates", &mut if_mesh);
    let _ = pip_get_logical("PointSizes", &mut do_point_sizes);
    let _ = pip_get_logical("TimesForContours", &mut do_times);
    let _ = pip_get_logical("ZCoordinatesFromZero", &mut z_from_zero);
    if z_from_zero && !floating {
        exit_error("You must output floating point values to have Z coordinates starting at 0");
    }
    if z_from_zero {
        z_offset = 0.5;
    }
    vertices = vec![0.0; 3 * 10];
    mesh_inds = vec![0; 10];
    memory_error(0, "initial arrays for mesh vertices");
    if do_point_sizes {
        point_sizes = vec![0.0; fm.max_pt.max(0) as usize];
        memory_error(0, "array for point sizes");
    }
    //
    // scan through all objects to get points
    //
    // An error exit flushes what the unit has buffered, as Fortran does at
    // `exit`, so the records written before it stay in the file.
    let fail = |out: &mut String, unit1: &mut std::fs::File, message: &str| -> ! {
        let _ = unit1.write_all(out.as_bytes());
        out.clear();
        exit_error(message)
    };
    let mut npnts = 0_i32;
    for imod_obj in 1..=nobj_tot {
        if !get_model_object_range(imod_obj, imod_obj, &mut fm) {
            let _ = unit1.write_all(out.as_bytes());
            println!();
            println!(
                "ERROR: MODEL2POINT - Loading data for object #{:>6}",
                imod_obj
            );
            let _ = std::io::stdout().flush();
            exit(1);
        }

        if !scaled {
            scale_model(0, &mut fm);
        }
        if do_point_sizes && getscatsize(imod_obj, &mut scat_size) != 0 {
            fail(&mut out, &mut unit1, "Getting object 3D point size from model");
        }
        //
        // Get mesh coordinates: first check size needed
        //
        if if_mesh > 0 {
            if if_mesh == 1 && fm.max_mod_obj > 0 {
                continue;
            }
            let mut new_vert_lim = 0_i32;
            let mut new_ind_lim = 0_i32;
            let ierr = getimodmesh(
                imod_obj,
                &mut vertices,
                &mut mesh_inds,
                &mut new_vert_lim,
                &mut new_ind_lim,
            );
            if ierr < 0 {
                fail(&mut out, &mut unit1, "Finding array size needed for mesh vertices");
            }
            if ierr > 0 || new_vert_lim == 0 {
                continue;
            }
            //
            // Allocate if necessary
            //
            if new_vert_lim > lim_vert || new_ind_lim > lim_index {
                lim_vert = new_vert_lim + 10;
                lim_index = new_ind_lim + 10;
                vertices = vec![0.0; (3 * lim_vert) as usize];
                mesh_inds = vec![0; lim_index as usize];
                memory_error(0, "arrays for mesh vertices");
            }
            //
            // Get and output the points
            //
            let (mut lv, mut li) = (lim_vert, lim_index);
            if getimodmesh(imod_obj, &mut vertices, &mut mesh_inds, &mut lv, &mut li) != 0 {
                fail(&mut out, &mut unit1, "Getting mesh vertices for object");
            }
            let mut ipnt = 1;
            while ipnt <= new_vert_lim {
                if print_obj {
                    out.push_str(&fmt_i(imod_obj - num_offset, 6));
                }
                // `104 format(3f12.2)`
                for ipt in 0..3 {
                    out.push_str(&format_f(vertices[((ipnt - 1) * 3 + ipt) as usize] as f64, 12, 2));
                }
                out.push('\n');
                npnts += 1;
                ipnt += 2;
            }
        } else {
            // Contour coordinates
            //
            if do_times {
                if fm.max_mod_obj > lim_time {
                    lim_time = fm.max_mod_obj + 10;
                    i_cont_times = vec![0; lim_time as usize];
                    memory_error(0, "array for contour times");
                }
                if getimodobjtimes(imod_obj, &mut i_cont_times) != 0 {
                    fail(&mut out, &mut unit1, "Getting contour time values");
                }
            }

            for iobject in 1..=fm.max_mod_obj {
                let ninobj = fm.npt_in_obj[(iobject - 1) as usize];
                if ninobj > 0 {
                    objtocont(iobject, &fm.obj_color, &mut mod_obj, &mut mod_cont);
                    if do_point_sizes
                        && getcontpointsizes(
                            mod_obj,
                            mod_cont,
                            &mut point_sizes,
                            fm.max_pt,
                            &mut num_sizes,
                        ) != 0
                    {
                        fail(&mut out, &mut unit1, "Getting point sizes for a contour");
                    }
                    for ipt in 1..=ninobj {
                        let ipnt =
                            fm.object[(ipt + fm.ibase_obj[(iobject - 1) as usize] - 1) as usize];
                        if ipnt > 0 {
                            let p = fm.p_coord[(ipnt - 1) as usize];
                            if do_point_sizes {
                                size = scat_size as f32;
                                if num_sizes > 0 && point_sizes[(ipt - 1) as usize] >= 0. {
                                    size = point_sizes[(ipt - 1) as usize];
                                }
                            }
                            if print_obj {
                                out.push_str(&fmt_i(mod_obj - num_offset, 6));
                            }
                            if print_cont || print_obj {
                                out.push_str(&fmt_i(mod_cont - num_offset, 6));
                            }
                            if (if_values < 0 && (ipt == 1 || !print_cont))
                                || if_values > 0
                                || do_point_sizes
                                || do_times
                            {
                                if floating {
                                    // `108 format(3f12.2, $)`
                                    out.push_str(&format_f(p[0] as f64, 12, 2));
                                    out.push_str(&format_f(p[1] as f64, 12, 2));
                                    out.push_str(&format_f((p[2] + z_offset) as f64, 12, 2));
                                } else {
                                    // `107 format(3i7, $)`
                                    out.push_str(&fmt_i(nint(p[0]), 7));
                                    out.push_str(&fmt_i(nint(p[1]), 7));
                                    out.push_str(&fmt_i(nint(p[2] + z_offset), 7));
                                }

                                if do_point_sizes {
                                    out.push_str(&format_f(size as f64, 12, 2));
                                }
                                if do_times && ipt == 1 {
                                    out.push_str(&fmt_i(i_cont_times[(iobject - 1) as usize], 6));
                                }

                                if (if_values < 0 && (ipt == 1 || !print_cont)) || if_values > 0 {
                                    if if_values < 0 {
                                        gen_val = fill_val;
                                        if ipt == 1
                                            && getcontvalue(mod_obj, mod_cont, &mut gen_val) != 0
                                        {
                                            gen_val = fill_val;
                                        }
                                    } else {
                                        gen_val = 0.;
                                        if getpointvalue(mod_obj, mod_cont, ipt, &mut gen_val) != 0
                                        {
                                            gen_val = fill_val;
                                        }
                                    }
                                    // `105 format(g16.6)`
                                    out.push_str(&densmatch_g_edit(gen_val, 16, 6));
                                    out.push('\n');
                                } else {
                                    // `write(1, 102)` with no items: ends the record
                                    out.push('\n');
                                }
                            } else if floating {
                                // `104 format(3f12.2)`
                                out.push_str(&format_f(p[0] as f64, 12, 2));
                                out.push_str(&format_f(p[1] as f64, 12, 2));
                                out.push_str(&format_f((p[2] + z_offset) as f64, 12, 2));
                                out.push('\n');
                            } else {
                                // `102 format(3i7)`
                                out.push_str(&fmt_i(nint(p[0]), 7));
                                out.push_str(&fmt_i(nint(p[1]), 7));
                                out.push_str(&fmt_i(nint(p[2] + z_offset), 7));
                                out.push('\n');
                            }
                            npnts += 1;
                        }
                    }
                }
            }
        }
    }
    // A record left open by a `$` format is ended when the unit closes.
    if !out.is_empty() && !out.ends_with('\n') {
        out.push('\n');
    }
    let _ = unit1.write_all(out.as_bytes());
    drop(unit1);
    println!("{:>12}  points output to file", npnts);
    let _ = std::io::stdout().flush();
    exit(0);
}
