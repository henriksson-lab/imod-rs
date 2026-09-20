//! Translation of `IMOD/ctfplotter/ctfutils.cpp` and its paired header
//! `IMOD/include/ctfutils.h`.
//!
//! Utility functions shared with `ctfphaseflip`.  The source keeps the defocus
//! records in an `Ilist`; the list is only ever sized, indexed and inserted
//! into, so it is a `Vec<SavedDefocus>` here, per `NATIVE.md`.

use std::io::Write as _;

use crate::imod::clip::clip::{ScanArg, sscanf};
use crate::imod::libcfshr::b3dutil::{
    CArg, ImodFile, angle_within_limits, c_format_bytes, fgetline,
};
use crate::imod::libcfshr::parse_params::exit_error;

const MAX_LINE: i32 = 100;

// `b3dutil.h`'s `RADIANS_PER_DEGREE` macro.
const RADIANS_PER_DEGREE: f64 = 0.01745329252;

pub const DEF_FILE_HAS_ASTIG: i32 = 1;
pub const DEF_FILE_ASTIG_IN_RAD: i32 = 2;
pub const DEF_FILE_HAS_PHASE: i32 = 4;
pub const DEF_FILE_PHASE_IN_RAD: i32 = 8;
pub const DEF_FILE_INVERT_ANGLES: i32 = 16;
pub const DEF_FILE_HAS_CUT_ON: i32 = 32;

/// When there is a cuton, frequency at which phase occurs (`ctfutils.h:19`).
pub const FREQ_FOR_PHASE: f64 = 0.3;

/// C `SavedDefocus` (`ctfutils.h:21`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SavedDefocus {
    /// Starting slice of range, numbered from 0
    pub starting_slice: i32,
    /// Ending slice og range
    pub ending_slice: i32,
    /// Low angle of range in degrees
    pub l_angle: f64,
    /// High angle of range in degrees
    pub h_angle: f64,
    /// Defocus value, or defocus on short axis
    pub defocus: f64,
    /// Defocus value on long axis
    pub defocus2: f64,
    /// Astigmatism angle of short axis in degrees or radians
    pub astig_angle: f64,
    /// Phase shift in degrees or radians
    pub plate_phase: f64,
    /// cut-on frequency in 1/nm
    pub cut_on_freq: f64,
}

/// C `readTiltAngles` (`ctfutils.cpp:25`).
///
/// Reads a file of tilt angles whose name is in angleFile, and which must
/// have at least nzz lines.  angleSign should be 1., or -1. to invert the
/// sign of the angles.  Minimum and maximum angles are returned in minAngle
/// and maxAngle.  The return value is the array of tilt angles.
pub fn read_tilt_angles(
    angle_file: &[u8],
    nzz: i32,
    angle_sign: f32,
    min_angle: &mut f32,
    max_angle: &mut f32,
) -> Vec<f32> {
    let mut angle_str = [0u8; MAX_LINE as usize];
    let mut err: i32;
    let mut curr_angle: f32 = 0.;
    let mut tilt_angles: Vec<f32> = vec![0.; nzz.max(0) as usize];
    *min_angle = 10000.;
    *max_angle = -10000.;
    let Some(mut fp_angle) = ImodFile::open(&*String::from_utf8_lossy(angle_file), "r") else {
        exit_error(&c_format_bytes(
            "Opening tilt angle file %s",
            &[CArg::Bytes(angle_file)],
        ));
    };
    for k in 0..nzz {
        loop {
            err = fgetline(&mut fp_angle, &mut angle_str, MAX_LINE);
            if err != 0 {
                break;
            }
        }
        if err == -1 || err == -2 {
            exit_error(&c_format_bytes(
                "%seading tilt angle file %s\n",
                &[
                    CArg::Str(if err == -1 {
                        "R"
                    } else {
                        "End of file while r"
                    }),
                    CArg::Bytes(angle_file),
                ],
            ));
        }
        let end = angle_str
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(angle_str.len());
        sscanf(
            &String::from_utf8_lossy(&angle_str[..end]),
            "%f",
            &mut [ScanArg::Flt(&mut curr_angle)],
        );
        curr_angle *= angle_sign;
        // `B3DMIN`/`B3DMAX` are `a < b ? a : b`, taking the second operand
        // when the comparison is false.
        *min_angle = if *min_angle < curr_angle {
            *min_angle
        } else {
            curr_angle
        };
        *max_angle = if *max_angle > curr_angle {
            *max_angle
        } else {
            curr_angle
        };
        tilt_angles[k as usize] = curr_angle;
    }
    tilt_angles
}

/// C `readDefocusFile` (`ctfutils.cpp:64`).
///
/// Reads a defocus file whose name is in fnDefocus and stores the values
/// in a list of SavedDefocus structures, eliminating duplications if any.
/// The return value is the list, which may be empty if the file does not
/// exist.  The version number is returned in defVersion, the flags in
/// versFlags.  Astigmatism angles are converted to angles if necessary, and
/// phase plate shift is converted to radians.
pub fn read_defocus_file(
    fn_defocus: &[u8],
    def_version: &mut i32,
    vers_flags: &mut i32,
) -> Vec<SavedDefocus> {
    let mut saved = SavedDefocus::default();
    let mut line = [0u8; MAX_LINE as usize];
    let mut nchar: i32;
    let mut vers_tmp: i32;
    let mut read_astig = false;
    let mut read_phase = false;
    let mut read_cuton = false;
    let mut langtmp: f32;
    let mut hangtmp: f32;
    let mut defoctmp: f32;
    let mut defoc2tmp: f32;
    let mut astigtmp: f32;
    let mut phasetmp: f32;
    let mut cutontmp: f32;
    let mut angle_sign: f32 = 1.;
    let mut lst_saved: Vec<SavedDefocus> = Vec::new();
    let fp = ImodFile::open(&*String::from_utf8_lossy(fn_defocus), "r");
    *def_version = 0;
    *vers_flags = 0;
    vers_tmp = 0;
    if let Some(mut fp) = fp {
        loop {
            nchar = fgetline(&mut fp, &mut line, MAX_LINE);
            if nchar == -2 {
                break;
            }
            if nchar == -1 {
                exit_error(&c_format_bytes(
                    "Error reading defocus file %s",
                    &[CArg::Bytes(fn_defocus)],
                ));
            }
            if nchar != 0 {
                defoc2tmp = 0.;
                astigtmp = 0.;
                phasetmp = 0.;
                cutontmp = 0.;
                langtmp = 0.;
                hangtmp = 0.;
                defoctmp = 0.;
                let end = line.iter().position(|&b| b == 0).unwrap_or(line.len());
                let text = String::from_utf8_lossy(&line[..end]).into_owned();
                if read_astig && read_phase && read_cuton {
                    sscanf(
                        &text,
                        "%d %d %f %f %f %f %f %f %f",
                        &mut [
                            ScanArg::Int(&mut saved.starting_slice),
                            ScanArg::Int(&mut saved.ending_slice),
                            ScanArg::Flt(&mut langtmp),
                            ScanArg::Flt(&mut hangtmp),
                            ScanArg::Flt(&mut defoctmp),
                            ScanArg::Flt(&mut defoc2tmp),
                            ScanArg::Flt(&mut astigtmp),
                            ScanArg::Flt(&mut phasetmp),
                            ScanArg::Flt(&mut cutontmp),
                        ],
                    );
                } else if read_astig && read_phase {
                    sscanf(
                        &text,
                        "%d %d %f %f %f %f %f %f",
                        &mut [
                            ScanArg::Int(&mut saved.starting_slice),
                            ScanArg::Int(&mut saved.ending_slice),
                            ScanArg::Flt(&mut langtmp),
                            ScanArg::Flt(&mut hangtmp),
                            ScanArg::Flt(&mut defoctmp),
                            ScanArg::Flt(&mut defoc2tmp),
                            ScanArg::Flt(&mut astigtmp),
                            ScanArg::Flt(&mut phasetmp),
                        ],
                    );
                } else if read_astig {
                    sscanf(
                        &text,
                        "%d %d %f %f %f %f %f",
                        &mut [
                            ScanArg::Int(&mut saved.starting_slice),
                            ScanArg::Int(&mut saved.ending_slice),
                            ScanArg::Flt(&mut langtmp),
                            ScanArg::Flt(&mut hangtmp),
                            ScanArg::Flt(&mut defoctmp),
                            ScanArg::Flt(&mut defoc2tmp),
                            ScanArg::Flt(&mut astigtmp),
                        ],
                    );
                } else if read_phase && read_cuton {
                    sscanf(
                        &text,
                        "%d %d %f %f %f %f %f",
                        &mut [
                            ScanArg::Int(&mut saved.starting_slice),
                            ScanArg::Int(&mut saved.ending_slice),
                            ScanArg::Flt(&mut langtmp),
                            ScanArg::Flt(&mut hangtmp),
                            ScanArg::Flt(&mut defoctmp),
                            ScanArg::Flt(&mut phasetmp),
                            ScanArg::Flt(&mut cutontmp),
                        ],
                    );
                } else if read_phase {
                    sscanf(
                        &text,
                        "%d %d %f %f %f %f",
                        &mut [
                            ScanArg::Int(&mut saved.starting_slice),
                            ScanArg::Int(&mut saved.ending_slice),
                            ScanArg::Flt(&mut langtmp),
                            ScanArg::Flt(&mut hangtmp),
                            ScanArg::Flt(&mut defoctmp),
                            ScanArg::Flt(&mut phasetmp),
                        ],
                    );
                } else {
                    sscanf(
                        &text,
                        "%d %d %f %f %f %d",
                        &mut [
                            ScanArg::Int(&mut saved.starting_slice),
                            ScanArg::Int(&mut saved.ending_slice),
                            ScanArg::Flt(&mut langtmp),
                            ScanArg::Flt(&mut hangtmp),
                            ScanArg::Flt(&mut defoctmp),
                            ScanArg::Int(&mut vers_tmp),
                        ],
                    );
                }
                if lst_saved.is_empty() && *def_version == 0 {
                    *def_version = vers_tmp;
                    if *def_version > 2 {
                        *vers_flags = saved.starting_slice;
                        read_astig = (*vers_flags & DEF_FILE_HAS_ASTIG) != 0;
                        read_phase = (*vers_flags & DEF_FILE_HAS_PHASE) != 0;
                        read_cuton = (*vers_flags & DEF_FILE_HAS_CUT_ON) != 0;
                        if *vers_flags & DEF_FILE_INVERT_ANGLES != 0 {
                            angle_sign = -1.;
                        }
                        // The source's `continue` jumps to the `while (1)`
                        // condition, skipping the `if (nchar < 0) break;` below.
                        continue;
                    }
                }
                saved.l_angle = (angle_sign * langtmp) as f64;
                saved.h_angle = (angle_sign * hangtmp) as f64;
                saved.defocus = defoctmp as f64 / 1000.;
                saved.starting_slice -= 1;
                saved.ending_slice -= 1;
                if *vers_flags & DEF_FILE_ASTIG_IN_RAD != 0 {
                    astigtmp = (astigtmp as f64 / RADIANS_PER_DEGREE) as f32;
                }
                astigtmp = angle_within_limits(astigtmp, -90., 90.) as f32;
                if *vers_flags & DEF_FILE_PHASE_IN_RAD == 0 {
                    phasetmp = (phasetmp as f64 * RADIANS_PER_DEGREE) as f32;
                }
                saved.defocus2 = defoc2tmp as f64 / 1000.;
                saved.astig_angle = astigtmp as f64;
                if saved.defocus2 > saved.defocus {
                    saved.defocus2 = saved.defocus;
                    saved.defocus = defoc2tmp as f64 / 1000.;
                    saved.astig_angle -= 90.;
                }
                saved.plate_phase = (phasetmp as f64).abs();
                saved.cut_on_freq = cutontmp as f64;
                add_item_to_defocus_list(&mut lst_saved, saved);
            }
            if nchar < 0 {
                break;
            }
        }
    }
    lst_saved
}

/// C `addItemToDefocusList` (`ctfutils.cpp:154`).
///
/// Adds one item to the defocus list, keeping the list in order and
/// and avoiding duplicate starting and ending view numbers
pub fn add_item_to_defocus_list(lst_saved: &mut Vec<SavedDefocus>, to_save: SavedDefocus) -> i32 {
    let mut match_ind: i32 = -1;
    let mut insert_ind: i32 = 0;

    // Look for match or place to insert
    for i in 0..lst_saved.len() as i32 {
        let item = &mut lst_saved[i as usize];
        if item.starting_slice == to_save.starting_slice
            && item.ending_slice == to_save.ending_slice
        {
            match_ind = i;
            *item = to_save;
            break;
        }
        if item.l_angle + item.h_angle <= to_save.l_angle + to_save.h_angle {
            insert_ind = i + 1;
        }
    }

    // If no match, now insert
    if match_ind < 0 {
        lst_saved.insert(insert_ind as usize, to_save);
    }
    if match_ind < 0 { insert_ind } else { match_ind }
}

/// C `checkAndFixDefocusList` (`ctfutils.cpp:184`).
///
/// Analyzes the list of angular ranges and defocuses to see if the view
/// numbers are correct, using the nz tilt angles in angles (which can be None
/// if there are no tilt angles).  If it detects that the views are low by one,
/// it will adjust them.  If there are other inconsistencies, it returns 1.
pub fn check_and_fix_defocus_list(
    list: &mut [SavedDefocus],
    angles: Option<&[f32]>,
    nz: i32,
    def_version: i32,
) -> i32 {
    let mut all_equal = 1;
    let mut all_off_by_one = 1;
    let mut num_empty = 0;
    let mut min_start = 10000;
    let mut max_end = -100;
    let mut start: i32;
    let mut end: i32;
    let mut start_ambig: i32;
    let mut end_ambig: i32;
    let tol: f32 = 0.02;
    let debug = 0;

    for k in 0..list.len() as i32 {
        let item = list[k as usize];
        min_start = if min_start < item.starting_slice {
            min_start
        } else {
            item.starting_slice
        };
        max_end = if max_end > item.ending_slice {
            max_end
        } else {
            item.ending_slice
        };
        start_ambig = 0;
        end_ambig = 0;
        if nz == 1 || angles.is_none() {
            start = 0;
            end = 0;
            if debug != 0 {
                let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                    "item %d  start %d end %d low %f high %f  has to start %d end %d\n",
                    &[
                        CArg::Int(k.into()),
                        CArg::Int(item.starting_slice.into()),
                        CArg::Int(item.ending_slice.into()),
                        CArg::Dbl(item.l_angle),
                        CArg::Dbl(item.h_angle),
                        CArg::Int(start.into()),
                        CArg::Int(end.into()),
                    ],
                ));
            }
        } else {
            let angles = angles.expect("checked above");
            start = -1;
            end = -1;

            // Find first and last slice whose angle is within the range
            for i in 0..nz {
                if (item.l_angle - tol as f64) < angles[i as usize] as f64
                    && (angles[i as usize] as f64) < item.h_angle + tol as f64
                {
                    if start < 0 {
                        start = i;
                    }
                    end = i;
                }

                // But also see if there could be ambiguity about an angle near one
                // end of the range, provided this is not a single-image range
                // 7/27/18: I have no idea why finding an angle that is exact match is called
                // ambiguous!  And why if that is the case, it is uninformative about whether
                // views don't match or are not off by one
                if (item.h_angle - item.l_angle).abs() > tol as f64 {
                    if (item.l_angle - angles[i as usize] as f64).abs() < 1.01 * tol as f64 {
                        if angles[0] <= angles[(nz - 1) as usize] {
                            start_ambig = 1;
                        } else {
                            end_ambig = 1;
                        }
                        if debug != 0 {
                            let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                                "Angle %d, %f ambiguous to low angle %f, start %d end %d\n",
                                &[
                                    CArg::Int(i.into()),
                                    CArg::Dbl(angles[i as usize] as f64),
                                    CArg::Dbl(item.l_angle),
                                    CArg::Int(start_ambig.into()),
                                    CArg::Int(end_ambig.into()),
                                ],
                            ));
                        }
                    }
                    if (item.h_angle - angles[i as usize] as f64).abs() < 1.01 * tol as f64 {
                        if angles[0] <= angles[(nz - 1) as usize] {
                            end_ambig = 1;
                        } else {
                            start_ambig = 1;
                        }
                        if debug != 0 {
                            let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                                "Angle %d, %f ambiguous to high angle %f, start %d end %d\n",
                                &[
                                    CArg::Int(i.into()),
                                    CArg::Dbl(angles[i as usize] as f64),
                                    CArg::Dbl(item.h_angle),
                                    CArg::Int(start_ambig.into()),
                                    CArg::Int(end_ambig.into()),
                                ],
                            ));
                        }
                    }
                }
            }
            if debug != 0 {
                let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                    "item %d  start %d end %d low %f high %f  implied start %d end %d\n",
                    &[
                        CArg::Int(k.into()),
                        CArg::Int(item.starting_slice.into()),
                        CArg::Int(item.ending_slice.into()),
                        CArg::Dbl(item.l_angle),
                        CArg::Dbl(item.h_angle),
                        CArg::Int(start.into()),
                        CArg::Int(end.into()),
                    ],
                ));
            }

            // If can't find, skip this one, count how many times this happens
            // This used to guarantee it thought they were off by 1!
            if start < 0 || end < 0 {
                num_empty += 1;
                continue;
            }
        }
        if (start != item.starting_slice && start_ambig == 0)
            || (end != item.ending_slice && end_ambig == 0)
        {
            all_equal = 0;
            if debug != 0 {
                let _ = ImodFile::Stdout.write_all(b"Set allEqual 0\n");
            }
        }
        if (start != item.starting_slice + 1 && start_ambig == 0)
            || (end != item.ending_slice + 1 && end_ambig == 0)
        {
            all_off_by_one = 0;
            if debug != 0 {
                let _ = ImodFile::Stdout.write_all(b"Set allOffByOne 0\n");
            }
        }
    }

    // If we didn't find out anything, see if min and max are at least
    // consistent with the bug, and go ahead and adjust
    if all_equal != 0 && all_off_by_one != 0 && min_start == -1 && max_end < nz - 1 {
        all_equal = 0;
    }

    // If we didn't find out anything and there were no interior entries with nothing
    // in the range and it is new version, assume it is OK
    if all_equal != 0 && all_off_by_one != 0 && num_empty < 3 && def_version > 1 {
        all_off_by_one = 0;
    }

    // Adjust them all if all off by one as far as we can tell, for old data
    if (all_equal == 0 || all_off_by_one != 0) && def_version < 2 {
        let _ = ImodFile::Stdout.write_all(
            b"View numbers in defocus file appear to be low by 1;\n  adding 1 to correct for old Ctfplotter bug\n",
        );
        for k in 0..list.len() {
            list[k].starting_slice += 1;
            list[k].ending_slice += 1;
        }
    }
    if (all_equal == 0 && all_off_by_one != 0 && def_version < 2)
        || (all_equal != 0 && all_off_by_one == 0)
    {
        return 0;
    }
    1
}
