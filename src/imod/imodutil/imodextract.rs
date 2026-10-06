//! Translation of `IMOD/imodutil/imodextract.c`: extracts (or, with `-d`,
//! deletes) a list of objects or object groups from a model.
//!
//! The program maps to [`imodextract`] and the static `usage` to [`usage`].
//! `subtomosetup -objects` calls [`imodextract_main`] in process, through the
//! same `argv` the command line would carry.

use std::io::Write;

use crate::imod::libcfshr::b3dutil::{
    CArg, ImodFile, c_format_bytes, exit, imod_backup_file, imod_prog_name, number_in_list,
    program_args,
};
use crate::imod::libcfshr::parse_params::{exit_error, setExitPrefix};
use crate::imod::libcfshr::parselist::parselist;
use crate::imod::libimod::imodel::imod_new_object;
use crate::imod::libimod::imodel_files::{imod_open_file, imod_read, imod_write_file};
use crate::imod::libimod::iobj::imod_object_copy;
use crate::imod::libimod::iview::imod_objview_complete;
use crate::imod::libimod::objgroup::obj_group_list_to_obj_list;

/// `exitError` with the source's variadic format.
macro_rules! exit_error_fmt {
    ($fmt:expr $(, $arg:expr)* $(,)?) => {
        exit_error(&c_format_bytes($fmt, &[$($arg),*]))
    };
}

/// Original static `usage` (`imodextract.c:16`).
fn usage() -> ! {
    let mut out = ImodFile::Stdout;
    let _ = out.write_all(b"Usage: imodextract [-g|-d] listOfObjects inputModel outputModel\n");
    let _ = out.write_all(b"       The list of objects can include ranges, e.g. 1-3,6,9,13-15\n");
    let _ = out.write_all(b"  Options:\n");
    let _ = out.write_all(b"  -g   The list has object group numbers instead of object numbers\n");
    let _ = out.write_all(b"  -d   Delete the list of objects or object groups\n");
    exit(3);
}

/// Original program `main` (`imodextract.c:26`).
pub fn imodextract() {
    let argv = program_args();
    imodextract_main(&argv);
}

/// The body of `main`, taking `argv` (so `subtomosetup` can run it in
/// process).  Ends through `exit`, as the source does.
pub fn imodextract_main(argv: &[String]) -> ! {
    let argc = argv.len();
    let mut list_ind: usize = 1;
    let mut groups = false;
    let mut delete = false;
    let progname = imod_prog_name(argv.first().map(String::as_str).unwrap_or(""));
    let prefix = format!("ERROR: {} - ", progname);
    setExitPrefix(prefix.as_bytes());

    if argc < 3 {
        usage();
    }

    let mut iarg = 1;
    while iarg < argc {
        let arg = argv[iarg].as_bytes();
        if arg.first() == Some(&b'-') {
            match arg.get(1).copied().unwrap_or(0) {
                b'g' => {
                    groups = true;
                    list_ind += 1;
                }
                b'd' => {
                    delete = true;
                    list_ind += 1;
                }
                _ => exit_error_fmt!("Invalid option %s", CArg::Str(&argv[iarg])),
            }
        } else {
            break;
        }
        iarg += 1;
    }

    if argc != 3 + list_ind {
        usage();
    }

    // `parselist` returns NULL for an empty string as well as for a bad entry
    // (`parselist.c:51-52`); the Rust routine returns an empty list there.
    let mut list = match parselist(&argv[list_ind]) {
        Ok(list) if !argv[list_ind].is_empty() => list,
        _ => exit_error_fmt!(
            "Parsing %s list",
            CArg::Str(if groups { "object group" } else { "object" })
        ),
    };
    let mut nlist = list.len();

    let mut in_model = match imod_read(&argv[list_ind + 1]) {
        Ok(model) => model,
        Err(_) => exit_error_fmt!("Reading model %s", CArg::Str(&argv[list_ind + 1])),
    };

    // For groups, check that they exist
    if groups && in_model.group_list.is_empty() {
        exit_error(b"The model has no object groups");
    }

    // Check entered numbers
    let tot_num = if groups {
        in_model.group_list.len() as i32
    } else {
        in_model.obj.len() as i32
    };
    for i in 0..nlist {
        if list[i] < 1 || list[i] > tot_num {
            exit_error_fmt!(
                "Invalid object%s number %d",
                CArg::Str(if groups { " group" } else { "" }),
                CArg::Int(list[i] as i64)
            );
        }
        list[i] -= 1;
    }

    // Replace the list with complement
    if delete {
        let mut inv_list = vec![0_i32; in_model.obj.len()];

        // Replace group list by object list, and now stop consider input as groups
        if groups {
            obj_group_list_to_obj_list(&in_model.group_list, &mut list, None, 0);
            nlist = list.len();
        }
        groups = false;
        let mut inv_num = 0usize;
        for i in 0..in_model.obj.len() {
            if number_in_list(i as i32, Some(&list), nlist as i32, 0) == 0 {
                inv_list[inv_num] = i as i32;
                inv_num += 1;
            }
        }
        if inv_num == 0 {
            exit_error(b"No objects are left after deleting the specified ones");
        }
        nlist = inv_num;
        inv_list.truncate(inv_num);
        list = inv_list;
    }

    // For extracting groups, convert to object list and get new group list
    let mut new_groups = Vec::with_capacity(nlist);
    if groups {
        // Convert to list of objects, get replacement group list
        let ob =
            obj_group_list_to_obj_list(&in_model.group_list, &mut list, Some(&mut new_groups), 0);
        nlist = list.len();
        if ob == 1 {
            exit_error_fmt!(
                "An object group number is out of range; there are %d groups",
                CArg::Int(in_model.group_list.len() as i64)
            );
        }
        if ob == 2 {
            exit_error(b"Allocating new object list or new object group");
        }
    }

    /* Make nlist new objects */
    let origsize = in_model.obj.len();
    for _ in 0..nlist {
        imod_new_object(&mut in_model);
    }

    /* Extend the object views then shift all existing objects and their views
    to top */
    imod_objview_complete(&mut in_model);
    for ob in (0..origsize).rev() {
        let from = in_model.obj[ob].clone();
        imod_object_copy(&from, &mut in_model.obj[ob + nlist]);
        for iview in 1..in_model.view.len() {
            in_model.view[iview].objview[ob + nlist] = in_model.view[iview].objview[ob].clone();
        }
    }

    /* Now copy all of the selected ones into place, copying the view down
    too */
    for i in 0..nlist {
        let ob = list[i] as usize;
        let from = in_model.obj[ob + nlist].clone();
        imod_object_copy(&from, &mut in_model.obj[i]);
        for iview in 1..in_model.view.len() {
            in_model.view[iview].objview[i] = in_model.view[iview].objview[ob + nlist].clone();
        }
    }

    // If no groups, fix the existing groups
    if !groups && !in_model.group_list.is_empty() {
        for i in 0..in_model.group_list.len() {
            let obj_group = &mut in_model.group_list[i];
            for ob in (0..obj_group.obj_list.len()).rev() {
                let nob = obj_group.obj_list[ob];
                let mut found = false;
                for new_ob in 0..nlist {
                    if list[new_ob] == nob {
                        obj_group.obj_list[ob] = new_ob as i32;
                        found = true;
                        break;
                    }
                }
                if !found {
                    obj_group.obj_list.remove(ob);
                }
            }
        }
    }

    /* Delete extra objects by just setting objsize, and the object view sizes,
    set current indexes to -1 to avoid problems */
    in_model.obj.truncate(nlist);
    in_model.cindex.point = -1;
    in_model.cindex.contour = -1;
    in_model.cindex.object = 0;
    for iview in 1..in_model.view.len() {
        in_model.view[iview].objview.truncate(nlist);
    }

    // Assign new list of groups (let it leak)
    if groups {
        in_model.group_list = new_groups;
    }

    let out_name = &argv[argc - 1];
    imod_backup_file(out_name);
    let mut fout = match imod_open_file(out_name, "wb", &mut in_model) {
        Ok(file) => file,
        Err(_) => exit_error_fmt!("Opening new model %s", CArg::Str(out_name)),
    };
    let _ = imod_write_file(&in_model, &mut fout);
    drop(fout);
    exit(0);
}
