//! `IMOD/imodutil/imodchopconts.cpp`: chops up contours from patch tracking,
//! or separates them by fine-grained color into new surfaces or objects.
//!
//! The source keeps its state in the small class `ChopConts`; it is
//! [`ChopConts`] here, with one method per member function.
//!
//! Aliasing: the source passes `Iobj *` pointers that may point at the same
//! object (`dupBreakAddCont(obj->cont, obj, ...)`,
//! `transferToSurfOrObj`'s `newObj`, `copyFineGrain(obj, obj, ...)`), which
//! Rust cannot borrow twice.  The methods therefore take object *indexes*
//! into `m_model.obj`; the "old contour" is always the object's contour 0 at
//! every call site (`obj->cont`), so it is addressed by its object index.

use std::collections::BTreeMap;
use std::io::Write;

use crate::imod::libcfshr::b3dutil::{
    CArg, ImodFile, c_format_bytes, exit, imod_backup_file, imod_prog_name, imod_usage_header,
    number_in_list, program_args,
};
use crate::imod::libcfshr::parse_params::{
    exit_error, pip_get_boolean, pip_get_in_out_file, pip_get_integer, pip_get_string,
    pip_print_help, pip_read_or_parse_options,
};
use crate::imod::libcfshr::parselist::parselist;
use crate::imod::libimod::icont::{imod_contour_break, imod_contour_clear, imod_contour_dup};
use crate::imod::libimod::imodel::{Imod, imod_new_object};
use crate::imod::libimod::imodel_files::{imod_open_file, imod_read, imod_write_file};
use crate::imod::libimod::iobj::{
    IMOD_OBJFLAG_THICK_CONT, imod_object_add_contour, imod_object_clean_surf,
    imod_object_copy_clear, imod_object_remove_contour, iobj_scat,
};
use crate::imod::libimod::istore::{
    DrawProps, GEN_STORE_COLOR, GEN_STORE_MINMAX1, GEN_STORE_SURFACE, Istore, StoreUnion,
    istore_add_min_max, istore_clear_range, istore_cont_surf_draw_props, istore_default_draw_props,
    istore_get_min_max, istore_insert, istore_list_point_props, istore_lookup,
};

/// The source's `printf`, through the C-format writer on libc-order stdout.
macro_rules! printf {
    ($fmt:expr $(, $arg:expr)* $(,)?) => {{
        let _ = ImodFile::Stdout.write_all(&c_format_bytes($fmt, &[$($arg),*]));
    }};
}

/// `exitError` with the source's variadic format.
macro_rules! exit_error_fmt {
    ($fmt:expr $(, $arg:expr)* $(,)?) => {
        exit_error(&c_format_bytes($fmt, &[$($arg),*]))
    };
}

/// `B3DNINT(a)`: `(int)floor((a) + 0.5)`.
macro_rules! b3dnint {
    ($a:expr) => {
        (($a) as f64 + 0.5).floor() as i32
    };
}

/// `B3DMAX(a,b)`: `((a) > (b) ? (a) : (b))`.
macro_rules! b3dmax {
    ($a:expr, $b:expr) => {{
        let (a, b) = ($a, $b);
        if a > b { a } else { b }
    }};
}

/// `B3DMIN(a,b)`: `((a) < (b) ? (a) : (b))`.
macro_rules! b3dmin {
    ($a:expr, $b:expr) => {{
        let (a, b) = ($a, $b);
        if a < b { a } else { b }
    }};
}

/// `RGB_VALUE(props)` (`imodchopconts.cpp:24-25`): one integer from a
/// `DrawProps` color.  `255.` is a double, so each float is widened first.
macro_rules! rgb_value {
    ($props:expr) => {
        (b3dnint!(255. * $props.red as f64) << 16)
            + (b3dnint!(255. * $props.green as f64) << 8)
            + b3dnint!(255. * $props.blue as f64)
    };
}

/// Original: class `ChopConts` (`imodchopconts.cpp:30-49`).
pub struct ChopConts {
    m_model: Imod,
    /// `DIMap mColorMap` (`std::map<double,int>`).  Every key the program
    /// makes is an integral double (`color + surf * 2^25`, computed in double
    /// as the source does), so the key converted to `i64` is exact and orders
    /// the keys as the `double`s do.
    m_color_map: BTreeMap<i64, i32>,
    m_surf_fac: f64,
    m_assign_surf: i32,
    m_num_after: i32,
}

/// Original: `main` (`imodchopconts.cpp:54`).
pub fn imodchopconts() {
    let mut chop = ChopConts::new();
    let argv = program_args();
    chop.main(&argv);
    exit(0);
}

/// `imodUsageHeader` as PIP's header callback.
fn imod_usage_header_for_pip(prog_name: &[u8]) {
    imod_usage_header(Some(&String::from_utf8_lossy(prog_name)));
    let _ = ImodFile::Stdout.flush();
}

impl ChopConts {
    /// Original: `ChopConts::ChopConts` (`imodchopconts.cpp:62`).
    pub fn new() -> Self {
        ChopConts {
            m_model: Imod::default(),
            m_color_map: BTreeMap::new(),
            m_surf_fac: 2f64.powf(25.),
            m_assign_surf: 0,
            m_num_after: 0,
        }
    }

    /// Original: `ChopConts::main` (`imodchopconts.cpp:71`).
    ///
    /// Defined behaviour (`BUGS.md`, imodchopconts): the source recovers the
    /// surface from a map key with `B3DNINT(mapKey) / mSurfFac`, an `int`
    /// conversion of `color + surf * 2^25` that is out of range (undefined
    /// behaviour; `INT_MIN` on x86) for any surface number of 64 or more.
    /// The rounding is done in double here, which gives the same result
    /// wherever the source's conversion is in range.
    pub fn main(&mut self, argv: &[String]) {
        let mut obj_list: Option<Vec<i32>> = None;
        let mut num_obj: i32 = 0;
        let mut len_contour: i32;
        let mut num_contour: i32 = 0;
        let mut len_contour_in: i32 = 0;
        let mut min_cont_overlap: i32 = 4;
        let mut no_overlap = 0;
        let mut break_at_color: i32 = 0;
        let mut ierr: i32 = 0;
        let mut surf: i32;
        let mut num_to_cut: i32;
        let mut max_len: i32;
        let mut len_conts: i32 = 0;
        let mut num_cont: i32;
        let mut lap_total: i32;
        let mut lap_base: i32 = 0;
        let mut lap_remainder: i32 = 0;
        let mut pt_base: i32;
        let mut last_new_ind: i32;
        let mut ind: i32;
        let mut ipnt: i32;
        let num_obj_orig: i32;
        let mut num_obj_tmp: i32 = 0;
        let mut num_before: i32;
        let mut cur_color: i32;
        let mut color: i32;
        let mut max_surf_or_obj: i32;
        let mut has_min_max: i32 = 0;
        let len_entered: i32;
        let mut val_min: f32 = 0.;
        let mut val_max: f32 = 0.;
        let mut store: Istore;
        let mut props = DrawProps::default();
        let mut dflt_props = DrawProps::default();
        let mut cont_props = DrawProps::default();
        let mut cont_state: i32 = 0;
        let mut surf_state: i32 = 0;
        let mut map_key: f64;
        let progname = imod_prog_name(argv.first().map(String::as_str).unwrap_or(""));
        let mut num_opt_args: i32 = 0;
        let mut num_non_opt_args: i32 = 0;

        // Fallbacks from    ../manpages/autodoc2man 2 1 imodchopconts
        let num_options: i32 = 7;
        let options: [&[u8]; 7] = [
            b"input:InputModel:FN:",
            b"output:OutputModel:FN:",
            b"length:LengthOfPieces:I:",
            b"overlap:MinimumOverlap:I:",
            b"number:NumberOfPieces:I:",
            b"surfaces:AssignSurfaces:B:",
            b"objects:ObjectsToDo:LI:",
        ];

        /* Startup with fallback */
        let argv_bytes = argv
            .iter()
            .map(|value| value.as_bytes().to_vec())
            .collect::<Vec<_>>();
        pip_read_or_parse_options(
            argv_bytes.len() as i32,
            &argv_bytes,
            &options,
            num_options,
            progname.as_bytes(),
            2,
            1,
            1,
            &mut num_opt_args,
            &mut num_non_opt_args,
            Some(imod_usage_header_for_pip),
        );
        if pip_get_boolean(b"usage", &mut ierr) == 0 {
            pip_print_help(progname.as_bytes(), 0, 1, 1);
            exit(0);
        }

        /* Get input and output files */
        let mut filename: Vec<u8> = Vec::new();
        if pip_get_in_out_file(b"InputModel", 0, &mut filename) != 0 {
            exit_error(b"No input model file specified");
        }

        let name = String::from_utf8_lossy(&filename).into_owned();
        self.m_model = match imod_read(&name) {
            Ok(model) => model,
            Err(_) => exit_error_fmt!("Reading model %s", CArg::Str(&name)),
        };

        let mut filename: Vec<u8> = Vec::new();
        if pip_get_in_out_file(b"OutputModel", 1, &mut filename) != 0 {
            exit_error(b"No output model file specified");
        }
        let filename = String::from_utf8_lossy(&filename).into_owned();

        // Get object list if any
        let mut list_string: Vec<u8> = Vec::new();
        if pip_get_string(b"ObjectsToDo", &mut list_string) == 0 {
            match parselist(&String::from_utf8_lossy(&list_string)) {
                Ok(list) => {
                    num_obj = list.len() as i32;
                    obj_list = Some(list);
                }
                Err(_) => exit_error(b"Bad entry in list of objects to do"),
            }
        }

        pip_get_integer(b"BreakAtColors", &mut break_at_color);
        if break_at_color > 0 {
            if pip_get_boolean(b"AssignSurfaces", &mut self.m_assign_surf) == 0
                || pip_get_integer(b"LengthOfPieces", &mut len_contour_in) == 0
                || pip_get_integer(b"MinimumOverlap", &mut min_cont_overlap) == 0
                || pip_get_integer(b"NumberOfPieces", &mut num_contour) == 0
            {
                exit_error(
                    b"You cannot enter -length, -number, -overlap, or -surfaces with -colors",
                );
            }
            self.m_assign_surf = (break_at_color == 1) as i32;
        }

        // Determine maximum length of contours
        max_len = 1;
        for ob_num in 0..self.m_model.obj.len() as i32 {
            if number_in_list(ob_num + 1, obj_list.as_deref(), num_obj, 1) == 0 {
                continue;
            }
            let obj = &self.m_model.obj[ob_num as usize];
            for co in 0..obj.cont.len() {
                max_len = b3dmax!(max_len, obj.cont[co].pts.len() as i32);
            }
        }

        // Set up default contour length as max of Z size of model and max contour length
        len_contour = b3dmax!(self.m_model.zmax, max_len);

        pip_get_boolean(b"AssignSurfaces", &mut self.m_assign_surf);

        // Get the length entry
        len_entered = 1 - pip_get_integer(b"LengthOfPieces", &mut len_contour_in);
        if len_entered != 0 {
            if len_contour_in == -1 {
                len_contour = b3dmax!(16, self.m_model.zmax / 5);
            } else {
                len_contour = len_contour_in;
            }
        }
        if len_contour <= 0 {
            exit_error(b"New contour length must be a positive number");
        }

        // Set new default of 0 for overlap if length is 1, then get overlap and process it
        if len_contour == 1 {
            min_cont_overlap = 0;
        }
        pip_get_integer(b"MinimumOverlap", &mut min_cont_overlap);
        if min_cont_overlap == -1 {
            no_overlap = 1;
            min_cont_overlap = 0;
        }
        if min_cont_overlap < -1 {
            exit_error(b"Contour overlap cannot be negative, other than -1 to enforce 0 overlap");
        }

        // Then process number option
        if pip_get_integer(b"NumberOfPieces", &mut num_contour) == 0 {
            if len_entered != 0 && len_contour_in > 0 {
                exit_error(
                    b"You cannot enter both -number and -length with a value greater than 0",
                );
            }

            // Derive length for new contours
            len_contour = b3dnint!((max_len + (num_contour - 1) * min_cont_overlap) / num_contour);
            printf!(
                "Maximum contour length = %d, length for new contours = %d\n",
                CArg::Int(max_len as i64),
                CArg::Int(len_contour as i64)
            );
        }
        if len_contour <= min_cont_overlap
            || (len_contour > 1 && len_contour < min_cont_overlap + 2)
        {
            exit_error_fmt!(
                "Contour length must be greater than the overlap%s",
                CArg::Str(if len_contour > 1 { " + 1" } else { "" })
            );
        }

        // Loop on objects
        num_before = 0;
        self.m_num_after = 0;
        num_obj_orig = self.m_model.obj.len() as i32;
        for ob_num in 0..num_obj_orig {
            if number_in_list(ob_num + 1, obj_list.as_deref(), num_obj, 1) == 0 {
                continue;
            }
            let obn = ob_num as usize;
            num_to_cut = self.m_model.obj[obn].cont.len() as i32;
            num_before += num_to_cut;

            if break_at_color <= 0 {
                surf = 1;
                max_len = 0;

                // Find maximum contour length and if it is less than the length to cut, skip
                // cutting
                for co in 0..num_to_cut as usize {
                    max_len = b3dmax!(max_len, self.m_model.obj[obn].cont[co].pts.len() as i32);
                }
                if max_len <= len_contour {
                    num_to_cut = 0;
                }

                // Loop on contours, each one is the first because it is deleted when done
                for _co in 0..num_to_cut {
                    // Set up the cutting as in tiltxcorr
                    ipnt = self.m_model.obj[obn].cont[0].pts.len() as i32;
                    if ipnt <= len_contour {
                        num_cont = 1;
                    } else {
                        len_conts = b3dmin!(len_contour, ipnt);
                        num_cont = (ipnt - 1) / (len_conts - min_cont_overlap) + 1;
                        lap_total = num_cont * len_conts - ipnt;
                        lap_base = lap_total / b3dmax!(num_cont - 1, 1);
                        lap_remainder = lap_total % b3dmax!(num_cont - 1, 1);

                        if no_overlap != 0 {
                            len_conts = ipnt / num_cont;
                            lap_remainder = ipnt % num_cont;
                            lap_base = 0;
                        }
                    }

                    // If only one contour, duplicate it and add it to the end
                    if num_cont == 1 {
                        last_new_ind = self.dup_break_add_cont(
                            obn,
                            obn,
                            -1,
                            -1,
                            if self.m_assign_surf != 0 { surf } else { -1 },
                        );
                    } else {
                        // Otherwise  loop on new contours, make a duplicate and use break
                        // function to get the contour and any associated data.  This takes
                        // care of fine-grained contour data
                        pt_base = 0;
                        last_new_ind = 0;
                        for new_co in 0..num_cont {
                            ind = pt_base + len_conts - 1;
                            if no_overlap != 0 && new_co < lap_remainder {
                                ind += 1;
                            }
                            last_new_ind = self.dup_break_add_cont(
                                obn,
                                obn,
                                pt_base,
                                ind,
                                if self.m_assign_surf != 0 { surf } else { -1 },
                            );

                            // Advance the base
                            pt_base += len_conts - lap_base;
                            if new_co < lap_remainder {
                                pt_base += if no_overlap != 0 { 1 } else { -1 };
                            }
                        }
                    }

                    // Copy any fine-grained object data for contour 0 for each new contour
                    self.copy_fine_grain(obn, obn, last_new_ind, num_cont, 0);

                    // Clear out the contour and delete it
                    let obj = &mut self.m_model.obj[obn];
                    imod_contour_clear(&mut obj.cont[0]);
                    imod_object_remove_contour(obj, 0);
                    surf += 1;
                }
                let obj = &mut self.m_model.obj[obn];
                obj.flags |= IMOD_OBJFLAG_THICK_CONT;
                self.m_num_after += obj.cont.len() as i32;
            } else {
                istore_default_draw_props(&self.m_model.obj[obn], &mut dflt_props);
                color = rgb_value!(dflt_props);

                // Get starting object or surface number for new surfs/objs, and start the
                // map with the index that needs to be used for contours at base color
                self.m_color_map.clear();
                if self.m_assign_surf != 0 {
                    imod_object_clean_surf(&mut self.m_model.obj[obn]);
                    max_surf_or_obj = self.m_model.obj[obn].surfsize + 1;
                    self.m_color_map
                        .insert((color as f64) as i64, max_surf_or_obj);
                    max_surf_or_obj += 1;
                } else {
                    max_surf_or_obj = self.m_model.obj.len() as i32;
                    self.m_color_map.insert((color as f64) as i64, ob_num);
                }

                // Loop on contours and find all the colors
                for co_num in 0..num_to_cut {
                    let obj = &self.m_model.obj[obn];
                    let contmp = &obj.cont[co_num as usize];
                    istore_cont_surf_draw_props(
                        &obj.store,
                        &dflt_props,
                        &mut cont_props,
                        co_num,
                        contmp.surf,
                        &mut cont_state,
                        &mut surf_state,
                    );
                    for pt_num in 0..contmp.pts.len() as i32 {
                        istore_list_point_props(&contmp.store, &cont_props, &mut props, pt_num);
                        color = rgb_value!(props);
                        map_key = color as f64;
                        if self.m_assign_surf != 0 {
                            map_key += contmp.surf as f64 * self.m_surf_fac;
                        }
                        if !self.m_color_map.contains_key(&((map_key) as i64)) {
                            self.m_color_map.insert((map_key) as i64, max_surf_or_obj);
                            max_surf_or_obj += 1;
                        }
                    }
                }

                // Add surface colors
                if self.m_assign_surf != 0 {
                    let entries: Vec<(i64, i32)> =
                        self.m_color_map.iter().map(|(k, v)| (*k, *v)).collect();
                    for (key, second) in entries {
                        map_key = key as f64;
                        // Defined behaviour: `B3DNINT(mapKey) / mSurfFac` in double
                        // (see the doc comment).
                        surf = ((map_key + 0.5).floor() / self.m_surf_fac) as i32;
                        color = b3dnint!(map_key - surf as f64 * self.m_surf_fac);
                        store = Istore::default();
                        store.type_ = GEN_STORE_COLOR;
                        store.index = StoreUnion::from_i(second);
                        store.flags = GEN_STORE_SURFACE;
                        store.value = StoreUnion::from_i(0);
                        let mut bytes = store.value.b();
                        bytes[0] = (color >> 16) as u8;
                        bytes[1] = ((color >> 8) & 255) as u8;
                        bytes[2] = (color & 255) as u8;
                        store.value.set_b(bytes);
                        if istore_insert(&mut self.m_model.obj[obn].store, store) != 0 {
                            exit_error(b"Adding fine-grained surface data");
                        }

                        // Duplicate non-color surface data for this surface into the new
                        // surface
                        let (first_ind, after_ind) =
                            istore_lookup(&self.m_model.obj[obn].store, surf);
                        if let Some(first_ind) = first_ind {
                            for ind in first_ind..after_ind {
                                let store_ptr = self.m_model.obj[obn].store[ind];
                                if (store_ptr.flags & GEN_STORE_SURFACE) != 0
                                    && store_ptr.type_ != GEN_STORE_COLOR
                                {
                                    store = store_ptr;
                                    store.index = StoreUnion::from_i(second);
                                    if istore_insert(&mut self.m_model.obj[obn].store, store) != 0 {
                                        exit_error(b"Adding fine-grained surface data");
                                    }
                                }
                            }
                        }
                    }
                } else {
                    // Or add the model objects
                    num_obj_tmp = self.m_model.obj.len() as i32;
                    for ob in num_obj_tmp..max_surf_or_obj {
                        if imod_new_object(&mut self.m_model) != 0 {
                            exit_error(b"Adding object to model");
                        }
                        let from = self.m_model.obj[obn].clone();
                        imod_object_copy_clear(&from, &mut self.m_model.obj[ob as usize]);
                    }
                    let obj = &self.m_model.obj[obn];
                    has_min_max = istore_get_min_max(
                        &obj.store,
                        obj.cont.len() as i32,
                        GEN_STORE_MINMAX1,
                        &mut val_min,
                        &mut val_max,
                    );

                    // Assign the colors
                    for (key, second) in self.m_color_map.iter() {
                        color = b3dnint!(*key as f64);
                        let ob = *second;
                        if ob != ob_num {
                            let target = &mut self.m_model.obj[ob as usize];
                            target.red = ((color >> 16) as f64 / 255.) as f32;
                            target.green = (((color >> 8) & 255) as f64 / 255.) as f32;
                            target.blue = ((color & 255) as f64 / 255.) as f32;
                        }
                    }

                    // Copy any non-color surface data to new objects
                    for ind in 0..self.m_model.obj[obn].store.len() {
                        let store_ptr = self.m_model.obj[obn].store[ind];
                        if (store_ptr.flags & GEN_STORE_SURFACE) != 0
                            && store_ptr.type_ != GEN_STORE_COLOR
                        {
                            for ob in num_obj_tmp..max_surf_or_obj {
                                if istore_insert(
                                    &mut self.m_model.obj[ob as usize].store,
                                    store_ptr,
                                ) != 0
                                {
                                    exit_error(b"Transferring fine-grained surface data");
                                }
                            }
                        }
                    }
                }

                // Loop on contours to cut them up.  Each time, the contour being cut is
                // contour 0.  Do not assign a cont pointer because it can change; be
                // careful with obj pointer
                for _co_num in 0..num_to_cut {
                    {
                        let obj = &self.m_model.obj[obn];
                        istore_cont_surf_draw_props(
                            &obj.store,
                            &dflt_props,
                            &mut cont_props,
                            0,
                            obj.cont[0].surf,
                            &mut cont_state,
                            &mut surf_state,
                        );
                        istore_list_point_props(&obj.cont[0].store, &cont_props, &mut props, 0);
                    }
                    cur_color = rgb_value!(props);
                    pt_base = 0;
                    let mut pt_num: i32 = 1;
                    while pt_num < self.m_model.obj[obn].cont[0].pts.len() as i32 {
                        istore_list_point_props(
                            &self.m_model.obj[obn].cont[0].store,
                            &cont_props,
                            &mut props,
                            pt_num,
                        );
                        color = rgb_value!(props);
                        if color != cur_color {
                            ind = pt_num;
                            if iobj_scat(self.m_model.obj[obn].flags) != 0 {
                                ind -= 1;
                            }
                            self.transfer_to_surf_or_obj(obn, cur_color, pt_base, ind, 0);
                            pt_base = pt_num;
                            cur_color = color;
                        }
                        pt_num += 1;
                    }

                    // Deal with the points left at the end: move whole contour to end if the
                    // final points are the whole thing, otherwise make final segment
                    self.transfer_to_surf_or_obj(obn, cur_color, pt_base, pt_num - 1, 1);

                    // Clear out the contour and delete it
                    let obj = &mut self.m_model.obj[obn];
                    imod_contour_clear(&mut obj.cont[0]);
                    imod_object_remove_contour(obj, 0);
                }

                // Assign all new objects the same min/max for consistent display
                if self.m_assign_surf == 0 && has_min_max != 0 {
                    for ob in num_obj_tmp..max_surf_or_obj {
                        istore_add_min_max(
                            &mut self.m_model.obj[ob as usize].store,
                            GEN_STORE_MINMAX1,
                            val_min,
                            val_max,
                        );
                    }
                }
            }
        }
        imod_backup_file(&filename);
        let mut file = match imod_open_file(&filename, "wb", &mut self.m_model) {
            Ok(file) => file,
            Err(_) => exit_error_fmt!("Opening new model %s", CArg::Str(&filename)),
        };
        let _ = imod_write_file(&self.m_model, &mut file);
        drop(file);
        printf!(
            "Number of contours in selected objects changed from %d to %d\n",
            CArg::Int(num_before as i64),
            CArg::Int(self.m_num_after as i64)
        );
        exit(0);
    }

    /// Original: `ChopConts::transferToSurfOrObj` (`imodchopconts.cpp:386`).
    ///
    /// Does the common tasks of transferring a contour segment to a new
    /// surface or object for the current color, either an internal segment
    /// (finish = 0) or a final segment (finish = 1).  `obj` is the index of
    /// the object whose contour 0 is being cut.
    fn transfer_to_surf_or_obj(
        &mut self,
        obj: usize,
        cur_color: i32,
        pt_base: i32,
        ind: i32,
        finish: i32,
    ) {
        let last_new_ind: i32;
        let mut map_key: f64 = cur_color as f64;
        if self.m_assign_surf != 0 {
            map_key += self.m_surf_fac * self.m_model.obj[obj].cont[0].surf as f64;
        }
        let surf = self.m_color_map[&((map_key) as i64)];
        let new_obj = if self.m_assign_surf == 0 {
            surf as usize
        } else {
            obj
        };
        if finish != 0 && pt_base == 0 {
            last_new_ind = self.dup_break_add_cont(
                obj,
                new_obj,
                -1,
                -1,
                if self.m_assign_surf != 0 { surf } else { -1 },
            );
        } else {
            last_new_ind = self.dup_break_add_cont(
                obj,
                new_obj,
                pt_base,
                ind,
                if self.m_assign_surf != 0 { surf } else { -1 },
            );
        }
        let cont = &mut self.m_model.obj[new_obj].cont[last_new_ind as usize];
        let end = cont.pts.len() as i32 - 1;
        istore_clear_range(&mut cont.store, GEN_STORE_COLOR, 0, end);

        // Copy non-color fine-grained object data for contour 0 for each new contour
        // one at a time, since they may be in different objects
        self.copy_fine_grain(obj, new_obj, last_new_ind, 1, 1);
        self.m_num_after += 1;
    }

    /// Original: `ChopConts::dupBreakAddCont` (`imodchopconts.cpp:416`).
    ///
    /// Duplicates a contour (contour 0 of object `old_obj`, the source's
    /// `oldCont` at every call), extracts just the segment between
    /// `ind_start` and `ind_end` inclusive if these are both non-negative,
    /// and adds the whole or segment to object `obj`.  Assigns the surface as
    /// `surf` if it is non-negative.
    fn dup_break_add_cont(
        &mut self,
        old_obj: usize,
        obj: usize,
        ind_start: i32,
        ind_end: i32,
        surf: i32,
    ) -> i32 {
        let Some(mut dup_cont) = imod_contour_dup(&self.m_model.obj[old_obj].cont[0]) else {
            exit_error(b"Duplicating contour")
        };
        let mut new_cont;
        if ind_start >= 0 && ind_end >= 0 {
            new_cont = match imod_contour_break(&mut dup_cont, ind_start, ind_end) {
                Some(cont) => cont,
                None => exit_error(b"Breaking out piece of contour"),
            };
            imod_contour_clear(&mut dup_cont);
        } else {
            new_cont = dup_cont;
        }
        if surf >= 0 {
            new_cont.surf = surf;
        }
        let last_new_ind = imod_object_add_contour(&mut self.m_model.obj[obj], new_cont);
        if last_new_ind < 0 {
            exit_error(b"Adding new contour to object");
        }
        last_new_ind
    }

    /// Original: `ChopConts::copyFineGrain` (`imodchopconts.cpp:445`).
    ///
    /// Copy any fine-grained object data for contour 0 in the old object to
    /// each new contour in the new object, `num_cont` contours ending with
    /// `last_new_ind`.  If `skip_color` is nonzero, it does not transfer color
    /// data.
    fn copy_fine_grain(
        &mut self,
        old_obj: usize,
        new_obj: usize,
        last_new_ind: i32,
        num_cont: i32,
        skip_color: i32,
    ) {
        let (first_ind, after_ind) = istore_lookup(&self.m_model.obj[old_obj].store, 0);
        if let Some(first_ind) = first_ind {
            for ind in first_ind..after_ind {
                let store_ptr = self.m_model.obj[old_obj].store[ind];
                if (skip_color != 0 && store_ptr.type_ == GEN_STORE_COLOR)
                    || (store_ptr.flags & GEN_STORE_SURFACE) != 0
                {
                    continue;
                }
                let mut store = store_ptr;
                for co_ind in last_new_ind + 1 - num_cont..=last_new_ind {
                    store.index = StoreUnion::from_i(co_ind);
                    if istore_insert(&mut self.m_model.obj[new_obj].store, store) != 0 {
                        exit_error(b"Adding fine-grained contour data");
                    }
                }
            }
        }
    }
}
