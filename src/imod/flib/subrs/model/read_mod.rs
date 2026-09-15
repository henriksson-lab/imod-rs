//! ASCII WIMP model reader from `IMOD/flib/subrs/model/read_mod.f`.
#![allow(dead_code)]

use std::io::BufRead;

use crate::imod::flib::subrs::model::fortmodel::{FortModel, MAX_CLABEL};

/// Original: `read_mod` (`read_mod.f:17`).
///
/// The source reads Fortran unit 20, opened by `readw_or_imod`, and returns
/// its result in the `fortmodel` module arrays; `unit20` and `fm` carry those
/// two pieces of program state.
///
/// Deviation: every `read(20,101,err=151)` here is an `err=` branch with no
/// `end=` branch, so hitting end of file is a gfortran runtime error that
/// terminates the program with status 2 and an address-bearing backtrace on
/// stderr (verified against the reference build with a one-line WIMP file).
/// That backtrace is not reproducible, so a truncated file returns `.false.`
/// here and `convertmod` reports `Error reading mode file` instead.
pub fn read_mod(unit20: &mut impl BufRead, fm: &mut FortModel) -> bool {
    // `character*80 string`, blank filled by the `(90a)` edit descriptor.
    let mut string = [b' '; 80];
    let mut record = Vec::new();
    // `character*10 label_c`
    let mut label_c = [b' '; 10];

    fm.n_clabel = 0;
    let read_mod = false;
    // `read(20,101,err=151)string`
    record.clear();
    if unit20.read_until(b'\n', &mut record).unwrap_or(0) == 0 {
        return read_mod;
    }
    if record.last() == Some(&b'\n') {
        record.pop();
    }
    string = [b' '; 80];
    for (index, byte) in record.iter().take(80).enumerate() {
        string[index] = *byte;
    }
    let indstr: usize = if &string[0..5] == b"Model" {
        1
    } else if &string[0..5] == b" Mode" {
        2
    } else {
        return read_mod;
    };

    // `ii`, `i` and `irec` are printed by the `151` handler before any of them
    // is necessarily set; the reference build prints zeros for the two that
    // are never assigned on the earliest failure path.
    let mut ii: i32 = 0;
    let mut i: i32 = 0;
    let mut irec: i32 = 0;
    let mut ninobj: i32 = 0;
    let reached_end = 'read151: {
        // `read(20,101,err=151)string` / `read(string(indstr:),'(38x,i5)')`
        for target in 0..3 {
            record.clear();
            if unit20.read_until(b'\n', &mut record).unwrap_or(0) == 0 {
                break 'read151 false;
            }
            if record.last() == Some(&b'\n') {
                record.pop();
            }
            string = [b' '; 80];
            for (index, byte) in record.iter().take(80).enumerate() {
                string[index] = *byte;
            }
            let mut digits = String::new();
            for byte in &string[indstr - 1 + 38..indstr - 1 + 43] {
                if *byte != b' ' {
                    digits.push(*byte as char);
                }
            }
            let value: i32 = if digits.is_empty() {
                0
            } else {
                match digits.parse::<i32>() {
                    Ok(value) => value,
                    Err(_) => break 'read151 false,
                }
            };
            match target {
                0 => fm.max_mod_obj = value,
                1 => fm.n_point = value,
                _ => fm.n_object = value,
            }
        }
        // `read(20,101,err=151) string` — Object sequence :
        record.clear();
        if unit20.read_until(b'\n', &mut record).unwrap_or(0) == 0 {
            break 'read151 false;
        }
        fm.n_object = 0;
        fm.ibase_free = 0;
        fm.ntot_in_obj = 0;
        fm.max_mod_obj = 0;
        i = 1;
        while i <= fm.max_obj_num {
            fm.npt_in_obj[i as usize - 1] = 0;
            i += 1;
        }
        // `100 read(20,101,err=151)string`
        loop {
            record.clear();
            if unit20.read_until(b'\n', &mut record).unwrap_or(0) == 0 {
                break 'read151 false;
            }
            if record.last() == Some(&b'\n') {
                record.pop();
            }
            string = [b' '; 80];
            for (index, byte) in record.iter().take(80).enumerate() {
                string[index] = *byte;
            }
            if &string[indstr - 1..indstr + 3] == b" Obj" {
                // `read(string(indstr:),'(10x,i15)',err=151)i`
                let mut digits = String::new();
                for byte in &string[indstr - 1 + 10..indstr - 1 + 25] {
                    if *byte != b' ' {
                        digits.push(*byte as char);
                    }
                }
                i = if digits.is_empty() {
                    0
                } else {
                    match digits.parse::<i32>() {
                        Ok(value) => value,
                        Err(_) => break 'read151 false,
                    }
                };
                // `read(20,101,err=151)string` / `'(11x,i15)'` -> ninobj
                record.clear();
                if unit20.read_until(b'\n', &mut record).unwrap_or(0) == 0 {
                    break 'read151 false;
                }
                if record.last() == Some(&b'\n') {
                    record.pop();
                }
                string = [b' '; 80];
                for (index, byte) in record.iter().take(80).enumerate() {
                    string[index] = *byte;
                }
                let mut digits = String::new();
                for byte in &string[indstr - 1 + 11..indstr - 1 + 26] {
                    if *byte != b' ' {
                        digits.push(*byte as char);
                    }
                }
                ninobj = if digits.is_empty() {
                    0
                } else {
                    match digits.parse::<i32>() {
                        Ok(value) => value,
                        Err(_) => break 'read151 false,
                    }
                };
                // `read(20,101,err=151)string` / `'(15x,i1,2x,i3)'`
                record.clear();
                if unit20.read_until(b'\n', &mut record).unwrap_or(0) == 0 {
                    break 'read151 false;
                }
                if record.last() == Some(&b'\n') {
                    record.pop();
                }
                string = [b' '; 80];
                for (index, byte) in record.iter().take(80).enumerate() {
                    string[index] = *byte;
                }
                if i < 1 || i > fm.max_obj_num {
                    // Fortran indexes `obj_color(1,i)` without a bound check;
                    // an out-of-range object number is source-level UB.
                    break 'read151 false;
                }
                let mut digits = String::new();
                for byte in &string[indstr - 1 + 15..indstr - 1 + 16] {
                    if *byte != b' ' {
                        digits.push(*byte as char);
                    }
                }
                fm.obj_color[i as usize - 1][0] = if digits.is_empty() {
                    0
                } else {
                    match digits.parse::<i32>() {
                        Ok(value) => value,
                        Err(_) => break 'read151 false,
                    }
                };
                let mut digits = String::new();
                for byte in &string[indstr - 1 + 18..indstr - 1 + 21] {
                    if *byte != b' ' {
                        digits.push(*byte as char);
                    }
                }
                fm.obj_color[i as usize - 1][1] = if digits.is_empty() {
                    0
                } else {
                    match digits.parse::<i32>() {
                        Ok(value) => value,
                        Err(_) => break 'read151 false,
                    }
                };
                // `read(20,101,err=151)string` — the column heading
                record.clear();
                if unit20.read_until(b'\n', &mut record).unwrap_or(0) == 0 {
                    break 'read151 false;
                }
                fm.n_object += 1;
                fm.obj_order[fm.n_object as usize - 1] = i;
                fm.ndx_order[i as usize - 1] = fm.n_object;
                fm.npt_in_obj[i as usize - 1] = ninobj;
                fm.ntot_in_obj += ninobj;
                fm.ibase_obj[i as usize - 1] = fm.ibase_free;
                fm.max_mod_obj = fm.max_mod_obj.max(i);
                ii = 1;
                while ii <= ninobj {
                    // `read(20,101,err=151)string`
                    record.clear();
                    if unit20.read_until(b'\n', &mut record).unwrap_or(0) == 0 {
                        break 'read151 false;
                    }
                    if record.last() == Some(&b'\n') {
                        record.pop();
                    }
                    string = [b' '; 80];
                    for (index, byte) in record.iter().take(80).enumerate() {
                        string[index] = *byte;
                    }
                    // `'(i6,1x,3(f7.2,1x),2x,i1,2x,a10)'`
                    let base = indstr - 1;
                    let mut digits = String::new();
                    for byte in &string[base..base + 6] {
                        if *byte != b' ' {
                            digits.push(*byte as char);
                        }
                    }
                    let mut ipt: i32 = if digits.is_empty() {
                        0
                    } else {
                        match digits.parse::<i32>() {
                            Ok(value) => value,
                            Err(_) => break 'read151 false,
                        }
                    };
                    let mut coordinate = [0.0f32; 3];
                    for (index, start) in [base + 7, base + 15, base + 23].iter().enumerate() {
                        let mut digits = String::new();
                        for byte in &string[*start..*start + 7] {
                            if *byte != b' ' {
                                digits.push(*byte as char);
                            }
                        }
                        coordinate[index] = if digits.is_empty() {
                            0.0
                        } else {
                            match digits.parse::<f32>() {
                                // `F7.2` supplies the two implied decimal
                                // places only when the field has no point.
                                Ok(value) => {
                                    if digits.contains('.') {
                                        value
                                    } else {
                                        value / 100.0
                                    }
                                }
                                Err(_) => break 'read151 false,
                            }
                        };
                    }
                    let p1 = coordinate[0];
                    let p2 = coordinate[1];
                    let p3 = coordinate[2];
                    let mut digits = String::new();
                    for byte in &string[base + 33..base + 34] {
                        if *byte != b' ' {
                            digits.push(*byte as char);
                        }
                    }
                    let imark: i32 = if digits.is_empty() {
                        0
                    } else {
                        match digits.parse::<i32>() {
                            Ok(value) => value,
                            Err(_) => break 'read151 false,
                        }
                    };
                    label_c.copy_from_slice(&string[base + 36..base + 46]);
                    irec = ipt;
                    if ipt > 0 {
                        // DNM: if index is > # of points, then it must be old
                        // model and needs fixing
                        if ipt > fm.n_point {
                            ipt %= 10000;
                        }
                        // `mod(ipt,10000)` is 0 for an exact multiple of
                        // 10000, and the Fortran then stores into
                        // `p_coord(1,0)` / `pt_label(0)`, one element before
                        // the allocated arrays.  That is source-level UB, not
                        // an error, so the store is dropped here and the read
                        // carries on exactly as the Fortran does.
                        if ipt >= 1 && ipt <= fm.max_pt {
                            fm.p_coord[ipt as usize - 1][0] = p1;
                            fm.p_coord[ipt as usize - 1][1] = p2;
                            fm.p_coord[ipt as usize - 1][2] = p3;
                            fm.pt_label[ipt as usize - 1] = imark as u8;
                        }
                        // DNM: if label is on a real point, add to list
                        if label_c != [b' '; 10] {
                            fm.n_clabel += 1;
                            // The Fortran writes past `clabel(max_clabel)`
                            // once more than 200 labels appear; that overrun
                            // is source-level UB and is dropped here.
                            if fm.n_clabel as usize <= MAX_CLABEL {
                                fm.label_list[fm.n_clabel as usize - 1] = ipt;
                                fm.clabel[fm.n_clabel as usize - 1] = label_c;
                            }
                        }
                    }
                    let slot = ii + fm.ibase_free;
                    if slot < 1 || slot > fm.len_object {
                        break 'read151 false;
                    }
                    fm.object[slot as usize - 1] = ipt;
                    ii += 1;
                }
                fm.ibase_free += ninobj;
            // `go to 100`
            } else if &string[0..5] == b"     " {
                break 'read151 true;
            } else {
                break 'read151 false;
            }
        }
    };
    if reached_end {
        fm.nin_order = fm.n_object;
        return true;
    }
    // `151 close(20)`
    println!(" Model file format incompatible,input a new one");
    println!("  ipt={ii:12}{i:12}{irec:12}");
    false
}
