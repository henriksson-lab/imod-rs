//! ASCII WIMP model writer from `IMOD/flib/subrs/model/store_mod.f`.
use std::io::Write;

use crate::imod::flib::subrs::model::fortmodel::FortModel;

/// Original: `store_mod` (`store_mod.f:18`).
///
/// The source writes Fortran unit 20, opened `status='new'` by the caller, and
/// takes the model from the `fortmodel` module arrays; `unit20` and `fm` carry
/// those two pieces of program state.  `model_file` is declared `character*50`
/// here regardless of the caller's longer string, and only its first 30
/// characters reach the file.
pub fn store_mod(unit20: &mut impl Write, model_file: &str, fm: &mut FortModel) {
    // `character*80 string,sttt`
    let mut dummy: [u8; 39];

    if fm.n_point <= 0 {
        return;
    }
    let mut maxlen: i32 = 1;
    let mut i: i32 = 1;
    while i <= fm.max_mod_obj {
        maxlen = maxlen.max(fm.npt_in_obj[i as usize - 1]);
        i += 1;
    }
    let needmax: i32 = fm.max_mod_obj.max(
        fm.max_obj_num
            .min((fm.len_object as f32 * fm.n_object as f32 / maxlen as f32).sqrt() as i32),
    );
    // `write(20,'(a80)',err=99)sttt` then `rewind(20)` writes one record of
    // uninitialised stack into the freshly created file and immediately
    // rewinds over it; gfortran truncates the sequential file at the end of
    // the records written after the rewind, so nothing of it survives.

    dummy = *b"Model file name........................";
    // `write(20,'(1x,a39,a30)',err=99)dummy,model_file(1:30)`
    let mut name = [b' '; 30];
    for (index, byte) in model_file.as_bytes().iter().take(30).enumerate() {
        name[index] = *byte;
    }
    if unit20.write_all(b" ").is_err()
        || unit20.write_all(&dummy).is_err()
        || unit20.write_all(&name).is_err()
        || unit20.write_all(b"\n").is_err()
    {
        println!(" format error, model file is truncated");
        return;
    }
    for (literal, value) in [
        (b"max # of object........................", needmax),
        (b"# of node..............................", fm.n_point),
        (b"# of object............................", fm.n_object),
    ] {
        dummy = *literal;
        // `write(20,'(1x,a38,i5)',err=99)dummy,<value>` — `a38` drops the
        // last character of the 39-character `dummy`.
        let field = format!("{value:5}");
        let field = if field.len() > 5 {
            "*****".to_string()
        } else {
            field
        };
        if unit20.write_all(b" ").is_err()
            || unit20.write_all(&dummy[0..38]).is_err()
            || unit20.write_all(field.as_bytes()).is_err()
            || unit20.write_all(b"\n").is_err()
        {
            println!(" format error, model file is truncated");
            return;
        }
    }

    // `dummy='  Object sequence :'` / `write(20,'(a20)',err=99)dummy(1:20)`
    if unit20.write_all(b"  Object sequence : \n").is_err() {
        println!(" format error, model file is truncated");
        return;
    }
    i = 1;
    while i <= fm.max_mod_obj {
        if fm.npt_in_obj[i as usize - 1] > 0 {
            let object_field = format!("{i:8}");
            let object_field = if object_field.len() > 8 {
                "********".to_string()
            } else {
                object_field
            };
            let point_field = format!("{:10}", fm.npt_in_obj[i as usize - 1]);
            let point_field = if point_field.len() > 10 {
                "**********".to_string()
            } else {
                point_field
            };
            let color1 = format!("{:1}", fm.obj_color[i as usize - 1][0]);
            let color1 = if color1.len() > 1 {
                "*".to_string()
            } else {
                color1
            };
            let color2 = format!("{:3}", fm.obj_color[i as usize - 1][1]);
            let color2 = if color2.len() > 3 {
                "***".to_string()
            } else {
                color2
            };
            let string_47 = b"     #    X       Y       Z      Mark    Label ";
            if unit20
                .write_all(format!("  Object #:{object_field}\n").as_bytes())
                .is_err()
                || unit20
                    .write_all(format!(" # of point:{point_field}\n").as_bytes())
                    .is_err()
                || unit20
                    .write_all(format!(" Display switch:{color1}  {color2}\n").as_bytes())
                    .is_err()
                || unit20.write_all(string_47).is_err()
                || unit20.write_all(b"\n").is_err()
            {
                println!(" format error, model file is truncated");
                return;
            }
            let mut ii: i32 = 1;
            while ii <= fm.npt_in_obj[i as usize - 1] {
                let ipnt = fm.object[(ii + fm.ibase_obj[i as usize - 1]) as usize - 1];
                let it = ipnt.abs();
                // `p_coord(1,it)` with `it == 0`, or with `it` past
                // `max_pt`, indexes outside the Fortran array; that is
                // source-level UB.  The reference build reads the zeroed heap
                // just before `p_coord` there, so an out-of-range point number
                // contributes zeros rather than neighbouring model data.
                let coordinate = if it >= 1 && it <= fm.max_pt {
                    fm.p_coord[it as usize - 1]
                } else {
                    [0.0; 3]
                };
                let mark = if it >= 1 && it <= fm.max_pt {
                    fm.pt_label[it as usize - 1]
                } else {
                    0
                };
                let mut record: Vec<u8> = Vec::new();
                record.push(b' ');
                let field = format!("{ipnt:6}");
                if field.len() > 6 {
                    record.extend_from_slice(b"******");
                } else {
                    record.extend_from_slice(field.as_bytes());
                }
                record.push(b' ');
                for axis in 0..3 {
                    let field = format!("{:7.2}", coordinate[axis]);
                    if field.len() > 7 {
                        record.extend_from_slice(b"*******");
                    } else {
                        record.extend_from_slice(field.as_bytes());
                    }
                    record.push(b' ');
                }
                let field = format!("{:3}", mark);
                if field.len() > 3 {
                    record.extend_from_slice(b"***");
                } else {
                    record.extend_from_slice(field.as_bytes());
                }
                // DNM: change format to allow >100000 points, pt_label 3 digits
                if fm.n_clabel != 0 {
                    // search for labels at "it"
                    let mut ilab: i32 = 1;
                    let mut indlab: i32 = 0;
                    while ilab <= fm.n_clabel && indlab == 0 {
                        if fm.label_list[ilab as usize - 1] == it {
                            indlab = ilab;
                        }
                        ilab += 1;
                    }
                    if indlab != 0 {
                        record.extend_from_slice(b"  ");
                        record.extend_from_slice(&fm.clabel[indlab as usize - 1]);
                    }
                }
                record.push(b'\n');
                if unit20.write_all(&record).is_err() {
                    println!(" format error, model file is truncated");
                    return;
                }
                ii += 1;
            }
        }
        i += 1;
    }
    // `write(20,'(a)')' '` and `write(20,'(a5)')'  END'`
    let _ = unit20.write_all(b" \n");
    let _ = unit20.write_all(b"  END\n");
}
