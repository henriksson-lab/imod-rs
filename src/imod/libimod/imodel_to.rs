//! Export routines from `IMOD/libimod/imodel_to.c`.

use std::io::Write;

use super::imodel::Imod;

/// Original: `imod_to_wmod` (`imodel_to.c:36`).
pub fn imod_to_wmod(imod: &Imod, output: &mut impl Write, filename: &str) -> Result<(), i32> {
    let contour_count: usize = imod.obj.iter().map(|object| object.cont.len()).sum();
    let point_total: usize = imod
        .obj
        .iter()
        .flat_map(|object| &object.cont)
        .map(|contour| contour.pts.len())
        .sum();
    write!(output, " Model file name........................{filename}").map_err(|_| 11)?;
    let blank_count = 31_i32 - filename.len() as i32;
    if blank_count < 0 {
        for _ in 0..blank_count {
            write!(output, " ").map_err(|_| 11)?;
        }
    }
    writeln!(output).map_err(|_| 11)?;
    writeln!(
        output,
        " max # of object....................... {:4}",
        2 * contour_count
    )
    .map_err(|_| 11)?;
    writeln!(
        output,
        " # of node............................. {:4}",
        2 * point_total
    )
    .map_err(|_| 11)?;
    writeln!(
        output,
        " # of object........................... {:4}",
        contour_count
    )
    .map_err(|_| 11)?;
    writeln!(output, "  Object sequence : ").map_err(|_| 11)?;
    let mut object_count = 1_i32;
    let mut point_count = 0_i32;
    let mut display_switch = 247_i32;
    for object in &imod.obj {
        if display_switch > 255 {
            display_switch = 247;
        }
        for contour in &object.cont {
            writeln!(output, "  Object #: {:11}", object_count).map_err(|_| 11)?;
            object_count += 1;
            writeln!(output, " # of point: {:11}", contour.pts.len()).map_err(|_| 11)?;
            writeln!(output, " Display switch:1  {display_switch}").map_err(|_| 11)?;
            writeln!(output, "     #    X       Y       Z      Mark    Label ").map_err(|_| 11)?;
            point_count += 1;
            for point in &contour.pts {
                write!(output, "{:7}", point_count).map_err(|_| 11)?;
                point_count += 1;
                write!(output, " {:7.2}", point.x).map_err(|_| 11)?;
                write!(output, " {:7.2}", point.y).map_err(|_| 11)?;
                writeln!(output, " {:7.2}   0", point.z).map_err(|_| 11)?;
            }
        }
    }
    writeln!(output, "\n  END").map_err(|_| 11)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::libimod::imodel::{Icont, Imod, Iobj, Ipoint};

    #[test]
    fn imod_to_wmod_has_c_source_layout_and_numbers() {
        let model = Imod {
            obj: vec![Iobj {
                cont: vec![Icont {
                    pts: vec![Ipoint {
                        x: 1.0,
                        y: 2.0,
                        z: 3.0,
                    }],
                    ..Icont::default()
                }],
                ..Iobj::default()
            }],
            ..Imod::default()
        };
        let mut text = Vec::new();
        imod_to_wmod(&model, &mut text, "out.wimp").unwrap();
        assert_eq!(
            String::from_utf8(text).unwrap(),
            concat!(
                " Model file name........................out.wimp\n",
                " max # of object.......................    2\n",
                " # of node.............................    2\n",
                " # of object...........................    1\n",
                "  Object sequence : \n",
                "  Object #:           1\n",
                " # of point:           1\n",
                " Display switch:1  247\n",
                "     #    X       Y       Z      Mark    Label \n",
                "      1    1.00    2.00    3.00   0\n\n  END\n"
            )
        );
    }
}
