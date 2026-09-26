//! Translation of `IMOD/flib/subrs/hvem/objtocont.f`.

/// Original `objtocont` (`objtocont.f:4`).
///
/// OBJTOCONT converts an internal or WIMP-style object number to an
/// IMOD object and contour number.  `obj_color(2,*)` is the column-major
/// `fortmodel` array, so `obj_color(2,i)` is `obj_color[i - 1][1]`.
pub fn objtocont(iobj: i32, obj_color: &[[i32; 2]], imodobj: &mut i32, imodcont: &mut i32) {
    let icolor = obj_color[(iobj - 1) as usize][1];
    *imodobj = 256 - icolor;
    *imodcont = 0;
    for i in 1..=iobj {
        if icolor == obj_color[(i - 1) as usize][1] {
            *imodcont += 1;
        }
    }
}
