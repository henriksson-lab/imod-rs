//! Translation of point operations in `IMOD/libimod/ipoint.c`.

use crate::imod::libimod::imodel::{ICONT_WILD, Icont, Ipoint};
use crate::imod::libimod::istore::istore_shift_index;

/// C `imodPointAdd` (`ipoint.c:44`).
pub fn imod_point_add(cont: &mut Icont, point: Option<Ipoint>, mut index: i32) -> i32 {
    if index > cont.pts.len() as i32 {
        index = cont.pts.len() as i32;
    }
    if index < 0 {
        return 0;
    }
    let Some(point) = point else {
        return cont.pts.len() as i32;
    };
    let start_z = cont.pts.first().map_or(point.z, |first| first.z);
    cont.pts.insert(index as usize, point);
    if !cont.sizes.is_empty() {
        cont.sizes.insert(index as usize, -1.0);
    }
    istore_shift_index(&mut cont.store, index, -1, 1);
    if (start_z + 0.5).floor() as i32 != (point.z + 0.5).floor() as i32 {
        cont.flags |= ICONT_WILD;
    }
    cont.pts.len() as i32
}
