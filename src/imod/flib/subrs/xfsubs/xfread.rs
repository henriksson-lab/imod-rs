//! Translation of `IMOD/flib/subrs/xfsubs/xfread.f`.

use crate::imod::flib::subrs::hvem::frefor::{ListItem, ListReadError, list_read};
use std::io::BufRead;

/// Original alternate returns from `xfread` (`xfread.f:1`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum XfReadError {
    End,
    Error,
}

/// Original `xfread` (`xfread.f:1`).
///
/// `read(iunit,*,end=10,err=20)((f(i,j),j=1,2),i=1,2),f(1,3),f(2,3)` is a
/// list-directed read, so it goes through [`list_read`]: blank records are
/// skipped, the six values may span records, anything after the sixth on the
/// last record is discarded, and running out of records part-way is `END=`.
pub fn xfread<R: BufRead>(iunit: &mut R, f: &mut [f32; 6]) -> Result<(), XfReadError> {
    let [a11, a21, a12, a22, dx, dy] = f;
    list_read(
        iunit,
        &mut [
            ListItem::Real(a11),
            ListItem::Real(a12),
            ListItem::Real(a21),
            ListItem::Real(a22),
            ListItem::Real(dx),
            ListItem::Real(dy),
        ],
    )
    .map_err(|err| match err {
        ListReadError::End => XfReadError::End,
        ListReadError::Error => XfReadError::Error,
    })
}
