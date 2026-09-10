//! Translation of `IMOD/flib/subrs/xfsubs/xfrdall.f`.

use crate::imod::flib::subrs::xfsubs::xfread::{XfReadError, xfread};
use std::io::BufRead;

/// Original `xfrdall` (`xfrdall.f:2`).
pub fn xfrdall<R: BufRead>(nunit: &mut R, flist: &mut Vec<[f32; 6]>) -> Result<(), ()> {
    flist.clear();
    loop {
        let mut transform = [0.0; 6];
        match xfread(nunit, &mut transform) {
            Ok(()) => flist.push(transform),
            Err(XfReadError::End) => return Ok(()),
            Err(XfReadError::Error) => return Err(()),
        }
    }
}

/// Original `xfrdall2` (`xfrdall.f:16`).
pub fn xfrdall2<R: BufRead>(nunit: &mut R, flist: &mut Vec<[f32; 6]>, limlist: i32) -> i32 {
    flist.clear();
    loop {
        let mut temporary = [0.0; 6];
        match xfread(nunit, &mut temporary) {
            Ok(()) => {
                if flist.len() as i32 + 1 > limlist {
                    return 1;
                }
                flist.push(temporary);
            }
            Err(XfReadError::End) => return 0,
            Err(XfReadError::Error) => return 2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{xfrdall, xfrdall2};
    use crate::imod::flib::subrs::xfsubs::xfread::XfReadError;
    use std::io::Cursor;

    #[test]
    fn preserves_list_directed_order_eof_error_and_limit() {
        let mut input = Cursor::new(b"1 0 0 1 2 3\n2 0 0 2 4 6\n");
        let mut list = Vec::new();
        assert_eq!(xfrdall2(&mut input, &mut list, 1), 1);
        assert_eq!(list, vec![[1., 0., 0., 1., 2., 3.]]);
        let mut input = Cursor::new(b"1 2 3\n");
        assert_eq!(xfrdall(&mut input, &mut list), Err(()));
        assert_eq!(XfReadError::End, XfReadError::End);
    }
}
