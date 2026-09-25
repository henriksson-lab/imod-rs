//! Translation of `IMOD/flib/subrs/piecesubs/read_piece_list.f`.

use crate::imod::flib::subrs::hvem::dopen::dopen;
use crate::imod::flib::subrs::hvem::frefor::{ListItem, ListReadError, list_read};
use std::io::BufReader;

/// Original `read_piece_list` (`read_piece_list.f:6`).
///
/// READ_PIECE_LIST will open the file whose name is in FILPCL, and
/// read the piece coordinates into IXPCLIST, IYPCLIST, IZPCLIST until
/// end of file.  NPCLIST has the number of coordinates, or 0 if the
/// file name is blank.
///
/// With no limit the source stores past the end of arrays that are too
/// small; the slices here stop that with a bounds panic instead.
pub fn read_piece_list(
    filpcl: &str,
    ixpclist: &mut [i32],
    iypclist: &mut [i32],
    izpclist: &mut [i32],
    npclist: &mut i32,
) {
    read_piece_list2(filpcl, ixpclist, iypclist, izpclist, npclist, 0);
}

/// Original `read_piece_list2` (`read_piece_list.f:17`).
///
/// A safe version that will only fill the array up to LIMPCL and issues a warning
/// if pieces don't fit
///
/// Unit 2 is the file `dopen` connects; a return without the `close(2)` of
/// label 20 still closes it here, when the reader is dropped.
pub fn read_piece_list2(
    filpcl: &str,
    ixpclist: &mut [i32],
    iypclist: &mut [i32],
    izpclist: &mut [i32],
    npclist: &mut i32,
    limpcl: i32,
) {
    let (mut ixt, mut iyt, mut izt) = (0i32, 0i32, 0i32);
    //
    *npclist = 0;
    if filpcl.bytes().all(|b| b == b' ') {
        return;
    }
    let mut unit = BufReader::new(dopen(2, filpcl, "ro", "f"));
    loop {
        // 10
        let i = *npclist + 1;
        // read(2,*,end=20,err=30)ixt, iyt, izt
        match list_read(
            &mut unit,
            &mut [
                ListItem::Integer(&mut ixt),
                ListItem::Integer(&mut iyt),
                ListItem::Integer(&mut izt),
            ],
        ) {
            Ok(()) => {}
            // 20 close(2)
            Err(ListReadError::End) => return,
            Err(ListReadError::Error) => {
                // 30 write(*,'(/,a)')'ERROR: read_piece_list - reading piece list file'
                println!("\nERROR: read_piece_list - reading piece list file");
                crate::imod::libcfshr::b3dutil::exit(1);
            }
        }
        if limpcl > 0 && i > limpcl {
            println!("\nWARNING: read_piece_list - Too many piece coordinates for arrays");
            return;
        }
        ixpclist[(i - 1) as usize] = ixt;
        iypclist[(i - 1) as usize] = iyt;
        izpclist[(i - 1) as usize] = izt;
        *npclist = i;
    }
}
