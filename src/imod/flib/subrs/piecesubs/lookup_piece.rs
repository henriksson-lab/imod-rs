//! Translation of `IMOD/flib/subrs/piecesubs/lookup_piece.f`.

/// Original `lookup_piece` (`lookup_piece.f:8`).
///
/// LOOKUP_PIECE takes a list of NPCLIST piece coordinates in
/// I[XYZ]PCLIST, the piece dimensions NX and NY, and index coordinates
/// in the montaged image IND[XYZ], and finds the piece that those
/// coordinates are in (IPCZ) and the coordinates IPCX, IPCY of the point
/// in that piece
pub fn lookup_piece(
    ixpclist: &[i32],
    iypclist: &[i32],
    izpclist: &[i32],
    npclist: i32,
    nx: i32,
    ny: i32,
    indx: i32,
    indy: i32,
    indz: i32,
    ipcx: &mut i32,
    ipcy: &mut i32,
    ipcz: &mut i32,
) {
    *ipcx = indx;
    *ipcy = indy;
    *ipcz = indz;
    if npclist == 0 {
        return;
    }
    *ipcx = -1;
    *ipcy = -1;
    *ipcz = -1;
    for ipc in 1..=npclist {
        let k = (ipc - 1) as usize;
        if indz == izpclist[k]
            && indx >= ixpclist[k]
            && indx < ixpclist[k] + nx
            && indy >= iypclist[k]
            && indy < iypclist[k] + ny
        {
            *ipcz = ipc - 1;
            *ipcx = indx - ixpclist[k];
            *ipcy = indy - iypclist[k];
            return;
        }
    }
}
