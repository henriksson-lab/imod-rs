//! Translation of `IMOD/libcfshr/piecefuncs.c`.

use super::b3dutil::ImodFile;
use super::readlinevalues::{RLFV_SEPARATE_LINES, ReadValueArray, read_lines_for_values};

/// Matches `checkPieceList` (`IMOD/libcfshr/piecefuncs.c:38`).
pub fn check_piece_list(
    piece_list: &[i32],
    stride: usize,
    number_piece_list: usize,
    reduction_factor: i32,
    frame_size: i32,
    minimum_piece: &mut i32,
    number_pieces: &mut i32,
    overlap: &mut i32,
) -> i32 {
    let mut minimum_difference = 100_000;
    *minimum_piece = 100_000;
    let mut maximum_piece = -100_000;
    // `coords` is allocated only for more than one piece; otherwise the old
    // pairwise way runs, which starts `minpiece` at 100000 (`piecefuncs.c:45-78`).
    if number_piece_list > 1 {
        let mut coordinates = (0..number_piece_list)
            .map(|index| piece_list[index * stride])
            .collect::<Vec<_>>();
        coordinates.sort_unstable();
        *minimum_piece = reduction_factor * coordinates[0];
        maximum_piece = reduction_factor * coordinates[coordinates.len() - 1];
        for index in 1..coordinates.len() {
            let difference = coordinates[index] - coordinates[index - 1];
            if difference != 0 {
                minimum_difference = minimum_difference.min(reduction_factor * difference);
            }
        }
    } else {
        for i in 0..number_piece_list {
            let ipc = reduction_factor * piece_list[i * stride];
            *minimum_piece = (*minimum_piece).min(ipc);
            maximum_piece = maximum_piece.max(ipc);
            for j in 0..number_piece_list {
                let jpc = reduction_factor * piece_list[j * stride];
                let difference = (ipc - jpc).abs();
                if difference != 0 {
                    minimum_difference = minimum_difference.min(difference);
                }
            }
        }
    }
    if minimum_difference == 100_000 {
        *number_pieces = 1;
        *overlap = 0;
        return 0;
    }
    if (minimum_difference as f64) > 1.1 * frame_size as f64 {
        let mut add_back = 1;
        let mut total_overlap = frame_size - minimum_difference;
        while total_overlap < 0 {
            total_overlap += frame_size;
            add_back += 1;
        }
        *overlap = total_overlap / add_back;
        minimum_difference = frame_size - *overlap;
    }
    *number_pieces = 0;
    for index in 0..number_piece_list {
        let difference = reduction_factor * piece_list[index * stride] - *minimum_piece;
        if difference % minimum_difference != 0 {
            *number_pieces = -1;
            return index as i32 + 1;
        }
        *number_pieces = (*number_pieces).max(difference / minimum_difference + 1);
    }
    *overlap = frame_size - minimum_difference;
    let _ = maximum_piece;
    0
}

/// Matches Fortran wrapper `checklist` (`IMOD/libcfshr/piecefuncs.c:122`).
pub fn checklist(
    piece_list: &[i32],
    reduction_factor: i32,
    frame_size: i32,
    minimum_piece: &mut i32,
    number_pieces: &mut i32,
    overlap: &mut i32,
) -> i32 {
    let retval = check_piece_list(
        piece_list,
        1,
        piece_list.len(),
        reduction_factor,
        frame_size,
        minimum_piece,
        number_pieces,
        overlap,
    );
    if retval != 0 {
        use std::io::Write;
        let _ = write!(
            ImodFile::Stdout,
            "Piece coordinates not regularly spaced apart or corrupted at line {retval}\n"
        );
        let _ = ImodFile::Stdout.flush();
    }
    retval
}

/// Matches `adjustPieceOverlap` (`IMOD/libcfshr/piecefuncs.c:140`).
pub fn adjust_piece_overlap(
    piece_list: &mut [i32],
    stride: usize,
    number_piece_list: usize,
    frame_size: i32,
    minimum_piece: i32,
    overlap: i32,
    new_overlap: i32,
) {
    for index in 0..number_piece_list {
        let piece = &mut piece_list[index * stride];
        let offset = (*piece - minimum_piece) / (frame_size - overlap);
        *piece = offset * (frame_size - new_overlap) + minimum_piece;
    }
}

/// Matches Fortran wrapper `adjustpieceoverlap` (`IMOD/libcfshr/piecefuncs.c:151`).
pub fn adjustpieceoverlap(
    piece_list: &mut [i32],
    frame_size: i32,
    minimum_piece: i32,
    overlap: i32,
    new_overlap: i32,
) {
    adjust_piece_overlap(
        piece_list,
        1,
        piece_list.len(),
        frame_size,
        minimum_piece,
        overlap,
        new_overlap,
    )
}

/// Matches `fillListOfPieceZ` (`IMOD/libcfshr/piecefuncs.c:162`).
pub fn fill_list_of_piece_z(piece_z: &[i32], list_z: &mut [i32], number_list_z: &mut usize) {
    *number_list_z = 0;
    for value in piece_z {
        let mut found = 0usize;
        while found < *number_list_z && list_z[found] < *value {
            found += 1;
        }
        if found == *number_list_z || list_z[found] != *value {
            for index in (found..*number_list_z).rev() {
                list_z[index + 1] = list_z[index];
            }
            list_z[found] = *value;
            *number_list_z += 1;
        }
    }
}

/// Matches Fortran wrapper `fill_listz` (`IMOD/libcfshr/piecefuncs.c:185`).
pub fn fill_listz(piece_z: &[i32], list_z: &mut [i32], number_list_z: &mut usize) {
    fill_list_of_piece_z(piece_z, list_z, number_list_z)
}

/// Matches `readPieceList` (`IMOD/libcfshr/piecefuncs.c:198`).
pub fn read_piece_list(
    piece_file: Option<&str>,
    piece_x: &mut [i32],
    piece_y: &mut [i32],
    piece_z: &mut [i32],
    number_piece_list: &mut i32,
    maximum_pieces: usize,
) -> i32 {
    *number_piece_list = 0;
    let Some(name) = piece_file.filter(|name| !name.is_empty()) else {
        return 0;
    };
    let Some(mut fp) = ImodFile::open(name, "r") else {
        return 1;
    };
    let mut line = [0u8; 160];
    let mut count = 0;
    let error = read_lines_for_values(
        &mut fp,
        &mut count,
        maximum_pieces as i32,
        &mut line,
        160,
        RLFV_SEPARATE_LINES,
        "iii",
        &mut [
            ReadValueArray::Integers(piece_x),
            ReadValueArray::Integers(piece_y),
            ReadValueArray::Integers(piece_z),
        ],
    );
    *number_piece_list = count;
    error
}
