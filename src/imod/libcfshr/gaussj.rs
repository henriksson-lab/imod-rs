//! Translation of `IMOD/libcfshr/gaussj.c`.
#![allow(dead_code)]

/// Original `gaussj` (`gaussj.c:40`).
pub unsafe fn gaussj(
    matrix: *mut f32,
    dimension: i32,
    matrix_pitch: i32,
    right_hand: *mut f32,
    vectors: i32,
    right_hand_pitch: i32,
) -> i32 {
    unsafe {
        let mut determinant = 0.0;
        gaussj_det(
            matrix,
            dimension,
            matrix_pitch,
            right_hand,
            vectors,
            right_hand_pitch,
            &mut determinant,
        )
    }
}

/// Original `gaussjDet` (`gaussj.c:49`).
pub unsafe fn gaussj_det(
    matrix: *mut f32,
    dimension: i32,
    matrix_pitch: i32,
    right_hand: *mut f32,
    vectors: i32,
    right_hand_pitch: i32,
    determinant: *mut f32,
) -> i32 {
    unsafe {
        let mut index = [[0_i16; 2]; 2000];
        let mut pivot = [0_f32; 2000];
        let mut pivot_count = [0_i16; 2000];
        *determinant = 1.0;
        if dimension > 2000 {
            return -1;
        }
        for column in 0..dimension {
            pivot_count[column as usize] = 0;
        }
        for row_iteration in 0..dimension {
            let mut maximum = 0.0_f32;
            let mut pivot_row = 0;
            let mut pivot_column = 0;
            for row in 0..dimension {
                if pivot_count[row as usize] != 1 {
                    for column in 0..dimension {
                        if pivot_count[column as usize] == 0 {
                            let mut absolute = *matrix.add((row * matrix_pitch + column) as usize);
                            if absolute < 0.0 {
                                absolute = -absolute;
                            }
                            if maximum < absolute {
                                pivot_row = row;
                                pivot_column = column;
                                maximum = absolute;
                            }
                        } else if pivot_count[column as usize] > 1 {
                            return 1;
                        }
                    }
                }
            }
            pivot_count[pivot_column as usize] += 1;
            if pivot_row != pivot_column {
                *determinant = -*determinant;
                for column in 0..dimension {
                    let temporary = *matrix.add((pivot_row * matrix_pitch + column) as usize);
                    *matrix.add((pivot_row * matrix_pitch + column) as usize) =
                        *matrix.add((pivot_column * matrix_pitch + column) as usize);
                    *matrix.add((pivot_column * matrix_pitch + column) as usize) = temporary;
                }
                for column in 0..vectors {
                    let temporary =
                        *right_hand.add((pivot_row * right_hand_pitch + column) as usize);
                    *right_hand.add((pivot_row * right_hand_pitch + column) as usize) =
                        *right_hand.add((pivot_column * right_hand_pitch + column) as usize);
                    *right_hand.add((pivot_column * right_hand_pitch + column) as usize) =
                        temporary;
                }
            }
            index[row_iteration as usize][0] = pivot_row as i16;
            index[row_iteration as usize][1] = pivot_column as i16;
            let pivot_multiple = *matrix.add((pivot_column * matrix_pitch + pivot_column) as usize);
            pivot[row_iteration as usize] = pivot_multiple;
            *determinant *= pivot_multiple;
            *matrix.add((pivot_column * matrix_pitch + pivot_column) as usize) = 1.0;
            for column in 0..dimension {
                *matrix.add((pivot_column * matrix_pitch + column) as usize) /= pivot_multiple;
            }
            for column in 0..vectors {
                *right_hand.add((pivot_column * right_hand_pitch + column) as usize) /=
                    pivot_multiple;
            }
            for other_row in 0..dimension {
                let temporary = *matrix.add((other_row * matrix_pitch + pivot_column) as usize);
                if temporary != 0.0 && other_row != pivot_column {
                    *matrix.add((other_row * matrix_pitch + pivot_column) as usize) = 0.0;
                    for column in 0..dimension {
                        *matrix.add((other_row * matrix_pitch + column) as usize) -= *matrix
                            .add((pivot_column * matrix_pitch + column) as usize)
                            * temporary;
                    }
                    for column in 0..vectors {
                        *right_hand.add((other_row * right_hand_pitch + column) as usize) -=
                            *right_hand.add((pivot_column * right_hand_pitch + column) as usize)
                                * temporary;
                    }
                }
            }
        }
        for row_iteration in 0..dimension {
            let reverse = dimension - 1 - row_iteration;
            if index[reverse as usize][0] != index[reverse as usize][1] {
                let pivot_row = index[reverse as usize][0] as i32;
                let pivot_column = index[reverse as usize][1] as i32;
                for row in 0..dimension {
                    let temporary = *matrix.add((row * matrix_pitch + pivot_row) as usize);
                    *matrix.add((row * matrix_pitch + pivot_row) as usize) =
                        *matrix.add((row * matrix_pitch + pivot_column) as usize);
                    *matrix.add((row * matrix_pitch + pivot_column) as usize) = temporary;
                }
            }
        }
        0
    }
}

/// Original Fortran wrapper `gaussjfw` (`gaussj.c:137`).
pub unsafe fn gaussjfw(
    matrix: *mut f32,
    dimension: *mut i32,
    matrix_pitch: *mut i32,
    right_hand: *mut f32,
    vectors: *mut i32,
    right_hand_pitch: *mut i32,
) -> i32 {
    unsafe {
        gaussj(
            matrix,
            *dimension,
            *matrix_pitch,
            right_hand,
            *vectors,
            *right_hand_pitch,
        )
    }
}

/// Original Fortran wrapper `gaussjdet` (`gaussj.c:142`).
pub unsafe fn gaussjdet(
    matrix: *mut f32,
    dimension: *mut i32,
    matrix_pitch: *mut i32,
    right_hand: *mut f32,
    vectors: *mut i32,
    right_hand_pitch: *mut i32,
    determinant: *mut f32,
) -> i32 {
    unsafe {
        gaussj_det(
            matrix,
            *dimension,
            *matrix_pitch,
            right_hand,
            *vectors,
            *right_hand_pitch,
            determinant,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn solves_row_major_system_and_preserves_source_determinant() {
        unsafe {
            let mut matrix = [2.0_f32, 1.0, 5.0, 7.0];
            let mut right_hand = [11.0_f32, 13.0];
            let mut determinant = 0.0;
            assert_eq!(
                gaussj_det(
                    matrix.as_mut_ptr(),
                    2,
                    2,
                    right_hand.as_mut_ptr(),
                    1,
                    1,
                    &mut determinant
                ),
                0
            );
            for (actual, expected) in
                matrix
                    .iter()
                    .zip([7.0 / 9.0, -1.0 / 9.0, -5.0 / 9.0, 2.0 / 9.0])
            {
                assert!((actual - expected).abs() < 1.0e-6);
            }
            assert!((right_hand[0] - 64.0 / 9.0).abs() < 1.0e-5);
            assert!((right_hand[1] + 29.0 / 9.0).abs() < 1.0e-5);
            assert_eq!(determinant, 9.0);
            assert_eq!(
                gaussj(core::ptr::null_mut(), 2001, 0, core::ptr::null_mut(), 0, 0),
                -1
            );
        }
    }
}
