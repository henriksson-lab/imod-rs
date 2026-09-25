//! Translation of `IMOD/flib/subrs/statsubs/polyterm.f`.

/// Original `polyTermReal` (`polyterm.f:16`).
///
/// Computes polynomial terms from [x] and [y] of order [norder], and
/// places then in array [vect].  These terms can then be used to fit a
/// polynomial in to measurements as a function of X and Y.  The number of
/// terms returned is {norder * (norder + 3) / 2}.  The first set of terms
/// is [x] and [y].  Each next set is the previous set multipled by [x],
/// plus the last term of the previous set multiplied by [y].
///
/// Indices are the source's 1-based ones, shifted by one at each access.
pub fn poly_term_real(x: f32, y: f32, norder: i32, vect: &mut [f32]) {
    vect[0] = x;
    vect[1] = y;
    let mut istr: i32 = 1;
    let mut iend: i32 = 2;
    for iorder in 2..=norder {
        for i in istr..=iend {
            vect[(i + iorder - 1) as usize] = vect[(i - 1) as usize] * x;
        }
        istr += iorder;
        vect[(iend + iorder) as usize] = vect[(iend - 1) as usize] * y;
        iend = iend + iorder + 1;
    }
}

/// Original `polyTerm` (`polyterm.f:40`).
///
/// Computes polynomial terms from integer arguments [ix] and [iy] of
/// order [norder], and places then in array [vect], the same as
/// polyTermReal does.
pub fn poly_term(ix: i32, iy: i32, norder: i32, vect: &mut [f32]) {
    poly_term_real(ix as f32, iy as f32, norder, vect);
}
