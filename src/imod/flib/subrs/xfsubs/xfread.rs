//! Translation of `IMOD/flib/subrs/xfsubs/xfread.f`.

use std::io::BufRead;

/// Original alternate returns from `xfread` (`xfread.f:1`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum XfReadError {
    End,
    Error,
}

/// Original `xfread` (`xfread.f:1`).
pub fn xfread<R: BufRead>(iunit: &mut R, f: &mut [f32; 6]) -> Result<(), XfReadError> {
    let mut line = String::new();
    if iunit.read_line(&mut line).map_err(|_| XfReadError::Error)? == 0 {
        return Err(XfReadError::End);
    }
    let mut values = [0.0_f32; 6];
    let mut count = 0;
    for word in line.split_whitespace() {
        if count == 6 {
            break;
        }
        values[count] = word
            .replace(['d', 'D'], "E")
            .parse()
            .map_err(|_| XfReadError::Error)?;
        count += 1;
    }
    if count != 6 {
        return Err(XfReadError::Error);
    }
    // `((f(i,j),j=1,2),i=1,2),f(1,3),f(2,3)` into column-major storage.
    *f = [
        values[0], values[2], values[1], values[3], values[4], values[5],
    ];
    Ok(())
}
