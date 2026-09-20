//! Translation of `IMOD/libcfshr/insidecontour.c`.

/// Original `InsideContour` (`insidecontour.c:31`).
pub fn inside_contour(pt_x: &[f32], pt_y: &[f32], np: i32, x: f32, y: f32) -> i32 {
    let mut nrcross = 0;
    let mut nlcross = 0;
    let mut yp: f32;
    let mut j: i32;

    if np <= 0 {
        return 0;
    }
    yp = pt_y[(np - 1) as usize];
    j = 0;
    while j < np {
        if yp < y {
            while j < np && pt_y[j as usize] < y {
                j += 1;
            }
        } else if yp > y {
            while j < np && pt_y[j as usize] > y {
                j += 1;
            }
        }

        if j < np {
            let mut jl = j - 1;
            if jl < 0 {
                jl = np - 1;
            }
            let xp = pt_x[jl as usize];
            yp = pt_y[jl as usize];
            let xc = pt_x[j as usize];
            let yc = pt_y[j as usize];

            if x == xc && y == yc {
                return 1;
            }

            let rstrad = (if yc > y { 1 } else { 0 }) != (if yp > y { 1 } else { 0 });
            let lstrad = (if yc < y { 1 } else { 0 }) != (if yp < y { 1 } else { 0 });
            if lstrad || rstrad {
                let xcross = xp + (y - yp) * (xc - xp) / (yc - yp);
                if rstrad && xcross > x {
                    nrcross += 1;
                }
                if lstrad && xcross < x {
                    nlcross += 1;
                }
            }
            yp = yc;
        }
        j += 1;
    }

    if nrcross % 2 != nlcross % 2 {
        return 1;
    }
    if nrcross % 2 > 0 { 1 } else { 0 }
}

/// Original Fortran wrapper `insidecontour` (`insidecontour.c:102`).
pub fn insidecontour(pt_x: &[f32], pt_y: &[f32], np: &i32, x: &f32, y: &f32) -> i32 {
    inside_contour(pt_x, pt_y, *np, *x, *y)
}

#[cfg(test)]
mod tests {
    use super::{inside_contour, insidecontour};

    #[test]
    fn source_polygon_and_wrapper_cases() {
        let x = [0.0, 4.0, 4.0, 0.0];
        let y = [0.0, 0.0, 4.0, 4.0];
        assert_eq!(inside_contour(&x, &y, 4, 2.0, 2.0), 1);
        assert_eq!(inside_contour(&x, &y, 4, 5.0, 2.0), 0);
        assert_eq!(inside_contour(&x, &y, 4, 0.0, 0.0), 1);
        assert_eq!(inside_contour(&x, &y, 4, 2.0, 0.0), 1);
        assert_eq!(inside_contour(&x, &y, 0, 2.0, 2.0), 0);
        assert_eq!(insidecontour(&x, &y, &4, &2.0, &2.0), 1);
    }
}
