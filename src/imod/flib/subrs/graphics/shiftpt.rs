//! Translation of `IMOD/flib/subrs/graphics/shiftpt.f`.

use std::cell::RefCell;

/// `parameter (limpts=500)`
const LIMPTS: usize = 500;

thread_local! {
    /// `save nlist,xst,yst`
    static LIST: RefCell<(usize, [f32; LIMPTS], [f32; LIMPTS])> =
        const { RefCell::new((0, [0.; LIMPTS], [0.; LIMPTS])) };
}

/// Original `shiftpt` (`shiftpt.f:9`): moves the point `x, y` so that it is
/// at least `sep` from the points plotted before, within `xlo..xhi`,
/// `ylo..yhi`; a separation of 0 starts a new list.
pub fn shiftpt(x: &mut f32, y: &mut f32, sep: f32, xlo: f32, xhi: f32, ylo: f32, yhi: f32) {
    LIST.with_borrow_mut(|(nlist, xst, yst)| {
        if sep == 0. {
            *nlist = 0;
            return;
        }
        let sepsq = sep * sep;
        let mut xx: f32 = *x;
        let mut yy: f32 = *y;
        'found: for irad in 0..=4 {
            let nang = 1.max(4 * irad);
            'angle: for iang in 1..=nang {
                let theta = (iang - 1) as f32 * 2. * 3.14159 / nang as f32;
                xx = *x + 0.5 * irad as f32 * sep * theta.cos();
                yy = *y + 0.5 * irad as f32 * sep * theta.sin();
                if xx < xlo || xx > xhi || yy < ylo || yy > yhi {
                    continue 'angle;
                }
                for list in 0..*nlist {
                    let dx = (xst[list] - xx).abs();
                    if dx < sep {
                        let dy = (yst[list] - yy).abs();
                        if dy < sep && dx * dx + dy * dy < sepsq {
                            continue 'angle;
                        }
                    }
                }
                break 'found;
            }
        }
        if *nlist < LIMPTS {
            *nlist += 1;
            xst[*nlist - 1] = xx;
            yst[*nlist - 1] = yy;
        }
        *x = xx;
        *y = yy;
    });
}
