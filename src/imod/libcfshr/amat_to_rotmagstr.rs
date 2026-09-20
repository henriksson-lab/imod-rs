//! Translation of `IMOD/libcfshr/amat_to_rotmagstr.c`.

// `amatToRotmagstr` and `rotmagstrToAmat` declare this local double.
const ATOR: f64 = 0.0174532925;
// The RotMag functions use b3dutil.h's `RADIANS_PER_DEGREE` macro.
const RADIANS_PER_DEGREE: f64 = 0.01745329252;

/// `rotmagstrToAmat`: store the matrix in Fortran `(2, *)` order.
pub fn rotmagstr_to_amat(theta: f32, smag: f32, str_: f32, phi: f32, amat: &mut [f32; 4]) {
    let costh = (ATOR * theta as f64).cos() as f32;
    let sinth = (ATOR * theta as f64).sin() as f32;
    let cosphi = (ATOR * phi as f64).cos() as f32;
    let sinphi = (ATOR * phi as f64).sin() as f32;
    let cosphisq = cosphi * cosphi;
    let sinphisq = sinphi * sinphi;
    let f1 = smag * (str_ * cosphisq + sinphisq);
    let f2 = (smag as f64 * (str_ as f64 - 1.0) * cosphi as f64 * sinphi as f64) as f32;
    let f3 = smag * (str_ * sinphisq + cosphisq);
    amat[0] = f1 * costh - f2 * sinth;
    amat[2] = f2 * costh - f3 * sinth;
    amat[1] = f1 * sinth + f2 * costh;
    amat[3] = f2 * sinth + f3 * costh;
}

/// `amatToRotMag`.
pub fn amat_to_rotmag(a11: f32, a12: f32, a21: f32, a22: f32) -> (f32, f32, f32, f32) {
    let xmag = ((a11 * a11 + a21 * a21) as f64).sqrt() as f32;
    let xtheta = ((a21 as f64).atan2(a11 as f64) / RADIANS_PER_DEGREE) as f32;
    let ymag = ((a12 * a12 + a22 * a22) as f64).sqrt() as f32;
    let ytheta = ((-(a12 as f64)).atan2(a22 as f64) / RADIANS_PER_DEGREE) as f32;
    let smag = (0.5_f64 * (xmag + ymag) as f64) as f32;
    let ydmag = ymag - xmag;
    let mut ydtheta = ytheta - xtheta;
    if ydtheta > 180.0 {
        ydtheta -= 360.0;
    }
    if ydtheta <= -180.0 {
        ydtheta += 360.0;
    }
    let mut theta = (xtheta as f64 + 0.5_f64 * ydtheta as f64) as f32;
    if theta > 180.0 {
        theta -= 360.0;
    }
    if theta <= -180.0 {
        theta += 360.0;
    }
    (theta, ydtheta, smag, ydmag)
}

/// `rotMagToAmat`, returned in Fortran `(2, *)` order.
pub fn rotmag_to_amat(theta: f32, ydtheta: f32, smag: f32, ydmag: f32) -> [f32; 4] {
    let xmag = (smag as f64 - ydmag as f64 / 2.0) as f32;
    let ymag = (smag as f64 + ydmag as f64 / 2.0) as f32;
    let xtheta = (theta as f64 - ydtheta as f64 / 2.0) as f32;
    let ytheta = (theta as f64 + ydtheta as f64 / 2.0) as f32;
    [
        (xmag as f64 * (xtheta as f64 * RADIANS_PER_DEGREE).cos()) as f32,
        (xmag as f64 * (xtheta as f64 * RADIANS_PER_DEGREE).sin()) as f32,
        (-(ymag as f64) * (ytheta as f64 * RADIANS_PER_DEGREE).sin()) as f32,
        (ymag as f64 * (ytheta as f64 * RADIANS_PER_DEGREE).cos()) as f32,
    ]
}

/// `amatToRotmagstr`.
pub fn amat_to_rotmagstr(a11: f32, mut a12: f32, a21: f32, mut a22: f32) -> (f32, f32, f32, f32) {
    let mut dtheta =
        (((a22 as f64).atan2(a12 as f64) - (a21 as f64).atan2(a11 as f64)) / ATOR) as f32;
    if dtheta > 180.0 {
        dtheta -= 360.0;
    }
    if dtheta <= -180.0 {
        dtheta += 360.0;
    }
    if dtheta < 0.0 {
        a12 = -a12;
        a22 = -a22;
    }

    let mut theta = 0.0_f32;
    if a21 != a12 || a22 != -a11 {
        theta = (((a21 - a12) as f64).atan2((a22 + a11) as f64) / ATOR) as f32;
    }
    let costh = (ATOR * theta as f64).cos() as f32;
    let sinth = (ATOR * theta as f64).sin() as f32;
    let f1 = a11 * costh + a21 * sinth;
    let f2 = a21 * costh - a11 * sinth;
    let f3 = a22 * costh - a12 * sinth;

    let mut cosphisq: f64;
    if f2 < 1.0e-10 && f2 > -1.0e-10 {
        cosphisq = 1.0;
    } else {
        let afac = (f3 - f1) as f64 * (f3 - f1) as f64;
        let bfac = 4.0_f64 * f2 as f64 * f2 as f64;
        cosphisq = 0.5 * (1.0 + (1.0 - bfac / (bfac + afac)).sqrt());
        let sinphisq = 1.0 - cosphisq;
        let mut fnum = (f1 as f64 * cosphisq - f3 as f64 * sinphisq) as f32;
        if fnum < 0.0 {
            fnum = -fnum;
        }
        let mut fden = (f3 as f64 * cosphisq - f1 as f64 * sinphisq) as f32;
        if fden < 0.0 {
            fden = -fden;
        }
        if (f2 > 0.0 && fnum < fden) || (f2 < 0.0 && fnum > fden) {
            cosphisq = 1.0 - cosphisq;
        }
    }
    let mut phi = (cosphisq.sqrt().acos() / ATOR) as f32;
    let sinphisq = 1.0 - cosphisq;

    let str_ = if cosphisq - 0.5 > 0.25 || cosphisq - 0.5 < -0.25 {
        ((f1 as f64 * cosphisq - f3 as f64 * sinphisq)
            / (f3 as f64 * cosphisq - f1 as f64 * sinphisq)) as f32
    } else {
        let factmp = ((f1 + f3) as f64 * (cosphisq * sinphisq).sqrt()) as f32;
        (factmp + f2) / (factmp - f2)
    };

    let dentmp = str_ * cosphisq as f32 + sinphisq as f32;
    let mut smag = if dentmp > 1.0e-5 || dentmp < -1.0e-5 {
        f1 / dentmp
    } else {
        (1.0_f64 / ((str_ as f64 - 1.0) * (cosphisq * sinphisq).sqrt())) as f32
    };

    let mut str_ = str_;
    let mut f1 = smag - 1.0;
    let mut f2 = str_ * smag - 1.0;
    if f1 < 0.0 {
        f1 = -f1;
    }
    if f2 < 0.0 {
        f2 = -f2;
    }
    if f1 > f2 {
        smag = smag * str_;
        str_ = 1.0 / str_;
        phi -= 90.0;
    }

    if dtheta < 0.0 {
        str_ = -str_;
        phi = -phi;
        theta = theta + 180.0 - 2.0 * phi;
        if theta > 180.0 {
            theta -= 360.0;
        }
    }
    (theta, smag, str_, phi)
}
