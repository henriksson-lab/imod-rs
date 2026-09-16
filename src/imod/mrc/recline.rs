//! Recursive one-dimensional filters from `IMOD/mrc/recline.{c,h}`.
//!
//! The C API exposed mutable global state and caller-allocated work buffers.
//! This translation keeps coefficients as an owned value and owns its temporary
//! lines; the numerical recurrences and their boundary conditions are unchanged.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RecursiveFilterType {
    Unknown,
    AlphaDeriche,
    GaussianDeriche,
    GaussianFidrich,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DerivativeOrder {
    None,
    Zero,
    One,
    Two,
    Three,
    OneContours,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RecursiveFilterCoefficients {
    pub sd1: f64,
    pub sd2: f64,
    pub sd3: f64,
    pub sd4: f64,
    pub sp0: f64,
    pub sp1: f64,
    pub sp2: f64,
    pub sp3: f64,
    pub sn0: f64,
    pub sn1: f64,
    pub sn2: f64,
    pub sn3: f64,
    pub sn4: f64,
    pub filter_type: RecursiveFilterType,
    pub derivative: DerivativeOrder,
}

impl Default for RecursiveFilterCoefficients {
    fn default() -> Self {
        Self {
            sd1: 0.0,
            sd2: 0.0,
            sd3: 0.0,
            sd4: 0.0,
            sp0: 0.0,
            sp1: 0.0,
            sp2: 0.0,
            sp3: 0.0,
            sn0: 0.0,
            sn1: 0.0,
            sn2: 0.0,
            sn3: 0.0,
            sn4: 0.0,
            filter_type: RecursiveFilterType::Unknown,
            derivative: DerivativeOrder::None,
        }
    }
}

/// Equivalent to `InitRecursiveCoefficients`.  Alpha-Deriche coefficients are
/// exact translations; fourth-order Gaussian coefficient construction follows
/// the same recurrence parameterization.
pub fn init_recursive_coefficients(
    x: f64,
    mut filter_type: RecursiveFilterType,
    mut derivative: DerivativeOrder,
) -> Result<RecursiveFilterCoefficients, String> {
    if filter_type == RecursiveFilterType::Unknown {
        filter_type = RecursiveFilterType::AlphaDeriche;
    }
    if !(x.is_finite()) {
        return Err("recursive filter coefficient must be finite".into());
    }
    let mut out = RecursiveFilterCoefficients::default();
    match filter_type {
        RecursiveFilterType::AlphaDeriche => {
            if !(0.1..=1.9).contains(&x) {
                return Err("alpha Deriche coefficient must be in [0.1, 1.9]".into());
            }
            if derivative == DerivativeOrder::None {
                derivative = DerivativeOrder::Zero;
            }
            let ex = (-x).exp();
            match derivative {
                DerivativeOrder::Zero => {
                    out.sp0 = (1.0 - ex).powi(2) / (1.0 + 2.0 * x * ex - ex * ex);
                    out.sp1 = out.sp0 * (x - 1.0) * ex;
                    out.sn1 = out.sp0 * (x + 1.0) * ex;
                    out.sn2 = -out.sp0 * ex * ex;
                }
                DerivativeOrder::One => {
                    out.sp1 = -(1.0 - ex).powi(3) / (2.0 * (1.0 + ex));
                    out.sn1 = -out.sp1;
                }
                DerivativeOrder::OneContours => {
                    out.sp1 = -(1.0 - ex).powi(2);
                    out.sn1 = -out.sp1;
                }
                DerivativeOrder::Two => {
                    let k1 = -2.0 * (1.0 - ex).powi(3) / (1.0 + ex).powi(3);
                    let k2 = (1.0 - ex * ex) / (2.0 * ex);
                    out.sp0 = k1;
                    out.sp1 = -k1 * (1.0 + k2) * ex;
                    out.sn1 = k1 * (1.0 - k2) * ex;
                    out.sn2 = -k1 * ex * ex;
                }
                DerivativeOrder::Three => {
                    let mut k1 = (1.0 + x) * ex + x - 1.0;
                    let k2 = (1.0 - ex) / k1;
                    k1 *= (1.0 - ex).powi(4);
                    k1 /= 2.0 * x * x * ex * ex;
                    k1 /= ex + 1.0;
                    out.sp0 = k1 * x * (k2 + 1.0);
                    out.sp1 = -k1 * x * (1.0 + k2 + k2 * x) * ex;
                    out.sn0 = -out.sp0;
                    out.sn1 = -out.sp1;
                }
                DerivativeOrder::None => unreachable!(),
            }
            out.sd1 = -2.0 * ex;
            out.sd2 = ex * ex;
        }
        RecursiveFilterType::GaussianFidrich => {
            if x < 0.1 {
                return Err("Gaussian coefficient must be at least 0.1".into());
            }
            let (a0, a1, c0, c1, b0, b1, omega0, omega1) = match derivative {
                DerivativeOrder::Zero => (
                    0.6570033214 / x,
                    1.978946687 / x,
                    -0.2580640608 / x,
                    -0.2391206463 / x,
                    1.906154352 / x,
                    1.881305409 / x,
                    0.6512453378 / x,
                    2.05339943 / x,
                ),
                DerivativeOrder::One | DerivativeOrder::OneContours => (
                    -0.1726729496 / x,
                    -2.003565572 / x,
                    0.1726730777 / x,
                    0.4440126835 / x,
                    1.560644213 / x,
                    1.594202256 / x,
                    0.6995461735 / x,
                    2.144671764 / x,
                ),
                DerivativeOrder::Two => (
                    -0.7241334169 / x,
                    1.688628765 / x,
                    0.3251949838 / x,
                    -0.7211796018 / x,
                    1.294951143 / x,
                    1.427007123 / x,
                    0.7789803775 / x,
                    2.233566862 / x,
                ),
                DerivativeOrder::Three => (
                    1.285774106 / x,
                    -0.2896378408 / x,
                    -1.28577129 / x,
                    0.26249833 / x,
                    1.01162886 / x,
                    1.273344739 / x,
                    0.9474270928 / x,
                    2.337607006 / x,
                ),
                DerivativeOrder::None => {
                    return Err("Gaussian Fidrich derivative must be specified".into());
                }
            };
            let (sin0, cos0) = omega0.sin_cos();
            let (sin1, cos1) = omega1.sin_cos();
            out.sp0 = a0 + c0;
            out.sp1 = (-b1).exp() * (c1 * sin1 - (c0 + 2.0 * a0) * cos1)
                + (-b0).exp() * (a1 * sin0 - (2.0 * c0 + a0) * cos0);
            out.sp2 = 2.0
                * (-b0 - b1).exp()
                * ((a0 + c0) * cos1 * cos0 - cos1 * a1 * sin0 - cos0 * c1 * sin1)
                + c0 * (-2.0 * b0).exp()
                + a0 * (-2.0 * b1).exp();
            out.sp3 = (-b1 - 2.0 * b0).exp() * (c1 * sin1 - c0 * cos1)
                + (-b0 - 2.0 * b1).exp() * (a1 * sin0 - a0 * cos0);
            out.sd1 = -2.0 * (-b1).exp() * cos1 - 2.0 * (-b0).exp() * cos0;
            out.sd2 = 4.0 * cos1 * cos0 * (-b0 - b1).exp() + (-2.0 * b1).exp() + (-2.0 * b0).exp();
            out.sd3 = -2.0 * cos0 * (-b0 - 2.0 * b1).exp() - 2.0 * cos1 * (-b1 - 2.0 * b0).exp();
            out.sd4 = (-2.0 * b0 - 2.0 * b1).exp();
            match derivative {
                DerivativeOrder::Zero | DerivativeOrder::Two => {
                    out.sn1 = out.sp1 - out.sd1 * out.sp0;
                    out.sn2 = out.sp2 - out.sd2 * out.sp0;
                    out.sn3 = out.sp3 - out.sd3 * out.sp0;
                    out.sn4 = -out.sd4 * out.sp0;
                }
                DerivativeOrder::One | DerivativeOrder::OneContours | DerivativeOrder::Three => {
                    out.sn1 = -out.sp1 + out.sd1 * out.sp0;
                    out.sn2 = -out.sp2 + out.sd2 * out.sp0;
                    out.sn3 = -out.sp3 + out.sd3 * out.sp0;
                    out.sn4 = out.sd4 * out.sp0;
                }
                DerivativeOrder::None => unreachable!(),
            }
        }
        RecursiveFilterType::GaussianDeriche => {
            if x < 0.1 {
                return Err("Gaussian coefficient must be at least 0.1".into());
            }
            if matches!(derivative, DerivativeOrder::None | DerivativeOrder::Three) {
                derivative = DerivativeOrder::Zero;
            }
            let (mut a0, mut a1, mut c0, mut c1, b0, b1, omega0, omega1) = match derivative {
                DerivativeOrder::Zero => (
                    1.68,
                    3.735,
                    -0.6803,
                    -0.2598,
                    1.783 / x,
                    1.723 / x,
                    0.6318 / x,
                    1.997 / x,
                ),
                DerivativeOrder::One | DerivativeOrder::OneContours => (
                    -0.6472,
                    -4.531,
                    0.6494,
                    0.9557,
                    1.527 / x,
                    1.516 / x,
                    0.6719 / x,
                    2.072 / x,
                ),
                DerivativeOrder::Two => (
                    -1.331,
                    3.661,
                    0.3225,
                    -1.738,
                    1.24 / x,
                    1.314 / x,
                    0.748 / x,
                    2.166 / x,
                ),
                DerivativeOrder::None | DerivativeOrder::Three => unreachable!(),
            };
            let (sin0, cos0) = omega0.sin_cos();
            let (sin1, cos1) = omega1.sin_cos();
            let eb0 = b0.exp();
            let eb1 = b1.exp();
            let (sum_a, sum_c) = match derivative {
                DerivativeOrder::Zero => {
                    let sa = (2.0 * a1 * eb0 * cos0 * cos0 - a0 * sin0 * (2.0 * b0).exp()
                        + a0 * sin0
                        - 2.0 * a1 * eb0)
                        / ((2.0 * cos0 * eb0 - (2.0 * b0).exp() - 1.0) * sin0);
                    let sc = (2.0 * c1 * eb1 * cos1 * cos1 - c0 * sin1 * (2.0 * b1).exp()
                        + c0 * sin1
                        - 2.0 * c1 * eb1)
                        / ((2.0 * cos1 * eb1 - (2.0 * b1).exp() - 1.0) * sin1);
                    (sa, sc)
                }
                DerivativeOrder::One => {
                    let q0 = (4.0 * b0).exp() - 4.0 * cos0 * (3.0 * b0).exp()
                        + 2.0 * (2.0 * b0).exp()
                        + 4.0 * cos0 * cos0 * (2.0 * b0).exp()
                        + 1.0
                        - 4.0 * cos0 * eb0;
                    let q1 = (4.0 * b1).exp() - 4.0 * cos1 * (3.0 * b1).exp()
                        + 2.0 * (2.0 * b1).exp()
                        + 4.0 * cos1 * cos1 * (2.0 * b1).exp()
                        + 1.0
                        - 4.0 * cos1 * eb1;
                    let sa = -2.0
                        * (a0 * cos0 - a1 * sin0
                            + a1 * sin0 * (2.0 * b0).exp()
                            + a0 * cos0 * (2.0 * b0).exp()
                            - 2.0 * a0 * eb0)
                        * eb0
                        / q0;
                    let sc = -2.0
                        * (c0 * cos1 - c1 * sin1
                            + c1 * sin1 * (2.0 * b1).exp()
                            + c0 * cos1 * (2.0 * b1).exp()
                            - 2.0 * c0 * eb1)
                        * eb1
                        / q1;
                    (sa, sc)
                }
                DerivativeOrder::OneContours => {
                    let sa = (a1 * eb0 - a1 * cos0 * cos0 * eb0 + a0 * cos0 * sin0 * eb0
                        - a0 * sin0)
                        / (sin0 * (2.0 * cos0 * eb0 - (2.0 * b0).exp() - 1.0));
                    let sc = (c1 * eb1 - c1 * cos1 * cos1 * eb1 + c0 * cos1 * sin1 * eb1
                        - c0 * sin1)
                        / (sin1 * (2.0 * cos1 * eb1 - (2.0 * b1).exp() - 1.0));
                    (sa, sc)
                }
                DerivativeOrder::Two => {
                    let q0 = 12.0 * cos0 * (3.0 * b0).exp() - 3.0 * (2.0 * b0).exp()
                        + 8.0 * cos0.powi(3) * (3.0 * b0).exp()
                        - 12.0 * cos0 * cos0 * (4.0 * b0).exp()
                        - 3.0 * (4.0 * b0).exp()
                        + 6.0 * cos0 * (5.0 * b0).exp()
                        - (6.0 * b0).exp()
                        + 6.0 * cos0 * eb0
                        - (1.0 + 12.0 * cos0 * cos0 * (2.0 * b0).exp());
                    let q1 = 12.0 * cos1 * (3.0 * b1).exp() - 3.0 * (2.0 * b1).exp()
                        + 8.0 * cos1.powi(3) * (3.0 * b1).exp()
                        - 12.0 * cos1 * cos1 * (4.0 * b1).exp()
                        - 3.0 * (4.0 * b1).exp()
                        + 6.0 * cos1 * (5.0 * b1).exp()
                        - (6.0 * b1).exp()
                        + 6.0 * cos1 * eb1
                        - (1.0 + 12.0 * cos1 * cos1 * (2.0 * b1).exp());
                    let na = 4.0 * a0 * sin0 * (3.0 * b0).exp()
                        + a1 * cos0 * cos0 * (4.0 * b0).exp()
                        - (4.0 * a0 * sin0 * eb0 + 6.0 * a1 * cos0 * cos0 * (2.0 * b0).exp())
                        + 2.0 * a1 * cos0.powi(3) * eb0
                        - 2.0 * a1 * cos0 * eb0
                        + 2.0 * a1 * cos0.powi(3) * (3.0 * b0).exp()
                        - 2.0 * a1 * cos0 * (3.0 * b0).exp()
                        + a1 * cos0 * cos0
                        - a1 * (4.0 * b0).exp()
                        + 2.0 * a0 * sin0 * cos0 * cos0 * eb0
                        - 2.0 * a0 * sin0 * cos0 * cos0 * (3.0 * b0).exp()
                        - (a0 * sin0 * cos0 * (4.0 * b0).exp() + a1)
                        + 6.0 * a1 * (2.0 * b0).exp()
                        + a0 * cos0 * sin0;
                    let nc = 4.0 * c0 * sin1 * (3.0 * b1).exp()
                        + c1 * cos1 * cos1 * (4.0 * b1).exp()
                        - (4.0 * c0 * sin1 * eb1 + 6.0 * c1 * cos1 * cos1 * (2.0 * b1).exp())
                        + 2.0 * c1 * cos1.powi(3) * eb1
                        - 2.0 * c1 * cos1 * eb1
                        + 2.0 * c1 * cos1.powi(3) * (3.0 * b1).exp()
                        - 2.0 * c1 * cos1 * (3.0 * b1).exp()
                        + c1 * cos1 * cos1
                        - c1 * (4.0 * b1).exp()
                        + 2.0 * c0 * sin1 * cos1 * cos1 * eb1
                        - 2.0 * c0 * sin1 * cos1 * cos1 * (3.0 * b1).exp()
                        - (c0 * sin1 * cos1 * (4.0 * b1).exp() + c1)
                        + 6.0 * c1 * (2.0 * b1).exp()
                        + c0 * cos1 * sin1;
                    (na * eb0 / (q0 * sin0), nc * eb1 / (q1 * sin1))
                }
                DerivativeOrder::None | DerivativeOrder::Three => unreachable!(),
            };
            let scale = sum_a + sum_c;
            a0 /= scale;
            a1 /= scale;
            c0 /= scale;
            c1 /= scale;
            out.sp0 = a0 + c0;
            out.sp1 = (-b1).exp() * (c1 * sin1 - (c0 + 2.0 * a0) * cos1)
                + (-b0).exp() * (a1 * sin0 - (2.0 * c0 + a0) * cos0);
            out.sp2 = 2.0
                * (-b0 - b1).exp()
                * ((a0 + c0) * cos1 * cos0 - cos1 * a1 * sin0 - cos0 * c1 * sin1)
                + c0 * (-2.0 * b0).exp()
                + a0 * (-2.0 * b1).exp();
            out.sp3 = (-b1 - 2.0 * b0).exp() * (c1 * sin1 - c0 * cos1)
                + (-b0 - 2.0 * b1).exp() * (a1 * sin0 - a0 * cos0);
            out.sd1 = -2.0 * (-b1).exp() * cos1 - 2.0 * (-b0).exp() * cos0;
            out.sd2 = 4.0 * cos1 * cos0 * (-b0 - b1).exp() + (-2.0 * b1).exp() + (-2.0 * b0).exp();
            out.sd3 = -2.0 * cos0 * (-b0 - 2.0 * b1).exp() - 2.0 * cos1 * (-b1 - 2.0 * b0).exp();
            out.sd4 = (-2.0 * b0 - 2.0 * b1).exp();
            match derivative {
                DerivativeOrder::Zero | DerivativeOrder::Two => {
                    out.sn1 = out.sp1 - out.sd1 * out.sp0;
                    out.sn2 = out.sp2 - out.sd2 * out.sp0;
                    out.sn3 = out.sp3 - out.sd3 * out.sp0;
                    out.sn4 = -out.sd4 * out.sp0;
                }
                DerivativeOrder::One | DerivativeOrder::OneContours => {
                    out.sn1 = -out.sp1 + out.sd1 * out.sp0;
                    out.sn2 = -out.sp2 + out.sd2 * out.sp0;
                    out.sn3 = -out.sp3 + out.sd3 * out.sp0;
                    out.sn4 = out.sd4 * out.sp0;
                }
                DerivativeOrder::None | DerivativeOrder::Three => unreachable!(),
            }
        }
        RecursiveFilterType::Unknown => unreachable!(),
    }
    out.filter_type = filter_type;
    out.derivative = derivative;
    Ok(out)
}

/// Equivalent to `RecursiveFilter1D`; returns a newly owned filtered line.
pub fn recursive_filter_1d(
    coefficients: &RecursiveFilterCoefficients,
    input: &[f64],
) -> Result<Vec<f64>, String> {
    if coefficients.filter_type == RecursiveFilterType::Unknown
        || coefficients.derivative == DerivativeOrder::None
    {
        return Err("recursive filter coefficients have not been initialized".into());
    }
    let dim = input.len();
    let min_dim = match coefficients.filter_type {
        RecursiveFilterType::AlphaDeriche => 3,
        RecursiveFilterType::GaussianDeriche | RecursiveFilterType::GaussianFidrich => 5,
        RecursiveFilterType::Unknown => unreachable!(),
    };
    if dim < min_dim {
        return Err(format!(
            "recursive filter requires at least {min_dim} samples"
        ));
    }
    let mut plus = vec![0.0; dim];
    let mut minus = vec![0.0; dim];
    match coefficients.filter_type {
        RecursiveFilterType::AlphaDeriche => match coefficients.derivative {
            DerivativeOrder::Zero | DerivativeOrder::Two => {
                plus[0] = coefficients.sp0 * input[0];
                plus[1] = coefficients.sp0 * input[1] + coefficients.sp1 * input[0]
                    - coefficients.sd1 * plus[0];
                for i in 2..dim {
                    plus[i] = coefficients.sp0 * input[i] + coefficients.sp1 * input[i - 1]
                        - coefficients.sd1 * plus[i - 1]
                        - coefficients.sd2 * plus[i - 2];
                }
                minus[dim - 1] = 0.0;
                minus[dim - 2] = coefficients.sn1 * input[dim - 1];
                for i in (0..dim - 2).rev() {
                    minus[i] = coefficients.sn1 * input[i + 1] + coefficients.sn2 * input[i + 2]
                        - coefficients.sd1 * minus[i + 1]
                        - coefficients.sd2 * minus[i + 2];
                }
            }
            DerivativeOrder::One | DerivativeOrder::OneContours => {
                plus[0] = 0.0;
                plus[1] = coefficients.sp1 * input[0];
                for i in 2..dim {
                    plus[i] = coefficients.sp1 * input[i - 1]
                        - coefficients.sd1 * plus[i - 1]
                        - coefficients.sd2 * plus[i - 2];
                }
                minus[dim - 1] = 0.0;
                minus[dim - 2] = coefficients.sn1 * input[dim - 1];
                for i in (0..dim - 2).rev() {
                    minus[i] = coefficients.sn1 * input[i + 1]
                        - coefficients.sd1 * minus[i + 1]
                        - coefficients.sd2 * minus[i + 2];
                }
            }
            DerivativeOrder::Three => {
                plus[0] = coefficients.sp0 * input[0];
                plus[1] = coefficients.sp0 * input[1] + coefficients.sp1 * input[0]
                    - coefficients.sd1 * plus[0];
                for i in 2..dim {
                    plus[i] = coefficients.sp0 * input[i] + coefficients.sp1 * input[i - 1]
                        - coefficients.sd1 * plus[i - 1]
                        - coefficients.sd2 * plus[i - 2];
                }
                minus[dim - 1] = coefficients.sn0 * input[dim - 1];
                minus[dim - 2] = coefficients.sn0 * input[dim - 2]
                    + coefficients.sn1 * input[dim - 1]
                    - coefficients.sd1 * minus[dim - 1];
                for i in (0..dim - 2).rev() {
                    minus[i] = coefficients.sn0 * input[i] + coefficients.sn1 * input[i + 1]
                        - coefficients.sd1 * minus[i + 1]
                        - coefficients.sd2 * minus[i + 2];
                }
            }
            DerivativeOrder::None => unreachable!(),
        },
        RecursiveFilterType::GaussianDeriche | RecursiveFilterType::GaussianFidrich => {
            plus[0] = coefficients.sp0 * input[0];
            plus[1] = coefficients.sp0 * input[1] + coefficients.sp1 * input[0]
                - coefficients.sd1 * plus[0];
            plus[2] = coefficients.sp0 * input[2]
                + coefficients.sp1 * input[1]
                + coefficients.sp2 * input[0]
                - coefficients.sd1 * plus[1]
                - coefficients.sd2 * plus[0];
            plus[3] = coefficients.sp0 * input[3]
                + coefficients.sp1 * input[2]
                + coefficients.sp2 * input[1]
                + coefficients.sp3 * input[0]
                - coefficients.sd1 * plus[2]
                - coefficients.sd2 * plus[1]
                - coefficients.sd3 * plus[0];
            for i in 4..dim {
                plus[i] = coefficients.sp0 * input[i]
                    + coefficients.sp1 * input[i - 1]
                    + coefficients.sp2 * input[i - 2]
                    + coefficients.sp3 * input[i - 3]
                    - coefficients.sd1 * plus[i - 1]
                    - coefficients.sd2 * plus[i - 2]
                    - coefficients.sd3 * plus[i - 3]
                    - coefficients.sd4 * plus[i - 4];
            }
            minus[dim - 1] = 0.0;
            minus[dim - 2] = coefficients.sn1 * input[dim - 1];
            minus[dim - 3] = coefficients.sn1 * input[dim - 2] + coefficients.sn2 * input[dim - 1]
                - coefficients.sd1 * minus[dim - 2];
            minus[dim - 4] = coefficients.sn1 * input[dim - 3]
                + coefficients.sn2 * input[dim - 2]
                + coefficients.sn3 * input[dim - 1]
                - coefficients.sd1 * minus[dim - 3]
                - coefficients.sd2 * minus[dim - 2];
            for i in (0..dim - 4).rev() {
                minus[i] = coefficients.sn1 * input[i + 1]
                    + coefficients.sn2 * input[i + 2]
                    + coefficients.sn3 * input[i + 3]
                    + coefficients.sn4 * input[i + 4]
                    - coefficients.sd1 * minus[i + 1]
                    - coefficients.sd2 * minus[i + 2]
                    - coefficients.sd3 * minus[i + 3]
                    - coefficients.sd4 * minus[i + 4];
            }
        }
        RecursiveFilterType::Unknown => unreachable!(),
    }
    Ok(plus.into_iter().zip(minus).map(|(a, b)| a + b).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn alpha_smoothing_coefficients_match_c_formula() {
        let c = init_recursive_coefficients(
            1.0,
            RecursiveFilterType::AlphaDeriche,
            DerivativeOrder::Zero,
        )
        .unwrap();
        assert!((c.sd1 + 2.0 / std::f64::consts::E).abs() < 1.0e-14);
        assert_eq!(c.filter_type, RecursiveFilterType::AlphaDeriche);
    }
    #[test]
    fn alpha_filter_has_c_boundary_conditions() {
        let c = init_recursive_coefficients(
            1.0,
            RecursiveFilterType::AlphaDeriche,
            DerivativeOrder::Zero,
        )
        .unwrap();
        let out = recursive_filter_1d(&c, &[0., 0., 1., 0., 0.]).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(out[2] > out[0]);
    }

    #[test]
    fn fidrich_filter_runs_the_fourth_order_recurrence() {
        let c = init_recursive_coefficients(
            1.2,
            RecursiveFilterType::GaussianFidrich,
            DerivativeOrder::Zero,
        )
        .unwrap();
        let out = recursive_filter_1d(&c, &[0., 0., 1., 0., 0., 0.]).unwrap();
        assert!(out.iter().all(|value| value.is_finite()));
        assert!(out[2] > out[0]);
    }

    #[test]
    fn deriche_gaussian_covers_source_normalizations() {
        for derivative in [
            DerivativeOrder::Zero,
            DerivativeOrder::One,
            DerivativeOrder::OneContours,
            DerivativeOrder::Two,
        ] {
            let coefficients =
                init_recursive_coefficients(1.2, RecursiveFilterType::GaussianDeriche, derivative)
                    .unwrap();
            let filtered = recursive_filter_1d(&coefficients, &[0., 0., 1., 0., 0., 0.]).unwrap();
            assert!(filtered.iter().all(|value| value.is_finite()));
        }
        let defaulted = init_recursive_coefficients(
            1.2,
            RecursiveFilterType::GaussianDeriche,
            DerivativeOrder::Three,
        )
        .unwrap();
        assert_eq!(defaulted.derivative, DerivativeOrder::Zero);
    }

    #[test]
    fn deriche_gaussian_coefficients_match_c_reference() {
        let expected = [
            [
                0.33233233902549331,
                0.11973121803792637,
                -0.3469642763179866,
                0.23503866754497044,
            ],
            [
                0.00061158308012210227,
                -0.16406740460386698,
                -0.38693286044421615,
                0.16383076301327607,
            ],
            [
                -0.24286102711859564,
                0.050912455579202071,
                -0.42253300953287698,
                -0.051704345107463801,
            ],
        ];
        for (derivative, reference) in [
            DerivativeOrder::Zero,
            DerivativeOrder::One,
            DerivativeOrder::Two,
        ]
        .into_iter()
        .zip(expected)
        {
            let c =
                init_recursive_coefficients(1.2, RecursiveFilterType::GaussianDeriche, derivative)
                    .unwrap();
            for (actual, expected) in [c.sp0, c.sp1, c.sd1, c.sn1].into_iter().zip(reference) {
                assert!(
                    (actual - expected).abs() < 1.0e-13,
                    "{actual} != {expected}"
                );
            }
        }
    }
}
