//! Translation of `IMOD/mrc/recline.{c,h}` — Gregoire Malandain's recursive
//! filtering of a 1D line.
//!
//! `InitRecursiveCoefficients` `malloc`s an `RFcoefficientType` and returns
//! `NULL` on every rejection, so the translation returns `Option`.  Everything
//! else keeps the source's shape: `RecursiveFilter1D` still takes the caller's
//! `in`, `out`, `work1` and `work2` lines and returns `EXIT_ON_FAILURE` /
//! `EXIT_ON_SUCCESS`, because the callers in `preNAD.cpp` and `preNID.cpp`
//! allocate those four buffers themselves.
//!
//! Every arithmetic statement is the source's statement, including where the
//! C accumulates with `+=` and `*=`: `sumA = A - B; sumA += C - D;` is
//! `A - B + (C - D)`, which is not the same double as `((A - B) + C) - D`.

use std::sync::atomic::{AtomicI32, Ordering};

static VERBOSE: AtomicI32 = AtomicI32::new(0);

const EXIT_ON_FAILURE: i32 = 0;
const EXIT_ON_SUCCESS: i32 = 1;

/// `recursiveFilterType`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RecursiveFilterType {
    /// `UNKNOWN_FILTER`
    Unknown,
    /// `ALPHA_DERICHE`
    AlphaDeriche,
    /// `GAUSSIAN_DERICHE`
    GaussianDeriche,
    /// `GAUSSIAN_FIDRICH`
    GaussianFidrich,
}

/// `derivativeOrder`.  `SMOOTHING` is `DERIVATIVE_0` and `DERIVATIVE_1_EDGES`
/// is `DERIVATIVE_1_CONTOURS`; the enum has one arm per distinct value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DerivativeOrder {
    /// `NODERIVATIVE` (-1)
    None,
    /// `DERIVATIVE_0` / `SMOOTHING` (0)
    Zero,
    /// `DERIVATIVE_1` (1)
    One,
    /// `DERIVATIVE_2` (2)
    Two,
    /// `DERIVATIVE_3` (3)
    Three,
    /// `DERIVATIVE_1_CONTOURS` / `DERIVATIVE_1_EDGES` (11)
    OneContours,
}

/// `RFcoefficientType`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RFcoefficientType {
    /*--- denominateur       ---*/
    pub sd1: f64,
    pub sd2: f64,
    pub sd3: f64,
    pub sd4: f64,
    /*--- numerateur positif ---*/
    pub sp0: f64,
    pub sp1: f64,
    pub sp2: f64,
    pub sp3: f64,
    /*--- numerateur negatif ---*/
    pub sn0: f64,
    pub sn1: f64,
    pub sn2: f64,
    pub sn3: f64,
    pub sn4: f64,
    /*--- type de filtre en cours ---*/
    pub type_filter: RecursiveFilterType,
    pub derivative: DerivativeOrder,
}

/// C `printRecursiveCoefficients`.
pub fn print_recursive_coefficients(rfc: &RFcoefficientType) {
    print!("denominator:\n");
    print!(
        "{:.6} {:.6} {:.6} {:.6}\n",
        rfc.sd1, rfc.sd2, rfc.sd3, rfc.sd4
    );
    print!("positive numerator:\n");
    print!(
        "{:.6} {:.6} {:.6} {:.6}\n",
        rfc.sp0, rfc.sp1, rfc.sp2, rfc.sp3
    );
    print!("negative numerator:\n");
    print!(
        "{:.6} {:.6} {:.6} {:.6} {:.6}\n",
        rfc.sn0, rfc.sn1, rfc.sn2, rfc.sn3, rfc.sn4
    );
    print!("\n");
}

/// C `InitRecursiveCoefficients`.
pub fn init_recursive_coefficients(
    x: f64,
    mut type_filter: RecursiveFilterType,
    mut derivative: DerivativeOrder,
) -> Option<RFcoefficientType> {
    let proc = "InitRecursiveCoefficients";
    let ex: f64;
    let mut k1: f64;
    let k2: f64;
    let (mut a0, mut a1, mut c0, mut c1, mut omega0, mut omega1, mut b0, mut b1);
    let (cos0, sin0, cos1, sin1);
    let mut sum_a: f64 = 0.0;
    let mut sum_c: f64 = 0.0;
    let mut aux: f64;

    let mut rfc = RFcoefficientType {
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
        type_filter: RecursiveFilterType::Unknown,
        derivative: DerivativeOrder::None,
    };

    a0 = 0.0;
    a1 = 0.0;
    c0 = 0.0;
    c1 = 0.0;
    b0 = 0.0;
    b1 = 0.0;
    omega0 = 0.0;
    omega1 = 0.0;

    /*--- Selon le type de filtrage (filtres de Deriche,
    ou approximation de la gaussienne), x designe
    soit alpha, soit sigma                         ---*/

    // The C `switch` runs the `default:` arm into `case ALPHA_DERICHE:`, so an
    // unrecognised filter type becomes Deriche's.
    if type_filter != RecursiveFilterType::GaussianFidrich
        && type_filter != RecursiveFilterType::GaussianDeriche
        && type_filter != RecursiveFilterType::AlphaDeriche
    {
        if VERBOSE.load(Ordering::Relaxed) != 0 {
            eprintln!("{proc}: switch to default recursive filter (Deriche's filters).");
        }
        type_filter = RecursiveFilterType::AlphaDeriche;
    }

    match type_filter {
        RecursiveFilterType::GaussianFidrich => {
            if x < 0.1 {
                if VERBOSE.load(Ordering::Relaxed) != 0 {
                    eprintln!("{proc}: improper value of coefficient (should be >= 0.1).");
                }
                return None;
            }

            match derivative {
                DerivativeOrder::Zero => {
                    a0 = 0.6570033214 / x;
                    a1 = 1.978946687 / x;
                    c0 = -0.2580640608 / x;
                    c1 = -0.2391206463 / x;
                    omega0 = 0.6512453378;
                    omega1 = 2.05339943;
                    b0 = 1.906154352;
                    b1 = 1.881305409;
                }
                DerivativeOrder::One | DerivativeOrder::OneContours => {
                    a0 = -0.1726729496 / x;
                    a1 = -2.003565572 / x;
                    c0 = 0.1726730777 / x;
                    c1 = 0.4440126835 / x;
                    b0 = 1.560644213;
                    b1 = 1.594202256;
                    omega0 = 0.6995461735;
                    omega1 = 2.144671764;
                }
                DerivativeOrder::Two => {
                    a0 = -0.7241334169 / x;
                    a1 = 1.688628765 / x;
                    c0 = 0.3251949838 / x;
                    c1 = -0.7211796018 / x;
                    b0 = 1.294951143;
                    b1 = 1.427007123;
                    omega0 = 0.7789803775;
                    omega1 = 2.233566862;
                }
                DerivativeOrder::Three => {
                    a0 = 1.285774106 / x;
                    a1 = -0.2896378408 / x;
                    c0 = -1.28577129 / x;
                    c1 = 0.26249833 / x;
                    b0 = 1.01162886;
                    b1 = 1.273344739;
                    omega0 = 0.9474270928;
                    omega1 = 2.337607006;
                }
                DerivativeOrder::None => {
                    if VERBOSE.load(Ordering::Relaxed) != 0 {
                        eprintln!("{proc}: improper value of derivative order.");
                    }
                    return None;
                }
            }

            omega0 /= x;
            sin0 = omega0.sin();
            cos0 = omega0.cos();
            omega1 /= x;
            sin1 = omega1.sin();
            cos1 = omega1.cos();
            b0 /= x;
            b1 /= x;

            rfc.sp0 = a0 + c0;
            rfc.sp1 = (-b1).exp() * (c1 * sin1 - (c0 + 2.0 * a0) * cos1);
            rfc.sp1 += (-b0).exp() * (a1 * sin0 - (2.0 * c0 + a0) * cos0);
            rfc.sp2 = 2.0
                * (-b0 - b1).exp()
                * ((a0 + c0) * cos1 * cos0 - cos1 * a1 * sin0 - cos0 * c1 * sin1);
            rfc.sp2 += c0 * (-2.0 * b0).exp() + a0 * (-2.0 * b1).exp();
            rfc.sp3 = (-b1 - 2.0 * b0).exp() * (c1 * sin1 - c0 * cos1);
            rfc.sp3 += (-b0 - 2.0 * b1).exp() * (a1 * sin0 - a0 * cos0);

            rfc.sd1 = -2.0 * (-b1).exp() * cos1 - 2.0 * (-b0).exp() * cos0;
            rfc.sd2 = 4.0 * cos1 * cos0 * (-b0 - b1).exp() + (-2.0 * b1).exp() + (-2.0 * b0).exp();
            rfc.sd3 = -2.0 * cos0 * (-b0 - 2.0 * b1).exp() - 2.0 * cos1 * (-b1 - 2.0 * b0).exp();
            rfc.sd4 = (-2.0 * b0 - 2.0 * b1).exp();

            match derivative {
                DerivativeOrder::Zero | DerivativeOrder::Two => {
                    rfc.sn1 = rfc.sp1 - rfc.sd1 * rfc.sp0;
                    rfc.sn2 = rfc.sp2 - rfc.sd2 * rfc.sp0;
                    rfc.sn3 = rfc.sp3 - rfc.sd3 * rfc.sp0;
                    rfc.sn4 = -rfc.sd4 * rfc.sp0;
                }
                DerivativeOrder::One | DerivativeOrder::OneContours | DerivativeOrder::Three => {
                    rfc.sn1 = -rfc.sp1 + rfc.sd1 * rfc.sp0;
                    rfc.sn2 = -rfc.sp2 + rfc.sd2 * rfc.sp0;
                    rfc.sn3 = -rfc.sp3 + rfc.sd3 * rfc.sp0;
                    rfc.sn4 = rfc.sd4 * rfc.sp0;
                }
                DerivativeOrder::None => {
                    if VERBOSE.load(Ordering::Relaxed) != 0 {
                        eprintln!("{proc}: improper value of derivative order.");
                    }
                    return None;
                }
            }

            rfc.type_filter = type_filter;
            rfc.derivative = derivative;
        }

        RecursiveFilterType::GaussianDeriche => {
            if x < 0.1 {
                if VERBOSE.load(Ordering::Relaxed) != 0 {
                    eprintln!("{proc}: improper value of coefficient (should be >= 0.1).");
                }
                return None;
            }

            // `default:` falls through into `case DERIVATIVE_0:`, so
            // NODERIVATIVE and DERIVATIVE_3 both become smoothing.
            if !matches!(
                derivative,
                DerivativeOrder::Zero
                    | DerivativeOrder::One
                    | DerivativeOrder::OneContours
                    | DerivativeOrder::Two
            ) {
                if VERBOSE.load(Ordering::Relaxed) != 0 {
                    eprintln!("{proc}: switch to default coefficients (smoothing).");
                }
                derivative = DerivativeOrder::Zero;
            }
            match derivative {
                DerivativeOrder::One | DerivativeOrder::OneContours => {
                    a0 = -0.6472;
                    omega0 = 0.6719;
                    a1 = -4.531;
                    b0 = 1.527;
                    c0 = 0.6494;
                    omega1 = 2.072;
                    c1 = 0.9557;
                    b1 = 1.516;
                }
                DerivativeOrder::Two => {
                    a0 = -1.331;
                    omega0 = 0.748;
                    a1 = 3.661;
                    b0 = 1.24;
                    c0 = 0.3225;
                    omega1 = 2.166;
                    c1 = -1.738;
                    b1 = 1.314;
                }
                _ => {
                    a0 = 1.68;
                    omega0 = 0.6318;
                    a1 = 3.735;
                    b0 = 1.783;
                    c0 = -0.6803;
                    omega1 = 1.997;
                    c1 = -0.2598;
                    b1 = 1.723;
                }
            }

            omega0 /= x;
            sin0 = omega0.sin();
            cos0 = omega0.cos();
            omega1 /= x;
            sin1 = omega1.sin();
            cos1 = omega1.cos();
            b0 /= x;
            b1 /= x;

            /*--- normalisation ---*/
            match derivative {
                DerivativeOrder::One => {
                    aux = (4.0 * b0).exp() - 4.0 * cos0 * (3.0 * b0).exp();
                    aux += 2.0 * (2.0 * b0).exp() + 4.0 * cos0 * cos0 * (2.0 * b0).exp();
                    aux += 1.0 - 4.0 * cos0 * b0.exp();
                    sum_a = a0 * cos0 - a1 * sin0 + a1 * sin0 * (2.0 * b0).exp();
                    sum_a += a0 * cos0 * (2.0 * b0).exp() - 2.0 * a0 * b0.exp();
                    sum_a *= b0.exp() / aux;
                    aux = (4.0 * b1).exp() - 4.0 * cos1 * (3.0 * b1).exp();
                    aux += 2.0 * (2.0 * b1).exp() + 4.0 * cos1 * cos1 * (2.0 * b1).exp();
                    aux += 1.0 - 4.0 * cos1 * b1.exp();
                    sum_c = c0 * cos1 - c1 * sin1 + c1 * sin1 * (2.0 * b1).exp();
                    sum_c += c0 * cos1 * (2.0 * b1).exp() - 2.0 * c0 * b1.exp();
                    sum_c *= b1.exp() / aux;
                    /*--- on multiplie les sommes par 2 car on n'a calcule que des demi-sommes
                    et on change le signe car la somme doit etre egale a -1              ---*/
                    sum_a *= -2.0;
                    sum_c *= -2.0;
                }
                DerivativeOrder::OneContours => {
                    /*--- la somme de 1 a l'infini est egale a 1 : cela introduit
                    un petit biais (reponse un rien superieur a la hauteur du step).
                    Avec une somme de 0 a l'infini, c'est pire                       ---*/
                    sum_a = a1 * b0.exp() - a1 * cos0 * cos0 * b0.exp();
                    sum_a += a0 * cos0 * sin0 * b0.exp() - a0 * sin0;
                    sum_a /= sin0 * (2.0 * cos0 * b0.exp() - (2.0 * b0).exp() - 1.0);
                    sum_c = c1 * b1.exp() - c1 * cos1 * cos1 * b1.exp();
                    sum_c += c0 * cos1 * sin1 * b1.exp() - c0 * sin1;
                    sum_c /= sin1 * (2.0 * cos1 * b1.exp() - (2.0 * b1).exp() - 1.0);
                }
                DerivativeOrder::Two => {
                    aux = 12.0 * cos0 * (3.0 * b0).exp() - 3.0 * (2.0 * b0).exp();
                    aux += 8.0 * cos0 * cos0 * cos0 * (3.0 * b0).exp()
                        - 12.0 * cos0 * cos0 * (4.0 * b0).exp();
                    aux -= 3.0 * (4.0 * b0).exp();
                    aux += 6.0 * cos0 * (5.0 * b0).exp() - (6.0 * b0).exp() + 6.0 * cos0 * b0.exp();
                    aux -= 1.0 + 12.0 * cos0 * cos0 * (2.0 * b0).exp();
                    sum_a =
                        4.0 * a0 * sin0 * (3.0 * b0).exp() + a1 * cos0 * cos0 * (4.0 * b0).exp();
                    sum_a -= 4.0 * a0 * sin0 * b0.exp() + 6.0 * a1 * cos0 * cos0 * (2.0 * b0).exp();
                    sum_a += 2.0 * a1 * cos0 * cos0 * cos0 * b0.exp() - 2.0 * a1 * cos0 * b0.exp();
                    sum_a += 2.0 * a1 * cos0 * cos0 * cos0 * (3.0 * b0).exp()
                        - 2.0 * a1 * cos0 * (3.0 * b0).exp();
                    sum_a += a1 * cos0 * cos0 - a1 * (4.0 * b0).exp();
                    sum_a += 2.0 * a0 * sin0 * cos0 * cos0 * b0.exp()
                        - 2.0 * a0 * sin0 * cos0 * cos0 * (3.0 * b0).exp();
                    sum_a -= a0 * sin0 * cos0 * (4.0 * b0).exp() + a1;
                    sum_a += 6.0 * a1 * (2.0 * b0).exp() + a0 * cos0 * sin0;
                    sum_a *= 2.0 * b0.exp() / (aux * sin0);
                    aux = 12.0 * cos1 * (3.0 * b1).exp() - 3.0 * (2.0 * b1).exp();
                    aux += 8.0 * cos1 * cos1 * cos1 * (3.0 * b1).exp()
                        - 12.0 * cos1 * cos1 * (4.0 * b1).exp();
                    aux -= 3.0 * (4.0 * b1).exp();
                    aux += 6.0 * cos1 * (5.0 * b1).exp() - (6.0 * b1).exp() + 6.0 * cos1 * b1.exp();
                    aux -= 1.0 + 12.0 * cos1 * cos1 * (2.0 * b1).exp();
                    sum_c =
                        4.0 * c0 * sin1 * (3.0 * b1).exp() + c1 * cos1 * cos1 * (4.0 * b1).exp();
                    sum_c -= 4.0 * c0 * sin1 * b1.exp() + 6.0 * c1 * cos1 * cos1 * (2.0 * b1).exp();
                    sum_c += 2.0 * c1 * cos1 * cos1 * cos1 * b1.exp() - 2.0 * c1 * cos1 * b1.exp();
                    sum_c += 2.0 * c1 * cos1 * cos1 * cos1 * (3.0 * b1).exp()
                        - 2.0 * c1 * cos1 * (3.0 * b1).exp();
                    sum_c += c1 * cos1 * cos1 - c1 * (4.0 * b1).exp();
                    sum_c += 2.0 * c0 * sin1 * cos1 * cos1 * b1.exp()
                        - 2.0 * c0 * sin1 * cos1 * cos1 * (3.0 * b1).exp();
                    sum_c -= c0 * sin1 * cos1 * (4.0 * b1).exp() + c1;
                    sum_c += 6.0 * c1 * (2.0 * b1).exp() + c0 * cos1 * sin1;
                    sum_c *= 2.0 * b1.exp() / (aux * sin1);
                    /*--- on divise les sommes par 2 (la somme doit etre egale a 2) ---*/
                    sum_a /= 2.0;
                    sum_c /= 2.0;
                }
                _ => {
                    sum_a = 2.0 * a1 * b0.exp() * cos0 * cos0 - a0 * sin0 * (2.0 * b0).exp();
                    sum_a += a0 * sin0 - 2.0 * a1 * b0.exp();
                    sum_a /= (2.0 * cos0 * b0.exp() - (2.0 * b0).exp() - 1.0) * sin0;
                    sum_c = 2.0 * c1 * b1.exp() * cos1 * cos1 - c0 * sin1 * (2.0 * b1).exp();
                    sum_c += c0 * sin1 - 2.0 * c1 * b1.exp();
                    sum_c /= (2.0 * cos1 * b1.exp() - (2.0 * b1).exp() - 1.0) * sin1;
                }
            }
            a0 /= sum_a + sum_c;
            a1 /= sum_a + sum_c;
            c0 /= sum_a + sum_c;
            c1 /= sum_a + sum_c;

            /*--- coefficients du calcul recursif ---*/
            rfc.sp0 = a0 + c0;
            rfc.sp1 = (-b1).exp() * (c1 * sin1 - (c0 + 2.0 * a0) * cos1);
            rfc.sp1 += (-b0).exp() * (a1 * sin0 - (2.0 * c0 + a0) * cos0);
            rfc.sp2 = 2.0
                * (-b0 - b1).exp()
                * ((a0 + c0) * cos1 * cos0 - cos1 * a1 * sin0 - cos0 * c1 * sin1);
            rfc.sp2 += c0 * (-2.0 * b0).exp() + a0 * (-2.0 * b1).exp();
            rfc.sp3 = (-b1 - 2.0 * b0).exp() * (c1 * sin1 - c0 * cos1);
            rfc.sp3 += (-b0 - 2.0 * b1).exp() * (a1 * sin0 - a0 * cos0);

            rfc.sd1 = -2.0 * (-b1).exp() * cos1 - 2.0 * (-b0).exp() * cos0;
            rfc.sd2 = 4.0 * cos1 * cos0 * (-b0 - b1).exp() + (-2.0 * b1).exp() + (-2.0 * b0).exp();
            rfc.sd3 = -2.0 * cos0 * (-b0 - 2.0 * b1).exp() - 2.0 * cos1 * (-b1 - 2.0 * b0).exp();
            rfc.sd4 = (-2.0 * b0 - 2.0 * b1).exp();

            match derivative {
                DerivativeOrder::One | DerivativeOrder::OneContours | DerivativeOrder::Three => {
                    rfc.sn1 = -rfc.sp1 + rfc.sd1 * rfc.sp0;
                    rfc.sn2 = -rfc.sp2 + rfc.sd2 * rfc.sp0;
                    rfc.sn3 = -rfc.sp3 + rfc.sd3 * rfc.sp0;
                    rfc.sn4 = rfc.sd4 * rfc.sp0;
                }
                _ => {
                    rfc.sn1 = rfc.sp1 - rfc.sd1 * rfc.sp0;
                    rfc.sn2 = rfc.sp2 - rfc.sd2 * rfc.sp0;
                    rfc.sn3 = rfc.sp3 - rfc.sd3 * rfc.sp0;
                    rfc.sn4 = -rfc.sd4 * rfc.sp0;
                }
            }

            rfc.type_filter = type_filter;
            rfc.derivative = derivative;
        }

        _ => {
            /*--- ALPHA_DERICHE ---*/
            if x < 0.1 || x > 1.9 {
                if VERBOSE.load(Ordering::Relaxed) != 0 {
                    eprintln!(
                        "{proc}: improper value of coefficient (should be >= 0.1 and <= 1.9)."
                    );
                }
                return None;
            }
            ex = (-x).exp();

            if !matches!(
                derivative,
                DerivativeOrder::Zero
                    | DerivativeOrder::One
                    | DerivativeOrder::OneContours
                    | DerivativeOrder::Two
                    | DerivativeOrder::Three
            ) {
                if VERBOSE.load(Ordering::Relaxed) != 0 {
                    eprintln!("{proc}: switch to default coefficients (smoothing).");
                }
                derivative = DerivativeOrder::Zero;
            }
            match derivative {
                DerivativeOrder::One => {
                    rfc.sp1 = -(1.0 - ex) * (1.0 - ex) * (1.0 - ex) / (2.0 * (1.0 + ex));
                    rfc.sn1 = -rfc.sp1;
                    rfc.sd1 = -2.0 * ex;
                    rfc.sd2 = ex * ex;
                }
                DerivativeOrder::OneContours => {
                    rfc.sp1 = -(1.0 - ex) * (1.0 - ex);
                    rfc.sn1 = -rfc.sp1;
                    rfc.sd1 = -2.0 * ex;
                    rfc.sd2 = ex * ex;
                }
                DerivativeOrder::Two => {
                    k1 = -2.0 * (1.0 - ex) * (1.0 - ex) * (1.0 - ex);
                    k1 /= (1.0 + ex) * (1.0 + ex) * (1.0 + ex);
                    k2 = (1.0 - ex * ex) / (2.0 * ex);
                    rfc.sp0 = k1;
                    rfc.sp1 = -k1 * (1.0 + k2) * ex;
                    rfc.sn1 = k1 * (1.0 - k2) * ex;
                    rfc.sn2 = -k1 * ex * ex;
                    rfc.sd1 = -2.0 * ex;
                    rfc.sd2 = ex * ex;
                }
                DerivativeOrder::Three => {
                    k1 = (1.0 + x) * ex + (x - 1.0);
                    k2 = (1.0 - ex) / k1;
                    k1 *= (1.0 - ex) * (1.0 - ex) * (1.0 - ex) * (1.0 - ex);
                    k1 /= 2.0 * x * x * ex * ex;
                    k1 /= ex + 1.0;
                    rfc.sp0 = k1 * x * (k2 + 1.0);
                    rfc.sp1 = -k1 * x * (1.0 + k2 + k2 * x) * ex;
                    rfc.sn0 = -rfc.sp0;
                    rfc.sn1 = -rfc.sp1;
                    rfc.sd1 = -2.0 * ex;
                    rfc.sd2 = ex * ex;
                }
                _ => {
                    rfc.sp0 = (1.0 - ex) * (1.0 - ex) / (1.0 + 2.0 * x * ex - ex * ex);
                    rfc.sp1 = rfc.sp0 * (x - 1.0) * ex;
                    rfc.sn1 = rfc.sp0 * (x + 1.0) * ex;
                    rfc.sn2 = -rfc.sp0 * ex * ex;
                    rfc.sd1 = -2.0 * ex;
                    rfc.sd2 = ex * ex;
                }
            }
            rfc.type_filter = type_filter;
            rfc.derivative = derivative;
        }
    }

    Some(rfc)
}

/// C `RecursiveFilter1D`.
///
/// `in`, `out`, `work1` and `work2` are the caller's lines, as in the source;
/// `work2` may be `out` when `out` is not `in`.  The two work lines are
/// separate here so that the borrow checker sees them as distinct, which is
/// what both callers in this tree pass anyway.
pub fn recursive_filter_1d(
    rfc: &RFcoefficientType,
    r#in: &[f64],
    out: &mut [f64],
    work1: &mut [f64],
    work2: &mut [f64],
    dim: i32,
) -> i32 {
    let proc = "RecursiveFilter1D";
    let (mut rp0, mut rp1, mut rp2, mut rp3) = (0.0, 0.0, 0.0, 0.0);
    let (mut rd1, mut rd2, mut rd3, mut rd4) = (0.0, 0.0, 0.0, 0.0);
    let (mut rn0, mut rn1, mut rn2, mut rn3, mut rn4) = (0.0, 0.0, 0.0, 0.0, 0.0);
    let mut i: i32;

    if rfc.type_filter == RecursiveFilterType::Unknown {
        if VERBOSE.load(Ordering::Relaxed) != 0 {
            eprintln!("{proc}: unknown type of recursive filter.");
        }
        return EXIT_ON_FAILURE;
    }
    if rfc.derivative == DerivativeOrder::None {
        if VERBOSE.load(Ordering::Relaxed) != 0 {
            eprintln!("{proc}: unknown type of derivative.");
        }
        return EXIT_ON_FAILURE;
    }

    let dimu = dim as usize;

    match rfc.type_filter {
        RecursiveFilterType::GaussianFidrich | RecursiveFilterType::GaussianDeriche => {
            /*--- filtrage generique d'ordre 4 ---*/
            rp0 = rfc.sp0;
            rp1 = rfc.sp1;
            rp2 = rfc.sp2;
            rp3 = rfc.sp3;
            rd1 = rfc.sd1;
            rd2 = rfc.sd2;
            rd3 = rfc.sd3;
            rd4 = rfc.sd4;
            rn1 = rfc.sn1;
            rn2 = rfc.sn2;
            rn3 = rfc.sn3;
            rn4 = rfc.sn4;

            /* the C positions w4..w0 at work1[0..4] and d3..d0 at in[1..4] */
            /*--- calcul de y+ ---*/
            work1[0] = rp0 * r#in[0];
            work1[1] = rp0 * r#in[1] + rp1 * r#in[0] - rd1 * work1[0];
            work1[2] =
                rp0 * r#in[2] + rp1 * r#in[1] + rp2 * r#in[0] - rd1 * work1[1] - rd2 * work1[0];
            work1[3] = rp0 * r#in[3] + rp1 * r#in[2] + rp2 * r#in[1] + rp3 * r#in[0]
                - rd1 * work1[2]
                - rd2 * work1[1]
                - rd3 * work1[0];
            i = 4;
            while i < dim {
                let k = i as usize;
                work1[k] =
                    rp0 * r#in[k] + rp1 * r#in[k - 1] + rp2 * r#in[k - 2] + rp3 * r#in[k - 3]
                        - rd1 * work1[k - 1]
                        - rd2 * work1[k - 2]
                        - rd3 * work1[k - 3]
                        - rd4 * work1[k - 4];
                i += 1;
            }

            /*--- calcul de y- ---*/
            work2[dimu - 1] = 0.0;
            work2[dimu - 2] = rn1 * r#in[dimu - 1];
            work2[dimu - 3] = rn1 * r#in[dimu - 2] + rn2 * r#in[dimu - 1] - rd1 * work2[dimu - 2];
            work2[dimu - 4] = rn1 * r#in[dimu - 3] + rn2 * r#in[dimu - 2] + rn3 * r#in[dimu - 1]
                - rd1 * work2[dimu - 3]
                - rd2 * work2[dimu - 2];
            i = dim - 5;
            while i >= 0 {
                let k = i as usize;
                work2[k] =
                    rn1 * r#in[k + 1] + rn2 * r#in[k + 2] + rn3 * r#in[k + 3] + rn4 * r#in[k + 4]
                        - rd1 * work2[k + 1]
                        - rd2 * work2[k + 2]
                        - rd3 * work2[k + 3]
                        - rd4 * work2[k + 4];
                i -= 1;
            }

            /*--- calcul final ---*/
            for k in 0..dimu {
                out[k] = work1[k] + work2[k];
            }
        }

        _ => {
            /*--- ALPHA_DERICHE ---*/
            match rfc.derivative {
                DerivativeOrder::One | DerivativeOrder::OneContours => {
                    rp1 = rfc.sp1;
                    rn1 = rfc.sn1;
                    rd1 = rfc.sd1;
                    rd2 = rfc.sd2;

                    /*--- calcul de y+ ---*/
                    work1[0] = 0.0;
                    work1[1] = rp1 * r#in[0];
                    i = 2;
                    while i < dim {
                        let k = i as usize;
                        work1[k] = rp1 * r#in[k - 1] - rd1 * work1[k - 1] - rd2 * work1[k - 2];
                        i += 1;
                    }

                    /*--- calcul de y- ---*/
                    work2[dimu - 1] = 0.0;
                    work2[dimu - 2] = rn1 * r#in[dimu - 1];
                    i = dim - 3;
                    while i >= 0 {
                        let k = i as usize;
                        work2[k] = rn1 * r#in[k + 1] - rd1 * work2[k + 1] - rd2 * work2[k + 2];
                        i -= 1;
                    }

                    /*--- calcul final ---*/
                    for k in 0..dimu {
                        out[k] = work1[k] + work2[k];
                    }
                }

                DerivativeOrder::Three => {
                    rp0 = rfc.sp0;
                    rp1 = rfc.sp1;
                    rd1 = rfc.sd1;
                    rd2 = rfc.sd2;
                    rn0 = rfc.sn0;
                    rn1 = rfc.sn1;

                    /*--- calcul de y+ ---*/
                    work1[0] = rp0 * r#in[0];
                    work1[1] = rp0 * r#in[1] + rp1 * r#in[0] - rd1 * work1[0];
                    i = 2;
                    while i < dim {
                        let k = i as usize;
                        work1[k] = rp0 * r#in[k] + rp1 * r#in[k - 1]
                            - rd1 * work1[k - 1]
                            - rd2 * work1[k - 2];
                        i += 1;
                    }

                    /*--- calcul de y- ---*/
                    work2[dimu - 1] = rn0 * r#in[dimu - 1];
                    work2[dimu - 2] =
                        rn0 * r#in[dimu - 2] + rn1 * r#in[dimu - 1] - rd1 * work2[dimu - 1];
                    i = dim - 3;
                    while i >= 0 {
                        let k = i as usize;
                        work2[k] = rn0 * r#in[k] + rn1 * r#in[k + 1]
                            - rd1 * work2[k + 1]
                            - rd2 * work2[k + 2];
                        i -= 1;
                    }

                    /*--- calcul final ---*/
                    for k in 0..dimu {
                        out[k] = work1[k] + work2[k];
                    }
                }

                _ => {
                    /*--- DERIVATIVE_0, DERIVATIVE_2 ---*/
                    rp0 = rfc.sp0;
                    rp1 = rfc.sp1;
                    rd1 = rfc.sd1;
                    rd2 = rfc.sd2;
                    rn1 = rfc.sn1;
                    rn2 = rfc.sn2;

                    /*--- calcul de y+ ---*/
                    work1[0] = rp0 * r#in[0];
                    work1[1] = rp0 * r#in[1] + rp1 * r#in[0] - rd1 * work1[0];
                    i = 2;
                    while i < dim {
                        let k = i as usize;
                        work1[k] = rp0 * r#in[k] + rp1 * r#in[k - 1]
                            - rd1 * work1[k - 1]
                            - rd2 * work1[k - 2];
                        i += 1;
                    }

                    /*--- calcul de y- ---*/
                    work2[dimu - 1] = 0.0;
                    work2[dimu - 2] = rn1 * r#in[dimu - 1];
                    i = dim - 3;
                    while i >= 0 {
                        let k = i as usize;
                        work2[k] = rn1 * r#in[k + 1] + rn2 * r#in[k + 2]
                            - rd1 * work2[k + 1]
                            - rd2 * work2[k + 2];
                        i -= 1;
                    }

                    /*--- calcul final ---*/
                    for k in 0..dimu {
                        out[k] = work1[k] + work2[k];
                    }
                }
            }
        }
    }
    EXIT_ON_SUCCESS
}

/// C `Recline_verbose`.
pub fn recline_verbose() {
    VERBOSE.store(1, Ordering::Relaxed);
}

/// C `Recline_noverbose`.
pub fn recline_noverbose() {
    VERBOSE.store(0, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn filter(rfc: &RFcoefficientType, line: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; line.len()];
        let mut w1 = vec![0.0; line.len()];
        let mut w2 = vec![0.0; line.len()];
        assert_eq!(
            recursive_filter_1d(rfc, line, &mut out, &mut w1, &mut w2, line.len() as i32),
            EXIT_ON_SUCCESS
        );
        out
    }

    #[test]
    fn alpha_deriche_rejects_out_of_range_coefficients() {
        assert!(
            init_recursive_coefficients(
                2.0,
                RecursiveFilterType::AlphaDeriche,
                DerivativeOrder::Zero
            )
            .is_none()
        );
        assert!(
            init_recursive_coefficients(
                0.05,
                RecursiveFilterType::GaussianDeriche,
                DerivativeOrder::Zero
            )
            .is_none()
        );
    }

    #[test]
    fn gaussian_deriche_smoothing_preserves_a_constant_line() {
        let rfc = init_recursive_coefficients(
            2.0,
            RecursiveFilterType::GaussianDeriche,
            DerivativeOrder::Zero,
        )
        .unwrap();
        let line = vec![5.0; 60];
        let out = filter(&rfc, &line);
        // Away from the ends the normalised filter reproduces the constant.
        for value in &out[20..40] {
            assert!((value - 5.0).abs() < 1.0e-6, "{value}");
        }
    }

    #[test]
    fn gaussian_deriche_first_derivative_of_a_ramp_is_one() {
        let rfc = init_recursive_coefficients(
            2.0,
            RecursiveFilterType::GaussianDeriche,
            DerivativeOrder::One,
        )
        .unwrap();
        let line: Vec<f64> = (0..80).map(|i| i as f64).collect();
        let out = filter(&rfc, &line);
        for value in &out[30..50] {
            assert!((value - 1.0).abs() < 1.0e-6, "{value}");
        }
    }

    #[test]
    fn derivative_three_and_unknown_fall_back_to_smoothing_for_deriche_gaussian() {
        let rfc = init_recursive_coefficients(
            1.2,
            RecursiveFilterType::GaussianDeriche,
            DerivativeOrder::Three,
        )
        .unwrap();
        assert_eq!(rfc.derivative, DerivativeOrder::Zero);
    }

    #[test]
    fn alpha_deriche_orders_all_run() {
        for derivative in [
            DerivativeOrder::Zero,
            DerivativeOrder::One,
            DerivativeOrder::OneContours,
            DerivativeOrder::Two,
            DerivativeOrder::Three,
        ] {
            let rfc =
                init_recursive_coefficients(1.0, RecursiveFilterType::AlphaDeriche, derivative)
                    .unwrap();
            let out = filter(&rfc, &[0., 0., 1., 0., 0., 0.]);
            assert!(out.iter().all(|v| v.is_finite()));
        }
    }
}
