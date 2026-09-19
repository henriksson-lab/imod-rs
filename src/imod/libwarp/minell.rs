//! Minimum enclosing ellipses from `IMOD/libwarp/minell.c`.
//!
//! The C API exposes an opaque owner whose observable purpose is to transform
//! points to and from its minimum-area enclosing ellipse.  Rust makes that
//! owner explicit while retaining the source's 2-D affine convention and
//! leaving `Point.z` untouched.

use super::nn::Point;

const TOLERANCE: f64 = 1.0e-13;
// The dual update is deliberately carried to source-level numerical precision:
// four-support-point ellipses converge much more slowly than triangles.
const MAX_ITERATIONS: usize = 1_000_000;

/// Rust owner for C's opaque `minell`.
#[derive(Clone, Debug)]
pub struct Minell {
    pub center: Point,
    /// Source `a` and `b` semiaxes.  `a` is the axis associated with
    /// `theta`, exactly as `minell_center2human` computes it.
    pub a: f64,
    pub b: f64,
    pub theta: f64,
    /// Center-form matrix `[[r, t], [t, s]]` where inside points satisfy
    /// `(p-center)^T M (p-center) <= 1`.
    pub r: f64,
    pub t: f64,
    pub s: f64,
}

impl Minell {
    fn empty() -> Self {
        Self {
            center: Point {
                x: f64::NAN,
                y: f64::NAN,
                z: f64::NAN,
            },
            a: f64::NAN,
            b: f64::NAN,
            theta: f64::NAN,
            r: f64::NAN,
            t: f64::NAN,
            s: f64::NAN,
        }
    }

    fn from_pair(first: Point, second: Point) -> Self {
        let center = Point {
            x: (first.x + second.x) / 2.,
            y: (first.y + second.y) / 2.,
            z: f64::NAN,
        };
        let a = (center.x - first.x).hypot(center.y - first.y);
        let theta = (first.y - second.y).atan2(first.x - second.x);
        Self {
            center,
            a,
            b: 0.,
            theta,
            r: f64::INFINITY,
            t: 0.,
            s: f64::INFINITY,
        }
    }

    fn from_matrix(center: Point, r: f64, t: f64, s: f64) -> Self {
        let sum = r + s;
        let difference = r - s;
        let root = difference.hypot(2. * t);
        let low = (sum - root) / 2.;
        let high = (sum + root) / 2.;
        let a = (1. / low).sqrt();
        let b = (1. / high).sqrt();
        let mut theta = if low == high {
            0.
        } else {
            -(2. * t / root).asin() / 2.
        };
        if s < r {
            theta = if theta > 0. {
                std::f64::consts::FRAC_PI_2 - theta
            } else {
                -std::f64::consts::FRAC_PI_2 - theta
            };
        }
        Self {
            center,
            a,
            b,
            theta,
            r,
            t,
            s,
        }
    }

    /// C `minell_scalepoints`: map points into the ellipse's unit-circle
    /// coordinates.  As in C, a degenerate one/two-point ellipse produces
    /// floating IEEE division results rather than an invented finite axis.
    pub fn scale_points(&self, points: &mut [Point]) {
        let (sine, cosine) = self.theta.sin_cos();
        for point in points {
            let (x, y) = (point.x - self.center.x, point.y - self.center.y);
            point.x = (x * cosine - y * sine) / self.a;
            point.y = (x * sine + y * cosine) / self.b;
        }
    }

    /// C `minell_rescalepoints`, inverse to [`Self::scale_points`].
    pub fn rescale_points(&self, points: &mut [Point]) {
        let (sine, cosine) = self.theta.sin_cos();
        for point in points {
            let (x, y) = (point.x * self.a, point.y * self.b);
            point.x = x * cosine + y * sine + self.center.x;
            point.y = -x * sine + y * cosine + self.center.y;
        }
    }

    pub fn contains(&self, point: Point, epsilon: f64) -> bool {
        let (x, y) = (point.x - self.center.x, point.y - self.center.y);
        (self.r * x + 2. * self.t * y) * x + self.s * y * y <= 1. + epsilon
    }
}

/// C `minell_build`.  For a non-degenerate cloud this uses the same
/// homogeneous minimum-volume enclosing-ellipse formulation that the C
/// Welzl support-point search solves, converging the dual weights to the
/// source center-form matrix.  Degenerate clouds retain C's point/pair form.
pub fn minell_build(points: &[Point]) -> Minell {
    match points {
        [] => return Minell::empty(),
        [point] => {
            return Minell {
                center: Point {
                    x: point.x,
                    y: point.y,
                    z: f64::NAN,
                },
                a: 0.,
                b: 0.,
                theta: 0.,
                r: f64::INFINITY,
                t: 0.,
                s: f64::INFINITY,
            };
        }
        [first, second] => return Minell::from_pair(*first, *second),
        _ => {}
    }
    let count = points.len();
    let mut weights = vec![1. / count as f64; count];
    for _ in 0..MAX_ITERATIONS {
        let mut x = [[0.; 3]; 3];
        for (weight, point) in weights.iter().zip(points) {
            let value = [point.x, point.y, 1.];
            for row in 0..3 {
                for column in 0..3 {
                    x[row][column] += weight * value[row] * value[column];
                }
            }
        }
        let Some(next_inverse) = invert_3(x) else {
            return farthest_pair(points);
        };
        let (index, maximum) = points
            .iter()
            .enumerate()
            .map(|(index, point)| (index, quadratic(next_inverse, [point.x, point.y, 1.])))
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .expect("nonempty points");
        if maximum <= 3. * (1. + TOLERANCE) {
            break;
        }
        let step = ((maximum - 3.) / (3. * (maximum - 1.))).clamp(0., 1.);
        for weight in &mut weights {
            *weight *= 1. - step;
        }
        weights[index] += step;
    }
    let center = Point {
        x: weights
            .iter()
            .zip(points)
            .map(|(weight, point)| weight * point.x)
            .sum(),
        y: weights
            .iter()
            .zip(points)
            .map(|(weight, point)| weight * point.y)
            .sum(),
        z: f64::NAN,
    };
    let mut covariance = [[0.; 2]; 2];
    for (weight, point) in weights.iter().zip(points) {
        let value = [point.x - center.x, point.y - center.y];
        for row in 0..2 {
            for column in 0..2 {
                covariance[row][column] += weight * value[row] * value[column];
            }
        }
    }
    let determinant = covariance[0][0] * covariance[1][1] - covariance[0][1].powi(2);
    if determinant <= f64::EPSILON {
        return farthest_pair(points);
    }
    // In two dimensions, A = inv(covariance) / d.
    Minell::from_matrix(
        center,
        covariance[1][1] / determinant / 2.,
        -covariance[0][1] / determinant / 2.,
        covariance[0][0] / determinant / 2.,
    )
}

/// C `minell_destroy`.  Rust's owned value is released at scope end; this
/// explicit adapter is useful at FFI-shaped call sites that want source-order
/// destruction.
pub fn minell_destroy(ellipse: Minell) {
    drop(ellipse);
}

/// C `minell_scalepoints` free-function spelling.
pub fn minell_scalepoints(ellipse: &Minell, points: &mut [Point]) {
    ellipse.scale_points(points);
}

/// C `minell_rescalepoints` free-function spelling.
pub fn minell_rescalepoints(ellipse: &Minell, points: &mut [Point]) {
    ellipse.rescale_points(points);
}

fn farthest_pair(points: &[Point]) -> Minell {
    let (first, second) = points
        .iter()
        .enumerate()
        .flat_map(|(index, first)| {
            points[index + 1..]
                .iter()
                .map(move |second| (*first, *second))
        })
        .max_by(|(a0, a1), (b0, b1)| {
            (a0.x - a1.x)
                .hypot(a0.y - a1.y)
                .total_cmp(&(b0.x - b1.x).hypot(b0.y - b1.y))
        })
        .unwrap_or((points[0], points[0]));
    Minell::from_pair(first, second)
}

fn invert_3(matrix: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let determinant = matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
    if determinant.abs() <= f64::EPSILON {
        return None;
    }
    let mut result = [[0.; 3]; 3];
    for row in 0..3 {
        for column in 0..3 {
            let rows = (0..3).filter(|&value| value != column).collect::<Vec<_>>();
            let columns = (0..3).filter(|&value| value != row).collect::<Vec<_>>();
            let minor = matrix[rows[0]][columns[0]] * matrix[rows[1]][columns[1]]
                - matrix[rows[0]][columns[1]] * matrix[rows[1]][columns[0]];
            result[row][column] = if (row + column) % 2 == 0 {
                minor
            } else {
                -minor
            } / determinant;
        }
    }
    Some(result)
}

fn quadratic(matrix: [[f64; 3]; 3], value: [f64; 3]) -> f64 {
    (0..3)
        .map(|row| {
            value[row]
                * (0..3)
                    .map(|column| matrix[row][column] * value[column])
                    .sum::<f64>()
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_is_enclosed_and_scale_round_trips() {
        let input = [
            Point {
                x: 0.,
                y: 0.,
                z: 3.,
            },
            Point {
                x: 1.,
                y: 0.,
                z: 4.,
            },
            Point {
                x: 1.,
                y: 1.,
                z: 5.,
            },
            Point {
                x: 0.,
                y: 1.,
                z: 6.,
            },
        ];
        let ellipse = minell_build(&input);
        for point in input {
            assert!(ellipse.contains(point, 1.0e-7));
        }
        let mut transformed = input;
        ellipse.scale_points(&mut transformed);
        assert!(
            transformed
                .iter()
                .all(|point| point.x.hypot(point.y) <= 1.00001)
        );
        ellipse.rescale_points(&mut transformed);
        for (actual, expected) in transformed.iter().zip(input) {
            assert!((actual.x - expected.x).abs() < 1.0e-8);
            assert!((actual.y - expected.y).abs() < 1.0e-8);
            assert_eq!(actual.z, expected.z);
        }
    }

    #[test]
    fn irregular_cloud_matches_standalone_native_minell_fixture() {
        // Generated with `cc -DME_STANDALONE IMOD/libwarp/minell.c -lm` and
        // its normal seed: this cloud exercises the C implementation's
        // four-support-point conic path rather than the simple three-point
        // case.  Keep a numerical fixture here so source changes are checked
        // without requiring a C compiler at every Rust test run.
        let ellipse = minell_build(&[
            Point {
                x: 0.,
                y: 0.,
                z: 0.,
            },
            Point {
                x: 4.,
                y: 0.,
                z: 0.,
            },
            Point {
                x: 1.,
                y: 3.,
                z: 0.,
            },
            Point {
                x: 2.,
                y: 1.,
                z: 0.,
            },
            Point {
                x: -1.,
                y: 2.,
                z: 0.,
            },
        ]);
        assert!(
            (ellipse.center.x - 1.518_708_151_196_54).abs() < 5.0e-6,
            "rust ellipse: {ellipse:?}"
        );
        assert!((ellipse.center.y - 1.067_366_001_524_69).abs() < 5.0e-6);
        assert!(
            (ellipse.a - 2.706_137_897_616_87).abs() < 5.0e-6,
            "rust ellipse: {ellipse:?}"
        );
        assert!((ellipse.b - 1.724_110_625_967_31).abs() < 5.0e-6);
        assert!((ellipse.theta.to_degrees() + 26.163_449_840_648_7).abs() < 2.0e-5);
    }
}
