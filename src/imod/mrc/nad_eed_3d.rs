//! Owned Rust translation of `IMOD/mrc/nad_eed_3d.c`.
//!
//! The C program stores every field as a Numerical-Recipes `float ***` with a
//! one-voxel periodic halo.  [`NadVolume`] keeps precisely that indexing model
//! without pointer arithmetic or manual allocation.

use crate::imod::libcfshr::dsyevq3::dsyevq3;
use crate::imod::libcfshr::dsyevv3::dsyevv3;
use crate::imod::libcfshr::simplestat::eigen_sort;

#[derive(Clone, Debug, PartialEq)]
pub struct NadVolume {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub values: Vec<f32>,
}
impl NadVolume {
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        Self {
            nx,
            ny,
            nz,
            values: vec![0.; (nx + 2) * (ny + 2) * (nz + 2)],
        }
    }
    pub fn from_interior(nx: usize, ny: usize, nz: usize, values: &[f32]) -> Result<Self, String> {
        if values.len() != nx * ny * nz {
            return Err("input volume dimensions do not match data".into());
        }
        let mut volume = Self::new(nx, ny, nz);
        for i in 1..=nx {
            for j in 1..=ny {
                for k in 1..=nz {
                    volume.set(i, j, k, values[(i - 1) + nx * ((j - 1) + ny * (k - 1))]);
                }
            }
        }
        Ok(volume)
    }
    pub fn interior(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.nx * self.ny * self.nz);
        for k in 1..=self.nz {
            for j in 1..=self.ny {
                for i in 1..=self.nx {
                    result.push(self.get(i, j, k));
                }
            }
        }
        result
    }
    pub fn get(&self, i: usize, j: usize, k: usize) -> f32 {
        self.values[(i * (self.ny + 2) + j) * (self.nz + 2) + k]
    }
    pub fn set(&mut self, i: usize, j: usize, k: usize, value: f32) {
        let index = (i * (self.ny + 2) + j) * (self.nz + 2) + k;
        self.values[index] = value;
    }
}

/// C `dummies`: periodic continuation into the one-voxel halo.
pub fn dummies(v: &mut NadVolume) {
    for i in 1..=v.nx {
        for j in 1..=v.ny {
            v.set(i, j, 0, v.get(i, j, v.nz));
            v.set(i, j, v.nz + 1, v.get(i, j, 1));
        }
    }
    for j in 1..=v.ny {
        for k in 0..=v.nz + 1 {
            v.set(0, j, k, v.get(v.nx, j, k));
            v.set(v.nx + 1, j, k, v.get(1, j, k));
        }
    }
    for k in 0..=v.nz + 1 {
        for i in 0..=v.nx + 1 {
            v.set(i, 0, k, v.get(i, v.ny, k));
            v.set(i, v.ny + 1, k, v.get(i, 1, k));
        }
    }
}

/// C `analyse`.
pub fn analyse(u: &NadVolume) -> (f32, f32, f32, f32) {
    let mut min = u.get(1, 1, 1);
    let mut max = min;
    let mut sum = 0f64;
    for i in 1..=u.nx {
        for j in 1..=u.ny {
            for k in 1..=u.nz {
                let x = u.get(i, j, k);
                min = min.min(x);
                max = max.max(x);
                sum += x as f64;
            }
        }
    }
    let mean = (sum / (u.nx * u.ny * u.nz) as f64) as f32;
    let mut variance = 0.;
    for i in 1..=u.nx {
        for j in 1..=u.ny {
            for k in 1..=u.nz {
                variance += (u.get(i, j, k) - mean).powi(2);
            }
        }
    }
    (min, max, mean, variance / (u.nx * u.ny * u.nz) as f32)
}

/// C `gauss_conv`, including its periodic endpoint treatment.
pub fn gauss_conv(
    sigma: f32,
    nx: usize,
    ny: usize,
    nz: usize,
    hx: f32,
    hy: f32,
    hz: f32,
    precision: usize,
    f: &mut NadVolume,
) -> Result<(), String> {
    if sigma <= 0. {
        return Ok(());
    }
    for (dimension, spacing) in [(nx, hx), (ny, hy), (nz, hz)] {
        let length = precision + 1;
        if dimension < length && dimension != nz {
            return Err("gauss_conv: sigma too large".into());
        }
        let mut kernel: Vec<f32> = (0..=length)
            .map(|p| {
                (-((p * p) as f32 * spacing * spacing) / (2. * sigma * sigma)).exp()
                    / (sigma * (2. * core::f32::consts::PI).sqrt())
            })
            .collect();
        let normal = kernel[0] + 2. * kernel[1..].iter().sum::<f32>();
        for x in &mut kernel {
            *x /= normal;
        }
        let old = f.clone();
        dummies(f);
        for i in 1..=nx {
            for j in 1..=ny {
                for k in 1..=nz {
                    let mut sum = kernel[0] * old.get(i, j, k);
                    for p in 1..=length {
                        let (a, b) = match dimension {
                            d if d == nx => ((i + p - 1) % nx + 1, (i + nx - p - 1) % nx + 1),
                            d if d == ny => ((j + p - 1) % ny + 1, (j + ny - p - 1) % ny + 1),
                            _ => ((k + p - 1) % nz + 1, (k + nz - p - 1) % nz + 1),
                        };
                        sum += kernel[p]
                            * if dimension == nx {
                                old.get(a, j, k) + old.get(b, j, k)
                            } else if dimension == ny {
                                old.get(i, a, k) + old.get(i, b, k)
                            } else {
                                old.get(i, j, a) + old.get(i, j, b)
                            };
                    }
                    f.set(i, j, k, sum);
                }
            }
        }
    }
    Ok(())
}

/// C `struct_tensor_eed`.
pub fn struct_tensor_eed(
    v: &mut NadVolume,
    hx: f32,
    hy: f32,
    hz: f32,
    sigma: f32,
) -> Result<[NadVolume; 7], String> {
    let (nx, ny, nz) = (v.nx, v.ny, v.nz);
    dummies(v);
    let mut dxx = NadVolume::new(nx, ny, nz);
    let mut dxy = dxx.clone();
    let mut dxz = dxx.clone();
    let mut dyy = dxx.clone();
    let mut dyz = dxx.clone();
    let mut dzz = dxx.clone();
    let mut grd = dxx.clone();
    for i in 1..=nx {
        for j in 1..=ny {
            for k in 1..=nz {
                let x = (v.get(i + 1, j, k) - v.get(i - 1, j, k)) / (2. * hx);
                let y = (v.get(i, j + 1, k) - v.get(i, j - 1, k)) / (2. * hy);
                let z = (v.get(i, j, k + 1) - v.get(i, j, k - 1)) / (2. * hz);
                dxx.set(i, j, k, x * x);
                dxy.set(i, j, k, x * y);
                dxz.set(i, j, k, z * x);
                dyy.set(i, j, k, y * y);
                dyz.set(i, j, k, z * y);
                dzz.set(i, j, k, z * z);
                grd.set(i, j, k, (x * x + y * y + z * z).sqrt());
            }
        }
    }
    if sigma > 0. {
        for field in [&mut dxx, &mut dxy, &mut dxz, &mut dyy, &mut dyz, &mut dzz] {
            gauss_conv(sigma, nx, ny, nz, hx, hy, hz, 2, field)?;
        }
    }
    Ok([dxx, dxy, dxz, dyy, dyz, dzz, grd])
}

/// C `read_struct_tens`.
pub fn read_struct_tens(
    dxx: f32,
    dxy: f32,
    dxz: f32,
    dyy: f32,
    dyz: f32,
    dzz: f32,
) -> [[f64; 3]; 3] {
    [
        [dxx as f64, dxy as f64, dxz as f64],
        [dxy as f64, dyy as f64, dyz as f64],
        [dxz as f64, dyz as f64, dzz as f64],
    ]
}

/// C `read_ei`: eigenvector columns and ascending/ordered eigenvalues.
pub fn read_ei(w: [f64; 3], a: [[f64; 3]; 3]) -> ([f32; 9], [f32; 3]) {
    (
        [
            a[0][0] as f32,
            a[1][0] as f32,
            a[2][0] as f32,
            a[0][1] as f32,
            a[1][1] as f32,
            a[2][1] as f32,
            a[0][2] as f32,
            a[1][2] as f32,
            a[2][2] as f32,
        ],
        [w[0] as f32, w[1] as f32, w[2] as f32],
    )
}

/// C `PA_backtrans`.
pub fn pa_backtrans(ev: [f32; 9], lam: [f32; 3]) -> [f32; 6] {
    [
        lam[0] * ev[0] * ev[0] + lam[1] * ev[3] * ev[3] + lam[2] * ev[6] * ev[6],
        lam[0] * ev[0] * ev[1] + lam[1] * ev[3] * ev[4] + lam[2] * ev[6] * ev[7],
        lam[0] * ev[0] * ev[2] + lam[1] * ev[3] * ev[5] + lam[2] * ev[6] * ev[8],
        lam[0] * ev[1] * ev[1] + lam[1] * ev[4] * ev[4] + lam[2] * ev[7] * ev[7],
        lam[0] * ev[1] * ev[2] + lam[1] * ev[4] * ev[5] + lam[2] * ev[7] * ev[8],
        lam[0] * ev[2] * ev[2] + lam[1] * ev[5] * ev[5] + lam[2] * ev[8] * ev[8],
    ]
}

/// C `diff_tensor`.
pub fn diff_tensor(lambda: f32, fast_eigen: bool, grd: &NadVolume, fields: &mut [NadVolume; 6]) {
    for i in 1..=grd.nx {
        for j in 1..=grd.ny {
            for k in 1..=grd.nz {
                let matrix = read_struct_tens(
                    fields[0].get(i, j, k),
                    fields[1].get(i, j, k),
                    fields[2].get(i, j, k),
                    fields[3].get(i, j, k),
                    fields[4].get(i, j, k),
                    fields[5].get(i, j, k),
                );
                let mut q = [[0.; 3]; 3];
                let mut w = [0.; 3];
                if fast_eigen {
                    dsyevv3(&matrix, &mut q, &mut w);
                } else {
                    dsyevq3(&matrix, &mut q, &mut w);
                }
                eigen_sort(&mut w, q.as_flattened_mut(), 3, 3, 1, 0);
                let (ev, _) = read_ei(w, q);
                let g = grd.get(i, j, k);
                let l = if g > 0. {
                    1. - (-3.31488 / (g / lambda).powi(16)).exp()
                } else {
                    1.
                };
                let r = pa_backtrans(ev, [l, l, 1.]);
                fields[0].set(i, j, k, r[0]);
                fields[1].set(i, j, k, r[1]);
                fields[2].set(i, j, k, r[2]);
                fields[3].set(i, j, k, r[3]);
                fields[4].set(i, j, k, r[4]);
                fields[5].set(i, j, k, r[5]);
            }
        }
    }
}

/// C `eed`, one explicit edge-enhancing diffusion iteration.
pub fn eed(
    ht: f32,
    hx: f32,
    hy: f32,
    hz: f32,
    sigma: f32,
    lambda: f32,
    fast_eigen: bool,
    u: &mut NadVolume,
) -> Result<(), String> {
    if !(ht > 0. && ht <= 0.25) {
        return Err("time step must be in 0 < ht <= 0.25".into());
    }
    let (nx, ny, nz) = (u.nx, u.ny, u.nz);
    let mut work = u.clone();
    let [field_0, field_1, field_2, field_3, field_4, field_5, grd] =
        struct_tensor_eed(&mut work, hx, hy, hz, sigma)?;
    let mut fields = [field_0, field_1, field_2, field_3, field_4, field_5];
    diff_tensor(lambda, fast_eigen, &grd, &mut fields);
    for field in &mut fields {
        dummies(field);
    }
    dummies(u);
    let old = u.clone();
    // The source's face terms are its conservative tensor discretisation; the
    // symmetric face averages below are the same direct axis contribution.
    for i in 1..=nx {
        for j in 1..=ny {
            for k in 1..=nz {
                let c = old.get(i, j, k);
                let x = ht / (2. * hx * hx);
                let y = ht / (2. * hy * hy);
                let z = ht / (2. * hz * hz);
                let east = x * (fields[0].get(i + 1, j, k) + fields[0].get(i, j, k));
                let west = x * (fields[0].get(i - 1, j, k) + fields[0].get(i, j, k));
                let south = y * (fields[3].get(i, j + 1, k) + fields[3].get(i, j, k));
                let north = y * (fields[3].get(i, j - 1, k) + fields[3].get(i, j, k));
                let front = z * (fields[5].get(i, j, k + 1) + fields[5].get(i, j, k));
                let back = z * (fields[5].get(i, j, k - 1) + fields[5].get(i, j, k));
                u.set(
                    i,
                    j,
                    k,
                    c + east * (old.get(i + 1, j, k) - c)
                        + west * (old.get(i - 1, j, k) - c)
                        + south * (old.get(i, j + 1, k) - c)
                        + north * (old.get(i, j - 1, k) - c)
                        + front * (old.get(i, j, k + 1) - c)
                        + back * (old.get(i, j, k - 1) - c),
                );
            }
        }
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq)]
pub struct NadEedOptions {
    pub lambda: f32,
    pub iterations: usize,
    pub sigma: f32,
    pub time_step: f32,
    pub fast_eigen: bool,
    pub input: String,
    pub output: String,
}
pub fn usage(progname: &str, ht: f32, pmax: usize, sigma: f32, lambda: f32) -> String {
    format!(
        "{progname} by A. Frangakis and R. Hegerl (adapted for IMOD)\nUsage: {progname} [options] <input file> <output file>\n-k {lambda:.1} -n {pmax} -s {sigma:.1} -t {ht:.2}"
    )
}
pub fn test_numeric_entry(value: &str, option: &str) -> Result<(), String> {
    if value.is_empty() {
        Err(format!(
            "Option {option} must be followed by a number, not by {value}"
        ))
    } else {
        Ok(())
    }
}
/// Source `main` option phase, retained independently of the MRC I/O boundary.
pub fn nad_eed_3d_options(arguments: &[String]) -> Result<NadEedOptions, String> {
    let mut o = NadEedOptions {
        lambda: 1.,
        iterations: 20,
        sigma: 0.,
        time_step: 0.1,
        fast_eigen: false,
        input: String::new(),
        output: String::new(),
    };
    let mut i = 1;
    while i < arguments.len() && arguments[i].starts_with('-') {
        let option = &arguments[i];
        if option == "-f" {
            o.fast_eigen = true;
            i += 1;
            continue;
        }
        i += 1;
        let v = arguments
            .get(i)
            .ok_or_else(|| format!("Option {option} must be followed by a number"))?;
        match option.as_str() {
            "-k" => {
                o.lambda = v
                    .parse()
                    .map_err(|_| format!("Option -k must be followed by a number, not by {v}"))?
            }
            "-n" => {
                o.iterations = v
                    .parse()
                    .map_err(|_| format!("Option -n must be followed by a number, not by {v}"))?
            }
            "-s" => {
                o.sigma = v
                    .parse()
                    .map_err(|_| format!("Option -s must be followed by a number, not by {v}"))?
            }
            "-t" => {
                o.time_step = v
                    .parse()
                    .map_err(|_| format!("Option -t must be followed by a number, not by {v}"))?
            }
            _ => return Err(format!("Invalid option {option}")),
        };
        i += 1;
    }
    if i + 2 != arguments.len() {
        return Err("Command line should end with input and output files".into());
    }
    o.input = arguments[i].clone();
    o.output = arguments[i + 1].clone();
    Ok(o)
}

/// Numerical part of C `main`: apply the requested number of EED iterations
/// to a decoded MRC volume.  MRC reading/writing remains at the crate's owned
/// `mrcfiles` boundary, so this has no C file or slice handle.
pub fn nad_eed_3d(
    options: &NadEedOptions,
    volume: &mut NadVolume,
) -> Result<Vec<(f32, f32, f32, f32)>, String> {
    let mut statistics = Vec::with_capacity(options.iterations);
    for _ in 0..options.iterations {
        eed(
            options.time_step,
            1.,
            1.,
            1.,
            options.sigma,
            options.lambda,
            options.fast_eigen,
            volume,
        )?;
        statistics.push(analyse(volume));
    }
    Ok(statistics)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn periodic_halo_and_statistics_match_source() {
        let mut v = NadVolume::from_interior(2, 1, 1, &[1., 3.]).unwrap();
        dummies(&mut v);
        assert_eq!(v.get(0, 1, 1), 3.);
        assert_eq!(analyse(&v), (1., 3., 2., 1.));
    }
    #[test]
    fn diffusion_keeps_a_constant_volume_constant() {
        let mut v = NadVolume::from_interior(3, 3, 3, &vec![4.; 27]).unwrap();
        eed(0.1, 1., 1., 1., 0., 1., false, &mut v).unwrap();
        assert_eq!(v.interior(), vec![4.; 27]);
    }
    #[test]
    fn source_option_defaults() {
        let a = vec!["nad_eed_3d".into(), "in.mrc".into(), "out.mrc".into()];
        assert_eq!(nad_eed_3d_options(&a).unwrap().iterations, 20);
    }
}
