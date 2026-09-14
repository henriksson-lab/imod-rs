//! Translation of `IMOD/libcfshr/metro.c`.
#![allow(dead_code, unsafe_op_in_unsafe_fn)]

use super::b3dutil::num_omp_threads;

/// C `MetroFunct` callback type (`cfsemshare.h`).
pub type MetroFunct = unsafe extern "C" fn(*mut i32, *mut f32, *mut f32, *mut f32);

/// C static `copyArray` (`metro.c:10`).
pub fn copy_array(to: &mut [f32], from: &[f32]) {
    to.copy_from_slice(from);
}

/// C static `deltaXlength` (`metro.c:334`).
pub fn delta_xlength(hessian: &[f32], argument: &[f32], argument_base: usize) -> f64 {
    argument
        .iter()
        .zip(&hessian[argument_base..argument_base + argument.len()])
        .map(|(&argument, &base)| {
            let delta = f64::from(argument) - f64::from(base);
            delta * delta
        })
        .sum::<f64>()
        .sqrt()
}

/// C static `dotProduct` (`metro.c:348`).
pub fn dot_product(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&a, &b)| f64::from(a) * f64::from(b))
        .sum()
}

/// C `metroSearch` (`metro.c:58`).
pub fn metro_search(
    n: usize,
    argument: &mut [f32],
    funct: MetroFunct,
    function: &mut f32,
    gradient: &mut [f32],
    step_initial: f32,
    epsilon: f32,
    limit_in: i32,
    ier: &mut i32,
    hessian: &mut [f32],
    num_iter: &mut i32,
    rms_scale: f32,
) {
    const BREAK_SUMS: usize = 8;
    assert!(argument.len() >= n && gradient.len() >= n);
    assert!(hessian.len() >= n * n + 3 * n);
    let argument = &mut argument[..n];
    let gradient = &mut gradient[..n];
    let mut num_threads = ((n as f32 / 150.).round() as i32).max(1);
    num_threads = num_omp_threads(num_threads).clamp(1, 4);
    let mu = 1.0e-4_f64;
    let argument_base = n * n;
    let gradient_base = n * n + n;
    let direction_base = n * n + 2 * n;
    let hgamma_base = direction_base;
    *ier = 0;
    *num_iter = 0;
    let iter_limit = limit_in.abs();
    let mut step = step_initial;
    let n_i32 = n as i32;
    unsafe {
        funct(
            &n_i32 as *const i32 as *mut i32,
            argument.as_mut_ptr(),
            function,
            gradient.as_mut_ptr(),
        )
    };
    let mut reinitialize = true;
    while *num_iter <= iter_limit {
        if reinitialize {
            let gdotg = dot_product(gradient, gradient);
            hessian[..n * n].fill(0.);
            for j in 0..n {
                hessian[j * n + j] = (1. / (gdotg / n as f64).sqrt()) as f32;
            }
        }
        *num_iter += 1;
        // The C diagnostic has no effect on its minimization state.
        let _ = rms_scale;
        hessian[argument_base..argument_base + n].copy_from_slice(argument);
        hessian[gradient_base..gradient_base + n].copy_from_slice(gradient);
        for j in 0..n {
            hessian[direction_base + j] =
                -dot_product(&hessian[j * n..(j + 1) * n], gradient) as f32;
        }
        let mut f_new = *function as f64;
        let gdotg = dot_product(gradient, gradient);
        let mut dir_dot_g_new = dot_product(&hessian[direction_base..direction_base + n], gradient);
        if gdotg == 0. {
            return;
        }
        reinitialize = dir_dot_g_new >= 0.;
        if reinitialize {
            continue;
        }
        let f_old = f_new;
        let dir_dot_g_old = dir_dot_g_new;
        for j in 0..n {
            argument[j] += step * hessian[direction_base + j];
        }
        unsafe {
            funct(
                &n_i32 as *const i32 as *mut i32,
                argument.as_mut_ptr(),
                function,
                gradient.as_mut_ptr(),
            )
        };
        f_new = *function as f64;
        dir_dot_g_new = dot_product(&hessian[direction_base..direction_base + n], gradient);
        let mut del_x_dot_g = 0.;
        for j in 0..n {
            let delta = f64::from(argument[j]) - f64::from(hessian[argument_base + j]);
            del_x_dot_g += delta * f64::from(hessian[gradient_base + j]);
        }
        if del_x_dot_g >= 0. {
            if delta_xlength(hessian, argument, argument_base) < epsilon as f64 / 5. {
                return;
            }
            *ier = 1;
            return;
        }
        if (f_new - f_old) / del_x_dot_g >= mu {
            step = step_initial;
        } else {
            let mut cut_by_ten = dir_dot_g_new < 0. && f_new < f_old;
            let mut backoff_step = 0.;
            if !cut_by_ten {
                let step_double = step as f64;
                let z = 3. * (f_old - f_new) / step_double + dir_dot_g_old + dir_dot_g_new;
                let w = (z * z - dir_dot_g_old * dir_dot_g_new).sqrt();
                backoff_step = step_double * (dir_dot_g_new + w - z)
                    / (dir_dot_g_new - dir_dot_g_old + 2. * w);
                if backoff_step.abs() > step_double {
                    backoff_step *= 0.90 * step_double / backoff_step.abs();
                }
                for j in 0..n {
                    argument[j] -= backoff_step as f32 * hessian[direction_base + j];
                }
                unsafe {
                    funct(
                        &n_i32 as *const i32 as *mut i32,
                        argument.as_mut_ptr(),
                        function,
                        gradient.as_mut_ptr(),
                    )
                };
                cut_by_ten = *function as f64 > f_old || *function as f64 > f_new;
            }
            if cut_by_ten {
                step *= 0.1;
                if step / step_initial < 1.0e-6 {
                    if delta_xlength(hessian, argument, argument_base) < epsilon as f64 {
                        return;
                    }
                    *ier = 2;
                    return;
                }
                copy_array(argument, &hessian[argument_base..argument_base + n]);
                copy_array(gradient, &hessian[gradient_base..gradient_base + n]);
                *function = f_old as f32;
                continue;
            } else {
                step *= (1. - backoff_step).max(0.1) as f32;
            }
        }
        let mut del_x_gamma = 0.;
        let mut del_x_length = 0.;
        del_x_dot_g = 0.;
        for j in 0..n {
            let dx = f64::from(argument[j]) - f64::from(hessian[argument_base + j]);
            let dg = f64::from(gradient[j]) - f64::from(hessian[gradient_base + j]);
            del_x_dot_g += dx * f64::from(hessian[gradient_base + j]);
            del_x_gamma += dx * dg;
            del_x_length += dx * dx;
        }
        if del_x_length.sqrt() <= epsilon as f64 {
            return;
        }
        if del_x_dot_g >= 0. {
            *ier = 1;
            return;
        }
        if *num_iter >= iter_limit {
            *ier = 3;
            return;
        }
        let mut thr_ghg = [0_f64; BREAK_SUMS * 4];
        let mut thr_posdef = [0_f64; BREAK_SUMS * 4];
        // `num_omp_threads` is the translated no-OpenMP implementation (one thread), so this retains source summation buckets.
        for j in 0..n {
            let ind = (j + 1) % BREAK_SUMS;
            let mut sig = 0.;
            let mut hdot_gamma = 0.;
            for k in 0..n {
                let h = f64::from(hessian[j * n + k]);
                sig += h * f64::from(gradient[k]);
                hdot_gamma += h * (f64::from(gradient[k]) - f64::from(hessian[gradient_base + k]));
            }
            hessian[hgamma_base + j] = hdot_gamma as f32;
            thr_ghg[ind] +=
                (f64::from(gradient[j]) - f64::from(hessian[gradient_base + j])) * hdot_gamma;
            thr_posdef[ind] += f64::from(gradient[j]) * sig;
        }
        let gamma_hgamma: f64 = thr_ghg[..BREAK_SUMS * num_threads as usize].iter().sum();
        let posdef: f64 = thr_posdef[..BREAK_SUMS * num_threads as usize].iter().sum();
        if posdef < 0. || gamma_hgamma < 0. {
            *ier = 4;
            return;
        }
        for j in 0..n {
            for k in 0..n {
                let dx = (f64::from(argument[j]) - f64::from(hessian[argument_base + j]))
                    * (f64::from(argument[k]) - f64::from(hessian[argument_base + k]));
                let hgh = f64::from(hessian[hgamma_base + j]) * f64::from(hessian[hgamma_base + k]);
                hessian[j * n + k] += (dx / del_x_gamma - hgh / gamma_hgamma) as f32;
            }
        }
        if del_x_gamma < gamma_hgamma {
            continue;
        }
        let sqrt_ghg = gamma_hgamma.sqrt();
        for j in 0..n {
            let nuj = sqrt_ghg
                * ((f64::from(argument[j]) - f64::from(hessian[argument_base + j])) / del_x_gamma
                    - f64::from(hessian[hgamma_base + j]) / gamma_hgamma);
            for k in 0..n {
                let nuk = sqrt_ghg
                    * ((f64::from(argument[k]) - f64::from(hessian[argument_base + k]))
                        / del_x_gamma
                        - f64::from(hessian[hgamma_base + k]) / gamma_hgamma);
                hessian[j * n + k] += (nuj * nuk) as f32;
            }
        }
    }
    *ier = 3;
}

/// C Fortran wrapper `metro` (`metro.c:365`).
pub unsafe fn metro(
    n: *mut i32,
    argument: *mut f32,
    funct: MetroFunct,
    function: *mut f32,
    gradient: *mut f32,
    step_initial: *mut f32,
    epsilon: *mut f32,
    limit_in: *mut i32,
    ier: *mut i32,
    hessian: *mut f32,
    num_iter: *mut i32,
    rms_scale: *mut f32,
) {
    let n_value = *n as usize;
    metro_search(
        n_value,
        core::slice::from_raw_parts_mut(argument, n_value),
        funct,
        &mut *function,
        core::slice::from_raw_parts_mut(gradient, n_value),
        *step_initial,
        *epsilon,
        *limit_in,
        &mut *ier,
        core::slice::from_raw_parts_mut(hessian, n_value * n_value + 3 * n_value),
        &mut *num_iter,
        *rms_scale,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    unsafe extern "C" fn quadratic(n: *mut i32, x: *mut f32, f: *mut f32, g: *mut f32) {
        *f = 0.;
        for i in 0..*n {
            let d = *x.add(i as usize) - (i + 1) as f32;
            *f += d * d;
            *g.add(i as usize) = 2. * d;
        }
    }
    #[test]
    fn metric_search_reaches_source_cycle_limit_near_quadratic_minimum() {
        let mut x = [5., -3.];
        let mut f = 0.;
        let mut g = [0.; 2];
        let mut h = [0.; 10];
        let mut error = 0;
        let mut iterations = 0;
        metro_search(
            2,
            &mut x,
            quadratic,
            &mut f,
            &mut g,
            0.1,
            1.0e-5,
            100,
            &mut error,
            &mut h,
            &mut iterations,
            0.,
        );
        assert_eq!(error, 3);
        assert!(f < 1.0e-6, "f {f}, x {x:?}");
        assert!((x[0] - 1.).abs() < 2.0e-4 && (x[1] - 2.).abs() < 2.0e-4);
    }
    #[test]
    fn dot_and_delta_are_double_precision_source_operations() {
        let a = [1., 2.];
        let b = [3., 4.];
        let mut h = [0.; 6];
        h[4] = 0.;
        h[5] = 0.;
        assert_eq!(dot_product(&a, &b), 11.);
        assert!((delta_xlength(&h, &a, 4) - 5_f64.sqrt()).abs() < 1.0e-10);
    }
}
