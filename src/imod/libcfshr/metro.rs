//! Translation of `IMOD/libcfshr/metro.c`.
#![allow(dead_code, unsafe_op_in_unsafe_fn)]

use super::b3dutil::num_omp_threads;

/// C `MetroFunct` callback type (`cfsemshare.h`).
pub type MetroFunct = unsafe extern "C" fn(*mut i32, *mut f32, *mut f32, *mut f32);

/// C static `copyArray` (`metro.c:10`).
pub unsafe fn copy_array(to_arr: *mut f32, to1: i32, to2: i32, from_arr: *mut f32, from1: i32) {
    core::ptr::copy_nonoverlapping(
        from_arr.add((from1 - 1) as usize),
        to_arr.add((to1 - 1) as usize),
        (to2 + 1 - to1) as usize,
    );
}

/// C static `deltaXlength` (`metro.c:334`).
pub unsafe fn delta_xlength(
    hessian: *mut f32,
    argument: *mut f32,
    argument_base: i32,
    n: i32,
) -> f64 {
    let mut length = 0.;
    for j in 1..=n {
        let delta = *argument.add((j - 1) as usize) as f64
            - *hessian.add((argument_base + j - 1) as usize) as f64;
        length += delta * delta;
    }
    length.sqrt()
}

/// C static `dotProduct` (`metro.c:348`).
pub unsafe fn dot_product(a: *mut f32, b: *mut f32, n: i32) -> f64 {
    let mut product = 0.;
    for j in 1..=n {
        product += *a.add((j - 1) as usize) as f64 * *b.add((j - 1) as usize) as f64;
    }
    product
}

/// C `metroSearch` (`metro.c:58`).
pub unsafe fn metro_search(
    n: i32,
    argument: *mut f32,
    funct: MetroFunct,
    function: *mut f32,
    gradient: *mut f32,
    step_initial: f32,
    epsilon: f32,
    limit_in: i32,
    ier: *mut i32,
    hessian: *mut f32,
    num_iter: *mut i32,
    rms_scale: f32,
) {
    const BREAK_SUMS: usize = 8;
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
    funct(&n as *const i32 as *mut i32, argument, function, gradient);
    let mut reinitialize = true;
    while *num_iter <= iter_limit {
        if reinitialize {
            let gdotg = dot_product(gradient, gradient, n);
            core::ptr::write_bytes(hessian, 0, (n * n) as usize);
            for j in 1..=n {
                *hessian.add(((j - 1) * n + j - 1) as usize) =
                    (1. / (gdotg / n as f64).sqrt()) as f32;
            }
        }
        *num_iter += 1;
        // The C diagnostic has no effect on its minimization state.
        let _ = rms_scale;
        copy_array(hessian, argument_base + 1, argument_base + n, argument, 1);
        copy_array(hessian, gradient_base + 1, gradient_base + n, gradient, 1);
        for j in 1..=n {
            *hessian.add((direction_base + j - 1) as usize) =
                -dot_product(hessian.add(((j - 1) * n) as usize), gradient, n) as f32;
        }
        let mut f_new = *function as f64;
        let gdotg = dot_product(gradient, gradient, n);
        let mut dir_dot_g_new = dot_product(hessian.add(direction_base as usize), gradient, n);
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
            *argument.add(j as usize) += step * *hessian.add((direction_base + j) as usize);
        }
        funct(&n as *const i32 as *mut i32, argument, function, gradient);
        f_new = *function as f64;
        dir_dot_g_new = dot_product(hessian.add(direction_base as usize), gradient, n);
        let mut del_x_dot_g = 0.;
        for j in 1..=n {
            let delta = *argument.add((j - 1) as usize) as f64
                - *hessian.add((argument_base + j - 1) as usize) as f64;
            del_x_dot_g += delta * *hessian.add((gradient_base + j - 1) as usize) as f64;
        }
        if del_x_dot_g >= 0. {
            if delta_xlength(hessian, argument, argument_base, n) < epsilon as f64 / 5. {
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
                    *argument.add(j as usize) -=
                        backoff_step as f32 * *hessian.add((direction_base + j) as usize);
                }
                funct(&n as *const i32 as *mut i32, argument, function, gradient);
                cut_by_ten = *function as f64 > f_old || *function as f64 > f_new;
            }
            if cut_by_ten {
                step *= 0.1;
                if step / step_initial < 1.0e-6 {
                    if delta_xlength(hessian, argument, argument_base, n) < epsilon as f64 {
                        return;
                    }
                    *ier = 2;
                    return;
                }
                copy_array(argument, 1, n, hessian, argument_base + 1);
                copy_array(gradient, 1, n, hessian, gradient_base + 1);
                *function = f_old as f32;
                continue;
            } else {
                step *= (1. - backoff_step).max(0.1) as f32;
            }
        }
        let mut del_x_gamma = 0.;
        let mut del_x_length = 0.;
        del_x_dot_g = 0.;
        for j in 1..=n {
            let dx = *argument.add((j - 1) as usize) as f64
                - *hessian.add((argument_base + j - 1) as usize) as f64;
            let dg = *gradient.add((j - 1) as usize) as f64
                - *hessian.add((gradient_base + j - 1) as usize) as f64;
            del_x_dot_g += dx * *hessian.add((gradient_base + j - 1) as usize) as f64;
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
        for j in 1..=n {
            let ind = (j as usize) % BREAK_SUMS;
            let mut sig = 0.;
            let mut hdot_gamma = 0.;
            for k in 1..=n {
                let h = *hessian.add(((j - 1) * n + k - 1) as usize) as f64;
                sig += h * *gradient.add((k - 1) as usize) as f64;
                hdot_gamma += h
                    * (*gradient.add((k - 1) as usize) as f64
                        - *hessian.add((gradient_base + k - 1) as usize) as f64);
            }
            *hessian.add((hgamma_base + j - 1) as usize) = hdot_gamma as f32;
            thr_ghg[ind] += (*gradient.add((j - 1) as usize) as f64
                - *hessian.add((gradient_base + j - 1) as usize) as f64)
                * hdot_gamma;
            thr_posdef[ind] += *gradient.add((j - 1) as usize) as f64 * sig;
        }
        let gamma_hgamma: f64 = thr_ghg[..BREAK_SUMS * num_threads as usize].iter().sum();
        let posdef: f64 = thr_posdef[..BREAK_SUMS * num_threads as usize].iter().sum();
        if posdef < 0. || gamma_hgamma < 0. {
            *ier = 4;
            return;
        }
        for j in 1..=n {
            for k in 1..=n {
                let dx = (*argument.add((j - 1) as usize) as f64
                    - *hessian.add((argument_base + j - 1) as usize) as f64)
                    * (*argument.add((k - 1) as usize) as f64
                        - *hessian.add((argument_base + k - 1) as usize) as f64);
                let hgh = *hessian.add((hgamma_base + j - 1) as usize) as f64
                    * *hessian.add((hgamma_base + k - 1) as usize) as f64;
                *hessian.add(((j - 1) * n + k - 1) as usize) +=
                    (dx / del_x_gamma - hgh / gamma_hgamma) as f32;
            }
        }
        if del_x_gamma < gamma_hgamma {
            continue;
        }
        let sqrt_ghg = gamma_hgamma.sqrt();
        for j in 1..=n {
            let nuj = sqrt_ghg
                * ((*argument.add((j - 1) as usize) as f64
                    - *hessian.add((argument_base + j - 1) as usize) as f64)
                    / del_x_gamma
                    - *hessian.add((hgamma_base + j - 1) as usize) as f64 / gamma_hgamma);
            for k in 1..=n {
                let nuk = sqrt_ghg
                    * ((*argument.add((k - 1) as usize) as f64
                        - *hessian.add((argument_base + k - 1) as usize) as f64)
                        / del_x_gamma
                        - *hessian.add((hgamma_base + k - 1) as usize) as f64 / gamma_hgamma);
                *hessian.add(((j - 1) * n + k - 1) as usize) += (nuj * nuk) as f32;
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
    metro_search(
        *n,
        argument,
        funct,
        function,
        gradient,
        *step_initial,
        *epsilon,
        *limit_in,
        ier,
        hessian,
        num_iter,
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
        unsafe {
            metro_search(
                2,
                x.as_mut_ptr(),
                quadratic,
                &mut f,
                g.as_mut_ptr(),
                0.1,
                1.0e-5,
                100,
                &mut error,
                h.as_mut_ptr(),
                &mut iterations,
                0.,
            );
        }
        assert_eq!(error, 3);
        assert!(f < 1.0e-6, "f {f}, x {x:?}");
        assert!((x[0] - 1.).abs() < 2.0e-4 && (x[1] - 2.).abs() < 2.0e-4);
    }
    #[test]
    fn dot_and_delta_are_double_precision_source_operations() {
        let mut a = [1., 2.];
        let mut b = [3., 4.];
        let mut h = [0.; 6];
        h[4] = 0.;
        h[5] = 0.;
        unsafe {
            assert_eq!(dot_product(a.as_mut_ptr(), b.as_mut_ptr(), 2), 11.);
            assert!(
                (delta_xlength(h.as_mut_ptr(), a.as_mut_ptr(), 4, 2) - 5_f64.sqrt()).abs()
                    < 1.0e-10
            );
        }
    }
}
