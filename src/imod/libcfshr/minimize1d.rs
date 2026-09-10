//! Translation of `IMOD/libcfshr/minimize1D.c`.
#![allow(dead_code)]

/// Original `minimize1D` (`minimize1D.c:53`).
pub unsafe fn minimize1d(
    cur_position: f32,
    cur_value: f32,
    initial_step: f32,
    scan_steps: i32,
    cuts_done: *mut i32,
    brackets: *mut f32,
    next_position: *mut f32,
) -> i32 {
    unsafe {
        let positions = brackets;
        let values = brackets.add(7);
        let walking = if *cuts_done != 0 { 0 } else { 1 };
        let mut step = initial_step;
        if *cuts_done < 0 {
            *cuts_done = 0;
            *positions = cur_position;
            *positions.add(1) = cur_position;
            *positions.add(2) = cur_position;
            *values.add(1) = cur_value;
            if scan_steps > 0 {
                *values.add(6) = 2.0;
                *next_position = cur_position + initial_step;
                *positions.add(5) = cur_position;
                *positions.add(6) = cur_position;
                *values.add(5) = cur_value;
                *values = cur_value - cur_value.abs();
            } else {
                *values.add(6) = -1.0;
                *next_position = cur_position - initial_step;
            }
            return 0;
        }
        let mut direction = (*values.add(6)).round() as i32;
        if !(-1..=2).contains(&direction) {
            return 1;
        }
        for _ in 0..*cuts_done {
            step /= 2.0;
        }
        if direction == 2 {
            let step_number = ((cur_position - *positions.add(6)) / initial_step).round() as i32;
            if step_number <= 0 || step_number > scan_steps {
                return 1;
            }
            *positions.add(3) = *positions.add(4);
            *positions.add(4) = *positions.add(5);
            *positions.add(5) = cur_position;
            *values.add(3) = *values.add(4);
            *values.add(4) = *values.add(5);
            *values.add(5) = cur_value;
            if step_number > 1 && *values.add(4) < *values.add(1) {
                for index in 0..3 {
                    *positions.add(index) = *positions.add(index + 3);
                    *values.add(index) = *values.add(index + 3);
                }
            }
            if step_number < scan_steps {
                *next_position = cur_position + initial_step;
                return 0;
            }
            if *values.add(5) < *values.add(1)
                || *values.add(3) < *values.add(1)
                || *values.add(2) < *values.add(1)
                || *values < *values.add(1)
            {
                return 2;
            }
            step /= 2.0;
            *cuts_done += 1;
            direction = if *values.add(2) > *values { -1 } else { 1 };
        } else if cur_value > *values.add(1) {
            *positions.add((direction + 1) as usize) = cur_position;
            *values.add((direction + 1) as usize) = cur_value;
            if (walking == 0
                && (*positions.add((1 - direction) as usize) - *positions.add(1)).abs()
                    < 1.1 * (cur_position - *positions.add(1)).abs())
                || (walking != 0 && (*positions.add(1) - *positions.add(2)).abs() > 0.01 * step)
            {
                step /= 2.0;
                *cuts_done += 1;
                direction = if *values.add(2) > *values { -1 } else { 1 };
            } else {
                direction *= -1;
            }
        } else {
            *positions.add((1 - direction) as usize) = *positions.add(1);
            *values.add((1 - direction) as usize) = *values.add(1);
            *positions.add(1) = cur_position;
            *values.add(1) = cur_value;
            if walking == 0 {
                step /= 2.0;
                *cuts_done += 1;
                direction = if *values.add(2) > *values { -1 } else { 1 };
            }
        }
        *next_position = *positions.add(1) + direction as f32 * step;
        *values.add(6) = direction as f32;
        0
    }
}

/// Original Fortran wrapper `minimize1d` (`minimize1D.c:174`).
pub unsafe fn minimize1d_f(
    cur_position: *mut f32,
    cur_value: *mut f32,
    initial_step: *mut f32,
    scan_steps: *mut i32,
    cuts_done: *mut i32,
    brackets: *mut f32,
    next_position: *mut f32,
) -> i32 {
    unsafe {
        minimize1d(
            *cur_position,
            *cur_value,
            *initial_step,
            *scan_steps,
            cuts_done,
            brackets,
            next_position,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_walk_state_machine_converges_by_successive_step_cuts() {
        unsafe {
            let mut cuts = -1;
            let mut brackets = [0.0_f32; 14];
            let mut position = 4.0;
            let mut next = 0.0;
            for _ in 0..12 {
                assert_eq!(
                    minimize1d(
                        position,
                        (position - 1.0).powi(2),
                        2.0,
                        0,
                        &mut cuts,
                        brackets.as_mut_ptr(),
                        &mut next
                    ),
                    0
                );
                position = next;
            }
            assert!(cuts > 0);
            assert!((brackets[1] - 1.0).abs() <= 0.25);
        }
    }
}
