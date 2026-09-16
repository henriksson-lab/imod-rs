//! Translation of `IMOD/libcfshr/minimize1D.c`.
#![allow(dead_code)]
pub const MINIMIZE1D_SOURCE_FUNCTIONS: &[&str] = &["minimize1D", "minimize1d"];

/// Original `minimize1D` (`minimize1D.c:53`).
pub fn minimize1d(
    cur_position: f32,
    cur_value: f32,
    initial_step: f32,
    scan_steps: i32,
    cuts_done: &mut i32,
    brackets: &mut [f32],
    next_position: &mut f32,
) -> i32 {
    if brackets.len() < 14 {
        return 1;
    }
    /* `float *positions = brackets; float *values = brackets + 7;` — the two
    C pointers are disjoint halves of the caller's 14-element array. */
    let (positions, values) = brackets.split_at_mut(7);
    let walking = if *cuts_done != 0 { 0 } else { 1 };
    let mut step = initial_step;
    if *cuts_done < 0 {
        *cuts_done = 0;
        positions[0] = cur_position;
        positions[1] = cur_position;
        positions[2] = cur_position;
        values[1] = cur_value;
        if scan_steps > 0 {
            values[6] = 2.0;
            *next_position = cur_position + initial_step;
            positions[5] = cur_position;
            positions[6] = cur_position;
            values[5] = cur_value;
            values[0] = cur_value - cur_value.abs();
        } else {
            values[6] = -1.0;
            *next_position = cur_position - initial_step;
        }
        return 0;
    }
    /* B3DNINT(a) is (int)floor((a) + 0.5) in double. */
    let mut direction = (values[6] as f64 + 0.5).floor() as i32;
    if !(-1..=2).contains(&direction) {
        return 1;
    }
    for _ in 0..*cuts_done {
        step /= 2.0;
    }
    if direction == 2 {
        let step_number =
            (((cur_position - positions[6]) / initial_step) as f64 + 0.5).floor() as i32;
        if step_number <= 0 || step_number > scan_steps {
            return 1;
        }
        positions[3] = positions[4];
        positions[4] = positions[5];
        positions[5] = cur_position;
        values[3] = values[4];
        values[4] = values[5];
        values[5] = cur_value;
        if step_number > 1 && values[4] < values[1] {
            for index in 0..3 {
                positions[index] = positions[index + 3];
                values[index] = values[index + 3];
            }
        }
        if step_number < scan_steps {
            *next_position = cur_position + initial_step;
            return 0;
        }
        if values[5] < values[1]
            || values[3] < values[1]
            || values[2] < values[1]
            || values[0] < values[1]
        {
            return 2;
        }
        step /= 2.0;
        *cuts_done += 1;
        direction = if values[2] > values[0] { -1 } else { 1 };
    } else if cur_value > values[1] {
        positions[(direction + 1) as usize] = cur_position;
        values[(direction + 1) as usize] = cur_value;
        /* fabs() promotes its float argument, so both comparisons run in double. */
        if (walking == 0
            && ((positions[(1 - direction) as usize] - positions[1]) as f64).abs()
                < 1.1 * ((cur_position - positions[1]) as f64).abs())
            || (walking != 0 && ((positions[1] - positions[2]) as f64).abs() > 0.01 * step as f64)
        {
            step /= 2.0;
            *cuts_done += 1;
            direction = if values[2] > values[0] { -1 } else { 1 };
        } else {
            direction *= -1;
        }
    } else {
        positions[(1 - direction) as usize] = positions[1];
        values[(1 - direction) as usize] = values[1];
        positions[1] = cur_position;
        values[1] = cur_value;
        if walking == 0 {
            step /= 2.0;
            *cuts_done += 1;
            direction = if values[2] > values[0] { -1 } else { 1 };
        }
    }
    *next_position = positions[1] + direction as f32 * step;
    values[6] = direction as f32;
    0
}

/// Original Fortran wrapper `minimize1d` (`minimize1D.c:174`).
pub fn minimize1d_f(
    cur_position: &f32,
    cur_value: &f32,
    initial_step: &f32,
    scan_steps: &i32,
    cuts_done: &mut i32,
    brackets: &mut [f32],
    next_position: &mut f32,
) -> i32 {
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_walk_state_machine_converges_by_successive_step_cuts() {
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
                    &mut brackets,
                    &mut next
                ),
                0
            );
            position = next;
        }
        assert!(cuts > 0);
        assert!((brackets[1] - 1.0).abs() <= 0.25);
    }
    #[test]
    fn source_bracket_precondition_is_safe_in_rust() {
        let mut cuts = -1;
        let mut next = 0.;
        assert_eq!(
            minimize1d(0., 0., 1., 0, &mut cuts, &mut [0.; 13], &mut next),
            1
        );
        assert_eq!(MINIMIZE1D_SOURCE_FUNCTIONS.len(), 2);
    }
}
