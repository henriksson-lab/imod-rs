//! Translation of `IMOD/libcfshr/convexbound.c`.

/// Matches static `cenorder` (`IMOD/libcfshr/convexbound.c:207`).
fn cenorder(
    sx: &[f32],
    sy: &[f32],
    number_points: usize,
    indices: &mut [usize],
    center_distance: &mut [f32],
    x_center: &mut f32,
    y_center: &mut f32,
) {
    let (sum_x, sum_y) = indices[..number_points]
        .iter()
        .fold((0.0_f64, 0.0_f64), |(x, y), index| {
            (x + sx[*index] as f64, y + sy[*index] as f64)
        });
    *x_center = (sum_x / number_points as f64) as f32;
    *y_center = (sum_y / number_points as f64) as f32;
    for index in indices[..number_points].iter().copied() {
        center_distance[index] = (sx[index] - *x_center).powi(2) + (sy[index] - *y_center).powi(2);
    }
    indices[..number_points]
        .sort_by(|left, right| center_distance[*left].total_cmp(&center_distance[*right]));
}

/// Matches static `angorder` (`IMOD/libcfshr/convexbound.c:240`).
fn angorder(
    sx: &[f32],
    sy: &[f32],
    number_points: usize,
    distance_indices: &[usize],
    x_center: f32,
    y_center: f32,
    angle_indices: &mut [usize],
    point_angles: &mut [usize],
    angles: &mut [f32],
) {
    for index in 0..number_points {
        let point = distance_indices[index];
        angles[point] = (sy[point] - y_center).atan2(sx[point] - x_center);
        angle_indices[index] = point;
    }
    angle_indices[..number_points].sort_by(|left, right| angles[*left].total_cmp(&angles[*right]));
    for index in 0..number_points {
        point_angles[angle_indices[index]] = index;
    }
}

/// Matches `convexBound` (`IMOD/libcfshr/convexbound.c:41`).
pub fn convex_bound(
    sx: &[f32],
    sy_input: &[f32],
    fraction_omit: f32,
    padding: f32,
    bx: &mut [f32],
    by: &mut [f32],
    number_vertices: &mut i32,
    x_center: &mut f32,
    y_center: &mut f32,
) {
    let number_points = sx.len().min(sy_input.len());
    if number_points < 3 {
        *number_vertices = number_points as i32;
        return;
    }
    let mut center_distance = vec![0.; number_points];
    let mut angles = vec![0.; number_points];
    let mut sy = sy_input.to_vec();
    let mut distance_indices = (0..number_points).collect::<Vec<_>>();
    let mut angle_indices = vec![0; number_points];
    let mut point_angles = vec![0; number_points];
    let mut vertex = vec![false; number_points];
    let (mut standard_x, mut standard_y) = (1., 1.);
    if fraction_omit < 0. {
        let average_x = sx.iter().sum::<f32>() / number_points as f32;
        let average_y = sy_input.iter().sum::<f32>() / number_points as f32;
        standard_x = (sx
            .iter()
            .map(|value| (value - average_x).powi(2))
            .sum::<f32>()
            / number_points as f32)
            .sqrt();
        standard_y = (sy_input
            .iter()
            .map(|value| (value - average_y).powi(2))
            .sum::<f32>()
            / number_points as f32)
            .sqrt();
    }
    for index in 0..number_points {
        sy[index] = standard_x * sy_input[index] / standard_y;
    }
    cenorder(
        sx,
        &sy,
        number_points,
        &mut distance_indices,
        &mut center_distance,
        x_center,
        y_center,
    );
    let mut use_points = number_points;
    if fraction_omit != 0. {
        use_points = (number_points as i32
            - (fraction_omit.abs() * number_points as f32).round() as i32)
            .max((3).min(number_points as i32)) as usize;
        if fraction_omit < 0. {
            sy.copy_from_slice(sy_input);
        }
        cenorder(
            sx,
            &sy,
            use_points,
            &mut distance_indices,
            &mut center_distance,
            x_center,
            y_center,
        );
    }
    angorder(
        sx,
        &sy,
        use_points,
        &distance_indices,
        *x_center,
        *y_center,
        &mut angle_indices,
        &mut point_angles,
        &mut angles,
    );
    let mut minimum_y = 1.0e30_f32;
    let mut maximum_x = 0.;
    let mut start = 0;
    for point in distance_indices[..use_points].iter().copied() {
        if sy[point] < minimum_y || (sy[point] == minimum_y && sx[point] > maximum_x) {
            start = point;
            minimum_y = sy[point];
            maximum_x = sx[point];
        }
        vertex[point] = true;
    }
    let mut point_one = start;
    let mut point_two = angle_indices[(point_angles[point_one] + 1) % use_points];
    let mut point_three = angle_indices[(point_angles[point_two] + 1) % use_points];
    *number_vertices = use_points as i32;
    while point_two != start && *number_vertices > 2 {
        if sx[point_one] * (sy[point_two] - sy[point_three])
            - sx[point_two] * (sy[point_one] - sy[point_three])
            + sx[point_three] * (sy[point_one] - sy[point_two])
            > 0.
        {
            point_one = point_two;
            point_two = point_three;
            point_three = angle_indices[(point_angles[point_three] + 1) % use_points];
        } else {
            vertex[point_two] = false;
            *number_vertices -= 1;
            if point_one == start {
                point_two = point_three;
                point_three = angle_indices[(point_angles[point_three] + 1) % use_points];
            } else {
                point_two = point_one;
                loop {
                    let position = if point_angles[point_one] == 0 {
                        use_points - 1
                    } else {
                        point_angles[point_one] - 1
                    };
                    point_one = angle_indices[position];
                    if vertex[point_one] {
                        break;
                    }
                }
            }
        }
    }
    *number_vertices = 0;
    for point in angle_indices[..use_points].iter().copied() {
        if vertex[point] {
            if *number_vertices as usize >= bx.len().min(by.len()) {
                *number_vertices = -2;
                break;
            }
            let padding_fraction = if padding > 0. && center_distance[point] > 1.0e-10 {
                padding / center_distance[point].sqrt()
            } else {
                0.
            };
            let index = *number_vertices as usize;
            bx[index] = sx[point] + padding_fraction * (sx[point] - *x_center);
            by[index] = sy[point] + padding_fraction * (sy[point] - *y_center);
            *number_vertices += 1;
        }
    }
}

/// Matches Fortran wrapper `convexbound` (`IMOD/libcfshr/convexbound.c:193`).
pub fn convexbound(
    sx: &[f32],
    sy_input: &[f32],
    number_points: usize,
    fraction_omit: f32,
    padding: f32,
    bx: &mut [f32],
    by: &mut [f32],
    number_vertices: &mut i32,
    x_center: &mut f32,
    y_center: &mut f32,
    maximum_vertices: usize,
) {
    convex_bound(
        &sx[..number_points],
        &sy_input[..number_points],
        fraction_omit,
        padding,
        &mut bx[..maximum_vertices],
        &mut by[..maximum_vertices],
        number_vertices,
        x_center,
        y_center,
    )
}
