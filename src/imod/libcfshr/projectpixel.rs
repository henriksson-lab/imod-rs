//! Translation of `IMOD/libcfshr/projectpixel.c`.

/// `b3dutil.h:68`: `#define RADIANS_PER_DEGREE 0.01745329252` -- a **double**
/// literal, so `angle * RADIANS_PER_DEGREE` promotes and `cos`/`sin` are the
/// double libm routines; only their results narrow to the `float` locals.
const RADIANS_PER_DEGREE: f64 = 0.01745329252;

// Operand types throughout follow the C: every local is `float`, every bare
// literal (`0.5`, `2.`, `0.`) is `double`, so a float-by-float product stays
// float and only the addition of a literal widens.  `B3DMAX(a, b)` is
// `a > b ? a : b`, `B3DMIN(a, b)` is `a < b ? a : b`, `ACCUM_MAX(m, v)` is
// `m = m > v ? m : v` (`b3dutil.h:29-41`): each returns the **second** operand
// when either is NaN, which `f32::max`/`min` do not.

/// Original `maxCornerDistFromCen` (`projectpixel.c:25`).
pub fn max_corner_dist_from_cen(angle: f32, dist: &mut f32) {
    let cos_ang = (angle as f64 * RADIANS_PER_DEGREE).cos() as f32;
    let sin_ang = (angle as f64 * RADIANS_PER_DEGREE).sin() as f32;
    let c1 = (0.5 * cos_ang as f64 + 0.5 * sin_ang as f64) as f32;
    let c2 = (0.5 * cos_ang as f64 - 0.5 * sin_ang as f64) as f32;
    let c3 = (-0.5 * cos_ang as f64 + 0.5 * sin_ang as f64) as f32;
    let c4 = (-0.5 * cos_ang as f64 - 0.5 * sin_ang as f64) as f32;
    *dist = if c1 > c2 { c1 } else { c2 };
    *dist = if *dist > c3 { *dist } else { c3 };
    *dist = if *dist > c4 { *dist } else { c4 };
}

/// Original Fortran wrapper `maxcornerdistfromcen` (`projectpixel.c:39`).
pub fn maxcornerdistfromcen(angle: &f32, dist: &mut f32) {
    max_corner_dist_from_cen(*angle, dist);
}

/// Original `rayLengthAtDistFromCen` (`projectpixel.c:47`).
pub fn ray_length_at_dist_from_cen(cos_ang: f32, sin_ang: f32, dist: f32, length: &mut f32) {
    let txp = (((dist * cos_ang) as f64 + 0.5) / sin_ang as f64) as f32;
    let txm = (((dist * cos_ang) as f64 - 0.5) / sin_ang as f64) as f32;
    let txmin = if txp < txm { txp } else { txm };
    let txmax = if txp > txm { txp } else { txm };
    let typ = (((-dist * sin_ang) as f64 + 0.5) / cos_ang as f64) as f32;
    let tym = (((-dist * sin_ang) as f64 - 0.5) / cos_ang as f64) as f32;
    let tymin = if typ < tym { typ } else { tym };
    let tymax = if typ > tym { typ } else { tym };
    let tmin = if txmin > tymin { txmin } else { tymin };
    let tmax = if txmax < tymax { txmax } else { tymax };
    // `B3DMAX(0., tmax - tmin)`: a float difference compared against a double
    // `0.`; widening a float is exact, so the comparison and the narrowed
    // result are the float ones, with the ternary's NaN/-0 operand choice.
    let diff = tmax - tmin;
    *length = (if 0. > diff as f64 { 0. } else { diff as f64 }) as f32;
}

/// Original `rayPixelIntersectingArea` (`projectpixel.c:72`).
pub fn ray_pixel_intersecting_area(
    angle: f32,
    cen_to_cen_dist: f32,
    ray_width: i32,
    del_dist: f32,
    area: &mut f32,
) {
    let cos_ang = (angle as f64 * RADIANS_PER_DEGREE).cos() as f32;
    let sin_ang = (angle as f64 * RADIANS_PER_DEGREE).sin() as f32;
    let mut max_corn_dist = 0.;
    max_corner_dist_from_cen(angle, &mut max_corn_dist);
    let mut left_dist = (cen_to_cen_dist as f64 - ray_width as f64 / 2.) as f32;
    let mut right_dist = (cen_to_cen_dist as f64 + ray_width as f64 / 2.) as f32;
    *area = 0.;
    if right_dist <= -max_corn_dist || left_dist >= max_corn_dist {
        return;
    }
    if left_dist <= -max_corn_dist && right_dist >= max_corn_dist {
        *area = 1.;
        return;
    }
    left_dist = if left_dist > -max_corn_dist {
        left_dist
    } else {
        -max_corn_dist
    };
    right_dist = if right_dist < max_corn_dist {
        right_dist
    } else {
        max_corn_dist
    };
    let num_rays = ((right_dist - left_dist) / del_dist) as i32;
    for ind in 0..num_rays {
        let dist = (left_dist as f64 + (ind as f64 + 0.5) * del_dist as f64) as f32;
        let mut length = 1.;
        if cos_ang != 0. && sin_ang != 0. {
            ray_length_at_dist_from_cen(cos_ang, sin_ang, dist, &mut length);
        }
        *area += length;
    }
    *area *= del_dist;
    let frac = (right_dist - left_dist) - del_dist * num_rays as f32;
    let dist = ((left_dist + num_rays as f32 * del_dist) as f64 + 0.5 * frac as f64) as f32;
    let mut length = 1.;
    if cos_ang != 0. && sin_ang != 0. {
        ray_length_at_dist_from_cen(cos_ang, sin_ang, dist, &mut length);
    }
    *area += length * frac;
}

/// Original `makeRayAreaLookupTable` (`projectpixel.c:124`).
pub fn make_ray_area_lookup_table(
    angle: f32,
    ray_width: i32,
    num_dists: i32,
    del_for_area: f32,
    del_ray_ind: &mut [i32],
    num_rays_hit: &mut [i32],
    areas: &mut [f32],
) {
    let del_table = ray_width as f32 / num_dists as f32;
    let mut max_dist = 0.;
    max_corner_dist_from_cen(angle, &mut max_dist);
    for index in 0..num_dists as usize {
        let dist = ((index as f64 + 0.5) * del_table as f64 - 0.5 * ray_width as f64) as f32;
        del_ray_ind[index] = 0;
        num_rays_hit[index] = 0;
        if dist as f64 - ray_width as f64 / 2. > (-max_dist) as f64 {
            del_ray_ind[index] = -1;
            num_rays_hit[index] = 1;
            ray_pixel_intersecting_area(
                angle,
                dist - ray_width as f32,
                ray_width,
                del_for_area,
                &mut areas[3 * index],
            );
        }
        let area_index = 3 * index + num_rays_hit[index] as usize;
        ray_pixel_intersecting_area(angle, dist, ray_width, del_for_area, &mut areas[area_index]);
        num_rays_hit[index] += 1;
        if dist as f64 + ray_width as f64 / 2. < max_dist as f64 {
            let area_index = 3 * index + num_rays_hit[index] as usize;
            ray_pixel_intersecting_area(
                angle,
                dist + ray_width as f32,
                ray_width,
                del_for_area,
                &mut areas[area_index],
            );
            num_rays_hit[index] += 1;
        }
    }
}

/// Original Fortran wrapper `makerayarealookuptable` (`projectpixel.c:159`).
pub fn makerayarealookuptable(
    angle: &f32,
    ray_width: &i32,
    num_dists: &i32,
    del_for_area: &f32,
    del_ray_ind: &mut [i32],
    num_rays_hit: &mut [i32],
    areas: &mut [f32],
) {
    make_ray_area_lookup_table(
        *angle,
        *ray_width,
        *num_dists,
        *del_for_area,
        del_ray_ind,
        num_rays_hit,
        areas,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_ray_area_and_lookup_cases() {
        let mut distance = 0.;
        max_corner_dist_from_cen(0., &mut distance);
        assert_eq!(distance, 0.5);
        maxcornerdistfromcen(&45., &mut distance);
        assert!((distance - 2_f32.sqrt() / 2.).abs() < 1.0e-6);
        let mut length = 0.;
        ray_length_at_dist_from_cen(2_f32.sqrt() / 2., 2_f32.sqrt() / 2., 0., &mut length);
        assert!((length - 2_f32.sqrt()).abs() < 1.0e-6);
        let mut area = 0.;
        ray_pixel_intersecting_area(0., 0., 1, 0.01, &mut area);
        assert!((area - 1.).abs() < 1.0e-5);
        ray_pixel_intersecting_area(0., 5., 1, 0.01, &mut area);
        assert_eq!(area, 0.);
        let mut deltas = [0; 4];
        let mut hits = [0; 4];
        let mut areas = [0.; 12];
        make_ray_area_lookup_table(30., 2, 4, 0.001, &mut deltas, &mut hits, &mut areas);
        for index in 0..4 {
            let sum: f32 = areas[3 * index..3 * index + hits[index] as usize]
                .iter()
                .sum();
            assert!((sum - 1.).abs() < 0.01);
        }
        makerayarealookuptable(&30., &2, &4, &0.001, &mut deltas, &mut hits, &mut areas);
    }
}
