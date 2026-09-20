//! Translation of `IMOD/libcfshr/projectpixel.c`.

const RADIANS_PER_DEGREE: f32 = core::f32::consts::PI / 180.;

/// Original `maxCornerDistFromCen` (`projectpixel.c:25`).
pub fn max_corner_dist_from_cen(angle: f32, dist: &mut f32) {
    let cosine = (angle * RADIANS_PER_DEGREE).cos();
    let sine = (angle * RADIANS_PER_DEGREE).sin();
    let c1 = 0.5 * cosine + 0.5 * sine;
    let c2 = 0.5 * cosine - 0.5 * sine;
    let c3 = -0.5 * cosine + 0.5 * sine;
    let c4 = -0.5 * cosine - 0.5 * sine;
    *dist = c1.max(c2);
    *dist = (*dist).max(c3);
    *dist = (*dist).max(c4);
}

/// Original Fortran wrapper `maxcornerdistfromcen` (`projectpixel.c:39`).
pub fn maxcornerdistfromcen(angle: &f32, dist: &mut f32) {
    max_corner_dist_from_cen(*angle, dist);
}

/// Original `rayLengthAtDistFromCen` (`projectpixel.c:47`).
pub fn ray_length_at_dist_from_cen(cosine: f32, sine: f32, dist: f32, length: &mut f32) {
    let txp = (dist * cosine + 0.5) / sine;
    let txm = (dist * cosine - 0.5) / sine;
    let txmin = txp.min(txm);
    let txmax = txp.max(txm);
    let typ = (-dist * sine + 0.5) / cosine;
    let tym = (-dist * sine - 0.5) / cosine;
    let tymin = typ.min(tym);
    let tymax = typ.max(tym);
    *length = 0_f32.max(txmax.min(tymax) - txmin.max(tymin));
}

/// Original `rayPixelIntersectingArea` (`projectpixel.c:72`).
pub fn ray_pixel_intersecting_area(
    angle: f32,
    cen_to_cen_dist: f32,
    ray_width: i32,
    del_dist: f32,
    area: &mut f32,
) {
    let cosine = (angle * RADIANS_PER_DEGREE).cos();
    let sine = (angle * RADIANS_PER_DEGREE).sin();
    let mut max_corner_dist = 0.;
    max_corner_dist_from_cen(angle, &mut max_corner_dist);
    let mut left_dist = cen_to_cen_dist - ray_width as f32 / 2.;
    let mut right_dist = cen_to_cen_dist + ray_width as f32 / 2.;
    *area = 0.;
    if right_dist <= -max_corner_dist || left_dist >= max_corner_dist {
        return;
    }
    if left_dist <= -max_corner_dist && right_dist >= max_corner_dist {
        *area = 1.;
        return;
    }
    left_dist = left_dist.max(-max_corner_dist);
    right_dist = right_dist.min(max_corner_dist);
    let num_rays = ((right_dist - left_dist) / del_dist) as i32;
    for index in 0..num_rays {
        let dist = left_dist + (index as f32 + 0.5) * del_dist;
        let mut length = 1.;
        if cosine != 0. && sine != 0. {
            ray_length_at_dist_from_cen(cosine, sine, dist, &mut length);
        }
        *area += length;
    }
    *area *= del_dist;
    let fraction = (right_dist - left_dist) - del_dist * num_rays as f32;
    let dist = left_dist + num_rays as f32 * del_dist + 0.5 * fraction;
    let mut length = 1.;
    if cosine != 0. && sine != 0. {
        ray_length_at_dist_from_cen(cosine, sine, dist, &mut length);
    }
    *area += length * fraction;
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
        let dist = (index as f32 + 0.5) * del_table - 0.5 * ray_width as f32;
        del_ray_ind[index] = 0;
        num_rays_hit[index] = 0;
        if dist - ray_width as f32 / 2. > -max_dist {
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
        if dist + ray_width as f32 / 2. < max_dist {
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
