use imod_rs::imod::libcfshr::convexbound::convex_bound;

#[test]
fn convex_boundary_omits_an_interior_point_in_angle_order() {
    let x = [0., 4., 4., 0., 2.];
    let y = [0., 0., 3., 3., 1.5];
    let mut bx = [0.; 8];
    let mut by = [0.; 8];
    let mut vertices = 0;
    let mut xc = 0.;
    let mut yc = 0.;
    convex_bound(
        &x,
        &y,
        0.,
        0.,
        &mut bx,
        &mut by,
        &mut vertices,
        &mut xc,
        &mut yc,
    );
    assert_eq!(vertices, 4);
    assert_eq!((xc, yc), (2., 1.5));
    assert!(
        bx[..vertices as usize]
            .iter()
            .zip(&by[..vertices as usize])
            .all(|(x, y)| (*x == 0. || *x == 4.) && (*y == 0. || *y == 3.))
    );
}

#[test]
fn convex_boundary_reports_too_small_output_arrays() {
    let x = [0., 4., 4., 0.];
    let y = [0., 0., 3., 3.];
    let mut bx = [0.; 3];
    let mut by = [0.; 3];
    let mut vertices = 0;
    let mut xc = 0.;
    let mut yc = 0.;
    convex_bound(
        &x,
        &y,
        0.,
        0.,
        &mut bx,
        &mut by,
        &mut vertices,
        &mut xc,
        &mut yc,
    );
    assert_eq!(vertices, -2);
}
