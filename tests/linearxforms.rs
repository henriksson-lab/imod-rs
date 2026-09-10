use imod_rs::imod::libcfshr::linearxforms::*;
#[test]
fn composition_and_inverse_follow_centered_transform_convention() {
    let a = [1., 0., 0., 1., 2., 3.];
    let b = [2., 0., 0., 2., 0., 0.];
    let mut p = [0.; 6];
    xf_mult(&a, &b, &mut p, 2);
    assert_eq!(xf_apply(&p, 0., 0., 1., 1., 2), (6., 8.));
    let mut i = [0.; 6];
    xf_invert(&p, &mut i, 2);
    assert_eq!(xf_apply(&i, 0., 0., 6., 8., 2), (1., 1.));
}
#[test]
fn angle_matrix_decomposition_round_trips() {
    let a = [10., 20., 30.];
    let mut m = [0.; 9];
    angles_to_matrix(&a, &mut m, 3);
    let (x, y, z) = matrix_to_angles(&m, 3).unwrap();
    assert!((x - 10.).abs() < 0.001 && (y - 20.).abs() < 0.001 && (z - 30.).abs() < 0.001);
}
