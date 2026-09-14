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
#[test]
fn icalc_angles_leaves_its_output_untouched_when_the_matrix_is_not_a_rotation() {
    // `linearxforms.c:302-323`: `icalc_angles` returns `void`.  On a matrix whose
    // determinant is not near 1 it never assigns `angles`; it prints the matrix,
    // the determinant and `ERROR: icalc_angles - Not a pure rotation matrix` to
    // stdout instead.  The printing is the only failure signal a caller gets, so
    // an `angles` array that came back changed would mean the guard was skipped.
    let mut angles = [-1.5f32, 2.5, -3.5];
    icalc_angles(&mut angles, &[2., 0., 0., 0., 2., 0., 0., 0., 2.]);
    assert_eq!(angles, [-1.5f32, 2.5, -3.5]);

    // `linearxforms.c:226`: the determinant it prints is the file-scope `sDet`
    // after `sDet -= 1.0`, so for this 2I matrix it is 8 - 1 = 7.
    assert_eq!(S_DET.get(), 7.0);

    // And the success path does write, through the same entry point.
    let mut m = [0.; 9];
    angles_to_matrix(&[10., 20., 30.], &mut m, 3);
    icalc_angles(&mut angles, &m);
    assert!(
        (angles[0] - 10.).abs() < 0.001
            && (angles[1] - 20.).abs() < 0.001
            && (angles[2] - 30.).abs() < 0.001
    );
}
