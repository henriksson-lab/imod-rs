use imod_rs::imod::libcfshr::amat_to_rotmagstr::*;
#[test]
fn axial_round_trip() {
    let a = rotmag_to_amat(20., 10., 1.2, 0.3);
    let (t, d, m, s) = amat_to_rotmag(a[0], a[2], a[1], a[3]);
    assert!(
        (t - 20.).abs() < 0.001
            && (d - 10.).abs() < 0.001
            && (m - 1.2).abs() < 0.001
            && (s - 0.3).abs() < 0.001
    );
}

#[test]
fn natural_nonzero_phi_stretch_round_trip() {
    let mut input = [0.; 4];
    rotmagstr_to_amat(17., 1.1, 1.3, 28., &mut input);
    let (theta, magnification, stretch, phi) =
        amat_to_rotmagstr(input[0], input[2], input[1], input[3]);
    let mut output = [0.; 4];
    rotmagstr_to_amat(theta, magnification, stretch, phi, &mut output);
    assert!(
        input
            .iter()
            .zip(output)
            .all(|(left, right)| (left - right).abs() < 0.001)
    );
}

#[test]
fn natural_axis_inversion_round_trip() {
    let mut input = [0.0; 4];
    rotmagstr_to_amat(-31.0, 0.9, -1.4, 37.0, &mut input);
    let (theta, magnification, stretch, phi) =
        amat_to_rotmagstr(input[0], input[2], input[1], input[3]);
    // Standalone execution of IMOD/libcfshr/amat_to_rotmagstr.c on this
    // input yields these values; this exercises its double/float boundaries.
    assert!((input[0] - -0.944_151_998).abs() < 0.000_001);
    assert!((input[1] - -0.643_851_399).abs() < 0.000_001);
    assert!((input[2] - -0.829_265_118).abs() < 0.000_001);
    assert!((input[3] - 0.635_571_778).abs() < 0.000_001);
    assert!((theta - -30.999_992_4).abs() < 0.000_01);
    assert!((magnification - 0.899_999_976).abs() < 0.000_001);
    assert!((stretch - -1.399_999_86).abs() < 0.000_001);
    assert!((phi - 36.999_996_2).abs() < 0.000_01);
    assert!(
        stretch < 0.0,
        "the inversion branch must retain negative stretch"
    );
    let mut output = [0.0; 4];
    rotmagstr_to_amat(theta, magnification, stretch, phi, &mut output);
    assert!(
        input
            .iter()
            .zip(output)
            .all(|(left, right)| (left - right).abs() < 0.001),
        "the decomposed inversion must reconstruct its input matrix"
    );
}
