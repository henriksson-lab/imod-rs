use imod_rs::imod::libcfshr::amoeba::{amoeba, amoeba_init, dual_amoeba};

#[test]
fn amoeba_minimizes_a_two_variable_quadratic() {
    let mut p = [0.; 6];
    let mut y = [0.; 3];
    let mut ptol = [0.; 2];
    let mut function = |point: &[f32]| (point[0] - 3.).powi(2) + (point[1] + 2.).powi(2);
    amoeba_init(
        &mut p,
        &mut y,
        3,
        2,
        2.,
        0.0001,
        &[0., 0.],
        &[1., 1.],
        &mut function,
        &mut ptol,
    );
    let mut iter = 0;
    let mut low = 0;
    amoeba(
        &mut p,
        &mut y,
        3,
        2,
        0.00001,
        &mut function,
        &mut iter,
        &ptol,
        &mut low,
    );
    assert!(iter < 1000);
    assert!((p[low] - 3.).abs() < 0.01 && (p[low + 3] + 2.).abs() < 0.01);
}

#[test]
fn dual_amoeba_restarts_from_the_first_minimum() {
    let mut y = [0.; 21];
    let mut initial = [5.];
    let mut function = |point: &[f32]| (point[0] - 1.).powi(2);
    let mut iter = 0;
    dual_amoeba(
        &mut y,
        1,
        1.,
        &[0.01, 0.0001],
        &[0.01, 0.00001],
        &mut initial,
        &[1.],
        &mut function,
        &mut iter,
    );
    assert!(
        iter > 0 && (initial[0] - 1.).abs() < 0.01,
        "iter={iter}, result={}",
        initial[0]
    );
}
