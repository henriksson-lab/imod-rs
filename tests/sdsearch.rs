use imod_rs::imod::libcfshr::sdsearch::*;
#[test]
fn sd_and_density_are_zero_for_identical_images() {
    let a = (0..25).map(|v| v as f32).collect::<Vec<_>>();
    let (mut sd, mut d) = (-1., -1.);
    mont_sd_calc(&a, &a, 5, 5, 0, 0, 4, 4, 0., 0., &mut sd, &mut d);
    assert_eq!((sd, d), (0., 0.));
}
#[test]
fn empty_comparison_leaves_dden_unchanged() {
    // montSdCalc writes *dden only when nsum > 0 (sdsearch.c:308-309); a box
    // displaced entirely off the image compares nothing.  Native: sd 9999,
    // dden untouched.
    let a = (0..25).map(|v| v as f32).collect::<Vec<_>>();
    let (mut sd, mut d) = (-1., 42.5);
    mont_sd_calc(&a, &a, 5, 5, 0, 0, 4, 4, 10., 0., &mut sd, &mut d);
    assert_eq!((sd, d), (9999., 42.5));
}
#[test]
fn search_finds_integer_translation() {
    let a = (0..49)
        .map(|v| ((v % 7) * (v % 7) + 3 * (v / 7) * (v / 7)) as f32)
        .collect::<Vec<_>>();
    let mut b = vec![0.; 49];
    for y in 0..7 {
        for x in 0..6 {
            b[x + 1 + y * 7] = a[x + y * 7];
        }
    }
    let (mut x, mut y, mut sd, mut d) = (0., 0., 0., 0.);
    mont_big_search(
        &a, &b, 7, 7, 1, 1, 5, 5, &mut x, &mut y, &mut sd, &mut d, 1, 2,
    );
    // Values from the reference libcfshr.so's montBigSearch.
    assert_eq!((x, y, sd, d), (1., 0., 0., 0.));
}
