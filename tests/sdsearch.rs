use imod_rs::imod::libcfshr::sdsearch::*;
#[test]
fn sd_and_density_are_zero_for_identical_images() {
    let a = (0..25).map(|v| v as f32).collect::<Vec<_>>();
    let (sd, d) = mont_sd_calc(&a, &a, 5, 5, 0, 0, 4, 4, 0., 0.);
    assert_eq!((sd, d), (0., 0.));
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
    assert!((x - 1.).abs() < 0.1 && y.abs() < 0.1);
}
