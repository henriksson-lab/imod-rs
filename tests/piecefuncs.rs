use imod_rs::imod::libcfshr::piecefuncs::{
    adjust_piece_overlap, check_piece_list, fill_list_of_piece_z, read_piece_list,
};

#[test]
fn reads_real_piece_coordinate_fixture_and_preserves_source_spacing_rules() {
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/piece-list.txt");
    let mut x = [0; 16];
    let mut y = [0; 16];
    let mut z = [0; 16];
    let mut count = 0;
    assert_eq!(
        read_piece_list(fixture.to_str(), &mut x, &mut y, &mut z, &mut count, 16),
        0
    );
    assert_eq!(count, 9);
    assert_eq!(
        (&x[..3], &y[3..6], &z[6..9]),
        (&[0, 960, 1920][..], &[960, 960, 960][..], &[1, 1, 1][..])
    );
    let mut minimum = 0;
    let mut pieces = 0;
    let mut overlap = 0;
    assert_eq!(
        check_piece_list(
            &x,
            1,
            count as usize,
            1,
            1024,
            &mut minimum,
            &mut pieces,
            &mut overlap
        ),
        0
    );
    assert_eq!((minimum, pieces, overlap), (0, 3, 64));
    let mut list_z = [0; 16];
    let mut number_z = 0;
    fill_list_of_piece_z(&z[..count as usize], &mut list_z, &mut number_z);
    assert_eq!((&list_z[..number_z], number_z), (&[0, 1][..], 2));
}

#[test]
fn adjusts_piece_overlap_in_source_coordinate_order() {
    let mut pieces = [0, 960, 1920];
    adjust_piece_overlap(&mut pieces, 1, 3, 1024, 0, 64, 128);
    assert_eq!(pieces, [0, 896, 1792]);
}
