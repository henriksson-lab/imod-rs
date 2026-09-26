//! `tif2mrc` on a 1-bit (bilevel) TIFF, the defined behaviour for `BUGS.md` §5.
//!
//! libtiff-based `iiTIFFCheck` refuses bilevel data, so `tif2mrc` falls back to
//! the old reader in `mrc/tiff.c`, whose `pixSize = BitsPerSample / 8` is 0:
//! native reads the strips with a zero element size and expands into a
//! zero-byte allocation (garbage output or an abort in `free()`).  Fixed in
//! translation: the source's 1-bit branch is carried out as intended, each bit
//! becoming one byte pixel, 0xff for a set bit and 0x00 for a clear one, with
//! every row starting on its own byte as TIFF requires.

mod common;

/// A little-endian 13 x 5 uncompressed 1-bit TIFF with an explicit
/// BitsPerSample = 1; rows are packed MSB first, 2 bytes per row.
fn one_bit_tiff(bits: &[[bool; 13]; 5]) -> Vec<u8> {
    let mut out = b"II*\0".to_vec();
    out.extend_from_slice(&8u32.to_le_bytes());
    let entries: [(u16, u16, u32); 9] = [
        (256, 4, 13), // ImageWidth
        (257, 4, 5),  // ImageLength
        (258, 3, 1),  // BitsPerSample
        (259, 3, 1),  // Compression: none
        (262, 3, 1),  // Photometric: min-is-black
        (273, 4, 0),  // StripOffsets, patched below
        (277, 3, 1),  // SamplesPerPixel
        (278, 4, 5),  // RowsPerStrip
        (279, 4, 10), // StripByteCounts
    ];
    let data_offset = 8 + 2 + entries.len() as u32 * 12 + 4;
    out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    for (tag, typ, value) in entries {
        out.extend_from_slice(&tag.to_le_bytes());
        out.extend_from_slice(&typ.to_le_bytes());
        out.extend_from_slice(&1u32.to_le_bytes());
        let v = if tag == 273 { data_offset } else { value };
        if typ == 3 {
            out.extend_from_slice(&(v as u16).to_le_bytes());
            out.extend_from_slice(&[0, 0]);
        } else {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out.extend_from_slice(&0u32.to_le_bytes());
    for row in bits {
        let mut packed = [0u8; 2];
        for (x, &bit) in row.iter().enumerate() {
            if bit {
                packed[x / 8] |= 0x80 >> (x % 8);
            }
        }
        out.extend_from_slice(&packed);
    }
    out
}

#[test]
fn tif2mrc_expands_one_bit_tiff_to_bytes() {
    let dir = std::env::temp_dir().join(format!("imod-rs-tif2mrc-1bit-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut bits = [[false; 13]; 5];
    let mut state: u32 = 7;
    for row in bits.iter_mut() {
        for bit in row.iter_mut() {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
            *bit = (state >> 16) & 1 == 1;
        }
    }
    std::fs::write(dir.join("bit.tif"), one_bit_tiff(&bits)).unwrap();
    let output = common::imod_cmd("tif2mrc")
        .args(["bit.tif", "bit.mrc"])
        .current_dir(&dir)
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");

    let b = std::fs::read(dir.join("bit.mrc")).unwrap();
    let int = |at: usize| i32::from_le_bytes(b[at..at + 4].try_into().unwrap());
    assert_eq!((int(0), int(4), int(8), int(12)), (13, 5, 1, 0));
    let data = &b[1024 + int(92) as usize..][..65];
    // Clear bits and set bits become the byte values 0 and 255 (stored as
    // whatever the byte-mode convention writes for them); tif2mrc flips Y.
    let (sx, sy) = (0..65)
        .map(|i| (i % 13, i / 13))
        .find(|&(x, y)| bits[y][x])
        .unwrap();
    let set_value = data[(4 - sy) * 13 + sx];
    let set = |v: u8| v == set_value;
    let mut distinct = data.to_vec();
    distinct.sort();
    distinct.dedup();
    assert_eq!(distinct.len(), 2);
    for y in 0..5 {
        for x in 0..13 {
            assert_eq!(set(data[(4 - y) * 13 + x]), bits[y][x], "pixel ({x}, {y})");
        }
    }
    // Header min/max span the expanded 0..255 (bytes 76..84: amin, amax; a
    // signed-byte file stores them shifted by -128).
    let float = |at: usize| f32::from_le_bytes(b[at..at + 4].try_into().unwrap());
    assert_eq!(float(80) - float(76), 255.0);
    let _ = std::fs::remove_dir_all(&dir);
    common::remove_command_links();
}
