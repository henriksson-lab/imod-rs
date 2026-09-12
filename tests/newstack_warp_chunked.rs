//! `newstack -distort`, `-gradient` and a warping `-xform` on the **chunked**
//! route: the same `warpInterp` correction as `tests/newstack_warp.rs`, but
//! with a memory limit small enough that the output is written in pieces.
//!
//! The grid itself is fetched once per section, before the chunk loop
//! (`newstack.f90:2059-2090`), and it is the chunk loop that moves it: the
//! grid start comes down by the first loaded *input* line when undistorting
//! and by the first *output* line when warping (`newstack.f90:2419-2427`),
//! `ycenIn` is measured from the same loaded window (`newstack.f90:2059`), and
//! the window is the one the source carries across chunks -- including the one
//! `scanSection` leaves behind when it scanned for the fill mean
//! (`newstack.f90:2317-2320, 3437, 3463-3464`).
//!
//! The 64x48x3 volume the other warp test uses never chunks under `-memory`,
//! whose smallest accepted limit is 39 MB, so this one authors a 1200x900x3
//! float volume -- 1,080,000 pixels a section -- from the same expression
//! `fixtures/make-newstack-warp-big-inputs.py` writes, and drives it with
//! `-test`, the option whose whole purpose is "to test the code for reading in
//! and binning images in chunks".  Every value is a multiple of 0.25 in
//! 60..240, so it is exact in `float` and the script and this file agree
//! bit-for-bit without depending on either runtime's libm.
//!
//! `fixtures/newstack-warp-big.{idf,mgt,xf}` are that script's grids, scaled to
//! the larger image.  The goldens are the reference `newstack`'s own `-verbose 1`
//! stdout with its `loadtime` line masked -- that line is wall-clock and differs
//! between two *native* runs -- and the SHA-256 of its output file with the
//! label block (bytes 224..1024) zeroed, which is the documented
//! non-achievable: `mrc_head_new` never clears `labels`, and the date stamp
//! moves by a second between runs.  The output is too large to commit, so the
//! digest stands in for it; `fixtures/FIXTURE-MANIFEST.md` records how it was
//! taken.

mod common;

const AUTODOC: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc");

#[test]
fn warping_corrections_on_the_chunked_route_match_the_reference() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let dir = std::env::temp_dir().join(format!("imod-rs-nswarpchunk-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for (name, target) in [
        ("newstack-warp-big.idf", "w.idf"),
        ("newstack-warp-big.mgt", "w.mgt"),
        ("newstack-warp-big.xf", "w.xf"),
        ("newstack-warp-big-sparse.idf", "ws.idf"),
    ] {
        std::fs::copy(root.join("fixtures").join(name), dir.join(target)).unwrap();
    }

    //
    // The volume, written exactly as `make-newstack-warp-big-inputs.py` writes
    // it: an MRC 2 header with one label, then the sections.
    //
    let (nx, ny, nz) = (1200_i32, 900_i32, 3_i32);
    let mut header = vec![0_u8; 1024];
    let mut put_i32 = |bytes: &mut Vec<u8>, at: usize, value: i32| {
        bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
    };
    for (at, value) in [
        (0, nx),
        (4, ny),
        (8, nz),
        (12, 2),
        (28, nx),
        (32, ny),
        (36, nz),
        (64, 1),
        (68, 2),
        (72, 3),
        (220, 1),
    ] {
        put_i32(&mut header, at, value);
    }
    for (at, value) in [
        (40, nx as f32),
        (44, ny as f32),
        (48, nz as f32),
        (52, 90.0),
        (56, 90.0),
        (60, 90.0),
    ] {
        header[at..at + 4].copy_from_slice(&f32::to_le_bytes(value));
    }
    header[208..212].copy_from_slice(b"MAP ");
    header[212..216].copy_from_slice(&[68, 65, 0, 0]);
    let label = b"newstack chunked warping fixture";
    header[224..224 + label.len()].copy_from_slice(label);
    let mut pixels = Vec::<u8>::with_capacity((nx * ny * nz) as usize * 4);
    let (mut amin, mut amax, mut asum) = (1.0e37_f32, -1.0e37_f32, 0.0_f64);
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let value = 100.0_f32
                    + 0.25 * (((ix * 7 + iy * 13 + iz * 29) % 401) as f32)
                    + 0.5 * ((ix % 53) as f32)
                    - 0.75 * ((iy % 71) as f32)
                    + 3.0 * (iz as f32);
                amin = amin.min(value);
                amax = amax.max(value);
                asum += f64::from(value);
                pixels.extend_from_slice(&value.to_le_bytes());
            }
        }
    }
    header[76..80].copy_from_slice(&amin.to_le_bytes());
    header[80..84].copy_from_slice(&amax.to_le_bytes());
    header[84..88].copy_from_slice(&((asum / f64::from(nx * ny * nz)) as f32).to_le_bytes());
    let mut volume = header;
    volume.extend_from_slice(&pixels);
    std::fs::write(dir.join("in.mrc"), &volume).unwrap();

    // SHA-256 (FIPS 180-4) of the label-masked output, so a 12.9 MB golden
    // does not have to be committed.
    let sha256 = |data: &[u8]| -> String {
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2,
        ];
        let mut h: [u32; 8] = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
            0x5be0cd19,
        ];
        let mut message = data.to_vec();
        let bits = (data.len() as u64) * 8;
        message.push(0x80);
        while message.len() % 64 != 56 {
            message.push(0);
        }
        message.extend_from_slice(&bits.to_be_bytes());
        for block in message.chunks_exact(64) {
            let mut w = [0_u32; 64];
            for (index, word) in block.chunks_exact(4).enumerate() {
                w[index] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
            }
            for index in 16..64 {
                let s0 = w[index - 15].rotate_right(7)
                    ^ w[index - 15].rotate_right(18)
                    ^ (w[index - 15] >> 3);
                let s1 = w[index - 2].rotate_right(17)
                    ^ w[index - 2].rotate_right(19)
                    ^ (w[index - 2] >> 10);
                w[index] = w[index - 16]
                    .wrapping_add(s0)
                    .wrapping_add(w[index - 7])
                    .wrapping_add(s1);
            }
            let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
                (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
            for index in 0..64 {
                let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
                let ch = (e & f) ^ ((!e) & g);
                let t1 = hh
                    .wrapping_add(s1)
                    .wrapping_add(ch)
                    .wrapping_add(K[index])
                    .wrapping_add(w[index]);
                let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
                let maj = (a & b) ^ (a & c) ^ (b & c);
                let t2 = s0.wrapping_add(maj);
                hh = g;
                g = f;
                f = e;
                e = d.wrapping_add(t1);
                d = c;
                c = b;
                b = a;
                a = t1.wrapping_add(t2);
            }
            for (slot, value) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
                *slot = slot.wrapping_add(value);
            }
        }
        h.iter().map(|word| format!("{word:08x}")).collect()
    };

    let digests = std::fs::read_to_string(root.join("fixtures/newstack-warp-big-outputs.txt"))
        .expect("the reference output digests must be present");

    //
    // Two chunks, three, and on up; `-test 1080001,1` is the pair that leaves
    // `scanSection`'s own window covering the first chunk, and `-test
    // 1200000,1` is the whole-input/chunked-output branch at
    // `newstack.f90:2209-2233` rather than the search below it.
    //
    for (name, limit, chunks, args) in [
        ("distort2", "1620001,1", 2, vec!["-distort", "w.idf"]),
        ("distort3", "1080001,1", 3, vec!["-distort", "w.idf"]),
        ("distort8", "300000,1", 8, vec!["-distort", "w.idf"]),
        ("distort10", "1200000,1", 10, vec!["-distort", "w.idf"]),
        ("gradient3", "1000000,1", 3, vec!["-gradient", "w.mgt"]),
        ("gradient16", "150000,1", 16, vec!["-gradient", "w.mgt"]),
        ("xfwarp2", "1620001,1", 2, vec!["-xform", "w.xf"]),
        ("xfwarp3", "1000000,1", 3, vec!["-xform", "w.xf"]),
        ("xfwarp9", "300000,1", 9, vec!["-xform", "w.xf"]),
        // `newstack-warp-big-sparse.idf` covers only 0..510 of the 1200
        // columns, so `getSizeAdjustedGrid` has to run
        // `expandAndExtrapGrid` (`warputils.c:707`) over most of the image
        // rather than use the grid as it stands.  That path is where the
        // `float`-versus-`double` widths inside `extrapolateGrid`
        // (`warputils.c:436, 489-495, 511-516, 583`) decide which blocks the
        // neighbour search finds, and getting them wrong moved 22113 bytes of
        // this output.
        ("distortsparse", "300000,1", 8, vec!["-distort", "ws.idf"]),
    ] {
        let output = common::imod_cmd("newstack")
            .current_dir(&dir)
            .env("AUTODOC_DIR", AUTODOC)
            .args(&args)
            .args([
                "-mode", "2", "-verbose", "1", "-test", limit, "-input", "in.mrc", "-output",
                "o.mrc",
            ])
            .output()
            .expect("newstack executable must start");
        assert!(
            output.status.success(),
            "{name}: newstack failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            output.stderr.is_empty(),
            "{name}: nothing goes to stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let got_text = String::from_utf8_lossy(&output.stdout)
            .lines()
            .map(|line| {
                if line.starts_with("loadtime ") {
                    "loadtime MASKED".to_string()
                } else {
                    line.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        assert!(
            got_text.contains(&format!(" number of chunks: {chunks:>11}")),
            "{name}: the run must actually reach {chunks} chunks, got:\n{got_text}"
        );
        let want_text =
            std::fs::read_to_string(root.join(format!("fixtures/newstack-warp-big-{name}.txt")))
                .unwrap();
        assert_eq!(
            got_text, want_text,
            "{name}: stdout must match the reference run"
        );
        let mut got = std::fs::read(dir.join("o.mrc")).unwrap();
        for byte in &mut got[224..1024] {
            *byte = 0;
        }
        let want = digests
            .lines()
            .find_map(|line| line.strip_prefix(&format!("{name} ")))
            .expect("a digest for every case");
        assert_eq!(
            sha256(&got),
            want,
            "{name}: the output file must match the reference byte for byte"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}
