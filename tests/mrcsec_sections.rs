//! `mrcsec.c` section-reading parity across every storage mode it converts.
//!
//! The goldens are produced by `fixtures/mrcsec-driver.c`, a C program linked
//! against the reference `libiimod`, which calls the same fifteen entry points
//! in the same order on the same fixture and prints a rolling hash of each
//! result.  The fixtures are authored by `fixtures/make-mrcsec-inputs.py`
//! straight from the MRC layout; their 21x13x4 size is odd on purpose so the
//! sub-rectangle reads are unaligned.

use imod_rs::imod::libiimod::mrcfiles::{MrcHeader, mrc_head_read, mrc_init_li};
use imod_rs::imod::libiimod::mrcsec::{
    mrc_read_section, mrc_read_section_byte, mrc_read_section_float, mrc_read_section_ushort,
    mrc_read_y, mrc_read_y_byte, mrc_read_y_float, mrc_read_y_ushort, mrc_read_z, mrc_read_z_byte,
    mrc_read_z_float, mrc_read_z_ushort,
};
use std::ffi::CString;
use std::fmt::Write as _;

fn dump(out: &mut String, tag: &str, buf: &[u8], n: usize, esize: usize) {
    let mut sum: u64 = 0;
    for byte in &buf[..n * esize] {
        sum = sum.wrapping_mul(131).wrapping_add(u64::from(*byte));
    }
    let first: String = buf[..8.min(n * esize)]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    let _ = writeln!(out, "{tag} n={n} sum={sum} first={first}");
}

#[test]
fn every_section_reader_matches_the_reference_for_every_mode() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    for mode in [0, 1, 2, 4, 6] {
        let path = root.join(format!("fixtures/mrcsec-mode{mode}.mrc"));
        let want =
            std::fs::read_to_string(root.join(format!("fixtures/mrcsec-mode{mode}.out.txt")))
                .unwrap();
        let mut got = String::new();
        unsafe {
            let name = CString::new(path.to_string_lossy().as_bytes()).unwrap();
            let mut fp =
                imod_rs::imod::libcfshr::b3dutil::ImodFile::open(&name.to_string_lossy(), "rb")
                    .unwrap();
            let mut header = MrcHeader::default();
            assert_eq!(
                mrc_head_read(&mut fp, &mut header),
                0,
                "mode {mode}: header"
            );
            header.fp = Some(fp.clone());
            let (nx, ny, nz) = (header.nx, header.ny, header.nz);
            let z = nz / 2;
            let mut li = std::mem::zeroed();
            mrc_init_li(Some(&mut li), None);
            mrc_init_li(Some(&mut li), Some(&header));
            let _ = writeln!(
                got,
                "li xmin {} xmax {} ymin {} ymax {} mode {}",
                li.xmin, li.xmax, li.ymin, li.ymax, header.mode
            );
            let mut buf = vec![0u8; nx as usize * ny.max(nz) as usize * 16 + 4096];
            let np = (nx * ny) as usize;
            let nyz = (nx * nz) as usize;

            macro_rules! step {
                ($tag:expr, $n:expr, $esize:expr, $call:expr) => {{
                    buf.iter_mut().for_each(|v| *v = 0);
                    let rc = $call;
                    let _ = write!(got, "rc={rc} ");
                    dump(&mut got, $tag, &buf, $n, $esize);
                }};
            }

            step!(
                "sec",
                np,
                1,
                mrc_read_section(&mut header, &mut li, buf.as_mut_ptr(), z)
            );
            step!(
                "secB",
                np,
                1,
                mrc_read_section_byte(&mut header, &mut li, buf.as_mut_ptr(), z)
            );
            step!(
                "secU",
                np,
                2,
                mrc_read_section_ushort(&mut header, &mut li, buf.as_mut_ptr(), z)
            );
            step!(
                "secF",
                np,
                4,
                mrc_read_section_float(&mut header, &mut li, buf.as_mut_ptr().cast(), z)
            );
            step!(
                "z",
                np,
                1,
                mrc_read_z(&mut header, &mut li, buf.as_mut_ptr(), z)
            );
            step!(
                "zB",
                np,
                1,
                mrc_read_z_byte(&mut header, &mut li, buf.as_mut_ptr(), z)
            );
            step!(
                "zU",
                np,
                2,
                mrc_read_z_ushort(&mut header, &mut li, buf.as_mut_ptr(), z)
            );
            step!(
                "zF",
                np,
                4,
                mrc_read_z_float(&mut header, &mut li, buf.as_mut_ptr().cast(), z)
            );
            step!(
                "y",
                nyz,
                1,
                mrc_read_y(&mut header, &mut li, buf.as_mut_ptr(), 1)
            );
            step!(
                "yB",
                nyz,
                1,
                mrc_read_y_byte(&mut header, &mut li, buf.as_mut_ptr(), 1)
            );
            step!(
                "yU",
                nyz,
                2,
                mrc_read_y_ushort(&mut header, &mut li, buf.as_mut_ptr(), 1)
            );
            step!(
                "yF",
                nyz,
                4,
                mrc_read_y_float(&mut header, &mut li, buf.as_mut_ptr().cast(), 1)
            );

            // Sub-rectangle plus a slope/offset scaling, then a Y read of it.
            li.xmin = 3;
            li.xmax = nx - 4;
            li.ymin = 2;
            li.ymax = ny - 3;
            li.slope = 1.5;
            li.offset = -3.;
            li.smin = 0.;
            li.smax = 0.;
            let rw = (li.xmax - li.xmin + 1) as usize;
            let rh = (li.ymax - li.ymin + 1) as usize;
            step!(
                "subB",
                rw * rh,
                1,
                mrc_read_section_byte(&mut header, &mut li, buf.as_mut_ptr(), z)
            );
            step!(
                "subF",
                rw * rh,
                4,
                mrc_read_section_float(&mut header, &mut li, buf.as_mut_ptr().cast(), z)
            );
            li.axis = 2;
            step!(
                "subYF",
                rw * nz as usize,
                4,
                mrc_read_y_float(&mut header, &mut li, buf.as_mut_ptr().cast(), 2)
            );
            drop(fp);
        }
        assert_eq!(got, want, "mode {mode} must match the reference driver");
    }
}
