//! `istoreDump` (`IMOD/libimod/istore.c:281`) writes through the C library's
//! `stdout`, as the source's `printf` does, so capturing its text needs
//! `dup2` on file descriptor 1.
//!
//! This test lives here rather than beside the unit because
//! `src/imod/libimod/istore.rs` is one of the modules NATIVE.md requires to
//! carry no `libc::`; the capture plumbing is a harness detail, not part of
//! the translation, so it moved out of the module rather than being dropped.
//! The expected text is verbatim from a driver compiled against the pinned
//! `IMOD/libimod/istore.c` and linked to the reference build's `libimod`
//! (`%12.6g` and the flag-name decoding at `istore.c:297-325` are part of the
//! contract).

use imod_rs::imod::libimod::istore::{Istore, StoreUnion, istore_dump};

#[test]
fn dump_matches_source_printf_text() {
    let mut list = vec![
        Istore {
            type_: 1,
            flags: (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7),
            index: StoreUnion::from_i(0),
            value: StoreUnion::from_i(0),
        },
        Istore {
            type_: 10,
            flags: 4,
            index: StoreUnion::from_i(3),
            value: StoreUnion::from_f(1234567.),
        },
        Istore {
            type_: 23,
            flags: (1 << 5) | (1 << 6) | (1 << 7),
            index: StoreUnion::from_i(0),
            value: StoreUnion::from_i(0),
        },
        Istore {
            type_: 24,
            flags: 2,
            index: StoreUnion::from_s([-3, 9]),
            value: StoreUnion::from_i(0),
        },
        Istore {
            type_: 99,
            flags: 0,
            index: StoreUnion::from_i(7),
            value: StoreUnion::from_i(8),
        },
        Istore {
            type_: 0,
            flags: 3,
            index: StoreUnion::from_b([1, 2, 3, 4]),
            value: StoreUnion::from_i(0),
        },
    ];
    list[3].value = StoreUnion::from_i(0);
    let path = std::env::temp_dir().join(format!("imod-rs-istore-dump-{}.txt", std::process::id()));
    let c_path = std::ffi::CString::new(path.to_str().unwrap()).unwrap();
    unsafe {
        libc::fflush(std::ptr::null_mut());
        let saved = libc::dup(1);
        let fd = libc::open(
            c_path.as_ptr(),
            libc::O_WRONLY | libc::O_CREAT | libc::O_TRUNC,
            0o644,
        );
        libc::dup2(fd, 1);
        istore_dump(&list);
        istore_dump(&[]);
        libc::fflush(std::ptr::null_mut());
        libc::dup2(saved, 1);
        libc::close(fd);
        libc::close(saved);
    }
    let text = std::fs::read_to_string(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    assert_eq!(
        text,
        concat!(
            " 6 items in list:\n",
            "     1-COLOR     360-NOIND|REVERT|SURF|ONEPT           0           0\n",
            "    10-VALUE1       4-           3  1.23457e+06\n",
            "    23-ISOTHRESH     340-CAP|DEL|OUTER           0           0\n",
            "    24-       2-     -3      9           0\n",
            "    99-       0-           7           8\n",
            "     0-       3-   1   2   3   4           0\n",
            " 0 items in list:\n",
        )
    );
}
