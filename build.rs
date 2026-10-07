fn main() {
    // The `imod` executable is linked as a fixed-address (non-PIE)
    // executable: every command is a separate process start, and a PIE of
    // this size carries ~77 000 relative relocations that the loader applies
    // -- touching ~190 extra pages -- at every start.  Short commands such as
    // `header` are dominated by start-up, where native IMOD's small
    // executables pay none of it.  Code generation is unchanged (still
    // position-independent code); only the final link differs.
    // `-no-pie` is a GNU ld/gold/lld spelling: macOS ld calls it `-no_pie`
    // (and deprecates it on arm64) and MSVC link.exe has neither, so it is
    // passed on Linux only.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-arg-bins=-no-pie");
    }
    // Without these, cargo reruns this script — and so recompiles the whole
    // library with a new timestamp — whenever *any* file in the package
    // changes, fixtures included: recording a golden manifest
    // (`fixtures/regen-golden.sh`) cost a full rebuild per suite.  The stamp
    // still moves whenever the code does, as `__DATE__` would.
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src");
    // The C sources embed `__DATE__`/`__TIME__` from the compilation of the
    // reference binary.  Those are build metadata, not input-dependent output;
    // emit the same two fields in the identical C formats ("Mmm dd yyyy" with a
    // space-padded day, and "HH:MM:SS") so the translated banners keep the
    // source field layout.
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;
    let days = now / 86_400;
    let secs = now % 86_400;
    let (mut y, mut d) = (1970_i64, days);
    loop {
        let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
        let len = if leap { 366 } else { 365 };
        if d < len {
            break;
        }
        d -= len;
        y += 1;
    }
    let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    let lengths = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let mut m = 0;
    while d >= lengths[m] {
        d -= lengths[m];
        m += 1;
    }
    println!(
        "cargo:rustc-env=IMOD_BUILD_DATE={} {:2} {}",
        names[m],
        d + 1,
        y
    );
    println!(
        "cargo:rustc-env=IMOD_BUILD_TIME={:02}:{:02}:{:02}",
        secs / 3600,
        (secs % 3600) / 60,
        secs % 60
    );
}
