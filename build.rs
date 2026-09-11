use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
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

    if env::var_os("CARGO_FEATURE_QT").is_none() {
        return;
    }
    println!("cargo:rerun-if-changed=src/imod/qttools/mrc2tif/mrc2tif_qimage.cpp");
    let flags = Command::new("pkg-config")
        .args(["--cflags", "Qt5Gui"])
        .output()
        .expect("Qt5 pkg-config is required for the source QImage boundary");
    assert!(flags.status.success(), "Qt5Gui pkg-config lookup failed");
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let object = out.join("mrc2tif_qimage.o");
    let mut compile = Command::new("c++");
    compile.args(["-std=c++17", "-fPIC", "-c"]);
    compile.args(String::from_utf8(flags.stdout).unwrap().split_whitespace());
    compile.arg("src/imod/qttools/mrc2tif/mrc2tif_qimage.cpp");
    compile.args(["-o", object.to_str().unwrap()]);
    assert!(
        compile
            .status()
            .expect("C++ compiler unavailable")
            .success()
    );
    let archive = out.join("libmrc2tif_qimage.a");
    assert!(
        Command::new("ar")
            .args(["crs", archive.to_str().unwrap(), object.to_str().unwrap()])
            .status()
            .expect("archiver unavailable")
            .success()
    );
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=mrc2tif_qimage");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    let libs = Command::new("pkg-config")
        .args(["--libs", "Qt5Gui"])
        .output()
        .expect("Qt5 pkg-config is required for the source QImage boundary");
    assert!(libs.status.success(), "Qt5Gui library lookup failed");
    for flag in String::from_utf8(libs.stdout).unwrap().split_whitespace() {
        if let Some(name) = flag.strip_prefix("-l") {
            println!("cargo:rustc-link-lib={name}");
        } else if let Some(path) = flag.strip_prefix("-L") {
            println!("cargo:rustc-link-search=native={path}");
        }
    }
}
