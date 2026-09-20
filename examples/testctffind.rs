//! Runs the translated `IMOD/librgctf/testctffind.cpp` as a program, so that a
//! native differential can drive `ctffind()` the way the library's own test
//! binary does.  `librgctf` has no entry in the `imod` launcher's dispatch
//! table, because upstream's only shipped consumer of the library is the Qt
//! program `ctfplotter`.
fn main() {
    let argv: Vec<Vec<u8>> = std::env::args_os()
        .map(|arg| std::os::unix::ffi::OsStrExt::as_bytes(arg.as_os_str()).to_vec())
        .collect();
    std::process::exit(imod_rs::imod::librgctf::testctffind::main(&argv));
}
