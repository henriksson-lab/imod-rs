//! Rust executable mapping `IMOD/qttools/sourcedoc/sourcedoc.cpp:main`.

fn main() {
    let arguments = std::env::args().collect::<Vec<_>>();
    std::process::exit(imod_rs::imod::qttools::sourcedoc::sourcedoc::sourcedoc(
        &arguments,
    ));
}
