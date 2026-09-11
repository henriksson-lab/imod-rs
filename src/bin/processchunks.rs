//! Rust executable mapping `IMOD/qttools/processchunks/processchunks.cpp:main`.

fn main() {
    let arguments = std::env::args().collect::<Vec<_>>();
    std::process::exit(
        imod_rs::imod::qttools::processchunks::processchunks::processchunks(&arguments),
    );
}
