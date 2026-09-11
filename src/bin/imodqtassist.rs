//! Rust executable mapping `IMOD/qttools/qtassist/imodqtassist.cpp:main`.

fn main() {
    let arguments = std::env::args().collect::<Vec<_>>();
    std::process::exit(imod_rs::imod::qttools::qtassist::imodqtassist::imodqtassist(&arguments));
}
