//! Rust executable mapping `IMOD/qttools/sendevent/imodsendevent.cpp:main`.

fn main() {
    let arguments = std::env::args().collect::<Vec<_>>();
    std::process::exit(imod_rs::imod::qttools::sendevent::imodsendevent::imodsendevent(&arguments));
}
