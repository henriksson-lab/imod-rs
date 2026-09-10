fn main() {
    std::process::exit(imod_rs::imod::pysrc::subm::subm(
        &std::env::args_os().collect::<Vec<_>>(),
    ));
}
