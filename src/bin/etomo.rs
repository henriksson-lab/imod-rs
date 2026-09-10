fn main() {
    std::process::exit(imod_rs::imod::pysrc::etomo::etomo(
        &std::env::args_os().collect::<Vec<_>>(),
    ));
}
