fn main() {
    std::process::exit(imod_rs::imod::pysrc::submfg::submfg(
        &std::env::args_os().collect::<Vec<_>>(),
    ));
}
