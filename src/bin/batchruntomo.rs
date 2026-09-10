//! Command entry point for `IMOD/pysrc/batchruntomo`.
fn main() {
    std::process::exit(imod_rs::imod::pysrc::batchruntomo::batchruntomo(
        &std::env::args_os().collect::<Vec<_>>(),
    ));
}
