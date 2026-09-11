//! Executable entry for `IMOD/midas/midas.cpp`.
fn main() {
    let arguments = std::env::args().collect::<Vec<_>>();
    match imod_rs::imod::midas::midas::midas_main(&arguments) {
        Ok(status) => std::process::exit(status),
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(1);
        }
    }
}
