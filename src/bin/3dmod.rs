//! Executable entry for the source-shaped `IMOD/3dmod/imod.cpp` translation.
fn main() {
    let arguments = std::env::args().collect::<Vec<_>>();
    match imod_rs::imod::three_dmod::imod::imod_main(&arguments) {
        Ok(status) => std::process::exit(status),
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(1);
        }
    }
}
