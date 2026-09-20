// TEMPORARY differential runner for the measuredrift retranslation.
// Deleted once `imod measuredrift` has a dispatch arm.
fn main() {
    let mut args: Vec<String> = std::env::args().collect();
    args[0] = "measuredrift".to_string();
    let rc = imod_rs::imod::mrc::measuredrift::measuredrift(&args);
    std::process::exit(rc);
}
