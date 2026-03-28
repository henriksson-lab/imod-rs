/// nogputxc - Non-GPU tilt cross-correlation stub.
///
/// This is a placeholder stub that provides a no-op GPU interface for tiltxcorr.
/// In the original IMOD, this was compiled when no GPU support was available,
/// providing dummy implementations of the TxcGPU class methods that always
/// report no GPU available and return error codes.
///
/// In the Rust port, GPU support is handled differently (via feature flags or
/// runtime detection), so this binary exists only as a compatibility placeholder.
fn main() {
    eprintln!("nogputxc: This is a stub program.");
    eprintln!("GPU cross-correlation is not available in this build.");
    eprintln!("Use tiltxcorr with CPU-based cross-correlation instead.");
    std::process::exit(1);
}
