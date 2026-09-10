use std::process::Command;

#[test]
fn requires_imod_runtime_before_java_boundary() {
    let output = Command::new(env!("CARGO_BIN_EXE_etomo"))
        .env_remove("IMOD_DIR")
        .output()
        .expect("run etomo");
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "The IMOD_DIR environment variable has not been set\nSet it to point to the directory where IMOD is installed\n"
    );
}
