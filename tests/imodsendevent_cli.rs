//! Command-line parity probes for `qttools/sendevent/imodsendevent.cpp`.

use std::process::Command;

#[test]
fn imodsendevent_rejects_non_numeric_action_before_opening_clipboard() {
    let output = Command::new(env!("CARGO_BIN_EXE_imodsendevent"))
        .args(["42", "not-an-action"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(3));
    assert_eq!(
        String::from_utf8(output.stderr).unwrap(),
        "ERROR: imodsendevent - invalid characters in action entry not-an-action\n"
    );
}

#[test]
fn imodsendevent_help_follows_the_original_option_parser() {
    // The C++ loop deliberately examines options only while a following token
    // exists, so `-h` needs a dummy token to reach its switch case.
    let output = Command::new(env!("CARGO_BIN_EXE_imodsendevent"))
        .args(["-h", "dummy"])
        .output()
        .unwrap();
    assert!(output.status.success());
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        "   Usage: imodsendevent [-t timeout] [-D] Window_ID action [arguments]\n"
    );
}
