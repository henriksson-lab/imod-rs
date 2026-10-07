//! Native-golden coverage for `imodkillgroup` (`IMOD/pysrc/imodkillgroup`, translated in
//! `src/imod/pysrc/imodkillgroup.rs`).
//!
//! Every row of `fixtures/imodkillgroup/cases.tsv` was run through the native Python
//! script by `fixtures/make-imodkillgroup-goldens.sh` (`make-pyscript-goldens.py`); see
//! `tests/pysetup_common` for what is compared.

mod common;
mod pysetup_common;

#[test]
fn imodkillgroup_matches_native_goldens() {
    let failures = pysetup_common::run_cases("imodkillgroup");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Kills a process group, and a process tree with `-t`, that the test
/// started; every process is gone afterwards and the status is 0.
#[cfg(unix)]
#[test]
fn kills_a_group_and_a_tree() {
    use std::os::unix::process::CommandExt as _;
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for tree in [false, true] {
        let mut leader = std::process::Command::new("sh")
            .args(["-c", "sleep 300 & sleep 300 & wait"])
            .process_group(0)
            .spawn()
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(300));
        let pid = leader.id().to_string();
        let mut args = vec![pid.as_str()];
        if tree {
            args.insert(0, "-t");
        }
        let output = common::imod_cmd("imodkillgroup")
            .args(&args)
            .env("IMOD_DIR", root.join("IMOD"))
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(0), "{output:?}");
        let status = leader.wait().unwrap();
        assert!(!status.success());
        // SAFETY: probing our own (now killed) process group.  The killed
        // children are reaped by init after their parent is gone, so allow
        // that a moment.
        let mut alive = true;
        for _ in 0..100 {
            alive = unsafe { libc::killpg(leader.id() as libc::pid_t, 0) } == 0;
            if !alive {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        assert!(!alive, "group {pid} still has processes");
    }
}
