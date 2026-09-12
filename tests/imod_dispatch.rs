//! The single-binary launcher (`src/bin/imod.rs`).
//!
//! Upstream IMOD installs one executable per command; this crate builds one
//! `imod` binary instead.  What must not change is what each program sees, so
//! these tests check that the two invocation forms — `imod <command> …` and a
//! link named `<command>` — produce byte-identical stdout, stderr and exit
//! status, including on the error paths whose messages are derived from
//! `argv[0]` through `imodProgName`.

mod common;

use std::process::Output;

/// Runs `command` with `arguments` through both invocation forms and asserts
/// the two results are identical, returning the shared result.
fn both_forms(command: &str, arguments: &[&str], environment: &[(&str, &str)]) -> Output {
    let subcommand = common::imod_cmd(command)
        .args(arguments)
        .envs(environment.iter().copied())
        .output()
        .unwrap_or_else(|error| panic!("run imod {command}: {error}"));
    let linked = common::imod_link_cmd(command)
        .args(arguments)
        .envs(environment.iter().copied())
        .output()
        .unwrap_or_else(|error| panic!("run linked {command}: {error}"));
    assert_eq!(
        subcommand.status.code(),
        linked.status.code(),
        "{command} {arguments:?}: exit status differs between invocation forms"
    );
    assert_eq!(
        String::from_utf8_lossy(&subcommand.stdout),
        String::from_utf8_lossy(&linked.stdout),
        "{command} {arguments:?}: stdout differs between invocation forms"
    );
    assert_eq!(
        String::from_utf8_lossy(&subcommand.stderr),
        String::from_utf8_lossy(&linked.stderr),
        "{command} {arguments:?}: stderr differs between invocation forms"
    );
    subcommand
}

#[test]
fn both_invocation_forms_give_the_program_the_same_argv() {
    // A Fortran program reached through `getinout`/`parse_input_params`.
    let missing_header = both_forms("header", &["-size", "imod-dispatch-no-such-file.mrc"], &[]);
    assert_eq!(missing_header.status.code(), Some(1));

    // A PIP error path in a Fortran program: the message embeds the program
    // name, so an `argv[0]` the launcher got wrong would show up here.
    let bad_mode = both_forms(
        "newstack",
        &["-mode", "9", "imod-dispatch-no-such-file.mrc", "out.mrc"],
        &[],
    );
    assert_eq!(bad_mode.status.code(), Some(1));

    // A C program whose exit prefix is built from `imodProgName(argv[0])`.
    let clip_missing = both_forms("clip", &["stats", "imod-dispatch-no-such-file.mrc"], &[]);
    assert_eq!(clip_missing.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&clip_missing.stdout).contains("ERROR: clip - "),
        "clip's exit prefix must name the program, not the launcher: {}",
        String::from_utf8_lossy(&clip_missing.stdout)
    );

    // Same, for the C++ command whose prefix is also `imodProgName`-derived.
    let mrc2tif_missing = both_forms(
        "mrc2tif",
        &["imod-dispatch-no-such-file.mrc", "out.tif"],
        &[],
    );
    assert_eq!(mrc2tif_missing.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&mrc2tif_missing.stdout).contains("ERROR: mrc2tif - "),
        "mrc2tif's exit prefix must name the program, not the launcher: {}",
        String::from_utf8_lossy(&mrc2tif_missing.stdout)
    );

    // A Python-launcher translation reading `args_os()`.
    let submfg_usage = both_forms("submfg", &[], &[("IMOD_DIR", "/fixture/imod")]);
    assert!(String::from_utf8_lossy(&submfg_usage.stdout).contains("Usage:  submfg"));

    common::remove_command_links();
}

#[test]
fn launcher_lists_its_commands_and_exits_nonzero() {
    for arguments in [vec![], vec!["--help"], vec!["-h"], vec!["no-such-command"]] {
        let listing = std::process::Command::new(env!("CARGO_BIN_EXE_imod"))
            .args(&arguments)
            .output()
            .expect("run imod");
        assert_eq!(
            listing.status.code(),
            Some(1),
            "imod {arguments:?} must exit 1"
        );
        assert!(listing.stdout.is_empty(), "the listing goes to stderr");
        let text = String::from_utf8_lossy(&listing.stderr);
        for command in [
            "3dmod",
            "alterheader",
            "batchruntomo",
            "binvol",
            "clip",
            "convertmod",
            "etomo",
            "header",
            "imodinfo",
            "imodjoin",
            "imodqtassist",
            "imodsendevent",
            "midas",
            "mrc2tif",
            "newstack",
            "processchunks",
            "sourcedoc",
            "subm",
            "submfg",
            "tif2mrc",
            "trimvol",
            "wmod2imod",
        ] {
            assert!(
                text.contains(&format!("\n  {command}\n")),
                "imod {arguments:?} must list {command}"
            );
        }
    }
}
