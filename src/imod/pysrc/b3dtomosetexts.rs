//! Translation of `IMOD/pysrc/b3dtomosetexts`: outputs the results of
//! `findRootAxisAndExtensions` on a dataset.
//!
//! A Python command script with no functions; its top level is
//! [`b3dtomosetexts`].

use super::imodpy::{find_root_axis_and_extensions, fmtstr, prnstr};
use std::ffi::OsString;
use std::io::Write as _;

/// The script's top level (`b3dtomosetexts:1-60`).  Returns the status of
/// its `sys.exit`.
pub fn b3dtomosetexts(arguments: &[OsString]) -> i32 {
    let progname = "b3dtomosetexts";
    let prefix = format!("ERROR: {progname} - ");
    let argv: Vec<String> = arguments
        .iter()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect();

    //
    // Setup runtime environment (bare minimum)
    if std::env::var_os("IMOD_DIR").is_none() {
        print!("{prefix} IMOD_DIR is not defined!\n");
        let _ = std::io::stdout().flush();
        return 1;
    }

    if argv.len() > 1 && argv[1].starts_with("-h") {
        prnstr(
            "Usage: b3dtomosetexts [directory]
  Examines data set files in the current directory or the optionally specified one.
  Outputs all on one line (\"none\" is output if a string is undetermined):
    type extension (naming style): old, mrc, hdf, or none
    raw stack extension or none
    root name of dataset or none
    2 for dual axis, 0 for single axis, or -1 if not determined
    com file extension: com, pcm, or none",
            "\n",
            false,
        );
        let _ = std::io::stdout().flush();
        return 0;
    }

    if argv.len() > 1 && std::env::set_current_dir(&argv[1]).is_err() {
        prnstr(
            &format!("Error changing directory to {}", argv[1]),
            "\n",
            false,
        );
        let _ = std::io::stdout().flush();
        return 1;
    }

    let (mut com_ext, dual_num, mut root, mut type_ext, mut stack_ext) =
        find_root_axis_and_extensions(0, None);
    if com_ext.is_empty() {
        com_ext = "none".to_owned();
    }
    if type_ext.is_none() {
        type_ext = Some("none".to_owned());
    }
    if type_ext.as_deref() == Some("") {
        type_ext = Some("old".to_owned());
    }
    if stack_ext.is_empty() {
        stack_ext = "none".to_owned();
    }
    if root.is_empty() {
        root = "none".to_owned();
    }
    prnstr(
        &fmtstr(
            "{} {} {} {} {}",
            &[
                type_ext.unwrap_or_default(),
                stack_ext,
                root,
                dual_num.to_string(),
                com_ext,
            ],
        ),
        "\n",
        false,
    );
    let _ = std::io::stdout().flush();
    0
}
