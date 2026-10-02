//! `IMOD/Etomo/src/etomo/logic/VersionControl.java`.

use crate::imod::etomo::base_manager;
use crate::imod::etomo::process::system_program::SystemProgram;
use crate::imod::etomo::storage::log_file::{LogFile, LogFileError};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::etomo_version::EtomoVersion;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::environment_variable;

/// Java private `DEPENDANT_IMOD_VERSION`.
const DEPENDANT_IMOD_VERSION: &str = "4.4.4";
/// Java private `DEPENDANT_PEET_VERSION`.
const DEPENDANT_PEET_VERSION: &str = "1.8.0";
/// Java `TIME_STAMP` (updated by a script).
pub const TIME_STAMP: &str = "8/14/2026 16:32";

/// Java `isCompatiblePeet(AxisID)`.
///
/// Fixed in translation (BUGS.md): Java calls `imodVersion.lt(...)` on a null
/// `imodVersion` when `imodinfo` printed nothing (NullPointerException).  A
/// missing IMOD version is treated as the null (earliest) version.
pub fn is_compatible_peet(axis_id: AxisID) -> bool {
    let peet_version = EtomoVersion::get_default_instance_with_version(get_peet_version().as_deref());
    let mut imod_version: Option<EtomoVersion> = None;
    let imod_info = get_imod_info(axis_id);
    if let Some(imod_info) = &imod_info
        && !imod_info.is_empty()
    {
        imod_version = Some(EtomoVersion::get_default_instance_with_version(Some(
            &imod_info[0],
        )));
    }
    let imod_version =
        imod_version.unwrap_or_else(|| EtomoVersion::get_default_instance_with_version(None));
    let mut ret_val = true;
    if imod_version.lt_string(Some(DEPENDANT_IMOD_VERSION)) {
        ret_val = peet_version.lt_string(Some(DEPENDANT_PEET_VERSION));
    } else if imod_version.ge_string(Some(DEPENDANT_IMOD_VERSION)) {
        ret_val = peet_version.ge_string(Some(DEPENDANT_PEET_VERSION));
    }
    if !ret_val {
        ui_harness::open_message_dialog_from_process(
            None,
            &format!(
                "The PEET version is incompatible with this IMOD version.  IMOD {DEPENDANT_IMOD_VERSION} or later requires PEET {DEPENDANT_PEET_VERSION} or later.  IMOD versions prior to {DEPENDANT_IMOD_VERSION} require a PEET version prior to {DEPENDANT_PEET_VERSION}.  The currently installed versions are IMOD: {imod_version}, and PEET: {peet_version}."
            ),
            "Wrong Version of PEET",
            None,
        );
    }
    ret_val
}

/// Java `getPeetVersion()`.
pub fn get_peet_version() -> Option<String> {
    let particle_dir = environment_variable::INSTANCE.get_value(
        None,
        None,
        environment_variable::PARTICLE_DIR,
        Some(AxisID::Only),
    );
    let peet_version_file = match LogFile::get_instance_dir(
        std::path::Path::new(&particle_dir),
        "PEETVersion.txt",
        None,
    ) {
        Ok(file) => file,
        Err(LogFileError::Lock(_)) => return None,
        Err(e) => {
            eprintln!("{e}");
            return None;
        }
    };
    let id = match peet_version_file.open_reader() {
        Ok(id) => id,
        // `catch (FileNotFoundException | LockException e) {}`
        Err(LogFileError::Lock(_)) => None,
        Err(LogFileError::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => {
            eprintln!("{e}");
            None
        }
    };
    if let Some(id) = &id {
        match peet_version_file.read_line(id) {
            Ok(Some(version)) if !version.trim_matches(char::is_whitespace).is_empty() => {
                peet_version_file.close_id(Some(&**id));
                return Some(version);
            }
            Ok(_) => {}
            Err(e) => eprintln!("{e}"),
        }
        peet_version_file.close_id(Some(&**id));
    }
    None
}

/// Java `getImodInfo(AxisID)`.
pub fn get_imod_info(axis_id: AxisID) -> Option<Vec<String>> {
    let command = vec![format!(
        "{}imodinfo",
        base_manager::get_imod_bin_path().unwrap_or_else(|| "null".to_owned())
    )];
    let three_dmod_h = SystemProgram::new_array(None, None, Some(command), axis_id);
    three_dmod_h.run();
    let stdout = three_dmod_h.get_std_output();
    let mut imod_info: Option<Vec<String>> = None;
    if let Some(stdout) = stdout
        && !stdout.is_empty()
    {
        let mut info = Vec::new();
        // Get version info
        if let Some(idx_version) = stdout[0].find("Version")
            && idx_version > 0
        {
            let no_path = &stdout[0][idx_version..];
            let tokens: Vec<&str> = no_path.split(' ').collect();
            if tokens.len() > 1 {
                info.push(tokens[1].to_owned());
            }
        }
        // Get copyright info
        if stdout.len() > 3 {
            info.push(stdout[1].clone());
            info.push(stdout[2].clone());
        }
        imod_info = Some(info);
    }
    imod_info
}
