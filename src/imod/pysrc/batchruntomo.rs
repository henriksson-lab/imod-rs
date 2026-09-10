//! Translation scaffold and command boundary for `IMOD/pysrc/batchruntomo`.
//!
//! `etomo`, `copytomocoms`, `processchunks`, and every IMOD executable named by
//! generated com files remain process boundaries, as in the Python program.
#![allow(clippy::needless_pass_by_value)]
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};

use super::imodpy::{add_imod_bin_ignore_sighup, print_pid};

/// Matches the top-level program in `IMOD/pysrc/batchruntomo`.
pub fn batchruntomo(arguments: &[OsString]) -> i32 {
    let Some(imod_dir) = std::env::var_os("IMOD_DIR") else {
        // The top-level Python startup writes directly to stdout; `prefix`
        // already has a trailing space and this literal starts with one.
        println!("ERROR: batchruntomo -  IMOD_DIR is not defined!");
        return 1;
    };
    // Source initialization does this before PIP parsing and process use.
    add_imod_bin_ignore_sighup();
    let mut directives = Vec::new();
    let mut validation = 0_i32;
    let mut do_pid = false;
    let mut index = 1;
    while index < arguments.len() {
        let argument = arguments[index].to_string_lossy();
        match argument.as_ref() {
            "-help" | "--help" => {
                println!("batchruntomo - run one or more data sets in batch mode");
                return 0;
            }
            "-directive" | "-DirectiveFile" => {
                index += 1;
                if let Some(value) = arguments.get(index) {
                    directives.push(PathBuf::from(value));
                }
            }
            "-root" | "-RootName" => {
                // `batchruntomo:5263-5274`: RootName has a value of its own;
                // it must not be mistaken for an unnamed directive file.
                index += 1;
            }
            "-validation" | "-ValidationType" => {
                // `batchruntomo:5490`: keep the PIP integer rather than
                // treating the presence of this option as a Boolean.
                if let Some(value) = arguments
                    .get(index + 1)
                    .and_then(|value| value.to_string_lossy().parse::<i32>().ok())
                {
                    validation = value;
                    index += 1;
                }
            }
            "-pid" | "-PID" => do_pid = true,
            value if !value.starts_with('-') => directives.push(PathBuf::from(value)),
            _ => {}
        }
        index += 1;
    }
    // `batchruntomo:5166`: report the launcher PID before validating files.
    print_pid(do_pid);
    if directives.is_empty() {
        println!("ERROR: batchruntomo - You must enter at least one directive file");
        return 1;
    }
    if validation >= 0 {
        let validation_file = PathBuf::from(imod_dir).join("com/directives.csv");
        if !validation_file.exists() {
            eprintln!(
                "ERROR: batchruntomo - Cannot find file for validating directives, {}",
                validation_file.display()
            );
            return 1;
        }
    }
    for directive in directives {
        if !directive.exists() {
            eprintln!(
                "ERROR: batchruntomo - Directive file {} does not exist",
                directive.display()
            );
            return 1;
        }
        if validation > 0 {
            match read_directive_or_template(&directive) {
                Ok(_) => println!("Directives all seem OK in that file"),
                Err(errors) => {
                    for error in errors {
                        eprintln!("ERROR: batchruntomo - {error}");
                    }
                    return 1;
                }
            }
        } else {
            // The Python top-level setup path calls Etomo with these same source
            // options.  Its GUI/JVM implementation is intentionally external.
            match std::process::Command::new("etomo")
                .args(["--fromBRT", "--directive"])
                .arg(&directive)
                .status()
            {
                Ok(status) if status.success() => (),
                Ok(status) => {
                    eprintln!(
                        "ERROR: batchruntomo - etomo setup failed for {} with status {}",
                        directive.display(),
                        status
                    );
                    return 1;
                }
                Err(error) => {
                    eprintln!(
                        "ERROR: batchruntomo - Starting etomo for {}: {error}",
                        directive.display()
                    );
                    return 1;
                }
            }
        }
    }
    0
}

/// File-system operations owned by the `deliverStack` source branch.  The caller owns
/// Etomo's mutable dataset state; this interface keeps symlink/rename as explicit IO.
pub trait DeliverStackWorkflow {
    fn same_directory(&self) -> bool;
    fn inspect_stacks(&mut self) -> Result<(), String>;
    fn select_extension_and_conflicts(&mut self) -> Result<(), String>;
    fn deliver_or_rename_stack(&mut self) -> Result<(), String>;
    fn deliver_ancillaries(&mut self) -> Result<(), String>;
}
/// Matches `deliverStack` (`IMOD/pysrc/batchruntomo:447`).
pub fn deliver_stack<Workflow: DeliverStackWorkflow>(
    workflow: &mut Workflow,
) -> Result<(), String> {
    if workflow.same_directory() {
        return Ok(());
    }
    workflow.inspect_stacks()?;
    workflow.select_extension_and_conflicts()?;
    workflow.deliver_or_rename_stack()?;
    workflow.deliver_ancillaries()
}

/// Process/file operations in `makeSeedAndTrack`; RAPTOR, transferfid, autofidseed,
/// and beadtrack remain separately observable external process boundaries.
pub trait SeedAndTrackWorkflow {
    fn validate_directives(&mut self) -> Result<(i32, i32, i32), String>;
    fn modify_track_and_edf(&mut self) -> Result<(), String>;
    fn run_raptor(&mut self) -> Result<(), String>;
    fn transfer_fiducials(&mut self) -> Result<bool, String>;
    fn auto_seed(&mut self, append_transfer: bool) -> Result<(), String>;
    fn mark_seeding_done(&mut self) -> Result<(), String>;
    fn track(&mut self, run: i32) -> Result<bool, String>;
}
/// Matches `makeSeedAndTrack` (`IMOD/pysrc/batchruntomo:2521`).
pub fn make_seed_and_track<Workflow: SeedAndTrackWorkflow>(
    workflow: &mut Workflow,
) -> Result<(), String> {
    let (tracking, runs, seeding) = workflow.validate_directives()?;
    workflow.modify_track_and_edf()?;
    if tracking == 2 {
        workflow.run_raptor()?;
    } else {
        let transferred = if seeding & 2 != 0 {
            workflow.transfer_fiducials()?
        } else {
            false
        };
        if seeding & 1 != 0 && !transferred {
            workflow.auto_seed(false)?;
        }
        workflow.mark_seeding_done()?;
    }
    for run in 0..runs.max(0) {
        if workflow.track(run)? {
            break;
        }
    }
    Ok(())
}

/// Source state/command boundary for `detectGoldIn3D`.
pub trait DetectGoldIn3dWorkflow {
    fn enabled(&self) -> bool;
    fn determine_binning(&mut self) -> Result<i32, String>;
    fn make_aligned_stack_if_needed(&mut self, binning: i32) -> Result<(), String>;
    fn determine_thickness(&mut self) -> Result<i32, String>;
    fn make_tilt_com(&mut self, binning: i32, thickness: i32) -> Result<(), String>;
    fn reconstruct(&mut self) -> Result<(), String>;
    fn find_beads(&mut self, binning: i32) -> Result<(), String>;
    fn record_edf(&mut self) -> Result<(), String>;
}
/// Matches `detectGoldIn3D` (`IMOD/pysrc/batchruntomo:3332`).
pub fn detect_gold_in_3d<Workflow: DetectGoldIn3dWorkflow>(
    workflow: &mut Workflow,
) -> Result<(), String> {
    if !workflow.enabled() {
        return Ok(());
    }
    let binning = workflow.determine_binning()?;
    workflow.make_aligned_stack_if_needed(binning)?;
    let thickness = workflow.determine_thickness()?;
    workflow.make_tilt_com(binning, thickness)?;
    workflow.reconstruct()?;
    workflow.find_beads(binning)?;
    workflow.record_edf()
}

/// State and command boundaries of `modifyTiltComFile`; the workflow owns all source
/// globals and calls `pysed` only after this source ordering has been established.
pub trait ModifyTiltComWorkflow {
    fn read_tilt(&mut self) -> Result<(), String>;
    fn resolve_thickness(&mut self, supplied: Option<i32>) -> Result<i32, String>;
    fn geometry(&mut self) -> Result<(i32, i32, i32, i32), String>;
    fn modify_tilt(&mut self, thickness: i32, geometry: (i32, i32, i32, i32))
    -> Result<(), String>;
}
/// Matches `modifyTiltComFile` (`IMOD/pysrc/batchruntomo:3514`).
pub fn modify_tilt_com_file<Workflow: ModifyTiltComWorkflow>(
    workflow: &mut Workflow,
    sample_thickness: Option<i32>,
) -> Result<(), String> {
    workflow.read_tilt()?;
    let thickness = workflow.resolve_thickness(sample_thickness)?;
    let geometry = workflow.geometry()?;
    workflow.modify_tilt(thickness, geometry)
}

/// Process operations of `make3dCtfCorrectedTomogram` in source command order.
pub trait Ctf3dTomogramWorkflow {
    fn prepare_aligned_stack(&mut self) -> Result<(), String>;
    fn make_ctf3d_com(&mut self, processors: i32) -> Result<(), String>;
    fn run_ctf3d(&mut self) -> Result<(), String>;
    fn rename_output(&mut self) -> Result<(), String>;
}
/// Matches `make3dCtfCorrectedTomogram` (`IMOD/pysrc/batchruntomo:3662`).
pub fn make_3d_ctf_corrected_tomogram<Workflow: Ctf3dTomogramWorkflow>(
    workflow: &mut Workflow,
    processors: i32,
) -> Result<(), String> {
    workflow.prepare_aligned_stack()?;
    workflow.make_ctf3d_com(processors)?;
    workflow.run_ctf3d()?;
    workflow.rename_output()
}

/// Source-owned branch interface for `positionTomogram`.
pub trait PositionTomogramWorkflow {
    fn center_on_gold_if_needed(&mut self) -> Result<bool, String>;
    fn determine_thickness(&mut self) -> Result<i32, String>;
    fn make_positioning_volume(&mut self, thickness: i32) -> Result<(), String>;
    fn find_position(&mut self) -> Result<Option<(i32, f64, f64, f64)>, String>;
    fn record_original_position(&mut self) -> Result<(), String>;
    fn modify_tilt_and_align(&mut self, result: (i32, f64, f64, f64)) -> Result<(), String>;
}
/// Matches `positionTomogram` (`IMOD/pysrc/batchruntomo:3800`).
pub fn position_tomogram<Workflow: PositionTomogramWorkflow>(
    workflow: &mut Workflow,
) -> Result<(), String> {
    if workflow.center_on_gold_if_needed()? {
        return Ok(());
    }
    let thickness = workflow.determine_thickness()?;
    workflow.make_positioning_volume(thickness)?;
    workflow.record_original_position()?;
    if let Some(result) = workflow.find_position()? {
        workflow.modify_tilt_and_align(result)?;
    }
    Ok(())
}

/// Full external-process sequence used by `setupCombine`.
pub trait SetupCombineWorkflow {
    fn select_sirt_reconstructions(&mut self) -> Result<(), String>;
    fn default_z_limits(&mut self) -> Result<(), String>;
    fn find_z_limits(&mut self) -> Result<(), String>;
    fn choose_match_and_surfaces(&mut self) -> Result<(), String>;
    fn run_setupcombine(&mut self) -> Result<(), String>;
    fn write_edf_and_matchvol_size(&mut self) -> Result<(), String>;
}
/// Matches `setupCombine` (`IMOD/pysrc/batchruntomo:4010`).
pub fn setup_combine<Workflow: SetupCombineWorkflow>(
    workflow: &mut Workflow,
) -> Result<(), String> {
    workflow.select_sirt_reconstructions()?;
    workflow.default_z_limits()?;
    workflow.find_z_limits()?;
    workflow.choose_match_and_surfaces()?;
    workflow.run_setupcombine()?;
    workflow.write_edf_and_matchvol_size()
}

/// Process/log parsing operations for `initialCombineMatch`.
pub trait InitialCombineMatchWorkflow {
    fn recover_volume_match_mode(&mut self) -> Result<(), String>;
    fn use_volume_matching(&self) -> bool;
    fn run_match(&mut self) -> Result<(bool, Vec<String>), String>;
    fn accept_volume_match_thickness(&mut self, log: &[String]) -> Result<(), String>;
    fn retry_one_surface_if_suggested(&mut self, log: &[String]) -> Result<bool, String>;
    fn accept_initial_shift_if_possible(&mut self, log: &[String]) -> Result<bool, String>;
}
/// Matches `initialCombineMatch` (`IMOD/pysrc/batchruntomo:4250`).
pub fn initial_combine_match<Workflow: InitialCombineMatchWorkflow>(
    workflow: &mut Workflow,
) -> Result<(), String> {
    workflow.recover_volume_match_mode()?;
    let loops = if workflow.use_volume_matching() { 1 } else { 2 };
    for _ in 0..loops {
        let (success, log) = workflow.run_match()?;
        if workflow.use_volume_matching() && success {
            workflow.accept_volume_match_thickness(&log)?;
        }
        if !workflow.use_volume_matching() && workflow.retry_one_surface_if_suggested(&log)? {
            continue;
        }
        if success || workflow.accept_initial_shift_if_possible(&log)? {
            return Ok(());
        }
    }
    Err("Error getting initial alignment between volumes".to_owned())
}

/// File/directive/process sequence of `trimVolume`; trimvol itself remains an explicit
/// command execution in the implementing workflow.
pub trait TrimVolumeWorkflow {
    fn enabled(&mut self) -> Result<bool, String>;
    fn replacement_before(&mut self) -> Result<bool, String>;
    fn configure_reorientation(&mut self) -> Result<(), String>;
    fn choose_input_volumes(&mut self) -> Result<(), String>;
    fn resolve_sizes_and_find_section(&mut self) -> Result<(), String>;
    fn run_trimvol(&mut self) -> Result<(), String>;
    fn write_edf(&mut self) -> Result<(), String>;
    fn replacement_after(&mut self) -> Result<(), String>;
}
/// Matches `trimVolume` (`IMOD/pysrc/batchruntomo:4429`).
pub fn trim_volume<Workflow: TrimVolumeWorkflow>(workflow: &mut Workflow) -> Result<(), String> {
    if workflow.replacement_before()? || !workflow.enabled()? {
        return Ok(());
    }
    workflow.configure_reorientation()?;
    workflow.choose_input_volumes()?;
    workflow.resolve_sizes_and_find_section()?;
    workflow.run_trimvol()?;
    workflow.write_edf()?;
    workflow.replacement_after()
}

/// Values accepted by `findTaggedValue` (`IMOD/pysrc/batchruntomo:242`).
#[derive(Clone, Debug, PartialEq)]
pub enum TaggedValue {
    String(String),
    Integer(i32),
    Float(f64),
}

/// Matches `findTaggedValue` value type dispatch (`IMOD/pysrc/batchruntomo:242`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TaggedValueType {
    String,
    Integer,
    Float,
}

/// Matches `findTaggedValue` (`IMOD/pysrc/batchruntomo:242`).
pub fn find_tagged_value(
    lines: &[String],
    tag: &str,
    separator: char,
    value_type: TaggedValueType,
) -> Option<TaggedValue> {
    for line in lines {
        let Some(index) = line.find(separator) else {
            continue;
        };
        if index == 0 || !line.contains(tag) || index + separator.len_utf8() >= line.len() {
            continue;
        }
        let text = line[index + separator.len_utf8()..].trim();
        match value_type {
            TaggedValueType::String => return Some(TaggedValue::String(text.to_owned())),
            TaggedValueType::Integer => {
                if let Some(value) = text
                    .split_whitespace()
                    .next()
                    .and_then(|word| word.parse().ok())
                {
                    return Some(TaggedValue::Integer(value));
                }
            }
            TaggedValueType::Float => {
                if let Some(value) = text
                    .split_whitespace()
                    .next()
                    .and_then(|word| word.parse().ok())
                {
                    return Some(TaggedValue::Float(value));
                }
            }
        }
    }
    None
}

/// Matches `translateParallelPath` (`IMOD/pysrc/batchruntomo:283`).
pub fn translate_parallel_path(
    path: &str,
    from_paths: &[String],
    to_paths: &[String],
    last_translate_index: &mut i32,
) -> String {
    *last_translate_index = -1;
    for (index, from) in from_paths.iter().enumerate() {
        if path.starts_with(from) {
            *last_translate_index = index as i32;
            return format!(
                "{}{}",
                to_paths.get(index).map(String::as_str).unwrap_or(""),
                &path[from.len()..]
            );
        }
    }
    path.to_owned()
}

/// Matches `reverseTranslatePath` (`IMOD/pysrc/batchruntomo:300`).
pub fn reverse_translate_path(
    path: &str,
    from_paths: &[String],
    to_paths: &[String],
    translation_index: i32,
) -> String {
    if translation_index >= 0 {
        if let Some(to) = to_paths.get(translation_index as usize) {
            if path.starts_with(to) {
                return format!(
                    "{}{}",
                    from_paths
                        .get(translation_index as usize)
                        .map(String::as_str)
                        .unwrap_or(""),
                    &path[to.len()..]
                );
            }
        }
    }
    path.to_owned()
}

/// Matches `findPossibleStacks` (`IMOD/pysrc/batchruntomo:314`).
pub fn find_possible_stacks(
    stack_root: &str,
    already_stack_ext: &str,
    possible_stack_exts: &[String],
    standard_extension_count: usize,
    from_extension: &str,
    expected_extension: &str,
) -> (i32, i32, i32) {
    let mut first = -1;
    let mut second = -1;
    let mut first_size = 0;
    let mut second_size = 0;
    for (index, extension) in possible_stack_exts.iter().enumerate() {
        if !from_extension.is_empty()
            && from_extension != extension
            && (already_stack_ext.is_empty() || already_stack_ext != extension)
        {
            continue;
        }
        let name = format!("{stack_root}{extension}");
        if let Ok(mut file) = std::fs::File::open(&name) {
            use std::io::Read;
            let mut header = [0u8; 12];
            if file.read_exact(&mut header).is_ok() {
                let z = i32::from_le_bytes(header[8..12].try_into().expect("MRC header slice"));
                if first < 0 {
                    first = index as i32;
                    first_size = z;
                } else if second < 0
                    || (!expected_extension.is_empty() && expected_extension == extension)
                {
                    second = index as i32;
                    second_size = z;
                }
            }
        }
        if index + 1 == standard_extension_count && first >= 0 {
            break;
        }
    }
    if second >= 0 && first_size == 1 {
        first = second;
        first_size = second_size;
        second = -1;
    }
    if second >= 0 && second_size == 1 {
        second = -1;
    }
    (first, second, first_size)
}

/// Matches `checkRenameStack` (`IMOD/pysrc/batchruntomo:361`).
pub fn check_rename_stack(
    stack: &str,
    extension_already_set: &str,
    possible_extensions: &[String],
    standard_extension_count: usize,
    type_extension: bool,
    stack_extension: &mut String,
    from_extension: &mut String,
    original_stack_extension: &mut String,
) -> Result<(), String> {
    let (first, second, _) = find_possible_stacks(
        stack,
        extension_already_set,
        possible_extensions,
        standard_extension_count,
        from_extension,
        stack_extension.trim_start_matches('.'),
    );
    if first < 0 {
        return Err(if from_extension.is_empty() {
            format!("Stack file does not exist with any allowed extension: {stack}")
        } else {
            format!("Stack file does not exist: {stack}{from_extension}")
        });
    }
    let extension = possible_extensions
        .get(first as usize)
        .ok_or_else(|| "Invalid stack extension index".to_owned())?;
    if second >= 0 {
        return Err(format!(
            "There are two possible stack files in the dataset directory: {stack}{extension} and {stack}{}",
            possible_extensions[second as usize]
        ));
    }
    if original_stack_extension.is_empty() {
        *original_stack_extension = extension.clone();
    }
    from_extension.clear();
    if stack_extension.is_empty() && (type_extension || first != 0) {
        *stack_extension = format!(".{extension}");
        return Ok(());
    }
    if (first > 0 && !type_extension)
        || (type_extension
            && !stack_extension.is_empty()
            && stack_extension.trim_start_matches('.') != extension)
    {
        let old_name = format!("{stack}{extension}");
        let new_name = if stack_extension.is_empty() {
            format!("{stack}st")
        } else {
            format!("{stack}{}", stack_extension.trim_start_matches('.'))
        };
        if Path::new(&new_name).exists() {
            return Err(format!(
                "Cannot rename stack from {old_name} to {new_name} because a single-image file with that name already exists"
            ));
        }
        fs::rename(&old_name, &new_name).map_err(|error| {
            format!("Error renaming stack from {old_name} to {new_name}: {error}")
        })?;
        if stack_extension.is_empty() {
            *stack_extension = ".st".to_owned();
        }
    }
    Ok(())
}

/// Matches `testForSymLink` (`IMOD/pysrc/batchruntomo:416`).
pub fn test_for_sym_link(make_symbolic_links: &mut Option<bool>, directive_value: bool) -> bool {
    if make_symbolic_links.is_none() {
        *make_symbolic_links = Some(directive_value && !cfg!(windows));
    }
    make_symbolic_links.unwrap_or(false)
}

/// Matches `deliverAncillary` (`IMOD/pysrc/batchruntomo:432`).
pub fn deliver_ancillary(
    source: &Path,
    destination: &Path,
    make_symbolic_links: bool,
) -> Result<(), String> {
    if source.exists() && !destination.exists() {
        if make_symbolic_links {
            #[cfg(unix)]
            std::os::unix::fs::symlink(source, destination)
                .map_err(|error| format!("Error moving file {}: {error}", source.display()))?;
            #[cfg(not(unix))]
            return Err("Symbolic links are unavailable on this platform".to_owned());
        } else {
            fs::rename(source, destination)
                .map_err(|error| format!("Error moving file {}: {error}", source.display()))?;
        }
    }
    Ok(())
}

/// Matches `edfDelAndAdd` (`IMOD/pysrc/batchruntomo:639`).
pub fn edf_del_and_add(option: &str, value: &str, delim: char) -> Vec<String> {
    vec![
        format!("{delim}^{option}{delim}d"),
        format!("{delim}^Setupset{delim}a{delim}{option} = {value}{delim}"),
    ]
}

/// Matches `boolStringForEdf` (`IMOD/pysrc/batchruntomo:646`).
pub fn bool_string_for_edf(value: bool) -> String {
    if value { "true" } else { "false" }.to_owned()
}

/// Matches `getQueueOptions` (`IMOD/pysrc/batchruntomo:573`).
pub fn get_queue_options(
    queue_option: Option<&str>,
    queue_environment: Option<&str>,
    max_option: Option<i32>,
    max_environment: Option<&str>,
    gpu_text: &str,
    cpu_list: &str,
    cores_per_cluster_job: i32,
    parallel_root: bool,
) -> Result<(Option<String>, bool, i32), String> {
    let (command, running_on_it) = match queue_environment {
        Some("None") => (None, false),
        Some(value) if !value.is_empty() => (Some(value.to_owned()), true),
        _ => (
            queue_option
                .filter(|value| !value.is_empty())
                .map(str::to_owned),
            false,
        ),
    };
    let Some(command) = command else {
        return Ok((None, false, 0));
    };
    if !cpu_list.is_empty() && cores_per_cluster_job == 0 {
        return Err("You cannot enter a CPU machine list with a queue command".to_owned());
    }
    if !parallel_root {
        return Err(
            "Use of a cluster queue is allowed only when running batch in parallel".to_owned(),
        );
    }
    let max = match max_environment {
        Some(value) => value
            .parse()
            .map_err(|_| "Converting queue environment maximum to integer".to_owned())?,
        None => max_option.unwrap_or(0),
    };
    if max <= 0 {
        return Err(format!(
            "Maximum number of {gpu_text}jobs must be entered if a {gpu_text}cluster command is entered"
        ));
    }
    Ok((Some(command), running_on_it, max))
}

/// Matches `getClusterJobOption` (`IMOD/pysrc/batchruntomo:609`).
pub fn get_cluster_job_option(
    option_name: &str,
    option: Option<i32>,
    environment: Option<&str>,
    parallel_root: bool,
) -> Result<i32, String> {
    let value = match environment {
        Some("None") => return Ok(0),
        Some(text) => text
            .parse()
            .map_err(|_| format!("Environment variable for {option_name} must be an integer"))?,
        None => match option {
            Some(value) => value,
            None => return Ok(0),
        },
    };
    if value <= 0 {
        return Err(format!("The value for {option_name} must be positive"));
    }
    if !parallel_root {
        return Err(
            "Cluster node options can be entered only when running batch in parallel".to_owned(),
        );
    }
    Ok(value)
}

/// Matches `checkExcludedViewsRemoved` (`IMOD/pysrc/batchruntomo:682`).
pub fn check_excluded_views_removed(set_root: &str, exclude_list: &str) -> Result<i32, String> {
    let path = Path::new(set_root);
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let stem = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    let mut candidates = fs::read_dir(parent)
        .map_err(|error| error.to_string())?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|entry| {
            entry
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| {
                    name.starts_with(&format!("{stem}_cutviews")) && name.ends_with(".info")
                })
        })
        .collect::<Vec<_>>();
    candidates.sort();
    let Some(file) = candidates.last() else {
        return Ok(0);
    };
    let contents = fs::read_to_string(file).map_err(|error| error.to_string())?;
    let Some(last) = contents.lines().last() else {
        return Err("Cutviews file is empty".to_owned());
    };
    let cut_text = last.replace(',', " ");
    let exclude_text = exclude_list.replace(',', " ");
    let cuts = cut_text.split_whitespace().collect::<Vec<_>>();
    let excluded = exclude_text.split_whitespace().collect::<Vec<_>>();
    Ok((cuts.len() == excluded.len() && cuts.iter().all(|cut| excluded.contains(cut))) as i32)
}

/// A directly representable directive dictionary entry from `readDirectiveOrTemplate`.
pub type DirectiveDictionary = std::collections::BTreeMap<String, (String, usize)>;

/// Matches the general directive conversion portion of `readDirectiveOrTemplate`
/// (`IMOD/pysrc/batchruntomo:720`).
pub fn read_directive_or_template(
    filename: &Path,
) -> Result<(Vec<String>, DirectiveDictionary), Vec<String>> {
    let text = fs::read_to_string(filename).map_err(|error| {
        vec![format!(
            "Error opening directive/template file {}: {error}",
            filename.display()
        )]
    })?;
    let lines = text.lines().map(str::to_owned).collect::<Vec<_>>();
    let mut dictionary = DirectiveDictionary::new();
    let mut errors = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();
        if trimmed.starts_with('#') || trimmed.is_empty() {
            continue;
        }
        let Some((key, value)) = trimmed.split_once('=') else {
            errors.push(format!(
                "Directive from {} lacks an = separator: {line}",
                filename.display()
            ));
            continue;
        };
        if key.trim().is_empty() {
            errors.push(format!(
                "Directive from {} lacks a key: {line}",
                filename.display()
            ));
        } else {
            dictionary.insert(key.trim().to_owned(), (value.trim().to_owned(), index));
        }
    }
    if errors.is_empty() {
        Ok((lines, dictionary))
    } else {
        Err(errors)
    }
}

/// Matches `getOneValueAfterToken` (`IMOD/pysrc/batchruntomo:1319`).
pub fn get_one_value_after_token(line: &str, token: char, integer: bool) -> Result<f64, String> {
    let value = line
        .split_once(token)
        .ok_or_else(|| format!("Token {token} is absent"))?
        .1
        .trim();
    if integer {
        value
            .parse::<i32>()
            .map(f64::from)
            .map_err(|error| error.to_string())
    } else {
        value.parse::<f64>().map_err(|error| error.to_string())
    }
}

/// Matches `getReconTypes` (`IMOD/pysrc/batchruntomo:1658`).
pub fn get_recon_types(
    do_sirt: bool,
    fake_sirt: Option<&str>,
    do_backproj_also: bool,
    do_regular_backproj_also: bool,
) -> (bool, Option<String>, bool) {
    let fake = fake_sirt
        .filter(|value| !value.is_empty())
        .map(str::to_owned);
    (
        do_sirt,
        fake.clone(),
        (do_sirt && do_backproj_also)
            || (fake.is_some() && (do_regular_backproj_also || do_backproj_also)),
    )
}

/// Matches `getSIRTrecName` (`IMOD/pysrc/batchruntomo:1674`).
pub fn get_sirt_rec_name(
    rec_root: &str,
    do_sirt: bool,
    fake_sirt: Option<&str>,
    do_both: bool,
    leave_iterations: Option<&str>,
) -> (bool, Option<String>, bool, Option<String>, Option<String>) {
    let rec = if do_sirt {
        leave_iterations
            .map(|list| {
                let last = list
                    .replace('-', ",")
                    .split(',')
                    .last()
                    .unwrap_or("0")
                    .to_owned();
                format!("{rec_root}.srec{:0>2}", last)
            })
            .or_else(|| Some("1".to_owned()))
    } else {
        None
    };
    let fake = fake_sirt
        .filter(|value| !value.is_empty())
        .map(|_| format!("{rec_root}.{}rec", if do_both { "slf" } else { "" }));
    (do_sirt, fake_sirt.map(str::to_owned), do_both, rec, fake)
}

/// Matches `needStep` (`IMOD/pysrc/batchruntomo:1715`).
pub fn need_step(step: f64, starting_step: f64, ending_step: f64) -> bool {
    step >= starting_step - 0.005 && step <= ending_step + 0.005
}

/// Matches `parseTomopitchLog` (`IMOD/pysrc/batchruntomo:1589`).  The log is
/// explicit here rather than obtained through the Python module-global axis letter.
pub fn parse_tomopitch_log(
    lines: &[String],
    no_x_axis_tilt: bool,
    position_sample_type: i32,
    values: &mut [f64; 4],
) -> i32 {
    if position_sample_type <= 0 {
        return 1;
    }
    let tags = [
        "x-tilted  lines",
        "X axis tilt -",
        "Angle offset -",
        "Z shift -",
    ];
    let mut original = [0.; 4];
    let mut untilted = [0.; 4];
    let mut all_lines = 0;
    for line in lines {
        if line.contains("ERROR: ") {
            return 2;
        }
        for (index, tag) in tags.iter().enumerate() {
            if line.contains(tag) {
                let Some(value) = line
                    .split_whitespace()
                    .last()
                    .and_then(|word| word.parse::<f64>().ok())
                else {
                    return -1;
                };
                values[index] = value;
                if no_x_axis_tilt {
                    let words = line.split_whitespace().collect::<Vec<_>>();
                    if let Some(place) = words.iter().position(|word| *word == "Original:") {
                        if let Some(value) = words.get(place + 1).and_then(|word| word.parse().ok())
                        {
                            original[index] = value;
                        }
                    }
                }
            }
        }
        if no_x_axis_tilt && line.contains("all line pairs") {
            all_lines = 1;
        } else if all_lines == 2 {
            let Some((_, rest)) = line.split_once("add") else {
                return -1;
            };
            let Some((number, _)) = rest.split_once("to") else {
                return -1;
            };
            untilted[2] = match number.trim().parse() {
                Ok(value) => value,
                Err(_) => return -1,
            };
            all_lines += 1;
        } else if all_lines == 3 {
            untilted[0] = match line
                .split_whitespace()
                .last()
                .and_then(|word| word.parse().ok())
            {
                Some(value) => value,
                None => return -1,
            };
            let Some((_, rest)) = line.split_once("shift of") else {
                return -1;
            };
            let Some((number, _)) = rest.split_once(';') else {
                return -1;
            };
            untilted[3] = match number.trim().parse() {
                Ok(value) => value,
                Err(_) => return -1,
            };
            all_lines += 1;
        } else if all_lines > 0 {
            all_lines += 1;
        }
    }
    if no_x_axis_tilt && all_lines > 0 {
        for index in 0..4 {
            values[index] = original[index] + untilted[index];
        }
    }
    0
}

/// Matches the `valType` cases in `lookupDirective` (`IMOD/pysrc/batchruntomo:925`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DirectiveValueType {
    Boolean,
    Integer,
    Float,
    String,
}

/// Matches `lookupDirective` (`IMOD/pysrc/batchruntomo:925`).
pub fn lookup_directive(
    dictionaries: &[DirectiveDictionary],
    prefix: &str,
    option: &str,
    start_dictionary: usize,
    batch_dictionary: usize,
    axis_number: usize,
    dual_axis: bool,
    value_type: DirectiveValueType,
) -> Result<Option<String>, String> {
    let keys = if prefix.starts_with("comparam.") {
        vec![
            format!("{prefix}.{option}"),
            format!("{prefix}a.{option}"),
            format!("{prefix}b.{option}"),
        ]
    } else if prefix == "setupset." {
        vec![format!("{prefix}{option}"); 3]
    } else {
        vec![
            format!("{prefix}.any.{option}"),
            format!("{prefix}.a.{option}"),
            format!("{prefix}.b.{option}"),
        ]
    };
    let checks: &[usize] = if axis_number == 2 { &[0, 2] } else { &[0, 1] };
    let mut selected: Option<(usize, usize, usize)> = None;
    for dictionary in start_dictionary..=batch_dictionary.min(dictionaries.len().saturating_sub(1))
    {
        for &key in checks {
            if let Some((_, line)) = dictionaries[dictionary].get(&keys[key]) {
                let better = match selected {
                    None => true,
                    Some((old_dictionary, old_line, old_key)) if dual_axis => {
                        (key == axis_number && old_key != axis_number)
                            || (key == old_key && (dictionary > old_dictionary || *line > old_line))
                    }
                    Some((old_dictionary, old_line, _)) => {
                        dictionary > old_dictionary || *line > old_line
                    }
                };
                if better {
                    selected = Some((dictionary, *line, key));
                }
            }
        }
    }
    let Some((dictionary, _, key)) = selected else {
        return Ok(if value_type == DirectiveValueType::Boolean {
            Some("0".to_owned())
        } else {
            None
        });
    };
    let value = dictionaries[dictionary][&keys[key]].0.clone();
    match value_type {
        DirectiveValueType::Boolean => Ok(Some(if value == "1" { "1" } else { "0" }.to_owned())),
        DirectiveValueType::Integer => {
            if value.is_empty() {
                Ok(None)
            } else {
                value
                    .parse::<i32>()
                    .map(|_| Some(value))
                    .map_err(|_| "ERROR".to_owned())
            }
        }
        DirectiveValueType::Float => {
            if value.is_empty() {
                Ok(None)
            } else {
                value
                    .parse::<f64>()
                    .map(|_| Some(value))
                    .map_err(|_| "ERROR".to_owned())
            }
        }
        DirectiveValueType::String => Ok(Some(value)),
    }
}

/// Matches `laterComDirectives` (`IMOD/pysrc/batchruntomo:989`).
pub fn later_com_directives(
    start_index: usize,
    defaults_file: &str,
    directive_file: &str,
    batch: &DirectiveDictionary,
) -> Vec<String> {
    let mut lines = Vec::new();
    if start_index < 1 {
        lines.push(format!("ChangeParametersFile {defaults_file}"));
    }
    for (index, name) in [
        (1, "setupset.scopeTemplate"),
        (2, "setupset.systemTemplate"),
        (3, "setupset.userTemplate"),
    ] {
        if start_index <= index {
            if let Some((value, _)) = batch.get(name) {
                lines.push(format!("ChangeParametersFile {value}"));
            }
        }
    }
    lines.push(format!("ChangeParametersFile {directive_file}"));
    lines
}

/// Matches `readEdfFileIfNeeded` (`IMOD/pysrc/batchruntomo:653`).
pub fn read_edf_file_if_needed(
    edf_lines: &mut Vec<String>,
    set_name: &str,
    bypass_etomo: bool,
) -> Result<(), String> {
    if !edf_lines.is_empty() {
        return Ok(());
    }
    match fs::read_to_string(format!("{set_name}.edf")) {
        Ok(text) => {
            *edf_lines = text.lines().map(str::to_owned).collect();
            Ok(())
        }
        Err(_) if bypass_etomo => {
            *edf_lines = vec![format!("Setup.DatasetName={set_name}")];
            Ok(())
        }
        Err(error) => Err(format!("Error opening {set_name}.edf: {error}")),
    }
}

/// Matches `modifyEdfLines` (`IMOD/pysrc/batchruntomo:668`).
pub fn modify_edf_lines(
    edf_lines: &mut Vec<String>,
    edf_changed: &mut bool,
    set_name: &str,
    bypass_etomo: bool,
    sed_commands: &[String],
) -> Result<(), String> {
    read_edf_file_if_needed(edf_lines, set_name, bypass_etomo)?;
    *edf_lines = super::pysed::pysed(sed_commands, edf_lines, false, '/')?;
    *edf_changed = true;
    Ok(())
}

/// Explicit state formerly read from Python module globals by `runOneProcess`.
#[derive(Clone, Debug)]
pub struct ProcessRunOptions {
    pub niceness: i32,
    pub axis_letter: String,
    pub cpu_list: String,
    pub remote_data_dir: Option<String>,
    pub first_cpu_limit: i32,
    pub local_cpu_limit: i32,
    pub use_first_cpu_for_single: bool,
    pub queue_command: Option<String>,
    pub max_queue_jobs: i32,
    pub running_on_queue: bool,
    pub gpu_queue_command: Option<String>,
    pub max_gpu_queue_jobs: i32,
    pub gpu_list: String,
    pub top_check_file: Option<String>,
    pub processchunks_check_file: String,
}

/// Matches `runOneProcess` (`IMOD/pysrc/batchruntomo:1067`) through the original
/// `processchunks` process boundary.
pub fn run_one_process(
    comfile: &str,
    single: bool,
    using_gpu: bool,
    message: &str,
    options: &ProcessRunOptions,
) -> Result<i32, String> {
    if (single || (!comfile.contains("sirt") && !comfile.contains("ctf3d")))
        && !Path::new(comfile).exists()
    {
        return Err(format!("Command file {comfile} does not exist"));
    }
    let root = Path::new(comfile)
        .file_stem()
        .and_then(|item| item.to_str())
        .unwrap_or(comfile);
    let mut command = vec![
        std::ffi::OsString::from("-n"),
        std::ffi::OsString::from(options.niceness.to_string()),
    ];
    if let Some(remote) = &options.remote_data_dir {
        command.extend([
            std::ffi::OsString::from("-w"),
            std::ffi::OsString::from(remote),
        ]);
    }
    let mut machines = options.cpu_list.clone();
    let process_file = if single {
        command.extend([
            std::ffi::OsString::from("-s"),
            std::ffi::OsString::from("-e"),
            std::ffi::OsString::from("1"),
        ]);
        let threads = if options.use_first_cpu_for_single {
            options.first_cpu_limit
        } else {
            machines = "1".to_owned();
            options.local_cpu_limit
        };
        if threads > 0 {
            command.extend([
                std::ffi::OsString::from("-O"),
                std::ffi::OsString::from(threads.to_string()),
            ]);
        }
        comfile.to_owned()
    } else {
        root.to_owned()
    };
    if let Some(queue) = &options.queue_command {
        if !using_gpu && !(single && options.running_on_queue) {
            machines = queue.clone();
            command.extend([
                std::ffi::OsString::from("-q"),
                std::ffi::OsString::from(options.max_queue_jobs.to_string()),
            ]);
        }
    }
    if using_gpu {
        if let Some(queue) = &options.gpu_queue_command {
            machines = queue.clone();
            command.extend([
                std::ffi::OsString::from("-q"),
                std::ffi::OsString::from(options.max_gpu_queue_jobs.to_string()),
            ]);
        } else {
            machines = options.gpu_list.clone();
            if machines != "1" {
                command.push(std::ffi::OsString::from("-G"));
            }
        }
    }
    if machines.is_empty() {
        machines = "1".to_owned();
    }
    command.extend([
        std::ffi::OsString::from(machines),
        std::ffi::OsString::from(process_file),
    ]);
    let outfile = format!("processchunks{}.out", options.axis_letter);
    let (error, finished, _, _, detail) = super::prochunks::run_processchunks(
        &command,
        &outfile,
        options.top_check_file.as_deref(),
        &options.processchunks_check_file,
        false,
        false,
    );
    if error != 0 {
        return Err(if detail.is_empty() {
            format!("Unable to run {comfile}: {message}")
        } else {
            detail
        });
    }
    Ok(finished)
}

/// Matches `makeAndRunOneCom` (`IMOD/pysrc/batchruntomo:1281`).
pub fn make_and_run_one_com(
    comlines: &mut Vec<String>,
    comfile: &str,
    message: &str,
    use_gpu: bool,
    skip_run: bool,
    naming_style: i32,
    stack_extension: &str,
    options: &ProcessRunOptions,
) -> Result<i32, String> {
    comlines.insert(0, format!("OutputFile\t{comfile}"));
    comlines.push(format!("NamingStyle\t{naming_style}"));
    comlines.push(format!(
        "StackExtension\t{}",
        stack_extension.trim_start_matches('.')
    ));
    super::imodpy::run_cmd(
        "makecomfile -StandardInput",
        Some(comlines),
        None,
        None,
        &[],
    )
    .map_err(|error| error.arguments.join("\n"))?;
    if skip_run {
        Ok(0)
    } else {
        run_one_process(comfile, true, use_gpu, message, options)
    }
}

/// Matches `modifyWriteAndRunCom` (`IMOD/pysrc/batchruntomo:1301`).
pub fn modify_write_and_run_com(
    comfile: &str,
    sed_commands: &[String],
    input_lines: Option<Vec<String>>,
    message: &str,
    skip_run: bool,
    options: &ProcessRunOptions,
) -> Result<i32, String> {
    let source = match input_lines {
        Some(lines) => lines,
        None => fs::read_to_string(comfile)
            .map_err(|error| format!("Opening {comfile}: {error}"))?
            .lines()
            .map(str::to_owned)
            .collect(),
    };
    let changed = super::pysed::pysed(sed_commands, &source, false, '/')?;
    fs::write(comfile, changed.join("\n") + "\n")
        .map_err(|error| format!("Writing {comfile}: {error}"))?;
    if skip_run {
        Ok(0)
    } else {
        run_one_process(comfile, true, false, message, options)
    }
}

/// Matches `processQuitAction` (`IMOD/pysrc/batchruntomo:1044`).
pub fn process_quit_action(
    action: &str,
    message: &str,
    parallel_root: bool,
    finish_set_and_quit: &mut bool,
) -> Result<(), String> {
    match action {
        "" => Ok(()),
        "Q" => Err(if message.is_empty() {
            if parallel_root {
                "RECEIVED SIGNAL TO QUIT, JUST EXITING"
            } else {
                "RECEIVED SIGNAL TO QUIT, JUST EXITING"
            }
        } else {
            message
        }
        .to_owned()),
        "F" => {
            *finish_set_and_quit = true;
            Ok(())
        }
        _ => Ok(()),
    }
}

/// Matches `checkForQuit` (`IMOD/pysrc/batchruntomo:1061`).
pub fn check_for_quit(
    check_file: Option<&str>,
    processchunks_check_file: Option<&str>,
    parallel_root: bool,
    finish_set_and_quit: &mut bool,
) -> Result<(), String> {
    let action = super::prochunks::check_for_pro_chunks_quit(
        check_file,
        processchunks_check_file,
        false,
        false,
        finish_set_and_quit,
    );
    process_quit_action(&action, "", parallel_root, finish_set_and_quit)
}

/// Matches `manageGPUallocation` (`IMOD/pysrc/batchruntomo:1185`), retaining
/// `gpuallocator` as the source external allocation authority.
pub fn manage_gpu_allocation(
    use_gpu: bool,
    max_parallel_gpus: i32,
    queue_command: bool,
    cores_per_cluster_job: i32,
    parallel_root: &str,
    gpu_list: &str,
    starting_directory: &str,
    maximum: i32,
    host: &str,
    process_id: u32,
) -> Result<Vec<String>, String> {
    if !use_gpu || max_parallel_gpus == 0 || queue_command || cores_per_cluster_job != 0 {
        return Ok(if use_gpu {
            gpu_list.split(',').map(str::to_owned).collect()
        } else {
            Vec::new()
        });
    }
    loop {
        let command = format!(
            "gpuallocator -root {parallel_root} -full \"{gpu_list}\" -comm \"{starting_directory}\" -max {maximum} -con {host} -pid {process_id}"
        );
        match super::imodpy::run_cmd(&command, None, None, None, &[]) {
            Ok(Some(lines)) if !lines.is_empty() => {
                return Ok(lines
                    .into_iter()
                    .map(|line| line.trim().to_owned())
                    .collect());
            }
            Ok(_) => return Err("Gpuallocator gave no output, cannot proceed".to_owned()),
            Err(error) => {
                if error.arguments.iter().any(|line| line.contains("[GPA1]")) {
                    std::thread::sleep(std::time::Duration::from_secs(5));
                } else {
                    return Err(error.arguments.join("\n"));
                }
            }
        }
    }
}

/// Matches `releaseGPUallocation` (`IMOD/pysrc/batchruntomo:1249`).
pub fn release_gpu_allocation(
    needs_release: &mut bool,
    parallel_root: &str,
    starting_directory: &str,
    host: &str,
    process_id: u32,
) -> Result<(), String> {
    if !*needs_release {
        return Ok(());
    }
    let command = format!(
        "gpuallocator -root {parallel_root} -comm \"{starting_directory}\" -con {host} -pid {process_id}"
    );
    let result = super::imodpy::run_cmd(&command, None, None, None, &[])
        .map_err(|error| error.arguments.join("\n"));
    *needs_release = false;
    result.map(|_| ())
}

/// Matches `numberOfProcessingUnits` (`IMOD/pysrc/batchruntomo:1266`) after GPU allocation.
pub fn number_of_processing_units(
    parallel_gpu: i32,
    queue_command: bool,
    gpu_queue_command: bool,
    max_queue_jobs: i32,
    parallel_cpu: i32,
    use_gpu: bool,
) -> i32 {
    if parallel_gpu > 1 {
        parallel_gpu
    } else if queue_command && !gpu_queue_command {
        max_queue_jobs
    } else if parallel_cpu > 0 && !use_gpu {
        parallel_cpu
    } else {
        1
    }
}

/// Matches `comAndProcessForAlignedStack` (`IMOD/pysrc/batchruntomo:1514`).
pub fn com_and_process_for_aligned_stack(prefix: &str, montage: bool) -> (String, String) {
    if montage {
        (format!("{prefix}blend"), "blendmont".to_owned())
    } else {
        (format!("{prefix}newst"), "newstack".to_owned())
    }
}

/// Matches `splitAndRunTilt` (`IMOD/pysrc/batchruntomo:1525`), retaining
/// `splittilt` as the original external command boundary.
pub fn split_and_run_tilt(
    comfile: &str,
    message: &str,
    mut num_processes: i32,
    using_gpu: bool,
    options: &ProcessRunOptions,
    parallel_gpu: i32,
    queue_command: bool,
    gpu_queue_command: bool,
    max_queue_jobs: i32,
    parallel_cpu: i32,
) -> Result<i32, String> {
    if num_processes == 0 {
        num_processes = number_of_processing_units(
            parallel_gpu,
            queue_command,
            gpu_queue_command,
            max_queue_jobs,
            parallel_cpu,
            using_gpu,
        );
    }
    if num_processes < 1 {
        return Err("No processing units are available".to_owned());
    }
    if num_processes > 1 {
        super::imodpy::run_cmd(
            &format!("splittilt -n {num_processes} {comfile}"),
            None,
            None,
            None,
            &[],
        )
        .map_err(|error| error.arguments.join("\n"))?;
    }
    run_one_process(comfile, num_processes < 2, using_gpu, message, options)
}

/// Matches `runClipStats` (`IMOD/pysrc/batchruntomo:1700`).
pub fn run_clip_stats(
    command: &str,
    data_name: &str,
    stack_suffix: &str,
    description: &str,
    com_extension: &str,
    options: &ProcessRunOptions,
) -> Result<Vec<String>, String> {
    let comfile = format!("{data_name}{stack_suffix}.st.stats{com_extension}");
    fs::write(&comfile, format!("{command}\n"))
        .map_err(|error| format!("Writing {comfile}: {error}"))?;
    let status = run_one_process(
        &comfile,
        true,
        false,
        &format!("Getting statistics for {description} stack"),
        options,
    );
    let _ = fs::remove_file(&comfile);
    status?;
    fs::read_to_string(format!("{data_name}{stack_suffix}.st.stats.log"))
        .map_err(|error| format!("Opening clip statistics log: {error}"))
        .map(|text| text.lines().map(str::to_owned).collect())
}

/// Matches the tuple entries consumed by `printTaggedMessages`.
pub type MessageTag = (String, i32, Option<String>);

/// Matches `printTaggedMessages` (`IMOD/pysrc/batchruntomo:159`).
pub fn print_tagged_messages(lines: &[String], tags: &[MessageTag]) -> Vec<String> {
    let mut output = Vec::new();
    let mut reading_multi = false;
    let mut any_error = false;
    for line in lines {
        if reading_multi {
            output.push(line.clone());
            if line.trim().is_empty() {
                reading_multi = false;
            }
            continue;
        }
        for (tag, flags, suffix) in tags {
            let matched = if let Some((left, right)) = tag.split_once(".*") {
                if flags & 4 != 0 {
                    line.contains(left) && line.contains(right)
                } else {
                    line.starts_with(left) && line.contains(right)
                }
            } else if flags & 4 != 0 {
                line.contains(tag)
            } else {
                line.starts_with(tag) && !line.contains("No GUI")
            };
            if !matched {
                continue;
            }
            if any_error && line.starts_with("ERROR:") && line.ends_with("exited with status 1") {
                break;
            }
            if tag.starts_with("ERROR") {
                any_error = true;
            }
            let mut rendered = if flags & 2 != 0 {
                line[tag.len()..].trim_start().to_owned()
            } else {
                line.clone()
            };
            if let Some(suffix) = suffix {
                rendered.push_str(suffix);
            }
            output.push(rendered);
            if flags & 1 != 0 {
                reading_multi = true;
            }
            break;
        }
    }
    output
}

/// Matches `writeTextFileReportErr` (`IMOD/pysrc/batchruntomo:264`).
pub fn write_text_file_report_err(filename: &str, lines: &[String]) -> Result<(), String> {
    fs::write(filename, lines.join("\n") + "\n")
        .map_err(|error| format!("Error writing {filename}: {error}"))
}

/// Matches `readTextFileReportErr` (`IMOD/pysrc/batchruntomo:272`).
pub fn read_text_file_report_err(
    filename: &str,
    message: Option<&str>,
) -> Result<Vec<String>, String> {
    fs::read_to_string(filename)
        .map_err(|error| format!("{}{}", message.unwrap_or("Error reading "), error))
        .map(|text| text.lines().map(str::to_owned).collect())
}

/// Matches `useFileAsReplacement` (`IMOD/pysrc/batchruntomo:1005`).
pub fn use_file_as_replacement(
    use_file: &Path,
    old_file: &Path,
    save_original: bool,
    make_backup: bool,
) -> Result<(), String> {
    let stem = old_file
        .file_stem()
        .and_then(|item| item.to_str())
        .unwrap_or("");
    let extension = old_file
        .extension()
        .and_then(|item| item.to_str())
        .map(|item| format!(".{item}"))
        .unwrap_or_default();
    let original = old_file.with_file_name(format!("{stem}_orig{extension}"));
    if save_original && !original.exists() {
        fs::rename(old_file, &original).map_err(|error| {
            format!(
                "Error renaming {} to {}: {error}",
                old_file.display(),
                original.display()
            )
        })?;
    } else if make_backup {
        let backup = old_file.with_file_name(format!(
            "{}~",
            old_file
                .file_name()
                .and_then(|item| item.to_str())
                .unwrap_or("")
        ));
        if old_file.exists() {
            fs::rename(old_file, &backup)
                .map_err(|error| format!("Error making backup {}: {error}", old_file.display()))?;
        }
    } else if old_file.exists() {
        fs::remove_file(old_file)
            .map_err(|error| format!("Error removing {}: {error}", old_file.display()))?;
    }
    fs::rename(use_file, old_file).map_err(|error| {
        format!(
            "Error renaming {} to {}: {error}",
            use_file.display(),
            old_file.display()
        )
    })
}

/// Matches `reportReachedStep` (`IMOD/pysrc/batchruntomo:1720`).
pub fn report_reached_step(step: f64, starting_step: f64, ending_step: f64) -> Option<String> {
    if [0., 5., 6., 7., 10., 13., 14.].contains(&step)
        && need_step(step, starting_step, ending_step)
    {
        Some(format!("Reached step {step}"))
    } else {
        None
    }
}

/// Matches `replaceOrRunAfterStep` (`IMOD/pysrc/batchruntomo:1730`).
pub fn replace_or_run_after_step(
    step: f64,
    run_after: bool,
    starting_step: f64,
    ending_step: f64,
    command: Option<&str>,
    data_name: &str,
    axis_com_extension: &str,
    options: &ProcessRunOptions,
) -> Result<i32, String> {
    if !need_step(step, starting_step, ending_step) {
        return Ok(0);
    }
    let Some(command) = command.filter(|value| !value.is_empty()) else {
        return Ok(0);
    };
    let directive = if run_after {
        "runAfterStep"
    } else {
        "replaceStep"
    };
    let comfile = format!("r{directive}{step}{axis_com_extension}");
    fs::write(
        &comfile,
        format!("${}\n", command.replace("%{setname}", data_name)),
    )
    .map_err(|error| format!("Writing {comfile}: {error}"))?;
    run_one_process(&comfile, true, false, "", options)?;
    Ok(if run_after { 0 } else { -1 })
}

/// Matches `analyzeAlignLog` (`IMOD/pysrc/batchruntomo:1330`) after the external
/// `alignlog` command boundary has provided its output.
pub fn analyze_align_log(
    lines: &[String],
    two_surfaces: bool,
    no_x_axis_tilt: bool,
    raw_y_size: f64,
    center_on_gold: bool,
    values: &mut [f64; 8],
) -> Result<(), String> {
    values[..5].fill(0.);
    let tags = [
        "Total tilt angle change",
        "X axis tilt needed",
        "Unbinned thickness",
        "Incremental unbinned shift",
        "Total unbinned shift",
    ];
    let mut bottom = -1.;
    let mut top = -1.;
    for line in lines {
        if line.contains("# of points") {
            let number = get_one_value_after_token(line, '=', true)?;
            if bottom < 0. {
                bottom = number;
            } else if top < 0. {
                bottom = number;
                top = 0.;
            } else {
                top = number;
            }
        }
        for (index, tag) in tags.iter().enumerate() {
            if line.contains(tag) {
                values[index] = get_one_value_after_token(line, '=', false)?;
            }
        }
    }
    if no_x_axis_tilt {
        values[2] = values[2] / values[1].to_radians().cos()
            + 0.9 * raw_y_size * values[1].to_radians().abs().tan();
    }
    if two_surfaces {
        values[5] = values[2];
        if bottom >= 4. && top >= 4. && !center_on_gold {
            values[5] -= values[4].abs();
        }
    }
    values[6] = bottom;
    values[7] = top.max(0.);
    Ok(())
}

/// Matches `montageFrameValues` (`IMOD/pysrc/batchruntomo:1397`), retaining the
/// original `goodframe` executable boundary.
pub fn montage_frame_values(
    nx_montage: i32,
    ny_montage: i32,
    transpose: bool,
    nx_aligned: i32,
    ny_aligned: i32,
    frame_values: &mut [i32; 10],
) -> Result<i32, String> {
    let (nx_out, ny_out) = if transpose {
        (ny_montage, nx_montage)
    } else {
        (nx_montage, ny_montage)
    };
    let good_aligned = super::imodpy::run_cmd(
        &format!("goodframe {nx_aligned} {ny_aligned}"),
        None,
        None,
        None,
        &[],
    )
    .map_err(|_| "Error running goodframe".to_owned())?
    .ok_or_else(|| "No output from goodframe".to_owned())?;
    let good_raw = super::imodpy::run_cmd(
        &format!("goodframe {nx_out} {ny_out}"),
        None,
        None,
        None,
        &[],
    )
    .map_err(|_| "Error running goodframe".to_owned())?
    .ok_or_else(|| "No output from goodframe".to_owned())?;
    let parse = |line: &String| -> Result<(i32, i32), String> {
        let parts = line.split_whitespace().collect::<Vec<_>>();
        Ok((
            parts
                .first()
                .ok_or_else(|| "Bad goodframe output".to_owned())?
                .parse()
                .map_err(|_| "Bad goodframe output".to_owned())?,
            parts
                .get(1)
                .ok_or_else(|| "Bad goodframe output".to_owned())?
                .parse()
                .map_err(|_| "Bad goodframe output".to_owned())?,
        ))
    };
    let (ax, ay) = parse(
        good_aligned
            .last()
            .ok_or_else(|| "No output from goodframe".to_owned())?,
    )?;
    let (rx, ry) = parse(
        good_raw
            .last()
            .ok_or_else(|| "No output from goodframe".to_owned())?,
    )?;
    frame_values[0] = ax;
    frame_values[1] = ay;
    frame_values[6] = rx;
    frame_values[7] = ry;
    frame_values[2] = -((ax - nx_montage) / 2);
    frame_values[3] = -((ay - ny_montage) / 2);
    for index in 0..2 {
        frame_values[index + 4] = frame_values[index + 2] + frame_values[index] - 1;
        frame_values[index + 8] = -((frame_values[index] - frame_values[index + 6]) / 2);
    }
    Ok(0)
}

/// Matches `filterAlignedStack` (`IMOD/pysrc/batchruntomo:3489`).
pub fn filter_aligned_stack(
    filter_in_2d: bool,
    starting_step: f64,
    ending_step: f64,
    axis_com_extension: &str,
    aligned_binning: f64,
    pixel_size: f64,
    data_name: &str,
    skip_aligned_stack_modifications: bool,
    options: &ProcessRunOptions,
) -> Result<i32, String> {
    const STEP: f64 = 13.;
    if !filter_in_2d || !need_step(STEP, starting_step, ending_step) {
        return Ok(0);
    }
    let comfile = format!("mtffilter{axis_com_extension}");
    let lines = read_text_file_report_err(&comfile, None)?;
    let sed = vec![super::pysed::sed_modify(
        "PixelSize",
        &(aligned_binning * pixel_size).to_string(),
        '/',
    )];
    modify_write_and_run_com(
        &comfile,
        &sed,
        Some(lines),
        "2D filtering the aligned stack with Mtffilter",
        skip_aligned_stack_modifications,
        options,
    )?;
    if skip_aligned_stack_modifications {
        return Ok(0);
    }
    use_file_as_replacement(
        &PathBuf::from(format!("{data_name}_filt.ali")),
        &PathBuf::from(format!("{data_name}.ali")),
        false,
        true,
    )?;
    Ok(0)
}

/// Matches `runCleanup` (`IMOD/pysrc/batchruntomo:4707`), retaining tomocleanup as
/// its source process boundary.
pub fn run_cleanup(
    do_cleanup: bool,
    keep_aligned: bool,
    keep_untrimmed: bool,
    keep_axis: bool,
    keep_sirt: bool,
) -> Result<Vec<String>, String> {
    if !do_cleanup {
        return Ok(Vec::new());
    }
    let mut command = "tomocleanup".to_owned();
    if keep_aligned {
        command.push_str(" -aligned");
    }
    if keep_untrimmed {
        command.push_str(" -untrimmed");
    }
    if keep_axis {
        command.push_str(" -axis");
    }
    if keep_sirt {
        command.push_str(" -sirt");
    }
    command.push_str(" .");
    super::imodpy::run_cmd(&command, None, None, None, &[])
        .map_err(|error| error.arguments.join("\n"))?
        .ok_or_else(|| "tomocleanup produced no output".to_owned())
        .map(|lines| print_tagged_messages(&lines, &[("WARNING:".to_owned(), 0, None)]))
}

/// The state-owning operation interface consumed by `runOneAxis`.  Its methods map
/// directly to the original operation functions and preserve their external command
/// boundaries rather than introducing alternate implementations.
pub trait AxisWorkflow {
    fn need_step(&self, step: f64) -> bool;
    fn starting_step(&self) -> f64;
    fn ending_step(&self) -> f64;
    fn dual_axis(&self) -> bool;
    fn initial_parameters(&mut self) -> i32;
    fn xray_preprocess(&mut self) -> i32;
    fn coarse_align(&mut self) -> i32;
    fn analyze_alignment(&mut self) -> i32;
    fn fiducialless(&self) -> bool;
    fn fidless_operations(&mut self) -> i32;
    fn prealigned_stack(&mut self) -> i32;
    fn patch_tracking(&self) -> bool;
    fn run_patch_tracking(&mut self) -> i32;
    fn seed_and_track(&mut self) -> i32;
    fn adjust_patch_tracking(&mut self) -> i32;
    fn vary_patch_tracking(&mut self) -> i32;
    fn tilt_align(&mut self) -> i32;
    fn position_tomogram(&mut self) -> i32;
    fn aligned_stack(&mut self) -> i32;
    fn ctf_plot(&mut self) -> i32;
    fn modify_tilt_com(&mut self) -> i32;
    fn detect_gold(&mut self) -> i32;
    fn ctf_correct(&mut self) -> i32;
    fn erase_gold(&mut self) -> i32;
    fn filter_aligned(&mut self) -> i32;
    fn generate_tomogram(&mut self) -> i32;
    fn trim_volume(&mut self) -> i32;
    fn run_nad(&mut self) -> i32;
    fn reduce_filt_vol(&mut self) -> i32;
    fn cleanup(&mut self) -> i32;
}

/// Matches `runOneAxis` (`IMOD/pysrc/batchruntomo:4734`).
pub fn run_one_axis<Workflow: AxisWorkflow>(workflow: &mut Workflow) -> i32 {
    if workflow.initial_parameters() != 0 {
        return 1;
    }
    if workflow.need_step(1.) && workflow.xray_preprocess() != 0 {
        return 1;
    }
    if workflow.need_step(2.) && workflow.coarse_align() != 0 {
        return 1;
    }
    if workflow.analyze_alignment() != 0 {
        return 1;
    }
    if workflow.fiducialless() {
        if workflow.fidless_operations() != 0 {
            return 1;
        }
    } else {
        if workflow.need_step(3.) && workflow.prealigned_stack() != 0 {
            return 1;
        }
        if workflow.patch_tracking() && workflow.need_step(4.) {
            if workflow.run_patch_tracking() != 0 {
                return 1;
            }
        } else if workflow.seed_and_track() != 0 {
            return 1;
        }
        if workflow.patch_tracking()
            && workflow.need_step(4.)
            && workflow.adjust_patch_tracking() != 0
        {
            return 1;
        }
        if workflow.patch_tracking()
            && workflow.need_step(4.)
            && workflow.vary_patch_tracking() != 0
        {
            return 1;
        }
        if workflow.ending_step() >= 6.
            && workflow.starting_step() <= 14.
            && workflow.tilt_align() != 0
        {
            return 1;
        }
    }
    if workflow.need_step(7.) && workflow.position_tomogram() != 0 {
        return 1;
    }
    if workflow.aligned_stack() != 0 || workflow.ctf_plot() != 0 {
        return 1;
    }
    if workflow.ending_step() >= 10.
        && workflow.starting_step() <= 14.
        && workflow.modify_tilt_com() != 0
    {
        return 1;
    }
    if workflow.detect_gold() != 0 {
        return 1;
    }
    if workflow.need_step(11.) && workflow.ctf_correct() != 0 {
        return 1;
    }
    if workflow.need_step(12.) && workflow.erase_gold() != 0 {
        return 1;
    }
    if workflow.filter_aligned() != 0 {
        return 1;
    }
    if workflow.need_step(14.) && workflow.generate_tomogram() != 0 {
        return 1;
    }
    if ((!workflow.dual_axis() && workflow.need_step(20.))
        || (workflow.dual_axis() && workflow.need_step(14.5)))
        && workflow.trim_volume() != 0
    {
        return 1;
    }
    if !workflow.dual_axis() && workflow.need_step(21.) && workflow.run_nad() != 0 {
        return 1;
    }
    if !workflow.dual_axis() && workflow.need_step(21.5) && workflow.reduce_filt_vol() != 0 {
        return 1;
    }
    if !workflow.dual_axis() && workflow.need_step(22.) && workflow.cleanup() != 0 {
        return 1;
    }
    0
}

/// Matches `runCombine` (`IMOD/pysrc/batchruntomo:5010`).  Generic closures retain
/// the source ordering while allowing the source-mapped operation units to own their
/// explicit state and process boundaries.
pub fn run_combine<Setup, Initial, Align, Trim, Nad, Reduce, Cleanup>(
    need_step: impl Fn(f64) -> bool,
    mut setup: Setup,
    mut initial: Initial,
    mut align: Align,
    mut trim: Trim,
    mut nad: Nad,
    mut reduce: Reduce,
    mut cleanup: Cleanup,
) -> i32
where
    Setup: FnMut() -> i32,
    Initial: FnMut() -> i32,
    Align: FnMut() -> i32,
    Trim: FnMut() -> i32,
    Nad: FnMut() -> i32,
    Reduce: FnMut() -> i32,
    Cleanup: FnMut() -> i32,
{
    if need_step(15.) && setup() != 0 {
        return 1;
    }
    if need_step(16.) && initial() != 0 {
        return 1;
    }
    if align() != 0 {
        return 1;
    }
    if need_step(20.) && trim() != 0 {
        return 1;
    }
    if need_step(21.) && nad() != 0 {
        return 1;
    }
    if need_step(21.5) && reduce() != 0 {
        return 1;
    }
    if need_step(22.) {
        return cleanup();
    }
    0
}

/// Matches `runFinalTiltalign` (`IMOD/pysrc/batchruntomo:3780`).
pub fn run_final_tiltalign(
    sed_commands: &[String],
    message: &str,
    axis_com_extension: &str,
    align_lines: &mut Vec<String>,
    options: &ProcessRunOptions,
) -> Result<(), String> {
    let prior = std::env::var_os("TILTALIGN_SKIP_CROSS_VAL");
    if prior.is_none() {
        unsafe { std::env::set_var("TILTALIGN_SKIP_CROSS_VAL", "1") };
    }
    let result = modify_write_and_run_com(
        &format!("align{axis_com_extension}"),
        sed_commands,
        Some(align_lines.clone()),
        &format!("Doing final alignment {message}"),
        false,
        options,
    );
    if prior.is_none() {
        unsafe { std::env::remove_var("TILTALIGN_SKIP_CROSS_VAL") };
    }
    result?;
    *align_lines = super::pysed::pysed(sed_commands, align_lines, false, '/')?;
    Ok(())
}

/// Matches `transformRawBoundaryModel` (`IMOD/pysrc/batchruntomo:1026`), retaining
/// the `imodtrans` command as the source model-transform boundary.
pub fn transform_raw_boundary_model(
    model_input: &str,
    model_output: &str,
    data_name: &str,
    stack_extension: &str,
    prealigned_image: &str,
    prealigned_x: i32,
    prealigned_y: i32,
    raw_x: i32,
    raw_y: i32,
    coarse_binning: i32,
) -> Result<(), String> {
    if coarse_binning <= 0 {
        return Err("Coarse binning must be positive".to_owned());
    }
    let scale = 1. / coarse_binning as f64;
    let tx = (prealigned_x - raw_x / coarse_binning) as f64 / 2.;
    let ty = (prealigned_y - raw_y / coarse_binning) as f64 / 2.;
    let command = format!(
        "imodtrans -I \"{data_name}{stack_extension}\" -i \"{prealigned_image}\" -2 \"{data_name}.prexg\" -S {scale} -tx {tx} -ty {ty} \"{model_input}\" \"{model_output}\""
    );
    super::imodpy::run_cmd(&command, None, None, None, &[])
        .map_err(|error| error.arguments.join("\n"))
        .map(|_| ())
}

/// Matches `printDirectiveErrors` (`IMOD/pysrc/batchruntomo:708`).
pub fn print_directive_errors(errors: &[String]) -> (usize, Vec<String>) {
    if errors.is_empty() {
        return (0, Vec::new());
    }
    let mut messages = vec!["ERROR: Incorrect directive(s) as listed below:".to_owned()];
    messages.extend_from_slice(errors);
    messages.push(String::new());
    (errors.len(), messages)
}

/// Matches the validation dictionaries built by `processValidationFile`.
#[derive(Clone, Debug)]
pub struct ValidationEntry {
    pub spelling: String,
    pub template_ok: bool,
    pub batch_ok: bool,
    pub boolean: bool,
}

/// Matches `processValidationFile` (`IMOD/pysrc/batchruntomo:1753`).
pub fn process_validation_file(
    filename: &Path,
    base_com: &mut std::collections::BTreeMap<String, String>,
    valid_com: &mut std::collections::BTreeMap<String, ValidationEntry>,
    valid_run: &mut std::collections::BTreeMap<String, ValidationEntry>,
    valid_other: &mut std::collections::BTreeMap<String, ValidationEntry>,
) -> Result<(), String> {
    let text = fs::read_to_string(filename)
        .map_err(|error| format!("Opening {}: {error}", filename.display()))?;
    for raw_line in text.lines() {
        // The IMOD validation CSV has simple fields; retain quoted commas exactly by
        // scanning this source row rather than splitting command input elsewhere.
        let mut row = Vec::new();
        let mut field = String::new();
        let mut quoted = false;
        for character in raw_line.chars() {
            match character {
                '"' => quoted = !quoted,
                ',' if !quoted => {
                    row.push(field.trim_matches('"').to_owned());
                    field.clear();
                }
                _ => field.push(character),
            }
        }
        row.push(field.trim_matches('"').to_owned());
        if row.get(1).is_none_or(|value| value.is_empty()) {
            continue;
        }
        let parts = row[0].split('.').collect::<Vec<_>>();
        if parts.len() < 2 {
            continue;
        }
        let entry = ValidationEntry {
            spelling: row[0].clone(),
            template_ok: row.get(4).is_some_and(|value| value.trim() == "Y"),
            batch_ok: row.get(3).is_some_and(|value| value.trim() == "Y"),
            boolean: row
                .get(2)
                .is_some_and(|value| value.trim().eq_ignore_ascii_case("bool")),
        };
        match parts[0] {
            "comparam" if parts.len() >= 4 => {
                let com = parts[1].to_owned();
                base_com.insert(com.to_lowercase(), com.clone());
                base_com.insert(format!("{}a", com.to_lowercase()), com.clone());
                base_com.insert(format!("{}b", com.to_lowercase()), com.clone());
                valid_com.insert(
                    format!("{}.{}.{}", parts[1], parts[2], parts[3]).to_lowercase(),
                    entry,
                );
            }
            "runtime" if parts.len() >= 4 => {
                valid_run.insert(format!("{}.{}", parts[1], parts[3]).to_lowercase(), entry);
            }
            _ => {
                valid_other.insert(row[0].to_lowercase(), entry);
            }
        }
    }
    Ok(())
}

/// Matches `checkAllDirectives` (`IMOD/pysrc/batchruntomo:1797`) for the installed
/// validation table; unknown process autodoc lookup remains an external PIP boundary.
pub fn check_all_directives(
    dictionaries: &[DirectiveDictionary],
    batch_dictionary: usize,
    template_file: bool,
    base_com: &std::collections::BTreeMap<String, String>,
    valid_com: &std::collections::BTreeMap<String, ValidationEntry>,
    valid_run: &std::collections::BTreeMap<String, ValidationEntry>,
    valid_other: &std::collections::BTreeMap<String, ValidationEntry>,
) -> Vec<String> {
    let mut errors = Vec::new();
    for (dictionary_index, dictionary) in dictionaries.iter().take(batch_dictionary + 1).enumerate()
    {
        let template = template_file || dictionary_index < batch_dictionary;
        for (directive, (value, _)) in dictionary {
            let parts = directive.split('.').collect::<Vec<_>>();
            let entry = match parts.first().copied() {
                Some("comparam") if parts.len() >= 4 => {
                    let com = parts[1].to_lowercase();
                    if !base_com.contains_key(&com) {
                        errors.push(format!(
                            "Directive does not include a known com file: {directive}"
                        ));
                        continue;
                    }
                    valid_com.get(&format!("{}.{}.{}", parts[1], parts[2], parts[3]).to_lowercase())
                }
                Some("runtime") if parts.len() >= 4 => {
                    if !["any", "a", "b"].contains(&parts[2]) {
                        errors.push(format!("Directive does not include a/b/any: {directive}"));
                        continue;
                    }
                    valid_run.get(&format!("{}.{}", parts[1], parts[3]).to_lowercase())
                }
                Some("comparam") | Some("runtime") => {
                    errors.push(format!("Directive too short: {directive}"));
                    continue;
                }
                _ => valid_other.get(&directive.to_lowercase()),
            };
            let Some(entry) = entry else {
                errors.push(format!("Unknown directive: {directive}"));
                continue;
            };
            if entry.spelling != *directive {
                errors.push(format!("Directive has incorrect case: {directive}"));
            }
            if (!template && !entry.batch_ok) || (template && !entry.template_ok) {
                errors.push(format!(
                    "Directive is not intended for use in this file: {directive}"
                ));
            }
            if entry.boolean && value != "0" && value != "1" {
                errors.push(format!(
                    "Directive is a boolean and must be 0 or 1: {directive}"
                ));
            }
        }
    }
    errors
}

/// Matches `getAutoPatchfitParams` (`IMOD/pysrc/batchruntomo:3999`).
pub fn get_auto_patchfit_params(
    extra_targets: Option<&str>,
    final_patch_size: Option<&str>,
    default_extra_targets: &str,
    default_final_patch_size: &str,
) -> (String, String) {
    (
        final_patch_size
            .filter(|value| !value.is_empty())
            .unwrap_or(default_final_patch_size)
            .to_owned(),
        extra_targets
            .filter(|value| !value.is_empty())
            .unwrap_or(default_extra_targets)
            .to_owned(),
    )
}

/// Matches `CTFPlotAlignedStack` (`IMOD/pysrc/batchruntomo:3232`).
pub fn ctf_plot_aligned_stack(
    starting_step: f64,
    ending_step: f64,
    correct_ctf: bool,
    auto_fit_ctf: Option<&str>,
    axis_com_extension: &str,
    data_name: &str,
    options: &ProcessRunOptions,
) -> Result<i32, String> {
    const STEP: f64 = 9.;
    if !need_step(STEP, starting_step, ending_step) || !correct_ctf {
        return Ok(0);
    }
    let Some(auto_fit) = auto_fit_ctf.filter(|value| !value.is_empty()) else {
        return Ok(0);
    };
    let source = read_text_file_report_err(&format!("ctfplotter{axis_com_extension}"), None)?;
    let sed = vec![
        "/^ExpectedDefocus/a/SaveAndExit\t1/".to_owned(),
        format!("/^ExpectedDefocus/a/AutoFitRangeAndStep\t{auto_fit}/"),
        "/^AngleRange/d".to_owned(),
    ];
    let defocus = PathBuf::from(format!("{data_name}.defocus"));
    if defocus.exists() {
        let backup = PathBuf::from(format!("{data_name}.defocus~"));
        fs::rename(&defocus, &backup)
            .map_err(|error| format!("Backing up {}: {error}", defocus.display()))?;
    }
    modify_write_and_run_com(
        &format!("ctfplotter_auto{axis_com_extension}"),
        &sed,
        Some(source),
        "Finding defocus for CTF correction with Ctfplotter",
        false,
        options,
    )?;
    Ok(0)
}

/// Matches `runReduceFiltVol` (`IMOD/pysrc/batchruntomo:4660`).
pub fn run_reduce_filt_vol(
    do_reduce_filt: bool,
    binning: Option<f64>,
    com_extension: &str,
    data_name: &str,
    rec_file: &Path,
    directive_files: &[String],
    options: &ProcessRunOptions,
) -> Result<i32, String> {
    if !do_reduce_filt {
        return Ok(0);
    }
    if !rec_file.exists() {
        return Err("You must post-process with Trimvol in order to run Reducefiltvol".to_owned());
    }
    let binning = binning.unwrap_or(1.);
    let comfile = format!("reducefiltvol{com_extension}");
    let mut lines = vec![
        format!("RootNameOfDataFiles {data_name}"),
        format!("InputFile {}", rec_file.display()),
        format!("BinningOfImages {binning}"),
        "OneParameterChange comparam.reducefiltvol.reducefiltvol.SetupChunksIfMemoryError=1"
            .to_owned(),
    ];
    lines.extend(
        directive_files
            .iter()
            .map(|file| format!("ChangeParametersFile {file}")),
    );
    make_and_run_one_com(
        &mut lines,
        &comfile,
        "Reducing and/or filtering final tomogram",
        false,
        false,
        0,
        "",
        options,
    )?;
    let log = read_text_file_report_err("reducefiltvol.log", None)?;
    let mut it_ran = true;
    let mut next_com = format!("rfvfilter{com_extension}");
    for line in log {
        if line.contains("[MTF1]") {
            it_ran = false;
        }
        if line.contains("processchunks") {
            if let Some(name) = line.split_whitespace().last() {
                next_com = name.to_owned();
            }
        }
    }
    if it_ran {
        Ok(0)
    } else {
        run_one_process(
            &next_com,
            false,
            false,
            "Filtering volume in chunks to overcome memory limit",
            options,
        )
    }
}

/// Matches `runNAD` (`IMOD/pysrc/batchruntomo:4615`).
pub fn run_nad(
    iterations: Option<i32>,
    k_value: Option<f64>,
    memory_mb: Option<i32>,
    com_extension: &str,
    rec_file: &Path,
    options: &ProcessRunOptions,
) -> Result<i32, String> {
    match (iterations, k_value) {
        (None, None) => return Ok(0),
        (Some(_), Some(_)) => (),
        _ => {
            return Err(
                "Both \"iterations\" and \"Kvalue\" directives need to be present to run NAD"
                    .to_owned(),
            );
        }
    }
    let iterations = iterations.unwrap();
    let k_value = k_value.unwrap();
    if !rec_file.exists() {
        return Err("You must post-process with Trimvol in order to run NAD".to_owned());
    }
    let memory = memory_mb.unwrap_or(512);
    let chunk_size = (memory / 36).max(5);
    let comfile = format!("autoNAD{com_extension}");
    let nad_file = rec_file.with_extension("nad");
    write_text_file_report_err(
        &comfile,
        &[format!(
            "$nad_eed_3d -n {iterations} -k {k_value} INPUTFILE OUTPUTFILE"
        )],
    )?;
    let padding = iterations.max(8);
    let command = format!(
        "chunksetup -m {chunk_size} -p {padding} -no {comfile} {} {}",
        rec_file.display(),
        nad_file.display()
    );
    super::imodpy::run_cmd(&command, None, None, None, &[])
        .map_err(|error| error.arguments.join("\n"))?;
    run_one_process(
        &comfile,
        false,
        false,
        "Filtering with anisotropic diffusion",
        options,
    )
}

/// Matches `alignAndCombineAxes` (`IMOD/pysrc/batchruntomo:4392`).
pub fn align_and_combine_axes(
    starting_step: f64,
    ending_step: f64,
    com_extension: &str,
    final_patch_size: &str,
    extra_targets: &str,
    parallel_cpu: i32,
    options: &ProcessRunOptions,
) -> Result<(), String> {
    if need_step(17., starting_step, ending_step) {
        run_one_process(
            &format!("matchvol1{com_extension}"),
            true,
            false,
            "Making initial matching volume",
            options,
        )?;
    }
    if need_step(18., starting_step, ending_step) {
        let mut lines = vec![
            "$autopatchfit -StandardInput".to_owned(),
            format!("FinalPatchTypeOrXYZ {final_patch_size}"),
        ];
        if !extra_targets.is_empty() {
            lines.push(format!("ExtraResidualTargets {extra_targets}"));
        }
        let comfile = format!("autopatchfit{com_extension}");
        write_text_file_report_err(&comfile, &lines)?;
        run_one_process(
            &comfile,
            true,
            false,
            "Doing patch correlation and fitting to local patches",
            options,
        )?;
    }
    if need_step(19., starting_step, ending_step) {
        if parallel_cpu > 1 {
            super::imodpy::run_cmd("splitcombine", None, None, None, &[])
                .map_err(|error| error.arguments.join("\n"))?;
        }
        run_one_process(
            &format!("volcombine{com_extension}"),
            parallel_cpu < 2,
            false,
            "Combining the two volumes",
            options,
        )?;
    }
    Ok(())
}

/// Matches `postProcessTiltalign` (`IMOD/pysrc/batchruntomo:2700`).
pub fn post_process_tiltalign(
    align_lines: &[String],
    axis_upper: &str,
    axis_edf: &str,
    edf_lines: &mut Vec<String>,
    edf_changed: &mut bool,
    set_name: &str,
    bypass_etomo: bool,
) -> Result<(), String> {
    let mut z_factors = 0i32;
    let mut local = false;
    let mut z_shift = 0.;
    let mut angle = 0.;
    for line in align_lines {
        let words = line.split_whitespace().collect::<Vec<_>>();
        if words.len() < 2 {
            continue;
        }
        match words[0] {
            "XStretchOption" => z_factors = words[1].parse().unwrap_or(0),
            "LocalAlignments" => local = words[1] == "1",
            "AxisZShift" => z_shift = words[1].parse().unwrap_or(0.),
            "AngleOffset" => angle = words[1].parse().unwrap_or(0.),
            _ => (),
        }
    }
    let mut sed = edf_del_and_add(
        &format!("MadeZFactors{axis_upper}"),
        &bool_string_for_edf(z_factors > 0),
        '/',
    );
    sed.extend(edf_del_and_add(
        &format!("UsedLocalAlignments{axis_upper}"),
        &bool_string_for_edf(local),
        '/',
    ));
    sed.extend(edf_del_and_add(
        &format!("{axis_edf}.align.AxisZShift"),
        &format!("{z_shift:.2}"),
        '/',
    ));
    sed.extend(edf_del_and_add(
        &format!("{axis_edf}.align.AngleOffset"),
        &format!("{angle:.3}"),
        '/',
    ));
    modify_edf_lines(edf_lines, edf_changed, set_name, bypass_etomo, &sed)
}

/// Matches `fidlessFileOperations` (`IMOD/pysrc/batchruntomo:2347`).
pub fn fidless_file_operations(
    data_name: &str,
    axis_letter: &str,
    axis_rotation: f64,
) -> Result<(), String> {
    let radians = axis_rotation.to_radians();
    let cosine = radians.cos();
    let sine = radians.sin();
    let rotation = format!("rotation{axis_letter}.xf");
    write_text_file_report_err(
        &rotation,
        &[format!(
            "{cosine:.6} {sine:.6} {:.6} {cosine:.6} 0. 0.",
            -sine
        )],
    )?;
    let prexf = read_text_file_report_err(&format!("{data_name}.prexf"), None)?;
    write_text_file_report_err(
        &format!("{data_name}.xtilt"),
        &vec!["0.".to_owned(); prexf.len()],
    )?;
    super::imodpy::run_cmd(
        &format!("xftoxg -nfit 0 {data_name}.prexf"),
        None,
        None,
        None,
        &[],
    )
    .map_err(|error| error.arguments.join("\n"))?;
    super::imodpy::run_cmd(
        &format!("xfproduct {data_name}.prexg {rotation} {data_name}_nonfid.xf"),
        None,
        None,
        None,
        &[],
    )
    .map_err(|error| error.arguments.join("\n"))?;
    fs::copy(format!("{data_name}_nonfid.xf"), format!("{data_name}.xf"))
        .map_err(|error| format!("Error copying xf: {error}"))?;
    fs::copy(format!("{data_name}.rawtlt"), format!("{data_name}.tlt"))
        .map_err(|error| format!("Error copying tlt: {error}"))?;
    Ok(())
}

/// Matches `runPatchTracking` (`IMOD/pysrc/batchruntomo:2384`).
pub fn run_patch_tracking(
    contour_pieces: Option<i32>,
    patch_size: Option<&str>,
    patch_count: Option<&str>,
    patch_overlap: Option<&str>,
    boundary_model: Option<&str>,
    piece_length: Option<i32>,
    axis_com_extension: &str,
    coarse_binning: i32,
    data_name: &str,
    directive_files: &[String],
    use_gpu: bool,
    options: &ProcessRunOptions,
) -> Result<(), String> {
    if patch_size.is_none_or(str::is_empty) {
        return Err("Size of patches must be specified to use patch tracking".to_owned());
    }
    if patch_count.is_some_and(|value| !value.is_empty())
        && patch_overlap.is_some_and(|value| !value.is_empty())
    {
        return Err("You cannot enter both the number of patches to track and the fractional overlap of patches".to_owned());
    }
    let mut lines = vec![
        format!("InputFile xcorr{axis_com_extension}"),
        format!("BinningOfImages {coarse_binning}"),
        format!("RootNameOfDataFiles {data_name}"),
    ];
    lines.extend(
        directive_files
            .iter()
            .map(|file| format!("ChangeParametersFile {file}")),
    );
    if use_gpu {
        lines.push("UseGPU  0".to_owned());
    }
    if let Some(model) = boundary_model.filter(|value| !value.is_empty()) {
        lines.push(format!(
            "OneParameterChange comparam.xcorr_pt.tiltxcorr.BoundaryModel={model}"
        ));
    }
    if let Some(number) = contour_pieces.filter(|value| *value > 1) {
        if piece_length.is_some_and(|value| value > 0) {
            return Err(
                "Both LengthOfPieces and PatchTracking.contourPieces were entered".to_owned(),
            );
        }
        lines.push(format!(
            "OneParameterChange comparam.xcorr_pt.imodchopconts.NumberOfPieces={number}"
        ));
    }
    make_and_run_one_com(
        &mut lines,
        &format!("xcorr_pt{axis_com_extension}"),
        "Tracking patches to make alignment model",
        false,
        false,
        0,
        "",
        options,
    )
    .map(|_| ())
}

/// Matches `OKtoAdjustPatchTrack` (`IMOD/pysrc/batchruntomo:2442`).
pub fn ok_to_adjust_patch_track(
    total_tilt_adjustment: f64,
    max_tilt_adjustment: Option<f64>,
    max_adjusted_angle: Option<f64>,
    raw_tilt_file: &Path,
) -> Result<bool, String> {
    let max_tilt_adjustment = max_tilt_adjustment
        .filter(|value| *value != 0.)
        .unwrap_or(20.);
    let max_adjusted_angle = max_adjusted_angle
        .filter(|value| *value != 0.)
        .unwrap_or(78.);
    if total_tilt_adjustment.abs() > max_tilt_adjustment {
        return Ok(false);
    }
    let text = match fs::read_to_string(raw_tilt_file) {
        Ok(text) => text,
        Err(_) => return Ok(true),
    };
    let mut highest: f64 = 0.;
    for line in text.lines() {
        let Some(value) = line
            .split_whitespace()
            .next()
            .and_then(|value| value.parse::<f64>().ok())
        else {
            return Ok(true);
        };
        highest = highest.max((value + total_tilt_adjustment).abs());
    }
    Ok(highest <= max_adjusted_angle)
}

/// Matches `varyPatchTrack` (`IMOD/pysrc/batchruntomo:2480`).
pub fn vary_patch_track(
    use_gpu: bool,
    axis_com_extension: &str,
    axis_letter: &str,
    directive_files: &[String],
    options: &ProcessRunOptions,
) -> Result<(PathBuf, Option<PathBuf>), String> {
    let mut lines = Vec::new();
    if use_gpu {
        lines.push("OneParameterChange comparam.varypatchtrack.varypatchtrack.UseGPU=0".to_owned());
    }
    lines.extend(
        directive_files
            .iter()
            .map(|file| format!("ChangeParametersFile {file}")),
    );
    make_and_run_one_com(
        &mut lines,
        &format!("varypatchtrack{axis_com_extension}"),
        "Optimizing patch tracking parameters",
        use_gpu,
        false,
        0,
        "",
        options,
    )?;
    let log = read_text_file_report_err(&format!("varypatchtrack{axis_letter}.log"), None)?;
    let mut xcorr = None;
    let mut align = None;
    for line in log {
        let parts = line.split_whitespace().collect::<Vec<_>>();
        if line.contains("[VPT1]") {
            xcorr = parts.iter().rev().nth(1).map(|value| PathBuf::from(value));
        }
        if line.contains("[VPT2]") {
            align = parts.iter().rev().nth(1).map(|value| PathBuf::from(value));
        }
    }
    let xcorr = xcorr.ok_or_else(|| {
        "No new Tiltxcorr command file was written after running Varypatchtrack".to_owned()
    })?;
    Ok((xcorr, align))
}

/// Matches `fixModelHeaderForImage` (`IMOD/pysrc/batchruntomo:2115`).
pub fn fix_model_header_for_image(model: &Path, image: &Path) -> Result<i32, String> {
    if !image.exists() || !model.exists() {
        return Ok(0);
    }
    let temporary = PathBuf::from(format!("{}.transtemp", model.display()));
    let command = format!(
        "imodtrans -I \"{}\" \"{}\" \"{}\"",
        image.display(),
        model.display(),
        temporary.display()
    );
    if super::imodpy::run_cmd(&command, None, None, None, &[]).is_err() {
        return Ok(0);
    }
    let backup = PathBuf::from(format!("{}~", model.display()));
    if model.exists() {
        fs::rename(model, &backup)
            .map_err(|error| format!("Backing up {}: {error}", model.display()))?;
    }
    fs::rename(&temporary, model).map_err(|error| {
        format!(
            "Error renaming fixed model {} to {}: {error}",
            temporary.display(),
            model.display()
        )
    })?;
    Ok(0)
}

/// Matches `fixAllSuppliedModelHeaders` (`IMOD/pysrc/batchruntomo:2137`).
pub fn fix_all_supplied_model_headers(
    models_a: &[PathBuf],
    image_a: &Path,
    models_b: &[PathBuf],
    image_b: Option<&Path>,
) -> Result<i32, String> {
    for model in models_a {
        if fix_model_header_for_image(model, image_a)? != 0 {
            return Ok(1);
        }
    }
    if let Some(image_b) = image_b {
        for model in models_b {
            if fix_model_header_for_image(model, image_b)? != 0 {
                return Ok(1);
            }
        }
    }
    Ok(0)
}

/// Matches `prnLog` (`IMOD/pysrc/batchruntomo:32`).
pub fn prn_log(message: &str, log_file: Option<&mut std::fs::File>) -> Result<(), String> {
    println!("{message}");
    if let Some(log_file) = log_file {
        use std::io::Write;
        writeln!(log_file, "{message}").map_err(|error| error.to_string())?;
    }
    Ok(())
}

/// Matches `closeLogFileWriteEdf` (`IMOD/pysrc/batchruntomo:39`).
pub fn close_log_file_write_edf(
    log_file: &mut Option<std::fs::File>,
    edf_lines: &mut Vec<String>,
    edf_changed: &mut bool,
    set_name: &str,
) -> Result<(), String> {
    if let Some(mut file) = log_file.take() {
        use std::io::Write;
        writeln!(file, "Batchruntomo finished with data set").map_err(|error| error.to_string())?;
    }
    if *edf_changed {
        let path = PathBuf::from(format!("{set_name}.edf"));
        if path.exists() {
            fs::rename(&path, PathBuf::from(format!("{}~", path.display())))
                .map_err(|error| error.to_string())?;
        }
        write_text_file_report_err(&path.to_string_lossy(), edf_lines)?;
        *edf_changed = false;
    }
    edf_lines.clear();
    Ok(())
}

/// Matches `warning` (`IMOD/pysrc/batchruntomo:62`).
pub fn warning(messages: &[String]) -> Vec<String> {
    let mut output = messages.to_vec();
    if let Some(first) = output.first_mut() {
        *first = format!("WARNING: {first}");
    }
    output.push(" ".to_owned());
    output
}

/// Matches `sendEmail` (`IMOD/pysrc/batchruntomo:75`), retaining SMTP as an external
/// boundary; Python's SMTP interaction has no native replacement here.
pub fn send_email(
    address: Option<&str>,
    subject: &str,
    message: &str,
    smtp_server: &str,
) -> Result<(), String> {
    if address.is_none_or(str::is_empty) {
        return Ok(());
    }
    Err(format!(
        "SMTP delivery remains external: server={smtp_server}, to={}, subject={subject}, message={message}",
        address.unwrap_or("")
    ))
}

/// Matches `abortSet` (`IMOD/pysrc/batchruntomo:95`).
pub fn abort_set(
    error: &str,
    suppress_abort: bool,
    renaming_only: bool,
    exit_on_error: bool,
) -> Result<(), String> {
    if suppress_abort {
        Ok(())
    } else if renaming_only || exit_on_error {
        Err(error.to_owned())
    } else {
        Ok(())
    }
}

/// Matches `reportImodError` (`IMOD/pysrc/batchruntomo:121`).
pub fn report_imod_error(
    error_lines: &[String],
    abort_text: Option<&str>,
) -> (Vec<String>, Option<String>) {
    let mut output = error_lines.to_vec();
    if let Some(last) = output.last_mut() {
        *last = format!("ERROR: {last}");
    }
    (output, abort_text.map(str::to_owned))
}

/// Direct state returned by `scanSetupDirectives`.
#[derive(Clone, Debug, PartialEq)]
pub struct SetupDirectives {
    pub dataset_dir: String,
    pub set_name: String,
    pub scan_header: bool,
    pub montage: bool,
    pub dual_axis: bool,
    pub pixel_size: f64,
    pub fiducial_size_nm: Option<f64>,
    pub defocus: f64,
}

/// Matches `scanSetupDirectives` (`IMOD/pysrc/batchruntomo:1913`).
pub fn scan_setup_directives(
    dictionaries: &[DirectiveDictionary],
    batch_dictionary: usize,
    validation: i32,
) -> Result<SetupDirectives, String> {
    let batch = dictionaries
        .get(batch_dictionary)
        .ok_or_else(|| "Missing batch directive dictionary".to_owned())?;
    let mut setup = SetupDirectives {
        dataset_dir: batch
            .get("setupset.datasetDirectory")
            .map(|value| value.0.clone())
            .unwrap_or_default(),
        set_name: batch
            .get("setupset.copyarg.name")
            .map(|value| value.0.clone())
            .unwrap_or_default(),
        scan_header: false,
        montage: false,
        dual_axis: false,
        pixel_size: 0.,
        fiducial_size_nm: None,
        defocus: -1_000_000.,
    };
    for dictionary in dictionaries.iter().take(batch_dictionary + 1) {
        for (key, (value, _)) in dictionary {
            match key.as_str() {
                "setupset.scanHeader" => setup.scan_header = value == "1",
                "setupset.copyarg.montage" => setup.montage = value != "0",
                "setupset.copyarg.dual" => setup.dual_axis = value != "0",
                "setupset.copyarg.pixel" if !value.is_empty() => {
                    setup.pixel_size = value
                        .parse()
                        .map_err(|_| format!("Error converting {key} to float"))?
                }
                "setupset.copyarg.gold" if !value.is_empty() => {
                    setup.fiducial_size_nm = Some(
                        value
                            .parse()
                            .map_err(|_| format!("Error converting {key} to float"))?,
                    )
                }
                "setupset.copyarg.defocus" if !value.is_empty() => {
                    setup.defocus = value
                        .parse()
                        .map_err(|_| format!("Error converting {key} to float"))?
                }
                _ => (),
            }
        }
    }
    if validation <= 0 && setup.pixel_size == 0. && !setup.scan_header {
        return Err(
            "Pixel size missing from directives, and header is not being scanned".to_owned(),
        );
    }
    if validation <= 0 && (setup.set_name.is_empty() || setup.dataset_dir.is_empty()) {
        return Err("Set name or dataset directory missing from directives".to_owned());
    }
    Ok(setup)
}

/// Matches `checkDefaultsInBatchFile` (`IMOD/pysrc/batchruntomo:1989`).
pub fn check_defaults_in_batch_file(
    dictionaries: &[DirectiveDictionary],
    batch_dictionary: usize,
) -> Vec<String> {
    let defaults = [
        ("setupset.scanHeader", "1"),
        ("runtime.Preprocessing.any.archiveOriginal", "1"),
        ("comparam.eraser.ccderaser.LineObjects", "2"),
        ("comparam.eraser.ccderaser.BoundaryObjects", "3"),
        ("comparam.eraser.ccderaser.AllSectionObjects", "1-3"),
        ("comparam.track.beadtrack.RoundsOfTracking", "4"),
        ("runtime.BeadTracking.any.numberOfRuns", "2"),
        ("comparam.align.tiltalign.RobustFitting", "1"),
        ("setupset.copyarg.gold", "0"),
    ];
    let batch = match dictionaries.get(batch_dictionary) {
        Some(value) => value,
        None => return Vec::new(),
    };
    let mut duplicates = 0;
    let mut masked = Vec::new();
    for (key, value) in defaults {
        if batch.get(key).is_some_and(|entry| entry.0 == value) {
            duplicates += 1;
            if dictionaries
                .iter()
                .take(batch_dictionary)
                .skip(1)
                .any(|dictionary| dictionary.get(key).is_some_and(|entry| entry.0 != value))
            {
                masked.push(format!("    {key} = {value}"));
            }
        }
    }
    if duplicates >= 4 && !masked.is_empty() {
        let mut output =
            vec!["Incorrect directives in batch file may be masking template values.".to_owned()];
        output.extend(masked);
        output
    } else {
        Vec::new()
    }
}

/// Explicit directive values consumed by `getAxisInitialParameters`.
#[derive(Clone, Debug)]
pub struct AxisInitialInput {
    pub fiducialless: bool,
    pub tracking_method: i32,
    pub erase_gold: i32,
    pub filter_in_2d: bool,
    pub do_sirt: bool,
    pub fake_sirt: bool,
    pub both_reconstructions: bool,
    pub correct_ctf: bool,
    pub auto_fit_ctf: Option<String>,
    pub defocus: f64,
    pub scan_defocus_range: Option<String>,
    pub ctf3d_slab_thickness: i32,
    pub raw_for_3d_ctf: bool,
    pub super_sample: i32,
    pub expand_input_lines: bool,
    pub coarse_binning: i32,
    pub raw_size: (i32, i32, i32),
}

/// Matches `getAxisInitialParameters` (`IMOD/pysrc/batchruntomo:2022`).
pub fn get_axis_initial_parameters(input: AxisInitialInput) -> Result<AxisInitialInput, String> {
    if input.erase_gold == 1 && (input.fiducialless || input.tracking_method == 1) {
        return Err(
            "Cannot erase gold with fiducials after fiducialless processing or patch tracking"
                .to_owned(),
        );
    }
    if input.correct_ctf && input.defocus < -999_999. && input.auto_fit_ctf.is_none() {
        return Err("Defocus must be entered to correct CTF without using Ctfplotter".to_owned());
    }
    if input.correct_ctf
        && input.defocus < -999_999.
        && input.auto_fit_ctf.is_some()
        && input.scan_defocus_range.is_none()
    {
        return Err(
            "Either defocus or a range to scan must be entered to use Ctfplotter".to_owned(),
        );
    }
    if input.ctf3d_slab_thickness > 0 && !input.correct_ctf {
        return Err("Directive for correcting CTF must also be present to do 3-D CTF-corrected reconstruction".to_owned());
    }
    if input.ctf3d_slab_thickness > 0
        && (input.do_sirt || input.fake_sirt)
        && !input.both_reconstructions
    {
        return Err(
            "You cannot do just a SIRT or SIRT-like reconstruction with 3-D CTF correction"
                .to_owned(),
        );
    }
    if input.super_sample > 1 && input.expand_input_lines && input.raw_for_3d_ctf {
        return Err(
            "You cannot expand input lines when doing 3D CTF correction with unaligned images"
                .to_owned(),
        );
    }
    if input.coarse_binning <= 0 {
        return Err("Coarse binning must be positive".to_owned());
    }
    if input.raw_size.0 <= 0 || input.raw_size.1 <= 0 || input.raw_size.2 <= 0 {
        return Err("Error getting size of raw stack".to_owned());
    }
    Ok(input)
}

/// Matches `analyzeSDsAdjustExcludes` (`IMOD/pysrc/batchruntomo:2174`) after source
/// clip statistics and optional histogram process output are available.
pub fn analyze_sds_adjust_excludes(
    clip_lines: &[String],
    criterion: Option<f64>,
    dark_ratio: Option<f64>,
    dark_fraction: Option<f64>,
    dark_excludes: &[i32],
    existing_excludes: &[i32],
) -> Result<Vec<i32>, String> {
    const NUM_AVERAGE: usize = 5;
    const MAX_EXCLUDE: usize = 3;
    let Some(criterion) = criterion.filter(|value| *value > 0.) else {
        return Ok(Vec::new());
    };
    let mut sds = Vec::new();
    for line in clip_lines {
        if line.contains("mean") || line.contains("-----") {
            continue;
        }
        if line.contains("all") {
            break;
        }
        let value = line
            .split_whitespace()
            .last()
            .ok_or_else(|| format!("Error converting standard deviation in: {line}"))?
            .parse::<f64>()
            .map_err(|_| format!("Error converting standard deviation in: {line}"))?;
        sds.push(value);
    }
    if sds.len() < NUM_AVERAGE + 2 + 2 * MAX_EXCLUDE {
        return Ok(Vec::new());
    }
    let dark_ratio = dark_ratio.unwrap_or(0.17);
    let dark_fraction = dark_fraction.unwrap_or(0.33);
    let mut output = Vec::new();
    for (base_index, base_view, direction) in
        [(0isize, 1i32, 1isize), (-1isize, sds.len() as i32, -1isize)]
    {
        let mut low = 0usize;
        for indent in 2..=MAX_EXCLUDE + 1 {
            let mean = (0..NUM_AVERAGE)
                .map(|index| {
                    sds[(base_index + direction * (index + indent) as isize)
                        .rem_euclid(sds.len() as isize) as usize]
                })
                .sum::<f64>()
                / NUM_AVERAGE as f64;
            low = (0..indent)
                .take_while(|index| {
                    sds[(base_index + direction * *index as isize).rem_euclid(sds.len() as isize)
                        as usize]
                        < criterion * mean
                })
                .count();
            if low > 0 && low < indent {
                output.extend((0..low).map(|index| base_view + direction as i32 * index as i32));
                break;
            }
        }
        if dark_ratio > 0. {
            for index in low..2 * MAX_EXCLUDE {
                let view = base_view + direction as i32 * index as i32;
                if dark_excludes.contains(&(view - 1)) && dark_fraction > 0. {
                    output.push(view);
                } else {
                    break;
                }
            }
        }
    }
    output.sort_unstable();
    output.dedup();
    Ok(output
        .into_iter()
        .filter(|view| !existing_excludes.contains(view))
        .collect())
}

/// Matches `getOrXformBoundaryModel` (`IMOD/pysrc/batchruntomo:1425`).
pub fn get_or_xform_boundary_model(
    raw_boundary: Option<&str>,
    prealigned_boundary: Option<&str>,
    data_name: &str,
    default_suffix: &str,
    prealigned_image: &str,
    prealigned_size: (i32, i32),
    raw_size: (i32, i32),
    coarse_binning: i32,
    stack_extension: &str,
) -> Result<Option<PathBuf>, String> {
    let direct = prealigned_boundary
        .filter(|value| !value.is_empty())
        .map(PathBuf::from);
    if direct.is_some() {
        return Ok(direct);
    }
    let Some(raw) = raw_boundary.filter(|value| !value.is_empty()) else {
        return Ok(None);
    };
    let output = PathBuf::from(format!("{data_name}{default_suffix}"));
    transform_raw_boundary_model(
        raw,
        &output.to_string_lossy(),
        data_name,
        stack_extension,
        prealigned_image,
        prealigned_size.0,
        prealigned_size.1,
        raw_size.0,
        raw_size.1,
        coarse_binning,
    )?;
    Ok(Some(output))
}

/// Matches `eraseGoldInAlignedStack` (`IMOD/pysrc/batchruntomo:3424`).
pub fn erase_gold_in_aligned_stack(
    erase_gold: i32,
    extend_model: bool,
    extra_diameter: f64,
    fiducial_size_pixels: f64,
    aligned_binning: f64,
    expansion: f64,
    data_name: &str,
    axis_com_extension: &str,
    axis_letter: &str,
    directive_files: &[String],
    skip_modifications: bool,
    options: &ProcessRunOptions,
) -> Result<(), String> {
    if erase_gold > 1 {
        let mut lines = vec![
            format!("InputFile tilt_3dfind{axis_com_extension}"),
            format!("RootNameOfDataFiles {data_name}"),
        ];
        make_and_run_one_com(
            &mut lines,
            &format!("tilt_3dfind_reproject{axis_com_extension}"),
            "Reprojecting model of beads in tomogram onto aligned stack",
            false,
            false,
            0,
            "",
            options,
        )?;
    } else {
        let suffix = if Path::new(&format!("{data_name}_nogaps.fid")).exists() {
            "_nogaps"
        } else {
            ""
        };
        super::imodpy::run_cmd(
            &format!("xfmodel -xf {data_name}.tltxf {data_name}{suffix}.fid {data_name}_erase.fid"),
            None,
            None,
            None,
            &[],
        )
        .map_err(|error| error.arguments.join("\n"))?;
    }
    if extend_model {
        let mut lines = vec![format!("RootNameOfDataFiles {data_name}")];
        if erase_gold > 1 {
            lines.push("ReplaceAboveAngle 0.".to_owned());
        }
        make_and_run_one_com(
            &mut lines,
            &format!("extenderasemod{axis_com_extension}"),
            "Extending model on aligned stack",
            false,
            false,
            0,
            "",
            options,
        )?;
        let _ = use_file_as_replacement(
            &PathBuf::from(format!("{data_name}_extended.fid")),
            &PathBuf::from(format!("{data_name}_erase.fid")),
            false,
            true,
        );
    }
    let mut lines = vec![
        format!("RootNameOfDataFiles {data_name}"),
        format!(
            "BeadSize {}",
            expansion * fiducial_size_pixels / aligned_binning + extra_diameter
        ),
    ];
    lines.extend(
        directive_files
            .iter()
            .map(|file| format!("ChangeParametersFile {file}")),
    );
    make_and_run_one_com(
        &mut lines,
        &format!("golderaser{axis_com_extension}"),
        "Erasing beads from aligned stack",
        false,
        skip_modifications,
        0,
        "",
        options,
    )?;
    if !skip_modifications {
        use_file_as_replacement(
            &PathBuf::from(format!("{data_name}_erase.ali")),
            &PathBuf::from(format!("{data_name}.ali")),
            false,
            true,
        )?;
    }
    let _ = axis_letter;
    Ok(())
}

/// Matches `CTFCorrectAlignedStack` (`IMOD/pysrc/batchruntomo:3258`).
pub fn ctf_correct_aligned_stack(
    data_name: &str,
    axis_com_extension: &str,
    z_size: i32,
    defocus: f64,
    aligned_binning: f64,
    pixel_size: f64,
    use_gpu: bool,
    x_axis_tilt: Option<f64>,
    skip_aligned_stack_modifications: bool,
    number_processes: i32,
    options: &ProcessRunOptions,
) -> Result<(), String> {
    let defocus_file = PathBuf::from(format!("{data_name}.defocus"));
    let simple = format!("{data_name}_simple.defocus");
    let mut sed = if defocus_file.exists() {
        vec![super::pysed::sed_modify(
            "DefocusFile",
            &defocus_file.to_string_lossy(),
            '/',
        )]
    } else {
        write_text_file_report_err(
            &simple,
            &[format!("{} {} 0. 0. {defocus}", z_size / 2, z_size / 2)],
        )?;
        vec![super::pysed::sed_modify("DefocusFile", &simple, '/')]
    };
    sed.push(super::pysed::sed_modify(
        "PixelSize",
        &(aligned_binning * pixel_size).to_string(),
        '/',
    ));
    if use_gpu {
        sed.extend(super::pysed::sed_del_and_add(
            "UseGPU",
            "0",
            "DefocusFile",
            '/',
        ));
    }
    if let Some(tilt) = x_axis_tilt.filter(|value| *value != 0.) {
        sed.extend(super::pysed::sed_del_and_add(
            "XAxisTilt",
            &tilt.to_string(),
            "DefocusFile",
            '/',
        ));
    }
    let comfile = format!("ctfcorrection{axis_com_extension}");
    let lines = read_text_file_report_err(&comfile, None)?;
    modify_write_and_run_com(
        &comfile,
        &sed,
        Some(lines),
        "Correcting for CTF with Ctfphaseflip",
        skip_aligned_stack_modifications,
        options,
    )?;
    if skip_aligned_stack_modifications {
        return Ok(());
    }
    if number_processes > 1 {
        let max_slices = (z_size + 2 * number_processes - 1) / (2 * number_processes);
        super::imodpy::run_cmd(
            &format!("splitcorrection -m {max_slices} {comfile}"),
            None,
            None,
            None,
            &[],
        )
        .map_err(|error| error.arguments.join("\n"))?;
        run_one_process(
            &comfile,
            false,
            use_gpu,
            "Correcting for CTF with Ctfphaseflip",
            options,
        )?;
    }
    use_file_as_replacement(
        &PathBuf::from(format!("{data_name}_ctfcorr.ali")),
        &PathBuf::from(format!("{data_name}.ali")),
        false,
        true,
    )?;
    Ok(())
}

/// State-owning source-operation interface for `generateTomogram`.
pub trait TomogramWorkflow {
    fn do_sirt(&self) -> bool;
    fn do_fake_sirt(&self) -> bool;
    fn do_both(&self) -> bool;
    fn ctf3d(&self) -> bool;
    fn processors(&mut self) -> Result<i32, String>;
    fn back_project(&mut self, processors: i32) -> Result<(), String>;
    fn prepare_fake_sirt_regular(&mut self) -> Result<(), String>;
    fn ctf3d_reconstruct(&mut self, processors: i32) -> Result<(), String>;
    fn sirt(&mut self, processors: i32) -> Result<(), String>;
    fn final_size_to_edf(&mut self) -> Result<(), String>;
    fn release_gpu(&mut self);
}

/// Matches `generateTomogram` (`IMOD/pysrc/batchruntomo:3692`).
pub fn generate_tomogram<Workflow: TomogramWorkflow>(
    workflow: &mut Workflow,
) -> Result<(), String> {
    if workflow.do_sirt() && workflow.do_fake_sirt() {
        return Err("You cannot do both SIRT and a SIRT-like filter".to_owned());
    }
    let processors = workflow.processors()?;
    if processors < 1 {
        return Err("No processing units are available".to_owned());
    }
    let result = (|| {
        if workflow.do_both() || (!workflow.do_sirt() && !workflow.ctf3d()) {
            workflow.back_project(processors)?;
        }
        if workflow.do_fake_sirt() && workflow.do_both() {
            workflow.prepare_fake_sirt_regular()?;
            if workflow.ctf3d() {
                workflow.ctf3d_reconstruct(processors)?;
            } else {
                workflow.back_project(processors)?;
            }
        }
        if workflow.do_sirt() {
            workflow.sirt(processors)?;
        }
        if !(workflow.do_fake_sirt() && workflow.do_both()) && workflow.ctf3d() {
            workflow.ctf3d_reconstruct(processors)?;
        }
        workflow.final_size_to_edf()
    })();
    workflow.release_gpu();
    result
}

/// Matches `makeAlignedStack` (`IMOD/pysrc/batchruntomo:3126`) for the non-montage
/// command branch; montage frame selection remains in `montage_frame_values`.
pub fn make_aligned_stack(
    position_binning: i32,
    configured_binning: Option<i32>,
    configured_size: Option<&str>,
    linear_interpolation: bool,
    raw_size: (i32, i32),
    expansion: f64,
    transpose: bool,
    axis_com_extension: &str,
    data_name: &str,
    correct_ctf: bool,
    need_step_eight: bool,
    need_ctf_correction: bool,
    skip_modifications: bool,
    raw_for_3d_ctf: bool,
    options: &ProcessRunOptions,
) -> Result<(i32, i32, i32), String> {
    let binning = if position_binning > 0 {
        position_binning
    } else {
        configured_binning.unwrap_or(1)
    };
    if binning <= 0 {
        return Err("Aligned stack binning must be positive".to_owned());
    }
    let (mut size_x, mut size_y) = match configured_size {
        Some(value) if !value.is_empty() => {
            let values = value
                .replace(',', " ")
                .split_whitespace()
                .map(str::parse::<i32>)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| "Error converting aligned stack output size".to_owned())?;
            if values.len() < 2 {
                return Err("Error converting aligned stack output size".to_owned());
            }
            (values[0], values[1])
        }
        _ => raw_size,
    };
    if transpose {
        std::mem::swap(&mut size_x, &mut size_y);
    }
    if position_binning < 0 && skip_modifications && raw_for_3d_ctf {
        return Ok((binning, size_x, size_y));
    }
    let comfile = format!("newst{axis_com_extension}");
    let lines = read_text_file_report_err(&comfile, None)?;
    let mut sed = vec![super::pysed::sed_modify(
        "SizeToOutputInXandY",
        &format!(
            "{},{}",
            (expansion * (size_x / binning) as f64) as i32,
            (expansion * (size_y / binning) as f64) as i32
        ),
        '/',
    )];
    sed.extend(super::pysed::sed_del_and_add(
        "LinearInterpolation",
        if linear_interpolation { "1" } else { "0" },
        "TransformFile",
        '/',
    ));
    sed.extend(super::pysed::sed_del_and_add(
        "BinByFactor",
        &binning.to_string(),
        "TransformFile",
        '/',
    ));
    let need_rebuild = !need_step_eight
        && need_ctf_correction
        && correct_ctf
        && Path::new(&format!("{data_name}.ali")).exists();
    if position_binning > 0 || need_step_eight || need_rebuild {
        modify_write_and_run_com(
            &comfile,
            &sed,
            Some(lines),
            if position_binning > 0 {
                "Making binned aligned stack for whole tomogram positioning"
            } else {
                "Making final aligned stack"
            },
            false,
            options,
        )?;
    }
    Ok((binning, size_x, size_y))
}

/// Matches `modifyRestrictAndRunAlign` (`IMOD/pysrc/batchruntomo:2717`).
pub fn modify_restrict_and_run_align(
    comfile: &str,
    sed_commands: &[String],
    local_alignment: bool,
    message: &str,
    skip_restrict: bool,
    axis_com_extension: &str,
    directive_files: &[String],
    align_lines: &mut Vec<String>,
    options: &ProcessRunOptions,
) -> Result<(i32, i32), String> {
    fs::write(
        comfile,
        super::pysed::pysed(sed_commands, align_lines, false, '/')?.join("\n") + "\n",
    )
    .map_err(|error| format!("Writing {comfile}: {error}"))?;
    *align_lines = super::pysed::pysed(sed_commands, align_lines, false, '/')?;
    let mut result = 0;
    let mut no_robust = 0;
    if !skip_restrict {
        let mut lines = vec![
            format!("InputFile {comfile}"),
            format!("OutputFile restrictalign{axis_com_extension}"),
        ];
        if local_alignment {
            lines.push("LocalAlignValidation 3".to_owned());
        }
        lines.extend(
            directive_files
                .iter()
                .map(|file| format!("ChangeParametersFile {file}")),
        );
        make_and_run_one_com(
            &mut lines,
            &format!("restrictalign{axis_com_extension}"),
            &format!(
                "Running restrictalign to optimize {}alignment parameters",
                if local_alignment { "local " } else { "global " }
            ),
            false,
            false,
            0,
            "",
            options,
        )?;
        let log =
            read_text_file_report_err(&format!("restrictalign.log"), None).unwrap_or_default();
        if log.iter().any(|line| line.contains("No restriction")) {
            result = 0;
        } else {
            result = 1;
        }
        if log.iter().any(|line| line.contains("[rsa2]")) {
            no_robust = if log
                .iter()
                .any(|line| line.contains("[rsa2]") && line.contains("fail"))
            {
                2
            } else {
                1
            };
        }
        if result != 0 {
            *align_lines = read_text_file_report_err(comfile, None)?;
        }
    }
    if run_one_process(comfile, true, false, message, options)? < 0 {
        return Ok((-2, no_robust));
    }
    Ok((result, no_robust))
}

/// State-owning source-operation interface for the multi-stage `runTiltalign` flow.
pub trait TiltAlignWorkflow {
    fn patch_tracking(&self) -> bool;
    fn skip_tiltalign(&self) -> bool;
    fn setup_parameters(&mut self) -> Result<(), String>;
    fn patch_tracking_alignment(&mut self) -> Result<(), String>;
    fn global_alignment(&mut self) -> Result<(), String>;
    fn restricted_alignment(&mut self) -> Result<(), String>;
    fn local_alignment(&mut self) -> Result<(), String>;
    fn analyze_results(&mut self) -> Result<(), String>;
}

/// Matches `runTiltalign` (`IMOD/pysrc/batchruntomo:2826`).
pub fn run_tiltalign<Workflow: TiltAlignWorkflow>(workflow: &mut Workflow) -> Result<(), String> {
    workflow.setup_parameters()?;
    if workflow.patch_tracking() {
        workflow.patch_tracking_alignment()?;
    }
    if workflow.skip_tiltalign() {
        return Ok(());
    }
    workflow.global_alignment()?;
    workflow.restricted_alignment()?;
    workflow.local_alignment()?;
    workflow.analyze_results()
}

/// Matches `renameAndAbort` (`IMOD/pysrc/batchruntomo:135`), returning the formatted
/// error to the caller that owns the source command's abort state.
pub fn rename_and_abort(
    from_name: &Path,
    to_name: &Path,
    format_string: Option<&str>,
) -> Result<(), String> {
    fs::rename(from_name, to_name).map_err(|error| {
        let text = format_string
            .unwrap_or("Renaming {} to {}")
            .replacen("{}", &from_name.display().to_string(), 1)
            .replacen("{}", &to_name.display().to_string(), 1);
        format!("{text}: {error}")
    })
}

/// Matches `testDirectiveValue` (`IMOD/pysrc/batchruntomo:145`).
pub fn test_directive_value(
    value: &Result<Option<String>, String>,
    directive: &str,
    data_type: &str,
) -> Result<(), String> {
    match value {
        Err(_) => Err(format!(
            "An error occurred converting the value of the directive {directive} to a {data_type}"
        )),
        Ok(_) => Ok(()),
    }
}
