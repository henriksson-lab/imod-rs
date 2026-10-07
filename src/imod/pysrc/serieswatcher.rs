//! Translation of `IMOD/pysrc/serieswatcher`: watches a directory for tilt
//! series stacks, delivers each one when it is complete, and optionally runs
//! batchruntomo on it through processchunks, alone or in several parallel
//! runs, keeping an eTomo batch project (`.ebt`) up to date.
//!
//! The script's module globals live in [`Watcher`]; its functions are its
//! methods and its top level is [`serieswatcher`].  The main loop's globals
//! that the functions read (`stack`, `stackBase`, `stackRoot`, `stackExt`,
//! `setRoot`, `runOneAxis`, `firstReadySlot`, `datasetDir`) are fields, as
//! module globals they are assigned both by the loop and by the functions.
//!
//! Commands: `extracttilts`, `montagesize` and `header` are our own programs
//! and run through `imodpy::run_cmd` (in process); `getmrcsize` is the direct
//! `imodpy::get_mrc_size`.  processchunks (and the batchruntomo it runs) is a
//! detached background job started by `prochunks::start_processchunks`, as
//! in the source.
//!
//! **`KeyboardInterrupt`.**  Python raises it wherever the script is when
//! SIGINT arrives; it is almost always in the two-second `time.sleep`.  Here
//! a SIGINT handler records the signal, the sleep is taken in short steps
//! that check it, and the loop also checks it at the top of each pass; the
//! `except KeyboardInterrupt` arm then runs as in the source.  An end of
//! file at its `input()` prompt (an `EOFError` traceback natively) is taken
//! as an empty answer.
//!
//! Upstream bugs fixed in translation (each in BUGS.md):
//! - Without a command file (delivery only), every pass of the loop calls
//!   `checkForProChunksQuit(topCheckFile, pcCheckFiles[ind])`, names that
//!   exist only when reconstructing, so the script dies on its first pass
//!   with a `NameError` traceback.  There is no check file to look at then,
//!   so the call is skipped.
//! - When the A stack of a dual-axis pair becomes ready while its B stack is
//!   being held back, the two lines clearing both wait states are indented
//!   under `if debugMode:`, so without `-debug` neither stack ever runs.
//!   They run unconditionally here.
//! - A failed move of the stack retries through `waitingList[ind]`, where
//!   `ind` is the module global left by the last slot loop, not the stack
//!   being delivered; the ready stack's own entry is used here.
//! - After a run, a failure to read the data set's cumulative
//!   `batchruntomo.log` makes the step scan iterate over the *characters*
//!   of the error message, though the comment says to fall back to the
//!   command-file log; that log's lines are scanned here.
//! - After `runBRTonStack` fails in `header`, the error lines are fetched
//!   into `errLines` but the loop prints `errStrings`, imodpy's module
//!   global as it was at import (empty); `errLines` is printed here.

use super::comchanger::abs_template_path;
use super::imodpy::{
    BOOL_VALUE, FLOAT_VALUE, INT_VALUE, OptionValue, STRING_VALUE, add_imod_bin_ignore_sighup,
    cleanup_files, complete_and_check_com_file, convert_to_integer, default_com_extension, fmtstr,
    get_err_strings, get_mrc_size, glob_glob, imod_abs_path, make_backup_file,
    move_or_copy_with_retry, option_value, os_path_splitext, prnstr, py_float, py_int,
    read_text_file, run_cmd, write_text_file,
};
use super::pip::{
    exit_error, pip_get_boolean, pip_get_err_no, pip_get_float, pip_get_integer, pip_get_string,
    pip_read_or_parse_options,
};
use super::prochunks::{
    ProChunksParam, check_for_pro_chunks_quit, check_pro_chunks_log,
    get_translation_from_remote_dir, start_processchunks, transfer_remote_directory,
};
use super::pysed::{PysedSrc, pysed};
use std::collections::HashMap;
use std::ffi::OsString;
use std::io::Write as _;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

const PROGNAME: &str = "serieswatcher";
const OPEN_TS_EXT: &str = ".openTS";
const ACTIVE_EBT: &str = ".ebt.active";

/// Rust-only: set by the SIGINT handler, standing for Python's
/// `KeyboardInterrupt`; see the module documentation.
static KEY_INTERRUPT: AtomicBool = AtomicBool::new(false);

extern "C" fn key_interrupt(_signal: libc::c_int) {
    KEY_INTERRUPT.store(true, Ordering::SeqCst);
}

/// `os.path.basename`
fn basename(path: &str) -> String {
    path.rsplit('/').next().unwrap_or(path).to_owned()
}

/// `os.path.dirname`
fn dirname(path: &str) -> String {
    match path.rfind('/') {
        Some(index) => {
            let head = &path[..index + 1];
            if head.chars().all(|c| c == '/') {
                head.to_owned()
            } else {
                head.trim_end_matches('/').to_owned()
            }
        }
        None => String::new(),
    }
}

/// `os.path.join`
fn join(parts: &[&str]) -> String {
    let mut joined = String::new();
    for (index, part) in parts.iter().enumerate() {
        if index == 0 || part.starts_with('/') {
            joined = (*part).to_owned();
        } else {
            if !joined.is_empty() && !joined.ends_with('/') {
                joined.push('/');
            }
            joined.push_str(part);
        }
    }
    joined
}

/// `os.path.getmtime`: a float of seconds
fn getmtime(path: &str) -> std::io::Result<f64> {
    let meta = std::fs::metadata(path)?;
    Ok(crate::imod::libcfshr::b3dutil::py_st_mtime(&meta))
}

/// `time.time()`
fn now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs_f64())
        .unwrap_or(0.)
}

/// One `waitingList` entry; the fields are the source's indexes `StackInd`
/// .. `ABstatusInd` (`serieswatcher:13-27`).
struct Waiter {
    /// `StackInd`: full stack name
    stack: String,
    /// `MtimeInd`: modification time, updated when mtime or size changes
    mtime: f64,
    /// `SizeInd`: size, updated when mtime or size changes
    size: u64,
    /// `TimeInd`: time of those changes
    time: f64,
    /// `OtsInd`: flag for OpenTS seen, initially 0, 1 when seen, 2 when
    /// disappears
    ots: i32,
    /// `MaturInd`: maturation time to allow for particular state of OTS seen,
    /// set to 0 when mature
    matur: f64,
    /// `ErrTimeInd`: 0 or time when error occurred for stack
    err_time: f64,
    /// `ABstatusInd`: initially 1 for A stack when setup to wait for B and set
    /// to 2 when A matures to prevent multiple checks, set to -1 for a B stack
    /// when it gets skipped to wait for A
    ab_status: i32,
}

/// The module globals of the script.
#[derive(Default)]
struct Watcher {
    // Options and parameters
    watch_dir: String,
    deliver_dir: String,
    pattern: String,
    never_ots_matured_time: f64,
    remote_start_dir: String,
    min_tilt_range: f64,
    min_zsize: i32,
    ots_there_matured_time: f64,
    debug_mode: bool,
    etomo_options: i32,
    remote_run_dir: String,
    remote_com_dir: String,
    master_com: String,
    project_root: String,
    out_proj_root: String,
    out_ebt_lines: Vec<String>,
    num_ebt_rows: i32,
    ending_step: f64,
    check_for_a_wait_for_b: bool,
    project_adoc_written: bool,
    recheck_erred_time: f64,
    max_retries_if_no_open_ts: i32,
    max_size_in_existing_dir: u64,
    dual_axis: i32,
    num_surfaces: i32,
    dual_text: String,
    montage_text: String,
    two_surf_text: String,
    cur_ext_text: String,
    cur_b_ext_text: String,
    line_num_dict: HashMap<String, (i64, String)>,
    batch_lines: Vec<String>,
    reduced_lines: Vec<String>,
    gpu_list_in: String,
    num_parallel: usize,
    cpu_lists: Vec<String>,
    cpu_for_brt: Vec<String>,
    running_dir: String,
    abs_running_dir: String,
    starting_dir: String,
    top_check_file: String,
    slot_busy: Vec<bool>,
    stack_in_slot: Vec<String>,
    log_for_slot: Vec<String>,
    ebt_line_for_slot: Vec<i64>,
    ebt_num_for_slot: Vec<String>,
    pc_check_files: Vec<String>,
    pc_out_files: Vec<String>,
    param_array: Vec<ProChunksParam>,

    // The watching
    waiting_list: Vec<Waiter>,
    erred_list: Vec<String>,
    done_list: Vec<String>,
    pausing: bool,
    killing: bool,

    // Globals of the main loop
    stack: String,
    stack_base: String,
    stack_root: String,
    stack_ext: String,
    set_root: String,
    run_one_axis: i32,
    first_ready_slot: i64,
    dataset_dir: String,
    ready_stack_ind: usize,
}

impl Watcher {
    /// Matches `renameEbtAtEnd` (`serieswatcher:28`).
    fn rename_ebt_at_end(&self) {
        if self.etomo_options % 2 == 1 {
            return;
        }
        let active = format!("{}{ACTIVE_EBT}", self.out_proj_root);
        if !self.project_root.is_empty() && Path::new(&active).exists() {
            let ebt = format!("{}.ebt", self.out_proj_root);
            make_backup_file(&ebt);
            // An OSError here is uncaught natively
            if let Err(error) = std::fs::rename(&active, &ebt) {
                exit_error(&format!("Renaming {active} to {ebt}: {error}"));
            }
        }
    }

    /// Matches `processCheckAction` (`serieswatcher:36`): give message and
    /// exit for an action of F or Q.
    fn process_check_action(&mut self, action: &str) {
        if action != "Q" && action != "F" {
            return;
        }
        if action == "Q" {
            prnstr("Sent signal to quit to all processes", "\n", true);
            self.killing = true;
        } else if !self.pausing {
            prnstr("Allowing all runs to finish then exiting", "\n", true);
            self.pausing = true;
        }
    }

    /// Matches `findStackInWaitingList` (`serieswatcher:49`): look for a stack
    /// in the waiting list, return index or -1 if not there.
    fn find_stack_in_waiting_list(&self, stack: &str) -> i64 {
        for ind in 0..self.waiting_list.len() {
            let stack_base = basename(&self.waiting_list[ind].stack);
            if stack == stack_base {
                return ind as i64;
            }
        }

        -1
    }

    /// Matches `checkForStacks` (`serieswatcher:59`): check for new stacks
    /// and completion of ones already found.  Returns `(numReady,
    /// firstReady)`.
    fn check_for_stacks(&mut self) -> (i32, i64) {
        let mut num_ready = 0;
        let mut first_ready: i64 = -1;

        // Get the full list of matching files and loop on it
        let mut full_list = glob_glob(&join(&[&self.watch_dir, &self.pattern]));
        full_list.sort();
        for stack in full_list {
            // get root and extension, eliminate unwanted extensions and a axis files
            let (root, ext) = os_path_splitext(&stack);
            if ext == ".mdoc"
                || ext == OPEN_TS_EXT
                || self.erred_list.contains(&stack)
                || self.done_list.contains(&stack)
            {
                continue;
            }

            // Check if stack is in waiting list already, if not add it
            if self.waiting_list.iter().any(|waiter| waiter.stack == stack) {
                continue;
            }
            let mut is_a_stack = 0;
            if self.check_for_a_wait_for_b && root.ends_with('a') {
                is_a_stack = 1;
            }
            let entry = getmtime(&stack)
                .and_then(|mtime| std::fs::metadata(&stack).map(|meta| (mtime, meta.len())));
            match entry {
                Ok((mtime, size)) => {
                    self.waiting_list.push(Waiter {
                        stack: stack.clone(),
                        mtime,
                        size,
                        time: now(),
                        ots: 0,
                        matur: self.never_ots_matured_time,
                        err_time: 0.,
                        ab_status: is_a_stack,
                    });
                    if self.debug_mode {
                        let mut mess = format!("Added {stack} to waitingList");
                        if is_a_stack != 0 {
                            mess += " just to keep track of while awaiting B";
                        }
                        prnstr(&mess, "\n", true);
                    }
                }
                Err(_) => {
                    prnstr(
                        &format!(
                            "{stack} gave error getting size or modification time; skipping it"
                        ),
                        "\n",
                        false,
                    );
                    self.erred_list.push(stack);
                }
            }
        }

        // Go through list checking for openTS and times
        // Do this with C-style loop index in order to go forward and be able to remove as
        // you go.  Increment index before any continue; drop index when remove an item
        let mut ind = 0;
        let mut now_time = 0.0_f64;
        let mut rem_mess = String::new();
        while ind < self.waiting_list.len() {
            let mut remove_it = false;
            let mut test_for_removal = false;
            let stack = self.waiting_list[ind].stack.clone();
            let err_time = self.waiting_list[ind].err_time;
            let stack_base = basename(&stack);
            let (stack_root, stack_ext) = os_path_splitext(&stack_base);
            let a_stack_busy = !self.master_com.is_empty()
                && self.dual_axis != 0
                && stack_root.ends_with('b')
                && self.stack_in_slot.contains(&format!(
                    "{}a{stack_ext}",
                    &stack_root[..stack_root.len() - 1]
                ));

            // If a stack disappears, remove it
            if err_time == 0. && !Path::new(&stack).exists() {
                self.waiting_list.remove(ind);
                continue;
            }

            // If a stack is mature, skip it if the A axis is running
            if self.waiting_list[ind].matur == 0. && self.waiting_list[ind].ab_status == 0 {
                if a_stack_busy {
                    ind += 1;
                    continue;
                }

                // OK to run, record if it is the first one and skip the rest of this
                if num_ready == 0 {
                    first_ready = ind as i64;
                }
                num_ready += 1;
                ind += 1;
                continue;
            }

            // Check for whether openTS file is there, change maturation time when it is
            // first seen and when it disappears.  Do not check and errored one for this
            if self.waiting_list[ind].ots < 2 && err_time == 0. {
                if Path::new(&format!("{stack}{OPEN_TS_EXT}")).exists() {
                    if self.waiting_list[ind].ots <= 0 {
                        if self.debug_mode {
                            prnstr(&format!("Saw .openTS {stack_base}"), "\n", false);
                        }
                        self.waiting_list[ind].ots = 1;
                        self.waiting_list[ind].matur = self.ots_there_matured_time;
                    }
                } else if self.waiting_list[ind].ots > 0 {
                    if self.debug_mode {
                        prnstr(&format!(".openTS gone for {stack_base}"), "\n", false);
                    }
                    self.waiting_list[ind].ots = 2;
                    self.waiting_list[ind].matur = 2.;
                }
            }

            // Check if the modification time or the size has changed and keep track of
            // time when that happened, for direct comparison with current time
            let checked: Result<bool, std::io::Error> = (|| {
                // Check an errored one only occasionally
                now_time = now();
                if err_time != 0.
                    && (now_time - err_time < self.recheck_erred_time
                        || !Path::new(&stack).exists())
                {
                    ind += 1;
                    return Ok(true);
                }

                let this_mtime = getmtime(&stack)?;
                let this_size = std::fs::metadata(&stack)?.len();

                // If there is change, record the time of it
                if this_mtime != self.waiting_list[ind].mtime
                    || this_size != self.waiting_list[ind].size
                {
                    self.waiting_list[ind].mtime = this_mtime;
                    self.waiting_list[ind].size = this_size;
                    self.waiting_list[ind].time = now_time;

                    // If an errored one changed, revive it, starting fresh
                    if err_time != 0. {
                        self.waiting_list[ind].err_time = 0.;
                        self.waiting_list[ind].ots = 0;
                        self.waiting_list[ind].matur = self.never_ots_matured_time;
                        if self.debug_mode {
                            prnstr(
                                &format!("{stack_base} has changed, so it will be watched again"),
                                "\n",
                                false,
                            );
                        }
                        return Ok(true);
                    }
                }

                let mod_diff = 0f64.max(now_time - self.waiting_list[ind].time);

                // It is mature, check that its angle range is enough
                if mod_diff > self.waiting_list[ind].matur
                    && self.waiting_list[ind].ab_status >= 0
                    && self.waiting_list[ind].ab_status < 2
                {
                    let mut test_header = true;
                    if let Ok(Some(tilt_lines)) =
                        run_cmd(&format!("extracttilts \"{stack}\""), None, None, None, &[])
                    {
                        if tilt_lines.len() as i64 > self.min_zsize as i64 {
                            let mut min_angle = 999.0_f64;
                            let mut max_angle = -999.0_f64;
                            let mut value_error = false;
                            for lin_ind in (0..tilt_lines.len()).rev() {
                                if tilt_lines[lin_ind].trim().is_empty() {
                                    continue;
                                }
                                let lsplit: Vec<&str> =
                                    tilt_lines[lin_ind].split_whitespace().collect();
                                if lsplit.len() > 1 {
                                    break;
                                }
                                match py_float(lsplit[0]) {
                                    Some(angle) => {
                                        // `min(minAngle, angle)` and `max(maxAngle, angle)`
                                        if angle < min_angle {
                                            min_angle = angle;
                                        }
                                        if angle > max_angle {
                                            max_angle = angle;
                                        }
                                    }
                                    None => {
                                        value_error = true;
                                        break;
                                    }
                                }
                            }

                            if !value_error {
                                if self.debug_mode {
                                    prnstr(
                                        &format!(
                                            "Min and max angle {min_angle:.1}  {max_angle:.1}"
                                        ),
                                        "\n",
                                        false,
                                    );
                                }

                                // Report a tilt series with low range, be silent about
                                // non-tilts
                                if max_angle >= min_angle {
                                    rem_mess = format!(
                                        "{stack_base} has a tilt range of only {:.0} deg; skipping it for now",
                                        max_angle - min_angle
                                    );
                                    test_header = false;
                                    test_for_removal = max_angle - min_angle < self.min_tilt_range;
                                }
                            }
                        }
                    }

                    // Test Z range if angle range could not be found
                    if test_header {
                        rem_mess = format!(
                            "{stack_base} gave an error reading the header, skipping it for now"
                        );

                        // First try to get montage size, then header Z size
                        let mut nz: i64 = 0;
                        let mont_nz =
                            run_cmd(&format!("montagesize \"{stack}\""), None, None, None, &[])
                                .ok()
                                .flatten()
                                .and_then(|mont_size| {
                                    let first = mont_size.first()?;
                                    let lsplit: Vec<&str> = first.split_whitespace().collect();
                                    py_int(lsplit.last()?)
                                });
                        match mont_nz {
                            Some(value) => nz = value,
                            None => match get_mrc_size(&stack) {
                                Ok((_nx, _ny, z)) => nz = z as i64,
                                Err(_) => test_for_removal = true,
                            },
                        }

                        if !test_for_removal && nz < self.min_zsize as i64 {
                            rem_mess = fmtstr(
                                "{} has only {} views; skipping it for now",
                                &[stack_base.clone(), nz.to_string()],
                            );
                            test_for_removal = true;
                        }
                    }

                    // If it passes all the tests, zero the mature time and consider it
                    // ready unless the A stack is running
                    if !test_for_removal && self.waiting_list[ind].ab_status >= 0 {
                        self.waiting_list[ind].matur = 0.;

                        // If this is a checked A stack, increase status to have this happen
                        // only once and clear out B status if it exists
                        if self.waiting_list[ind].ab_status > 0 {
                            self.waiting_list[ind].ab_status = 2;
                            let b_stack_name =
                                format!("{}b{stack_ext}", &stack_root[..stack_root.len() - 1]);
                            let b_ind = self.find_stack_in_waiting_list(&b_stack_name);
                            if b_ind >= 0 && self.waiting_list[b_ind as usize].ab_status != 0 {
                                if self.debug_mode {
                                    prnstr(
                                        &format!(
                                            "A is ready: clearing wait status for A and B stacks {b_stack_name}"
                                        ),
                                        "\n",
                                        false,
                                    );
                                }
                                // Fixed in translation (BUGS.md): native has these two
                                // under `if debugMode:`
                                self.waiting_list[b_ind as usize].ab_status = 0;
                                self.waiting_list[ind].ab_status = 0;
                            }
                        }

                        if !a_stack_busy && self.waiting_list[ind].ab_status == 0 {
                            if num_ready == 0 {
                                first_ready = ind as i64;
                            }
                            num_ready += 1;
                        }
                    }
                }
                Ok(false)
            })();
            match checked {
                Ok(true) => continue,
                Ok(false) => {}
                Err(_) => {
                    if err_time == 0. {
                        prnstr(
                            &format!(
                                "{stack_base} gave error getting size or modification time; skipping it"
                            ),
                            "\n",
                            true,
                        );
                    }
                    remove_it = true;
                }
            }

            // If no openTS was seen, retry a few times, otherwise remove it
            if test_for_removal {
                if self.waiting_list[ind].ots <= 0
                    && -self.waiting_list[ind].ots < self.max_retries_if_no_open_ts
                {
                    self.waiting_list[ind].ots -= 1;
                    self.waiting_list[ind].time = now_time;
                    if self.debug_mode {
                        prnstr(
                            &format!("{stack_base} does not pass tests, retry it"),
                            "\n",
                            false,
                        );
                    }
                } else {
                    remove_it = true;
                    prnstr(&rem_mess, "\n", true);
                }
            }

            // Mark stack as errored, actually remove it if this didn't follow a test
            if remove_it {
                self.waiting_list[ind].err_time = now_time;
                if !test_for_removal {
                    self.waiting_list.remove(ind);
                    continue;
                }
            }

            // Increment ind at loop end
            ind += 1;
        }

        (num_ready, first_ready)
    }

    /// Matches `deliverStackAndMdoc` (`serieswatcher:255`): deliver the stack
    /// if there is a delivery directory.  Returns `skipIt`.
    fn deliver_stack_and_mdoc(&mut self) -> i32 {
        let do_deliver = !self.deliver_dir.is_empty()
            && imod_abs_path(&self.deliver_dir) != imod_abs_path(&self.watch_dir);
        let mut skip_it = 0;
        let mut mess = String::new();
        let stack_base = self.stack_base.clone();

        // Does the stack itself exist in other place?
        if do_deliver && Path::new(&join(&[&self.deliver_dir, &stack_base])).exists() {
            mess = format!("stack already exists in {}", self.deliver_dir);
            skip_it = 1;
        }

        if skip_it == 0 && !self.master_com.is_empty() {
            // For dual axis, A must exist somewhere or the directory must
            let set_dir = join(&[&self.running_dir, &self.set_root]);
            let data_dir_exists = Path::new(&set_dir).exists();
            let mut data_dir_reusable = false;

            // If the dir exists and the stack does not exist there, test for the size of
            // files to see if it is "reusable"
            if data_dir_exists && !Path::new(&join(&[&set_dir, &stack_base])).exists() {
                // An OSError in listdir or getsize is uncaught natively
                let all_files: Vec<String> = match std::fs::read_dir(&set_dir) {
                    Ok(entries) => entries
                        .flatten()
                        .map(|entry| entry.file_name().to_string_lossy().into_owned())
                        .collect(),
                    Err(error) => exit_error(&format!("Listing {set_dir}: {error}")),
                };
                data_dir_reusable = true;
                for inside in all_files {
                    let size = std::fs::metadata(join(&[&set_dir, &inside]))
                        .map(|meta| meta.len())
                        .unwrap_or(0);
                    if size > self.max_size_in_existing_dir {
                        data_dir_reusable = false;
                        break;
                    }
                }
            }

            // Do first axis if that is what we got
            self.run_one_axis = 0;
            if self.dual_axis != 0 && self.stack_root.ends_with('a') {
                self.run_one_axis = 1;
            }
            if self.dual_axis != 0 && self.run_one_axis == 0 {
                // If we have B axis, test for a stack, whether it exists and needs to be
                // moved
                let a_stack_base = format!("{}a{}", self.set_root, self.stack_ext);
                let a_stack = join(&[&self.watch_dir, &a_stack_base]);
                let mut a_exists_in_watch = Path::new(&a_stack).exists();

                // If A exists here, look it up in list and say it doesn't exist if it
                // isn't mature and ready
                if a_exists_in_watch {
                    let a_ind = self.find_stack_in_waiting_list(&a_stack_base);
                    if a_ind < 0 || self.waiting_list[a_ind as usize].matur != 0. {
                        if self.debug_mode {
                            prnstr(
                                &format!(
                                    "A stack {a_stack_base} is not ready; acting like it does not exist"
                                ),
                                "\n",
                                false,
                            );
                        }
                        a_exists_in_watch = false;
                    }
                }

                let a_is_moved = Path::new(&join(&[&self.running_dir, &a_stack_base])).exists();
                let move_a_stack = !self.deliver_dir.is_empty() && a_exists_in_watch;

                // If we still think the data dir is reusable but the stack exists there,
                // it is not reusable - this is a weak test since stacks can be renamed
                if data_dir_exists
                    && data_dir_reusable
                    && Path::new(&join(&[&self.running_dir, &self.set_root, &a_stack_base]))
                        .exists()
                {
                    data_dir_reusable = false;
                }

                // Error if there is more than one A stack or if there is an A stack
                // and the data directory exists
                if move_a_stack && a_is_moved {
                    mess = format!(
                        "the first axis stack exists in both {} and {}",
                        self.running_dir, self.watch_dir
                    );
                    skip_it = 1;
                } else if (move_a_stack || a_is_moved) && data_dir_exists && !data_dir_reusable {
                    mess =
                        "the data set directory already exists but there is a first axis stack in "
                            .to_owned();
                    if move_a_stack {
                        mess += &self.watch_dir;
                    } else {
                        mess += &self.running_dir;
                    }
                    skip_it = 1;
                }
                // Error if nothing exists anywhere
                else if !(move_a_stack || a_is_moved || (data_dir_exists && !data_dir_reusable)) {
                    mess = "there is no first axis stack or data set directory".to_owned();
                    skip_it = -1;
                }
                // If the directory does exist and could have the first axis, then just do
                // the second axis
                else if data_dir_exists && !data_dir_reusable {
                    self.run_one_axis = 2;
                }
                // Otherwise now copy the A stack and mdoc file
                else if move_a_stack {
                    if move_or_copy_with_retry(
                        &a_stack,
                        &self.deliver_dir,
                        "moving first axis stack",
                        false,
                        1,
                    ) != 0
                    {
                        skip_it = 1;
                    } else {
                        let mdoc_file = format!("{a_stack}.mdoc");
                        if Path::new(&mdoc_file).exists() {
                            move_or_copy_with_retry(
                                &mdoc_file,
                                &self.deliver_dir,
                                "moving first axis .mdoc file",
                                false,
                                5,
                            );
                        }
                    }
                }
            }

            // Now apply similar tests to single/b/a only stack that was watched
            if skip_it == 0 {
                if do_deliver && Path::new(&join(&[&self.deliver_dir, &stack_base])).exists() {
                    mess = format!(
                        "There is already a stack {stack_base} in {}",
                        self.deliver_dir
                    );
                    skip_it = 1;
                } else if self.run_one_axis < 2 && data_dir_exists && !data_dir_reusable {
                    mess = format!(
                        "The data set directory for {stack_base} already exists and has sizable files in it"
                    );
                    skip_it = 1;
                }
            }
        }

        // if not skipping, copy the main stack now
        if skip_it == 0 && do_deliver {
            if move_or_copy_with_retry(&self.stack, &self.deliver_dir, "moving", false, 1) != 0 {
                // If it fails there could be a file lock: give it one or multiple retries
                // as for original testing of stack size etc
                // Fixed in translation (BUGS.md): native indexes the waiting list with
                // the module global `ind`, not the stack being delivered
                let ind = self.ready_stack_ind;
                skip_it = 1;
                let ots = self.waiting_list[ind].ots;
                if (ots <= 0 && -ots < self.max_retries_if_no_open_ts) || ots == 2 {
                    if ots == 2 {
                        self.waiting_list[ind].ots = 3;
                    } else {
                        self.waiting_list[ind].ots -= 1;
                    }
                    self.waiting_list[ind].time = now();
                    skip_it = 2;
                    if self.debug_mode {
                        prnstr(
                            &format!("{stack_base} failed to move, retry it"),
                            "\n",
                            false,
                        );
                    }
                }
            } else {
                if self.master_com.is_empty() {
                    prnstr(&format!("Moved stack {stack_base}"), "\n", true);
                }
                let mdoc_file = format!("{}.mdoc", self.stack);
                if Path::new(&mdoc_file).exists() {
                    move_or_copy_with_retry(&mdoc_file, &self.deliver_dir, ".mdoc file", false, 5);
                }
            }
        }

        if skip_it == 1 || skip_it == -1 {
            let mut mess_out = format!("Skipping stack {stack_base}");
            if !mess.is_empty() {
                mess_out += &format!(" ; {mess}");
            }
            prnstr(&mess_out, "\n", true);
            if skip_it > 0 {
                self.erred_list.push(self.stack.clone());
            }
        }

        skip_it
    }

    /// Matches `changeOrAddEBTline` (`serieswatcher:437`): change or add one
    /// line in ebt.  Pass 'key=' to remove line with key.
    fn change_or_add_ebt_line(&mut self, text: &str) {
        let tsplit: Vec<&str> = text.split('=').collect();
        for ind in 0..self.out_ebt_lines.len() {
            let line = self.out_ebt_lines[ind].clone();
            let lsplit: Vec<&str> = line.split('=').collect();
            if lsplit[0] == tsplit[0] {
                if tsplit.len() == 1 || tsplit[1].is_empty() {
                    self.out_ebt_lines[ind] = String::new();
                } else {
                    self.out_ebt_lines[ind] = text.to_owned();
                }
                return;
            }
        }

        self.out_ebt_lines.push(text.to_owned());
    }

    /// Matches `changeEBTlines` (`serieswatcher:454`): change a set of lines
    /// in the ebt file and rewrite the file.
    #[allow(clippy::too_many_arguments)]
    fn change_ebt_lines(
        &mut self,
        _start_line: i64,
        num_text: &str,
        path: &str,
        run: bool,
        log_enabled: bool,
        etomo_enabled: bool,
        tomo_done: bool,
        trim_done: bool,
        rec_enabled: bool,
        status: &str,
        end_step: i32,
    ) {
        let tf_text = |value: bool| if value { "true" } else { "false" };
        let start_text = format!("meta.row.{num_text}");
        if !path.is_empty() {
            self.change_or_add_ebt_line(&format!("meta.ref.{num_text}={path}"));
        }
        self.change_or_add_ebt_line(&format!("{start_text}.Run={}", tf_text(run)));
        self.change_or_add_ebt_line(&format!(
            "{start_text}.Log.Enabled={}",
            tf_text(log_enabled)
        ));
        self.change_or_add_ebt_line(&format!(
            "{start_text}.Etomo.Enabled={}",
            tf_text(etomo_enabled)
        ));
        self.change_or_add_ebt_line(&format!(
            "{start_text}.Tomogram.Done={}",
            tf_text(tomo_done)
        ));
        self.change_or_add_ebt_line(&format!("{start_text}.Trimvol.Done={}", tf_text(trim_done)));
        self.change_or_add_ebt_line(&format!(
            "{start_text}.Rec.Enabled={}",
            tf_text(rec_enabled)
        ));
        if !status.is_empty() {
            self.change_or_add_ebt_line(&format!("{start_text}.DatasetStatus={status}"));
        }
        if end_step >= 0 {
            self.change_or_add_ebt_line(&format!("{start_text}.EndingStep={end_step}"));
        } else {
            self.change_or_add_ebt_line(&format!("{start_text}.EndingStep="));
        }

        if let Err(err) = write_text_file(
            &format!("{}{ACTIVE_EBT}", self.out_proj_root),
            &self.out_ebt_lines,
            true,
        ) {
            prnstr(&format!("WARNING: {err}"), "\n", true);
        }
    }

    /// Matches `initializeEBTentry` (`serieswatcher:480`): set up a new set of
    /// entries in the ebt lines or modify an old one, or leave a axis alone.
    /// Returns the index of the line with the path that starts the set of 9
    /// lines, and the `(ebtN)` text.
    fn initialize_ebt_entry(&mut self, path: &str) -> (i64, String) {
        let mut do_change = true;
        let mut ind_new: i64 = -1;
        let mut num_text = String::new();

        // If dual axis and this is the b axis, look for an existing entry
        if self.dual_axis != 0 {
            let (path_root, _ext) = os_path_splitext(path);
            let path_base = basename(&path_root);
            if path_root.ends_with('b') {
                let set_base = &path_base[..path_base.len() - 1];
                for ind in 0..self.out_ebt_lines.len() {
                    let line = &self.out_ebt_lines[ind];
                    if line.starts_with("meta.ref.ebt") && line.contains(set_base) {
                        if let Some(ind_equal) = line.find('=').filter(|index| *index > 0) {
                            // IF found a valid entry with =, set the index, and set flag
                            // to change the existing entry if it is a b axis entry
                            let (ref_base, _ext) =
                                os_path_splitext(&basename(&line[ind_equal + 1..]));
                            if ref_base.starts_with(set_base) {
                                ind_new = ind as i64;
                                do_change = ref_base.ends_with('b');
                                num_text = line[9..ind_equal].to_owned();
                                break;
                            }
                        }
                    }
                }
            }
        }

        // If not an existing set, make new lines
        if ind_new < 0 {
            self.num_ebt_rows += 1;
            num_text = format!("ebt{}", self.num_ebt_rows);

            // Set the last ID and row #, then do standard change
            self.change_or_add_ebt_line(&format!("meta.ref.ebt.lastID={num_text}"));
            self.out_ebt_lines.push(format!(
                "meta.row.{num_text}.RowNumber={}",
                self.num_ebt_rows
            ));
            self.out_ebt_lines.push(format!(
                "meta.row.{num_text}.dual={}",
                if self.dual_axis != 0 { "true" } else { "false" }
            ));
            self.out_ebt_lines
                .push(format!("meta.row.{num_text}.OrigStack={path}"));
            ind_new = self.out_ebt_lines.len() as i64;
        }

        // Set the lines
        if do_change {
            self.change_ebt_lines(
                ind_new, &num_text, path, true, true, false, false, false, false, "Running", -1,
            );
        }

        (ind_new, format!("({num_text})"))
    }

    /// Matches `parseBRTlogUpdateEBT` (`serieswatcher:526`): go through the
    /// log to determine state of data set and update the ebt lines.
    /// `log_lines` is `Err` when the source's `logLines` is the error string.
    fn parse_brt_log_update_ebt(
        &mut self,
        stack: &str,
        log_lines: &Result<Vec<String>, String>,
        line_in_ebt: i64,
    ) {
        let mut reached_step = [-1i32, -1];
        let mut failed = [false, false];
        let mut new_name = [String::new(), String::new()];
        let mut delivered = false;
        let mut axis_ind = 0;

        // Read the cumulative log in the dataset if possible, fall back to the com log
        let cumul_log = join(&[&self.running_dir, &self.dataset_dir, "batchruntomo.log"]);
        let cumul_lines = match read_text_file(&cumul_log, None, true, None) {
            Ok(lines) => lines,
            Err(cumul_error) => match log_lines {
                Err(_) => {
                    prnstr(
                        &format!("WARNING: Cannot update .ebt file: {cumul_error}"),
                        "\n",
                        true,
                    );
                    return;
                }
                // Fixed in translation (BUGS.md): native goes on to iterate
                // over the characters of the error message
                Ok(lines) => lines.clone(),
            },
        };

        // detect rename and delivery from the com log
        if let Ok(lines) = log_lines {
            for line in lines {
                if line.contains("[brt9]") {
                    if let Some(to_ind) = line.find(" to: ").filter(|index| *index > 0) {
                        new_name[axis_ind] = line[to_ind + 5..].to_owned();
                    }
                }

                if line.contains("[brt8]") {
                    delivered = true;
                }
            }
        }

        // Loop on lines of dataset log
        for line in &cumul_lines {
            // Keep track of step reached
            if line.starts_with("Reached step") {
                if let Some(step) = py_int(&line[12..]) {
                    reached_step[axis_ind] = (step as i32).max(reached_step[axis_ind]);
                }
            }
            if line.contains("Successfully finished volcombine") {
                reached_step[0] = 19;
            }
            if line.contains("Successfully finished trimvol") {
                reached_step[0] = 20;
            }

            // Detect what axis we are in and look for axis failure/success
            if self.dual_axis != 0 {
                if line.contains("Starting axis") {
                    axis_ind = 0;
                    if line.contains("axis B") {
                        axis_ind = 1;
                    }
                }
                if line.contains("Completed axis") {
                    failed[axis_ind] = false;
                }
                if line.contains("ABORT AXIS") {
                    failed[axis_ind] = true;
                }
            }

            if line.contains("ABORT SET") {
                failed[0] = true;
            }
        }

        // Condense arrays into first/only axis slot
        if self.dual_axis != 0 {
            if failed[1] {
                failed[0] = true;
            }
            if reached_step[1] > -1 && reached_step[1] < 14 {
                reached_step[0] = reached_step[0].min(reached_step[1]);
            }
            if new_name[0].is_empty() {
                new_name[0] = new_name[1].clone();
            }
        }

        // Set the name to the undelivered stack
        let mut name = join(&[&self.running_dir, stack]);
        let path_line = self.out_ebt_lines[line_in_ebt as usize].clone();
        let ind_equal = path_line.find('=');
        let mut num_text = path_line.get(9..).unwrap_or("").to_owned();
        if let Some(ind_equal) = ind_equal.filter(|index| *index > 0) {
            // If old name found and it ends with a and this is b axis, leave it alone
            num_text = path_line.get(9..ind_equal).unwrap_or("").to_owned();
            let old_path_base = basename(&path_line[ind_equal + 1..]);
            let (old_root, _old_ext) = os_path_splitext(&old_path_base);
            if self.dual_axis != 0 && self.stack_root.ends_with('b') && old_root.ends_with('a') {
                name = String::new();
            }
        }

        // Modify name for delivery and rename
        if !name.is_empty() && delivered {
            name = join(&[&self.running_dir, &self.dataset_dir, stack]);
        }
        if !name.is_empty() && !new_name[0].is_empty() {
            let (old_root, _old_ext) = os_path_splitext(&name);
            let (_new_root, new_ext) = os_path_splitext(&new_name[0]);
            name = old_root + &new_ext;
        }

        // Figure out status
        let mut end_step = reached_step[0];
        let status;
        if self.killing {
            status = "Killed";
        } else if failed[0] {
            status = "Failed";
        } else if self.ending_step >= 0. {
            status = "Stopped";
        } else if self.dual_axis != 0 && self.stack_root.ends_with('a') {
            status = "Awaiting B";
            end_step = 0;
        } else {
            status = "Done";
            end_step = -1;
        }

        let tomo_done = (self.dual_axis == 0 && reached_step[0] >= 14) || reached_step[0] >= 19;
        self.change_ebt_lines(
            line_in_ebt,
            &num_text,
            &name,
            status != "Done",
            true,
            reached_step[0] >= 0,
            tomo_done,
            reached_step[0] >= 20,
            tomo_done,
            status,
            end_step,
        );
    }

    /// Matches `runBRTonStack` (`serieswatcher:633`): run the stack with
    /// batchruntomo.  Returns `(err, mess)`.
    fn run_brt_on_stack(&mut self) -> (i32, String) {
        let slot = self.first_ready_slot as usize;
        let proc_root = format!("swbrt_{}.{}", self.set_root, std::process::id());
        let out_com_file = format!("{proc_root}.{}", default_com_extension());
        let mut adoc_file = format!("{}.adoc", join(&[&self.running_dir, &proc_root]));
        if !self.project_root.is_empty() {
            adoc_file = format!(
                "{}_{}.adoc",
                join(&[&self.running_dir, &self.out_proj_root]),
                self.set_root
            );
        }
        let mut adoc_lines = self.batch_lines.clone();
        self.log_for_slot[slot] = format!("{}.log", join(&[&self.running_dir, &proc_root]));

        // Determine if montage from header output
        let mut montage = 0;
        match run_cmd(
            &format!(
                "header \"{}\"",
                join(&[&self.running_dir, &self.stack_base])
            ),
            None,
            None,
            None,
            &[],
        ) {
            Ok(head_lines) => {
                for line in head_lines.unwrap_or_default() {
                    if line.contains("Piece coordinates") {
                        montage = 1;
                        break;
                    }
                }
            }
            Err(_) => return (2, "Error running header on stack".to_owned()),
        }

        // Modify or add directive lines
        let mut set_direc = |text: &str, value: String| {
            let direc = format!("{text} = {value}");
            let line_num = self.line_num_dict[text].0;
            if line_num >= 0 {
                adoc_lines[line_num as usize] = direc;
            } else {
                adoc_lines.push(direc);
            }
        };
        set_direc(&self.dual_text.clone(), self.dual_axis.to_string());
        set_direc(&self.montage_text.clone(), montage.to_string());
        set_direc(&self.two_surf_text.clone(), self.num_surfaces.to_string());

        let mut stack_ext_text = self.cur_ext_text.clone();
        // `not runOneAxis != 1` is `not (runOneAxis != 1)`
        if self.dual_axis != 0 && self.run_one_axis == 1 {
            stack_ext_text = self.cur_b_ext_text.clone();
        }
        let stack_ext_value = self.stack_ext.get(1..).unwrap_or("").to_owned();
        set_direc(&stack_ext_text, stack_ext_value);

        // Write the adoc
        if let Err(err) = write_text_file(&adoc_file, &adoc_lines, true) {
            return (1, format!("Error {err}"));
        }

        if !self.project_root.is_empty() && !self.project_adoc_written {
            let _ = write_text_file(&format!("{}.adoc", self.out_proj_root), &adoc_lines, false);
            self.project_adoc_written = true;
        }

        // Modify the com file
        let mut sedcom = vec![
            format!("|batchruntomo -St|a|RootName\t{}|", self.set_root),
            format!("|batchruntomo -St|a|DirectiveFile\t{adoc_file}|"),
            format!(
                "|batchruntomo -St|a|CurrentLocation\t{}|",
                self.abs_running_dir
            ),
            format!("|batchruntomo -St|a|CheckFile\t{proc_root}.cmds|"),
            "|batchruntomo -St|a|MakeSubDirectory\t1|".to_owned(),
            format!(
                "|batchruntomo -St|a|CPUMachineList\t{}|",
                self.cpu_lists[slot]
            ),
        ];
        if !self.remote_start_dir.is_empty() || !self.remote_com_dir.is_empty() {
            sedcom.push(format!(
                "|batchruntomo -St|a|RemoteDirectory\t{}|",
                self.remote_run_dir
            ));
        }

        if self.dual_axis != 0 && self.run_one_axis != 0 {
            sedcom.push(format!(
                "|batchruntomo -St|a|ProcessOneAxis\t{}|",
                self.run_one_axis
            ));
        }
        if self.num_parallel > 1 && !self.gpu_list_in.is_empty() {
            sedcom.push(format!(
                "|batchruntomo -St|a|GPUMachineList\t{}|",
                self.gpu_list_in
            ));
        }

        // This is needed to trigger reading translations
        if self.num_parallel > 1 {
            sedcom.push(format!(
                "|batchruntomo -St|a|ParallelBatchRootName\t{}|",
                self.project_root
            ));
        }

        if pysed(
            &sedcom,
            PysedSrc::Lines(&self.reduced_lines),
            Some(&join(&[&self.running_dir, &out_com_file])),
            false,
            '|',
            true,
        )
        .is_err()
        {
            return (1, "Error making batchruntomo command file".to_owned());
        }

        // Run the process from the runningDir and cd back
        let com_array = vec![
            OsString::from(&self.cpu_for_brt[slot]),
            OsString::from("-s"),
            OsString::from(&out_com_file),
        ];
        if std::env::set_current_dir(&self.running_dir).is_err() {
            exit_error(&format!("Changing to directory {}", self.running_dir));
        }

        let mut notify = format!("Starting to process stack: {}    [SRW1]", self.stack_base);
        if self.dual_axis != 0 && self.run_one_axis == 0 {
            notify = format!("Starting to process data set: {}    [SRW1]", self.set_root);
        }
        let mut log_mess = format!(
            "  with log in: {}.log    [SRW2]",
            join(&[&self.running_dir, &proc_root])
        );
        let save_print = std::env::var("PIP_PRINT_ENTRIES").ok();
        // SAFETY: the script sets its own environment, as `os.environ[...] =` does
        unsafe { std::env::set_var("PIP_PRINT_ENTRIES", "1") };
        let mess = start_processchunks(
            &com_array,
            &self.pc_out_files[slot],
            &self.pc_check_files[slot].clone(),
            &mut self.param_array[slot],
        );
        if save_print
            .as_deref()
            .is_none_or(|value| value.is_empty() || value == "0")
        {
            // SAFETY: as above
            unsafe { std::env::set_var("PIP_PRINT_ENTRIES", "0") };
        }
        if let Some(mess) = mess.filter(|mess| !mess.is_empty()) {
            prnstr(&notify, "\n", true);
            prnstr(&log_mess, "\n", true);
            return (1, mess);
        }

        if std::env::set_current_dir(&self.starting_dir).is_err() {
            exit_error(&format!("Changing to directory {}", self.starting_dir));
        }

        self.slot_busy[slot] = true;
        self.stack_in_slot[slot] = self.stack_base.clone();
        if !self.project_root.is_empty() {
            let (line, num) =
                self.initialize_ebt_entry(&join(&[&self.running_dir, &self.stack_base]));
            self.ebt_line_for_slot[slot] = line;
            self.ebt_num_for_slot[slot] = num;
            notify = notify.replace("[SRW", &format!("{} [SRW", self.ebt_num_for_slot[slot]));
            log_mess = log_mess.replace("[SRW", &format!("{} [SRW", self.ebt_num_for_slot[slot]));
        }

        prnstr(&notify, "\n", true);
        if self.num_parallel > 1 {
            prnstr(&format!("  on machine slot {}", slot + 1), "\n", false);
        }
        prnstr(&log_mess, "\n", true);
        (0, String::new())
    }
}

/// The script's top level (`serieswatcher:757-1306`).  Returns the status of
/// its `sys.exit`; error paths exit the process as `exitError` does.
pub fn serieswatcher(arguments: &[OsString]) -> i32 {
    let prefix = format!("ERROR: {PROGNAME} - ");
    let argv: Vec<String> = arguments
        .iter()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect();
    let done = |status: i32| {
        let _ = std::io::stdout().flush();
        status
    };

    //
    // Setup runtime environment
    if std::env::var_os("IMOD_DIR").is_some() {
        add_imod_bin_ignore_sighup();
    } else {
        print!("{prefix} IMOD_DIR is not defined!\n");
        return done(1);
    }

    // Python raises KeyboardInterrupt on SIGINT
    // SAFETY: installs a handler that only stores to an atomic
    unsafe {
        libc::signal(
            libc::SIGINT,
            key_interrupt as extern "C" fn(libc::c_int) as libc::sighandler_t,
        );
    }

    // Fallbacks from ../manpages/autodoc2man 3 1 serieswatcher
    let options: Vec<String> = [
        "watch:WatchDirectory:FN:",
        "deliver:DeliverToDirectory:FN:",
        "remote:RemoteDirectory:FN:",
        "match:MatchPatternOrExt:CH:",
        "age:MinimumAgeOfStacks:F:",
        "opents:MinHoursIfOpenTSPresent:F:",
        "range:MinimumTiltRange:F:",
        "views:MinimumNumberOfViews:I:",
        "project:EtomoProjectRoot:FN:",
        "com:CommandFile:FN:",
        "adoc:DirectiveFile:FN:",
        "check:CheckFile:FN:",
        "dual:DualAxis:I:",
        "two:TwoSurfaces:I:",
        "cpus:CPUMachineList:CH:",
        "gpus:GPUMachineList:CH:",
        "parallel:ParallelRuns:I:",
        "etomo:EtomoOptions:I:",
        "DebugMode:debug:B:",
        "help:usage:B:",
    ]
    .iter()
    .map(|option| (*option).to_owned())
    .collect();

    let (_opts, _nonopts) = pip_read_or_parse_options(&argv, &options, PROGNAME, 1, 0, 0);

    // Constants and parameters
    let copy_prefix = "setupset.copyarg.";
    let setup_prefix = "setupset.";
    let scope_tmpl_text = format!("{setup_prefix}scopeTemplate");
    let user_tmpl_text = format!("{setup_prefix}userTemplate");
    let sys_tmpl_text = format!("{setup_prefix}systemTemplate");
    let mut w = Watcher {
        cur_ext_text: format!("{setup_prefix}currentStackExt"),
        cur_b_ext_text: format!("{setup_prefix}currentBStackExt"),
        dual_text: format!("{copy_prefix}dual"),
        montage_text: format!("{copy_prefix}montage"),
        two_surf_text: "comparam.align.tiltalign.SurfacesToAnalyze".to_owned(),
        recheck_erred_time: 300.,
        max_retries_if_no_open_ts: 5,
        max_size_in_existing_dir: 1000000,
        ending_step: -1.,
        ..Default::default()
    };
    let mut user_template_dir: Option<std::path::PathBuf> = None;
    let ots_there_matured_hours = 17.;

    let empty_val = "-12345";
    for key in [
        scope_tmpl_text.clone(),
        sys_tmpl_text.clone(),
        user_tmpl_text.clone(),
        w.cur_ext_text.clone(),
        w.cur_b_ext_text.clone(),
        w.dual_text.clone(),
        w.two_surf_text.clone(),
        w.montage_text.clone(),
    ] {
        w.line_num_dict.insert(key, (-1, String::new()));
    }
    let templ_file_error_names = ["scope template", "system template", "user template"];
    let month_name = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];

    // Get options
    w.watch_dir = pip_get_string("WatchDirectory", ".").unwrap_or_default();
    w.deliver_dir = pip_get_string("DeliverToDirectory", "").unwrap_or_default();
    w.pattern = pip_get_string("MatchPatternOrExt", ".mrc").unwrap_or_default();
    w.never_ots_matured_time = pip_get_float("MinimumAgeOfStacks", 300.).unwrap_or(300.);
    w.remote_start_dir = pip_get_string("RemoteDirectory", "").unwrap_or_default();
    w.min_tilt_range = pip_get_float("MinimumTiltRange", 40.).unwrap_or(40.);
    w.min_zsize = pip_get_integer("MinimumNumberOfViews", 12).unwrap_or(12);
    let ots_there_matured_hours = pip_get_float("MinHoursIfOpenTSPresent", ots_there_matured_hours)
        .unwrap_or(ots_there_matured_hours);
    w.ots_there_matured_time = ots_there_matured_hours * 3600.;
    w.debug_mode = pip_get_boolean("DebugMode", 0).unwrap_or(0) != 0;
    w.etomo_options = pip_get_integer("EtomoOptions", 0).unwrap_or(0);

    w.master_com = pip_get_string("CommandFile", "").unwrap_or_default();
    let mut batch_file = pip_get_string("DirectiveFile", "").unwrap_or_default();
    w.project_root = pip_get_string("EtomoProjectRoot", "").unwrap_or_default();
    let dual_axis_opt = pip_get_integer("DualAxis", 0).unwrap_or(0);
    let dual_entered = 1 - pip_get_err_no();
    let two_surfaces_opt = pip_get_integer("TwoSurfaces", 0).unwrap_or(0);
    let two_surf_entered = 1 - pip_get_err_no();

    // If there is a project root, get a new root and strip the ebt lines
    if !w.project_root.is_empty() {
        if !w.master_com.is_empty() || !batch_file.is_empty() {
            exit_error("You cannot enter a command or directive file with a project root name");
        }
        w.master_com = w.project_root.clone();

        // Figure out which adoc to use: prefer a dataset specific one unless the two
        // relevant things from there are entered
        let batch_list = glob_glob(&format!("{}_*.adoc", w.project_root));
        let dflt_file = format!("{}.adoc", w.project_root);
        if Path::new(&dflt_file).exists() && dual_entered != 0 && two_surf_entered != 0 {
            batch_file = dflt_file;
        } else if !batch_list.is_empty() {
            batch_file = batch_list[0].clone();
            if batch_list.len() > 1 {
                prnstr(
                    &format!(
                        "WARNING: There is more than one data set specific .adoc file, using {batch_file}"
                    ),
                    "\n",
                    true,
                );
            }
        } else if Path::new(&dflt_file).exists() {
            batch_file = dflt_file;
        } else {
            exit_error("There is no .adoc file with that project root name");
        }

        let ebt_lines = read_text_file(&format!("{}.ebt", w.project_root), None, false, None)
            .unwrap_or_default();
        let now_date = chrono::Local::now();
        use chrono::Datelike as _;
        w.out_proj_root = format!(
            "batch{}{}",
            month_name[now_date.month0() as usize],
            now_date.format("%d-%H%M%S")
        );
        prnstr(
            &format!("New project root name: {}     [SRW6]", w.out_proj_root),
            "\n",
            true,
        );

        // Clean out all row information from the ebt
        w.out_ebt_lines = vec![format!("meta.RootName={}", w.out_proj_root)];
        let mut max_id: i64 = 0;
        let skip_stripping = (w.etomo_options / 2) % 2 == 1;
        for line in &ebt_lines {
            if line.contains("meta.ref.ebt") && !line.contains("LastID") {
                if let Some(equals) = line.find('=').filter(|index| *index > 0) {
                    if let Some(id) = line.get(12..equals).and_then(py_int) {
                        max_id = max_id.max(id);
                    }
                }
            }
            if skip_stripping
                || !(line.contains("meta.row.ebt")
                    || line.contains("meta.ref.ebt")
                    || line.contains("meta.RootName"))
            {
                w.out_ebt_lines.push(line.clone());
            }
        }

        if skip_stripping || (w.etomo_options / 4) % 2 == 1 {
            w.num_ebt_rows = max_id as i32;
        }
    }

    w.dual_axis = 0;
    let mut two_surfaces = 0;
    let mut num_parallel = pip_get_integer("ParallelRuns", 1).unwrap_or(1) as i64;

    if (!w.master_com.is_empty() && batch_file.is_empty())
        || (!batch_file.is_empty() && w.master_com.is_empty())
    {
        exit_error("You must enter both a command file and a batch directive file");
    }
    if w.master_com.is_empty() && w.deliver_dir.is_empty() {
        exit_error("You must enter a directory to deliver to when not reconstructing");
    }

    w.starting_dir = std::env::current_dir()
        .map(|dir| dir.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut master_lines: Vec<String> = Vec::new();
    if !w.master_com.is_empty() {
        let (master_com, _root) = complete_and_check_com_file(&w.master_com);
        w.master_com = master_com;
        master_lines = read_text_file(&w.master_com, None, false, None).unwrap_or_default();
        w.batch_lines = read_text_file(&batch_file, None, false, None).unwrap_or_default();

        // Set up check file
        let top_check_file = pip_get_string("CheckFile", "").unwrap_or_default();
        if !top_check_file.is_empty() {
            w.top_check_file = imod_abs_path(&top_check_file);
        } else {
            w.top_check_file = imod_abs_path(&join(&[
                &w.starting_dir,
                &format!("{PROGNAME}.{}.input", std::process::id()),
            ]));
        }
        prnstr(
            &format!(
                "To quit all processing, place a Q in the file: {}",
                w.top_check_file
            ),
            "\n",
            true,
        );
        if Path::new(&w.top_check_file).exists() {
            cleanup_files(&[w.top_check_file.clone()]);
        }

        // Find lines in the directive file for things we are looking for
        for ind in 0..w.batch_lines.len() {
            let line = w.batch_lines[ind].trim_start();
            let lsplit: Vec<&str> = line.split('=').collect();
            if lsplit.len() < 2 {
                continue;
            }
            let line_direc = lsplit[0].trim();
            if w.line_num_dict.contains_key(line_direc) {
                w.line_num_dict.insert(
                    line_direc.to_owned(),
                    (ind as i64, lsplit[1].trim().to_owned()),
                );
            }
        }

        // Read in template files after getting their absolute paths
        let mut all_lines: Vec<Vec<String>> =
            vec![Vec::new(), Vec::new(), Vec::new(), w.batch_lines.clone()];
        for (tmpl_text, ind) in [
            (&scope_tmpl_text, 0usize),
            (&sys_tmpl_text, 1),
            (&user_tmpl_text, 2),
        ] {
            let (line_ind, value) = w.line_num_dict[tmpl_text.as_str()].clone();
            if !value.is_empty() {
                let (abs_tmpl_name, err, user_dir, err_mess) = abs_template_path(
                    &value,
                    ind as i32,
                    user_template_dir.take(),
                    templ_file_error_names[ind],
                );
                user_template_dir = user_dir;
                if err < 0 {
                    exit_error(&err_mess);
                }
                let abs_tmpl_name = abs_tmpl_name
                    .map(|path| path.to_string_lossy().into_owned())
                    .unwrap_or_default();
                all_lines[ind] = read_text_file(
                    &abs_tmpl_name,
                    Some(templ_file_error_names[ind]),
                    false,
                    None,
                )
                .unwrap_or_default();
                w.batch_lines[line_ind as usize] = format!("{tmpl_text} = {abs_tmpl_name}");
            }
        }

        // Now evaluate a few directives through the heirarchy
        for ind in 0..4 {
            let dual_value = option_value(
                &all_lines[ind],
                &w.dual_text,
                BOOL_VALUE,
                false,
                0,
                Some('='),
                None,
            );
            if let Some(OptionValue::Boolean(value)) = dual_value {
                w.dual_axis = 0;
                if value {
                    w.dual_axis = 1;
                }
            }
            let surf_value = match option_value(
                &all_lines[ind],
                &w.two_surf_text,
                INT_VALUE,
                false,
                1,
                Some('='),
                Some(empty_val),
            ) {
                Some(OptionValue::Integers(values)) => values[0] as i64,
                Some(OptionValue::String(_)) => -12345,
                _ => 0,
            };
            if surf_value != 0 {
                two_surfaces = 0;
                if surf_value >= 2 {
                    two_surfaces = 1;
                }
            }
        }
    }

    // Finally override with option entries
    if dual_entered != 0 {
        w.dual_axis = dual_axis_opt;
    }
    if two_surf_entered != 0 {
        two_surfaces = two_surfaces_opt;
    }
    w.num_surfaces = 1;
    if two_surfaces != 0 {
        w.num_surfaces = 2;
    }

    // Modify the pattern with a * if it has no wild cards
    if !w.pattern.contains('*')
        && !w.pattern.contains('?')
        && !(w.pattern.contains('[') && w.pattern.contains(']'))
    {
        if !w.pattern.contains('.') {
            w.pattern = format!(".{}", w.pattern);
        }
        w.pattern = format!("*{}", w.pattern);
    }

    // If pattern is b*.ext, set up to check for A stack and wait for B
    if w.dual_axis != 0 && !w.master_com.is_empty() {
        if let Some(dot_ind) = w.pattern.rfind('.').filter(|index| *index > 0) {
            if w.pattern.as_bytes()[dot_ind - 1] == b'b' {
                w.check_for_a_wait_for_b = true;
                // `'[ab].'.join(pattern.rsplit('b.', 1))`
                if let Some(split) = w.pattern.rfind("b.") {
                    w.pattern = format!("{}[ab].{}", &w.pattern[..split], &w.pattern[split + 2..]);
                }
                if w.debug_mode {
                    prnstr(
                        &format!("Set checkForAWaitForB and changed pattern to {}", w.pattern),
                        "\n",
                        false,
                    );
                }
            }
        }
    }

    // Set up for making a command file and running
    if !w.master_com.is_empty() {
        w.reduced_lines =
            vec!["# Batchruntomo file for new project created by Serieswatcher".to_owned()];
        let reduce_list = [
            "RootName",
            "CurrentLocation",
            "DeliverToDirectory",
            "DirectiveFile",
            "CheckFile",
            "RemoteDirectory",
            "MakeSubDirectory",
            "CPUMachineList",
            "GPUMachineList",
            "ProcessOneAxis",
            "SingleOnFirstCPU",
            "ParallelBatchRootName",
        ];

        // Get needed options from com file, and then get option entries to override the
        // machine lists
        let string_value = |option: &str| -> String {
            match option_value(&master_lines, option, STRING_VALUE, false, 0, None, None) {
                Some(OptionValue::String(value)) => value,
                _ => String::new(),
            }
        };
        w.remote_com_dir = string_value("RemoteDirectory");
        let deliver_dir_com = string_value("DeliverToDirectory");
        let cpu_list_in = string_value("CPUMachineList");
        let gpu_list_in = string_value("GPUMachineList");

        let cpu_list_in = pip_get_string("CPUMachineList", &cpu_list_in).unwrap_or_default();
        w.gpu_list_in = pip_get_string("GPUMachineList", &gpu_list_in).unwrap_or_default();
        w.ending_step = match option_value(
            &master_lines,
            "EndingStep",
            FLOAT_VALUE,
            false,
            1,
            None,
            None,
        ) {
            Some(OptionValue::Floats(values)) => values[0],
            _ => -1.,
        };

        if cpu_list_in.is_empty() {
            exit_error("You must enter a CPU machine list; there is none in the command file");
        }
        if w.deliver_dir.is_empty() {
            w.deliver_dir = deliver_dir_com;
        }

        // Get rid of all the possible multiple entries
        for line in &master_lines {
            if !reduce_list.iter().any(|opt| line.contains(opt)) {
                w.reduced_lines.push(line.clone());
            }
        }

        // Get translation options from remote entry
        let (com_dir_root, remote_root) =
            get_translation_from_remote_dir(&master_lines, &w.reduced_lines, &w.master_com);
        if !com_dir_root.is_empty() {
            w.reduced_lines
                .push(format!("TranslatePathsFrom {com_dir_root}"));
            w.reduced_lines
                .push(format!("TranslatePathsTo {remote_root}"));
        }

        // Add back the current resources to a copy and write the new project com file
        if !w.project_root.is_empty() {
            let mut new_proj_com_lines = w.reduced_lines.clone();
            new_proj_com_lines.push(format!("CPUMachineList {cpu_list_in}"));
            if !w.gpu_list_in.is_empty() {
                new_proj_com_lines.push(format!("GPUMachineList {}", w.gpu_list_in));
            }
            let _ = write_text_file(
                &format!("{}.{}", w.out_proj_root, default_com_extension()),
                &new_proj_com_lines,
                false,
            );
        }

        // Set up machine lists for parallel situation
        num_parallel = num_parallel.max(1);
        let mut total_cpu: i64 = 0;
        let mut cpu_machines: Vec<String> = Vec::new();
        let mut cpu_cores: Vec<i64> = Vec::new();
        if num_parallel > 1 {
            // Parse the CPU list and get arrays of machines and number of cores
            for mach in cpu_list_in.split(',') {
                let msplit: Vec<&str> = mach.split(':').collect();
                let mut num = 1;
                if msplit.len() > 1 {
                    num = convert_to_integer(msplit[1], "number of cores in CPUMachineList") as i64;
                }
                cpu_machines.push(msplit[0].to_owned());
                cpu_cores.push(num);
                total_cpu += num;
            }

            if total_cpu == 1 {
                prnstr(
                    "WARNING: Only one core is provided in CPUMachineList so runs will not be in parallel",
                    "\n",
                    true,
                );
                num_parallel = 1;
            }
            if total_cpu < num_parallel {
                prnstr(
                    &fmtstr(
                        "WARNING: Only {} cores are provided in CPUMachineList so only {} runs will be done in parallel",
                        &[total_cpu.to_string(), total_cpu.to_string()],
                    ),
                    "\n",
                    true,
                );
                num_parallel = total_cpu;
            }
        }

        // For parallel runs, make lists of resources so they can be divided up
        if num_parallel > 1 {
            w.cpu_lists = Vec::new();
            w.cpu_for_brt = Vec::new();

            // Initialize to make CPU lists
            let num_cpu_per_run = total_cpu.div_euclid(num_parallel);
            let extra_cpu = total_cpu.rem_euclid(num_parallel);
            let mut cur_machine = 0usize;
            let mut cur_core = 0i64;

            // Loop on runs and get number of CPUs for a run
            for ind in 0..num_parallel {
                let mut one_list = String::new();
                let mut num_cpu = num_cpu_per_run;
                if ind < extra_cpu {
                    num_cpu += 1;
                }

                // Add machines and CPUs sequentially
                let mut num_added = 0;
                w.cpu_for_brt.push(cpu_machines[cur_machine].clone());
                while num_added < num_cpu {
                    if !one_list.is_empty() {
                        one_list.push(',');
                    }
                    one_list += &cpu_machines[cur_machine];
                    let num = (num_cpu - num_added).min(cpu_cores[cur_machine] - cur_core);
                    one_list += &format!(":{num}");
                    num_added += num;
                    cur_core += num;
                    if cur_core >= cpu_cores[cur_machine] {
                        cur_core = 0;
                        cur_machine += 1;
                    }
                }

                w.cpu_lists.push(one_list);
            }
        } else {
            w.cpu_lists = vec![cpu_list_in.clone()];
            w.cpu_for_brt = vec!["localhost".to_owned()];
        }

        // Set up the directory to place com's and cd to for running
        w.running_dir = w.watch_dir.clone();
        if !w.deliver_dir.is_empty() {
            w.running_dir = w.deliver_dir.clone();
        }
        let mut current_com_dir = dirname(&w.master_com);
        if current_com_dir.is_empty() {
            current_com_dir = ".".to_owned();
        }
        w.abs_running_dir = imod_abs_path(&w.running_dir);

        // Transfer the remote directory to the running direction, either from the
        // starting dir for option entry, or from the com file's location for an entry
        // in the com file
        if !w.remote_start_dir.is_empty() || !w.remote_com_dir.is_empty() {
            let remote = if !w.remote_start_dir.is_empty() {
                w.remote_start_dir.clone()
            } else {
                w.remote_com_dir.clone()
            };
            let (remote_run_dir, err_mess) =
                transfer_remote_directory(&remote, &current_com_dir, &w.running_dir);
            w.remote_run_dir = remote_run_dir;
            if w.remote_run_dir.is_empty() {
                exit_error(&err_mess);
            }
        }

        let n = num_parallel as usize;
        w.num_parallel = n;
        w.slot_busy = vec![false; n];
        w.stack_in_slot = vec![String::new(); n];
        w.log_for_slot = vec![String::new(); n];
        w.ebt_line_for_slot = vec![-1; n];
        w.ebt_num_for_slot = vec![String::new(); n];

        // Note that [[0] * 16] * numParallel creates SHALLOW copies of the 16 0's!
        for ind in 0..n {
            w.pc_check_files.push(format!(
                "{}/watcherbatch{ind}.{}.input",
                w.abs_running_dir,
                std::process::id()
            ));
            w.pc_out_files.push(format!(
                "{}/prochunks{ind}.{}.out",
                w.abs_running_dir,
                std::process::id()
            ));
            w.param_array.push(ProChunksParam::default());
        }
    } else {
        w.num_parallel = num_parallel.max(0) as usize;
    }

    // Initialize the watching
    let sleep_time = 2.;
    // The module global `ind` the loop over slots leaves behind
    let last_slot = w.num_parallel.saturating_sub(1);

    // START WATCHING

    loop {
        let mut interrupted = KEY_INTERRUPT.swap(false, Ordering::SeqCst);
        if !interrupted {
            let start_time = now();

            // Check the stacks then check the run slots
            let (_num_ready, ready_stack_ind) = w.check_for_stacks();
            let mut all_ready = true;
            if !w.master_com.is_empty() {
                let mut top_quit = String::new();
                w.first_ready_slot = -1;
                for ind in 0..w.num_parallel {
                    if !w.slot_busy[ind] {
                        if w.first_ready_slot < 0 {
                            w.first_ready_slot = ind as i64;
                        }
                        continue;
                    }

                    all_ready = false;
                    let top_check_file = w.top_check_file.clone();
                    let (error, finished, quit, _num_done, message) = check_pro_chunks_log(
                        Some(&top_check_file),
                        &mut w.param_array[ind],
                        false,
                        false,
                        false,
                    );
                    top_quit = quit;
                    if error != 0 || finished != 0 {
                        // A run is done, process the result and free the slot
                        if w.first_ready_slot < 0 {
                            w.first_ready_slot = ind as i64;
                        }
                        if error != 0 {
                            let message = format!(
                                "Processchunks error on {}: {message}",
                                w.stack_in_slot[ind]
                            );
                            prnstr(&message, "\n", true);
                        } else {
                            cleanup_files(&[w.pc_out_files[ind].clone()]);
                            let mut mess =
                                format!("Finished processing stack {}", w.stack_in_slot[ind]);
                            let (stack_root, _ext) = os_path_splitext(&w.stack_in_slot[ind]);
                            w.stack_root = stack_root.clone();
                            w.dataset_dir = stack_root;
                            if w.dual_axis != 0 {
                                w.dataset_dir.pop();
                                if w.run_one_axis != 1 {
                                    mess =
                                        format!("Finished processing data set: {}", w.dataset_dir);
                                }
                            }

                            let log_lines =
                                read_text_file(&w.log_for_slot[ind].clone(), None, true, None);
                            match &log_lines {
                                Err(_) => {
                                    mess +=
                                        " with unknown result - there was an error reading the log"
                                }
                                Ok(lines) => {
                                    // `logLines[max(-len(logLines) + 1, -5):]`
                                    let n = lines.len() as i64;
                                    let start = (-n + 1).max(-5);
                                    let start = if start < 0 { n + start } else { start };
                                    let start = start.clamp(0, n) as usize;
                                    for line in &lines[start..] {
                                        if line.contains("failures occurred for") {
                                            mess += " with a processing error";
                                        }
                                        if line.contains("no failures occurred") {
                                            mess += " with successful completion";
                                        }
                                    }
                                }
                            }

                            mess += "    [SRW3]";
                            if !w.project_root.is_empty() && w.ebt_line_for_slot[ind] > 0 {
                                mess = mess
                                    .replace("[SRW", &format!("{} [SRW", w.ebt_num_for_slot[ind]));
                                let stack = w.stack_in_slot[ind].clone();
                                let line = w.ebt_line_for_slot[ind];
                                w.parse_brt_log_update_ebt(&stack, &log_lines, line);
                            }
                            if !w.killing {
                                prnstr(&mess, "\n", true);
                            }
                        }

                        w.slot_busy[ind] = false;
                        w.stack_in_slot[ind] = String::new();
                        w.ebt_line_for_slot[ind] = -1;
                        w.ebt_num_for_slot[ind] = String::new();
                    } else {
                        w.param_array[ind].elapsed += sleep_time;
                    }
                }

                if !top_quit.is_empty() {
                    w.process_check_action(&top_quit);
                }
            }

            // Fixed in translation (BUGS.md): without a command file there is
            // no check file, and native dies here on `topCheckFile`
            if all_ready && !w.master_com.is_empty() {
                let action = check_for_pro_chunks_quit(
                    Some(&w.top_check_file),
                    Some(&w.pc_check_files[last_slot]),
                    false,
                    false,
                );
                w.process_check_action(&action);
            }

            // IF pausing and all slots are ready, exit
            if (w.pausing || w.killing) && all_ready {
                w.rename_ebt_at_end();
                if w.killing {
                    prnstr("All running sets killed    [SRW5]", "\n", true);
                } else {
                    prnstr("All running sets finished    [SRW4]", "\n", true);
                }
                return done(0);
            }

            // IF there is a stack ready and either not running, or a free run slot,
            // process it
            if ready_stack_ind >= 0
                && (w.master_com.is_empty() || w.first_ready_slot >= 0)
                && !w.pausing
                && !w.killing
            {
                let ready = ready_stack_ind as usize;
                w.ready_stack_ind = ready;
                w.stack = w.waiting_list[ready].stack.clone();
                w.stack_base = basename(&w.stack);
                let (stack_root, stack_ext) = os_path_splitext(&w.stack_base);
                w.stack_root = stack_root;
                w.stack_ext = stack_ext;
                w.set_root = w.stack_root.clone();
                if w.dual_axis != 0 {
                    w.set_root.pop();
                }

                // Deliver it if necessary and add it to either error or done list and
                // remove from waiting list.  Whatever happens now, we don't want to see
                // this stack again for most errors, but skipIt = 2 means retry
                let skip_it = w.deliver_stack_and_mdoc();
                if skip_it < 0 {
                    if w.debug_mode {
                        prnstr("Setting B axis status to -1", "\n", false);
                    }
                    w.waiting_list[ready].ab_status = -1;
                } else if skip_it < 2 {
                    if skip_it != 0 {
                        w.erred_list.push(w.stack.clone());
                    } else {
                        w.done_list.push(w.stack.clone());
                    }
                    w.waiting_list.remove(ready);
                }

                if skip_it == 0 && !w.master_com.is_empty() {
                    let (err, mess) = w.run_brt_on_stack();
                    if err == 2 {
                        // Fixed in translation (BUGS.md): native prints imodpy's
                        // import-time `errStrings`, not the lines it just fetched
                        let err_lines = get_err_strings();
                        for line in err_lines {
                            prnstr(&line, "\n", true);
                        }
                    }
                    if err != 0 {
                        prnstr(
                            &format!("{mess}; skipping stack {}", w.stack_base),
                            "\n",
                            true,
                        );
                    } else if w.run_one_axis == 1 {
                        // If doing A axis, look for B and clear its error from being
                        // there first
                        let b_ind = w
                            .find_stack_in_waiting_list(&format!("{}b{}", w.set_root, w.stack_ext));
                        if b_ind >= 0 {
                            if w.debug_mode && w.waiting_list[b_ind as usize].ab_status != 0 {
                                prnstr("Clearing wait status for B axis stack", "\n", false);
                            }
                            w.waiting_list[b_ind as usize].ab_status = 0;
                        }
                    }
                }
            }

            // Check for quit or finish
            if !w.master_com.is_empty() && !w.pausing && !w.killing {
                for ind in 0..w.num_parallel {
                    if w.slot_busy[ind] {
                        let action = check_for_pro_chunks_quit(
                            Some(&w.top_check_file),
                            Some(&w.pc_check_files[ind]),
                            false,
                            false,
                        );
                        w.process_check_action(&action);
                    }
                }
            }

            // Sleep, subtracting off any lost time if possible
            let sleep_end = now() + 0.1f64.max(sleep_time - (now() - start_time));
            while now() < sleep_end {
                if KEY_INTERRUPT.load(Ordering::SeqCst) {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_secs_f64(
                    0.05f64.min((sleep_end - now()).max(0.)),
                ));
            }
            interrupted = KEY_INTERRUPT.swap(false, Ordering::SeqCst);

            // Adjust elapsed times now with true interval
            if !interrupted && !w.master_com.is_empty() {
                let delta = now() - start_time;
                for ind in 0..w.num_parallel {
                    if w.slot_busy[ind] {
                        w.param_array[ind].elapsed += delta;
                    }
                }
            }
        }

        if interrupted {
            // `except KeyboardInterrupt:`
            if w.master_com.is_empty() || !w.slot_busy.contains(&true) {
                w.process_check_action("Q");
                return done(0);
            }
            print!("Enter Q to quit all processing or F to just exit {PROGNAME}: ");
            let _ = std::io::stdout().flush();
            let mut action = String::new();
            let _ = std::io::stdin().read_line(&mut action);
            let action = action.trim_end_matches(['\n', '\r']).to_owned();

            if action == "Q" {
                let _ = write_text_file(&w.top_check_file, &["Q".to_owned()], true);
                for ind in 0..w.num_parallel {
                    if w.slot_busy[ind] {
                        check_for_pro_chunks_quit(
                            Some(&w.top_check_file),
                            Some(&w.pc_check_files[ind]),
                            false,
                            false,
                        );
                    }
                }
            }

            w.process_check_action(&action);
        }
    }
}
