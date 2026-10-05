//! `IMOD/Etomo/src/etomo/process/ProcesschunksBatchRunTomoMonitor.java`.
//!
//! A `ProcesschunksProcessMonitor` subclass for a parallel batchruntomo run; see the
//! module comment of `processchunks_process_monitor.rs` for how subclasses are
//! represented.  Each chunk (one dataset) gets a `BatchRunTomoChunkMonitor` that
//! follows that chunk's log.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use super::batch_run_tomo_chunk_monitor;
use super::batch_run_tomo_process_monitor::{self, BatchRunTomoProcessMonitor};
use super::monitor::Monitor;
use super::process_data::ProcessData;
use super::process_interface::SystemProcessInterface;
use super::process_messages::{MessagesArray, ProcessMessages};
use super::process_output_strings;
use super::processchunks_process_monitor::{
    ParallelProgressDisplayRef, ProcesschunksProcessMonitor, ProcesschunksProcessMonitorImpl,
};
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::batch_run_tomo_manager::BatchRunTomoManager;
use crate::imod::etomo::comscript::processchunks_param;
use crate::imod::etomo::storage::log_file::{Handle, LogFile, LogFileError, WriterId};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::extension::Extension;
use crate::imod::etomo::r#type::process_end_state::ProcessEndState;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::r#type::run_list::RunList;
use crate::imod::etomo::r#type::run_type::RunType;
use crate::imod::etomo::util::dataset_files;
use crate::imod::etomo::util::event_queue;

/// Java private static final `RUNNING_COMSCRIPT_NAME_INDEX`.
const RUNNING_COMSCRIPT_NAME_INDEX: usize = 1;

/// `ProcesschunksBatchRunTomoMonitor`: the parent with this subclass.
pub type ProcesschunksBatchRunTomoMonitorRef =
    Arc<ProcesschunksProcessMonitor<ProcesschunksBatchRunTomoMonitor>>;

/// Java `ProcesschunksBatchRunTomoMonitor`'s own fields.
pub struct ProcesschunksBatchRunTomoMonitor {
    /// Java private volatile `running`.  Overrides the parent running boolean so that
    /// this monitor has time to finish its run function.
    running: AtomicBool,
    /// Java private final `manager`.
    manager: &'static BatchRunTomoManager,
    /// Java private final `messagesArray`.
    messages_array: MessagesArray,
    /// Java private final `runList`.
    run_list: Option<Arc<RunList>>,
    /// Java private final `chunkMonitorList`.
    chunk_monitor_list: Option<Mutex<Vec<Option<Arc<BatchRunTomoProcessMonitor>>>>>,
    /// Java private final `processData`.
    process_data: Option<Arc<Mutex<ProcessData>>>,
    /// Java private final `processingMethod`.
    processing_method: Option<ProcessingMethod>,
    /// Java private final `secondaryQueue`.
    secondary_queue: Option<String>,
    /// Java private `useBrtCommandsPipe`, initially true.
    use_brt_commands_pipe: AtomicBool,
    /// Java private `brtCommandsPipe`, initially null.
    brt_commands_pipe: Mutex<Option<Arc<Handle>>>,
    /// Java private `brtCommandsPipeWriterId`, initially null.
    brt_commands_pipe_writer_id: Mutex<Option<WriterId>>,
    /// Java `synchronized` on `halt`.
    synchronized: Mutex<()>,
}

/// Java private constructor `ProcesschunksBatchRunTomoMonitor(BatchRunTomoManager,
/// AxisID, String, Map, String, RunType, RunList, ProcessData, List<ProcessMessages>,
/// boolean, boolean, ProcessingMethod)`.
#[allow(clippy::too_many_arguments)]
fn new(
    manager: &'static BatchRunTomoManager,
    axis_id: AxisID,
    root_name: Option<&str>,
    computer_map: Option<BTreeMap<String, String>>,
    secondary_queue: Option<&str>,
    run_type: Option<RunType>,
    run_list: Option<Arc<RunList>>,
    process_data: Option<Arc<Mutex<ProcessData>>>,
    messages_array: MessagesArray,
    multi_line_messages: bool,
    reconnect: bool,
    processing_method: Option<ProcessingMethod>,
) -> ProcesschunksBatchRunTomoMonitorRef {
    let run_list = match run_list {
        Some(run_list) => Some(run_list),
        None => manager.create_run_list(run_type),
    };
    let chunk_monitor_list = run_list.as_ref().map(|run_list| {
        // RunList contains all the row involved in the original run - before
        // kills/pauses/exits/reconnects/resumes. Only actively running rows with get a
        // chunk monitor.
        let size = run_list.size().max(0) as usize;
        Mutex::new(vec![None; size])
    });
    let base_manager: &'static dyn BaseManager = manager;
    let instance = ProcesschunksProcessMonitor::new_subclass(
        base_manager,
        axis_id,
        root_name,
        computer_map,
        multi_line_messages,
        ProcesschunksBatchRunTomoMonitor {
            running: AtomicBool::new(false),
            manager,
            messages_array,
            run_list,
            chunk_monitor_list,
            process_data,
            processing_method,
            secondary_queue: secondary_queue.map(str::to_owned),
            use_brt_commands_pipe: AtomicBool::new(true),
            brt_commands_pipe: Mutex::new(None),
            brt_commands_pipe_writer_id: Mutex::new(None),
            synchronized: Mutex::new(()),
        },
    );
    instance.erased().set_reconnect(reconnect);
    instance
}

/// Java package-private static `getInstance(...)`.
#[allow(clippy::too_many_arguments)]
pub fn get_instance(
    manager: &'static BatchRunTomoManager,
    axis_id: AxisID,
    root_name: Option<&str>,
    computer_map: Option<BTreeMap<String, String>>,
    secondary_queue: Option<&str>,
    run_type: Option<RunType>,
    run_list: Option<Arc<RunList>>,
    process_data: Option<Arc<Mutex<ProcessData>>>,
    messages_array: MessagesArray,
    multi_line_messages: bool,
    processing_method: Option<ProcessingMethod>,
) -> ProcesschunksBatchRunTomoMonitorRef {
    new(
        manager,
        axis_id,
        root_name,
        computer_map,
        secondary_queue,
        run_type,
        run_list,
        process_data,
        messages_array,
        multi_line_messages,
        false,
        processing_method,
    )
}

/// Java package-private static `getReconnectInstance(BatchRunTomoManager, AxisID,
/// ProcessData, List<ProcessMessages>, boolean)`.
pub fn get_reconnect_instance(
    manager: &'static BatchRunTomoManager,
    axis_id: AxisID,
    process_data: Arc<Mutex<ProcessData>>,
    messages_array: MessagesArray,
    multi_line_messages: bool,
) -> ProcesschunksBatchRunTomoMonitorRef {
    let (sub_process_name, computer_map, secondary_queue, processing_method) = {
        let data = process_data.lock().unwrap();
        (
            data.get_sub_process_name(),
            data.get_computer_map().cloned(),
            data.get_secondary_queue(),
            data.get_processing_method(),
        )
    };
    new(
        manager,
        axis_id,
        sub_process_name.as_deref(),
        computer_map,
        secondary_queue.as_deref(),
        Some(RunType::Reconnect),
        None,
        Some(process_data),
        messages_array,
        multi_line_messages,
        true,
        processing_method,
    )
}

impl ProcesschunksBatchRunTomoMonitor {
    /// Java package-private `writeBrtCommand(String)`.  Write to the brt command pipe
    /// one time.  All the brt chunks use the same commend pipe (a different pipe from a
    /// brt processchunks run).  To avoid any timing problems, write even if no brt
    /// monitors have been created yet.
    pub fn write_brt_command(
        &self,
        this: &ProcesschunksProcessMonitor,
        command: &str,
    ) -> Result<(), LogFileError> {
        if !self.use_brt_commands_pipe.load(Ordering::SeqCst) {
            return Ok(());
        }
        let pipe = {
            let mut pipe = self.brt_commands_pipe.lock().unwrap();
            if pipe.is_none() {
                // Get the batchruntomo (standard) check file.
                *pipe = Some(LogFile::get_instance_user_dir(
                    &self.manager.get_property_user_dir().unwrap_or_default(),
                    &dataset_files::get_commands_file_name(
                        this.get_subdir_name().as_deref(),
                        &this.get_root_name().unwrap_or_else(|| "null".to_owned()),
                        None,
                    ),
                    Some(self.manager.get_emergency_monitor(Some(this.axis_id))),
                )?);
            }
            pipe.clone().unwrap()
        };
        let open = self
            .brt_commands_pipe_writer_id
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|id| !id.is_empty());
        if !open {
            match pipe.open_writer_append(true) {
                Ok(id) => *self.brt_commands_pipe_writer_id.lock().unwrap() = Some(id),
                Err(LogFileError::Lock(e)) => {
                    this.handle_lock_exception(Some(&e), true);
                    if !this.is_running() {
                        return Ok(());
                    }
                }
                Err(e) => return Err(e),
            }
        }
        let writer_id = self.brt_commands_pipe_writer_id.lock().unwrap().clone();
        let Some(writer_id) = writer_id.filter(|id| !id.is_empty()) else {
            return Ok(());
        };
        pipe.write(Some(command), &writer_id)?;
        pipe.new_line(&writer_id)?;
        pipe.flush(&writer_id)?;
        // Close writer after each write. If it is kept open, the file would not be
        // writeable from the command line in Windows.
        pipe.close_id(Some(&*writer_id));
        *self.brt_commands_pipe_writer_id.lock().unwrap() = None;
        Ok(())
    }

    /// Java private `getChunkIndex(String)`.  Returns an index based on the chunk file
    /// name ("-ddd" minus 1).  Index will be >=0.  Returns null if can't find a valid
    /// index.
    fn get_chunk_index(&self, comscript_name: &str) -> Option<i32> {
        if !Extension::is_comscript(comscript_name) {
            return None;
        }
        let chunk_number = Extension::get_chunk_number(Some(comscript_name))?;
        if chunk_number.is_null() || !chunk_number.is_valid() {
            return None;
        }
        let n_chunk = chunk_number.get_int();
        Some(n_chunk - 1)
    }

    /// Every chunk monitor that exists.
    fn chunk_monitors(&self) -> Vec<Arc<BatchRunTomoProcessMonitor>> {
        self.chunk_monitor_list
            .as_ref()
            .map(|list| list.lock().unwrap().iter().flatten().cloned().collect())
            .unwrap_or_default()
    }
}

impl ProcesschunksProcessMonitorImpl for ProcesschunksBatchRunTomoMonitor {
    /// Java final `run()`.
    fn run(&self, this: &ProcesschunksProcessMonitor) {
        self.running.store(true, Ordering::SeqCst);
        // try
        if !this.is_reconnect() {
            let manager = self.manager;
            event_queue::invoke_later(move || {
                manager.log_message(Some(batch_run_tomo_process_monitor::STARTING_MESSAGE))
            });
        }
        // Logging much go to the dataset logs so don't send anything to the primary.
        self.manager.set_allow_primary_logging(false);
        this.run_super();
        self.manager.set_allow_primary_logging(true);
        // finally
        self.running.store(false, Ordering::SeqCst);
    }

    /// Java `isRunning()`.
    fn is_running(&self, _this: &ProcesschunksProcessMonitor) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Java `updateParallelProgressDisplay(ParallelProgressDisplay)`.
    fn update_parallel_progress_display(
        &self,
        this: &ProcesschunksProcessMonitor,
        display: Option<&ParallelProgressDisplayRef>,
    ) {
        this.update_parallel_progress_display_super(display);
        if let Some(display) = display {
            let display = display.clone();
            let secondary_queue = self.secondary_queue.clone();
            event_queue::invoke_later(move || {
                display
                    .get()
                    .set_secondary_queue(secondary_queue.as_deref())
            });
        }
    }

    /// Java `getCheckFile()`.  When processchunks runs batchruntomo they must have
    /// different check files.
    fn get_check_file(&self, this: &ProcesschunksProcessMonitor) -> String {
        dataset_files::get_commands_file_name(
            this.get_subdir_name().as_deref(),
            &this.get_root_name().unwrap_or_else(|| "null".to_owned()),
            Some(processchunks_param::BRT_CHECK_NAME_SUFFIX),
        )
    }

    /// Java `writeKillCommand()`.  For a kill command, pause processchunks.  If its
    /// killed then batchruntomo can't end correctly.  Send a separate command to
    /// batchruntomo to kill it.
    ///
    /// This is not necessary for pause.  For pause, just pause processchunks in the
    /// usual way.  This allows the current brt dataset to complete.
    fn write_kill_command(&self, this: &ProcesschunksProcessMonitor) -> Result<(), LogFileError> {
        this.write_command("P")?;
        // All brt commands share the same .cmds file, and
        self.write_brt_command(this, "Q")?;
        for chunk_monitor in self.chunk_monitors() {
            chunk_monitor.erased().set_killing(true);
        }
        Ok(())
    }

    /// Java `setUseCommandsPipe(boolean)`.
    fn set_use_commands_pipe(&self, this: &ProcesschunksProcessMonitor, use_: bool) {
        self.use_brt_commands_pipe.store(use_, Ordering::SeqCst);
        this.set_use_commands_pipe_super(use_);
    }

    /// Java `handlePauseMessage()`.
    fn handle_pause_message(&self, this: &ProcesschunksProcessMonitor) {
        if this.is_killing() {
            this.end_monitor(ProcessEndState::Killed);
        } else {
            this.handle_pause_message_super();
        }
    }

    /// Java synchronized `halt()`.
    fn halt(&self, this: &ProcesschunksProcessMonitor) {
        let _synchronized = self.synchronized.lock().unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1000));
        // `super.halt()` is empty.
        this.set_halt(true);
        eprintln!("Halting chunk monitors");
        for chunk_monitor in self.chunk_monitors() {
            chunk_monitor.erased().halt();
        }
    }

    /// Java final `pause(SystemProcessInterface, AxisID)`.
    fn pause(
        &self,
        this: &ProcesschunksProcessMonitor,
        process: &dyn SystemProcessInterface,
        axis_id: AxisID,
    ) -> bool {
        let mut pause_succeeded = this.pause_super(process, axis_id);
        for chunk_monitor in self.chunk_monitors() {
            pause_succeeded =
                batch_run_tomo_chunk_monitor::pause(chunk_monitor.erased()) || pause_succeeded;
        }
        pause_succeeded
    }

    /// Java `setProcess(SystemProcessInterface)`.
    fn set_process(
        &self,
        this: &ProcesschunksProcessMonitor,
        process: Option<Arc<dyn SystemProcessInterface>>,
    ) {
        this.set_process_super(process.clone());
        if !this.is_reconnect()
            && let Some(process) = process
        {
            process.set_secondary_queue(self.secondary_queue.as_deref());
        }
    }

    /// Java `processMessage(String)`.  Handle comscript starting and failing messages
    /// from processchunks.out.  Runs a chunk monitor associated with the comscript name
    /// each time first starting message is received.  The fail message causes the
    /// monitor to be stopped and reset.
    fn process_message(&self, this: &ProcesschunksProcessMonitor, line: &str) {
        // Look for comscript started message.
        if line.contains(process_output_strings::PROCESSCHUNK_RUNNING_COMSCRIPT_MSG_ID) {
            // Java `line.split("\\s+")`: a leading empty field when the line starts
            // with whitespace.
            let mut string_array: Vec<&str> = line.split_whitespace().collect();
            if line.starts_with(char::is_whitespace) {
                string_array.insert(0, "");
            }
            if string_array.len() <= RUNNING_COMSCRIPT_NAME_INDEX + 1 {
                return;
            }
            let Some(chunk_index) =
                self.get_chunk_index(string_array[RUNNING_COMSCRIPT_NAME_INDEX])
            else {
                return;
            };
            // Create and start string feed on a process messages instance for the
            // chunk monitor.
            let messages = {
                let mut messages_array = self.messages_array.lock().unwrap();
                let index = chunk_index as usize;
                if index > messages_array.len() {
                    // Chunks won't come in order. Fill in the gaps in the array to avoid
                    // an out of bounds error.
                    messages_array.resize(index, None);
                }
                if index == messages_array.len() || messages_array[index].is_none() {
                    let messages = Arc::new(Mutex::new(
                        batch_run_tomo_process_monitor::create_process_messages_instance(
                            self.manager,
                        ),
                    ));
                    // Java `messagesArray.add(chunkIndex, messages)` inserts, which
                    // shifts every later chunk's messages when it fills a gap; fixed
                    // in translation: the slot is set (BUGS.md).
                    if index == messages_array.len() {
                        messages_array.push(Some(messages.clone()));
                    } else {
                        messages_array[index] = Some(messages.clone());
                    }
                    messages
                } else {
                    messages_array[index].clone().unwrap()
                }
            };
            ProcessMessages::start_string_feed(&messages);
            messages.lock().unwrap().clear();
            // Add or restart chunk
            if let Some(list) = &self.chunk_monitor_list {
                let mut list = list.lock().unwrap();
                if chunk_index >= 0 && (chunk_index as usize) < list.len() {
                    let chunk_monitor = match list[chunk_index as usize].clone() {
                        None => {
                            let chunk_monitor = if !this.is_reconnect() {
                                batch_run_tomo_chunk_monitor::get_instance(
                                    self.manager,
                                    self.run_list.clone(),
                                    chunk_index,
                                    self.process_data.clone(),
                                    Some(messages),
                                    self.processing_method,
                                )
                            } else {
                                batch_run_tomo_chunk_monitor::get_reconnect_instance(
                                    self.manager,
                                    self.run_list.clone(),
                                    chunk_index,
                                    self.process_data.clone(),
                                    Some(messages),
                                )
                            };
                            // If etomo is exiting, then clean up these monitors as soon as
                            // they are created.
                            if this.is_halt() {
                                chunk_monitor.erased().halt();
                            }
                            list[chunk_index as usize] = Some(chunk_monitor.clone());
                            chunk_monitor
                        }
                        Some(chunk_monitor) => {
                            chunk_monitor.erased().reset_run();
                            chunk_monitor
                        }
                    };
                    // `new Thread(new ThreadGroup("BatchRunTomo Chunk " + chunkIndex + 1),
                    // chunkMonitor).start()`.
                    let _ = std::thread::Builder::new()
                        .name(format!("BatchRunTomo Chunk {}{}", chunk_index, 1))
                        .spawn(move || Monitor::run(&*chunk_monitor));
                }
            }
        }
        // Look for comscript failed message.
        else if line.contains(process_output_strings::PROCESSCHUNK_FAILED_NEED_RESTART_MSG_ID) {
            // Comscript should be at the beginning of the line
            let Some(index) = line.find(' ') else {
                return;
            };
            // Java unboxes a null chunk index here (NullPointerException on the
            // monitor thread); fixed in translation: the line is ignored (BUGS.md).
            let Some(chunk_index) = self.get_chunk_index(&line[..index]) else {
                return;
            };
            if let Some(list) = &self.chunk_monitor_list {
                let list = list.lock().unwrap();
                if chunk_index >= 0
                    && (chunk_index as usize) < list.len()
                    && let Some(chunk_monitor) = &list[chunk_index as usize]
                {
                    chunk_monitor.erased().stop();
                }
            }
        }
    }

    /// Java `hasProgressBarAccess()`: inherited.
    fn has_progress_bar_access(&self, _this: &ProcesschunksProcessMonitor) -> bool {
        true
    }
}
