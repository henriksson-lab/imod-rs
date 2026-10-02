//! `IMOD/Etomo/src/etomo/storage/BatchRunTomoLog.java`.
//!
//! Reads the batchruntomo dataset log and turns its recognised lines into
//! `ProcessSignal`s.  The Java inner (non-static) class `Iterator` reads the outer
//! instance's `log` field; that field is assigned once, before the first iterator is
//! made, and never changes afterwards, so each `Iterator` carries its own handle to the
//! same `LogFile.Handle`.
//!
//! Java `printStackTrace()` for a caught exception is an `eprintln!` of the error.

use std::sync::{Arc, Mutex};
// This module defines a struct named `Iterator` (Java `BatchRunTomoLog.Iterator`),
// which shadows the prelude trait; import the trait anonymously for its methods.
use std::iter::Iterator as _;

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::process::process_output_strings::{BRT_TRANSFER_FID_A, BRT_TRANSFER_FID_B};
use crate::imod::etomo::process::process_state::ProcessState;
use crate::imod::etomo::storage::log_file::{Handle, LogFile, LogFileError, ReaderId};
use crate::imod::etomo::storage::process_signal::ProcessSignal;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::const_meta_data::ConstMetaData;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::file_type::CLASS;
use crate::imod::etomo::r#type::meta_data::MetaData;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::step::Step;

/// Java final `BatchRunTomoLog`.
pub struct BatchRunTomoLog {
    /// Java private field `log`, initialised to null.
    log: Mutex<Option<Arc<Handle>>>,
}

impl Default for BatchRunTomoLog {
    fn default() -> BatchRunTomoLog {
        BatchRunTomoLog::new()
    }
}

impl BatchRunTomoLog {
    /// Java `BatchRunTomoLog()`.
    pub fn new() -> BatchRunTomoLog {
        BatchRunTomoLog {
            log: Mutex::new(None),
        }
    }

    /// Java `seek(BaseManager, MetaData)`.  Seeks a location based on metadata and
    /// returns a new iterator.  (The source's `manager != null` test is always true
    /// for the non-null manager here.)
    pub fn seek<'a>(
        &self,
        manager: &'static dyn BaseManager,
        meta_data: &'a MetaData,
    ) -> Option<Iterator<'a>> {
        let result: Result<Option<Iterator<'a>>, LogFileError> = (|| {
            let mut log = self.log.lock().unwrap();
            if log.is_none() {
                *log = Some(LogFile::get_instance_file(
                    CLASS
                        .batch_run_tomo_dataset_log
                        .get_file(Some(manager), None)
                        .as_deref(),
                    Some(manager.get_emergency_monitor(None)),
                )?);
            }
            let mut iterator = Iterator::new((*log).clone(), meta_data);
            drop(log);
            if iterator.seek() {
                return Ok(Some(iterator));
            }
            Ok(None)
        })();
        match result {
            Ok(Some(iterator)) => return Some(iterator),
            Ok(None) => {}
            Err(e) => {
                // catch (final FileException | IOException e)
                eprintln!("{}", e);
            }
        }
        None
    }
}

/// Java public final inner class `BatchRunTomoLog.Iterator`.
pub struct Iterator<'a> {
    /// The enclosing instance's `log` field.
    log: Option<Arc<Handle>>,
    /// Java private final field `metaData`.
    meta_data: &'a MetaData,
    /// Java private final field `axisType`.
    axis_type: Option<AxisType>,
    /// Java private field `readId`, initialised to null.
    read_id: Option<ReaderId>,
    /// Java private field `next`, initialised to null.
    next: Option<ProcessSignal>,
    /// Java private field `axisID`, initialised to `AxisID.ONLY`.
    axis_id: Option<AxisID>,
    /// Java private field `completedDialogTypeA`, initialised to null.
    completed_dialog_type_a: Option<DialogType>,
    /// Java private field `completedDialogTypeB`, initialised to null.
    completed_dialog_type_b: Option<DialogType>,
    /// Java private field `completedStepA`, initialised to null.
    completed_step_a: Option<Step>,
    /// Java private field `completedStepB`, initialised to null.
    completed_step_b: Option<Step>,
}

impl<'a> Iterator<'a> {
    /// Java private `Iterator(MetaData)`.  (The source's null-metaData branch, which
    /// sets `axisType` to null, cannot be taken with a reference.)
    fn new(log: Option<Arc<Handle>>, meta_data: &'a MetaData) -> Iterator<'a> {
        let axis_type = Some(ConstMetaData::get_axis_type(meta_data));
        Iterator {
            log,
            meta_data,
            axis_type,
            read_id: None,
            next: None,
            axis_id: Some(AxisID::Only),
            completed_dialog_type_a: None,
            completed_dialog_type_b: None,
            completed_step_a: None,
            completed_step_b: None,
        }
    }

    /// Java `seek()`.  Sets readId to the line after the location specified by
    /// metadata.  Sets readId to beginning of the file if there is no metadata or the
    /// location is not found.  If seek is called more then once, will continue to scan
    /// the file.  A failed scan of part of the file will cause it to scan the whole
    /// file.  Returns true if a location was successfully set.
    pub fn seek(&mut self) -> bool {
        let required = false;
        let log = match &self.log {
            None => return false,
            Some(log) => log.clone(),
        };
        // Start the reader if necessary.
        let mut read_from_start = false;
        let result: Result<bool, LogFileError> = (|| {
            if self.read_id.is_none() {
                read_from_start = true;
                self.read_id = log.open_reader_required(required)?;
                if self.read_id.is_none() {
                    return Ok(false);
                }
            }
            // Get the last axisID saved from a previous scan.
            if self.axis_type == Some(AxisType::DualAxis) {
                self.axis_id = self.meta_data.get_batch_run_tomo_log_read_axis_id();
            }
            // Get the timestamp saved from a previous scan.
            let timestamp = self.meta_data.get_batch_run_tomo_log_read_timestamp();
            let timestamp = match timestamp {
                None => {
                    // No last location is available - scanning will start at the
                    // beginning of the file.
                    if !read_from_start {
                        self.done();
                        self.read_id = log.open_reader_required(required)?;
                        if self.read_id.is_none() {
                            return Ok(false);
                        }
                    }
                    return Ok(true);
                }
                Some(timestamp) => timestamp,
            };
            // If the finish line wasn't found, processing will start from the started
            // line.  This means that some settings may be applied twice.
            let finished = self.meta_data.is_batch_run_tomo_log_read_finished();
            // Scan for the timestamp in a dataset started or finished line
            let mut seeks = 0;
            loop {
                while let Some(process_signal) = self.next_boolean(true) {
                    if process_signal.is_timestamp()
                        && finished == process_signal.is_finished()
                        && process_signal.equals_timestamp(Some(&timestamp))
                    {
                        return Ok(true);
                    }
                }
                seeks += 1;
                // Seek failed. Make sure entire file has been sought.
                // Go to the beginning of the file.
                self.done();
                self.read_id = log.open_reader_required(required)?;
                if self.read_id.is_none() {
                    return Ok(false);
                }
                if read_from_start {
                    // Last location has not been found - scanning will start at the
                    // beginning of the file.
                    return Ok(true);
                }
                if !(seeks < 2) {
                    break;
                }
            }
            // Last location has not been found - scanning will start at the beginning of
            // the file.
            Ok(true)
        })();
        match result {
            Ok(found) => return found,
            // catch (FileNotFoundException | LockException e)
            Err(LogFileError::Lock(_)) => {
                // No batchruntomo log available for this dataset.
            }
            Err(LogFileError::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => {
                // No batchruntomo log available for this dataset.
            }
            // catch (final LogFileException | IOException e)
            Err(e) => {
                eprintln!("{}", e);
            }
        }
        // Exception caught
        self.done();
        false
    }

    /// Java `hasNext()`.  A boolean peek.  Attempts to set the next member variable if
    /// it is null.  Returns true if next member variable is not null.
    pub fn has_next(&mut self) -> bool {
        if self.next.is_none() {
            self.next = self.find(false);
        }
        self.next.is_some()
    }

    /// Java `next()`.  Get the next recognized process name or step process signal.
    pub fn next(&mut self) -> Option<ProcessSignal> {
        self.next_boolean(false)
    }

    /// Java private `next(boolean)`.  Returns and deletes the next member variable if
    /// available.  Otherwise return the result of find().  `timestamp`: if true, only
    /// finds a process signal with a timestamp.
    fn next_boolean(&mut self, timestamp: bool) -> Option<ProcessSignal> {
        if self.next.is_some() {
            // The next signal was retrieved by hasNext.
            return self.next.take();
        }
        // Return the next signal
        self.find(timestamp)
    }

    /// Java private `find(boolean)`.  Find the next process signal with meaningful
    /// information.  Adds the dialogType and the process state (when necessary) to the
    /// signal to be returned.  With timestamp off a timestamp is not return - it is
    /// used to modify meta data.  With timestamp on, only timestamps are returned.
    fn find(&mut self, timestamp: bool) -> Option<ProcessSignal> {
        let (log, read_id) = match (&self.log, &self.read_id) {
            (Some(log), Some(read_id)) => (log.clone(), read_id.clone()),
            _ => return None,
        };
        loop {
            let line = match log.read_line(&read_id) {
                Ok(Some(line)) => line,
                Ok(None) => break,
                Err(e) => {
                    // catch (final LogFileException e) / catch (IOException e)
                    eprintln!("{}", e);
                    break;
                }
            };
            let mut process_signal = match ProcessSignal::get_instance_axis_type_string_boolean(
                self.axis_type,
                Some(&line),
                timestamp,
            ) {
                None => continue,
                Some(process_signal) => process_signal,
            };
            if timestamp {
                // Only looking for timestamps.
                return Some(process_signal);
            }
            // Save timestamps. The last timestamp saved will be used the next time the
            // log is checked.
            if process_signal.is_timestamp() {
                self.meta_data
                    .set_batch_run_tomo_log_read_timestamp(process_signal.get_timestamp());
                self.meta_data
                    .set_batch_run_tomo_log_read_finished(process_signal.is_finished());
                continue;
            }
            if process_signal.is_axis_id() {
                // Save the axisID being worked on
                self.axis_id = process_signal.get_axis_id();
                self.meta_data
                    .set_batch_run_tomo_log_read_axis_id(self.axis_id);
                continue;
            }
            // Get the dialog.
            let mut dialog_type: Option<DialogType> = None;
            let mut after_step: Option<Step> = None;
            let mut process_state = process_signal.get_process_state();
            // Check for a step signal
            let step = process_signal.get_step();
            if let Some(step) = step {
                // Update completed step
                if self.axis_id == Some(AxisID::Second) {
                    self.completed_step_b = Some(step);
                } else {
                    self.completed_step_a = Some(step);
                }
                // Step line cantains no axisID information - get it from a previous
                // line.
                process_signal.set_axis_id(self.axis_id);
                if step == Step::GOLD_DETECTION_3D {
                    process_signal.set_process_state(Some(ProcessState::InProgress));
                    dialog_type = Some(DialogType::FinalAlignedStack);
                } else {
                    process_state = Some(ProcessState::Complete);
                    if step == Step::BEAD_TRACKING {
                        process_signal.set_process_state(process_state);
                        dialog_type = Some(DialogType::FiducialModel);
                    } else if step == Step::FINE_ALIGNMENT {
                        process_signal.set_process_state(process_state);
                        dialog_type = Some(DialogType::FineAlignment);
                    } else if step == Step::POSITIONING {
                        process_signal.set_process_state(process_state);
                        dialog_type = Some(DialogType::TomogramPositioning);
                    } else if step == Step::TWO_D_FILTERING {
                        process_signal.set_process_state(process_state);
                        dialog_type = Some(DialogType::FinalAlignedStack);
                    } else if step == Step::RECONSTRUCTION {
                        process_signal.set_process_state(process_state);
                        dialog_type = Some(DialogType::TomogramGeneration);
                    }
                }
            } else {
                // Check for a signal with process name. Process state will already be
                // set.  Add the dialog type to the signal
                let process_name = process_signal.get_process_name();
                let (state, process_name) = match (process_state, process_name) {
                    (Some(state), Some(process_name)) => (state, process_name),
                    _ => continue,
                };
                if state == ProcessState::InProgress {
                    if process_name == ProcessName::XCORR {
                        dialog_type = Some(DialogType::CoarseAlignment);
                    } else if process_name == ProcessName::AUTOFIDSEED
                        || process_name == ProcessName::RUNRAPTOR
                        || process_name == ProcessName::TRACK
                        || process_name == ProcessName::XCORR_PT
                    {
                        dialog_type = Some(DialogType::FiducialModel);
                    } else if process_name == ProcessName::TRANSFERFID {
                        dialog_type = Some(DialogType::FiducialModel);
                        if line.find(BRT_TRANSFER_FID_B).is_some() {
                            process_signal.set_axis_id(Some(AxisID::Second));
                        } else if line.find(BRT_TRANSFER_FID_A).is_some() {
                            process_signal.set_axis_id(Some(AxisID::First));
                        }
                    } else if process_name == ProcessName::ALIGN {
                        dialog_type = Some(DialogType::FineAlignment);
                        after_step = Some(Step::BEAD_TRACKING);
                    } else if process_name == ProcessName::TOMOPITCH
                        || process_name == ProcessName::CRYO_POSITION
                        || process_name == ProcessName::FIND_SECTION_POS
                    {
                        dialog_type = Some(DialogType::TomogramPositioning);
                        after_step = Some(Step::FINE_ALIGNMENT);
                    } else if process_name == ProcessName::NEWST
                        || process_name == ProcessName::BLEND
                    {
                        dialog_type = Some(DialogType::FinalAlignedStack);
                        after_step = Some(Step::POSITIONING);
                    } else if process_name == ProcessName::TILT
                        || process_name == ProcessName::SIRTSETUP
                    {
                        dialog_type = Some(DialogType::TomogramGeneration);
                        after_step = Some(Step::TWO_D_FILTERING);
                    } else if process_name == ProcessName::SOLVEMATCH
                        || process_name == ProcessName::FIND_SECTION_LIM
                    {
                        dialog_type = Some(DialogType::TomogramCombination);
                        after_step = Some(Step::RECONSTRUCTION);
                    }
                } else if state == ProcessState::Complete {
                    if process_name == ProcessName::ERASER {
                        dialog_type = Some(DialogType::PreProcessing);
                    } else if process_name == ProcessName::PRENEWST
                        || process_name == ProcessName::PREBLEND
                    {
                        dialog_type = Some(DialogType::CoarseAlignment);
                    } else if process_name == ProcessName::VOLCOMBINE {
                        dialog_type = Some(DialogType::TomogramCombination);
                        after_step = Some(Step::RECONSTRUCTION);
                    } else if process_name == ProcessName::TRIMVOL {
                        dialog_type = Some(DialogType::PostProcessing);
                        after_step = Some(Step::RECONSTRUCTION);
                    }
                }
            }
            // Processes can be used outside of the dialog they are associated with.
            if let Some(dialog_type) = dialog_type {
                let completed_dialog_type = if self.axis_id == Some(AxisID::Second) {
                    self.completed_dialog_type_b
                } else {
                    self.completed_dialog_type_a
                };
                // Ignore a process that appears after its dialog is completed.
                if let Some(completed_dialog_type) = completed_dialog_type {
                    if completed_dialog_type.get_index() >= dialog_type.get_index() {
                        continue;
                    }
                }
            }
            if let Some(after_step) = after_step {
                let completed_step = if self.axis_id == Some(AxisID::Second) {
                    self.completed_step_b
                } else {
                    self.completed_step_a
                };
                // Ignore a process that comes too early - it has to come after a
                // specific step.
                match completed_step {
                    None => continue,
                    Some(completed_step) if completed_step.lt(Some(after_step)) => continue,
                    Some(_) => {}
                }
            }
            // Update completed dialog
            if dialog_type.is_some() && process_state == Some(ProcessState::Complete) {
                if self.axis_id == Some(AxisID::Second) {
                    self.completed_dialog_type_b = dialog_type;
                } else {
                    self.completed_dialog_type_a = dialog_type;
                }
            }
            if dialog_type.is_some() {
                process_signal.set_dialog_type(dialog_type);
                return Some(process_signal);
            }
        }
        None
    }

    /// Java `done()`.  Close the reader.
    pub fn done(&mut self) {
        if self.log.is_some() && self.read_id.is_some() {
            let read_id = self.read_id.take().unwrap();
            self.log.as_ref().unwrap().close_id(Some(&*read_id));
        }
        self.next = None;
    }
}
