//! `IMOD/Etomo/src/etomo/process/ProcessManager.java`.
//!
//! The application-manager process dispatcher.  Every entry point in the Java source
//! is retained below.  The source's `BaseProcessManager`, `ApplicationManager`, command
//! parameter classes, process classes, and output/UI classes have not yet been ported;
//! their nullable references are `Option<Infallible>`.  This is intentionally a hard
//! execution frontier: returning a fabricated process name or claiming a command
//! completed would be less faithful than retaining Java's null/error boundary.
#![allow(dead_code)]

use std::convert::Infallible;
use std::io;
use std::path::{Path, PathBuf};

use crate::imod::etomo::comscript::com_script_file::ComScriptFile;
use crate::imod::etomo::process::system_program::{ProcessCommand, SystemProgram};
use crate::imod::etomo::process::workflow::{ProcessWorkflow, WorkflowResult, WorkflowState};
use crate::imod::etomo::r#type::axis_id::AxisID;

/// Java final `ProcessManager extends BaseProcessManager`.
///
/// TODO(unit): `etomo/process/BaseProcessManager.java` and
/// `etomo/ApplicationManager.java`.  Rust has no base field here until the former is
/// translated; `app_manager` preserves the latter's Java nullable/reference boundary.
pub struct ProcessManager {
    app_manager: Option<Infallible>,
    /// Java `transferfidCommandLine`, initially null.
    transferfid_command_line: Option<String>,
    /// The application manager normally owns these values.  Keep explicit
    /// Rust ownership here until `ApplicationManager` is fully translated so
    /// immediate utility commands have a real dataset and process directory.
    dataset_name: Option<String>,
    working_directory: Option<PathBuf>,
    imod_bin_directory: Option<PathBuf>,
    /// `(starting angle, increment, raw-stack section count)` per AxisID.
    /// This is the metadata/header input `ApplicationManager.makeRawtltFile`
    /// reads before the full application manager is available here.
    raw_tilt_specs: [Option<(f64, f64, usize)>; 3],
    /// Typed replacement for the local subset of Java's process dispatch.
    local_workflow: ProcessWorkflow,
}

impl ProcessManager {
    /// Java `ProcessManager(ApplicationManager)`.
    pub fn new(app_manager: Option<Infallible>) -> ProcessManager {
        ProcessManager {
            app_manager,
            transferfid_command_line: None,
            dataset_name: None,
            working_directory: None,
            imod_bin_directory: None,
            raw_tilt_specs: [None; 3],
            local_workflow: ProcessWorkflow::new(),
        }
    }

    pub fn set_dataset_name(&mut self, dataset_name: Option<&str>) {
        self.dataset_name = dataset_name.map(str::to_owned);
    }

    pub fn set_working_directory(&mut self, directory: Option<&Path>) {
        self.working_directory = directory.map(Path::to_owned);
    }

    pub fn set_imod_bin_directory(&mut self, directory: Option<&Path>) {
        self.imod_bin_directory = directory.map(Path::to_owned);
    }

    pub fn set_raw_tilt_spec(&mut self, axis_id: AxisID, start: f64, step: f64, sections: usize) {
        self.raw_tilt_specs[axis_id.get_axis_of_extension() as usize] =
            Some((start, step, sections));
    }

    fn make_rawtlt_file(&self, axis_id: AxisID) -> Result<(), String> {
        let dataset = self
            .get_dataset_name()
            .ok_or_else(|| "no dataset name configured".to_owned())?;
        let directory = self
            .working_directory
            .as_ref()
            .ok_or_else(|| "no working directory configured".to_owned())?;
        let (start, step, sections) = self.raw_tilt_specs[axis_id.get_axis_of_extension() as usize]
            .ok_or_else(|| "no raw tilt range/section configuration".to_owned())?;
        let rawtlt = directory.join(format!("{dataset}{}.rawtlt", axis_id.get_extension()));
        if rawtlt.exists() {
            let backup = directory.join(format!("{dataset}{}.rawtlt~", axis_id.get_extension()));
            std::fs::rename(&rawtlt, backup).map_err(|error| error.to_string())?;
        }
        let mut output = String::new();
        for section in 0..sections {
            let value = start + step * section as f64;
            output.push_str(&format!("{value:?}\n"));
        }
        std::fs::write(rawtlt, output).map_err(|error| error.to_string())
    }

    fn imod_command(&self, name: &str) -> PathBuf {
        self.imod_bin_directory
            .as_ref()
            .map_or_else(|| PathBuf::from(name), |directory| directory.join(name))
    }

    /// Queue a concrete local command.  Parameter-specific Java entry points
    /// can feed this as their command/comscript types become available.
    pub fn queue_local_command(&mut self, name: impl Into<String>, command: ProcessCommand) {
        self.local_workflow.push(name, command);
    }
    pub fn start_local_workflow(&mut self) -> io::Result<bool> {
        self.local_workflow.start()
    }
    pub fn poll_local_workflow(&mut self) -> io::Result<WorkflowState> {
        self.local_workflow.poll()
    }
    /// Toggle pause/resume for the active local process.
    pub fn pause_local_workflow(&mut self) -> io::Result<bool> {
        self.local_workflow.pause()
    }
    pub fn cancel_local_workflow(&mut self) -> io::Result<()> {
        self.local_workflow.cancel()
    }
    pub fn local_workflow_state(&self) -> WorkflowState {
        self.local_workflow.state()
    }
    /// Concrete terminal results for locally translated command/com-script
    /// dispatch.  Manager/UI code can use these records for output and failure
    /// handling without reconstructing status from nullable Java references.
    pub fn local_workflow_results(&self) -> &[WorkflowResult] {
        self.local_workflow.results()
    }
    /// Load an ordinary eTomo `.com` file into the typed process queue.
    pub fn queue_com_script(&mut self, path: &Path) -> std::io::Result<usize> {
        let script = ComScriptFile::load(path)?;
        let commands = script.process_commands();
        let count = commands.len();
        for (name, command) in commands {
            self.queue_local_command(name, command);
        }
        Ok(count)
    }

    /// Java `setupCtfPlotterComScript`.
    pub fn setup_ctf_plotter_com_script(
        &self,
        ctf_phase_flip_param: Option<Infallible>,
        axis_id: Option<AxisID>,
    ) {
        let _ = (ctf_phase_flip_param, axis_id);
    }
    /// Java `setupCtfCorrectionComScript`.
    pub fn setup_ctf_correction_com_script(&self, axis_id: Option<AxisID>) {
        let _ = axis_id;
    }
    /// Java public `setupComScripts(AxisID, CopyTomoComs, AxisType)`.
    pub fn setup_com_scripts(
        &self,
        axis_id: Option<AxisID>,
        param: Option<Infallible>,
        axis_type: Option<Infallible>,
    ) -> Option<Infallible> {
        let _ = (axis_id, param, axis_type);
        None
    }
    /// Java private `setupComScripts(CopyTomoComs, AxisID, AxisType)`.
    fn setup_com_scripts_from_copy_tomo_coms(
        &self,
        copy_tomo_coms: Option<Infallible>,
        axis_id: Option<AxisID>,
        axis_type: Option<Infallible>,
    ) -> Option<Infallible> {
        let _ = (copy_tomo_coms, axis_id, axis_type);
        None
    }
    /// Java `batchruntomo`.
    pub fn batchruntomo(&self, axis_id: Option<AxisID>, param: Option<Infallible>) -> bool {
        let _ = (axis_id, param);
        false
    }
    /// Java `makecomfile`.
    pub fn makecomfile(&self, axis_id: Option<AxisID>, param: Option<Infallible>) -> bool {
        let _ = (axis_id, param);
        false
    }
    /// Java `eraser`.
    pub fn eraser(
        &self,
        axis_id: Option<AxisID>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
    ) -> Option<String> {
        let _ = (axis_id, process_result_display, process_series);
        None
    }
    /// Java `goldEraser`.
    pub fn gold_eraser(
        &self,
        axis_id: Option<AxisID>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
    ) -> Option<String> {
        let _ = (axis_id, process_result_display, process_series);
        None
    }
    /// Java `clipStats`.
    pub fn clip_stats(
        &self,
        param: Option<Infallible>,
        axis_id: Option<AxisID>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
    ) -> Option<String> {
        let _ = (param, axis_id, process_result_display, process_series);
        None
    }
    /// Java `findSection`.
    pub fn find_section(
        &self,
        param: Option<Infallible>,
        axis_id: Option<AxisID>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
    ) -> Option<String> {
        let _ = (param, axis_id, process_result_display, process_series);
        None
    }
    /// Java private `getDatasetName`.
    fn get_dataset_name(&self) -> Option<String> {
        self.dataset_name.clone()
    }
    /// Java `crossCorrelate`.
    pub fn cross_correlate(
        &self,
        param: Option<Infallible>,
        axis_id: Option<AxisID>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
    ) -> Option<String> {
        let _ = (param, axis_id, process_result_display, process_series);
        None
    }
    /// Java `tiltxcorr`.
    pub fn tiltxcorr(
        &self,
        param: Option<Infallible>,
        process_name: Option<Infallible>,
        axis_id: Option<AxisID>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
    ) -> Option<String> {
        let _ = (
            param,
            process_name,
            axis_id,
            process_result_display,
            process_series,
        );
        None
    }
    /// Java `autofidseed`.
    pub fn autofidseed(
        &self,
        param: Option<Infallible>,
        axis_id: Option<AxisID>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
    ) -> Option<String> {
        let _ = (param, axis_id, process_result_display, process_series);
        None
    }
    /// Java `makeDistortionCorrectedStack`.
    pub fn make_distortion_corrected_stack(
        &self,
        axis_id: Option<AxisID>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
    ) -> Option<String> {
        let _ = (axis_id, process_result_display, process_series);
        None
    }
    /// Java `coarseAlign`.
    pub fn coarse_align(
        &self,
        param: Option<Infallible>,
        axis_id: Option<AxisID>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
    ) -> Option<String> {
        let _ = (param, axis_id, process_result_display, process_series);
        None
    }
    /// Java `preblend`.
    pub fn preblend(
        &self,
        param: Option<Infallible>,
        axis_id: Option<AxisID>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
    ) -> Option<String> {
        let _ = (param, axis_id, process_result_display, process_series);
        None
    }
    /// Java `blend`.
    pub fn blend(
        &self,
        blendmont_param: Option<Infallible>,
        axis_id: Option<AxisID>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
    ) -> Option<String> {
        let _ = (
            blendmont_param,
            axis_id,
            process_result_display,
            process_series,
        );
        None
    }
    /// Java `generatePreXG`.
    pub fn generate_pre_xg(&self, axis_id: Option<AxisID>) -> Result<(), String> {
        let axis_id = axis_id.ok_or_else(|| "no axis supplied".to_owned())?;
        let dataset = self
            .get_dataset_name()
            .ok_or_else(|| "no dataset name configured".to_owned())?;
        let extension = axis_id.get_extension();
        let command = vec![
            self.imod_command("xftoxg").to_string_lossy().into_owned(),
            "-NumberToFit".to_owned(),
            "0".to_owned(),
            format!("{dataset}{extension}.prexf"),
            format!("{dataset}{extension}.prexg"),
        ];
        self.run_command(Some(&command), Some(axis_id), None)
    }
    /// Java `generateNonFidXF`.
    pub fn generate_non_fid_xf(&self, axis_id: Option<AxisID>) -> Result<(), String> {
        let axis_id = axis_id.ok_or_else(|| "no axis supplied".to_owned())?;
        let dataset = self
            .get_dataset_name()
            .ok_or_else(|| "no dataset name configured".to_owned())?;
        let extension = axis_id.get_extension();
        let command = vec![
            self.imod_command("xfproduct")
                .to_string_lossy()
                .into_owned(),
            format!("{dataset}{extension}.prexg"),
            format!("rotation{extension}.xf"),
            format!("{dataset}{extension}_nonfid.xf"),
        ];
        self.run_command(Some(&command), Some(axis_id), None)
    }
    /// Java `setupNonFiducialAlign`.
    pub fn setup_non_fiducial_align(&self, axis_id: Option<AxisID>) -> Result<(), String> {
        let axis_id = axis_id.ok_or_else(|| "no axis supplied".to_owned())?;
        let dataset = self
            .get_dataset_name()
            .ok_or_else(|| "no dataset name configured".to_owned())?;
        let directory = self
            .working_directory
            .as_ref()
            .ok_or_else(|| "no working directory configured".to_owned())?;
        let axis_dataset = format!("{dataset}{}", axis_id.get_extension());
        let copy = |from: PathBuf, to: PathBuf| {
            std::fs::copy(&from, &to)
                .map(|_| ())
                .map_err(|error| format!("{} -> {}: {error}", from.display(), to.display()))
        };
        copy(
            directory.join(format!("{axis_dataset}_nonfid.xf")),
            directory.join(format!("{axis_dataset}.xf")),
        )?;
        let rawtlt = directory.join(format!("{axis_dataset}.rawtlt"));
        if !rawtlt.exists() {
            self.make_rawtlt_file(axis_id)?;
        }
        copy(rawtlt, directory.join(format!("{axis_dataset}.tlt")))
    }
    /// Java `setupFiducialAlign`.
    pub fn setup_fiducial_align(
        &self,
        axis_id: Option<AxisID>,
        process_result_display: Option<Infallible>,
    ) -> Result<bool, String> {
        let _ = process_result_display;
        let axis_id = axis_id.ok_or_else(|| "no axis supplied".to_owned())?;
        let dataset = self
            .get_dataset_name()
            .ok_or_else(|| "no dataset name configured".to_owned())?;
        let directory = self
            .working_directory
            .as_ref()
            .ok_or_else(|| "no working directory configured".to_owned())?;
        let axis_dataset = format!("{dataset}{}", axis_id.get_extension());
        let xf = directory.join(format!("{axis_dataset}.xf"));
        let fid_xf = directory.join(format!("{axis_dataset}_fid.xf"));
        let nonfid_xf = directory.join(format!("{axis_dataset}_nonfid.xf"));
        let tlt = directory.join(format!("{axis_dataset}.tlt"));
        let fid_tlt = directory.join(format!("{axis_dataset}_fid.tlt"));
        let tltxf = directory.join(format!("{axis_dataset}.tltxf"));
        let copy = |from: &Path, to: &Path| {
            std::fs::copy(from, to)
                .map(|_| ())
                .map_err(|error| format!("{} -> {}: {error}", from.display(), to.display()))
        };
        let remove_if_present = |path: &Path| match std::fs::remove_file(path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(format!("{}: {error}", path.display())),
        };
        if tltxf.exists() {
            if fid_xf.exists() {
                copy(&fid_xf, &xf)?;
                copy(&fid_tlt, &tlt)?;
            } else if nonfid_xf.exists() {
                remove_if_present(&xf)?;
                remove_if_present(&tlt)?;
            } else {
                copy(&xf, &fid_xf)?;
                copy(&tlt, &fid_tlt)?;
            }
        } else {
            remove_if_present(&xf)?;
            remove_if_present(&tlt)?;
        }
        Ok(true)
    }
    /// Java `midasRawStack`.
    pub fn midas_raw_stack(&self, axis_id: Option<AxisID>, image_rotation: f64) {
        let _ = (axis_id, image_rotation);
    }
    /// Java `getMidasRawStackCommandLine`.
    pub fn get_midas_raw_stack_command_line(
        &self,
        dataset_name: Option<&str>,
        axis_id: Option<AxisID>,
        image_rotation: f64,
    ) -> Option<Vec<String>> {
        let _ = (dataset_name, axis_id, image_rotation);
        None
    }
    /// Java `midasBlendStack`.
    pub fn midas_blend_stack(&self, axis_id: Option<AxisID>, image_rotation: f64) {
        let _ = (axis_id, image_rotation);
    }
    /// Java `getMidasBlendStackCommandLine`.
    pub fn get_midas_blend_stack_command_line(
        &self,
        dataset_name: Option<&str>,
        axis_id: Option<AxisID>,
        image_rotation: f64,
    ) -> Option<Vec<String>> {
        let _ = (dataset_name, axis_id, image_rotation);
        None
    }
    /// Java `midasFixEdges`.
    pub fn midas_fix_edges(&self, axis_id: Option<AxisID>) {
        let _ = axis_id;
    }
    /// Java `getMidasFixEdgesCommandLine`.
    pub fn get_midas_fix_edges_command_line(
        &self,
        dataset_name: Option<&str>,
        axis_id: Option<AxisID>,
    ) -> Option<Vec<String>> {
        let _ = (dataset_name, axis_id);
        None
    }
    /// Java `fiducialModelTrack`.
    pub fn fiducial_model_track(
        &self,
        param: Option<Infallible>,
        axis_id: Option<AxisID>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
    ) -> Option<String> {
        let _ = (param, axis_id, process_result_display, process_series);
        None
    }
    /// Java `fineAlignment`.
    pub fn fine_alignment(
        &self,
        param: Option<Infallible>,
        axis_id: Option<AxisID>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
    ) -> Option<String> {
        let _ = (param, axis_id, process_result_display, process_series);
        None
    }
    /// Java `generateAlignLogs`.
    pub fn generate_align_logs(&self, axis_id: Option<AxisID>) {
        let _ = axis_id;
    }
    /// Java `generateAlignLogForProjectLog`.
    pub fn generate_align_log_for_project_log(
        &self,
        axis_id: Option<AxisID>,
    ) -> Option<Infallible> {
        let _ = axis_id;
        None
    }
    /// Java `copyFiducialAlignFiles`.
    pub fn copy_fiducial_align_files(&self, axis_id: Option<AxisID>) {
        let _ = axis_id;
    }
    /// Java `transferFiducials`.
    pub fn transfer_fiducials(
        &mut self,
        transferfid_param: Option<Infallible>,
        axis_id: Option<AxisID>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
    ) -> Option<String> {
        let _ = (
            transferfid_param,
            axis_id,
            process_result_display,
            process_series,
        );
        None
    }
    /// Java `createSample`.
    pub fn create_sample(
        &self,
        axis_id: Option<AxisID>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
    ) -> Option<String> {
        let _ = (axis_id, process_result_display, process_series);
        None
    }
    /// Java `tomopitch`.
    pub fn tomopitch(
        &self,
        axis_id: Option<AxisID>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
    ) -> Option<String> {
        let _ = (axis_id, process_result_display, process_series);
        None
    }
    /// Java `processchunks`.
    pub fn processchunks(
        &self,
        axis_id: Option<AxisID>,
        param: Option<Infallible>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
    ) -> Option<String> {
        let _ = (axis_id, param, process_result_display, process_series);
        None
    }
    /// Java `newst`.
    pub fn newst(
        &self,
        newst_param: Option<Infallible>,
        axis_id: Option<AxisID>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
    ) -> Option<String> {
        let _ = (newst_param, axis_id, process_result_display, process_series);
        None
    }
    /// Java `cryoPosition`.
    pub fn cryo_position(
        &self,
        param: Option<Infallible>,
        axis_id: Option<AxisID>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
    ) -> Option<String> {
        let _ = (param, axis_id, process_result_display, process_series);
        None
    }
    /// Java `findBeads3d`.
    pub fn find_beads_3d(
        &self,
        param: Option<Infallible>,
        axis_id: Option<AxisID>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
    ) -> Option<String> {
        let _ = (param, axis_id, process_result_display, process_series);
        None
    }
    /// Java `mtffilter`.
    pub fn mtffilter(
        &self,
        param: Option<Infallible>,
        axis_id: Option<AxisID>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
    ) -> Option<String> {
        let _ = (param, axis_id, process_result_display, process_series);
        None
    }
    /// Java `ctfPlotter`.
    pub fn ctf_plotter(&self, axis_id: Option<AxisID>, process_result_display: Option<Infallible>) {
        let _ = (axis_id, process_result_display);
    }
    /// Java `ctfCorrection`.
    pub fn ctf_correction(
        &self,
        param: Option<Infallible>,
        axis_id: Option<AxisID>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
    ) -> Option<String> {
        let _ = (param, axis_id, process_result_display, process_series);
        None
    }
    /// Java `reconnectTilt`.
    pub fn reconnect_tilt(
        &self,
        axis_id: Option<AxisID>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
    ) -> bool {
        let _ = (axis_id, process_result_display, process_series);
        false
    }
    /// Java `sirtsetup`.
    pub fn sirtsetup(
        &self,
        axis_id: Option<AxisID>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
    ) -> Option<String> {
        let _ = (axis_id, process_result_display, process_series);
        None
    }
    /// Java `multifiltSetup`.
    pub fn multifilt_setup(
        &self,
        axis_id: Option<AxisID>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
    ) -> Option<String> {
        let _ = (axis_id, process_result_display, process_series);
        None
    }
    /// Java `ctf3dSetup`.
    pub fn ctf_3d_setup(
        &self,
        axis_id: Option<AxisID>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
    ) -> Option<String> {
        let _ = (axis_id, process_result_display, process_series);
        None
    }
    /// Java `subtomoSetup`.
    pub fn subtomo_setup(
        &self,
        axis_id: Option<AxisID>,
        process_series: Option<Infallible>,
        process_result_display: Option<Infallible>,
    ) -> Option<String> {
        let _ = (axis_id, process_series, process_result_display);
        None
    }
    /// Java `altTomoSetup`.
    pub fn alt_tomo_setup(
        &self,
        axis_id: Option<AxisID>,
        process_series: Option<Infallible>,
        process_result_display: Option<Infallible>,
    ) -> Option<String> {
        let _ = (axis_id, process_series, process_result_display);
        None
    }
    /// Java `tilt`.
    pub fn tilt(
        &self,
        axis_id: Option<AxisID>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
    ) -> Option<String> {
        let _ = (axis_id, process_result_display, process_series);
        None
    }
    /// Java `tilt3dFind`.
    pub fn tilt_3d_find(
        &self,
        axis_id: Option<AxisID>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
    ) -> Option<String> {
        let _ = (axis_id, process_result_display, process_series);
        None
    }
    /// Java `tilt3dFindReproject`.
    pub fn tilt_3d_find_reproject(
        &self,
        axis_id: Option<AxisID>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
    ) -> Option<String> {
        let _ = (axis_id, process_result_display, process_series);
        None
    }
    /// Java `splittilt`.
    pub fn splittilt(
        &self,
        param: Option<Infallible>,
        axis_id: Option<AxisID>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
    ) -> Option<String> {
        let _ = (param, axis_id, process_result_display, process_series);
        None
    }
    /// Java `splitCorrection`.
    pub fn split_correction(
        &self,
        param: Option<Infallible>,
        axis_id: Option<AxisID>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
    ) -> Option<String> {
        let _ = (param, axis_id, process_result_display, process_series);
        None
    }
    /// Java `extracttilts`.
    pub fn extracttilts(
        &self,
        axis_id: Option<AxisID>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
    ) -> Option<String> {
        let _ = (axis_id, process_result_display, process_series);
        None
    }
    /// Java `extractpieces`.
    pub fn extractpieces(
        &self,
        axis_id: Option<AxisID>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
    ) -> Option<String> {
        let _ = (axis_id, process_result_display, process_series);
        None
    }
    /// Java `extractmagrad`.
    pub fn extractmagrad(
        &self,
        param: Option<Infallible>,
        axis_id: Option<AxisID>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
    ) -> Option<String> {
        let _ = (param, axis_id, process_result_display, process_series);
        None
    }
    /// Java `splitcombine`.
    pub fn splitcombine(
        &self,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
    ) -> Option<String> {
        let _ = (process_result_display, process_series);
        None
    }
    /// Java `setupCombineScripts`.
    pub fn setup_combine_scripts(&self, process_result_display: Option<Infallible>) -> bool {
        let _ = process_result_display;
        false
    }
    /// Java `setupCombineOnlyMakeCombineCom`.
    pub fn setup_combine_only_make_combine_com(&self) -> bool {
        false
    }
    /// Java `modelToPatch`.
    pub fn model_to_patch(&self, axis_id: Option<AxisID>) {
        let _ = axis_id;
    }
    /// Java `combine`.
    pub fn combine(
        &self,
        combine_comscript_state: Option<Infallible>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
    ) -> Option<String> {
        let _ = (
            combine_comscript_state,
            process_result_display,
            process_series,
        );
        None
    }
    /// Java private `solvematch`.
    fn solvematch(&self, process_series: Option<Infallible>) -> Option<String> {
        let _ = process_series;
        None
    }
    /// Java private `matchvol1`.
    fn matchvol1(&self, process_series: Option<Infallible>) -> Option<String> {
        let _ = process_series;
        None
    }
    /// Java `matchorwarp`.
    pub fn matchorwarp(&self, process_series: Option<Infallible>) -> Option<String> {
        let _ = process_series;
        None
    }
    /// Java `trimVolume`.
    pub fn trim_volume(
        &self,
        trimvol_param: Option<Infallible>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
    ) -> Option<String> {
        let _ = (trimvol_param, process_result_display, process_series);
        None
    }
    /// Java `excludeViews`.
    pub fn exclude_views(
        &self,
        param: Option<Infallible>,
        axis_id: Option<AxisID>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
    ) -> Option<String> {
        let _ = (param, axis_id, process_result_display, process_series);
        None
    }
    /// Java `archiveOrig`.
    pub fn archive_orig(
        &self,
        param: Option<Infallible>,
        process_series: Option<Infallible>,
    ) -> Option<String> {
        let _ = (param, process_series);
        None
    }
    /// Java `restrictalign`.
    pub fn restrictalign(
        &self,
        param: Option<Infallible>,
        axis_id: Option<AxisID>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
    ) -> Option<String> {
        let _ = (param, axis_id, process_result_display, process_series);
        None
    }
    /// Java `flatten`.
    pub fn flatten(
        &self,
        param: Option<Infallible>,
        axis_id: Option<AxisID>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
    ) -> Option<String> {
        let _ = (param, axis_id, process_result_display, process_series);
        None
    }
    /// Java `flattenWarp`.
    pub fn flatten_warp(
        &self,
        param: Option<Infallible>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
    ) -> Option<String> {
        let _ = (param, process_result_display, process_series);
        None
    }
    /// Java `runraptor`.
    pub fn runraptor(
        &self,
        param: Option<Infallible>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
    ) -> Option<String> {
        let _ = (param, process_result_display, process_series);
        None
    }
    /// Java `squeezeVolume`.
    pub fn squeeze_volume(
        &self,
        param: Option<Infallible>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
    ) -> Option<String> {
        let _ = (param, process_result_display, process_series);
        None
    }
    /// Java `reduceFiltVol`.
    pub fn reduce_filt_vol(
        &self,
        param: Option<Infallible>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
    ) -> Option<String> {
        let _ = (param, process_result_display, process_series);
        None
    }
    /// Java private `showLogFile`.
    fn show_log_file(&self, log_file: Option<&Path>) {
        let _ = log_file;
    }
    /// Java private `runCommand`.  The original executes this path
    /// synchronously, then surfaces stderr and the trailing `ERROR:` stdout
    /// block when the child fails.  A translated log handle is not required to
    /// run the command; callers that have one can attach it at the manager/UI
    /// layer while this preserves the child and error contract.
    fn run_command(
        &self,
        command_array: Option<&[String]>,
        axis_id: Option<AxisID>,
        log_file: Option<Infallible>,
    ) -> Result<(), String> {
        let _ = (axis_id, log_file);
        let command_array = command_array.ok_or_else(|| "no command supplied".to_owned())?;
        let Some((program, args)) = command_array.split_first() else {
            return Err("no command supplied".to_owned());
        };
        let mut command = ProcessCommand::new(program).args(args.iter().cloned());
        if let Some(directory) = &self.working_directory {
            command.set_working_directory(directory);
        }
        let mut program = SystemProgram::spawn(&command).map_err(|error| error.to_string())?;
        let (status, _) = program
            .wait_and_drain()
            .map_err(|error| error.to_string())?;
        if status.success() {
            return Ok(());
        }
        let mut message = String::new();
        for line in program.get_std_error() {
            message.push_str(line);
            message.push('\n');
        }
        let mut found_error = false;
        for line in program.get_std_output() {
            if !found_error && line.contains("ERROR:") {
                found_error = true;
            }
            if found_error {
                message.push_str(line);
            }
        }
        Err(message)
    }
    /// Java override `postProcess(DetachedProcess)`.
    pub fn post_process_detached_process(&self, process: Option<Infallible>) {
        let _ = process;
    }
    /// Java private `postProcess(String, AxisID)`.
    fn post_process_name(&self, process_name: Option<&str>, axis_id: Option<AxisID>) {
        let _ = (process_name, axis_id);
    }
    /// Java override `postProcess(ComScriptProcess)`.
    pub fn post_process_com_script_process(&self, script: Option<Infallible>) {
        let _ = script;
    }
    /// Java override `errorProcess(ComScriptProcess)`.
    pub fn error_process_com_script_process(&self, script: Option<Infallible>) {
        let _ = script;
    }
    /// Java private `setInvalidEdgeFunctions`.
    fn set_invalid_edge_functions(&self, command: Option<Infallible>, succeeded: bool) {
        let _ = (command, succeeded);
    }
    /// Java override `postProcess(BackgroundProcess)`.
    pub fn post_process_background_process(&self, process: Option<Infallible>) {
        let _ = process;
    }
    /// Java override `errorProcess(BackgroundProcess)`.
    pub fn error_process_background_process(&self, process: Option<Infallible>) {
        let _ = process;
    }
    /// Java package-private `getManager`.
    pub fn get_manager(&self) -> Option<Infallible> {
        self.app_manager
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn com_script_blocks_execute_and_expose_manager_level_results() {
        let path = std::env::temp_dir().join(format!(
            "imod-rs-process-manager-{}.com",
            std::process::id()
        ));
        std::fs::write(&path, "$true\n$true\n").unwrap();
        let mut manager = ProcessManager::new(None);
        assert_eq!(manager.queue_com_script(&path).unwrap(), 2);
        assert!(manager.start_local_workflow().unwrap());
        loop {
            if !matches!(
                manager.poll_local_workflow().unwrap(),
                WorkflowState::Running
            ) {
                break;
            }
            std::thread::yield_now();
        }
        assert_eq!(manager.local_workflow_state(), WorkflowState::Complete);
        assert_eq!(manager.local_workflow_results().len(), 2);
        assert!(
            manager
                .local_workflow_results()
                .iter()
                .all(|result| result.success)
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn immediate_runner_retains_source_failure_message_selection() {
        let manager = ProcessManager::new(None);
        let command = [
            "sh".to_owned(),
            "-c".to_owned(),
            "printf 'before\\nERROR: first\\nafter\\n'; printf 'stderr\\n' >&2; exit 3".to_owned(),
        ];
        assert_eq!(
            manager.run_command(Some(&command), Some(AxisID::First), None),
            Err("stderr\nERROR: firstafter".to_owned())
        );
        assert_eq!(
            manager.run_command(None, None, None),
            Err("no command supplied".into())
        );
    }

    #[cfg(unix)]
    #[test]
    fn transform_generators_run_configured_imod_tools_with_source_argument_order() {
        use std::os::unix::fs::PermissionsExt;

        let directory = std::env::temp_dir().join(format!(
            "imod-rs-process-manager-generators-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("unnamed")
        ));
        std::fs::create_dir(&directory).unwrap();
        for tool in ["xftoxg", "xfproduct"] {
            let path = directory.join(tool);
            std::fs::write(&path, "#!/bin/sh\nprintf '%s\\n' \"$@\" > command.args\n").unwrap();
            let mut permissions = std::fs::metadata(&path).unwrap().permissions();
            permissions.set_mode(0o755);
            std::fs::set_permissions(path, permissions).unwrap();
        }
        let mut manager = ProcessManager::new(None);
        manager.set_dataset_name(Some("sample"));
        manager.set_working_directory(Some(&directory));
        manager.set_imod_bin_directory(Some(&directory));
        manager.generate_pre_xg(Some(AxisID::First)).unwrap();
        assert_eq!(
            std::fs::read_to_string(directory.join("command.args")).unwrap(),
            "-NumberToFit\n0\nsamplea.prexf\nsamplea.prexg\n"
        );
        manager.generate_non_fid_xf(Some(AxisID::First)).unwrap();
        assert_eq!(
            std::fs::read_to_string(directory.join("command.args")).unwrap(),
            "samplea.prexg\nrotationa.xf\nsamplea_nonfid.xf\n"
        );
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn nonfiducial_setup_copies_generated_transform_and_raw_tilt_files() {
        let directory = std::env::temp_dir().join(format!(
            "imod-rs-process-manager-nonfid-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("unnamed")
        ));
        std::fs::create_dir(&directory).unwrap();
        std::fs::write(directory.join("samplea_nonfid.xf"), "transform").unwrap();
        let mut manager = ProcessManager::new(None);
        manager.set_dataset_name(Some("sample"));
        manager.set_working_directory(Some(&directory));
        manager.set_raw_tilt_spec(AxisID::First, -2., 1.5, 3);
        manager
            .setup_non_fiducial_align(Some(AxisID::First))
            .unwrap();
        assert_eq!(
            std::fs::read_to_string(directory.join("samplea.xf")).unwrap(),
            "transform"
        );
        assert_eq!(
            std::fs::read_to_string(directory.join("samplea.tlt")).unwrap(),
            "-2.0\n-0.5\n1.0\n"
        );
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn fiducial_setup_restores_protected_files_or_removes_stale_nonfiducial_outputs() {
        let directory = std::env::temp_dir().join(format!(
            "imod-rs-process-manager-fid-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("unnamed")
        ));
        std::fs::create_dir(&directory).unwrap();
        std::fs::write(directory.join("samplea.tltxf"), "ran").unwrap();
        std::fs::write(directory.join("samplea_fid.xf"), "fid-transform").unwrap();
        std::fs::write(directory.join("samplea_fid.tlt"), "fid-tilts").unwrap();
        let mut manager = ProcessManager::new(None);
        manager.set_dataset_name(Some("sample"));
        manager.set_working_directory(Some(&directory));
        assert!(
            manager
                .setup_fiducial_align(Some(AxisID::First), None)
                .unwrap()
        );
        assert_eq!(
            std::fs::read_to_string(directory.join("samplea.xf")).unwrap(),
            "fid-transform"
        );
        assert_eq!(
            std::fs::read_to_string(directory.join("samplea.tlt")).unwrap(),
            "fid-tilts"
        );
        std::fs::remove_file(directory.join("samplea_fid.xf")).unwrap();
        std::fs::remove_file(directory.join("samplea_fid.tlt")).unwrap();
        std::fs::write(directory.join("samplea_nonfid.xf"), "stale").unwrap();
        assert!(
            manager
                .setup_fiducial_align(Some(AxisID::First), None)
                .unwrap()
        );
        assert!(!directory.join("samplea.xf").exists());
        assert!(!directory.join("samplea.tlt").exists());
        std::fs::remove_dir_all(directory).unwrap();
    }
}
