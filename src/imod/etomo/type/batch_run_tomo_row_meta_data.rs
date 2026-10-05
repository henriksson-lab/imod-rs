//! `IMOD/Etomo/src/etomo/type/BatchRunTomoRowMetaData.java`.
//!
//! Meta data for a row of the BatchRunTomo Interface table.
//!
//! **Representation.**  The batch meta data's row map hands the same instance to the
//! dataset table (event dispatch thread) and stores it (possibly from a process
//! thread), so it is shared as an `Arc` with its fields behind one lock.  Java's
//! getters that return a field for the caller to modify (`getDualCheckBox()`,
//! `getRowProperties()`, ...) take a closure that receives the field.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use super::axis_id::AxisID;
use super::batch_run_tomo_dataset_meta_data::BatchRunTomoDatasetMetaData;
use super::batch_run_tomo_dataset_state::BatchRunTomoDatasetState;
use super::batch_run_tomo_status::BatchRunTomoStatus;
use super::const_etomo_number::java_lang_string_matches_whitespace;
use super::ending_step::EndingStep;
use super::enumerated_type::EnumeratedType;
use super::etomo_boolean2::EtomoBoolean2;
use super::etomo_number::EtomoNumber;
use super::field_properties::{FieldProperties, PropInstr};
use super::process_name::ProcessName;
use super::run_status::{self, RunStatus};
use super::status::{Status, StatusRef};
use super::step::Step;
use super::string_property::StringProperty;
use crate::imod::etomo::r#type::const_etomo_number::java_lang_string_trim;
use crate::imod::etomo::util::utilities;

/// Java private static final `GROUP_KEY`.
const GROUP_KEY: &str = "row";
/// Java private static final `ROW_NUMBER_KEY`.
const ROW_NUMBER_KEY: &str = "RowNumber";
/// Java private static final `DATASET_STATE_KEY`.
const DATASET_STATE_KEY: &str = "DatasetStatus";
/// Java private static final `ENDING_STEP_KEY`.
const ENDING_STEP_KEY: &str = "EndingStep";

/// Java private static final `ENDING_STEP_A_KEY = ENDING_STEP_KEY + "." +
/// AxisID.FIRST.getCapitalizedKeyString()`.
fn ending_step_a_key() -> String {
    format!(
        "{}.{}",
        ENDING_STEP_KEY,
        AxisID::First.get_capitalized_key_string()
    )
}

/// The fields of Java `BatchRunTomoRowMetaData`.
pub struct Fields {
    /// Java private final `rowNumber`.
    row_number: EtomoNumber,
    /// Java private final `dualCheckBox`.
    dual_check_box: FieldProperties,
    /// Java private final `bskip`.
    bskip: StringProperty,
    /// Java private final `runCheckBox`.
    run_check_box: FieldProperties,
    /// Java private final `origStack`.
    orig_stack: StringProperty,
    /// Java private final `openDatasetButton`.
    open_dataset_button: FieldProperties,
    /// Java private final `tomogramButton`.
    tomogram_button: FieldProperties,
    /// Java private final `projLogButton`.
    proj_log_button: FieldProperties,
    /// Java private final `brtLogButton`.
    brt_log_button: FieldProperties,
    /// Java private final `imageStackAEditable`.
    image_stack_a_editable: EtomoBoolean2,
    /// Java private final `imageStackBEditable`.
    image_stack_b_editable: EtomoBoolean2,
    /// Java private final `tomogramDone`.
    tomogram_done: EtomoBoolean2,
    /// Java private final `trimvolDone`.
    trimvol_done: EtomoBoolean2,
    /// Java private final `runStatus = new StringProperty(RunStatus.NAME)`.
    run_status: StringProperty,
    /// Java private final `curEndingStep`.
    cur_ending_step: StringProperty,
    /// Java private final `curAxisLetter`.
    cur_axis_letter: StringProperty,
    /// Java private final `stateProperties`.
    state_properties: StateProperties,
    /// Java private `datasetMetaData`, initially null.
    dataset_meta_data: Option<Arc<BatchRunTomoDatasetMetaData>>,
    /// Java private `datasetState`, initially null.
    dataset_state: Option<BatchRunTomoDatasetState>,
    /// Java private `endingStep`, initially null.
    ending_step: Option<EndingStep>,
    /// Java private `endingStepA`, initially null.
    ending_step_a: Option<EndingStep>,
}

/// Java `public final class BatchRunTomoRowMetaData`.
pub struct BatchRunTomoRowMetaData {
    /// Java private final `stackID`.
    stack_id: String,
    fields: Mutex<Fields>,
}

impl BatchRunTomoRowMetaData {
    /// Java package-private `BatchRunTomoRowMetaData(String)`.
    pub fn new(stack_id: &str) -> BatchRunTomoRowMetaData {
        BatchRunTomoRowMetaData {
            stack_id: stack_id.to_owned(),
            fields: Mutex::new(Fields {
                row_number: EtomoNumber::new_with_name(ROW_NUMBER_KEY),
                dual_check_box: FieldProperties::get_toggle_button_instance(
                    "dual",
                    Some(PropInstr::Store),
                    None,
                    Some(PropInstr::Remove),
                ),
                bskip: StringProperty::new_with_key(Some("bskip")),
                run_check_box: FieldProperties::get_toggle_button_instance(
                    "Run",
                    Some(PropInstr::Store),
                    None,
                    Some(PropInstr::Remove),
                ),
                orig_stack: StringProperty::new_with_key(Some("OrigStack")),
                open_dataset_button: FieldProperties::get_button_instance(
                    "Etomo",
                    Some(PropInstr::Store),
                    Some(PropInstr::Remove),
                ),
                tomogram_button: FieldProperties::get_button_instance(
                    "Rec",
                    Some(PropInstr::Store),
                    Some(PropInstr::Remove),
                ),
                proj_log_button: FieldProperties::get_button_instance(
                    "ProjLog",
                    Some(PropInstr::Store),
                    Some(PropInstr::Remove),
                ),
                brt_log_button: FieldProperties::get_button_instance(
                    "Log",
                    Some(PropInstr::Store),
                    Some(PropInstr::Remove),
                ),
                image_stack_a_editable: EtomoBoolean2::new_with_name("ImageStack.A.Editable"),
                image_stack_b_editable: EtomoBoolean2::new_with_name("ImageStack.B.Editable"),
                tomogram_done: EtomoBoolean2::new_with_name("Tomogram.Done"),
                trimvol_done: EtomoBoolean2::new_with_name("Trimvol.Done"),
                run_status: StringProperty::new_with_key(Some(run_status::NAME)),
                cur_ending_step: StringProperty::new_with_key(Some("Cur.EndingStep")),
                cur_axis_letter: StringProperty::new_with_key(Some("Cur.Axis.Letter")),
                state_properties: StateProperties::new(Some(stack_id)),
                dataset_meta_data: None,
                dataset_state: None,
                ending_step: None,
                ending_step_a: None,
            }),
        }
    }

    /// Java static `isRowNumberNull(Properties, String, String)`.  This function is used
    /// to decide whether to load the instance associated with stackID.
    pub fn is_row_number_null_in(
        props: &BTreeMap<String, String>,
        prepend: Option<&str>,
        stack_id: &str,
    ) -> bool {
        let prepend = Self::create_prepend_static(prepend, stack_id);
        let mut number = EtomoNumber::new_with_name(ROW_NUMBER_KEY);
        number.load_with_prepend(props, Some(&prepend));
        number.is_null()
    }

    // <p>Updates done</p>

    /// Java private static `getGroupKey(String)`.
    fn get_group_key_static(stack_id: &str) -> String {
        format!("{}.{}", GROUP_KEY, stack_id)
    }

    /// Java private static `createPrepend(String, String)`.
    fn create_prepend_static(prepend: Option<&str>, stack_id: &str) -> String {
        let Some(prepend) = prepend else {
            return Self::get_group_key_static(stack_id);
        };
        if java_lang_string_matches_whitespace(prepend) {
            return Self::get_group_key_static(stack_id);
        }
        let prepend = java_lang_string_trim(prepend);
        if prepend.ends_with('.') {
            return format!("{}{}", prepend, Self::get_group_key_static(stack_id));
        }
        format!("{}.{}", prepend, Self::get_group_key_static(stack_id))
    }

    /// Java private `createPrepend(String)`.
    fn create_prepend(&self, prepend: Option<&str>) -> String {
        Self::create_prepend_static(prepend, &self.stack_id)
    }

    /// Java `load(Properties, String)`.
    pub fn load(&self, props: &mut BTreeMap<String, String>, prepend: Option<&str>) {
        let mut f = self.fields.lock().unwrap();
        // reset
        f.row_number.reset();
        f.dual_check_box.reset();
        f.bskip.reset();
        f.run_check_box.reset();
        f.dataset_state = None;
        f.ending_step = None;
        f.ending_step_a = None;
        f.cur_ending_step.reset();
        f.cur_axis_letter.reset();
        f.orig_stack.reset();
        f.open_dataset_button.reset();
        f.tomogram_button.reset();
        f.proj_log_button.reset();
        f.brt_log_button.reset();
        f.image_stack_a_editable.reset();
        f.image_stack_b_editable.reset();
        f.tomogram_done.reset();
        f.trimvol_done.reset();
        f.run_status.reset();
        let prepend = self.create_prepend(prepend);
        let group = format!("{}.", prepend);
        let p = Some(prepend.as_str());
        // `StringProperty.load` may remove a backward-compatible key from `props`; none
        // of these declares one.
        f.row_number.load_with_prepend(props, p);
        f.dual_check_box.load(props, p);
        f.bskip.load_with_prepend(Some(&mut *props), p);
        f.run_check_box.load(props, p);
        f.dataset_state = BatchRunTomoDatasetState::get_instance(
            props
                .get(&format!("{}{}", group, DATASET_STATE_KEY))
                .map(String::as_str),
        );
        f.ending_step = EndingStep::get_instance_from_step_value(
            props
                .get(&format!("{}{}", group, ENDING_STEP_KEY))
                .map(String::as_str),
        );
        f.ending_step_a = EndingStep::get_instance_from_step_value(
            props
                .get(&format!("{}{}", group, ending_step_a_key()))
                .map(String::as_str),
        );
        f.cur_ending_step.load_with_prepend(Some(&mut *props), p);
        f.cur_axis_letter.load_with_prepend(Some(&mut *props), p);
        f.orig_stack.load_with_prepend(Some(&mut *props), p);
        f.open_dataset_button.load(props, p);
        f.tomogram_button.load(props, p);
        f.proj_log_button.load(props, p);
        f.brt_log_button.load(props, p);
        f.image_stack_a_editable.load_with_prepend(props, p);
        f.image_stack_b_editable.load_with_prepend(props, p);
        f.tomogram_done.load_with_prepend(props, p);
        f.trimvol_done.load_with_prepend(props, p);
        f.run_status.load_with_prepend(Some(&mut *props), p);
        f.state_properties.load(props, p);
        if BatchRunTomoDatasetMetaData::exists(props, p) {
            if f.dataset_meta_data.is_none() {
                f.dataset_meta_data = Some(Arc::new(BatchRunTomoDatasetMetaData::new()));
            }
            f.dataset_meta_data.as_ref().unwrap().load(props, p);
        }
    }

    /// Java `store(Properties, String)`.
    pub fn store(&self, props: &mut BTreeMap<String, String>, prepend: Option<&str>) {
        let f = self.fields.lock().unwrap();
        let prepend = self.create_prepend(prepend);
        let group = format!("{}.", prepend);
        let p = Some(prepend.as_str());
        f.row_number.store_with_prepend(props, p);
        if !f.row_number.is_null() {
            f.dual_check_box.store(props, p);
            f.bskip.store_with_prepend(Some(props), p);
            f.run_check_box.store(props, p);
            match f.dataset_state {
                Some(dataset_state) => {
                    props.insert(
                        format!("{}{}", group, DATASET_STATE_KEY),
                        dataset_state.get_key().to_owned(),
                    );
                }
                None => {
                    props.remove(&format!("{}{}", group, DATASET_STATE_KEY));
                }
            }
            match f.ending_step {
                Some(ending_step) => {
                    props.insert(
                        format!("{}{}", group, ENDING_STEP_KEY),
                        ending_step.get_value().to_string(),
                    );
                }
                None => {
                    props.remove(&format!("{}{}", group, ENDING_STEP_KEY));
                }
            }
            match f.ending_step_a {
                Some(ending_step_a) => {
                    props.insert(
                        format!("{}{}", group, ending_step_a_key()),
                        ending_step_a.get_value().to_string(),
                    );
                }
                None => {
                    props.remove(&format!("{}{}", group, ending_step_a_key()));
                }
            }
            f.cur_ending_step.store_with_prepend(Some(props), p);
            f.cur_axis_letter.store_with_prepend(Some(props), p);
            f.orig_stack.store_with_prepend(Some(props), p);
            f.open_dataset_button.store(props, p);
            f.tomogram_button.store(props, p);
            f.proj_log_button.store(props, p);
            f.brt_log_button.store(props, p);
            f.image_stack_a_editable.store_with_prepend(props, p);
            f.image_stack_b_editable.store_with_prepend(props, p);
            f.tomogram_done.store_with_prepend(props, p);
            f.trimvol_done.store_with_prepend(props, p);
            f.run_status.store_with_prepend(Some(props), p);
            f.state_properties.store(props, p);
            if let Some(dataset_meta_data) = &f.dataset_meta_data {
                dataset_meta_data.store(props, p);
            }
        }
    }

    /// Java `setTomogramDone(boolean)`.
    pub fn set_tomogram_done(&self, input: bool) {
        self.fields.lock().unwrap().tomogram_done.set_boolean(input);
    }

    /// Java `setTrimvolDone(boolean)`.
    pub fn set_trimvol_done(&self, input: bool) {
        self.fields.lock().unwrap().trimvol_done.set_boolean(input);
    }

    /// Java `isTomogramDone()`.
    pub fn is_tomogram_done(&self) -> bool {
        self.fields.lock().unwrap().tomogram_done.is()
    }

    /// Java `isTrimvolDone()`.
    pub fn is_trimvol_done(&self) -> bool {
        self.fields.lock().unwrap().trimvol_done.is()
    }

    /// Java `setOrigStack(String)`.
    pub fn set_orig_stack(&self, input: Option<&str>) {
        self.fields.lock().unwrap().orig_stack.set(input);
    }

    /// Java `setRunStatus(RunStatus)`.
    pub fn set_run_status(&self, input: Option<RunStatus>) {
        let mut f = self.fields.lock().unwrap();
        match input {
            Some(input) => f.run_status.set(Some(input.value())),
            None => f.run_status.reset(),
        }
    }

    /// Java `getRunStatus()`.
    pub fn get_run_status(&self) -> Option<RunStatus> {
        RunStatus::get_instance(Some(&self.fields.lock().unwrap().run_status.to_string()))
    }

    /// Java `setEndingStep(EndingStep)`.
    pub fn set_ending_step(&self, input: Option<EndingStep>) {
        self.fields.lock().unwrap().ending_step = input;
    }

    /// Java `setEndingStepA(EndingStep)`.
    pub fn set_ending_step_a(&self, input: Option<EndingStep>) {
        self.fields.lock().unwrap().ending_step_a = input;
    }

    /// Java `setCurEndingStep(String)`.
    pub fn set_cur_ending_step(&self, cur_ending_step: Option<&str>) {
        self.fields
            .lock()
            .unwrap()
            .cur_ending_step
            .set(cur_ending_step);
    }

    /// Java `setCurAxisLetter(String)`.
    pub fn set_cur_axis_letter(&self, cur_axis_letter: Option<&str>) {
        self.fields
            .lock()
            .unwrap()
            .cur_axis_letter
            .set(cur_axis_letter);
    }

    /// Java `setDatasetState(BatchRunTomoDatasetState)`.
    pub fn set_dataset_state(&self, input: Option<BatchRunTomoDatasetState>) {
        self.fields.lock().unwrap().dataset_state = input;
    }

    /// Java `getOrigStack()`.
    pub fn get_orig_stack(&self) -> String {
        self.fields.lock().unwrap().orig_stack.to_string()
    }

    /// Java `getEndingStep()`.
    pub fn get_ending_step(&self) -> Option<EndingStep> {
        self.fields.lock().unwrap().ending_step
    }

    /// Java `getEndingStepA()`.
    pub fn get_ending_step_a(&self) -> Option<EndingStep> {
        self.fields.lock().unwrap().ending_step_a
    }

    /// Java `getCurEndingStep()`.
    pub fn get_cur_ending_step(&self) -> String {
        self.fields.lock().unwrap().cur_ending_step.to_string()
    }

    /// Java `getCurAxisLetter()`.
    pub fn get_cur_axis_letter(&self) -> String {
        self.fields.lock().unwrap().cur_axis_letter.to_string()
    }

    /// Java `getDatasetState()`.
    pub fn get_dataset_state(&self) -> Option<BatchRunTomoDatasetState> {
        self.fields.lock().unwrap().dataset_state
    }

    /// Java `getStackID()`.
    pub fn get_stack_id(&self) -> &str {
        &self.stack_id
    }

    /// Java `setDatasetDialog(boolean)`.
    pub fn set_dataset_dialog(&self, input: bool) {
        let mut f = self.fields.lock().unwrap();
        if input && f.dataset_meta_data.is_none() {
            f.dataset_meta_data = Some(Arc::new(BatchRunTomoDatasetMetaData::new()));
        }
        if let Some(dataset_meta_data) = &f.dataset_meta_data {
            dataset_meta_data.set_dataset(input);
        }
    }

    /// Java `isDatasetDialog()`.
    pub fn is_dataset_dialog(&self) -> bool {
        self.fields.lock().unwrap().dataset_meta_data.is_some()
    }

    /// Java `getDatasetMetaData()`.
    pub fn get_dataset_meta_data(&self) -> Arc<BatchRunTomoDatasetMetaData> {
        let mut f = self.fields.lock().unwrap();
        if f.dataset_meta_data.is_none() {
            f.dataset_meta_data = Some(Arc::new(BatchRunTomoDatasetMetaData::new()));
        }
        f.dataset_meta_data.clone().unwrap()
    }

    /// Java `setRowNumber(String)`.
    pub fn set_row_number(&self, input: Option<&str>) {
        self.fields.lock().unwrap().row_number.set_string(input);
    }

    /// Java `getRowNumber()`.
    pub fn get_row_number(&self) -> i32 {
        self.fields.lock().unwrap().row_number.get_int()
    }

    /// Java `isRowNumberNull()`.
    pub fn is_row_number_null(&self) -> bool {
        self.fields.lock().unwrap().row_number.is_null()
    }

    /// Java `getDualCheckBox()`: the field, handed to `f`.
    pub fn get_dual_check_box<R>(&self, f: impl FnOnce(&mut FieldProperties) -> R) -> R {
        f(&mut self.fields.lock().unwrap().dual_check_box)
    }

    /// Java `getRunCheckBox()`: the field, handed to `f`.
    pub fn get_run_check_box<R>(&self, f: impl FnOnce(&mut FieldProperties) -> R) -> R {
        f(&mut self.fields.lock().unwrap().run_check_box)
    }

    /// Java `getOpenDatasetButton()`: the field, handed to `f`.
    pub fn get_open_dataset_button<R>(&self, f: impl FnOnce(&mut FieldProperties) -> R) -> R {
        f(&mut self.fields.lock().unwrap().open_dataset_button)
    }

    /// Java `getTomogramButton()`: the field, handed to `f`.
    pub fn get_tomogram_button<R>(&self, f: impl FnOnce(&mut FieldProperties) -> R) -> R {
        f(&mut self.fields.lock().unwrap().tomogram_button)
    }

    /// Java `getProjLogButton()`: the field, handed to `f`.
    pub fn get_proj_log_button<R>(&self, f: impl FnOnce(&mut FieldProperties) -> R) -> R {
        f(&mut self.fields.lock().unwrap().proj_log_button)
    }

    /// Java `getBRTLogButton()`: the field, handed to `f`.
    pub fn get_brt_log_button<R>(&self, f: impl FnOnce(&mut FieldProperties) -> R) -> R {
        f(&mut self.fields.lock().unwrap().brt_log_button)
    }

    /// Java `isImageStackAEditable()`.
    pub fn is_image_stack_a_editable(&self) -> bool {
        self.fields.lock().unwrap().image_stack_a_editable.is()
    }

    /// Java `isImageStackBEditable()`.
    pub fn is_image_stack_b_editable(&self) -> bool {
        self.fields.lock().unwrap().image_stack_b_editable.is()
    }

    /// Java `setImageStackAEditable(boolean)`.
    pub fn set_image_stack_a_editable(&self, input: bool) {
        self.fields
            .lock()
            .unwrap()
            .image_stack_a_editable
            .set_boolean(input);
    }

    /// Java `setImageStackBEditable(boolean)`.
    pub fn set_image_stack_b_editable(&self, input: bool) {
        self.fields
            .lock()
            .unwrap()
            .image_stack_b_editable
            .set_boolean(input);
    }

    /// Java `setBskip(String)`.
    pub fn set_bskip(&self, input: Option<&str>) {
        self.fields.lock().unwrap().bskip.set(input);
    }

    /// Java `getRowProperties()`: the state properties, handed to `f`.
    pub fn get_row_properties<R>(&self, f: impl FnOnce(&mut StateProperties) -> R) -> R {
        f(&mut self.fields.lock().unwrap().state_properties)
    }

    /// Java `getBskip()`.
    pub fn get_bskip(&self) -> String {
        self.fields.lock().unwrap().bskip.to_string()
    }
}

/// Java `toString()`: `rowNumber.toString()`.
impl std::fmt::Display for BatchRunTomoRowMetaData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.fields.lock().unwrap().row_number)
    }
}

/// Java private static final `StateProperties.GROUP_KEY`.
const STATE_GROUP_KEY: &str = "StateProperties";
/// Java private static final `StateProperties.DATASET_STATE_KEY`.
const STATE_DATASET_STATE_KEY: &str = "DatasetState";
/// Java private static final `StateProperties.STEP_KEY`.
const STEP_KEY: &str = "Step";
/// Java private static final `StateProperties.ENDING_STEP_KEY`.
const STATE_ENDING_STEP_KEY: &str = "EndingStep";
/// Java private static final `StateProperties.PROCESS_NAME_KEY`.
const PROCESS_NAME_KEY: &str = "ProcessName";

/// Java `public static final class StateProperties`.
pub struct StateProperties {
    /// Java private final `status` (BatchRunTomoStatus).
    status: StringProperty,
    /// Java private final `datasetState` (BatchRunTomoDatasetState).
    dataset_state: StringProperty,
    /// Java private final `datasetStateInit`.
    dataset_state_init: EtomoBoolean2,
    /// Java private final `step`.
    step: EtomoNumber,
    /// Java private final `endingStep`.
    ending_step: EtomoNumber,
    /// Java private final `endingStepAxisID`.
    ending_step_axis_id: StringProperty,
    /// Java private final `processName`.
    process_name: StringProperty,
    /// Java private final `curReconStep`.
    cur_recon_step: StringProperty,
    /// Java private `print`, initially false.
    print: bool,
    /// Java private `printMsg`, initially null.
    print_msg: Option<String>,
}

impl StateProperties {
    /// Java private `StateProperties(String)`.
    fn new(stack_id: Option<&str>) -> StateProperties {
        StateProperties {
            status: StringProperty::new_with_key(Some("Status")),
            dataset_state: StringProperty::new_with_key(Some(STATE_DATASET_STATE_KEY)),
            dataset_state_init: EtomoBoolean2::new_with_name(&format!(
                "{}.Init",
                STATE_DATASET_STATE_KEY
            )),
            step: EtomoNumber::new_with_name(STEP_KEY),
            ending_step: EtomoNumber::new_with_name(STATE_ENDING_STEP_KEY),
            ending_step_axis_id: StringProperty::new_with_key(Some(&format!(
                "{}.AxisID",
                STATE_ENDING_STEP_KEY
            ))),
            process_name: StringProperty::new_with_key(Some(PROCESS_NAME_KEY)),
            cur_recon_step: StringProperty::new_with_key(Some("Cur.Recon.Step")),
            print: stack_id == Some("ebt1"),
            print_msg: None,
        }
    }

    /// Java `getStatus()`.
    pub fn get_status(&self) -> String {
        self.status.to_string()
    }

    /// Java `getDatasetState()`.
    pub fn get_dataset_state(&self) -> String {
        self.dataset_state.to_string()
    }

    /// Java `getDatasetStateInit()`.
    pub fn get_dataset_state_init(&self) -> Option<bool> {
        if self.dataset_state_init.is_null() {
            return None;
        }
        Some(self.dataset_state_init.is())
    }

    /// Java `getStep()`.
    pub fn get_step(&self) -> String {
        self.step.to_string()
    }

    /// Java `getEndingStep()`.
    pub fn get_ending_step(&self) -> i32 {
        self.ending_step.get_int()
    }

    /// Java `getEndingStepAxisID()`.
    pub fn get_ending_step_axis_id(&self) -> String {
        self.ending_step_axis_id.to_string()
    }

    /// Java `getProcessName()`.
    pub fn get_process_name(&self) -> String {
        self.process_name.to_string()
    }

    /// Java `setStatus(BatchRunTomoStatus)`.
    pub fn set_status(&mut self, input: Option<BatchRunTomoStatus>) {
        match input {
            Some(input) => self.status.set(input.get_text()),
            None => self.status.reset(),
        }
    }

    /// Java `setDatasetState(BatchRunTomoDatasetState)`.
    pub fn set_dataset_state(&mut self, input: Option<BatchRunTomoDatasetState>) {
        match input {
            Some(input) => self.dataset_state.set(Some(input.get_key())),
            None => self.dataset_state.reset(),
        }
    }

    /// Java `setDatasetStateInit(boolean)`.
    pub fn set_dataset_state_init(&mut self, input: bool) {
        self.dataset_state_init.set_boolean(input);
    }

    /// Java `setCurReconStep(Status)`.
    pub fn set_cur_recon_step(&mut self, input: Option<StatusRef>) {
        match input {
            None => self.cur_recon_step.reset(),
            Some(StatusRef::Step(_)) => self.cur_recon_step.set(Some(STEP_KEY)),
            Some(StatusRef::EndingStep(_)) => self.cur_recon_step.set(Some(STATE_ENDING_STEP_KEY)),
            Some(StatusRef::ProcessName(_)) => self.cur_recon_step.set(Some(PROCESS_NAME_KEY)),
            Some(_) => self.cur_recon_step.reset(),
        }
    }

    /// Java `isStepCurrent()`.
    pub fn is_step_current(&self) -> bool {
        self.cur_recon_step.equals(Some(STEP_KEY))
    }

    /// Java `isEndingStepCurrent()`.
    pub fn is_ending_step_current(&self) -> bool {
        self.cur_recon_step.equals(Some(STATE_ENDING_STEP_KEY))
    }

    /// Java `isProcessNameCurrent()`.
    pub fn is_process_name_current(&self) -> bool {
        self.cur_recon_step.equals(Some(PROCESS_NAME_KEY))
    }

    /// Java `setStep(Step)`.
    pub fn set_step(&mut self, input: Option<Step>) {
        match input {
            None => {
                self.step.reset();
            }
            Some(input) => {
                let value = input.get_value();
                self.step.set_const_etomo_number(Some(&value));
            }
        }
    }

    /// Java `setEndingStep(EndingStep)`.
    pub fn set_ending_step(&mut self, input: Option<EndingStep>) {
        match input {
            Some(input) => {
                self.ending_step.set_int(input.get_index());
            }
            None => {
                self.ending_step.reset();
            }
        }
    }

    /// Java `setEndingStepAxisID(AxisID)`.
    pub fn set_ending_step_axis_id(&mut self, input: Option<AxisID>) {
        match input {
            Some(input) => self.ending_step_axis_id.set(Some(input.get_key())),
            None => self.ending_step_axis_id.reset(),
        }
    }

    /// Java `setProcessName(ProcessName)`.
    pub fn set_process_name(&mut self, input: Option<ProcessName>) {
        match input {
            None => self.process_name.reset(),
            Some(input) => self.process_name.set(Some(&input.to_string())),
        }
    }

    /// Java private static `getGroupKey()`.
    fn get_group_key() -> &'static str {
        STATE_GROUP_KEY
    }

    /// Java private static `createPrepend(String)`.
    fn create_prepend(prepend: Option<&str>) -> String {
        if utilities::is_empty(prepend) {
            return Self::get_group_key().to_owned();
        }
        let prepend = java_lang_string_trim(prepend.unwrap());
        if prepend.ends_with('.') {
            return format!("{}{}", prepend, Self::get_group_key());
        }
        format!("{}.{}", prepend, Self::get_group_key())
    }

    /// Java private `load(Properties, String)`.
    fn load(&mut self, props: &mut BTreeMap<String, String>, prepend: Option<&str>) {
        // reset
        self.status.reset();
        self.dataset_state.reset();
        self.dataset_state_init.reset();
        self.step.reset();
        self.ending_step.reset();
        self.ending_step_axis_id.reset();
        self.process_name.reset();
        self.cur_recon_step.reset();

        // load
        let prepend = Self::create_prepend(prepend);
        let p = Some(prepend.as_str());
        self.status.load_with_prepend(Some(&mut *props), p);
        self.dataset_state.load_with_prepend(Some(&mut *props), p);
        self.dataset_state_init.load_with_prepend(props, p);
        self.step.load_with_prepend(props, p);
        self.ending_step.load_with_prepend(props, p);
        self.ending_step_axis_id
            .load_with_prepend(Some(&mut *props), p);
        self.process_name.load_with_prepend(Some(&mut *props), p);
        self.cur_recon_step.load_with_prepend(Some(&mut *props), p);
    }

    /// Java private `store(Properties, String)`.
    fn store(&self, props: &mut BTreeMap<String, String>, prepend: Option<&str>) {
        let prepend = Self::create_prepend(prepend);
        let p = Some(prepend.as_str());
        self.status.store_with_prepend(Some(props), p);
        self.dataset_state.store_with_prepend(Some(props), p);
        self.dataset_state_init.store_with_prepend(props, p);
        self.step.store_with_prepend(props, p);
        self.ending_step.store_with_prepend(props, p);
        self.ending_step_axis_id.store_with_prepend(Some(props), p);
        self.process_name.store_with_prepend(Some(props), p);
        self.cur_recon_step.store_with_prepend(Some(props), p);
    }
}
