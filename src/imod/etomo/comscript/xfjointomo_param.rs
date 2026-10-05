//! `IMOD/Etomo/src/etomo/comscript/XfjointomoParam.java`.
//!
//! The `xfjointomo` command line (Model tab, "Find Transformations"), built from the
//! join state of the join or trial join the refining model was made on.

use crate::imod::etomo::join_manager::JoinManager;
use crate::imod::etomo::r#type::const_etomo_number::ConstEtomoNumber;
use crate::imod::etomo::r#type::const_join_state::ConstJoinState;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::transform::Transform;
use crate::imod::etomo::util::dataset_files;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `public static final String BOUNDARIES_TO_ANALYZE_KEY`.
pub const BOUNDARIES_TO_ANALYZE_KEY: &str = "BoundariesToAnalyze";
/// Java `public static final String OBJECTS_TO_INCLUDE`.
pub const OBJECTS_TO_INCLUDE: &str = "ObjectsToInclude";
/// Java `public static final String GAP_START_END_INC`.
pub const GAP_START_END_INC: &str = "GapStartEndInc";
/// Java `public static final String POINTS_TO_FIT`.
pub const POINTS_TO_FIT: &str = "PointsToFit";

/// Java package-private static final `COMMAND_NAME = ProcessName.XFJOINTOMO.toString()`.
pub(crate) fn command_name() -> String {
    ProcessName::XFJOINTOMO.to_string()
}

/// Java private static final `debug`.
const DEBUG: bool = false;

/// Java `public final class XfjointomoParam`.
pub struct XfjointomoParam {
    /// Java private final `manager`.
    manager: &'static JoinManager,
    /// Java private `transform`, initially null.
    transform: Option<Transform>,
    /// Java private `boundariesToAnalyze`, initially null.
    boundaries_to_analyze: Option<String>,
    /// Java private `pointsToFit`, initially null.
    points_to_fit: Option<String>,
    /// Java private `gapStartEndInc`, initially null.
    gap_start_end_inc: Option<String>,
    /// Java private `objectsToInclude`, initially null.
    objects_to_include: Option<String>,
    /// Java private `commandArray`, initially null.
    command_array: Option<Vec<String>>,
    /// Java private final `trial`.
    trial: bool,
}

impl XfjointomoParam {
    /// Java `XfjointomoParam(JoinManager, boolean)`.
    pub fn new(manager: &'static JoinManager, trial: bool) -> XfjointomoParam {
        XfjointomoParam {
            manager,
            transform: None,
            boundaries_to_analyze: None,
            points_to_fit: None,
            gap_start_end_inc: None,
            objects_to_include: None,
            command_array: None,
            trial,
        }
    }

    /// Java `getCommandArray()`.
    pub fn get_command_array(&mut self) -> Vec<String> {
        self.create_command_array();
        self.command_array.clone().unwrap_or_default()
    }

    /// Java private `createCommandArray()`.
    fn create_command_array(&mut self) {
        if self.command_array.is_some() {
            return;
        }
        let options = self.gen_options();
        let mut command_array = vec![String::new(); options.len() + 1];
        command_array[0] = command_name();
        let mut index = 1;
        for option in &options {
            command_array[index] = option.clone();
            index += 1;
        }
        if DEBUG {
            let mut buffer = String::new();
            for i in 0..command_array.len() {
                buffer.push_str(&command_array[i]);
                if i < command_array.len() - 1 {
                    buffer.push(' ');
                }
            }
            eprintln!("{buffer}");
        }
        self.command_array = Some(command_array);
    }

    /// Java `setBoundariesToAnalyze(String)`.
    pub fn set_boundaries_to_analyze(&mut self, boundaries_to_analyze: Option<&str>) {
        self.boundaries_to_analyze = boundaries_to_analyze.map(str::to_owned);
    }

    /// Java `setObjectsToInclude(String)`.
    pub fn set_objects_to_include(&mut self, objects_to_include: Option<&str>) {
        self.objects_to_include = objects_to_include.map(str::to_owned);
    }

    /// Java `setGapStartEndInc(String, String, String)`.
    pub fn set_gap_start_end_inc(
        &mut self,
        start: Option<&str>,
        end: Option<&str>,
        inc: Option<&str>,
    ) {
        if XfjointomoParam::is_null(start)
            || XfjointomoParam::is_null(end)
            || XfjointomoParam::is_null(inc)
        {
            self.gap_start_end_inc = None;
        } else {
            self.gap_start_end_inc = Some(format!(
                "{},{},{}",
                start.unwrap(),
                end.unwrap(),
                inc.unwrap()
            ));
        }
    }

    /// Java `setTransform(Transform)`.
    pub fn set_transform(&mut self, transform: Option<Transform>) {
        self.transform = transform;
    }

    /// Java `setPointsToFit(String, String)`.
    pub fn set_points_to_fit(&mut self, min: Option<&str>, max: Option<&str>) {
        if XfjointomoParam::is_null(min) || XfjointomoParam::is_null(max) {
            self.points_to_fit = None;
        } else {
            self.points_to_fit = Some(format!("{},{}", min.unwrap(), max.unwrap()));
        }
    }

    /// Java private `genOptions()`.
    fn gen_options(&self) -> Vec<String> {
        let mut command: Vec<String> = Vec::new();
        command.push("-InputFile".to_string());
        command.push(dataset_files::get_refine_model_file_name(self.manager));
        command.push("-FOutputFile".to_string());
        command.push(dataset_files::get_refine_xf_file_name(self.manager));
        command.push("-GOutputFile".to_string());
        command.push(dataset_files::get_refine_join_xg_file_name(self.manager));
        command.push("-EditExistingFile".to_string());
        command.push("-SizesOfSections".to_string());
        let state = self.manager.get_state();
        let mut sizes_of_sections = String::new();
        let mut start_list_walker = state.get_join_start_list_walker(self.trial);
        let mut end_list_walker = state.get_join_end_list_walker(self.trial);
        // check for valid lists
        if start_list_walker.size() == end_list_walker.size() {
            while start_list_walker.has_next() {
                // Fixed in translation: a key missing from the end list makes the
                // source dereference a null number (`start.gt(end)`); it counts as an
                // empty number here.
                let start: ConstEtomoNumber = start_list_walker
                    .next_etomo_number()
                    .map(|number| number.base)
                    .unwrap_or_else(ConstEtomoNumber::new);
                let end: ConstEtomoNumber = end_list_walker
                    .next_etomo_number()
                    .map(|number| number.base)
                    .unwrap_or_else(ConstEtomoNumber::new);
                if start.gt_const_etomo_number(Some(&end)) {
                    sizes_of_sections.push_str(
                        &start
                            .get_int()
                            .wrapping_sub(end.get_int())
                            .wrapping_add(1)
                            .to_string(),
                    );
                } else {
                    sizes_of_sections.push_str(
                        &end.get_int()
                            .wrapping_sub(start.get_int())
                            .wrapping_add(1)
                            .to_string(),
                    );
                }
                if start_list_walker.has_next() {
                    sizes_of_sections.push(',');
                }
            }
            command.push(sizes_of_sections);
        }
        command.push("-OffsetOfJoin".to_string());
        command.push(format!(
            "{},{}",
            XfjointomoParam::calc_offset(&state.get_join_shift_in_x(self.trial)),
            XfjointomoParam::calc_offset(&state.get_join_shift_in_y(self.trial))
        ));
        if self.trial {
            command.push("-BinningOfJoin".to_string());
            command.push(state.get_join_trial_binning().to_string());
        }
        if !state.get_join_alignment_ref_section(self.trial).is_null() {
            command.push("-ReferenceSection".to_string());
            command.push(state.get_join_alignment_ref_section(self.trial).to_string());
        }
        if self.transform == Some(Transform::Translation) {
            command.push("-TranslationOnly".to_string());
        } else if self.transform == Some(Transform::RotationTranslation) {
            command.push("-RotationTranslation".to_string());
        } else if self.transform == Some(Transform::RotationTranslationMagnification) {
            command.push("-MagRotTrans".to_string());
        }
        if !XfjointomoParam::is_null(self.boundaries_to_analyze.as_deref()) {
            command.push(format!("-{}", BOUNDARIES_TO_ANALYZE_KEY));
            command.push(self.boundaries_to_analyze.clone().unwrap());
        }
        if !XfjointomoParam::is_null(self.points_to_fit.as_deref()) {
            command.push(format!("-{}", POINTS_TO_FIT));
            command.push(self.points_to_fit.clone().unwrap());
        }
        if !XfjointomoParam::is_null(self.gap_start_end_inc.as_deref()) {
            command.push(format!("-{}", GAP_START_END_INC));
            command.push(self.gap_start_end_inc.clone().unwrap());
        }
        if !XfjointomoParam::is_null(self.objects_to_include.as_deref()) {
            command.push(format!("-{}", OBJECTS_TO_INCLUDE));
            command.push(self.objects_to_include.clone().unwrap());
        }
        command
    }

    /// Java private `isNull(String)`.
    fn is_null(string: Option<&str>) -> bool {
        match string {
            None => true,
            Some(string) => {
                string.is_empty()
                    || string
                        .chars()
                        .all(|c| matches!(c, ' ' | '\t' | '\n' | '\x0B' | '\x0C' | '\r'))
            }
        }
    }

    /// Java private `calcOffset(ConstEtomoNumber)`.
    fn calc_offset(shift: &ConstEtomoNumber) -> String {
        shift.get_int().wrapping_mul(-1).to_string()
    }
}
