//! `IMOD/Etomo/src/etomo/comscript/TomodataplotsParam.java`.
//!
//! The nested typesafe enums `TypeOfDataToPlot` and `Task` are Rust enums: each Java
//! instance is a variant, and each private final field is a `match` over the variants
//! giving the value its constructor stored.

use std::sync::Arc;

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::task_interface::TaskInterface;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::process_name::ProcessName;

/// Java final `TomodataplotsParam`.
#[derive(Clone, Debug, Default)]
pub struct TomodataplotsParam {
    task: Option<Task>,
    alternative_input_file_absolute_path: Option<String>,
}

impl TomodataplotsParam {
    /// Java's implicit `TomodataplotsParam()`.
    pub fn new() -> TomodataplotsParam {
        TomodataplotsParam::default()
    }

    /// Java `setTask(TaskInterface)`.
    pub fn set_task(&mut self, input: Option<&dyn TaskInterface>) {
        match input.and_then(|input| (input as &dyn std::any::Any).downcast_ref::<Task>()) {
            Some(task) => self.task = Some(*task),
            None => self.task = None,
        }
    }

    /// Java `setAlternativeInputFileAbsolutePath(String)`.
    pub fn set_alternative_input_file_absolute_path(
        &mut self,
        alternative_input_file_absolute_path: Option<&str>,
    ) {
        self.alternative_input_file_absolute_path =
            alternative_input_file_absolute_path.map(str::to_owned);
    }

    /// Java `getCommandArray(BaseManager, AxisID)`.
    pub fn get_command_array(
        &self,
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
    ) -> Vec<String> {
        let mut command: Vec<String> = Vec::new();
        command.push("python".to_owned());
        command.push("-u".to_owned());
        // Java string concatenation writes a null path as "null"
        let script_path = etomo_director::INSTANCE
            .get_python_script_path()
            .as_deref()
            .unwrap_or("null")
            .to_owned();
        command.push(format!("{script_path}{}", ProcessName::TOMODATAPLOTS));
        if let Some(task) = self.task {
            if let Some(type_of_data_to_plot) = task.type_of_data_to_plot() {
                command.push("-TypeOfDataToPlot".to_owned());
                command.push(type_of_data_to_plot.value().to_owned());
            }
            if let Some(alternative_input_file_absolute_path) =
                &self.alternative_input_file_absolute_path
            {
                command.push("-InputFile".to_owned());
                command.push(alternative_input_file_absolute_path.clone());
            } else if let Some(input_file) = task.input_file() {
                // A null file name is a null list element in Java, which
                // `ProcessBuilder` rejects; here the option is left out.
                if let Some(file_name) = input_file.get_file_name(Some(manager), Some(axis_id)) {
                    command.push("-InputFile".to_owned());
                    command.push(file_name);
                }
            }
            if let Some(xaxis_label) = task.xaxis_label() {
                command.push("-XaxisLabel".to_owned());
                command.push(xaxis_label.to_owned());
            }
        }
        command
    }
}

/// Java private static final nested `TypeOfDataToPlot`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TypeOfDataToPlot {
    MeanMax,
    Rotation,
    TiltSkew,
    Mag,
    Xstretch,
    Resid,
    AverResid,
    MinMax,
    Shifts,
    MeanResiduals,
    MaxOfMaxResiduals,
}

impl TypeOfDataToPlot {
    /// Java private final field `value`.
    fn value(self) -> &'static str {
        match self {
            TypeOfDataToPlot::MeanMax => "4",
            TypeOfDataToPlot::Rotation => "5",
            TypeOfDataToPlot::TiltSkew => "6",
            TypeOfDataToPlot::Mag => "7",
            TypeOfDataToPlot::Xstretch => "8",
            TypeOfDataToPlot::Resid => "9",
            TypeOfDataToPlot::AverResid => "10",
            TypeOfDataToPlot::MinMax => "13",
            TypeOfDataToPlot::Shifts => "20",
            TypeOfDataToPlot::MeanResiduals => "21",
            TypeOfDataToPlot::MaxOfMaxResiduals => "22",
        }
    }
}

/// Java public static final nested `Task implements TaskInterface`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Task {
    /// Java `MIN_MAX`.
    MinMax,
    /// Java `FIXED_MIN_MAX`.
    FixedMinMax,
    /// Java `COARSE_MEAN_MAX`.
    CoarseMeanMax,
    /// Java `ROTATION`.
    Rotation,
    /// Java `TILT_SKEW`.
    TiltSkew,
    /// Java `MAG`.
    Mag,
    /// Java `XSTRETCH`.
    Xstretch,
    /// Java `RESID`.
    Resid,
    /// Java `AVER_RESID`.
    AverResid,
    /// Java `SERIAL_SECTIONS_MEAN_MAX`.
    SerialSectionsMeanMax,
    /// Java `ALIGN_FRAMES_SHIFTS`.
    AlignFramesShifts,
    /// Java `ALIGN_FRAMES_MEAN_RESIDUALS`.
    AlignFramesMeanResiduals,
    /// Java `ALIGN_FRAMES_MAXOFMAX_RESIDUALS`.
    AlignFramesMaxofmaxResiduals,
}

impl Task {
    /// Java private final field `label`.
    fn label(self) -> &'static str {
        match self {
            Task::MinMax => "Plot min/max",
            Task::FixedMinMax => "Plot fixed min/max",
            Task::CoarseMeanMax => "Plot edge errors",
            Task::Rotation => "Plot rotation",
            Task::TiltSkew => "Plot delta tilt and skew",
            Task::Mag => "Plot magnification",
            Task::Xstretch => "Plot X-stretch (dmag)",
            Task::Resid => "Plot global mean residual",
            Task::AverResid => "Plot average of local mean residual",
            Task::SerialSectionsMeanMax => "Plot edge errors",
            Task::AlignFramesShifts => "Plot shifts",
            Task::AlignFramesMeanResiduals => "Plot mean residuals",
            Task::AlignFramesMaxofmaxResiduals => "Plot max of max residuals",
        }
    }

    /// Java private final field `typeOfDataToPlot`.
    fn type_of_data_to_plot(self) -> Option<TypeOfDataToPlot> {
        Some(match self {
            Task::MinMax => TypeOfDataToPlot::MinMax,
            Task::FixedMinMax => TypeOfDataToPlot::MinMax,
            Task::CoarseMeanMax => TypeOfDataToPlot::MeanMax,
            Task::Rotation => TypeOfDataToPlot::Rotation,
            Task::TiltSkew => TypeOfDataToPlot::TiltSkew,
            Task::Mag => TypeOfDataToPlot::Mag,
            Task::Xstretch => TypeOfDataToPlot::Xstretch,
            Task::Resid => TypeOfDataToPlot::Resid,
            Task::AverResid => TypeOfDataToPlot::AverResid,
            Task::SerialSectionsMeanMax => TypeOfDataToPlot::MeanMax,
            Task::AlignFramesShifts => TypeOfDataToPlot::Shifts,
            Task::AlignFramesMeanResiduals => TypeOfDataToPlot::MeanResiduals,
            Task::AlignFramesMaxofmaxResiduals => TypeOfDataToPlot::MaxOfMaxResiduals,
        })
    }

    /// Java private final field `inputFile`.
    fn input_file(self) -> Option<&'static Arc<FileType>> {
        let class = &*file_type::CLASS;
        Some(match self {
            Task::MinMax => &class.stats_log,
            Task::FixedMinMax => &class.fixed_stats_log,
            Task::CoarseMeanMax => &class.cross_correlation_log,
            Task::Rotation
            | Task::TiltSkew
            | Task::Mag
            | Task::Xstretch
            | Task::Resid
            | Task::AverResid => &class.align_solution_log,
            Task::SerialSectionsMeanMax => &class.preblend_log,
            Task::AlignFramesShifts
            | Task::AlignFramesMeanResiduals
            | Task::AlignFramesMaxofmaxResiduals => &class.align_frames_log,
        })
    }

    /// Java private final field `xaxisLabel`: set only by the four-argument
    /// constructor.
    fn xaxis_label(self) -> Option<&'static str> {
        match self {
            Task::MinMax => Some("View number in raw stack "),
            Task::FixedMinMax => Some("View number in fixed stack "),
            _ => None,
        }
    }

    /// Java `isAvailable(BaseManager, AxisID)`.
    pub fn is_available(self, manager: &'static dyn BaseManager, axis_id: AxisID) -> bool {
        match self.input_file() {
            Some(input_file) => input_file.exists(Some(manager), Some(axis_id)),
            None => false,
        }
    }
}

impl TaskInterface for Task {
    /// Java `getDescr`.
    fn get_descr(&self) -> Option<String> {
        Some(self.label().to_owned())
    }

    /// Java `okToDrop`.
    fn ok_to_drop(&self) -> bool {
        true
    }
}

/// Java `toString`.
impl std::fmt::Display for Task {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}
