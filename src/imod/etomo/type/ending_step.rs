//! `IMOD/Etomo/src/etomo/type/EndingStep.java`.
//!
//! The batchruntomo ending steps offered by the batch interface.  The Java
//! singletons are an enum.

use super::const_etomo_number::ConstEtomoNumber;
use super::enumerated_type::EnumeratedType;
use super::status::Status;
use super::step::Step;

/// Java `public final class EndingStep implements EnumeratedType, Status`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum EndingStep {
    /// Java `BEAD_TRACKING = new EndingStep(0, Step.BEAD_TRACKING, ...)`.
    BeadTracking,
    /// Java private `FINE_ALIGNMENT = new EndingStep(1, Step.FINE_ALIGNMENT, ...)`.
    FineAlignment,
    /// Java `POSITIONING = new EndingStep(2, Step.POSITIONING, ...)`.
    Positioning,
    /// Java private `GOLD_DETECTION_3D = new EndingStep(3, Step.GOLD_DETECTION_3D, ...)`.
    GoldDetection3d,
    /// Java private `TWO_D_FILTERING = new EndingStep(4, Step.TWO_D_FILTERING, ...)`.
    TwoDFiltering,
}

// <p>Updates done.</p>

/// Java `MAX = TWO_D_FILTERING`.
pub const MAX: EndingStep = EndingStep::TwoDFiltering;

/// The instances in declaration order.
const DECLARED: [EndingStep; 5] = [
    EndingStep::BeadTracking,
    EndingStep::FineAlignment,
    EndingStep::Positioning,
    EndingStep::GoldDetection3d,
    EndingStep::TwoDFiltering,
];

impl EndingStep {
    /// Java private final `index`.
    fn index(self) -> i32 {
        match self {
            EndingStep::BeadTracking => 0,
            EndingStep::FineAlignment => 1,
            EndingStep::Positioning => 2,
            EndingStep::GoldDetection3d => 3,
            EndingStep::TwoDFiltering => 4,
        }
    }

    /// Java private final `step`.
    fn step(self) -> Step {
        match self {
            EndingStep::BeadTracking => Step::BEAD_TRACKING,
            EndingStep::FineAlignment => Step::FINE_ALIGNMENT,
            EndingStep::Positioning => Step::POSITIONING,
            EndingStep::GoldDetection3d => Step::GOLD_DETECTION_3D,
            EndingStep::TwoDFiltering => Step::TWO_D_FILTERING,
        }
    }

    /// Java package-private final `tooltip`.
    fn tooltip(self) -> &'static str {
        match self {
            EndingStep::BeadTracking => {
                "Stop after fiducial model generation or patch tracking."
            }
            EndingStep::FineAlignment => "Stop after fine alignment with Tiltalign.",
            EndingStep::Positioning => "Stop after tomogram positioning (if any).",
            EndingStep::GoldDetection3d => {
                "Stop after CTF estimation and detection of gold in 3D (if any)."
            }
            EndingStep::TwoDFiltering => {
                "Stop when all steps on the aligned stack are completed."
            }
        }
    }

    /// Java static `getInstance(String)`.
    pub fn get_instance_from_step_value(step_value: Option<&str>) -> Option<EndingStep> {
        Self::get_instance_from_step(Step::get_instance(step_value))
    }

    /// Java static `getInstanceFromText(String)`.
    pub fn get_instance_from_text(step_text: Option<&str>) -> Option<EndingStep> {
        Self::get_instance_from_step(Step::get_instance_from_text(step_text))
    }

    /// Java static `getInstance(ConstEtomoNumber)`.
    pub fn get_instance_from_number(index: Option<&ConstEtomoNumber>) -> Option<EndingStep> {
        let index = index?;
        if index.is_null() {
            return None;
        }
        Self::get_instance(Some(index.get_int()))
    }

    /// Java static `getInstance(Integer)`.
    pub fn get_instance(index: Option<i32>) -> Option<EndingStep> {
        let index = index?;
        DECLARED.into_iter().find(|step| index == step.index())
    }

    /// Java private static `getInstance(Step)`.
    fn get_instance_from_step(step: Option<Step>) -> Option<EndingStep> {
        let step = step?;
        DECLARED
            .into_iter()
            .find(|ending_step| step == ending_step.step())
    }

    /// Java `isFirst()`.  True if this is the first ending step to be executed by
    /// batchruntomo.
    pub fn is_first(self) -> bool {
        self == EndingStep::FineAlignment
    }

    /// Java `getIndex()`.
    pub fn get_index(self) -> i32 {
        self.index()
    }

    /// Java `getTooltip()`.
    pub fn get_tooltip(self) -> &'static str {
        self.tooltip()
    }

    /// Java `getStep()`.
    pub fn get_step(self) -> Step {
        self.step()
    }

    /// Java `le(EndingStep)`.
    pub fn le(self, input: Option<EndingStep>) -> bool {
        let Some(input) = input else {
            return true;
        };
        self.step().le(input.step())
    }

    /// Java `lt(EndingStep)`.
    pub fn lt(self, input: Option<EndingStep>) -> bool {
        let Some(input) = input else {
            return true;
        };
        self.step().lt(Some(input.step()))
    }
}

impl EnumeratedType for EndingStep {
    /// Java `isDefault()`.
    fn is_default(&self) -> bool {
        *self == EndingStep::GoldDetection3d
    }

    /// Java `getValue()`.
    fn get_value(&self) -> ConstEtomoNumber {
        let value = self.step().get_value();
        let number: &ConstEtomoNumber = &value;
        number.clone()
    }

    /// Java `getLabel()`.
    fn get_label(&self) -> Option<String> {
        self.step().get_label().map(str::to_owned)
    }
}

impl Status for EndingStep {
    /// Java `getText()`.
    fn get_text(&self) -> Option<&'static str> {
        self.step().get_text()
    }
}

/// Java `toString()`.
impl std::fmt::Display for EndingStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[EndingStep:index:{},\nstep:{}]",
            self.index(),
            self.step()
        )
    }
}
