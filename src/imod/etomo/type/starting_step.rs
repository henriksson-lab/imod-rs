//! `IMOD/Etomo/src/etomo/type/StartingStep.java`.
//!
//! The batchruntomo starting steps offered by the batch interface.  The Java
//! singletons are an enum.

use super::const_etomo_number::ConstEtomoNumber;
use super::enumerated_type::EnumeratedType;
use super::step::Step;

/// Java `public final class StartingStep implements EnumeratedType`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum StartingStep {
    /// Java private `FINE_ALIGNMENT = new StartingStep(0, Step.FINE_ALIGNMENT, ...)`.
    FineAlignment,
    /// Java private `POSITIONING = new StartingStep(1, Step.POSITIONING, ...)`.
    Positioning,
    /// Java private `ALIGNED_STACK_GENERATION = new StartingStep(2,
    /// Step.ALIGNED_STACK_GENERATION, ...)`.
    AlignedStackGeneration,
    /// Java private `CTF_CORRECTION = new StartingStep(3, Step.CTF_CORRECTION, ...)`.
    CtfCorrection,
    /// Java private `RECONSTRUCTION = new StartingStep(4, Step.RECONSTRUCTION, ...)`.
    Reconstruction,
}

impl StartingStep {
    /// Java private final `index`.
    fn index(self) -> i32 {
        match self {
            StartingStep::FineAlignment => 0,
            StartingStep::Positioning => 1,
            StartingStep::AlignedStackGeneration => 2,
            StartingStep::CtfCorrection => 3,
            StartingStep::Reconstruction => 4,
        }
    }

    /// Java private final `step`.
    fn step(self) -> Step {
        match self {
            StartingStep::FineAlignment => Step::FINE_ALIGNMENT,
            StartingStep::Positioning => Step::POSITIONING,
            StartingStep::AlignedStackGeneration => Step::ALIGNED_STACK_GENERATION,
            StartingStep::CtfCorrection => Step::CTF_CORRECTION,
            StartingStep::Reconstruction => Step::RECONSTRUCTION,
        }
    }

    /// Java private final `tooltip`.
    fn tooltip(self) -> &'static str {
        match self {
            StartingStep::FineAlignment => "Start from fine alignment with Tiltalign.",
            StartingStep::Positioning => "Start with tomogram positioning (if any).",
            StartingStep::AlignedStackGeneration => {
                "Start with generating the aligned stack from the raw stack."
            }
            StartingStep::CtfCorrection => {
                "Start with correcting the CTF then erasing the gold (if any)."
            }
            StartingStep::Reconstruction => "Start with making the reconstruction.",
        }
    }

    /// Java static `getInstance(String)`.
    pub fn get_instance_from_step_value(step_value: Option<&str>) -> Option<StartingStep> {
        Self::get_instance_from_step(Step::get_instance(step_value))
    }

    /// Java static `getInstance(int)`.
    pub fn get_instance(index: i32) -> Option<StartingStep> {
        for step in [
            StartingStep::FineAlignment,
            StartingStep::Positioning,
            StartingStep::AlignedStackGeneration,
            StartingStep::CtfCorrection,
            StartingStep::Reconstruction,
        ] {
            if index == step.index() {
                return Some(step);
            }
        }
        None
    }

    /// Java private static `getInstance(Step)`.
    fn get_instance_from_step(step: Option<Step>) -> Option<StartingStep> {
        let step = step?;
        for starting_step in [
            StartingStep::FineAlignment,
            StartingStep::Positioning,
            StartingStep::AlignedStackGeneration,
            StartingStep::CtfCorrection,
            StartingStep::Reconstruction,
        ] {
            if step == starting_step.step() {
                return Some(starting_step);
            }
        }
        None
    }

    /// Java `getTooltip()`.
    pub fn get_tooltip(self) -> &'static str {
        self.tooltip()
    }

    /// Java `getIndex()`.
    pub fn get_index(self) -> i32 {
        self.index()
    }
}

impl EnumeratedType for StartingStep {
    /// Java `isDefault()`.
    fn is_default(&self) -> bool {
        *self == StartingStep::CtfCorrection
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

/// Java `toString()`: `step.toString()`.
impl std::fmt::Display for StartingStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.step())
    }
}
