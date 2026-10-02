//! `IMOD/Etomo/src/etomo/type/Step.java`.
//!
//! The batchruntomo step numbers.  Java's typesafe-enum pattern (a private constructor
//! plus `static final` singletons) is mirrored as a `Copy` struct with associated
//! constants; the singleton's `EtomoNumber value` field is built from the constructor's
//! string on each access (`value()`), exactly as the private constructor builds it.

use crate::imod::etomo::r#type::const_etomo_number::Type;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::status::Status;
use crate::imod::etomo::util::utilities::{to_string_if_set, to_string_if_set_etomo_number};

/// Java final `Step implements Status`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Step {
    /// The string the private constructor passes to `this.value.set(value)`.
    value: &'static str,
    /// Java private final field `label`.
    label: Option<&'static str>,
    /// Java private final field `text`.
    text: Option<&'static str>,
}

impl Step {
    // <p>Updates done</p>

    /// Java `SETUP = new Step("0")`.
    pub const SETUP: Step = Step::new_string("0");
    /// Java `BEAD_TRACKING = new Step("5", "Tracking", "Track")`.
    pub const BEAD_TRACKING: Step =
        Step::new_string_string_string("5", Some("Tracking"), Some("Track"));
    /// Java `FINE_ALIGNMENT = new Step("6", "Fine alignment", "Align")`.
    pub const FINE_ALIGNMENT: Step =
        Step::new_string_string_string("6", Some("Fine alignment"), Some("Align"));
    /// Java `POSITIONING = new Step("7", "Positioning", "Pos")`.
    pub const POSITIONING: Step =
        Step::new_string_string_string("7", Some("Positioning"), Some("Pos"));
    /// Java package-private `ALIGNED_STACK_GENERATION = new Step("8", "Aligned stack")`.
    pub(crate) const ALIGNED_STACK_GENERATION: Step =
        Step::new_string_string("8", Some("Aligned stack"));
    /// Java `GOLD_DETECTION_3D = new Step("10", "CTF/gold detection", "CTF/gold")`.
    pub const GOLD_DETECTION_3D: Step =
        Step::new_string_string_string("10", Some("CTF/gold detection"), Some("CTF/gold"));
    /// Java package-private `CTF_CORRECTION = new Step("11", "Finish CTF/gold")`.
    pub(crate) const CTF_CORRECTION: Step = Step::new_string_string("11", Some("Finish CTF/gold"));
    /// Java `TWO_D_FILTERING = new Step("13", "Finished stack", "Stack")`.
    pub const TWO_D_FILTERING: Step =
        Step::new_string_string_string("13", Some("Finished stack"), Some("Stack"));
    /// Java `RECONSTRUCTION = new Step("14", "Reconstruction")`.
    pub const RECONSTRUCTION: Step = Step::new_string_string("14", Some("Reconstruction"));

    /// Java private `Step(String, String, String)`.
    const fn new_string_string_string(
        value: &'static str,
        label: Option<&'static str>,
        text: Option<&'static str>,
    ) -> Step {
        Step { value, label, text }
    }

    /// Java private `Step(String)`: `this(value, null, null)`.
    const fn new_string(value: &'static str) -> Step {
        Step::new_string_string_string(value, None, None)
    }

    /// Java private `Step(String, String)`: `this(value, label, null)`.
    const fn new_string_string(value: &'static str, label: Option<&'static str>) -> Step {
        Step::new_string_string_string(value, label, None)
    }

    /// Java private final field `value`, as the constructor builds it: an integer
    /// `EtomoNumber` unless the string contains a '.', then a double one; then
    /// `this.value.set(value)`.
    fn value(&self) -> EtomoNumber {
        let mut value = if self.value.find('.').is_none() {
            EtomoNumber::new()
        } else {
            EtomoNumber::new_with_type(Some(Type::Double))
        };
        value.set_string(Some(self.value));
        value
    }

    /// Java static `getInstance(String)`.
    pub fn get_instance(value: Option<&str>) -> Option<Step> {
        let value = value?;
        if Self::SETUP.value().equals_string(Some(value)) {
            return Some(Self::SETUP);
        }
        if Self::BEAD_TRACKING.value().equals_string(Some(value)) {
            return Some(Self::BEAD_TRACKING);
        }
        if Self::FINE_ALIGNMENT.value().equals_string(Some(value)) {
            return Some(Self::FINE_ALIGNMENT);
        }
        if Self::POSITIONING.value().equals_string(Some(value)) {
            return Some(Self::POSITIONING);
        }
        if Self::ALIGNED_STACK_GENERATION
            .value()
            .equals_string(Some(value))
        {
            return Some(Self::ALIGNED_STACK_GENERATION);
        }
        if Self::GOLD_DETECTION_3D.value().equals_string(Some(value)) {
            return Some(Self::GOLD_DETECTION_3D);
        }
        if Self::CTF_CORRECTION.value().equals_string(Some(value)) {
            return Some(Self::CTF_CORRECTION);
        }
        if Self::TWO_D_FILTERING.value().equals_string(Some(value)) {
            return Some(Self::TWO_D_FILTERING);
        }
        if Self::RECONSTRUCTION.value().equals_string(Some(value)) {
            return Some(Self::RECONSTRUCTION);
        }
        None
    }

    /// Java package-private static `getInstanceFromText(String)`.
    pub(crate) fn get_instance_from_text(text: Option<&str>) -> Option<Step> {
        let text = text?;
        if Self::SETUP.text.is_some() && Self::SETUP.text.unwrap() == text {
            return Some(Self::SETUP);
        }
        if Self::BEAD_TRACKING.text.is_some() && Self::BEAD_TRACKING.text.unwrap() == text {
            return Some(Self::BEAD_TRACKING);
        }
        if Self::FINE_ALIGNMENT.text.is_some() && Self::FINE_ALIGNMENT.text.unwrap() == text {
            return Some(Self::FINE_ALIGNMENT);
        }
        if Self::POSITIONING.text.is_some() && Self::POSITIONING.text.unwrap() == text {
            return Some(Self::POSITIONING);
        }
        if Self::ALIGNED_STACK_GENERATION.text.is_some()
            && Self::ALIGNED_STACK_GENERATION.text.unwrap() == text
        {
            return Some(Self::ALIGNED_STACK_GENERATION);
        }
        if Self::GOLD_DETECTION_3D.text.is_some() && Self::GOLD_DETECTION_3D.text.unwrap() == text {
            return Some(Self::GOLD_DETECTION_3D);
        }
        if Self::CTF_CORRECTION.text.is_some() && Self::CTF_CORRECTION.text.unwrap() == text {
            return Some(Self::CTF_CORRECTION);
        }
        if Self::TWO_D_FILTERING.text.is_some() && Self::TWO_D_FILTERING.text.unwrap() == text {
            return Some(Self::TWO_D_FILTERING);
        }
        if Self::RECONSTRUCTION.text.is_some() && Self::RECONSTRUCTION.text.unwrap() == text {
            return Some(Self::RECONSTRUCTION);
        }
        None
    }

    /// Java package-private `getLabel()`.
    pub(crate) fn get_label(&self) -> Option<&'static str> {
        self.label
    }

    /// Java package-private `getValue()`.  Java returns the singleton's own
    /// `EtomoNumber`; it is rebuilt here from the constructor's string.
    pub(crate) fn get_value(&self) -> EtomoNumber {
        self.value()
    }

    /// Java `lt(Step)`.  Less then function bases order on the parameter value in
    /// batchruntomo.
    pub fn lt(&self, input: Option<Step>) -> bool {
        let input = match input {
            None => return false,
            Some(input) => input,
        };
        self.value().lt_const_etomo_number(Some(&input.value()))
    }

    /// Java package-private `le(Step)`.  Less then or equal to function bases order on
    /// the parameter value in batchruntomo.
    pub(crate) fn le(&self, input: Step) -> bool {
        self.value().le_const_etomo_number(&input.value())
    }
}

impl Status for Step {
    /// Java `getText()`.
    fn get_text(&self) -> Option<&'static str> {
        if self.text.is_some() {
            return self.text;
        }
        Some("")
    }
}

/// Java `toString()`.
impl std::fmt::Display for Step {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let value = self.value();
        // `Utilities.getClassString(getClass())` is `getExtension("class etomo.type.Step")`,
        // i.e. "Step".
        write!(
            f,
            "[{}{}{}{}]",
            "Step",
            to_string_if_set(Some(":label:"), self.label),
            to_string_if_set(Some(",text:"), self.text),
            to_string_if_set_etomo_number(Some(",value:"), Some(&value))
        )
    }
}
