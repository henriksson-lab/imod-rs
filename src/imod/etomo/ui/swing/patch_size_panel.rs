//! `IMOD/Etomo/src/etomo/ui/swing/PatchSizePanel.java`.
//!
//! Panel to hold the patch size type or x, y, and z values.  Used twice by
//! `SetupCombinePanel`: once for the starting patch size (Small / Medium /
//! Large / Custom) and once, with `finalSize` set, for the maximum patch size
//! of automatic patch fitting (Medium / Large / Extra Large / Custom).
//!
//! Java `final class PatchSizePanel implements ActionListener`: an EDT object
//! created as `Rc<Self>` by [`PatchSizePanel::get_instance`]; the panel itself
//! is the action listener of its radio buttons, which is a closure holding a
//! weak reference to it.
//!
//! The radio buttons are enum-constructed with `CombinePatchSize` instances,
//! so `rbTypeMedium` selects itself on construction
//! (`CombinePatchSize.isDefault()` is `this == MEDIUM`) and the group always
//! has a selection.

use std::rc::{Rc, Weak};

use super::abstract_radio_button_model::AbstractRadioButtonModel;
use super::etched_border::EtchedBorder;
use super::labeled_text_field::LabeledTextField;
use super::radio_button::{RadioButton, RadioButtonModel};
use super::radio_button_interface::EnumeratedTypeRef;
use crate::imod::etomo::comscript::combine_params::CombineParams;
use crate::imod::etomo::comscript::const_combine_params::ConstCombineParams;
use crate::imod::etomo::comscript::const_patchcrawl3d_param::ConstPatchcrawl3DParam;
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, ButtonGroup, JComponent};
use crate::imod::etomo::r#type::combine_patch_size::CombinePatchSize;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_type::FieldType;

/// Java public static final `DEFAULT_FINAL_SIZE = CombinePatchSize.EXTRA_LARGE`.
pub const DEFAULT_FINAL_SIZE: CombinePatchSize = CombinePatchSize::ExtraLarge;
/// Java private static final `X_INDEX`.
const X_INDEX: usize = 0;
/// Java private static final `Y_INDEX`.
const Y_INDEX: usize = 1;
/// Java private static final `Z_INDEX`.
const Z_INDEX: usize = 2;

/// Java `final class PatchSizePanel implements ActionListener`.
pub struct PatchSizePanel {
    /// Java private final `pnlRoot = new JPanel()`.
    pnl_root: Rc<JComponent>,
    /// Java private final `bgType = new ButtonGroup()`.
    bg_type: Rc<ButtonGroup>,
    /// Java private final `rbTypeMedium`.
    rb_type_medium: Rc<RadioButton>,
    /// Java private final `rbTypeLarge`.
    rb_type_large: Rc<RadioButton>,
    /// Java private final `rbTypeCustom`.
    rb_type_custom: Rc<RadioButton>,
    /// Java private final `xyzLabels = { "X: ", "Y: ", "Z: " }`.
    #[allow(dead_code)]
    xyz_labels: [&'static str; 3],
    /// Java private final `ltfXYZ = new LabeledTextField[xyzLabels.length]`.
    ltf_xyz: Vec<Rc<LabeledTextField>>,

    /// Java private final `rbTypeSmall`; null for the final (maximum) size.
    rb_type_small: Option<Rc<RadioButton>>,
    /// Java private final `rbTypeExtraLarge`; null for the starting size.
    rb_type_extra_large: Option<Rc<RadioButton>>,
    /// Java private final `title`.
    title: String,
    /// Java private final `finalSize`.
    final_size: bool,

    /// Java `this` as the `ActionListener` of the radio buttons.
    this: Weak<PatchSizePanel>,
}

impl PatchSizePanel {
    /// Java private constructor `PatchSizePanel(boolean finalSize)`.
    fn new(final_size: bool) -> Rc<PatchSizePanel> {
        Rc::new_cyclic(|this: &Weak<PatchSizePanel>| {
            // Field initializers, in declaration order.
            let pnl_root = JComponent::new_panel();
            let bg_type = ButtonGroup::new();
            let rb_type_medium = RadioButton::new_string_enumerated_type_button_group(
                Some("Medium patches"),
                Some(EnumeratedTypeRef::new(CombinePatchSize::Medium)),
                Some(&bg_type),
            );
            let rb_type_large = RadioButton::new_string_enumerated_type_button_group(
                Some("Large patches"),
                Some(EnumeratedTypeRef::new(CombinePatchSize::Large)),
                Some(&bg_type),
            );
            let rb_type_custom = RadioButton::new_string_enumerated_type_button_group(
                Some("Custom"),
                Some(EnumeratedTypeRef::new(CombinePatchSize::Custom)),
                Some(&bg_type),
            );
            let xyz_labels: [&'static str; 3] = ["X: ", "Y: ", "Z: "];
            // Constructor body.
            // this.finalSize = finalSize;
            let base_title = "Patch Size";
            let mut ltf_xyz: Vec<Rc<LabeledTextField>> = Vec::with_capacity(xyz_labels.len());
            for i in 0..xyz_labels.len() {
                let ltf = LabeledTextField::new_field_type_string(
                    FieldType::Integer,
                    Some(xyz_labels[i]),
                );
                ltf.set_required(true);
                ltf_xyz.push(ltf);
            }
            let title;
            let rb_type_small;
            let rb_type_extra_large;
            if !final_size {
                title = base_title.to_string();
                rb_type_small = Some(RadioButton::new_string_enumerated_type_button_group(
                    Some("Small patches"),
                    Some(EnumeratedTypeRef::new(CombinePatchSize::Small)),
                    Some(&bg_type),
                ));
                rb_type_extra_large = None;
            } else {
                title = format!("Max {base_title}");
                rb_type_small = None;
                rb_type_extra_large = Some(RadioButton::new_string_enumerated_type_button_group(
                    Some("Extra Large patches"),
                    Some(EnumeratedTypeRef::new(CombinePatchSize::ExtraLarge)),
                    Some(&bg_type),
                ));
            }
            PatchSizePanel {
                pnl_root,
                bg_type,
                rb_type_medium,
                rb_type_large,
                rb_type_custom,
                xyz_labels,
                ltf_xyz,
                rb_type_small,
                rb_type_extra_large,
                title,
                final_size,
                this: this.clone(),
            }
        })
    }

    /// Java package-private static `getInstance(boolean finalSize)`.
    pub fn get_instance(final_size: bool) -> Rc<PatchSizePanel> {
        let instance = PatchSizePanel::new(final_size);
        instance.create_panel(final_size);
        instance.add_listeners();
        instance
    }

    /// Java private `createPanel(boolean maxSize)`.  The parameter is not read.
    fn create_panel(&self, _max_size: bool) {
        // init
        // Swing layout: rbTypeSmall, rbTypeMedium, rbTypeLarge, rbTypeExtraLarge
        // and rbTypeCustom .setAlignmentX(Component.LEFT_ALIGNMENT) (where not
        // null).
        // panels
        let pnl_type = JComponent::new_panel();
        let pnl_xyz = JComponent::new_panel();
        // Root
        // Swing layout: pnlRoot.setLayout(new BoxLayout(pnlRoot, BoxLayout.X_AXIS)).
        self.pnl_root
            .set_border_title(EtchedBorder::new(Some(&self.title)).get_title().as_deref());
        // Swing layout: pnlRoot.setAlignmentX(Component.CENTER_ALIGNMENT).
        self.pnl_root.add(&pnl_type);
        // Swing layout: pnlRoot.add(Box.createHorizontalGlue()).
        self.pnl_root.add(&pnl_xyz);
        // Type
        // Swing layout: pnlType.setLayout(new BoxLayout(pnlType, BoxLayout.Y_AXIS)).
        if let Some(rb_type_small) = &self.rb_type_small {
            pnl_type.add(&rb_type_small.get_component());
        }
        pnl_type.add(&self.rb_type_medium.get_component());
        pnl_type.add(&self.rb_type_large.get_component());
        if let Some(rb_type_extra_large) = &self.rb_type_extra_large {
            pnl_type.add(&rb_type_extra_large.get_component());
        }
        pnl_type.add(&self.rb_type_custom.get_component());
        // XYZ
        // Swing layout: pnlXYZ.setLayout(new BoxLayout(pnlXYZ, BoxLayout.Y_AXIS)).
        pnl_xyz.set_border_title(EtchedBorder::new(Some("In Pixels")).get_title().as_deref());
        for i in 0..self.ltf_xyz.len() {
            pnl_xyz.add(&self.ltf_xyz[i].get_component());
            if i < self.ltf_xyz.len() - 1 {
                // Swing layout: pnlXYZ.add(Box.createVerticalGlue()).
            }
        }
        // update
        self.action_performed(None);
    }

    /// Java package-private `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.clone()
    }

    /// Java private `addListeners()`: the panel itself is the listener.
    fn add_listeners(&self) {
        let adaptee = self.this.clone();
        let listener: ActionListener = Rc::new(move |event: &ActionEvent| {
            if let Some(adaptee) = adaptee.upgrade() {
                adaptee.action_performed(Some(event));
            }
        });
        if let Some(rb_type_small) = &self.rb_type_small {
            rb_type_small.add_action_listener(listener.clone());
        }
        self.rb_type_medium.add_action_listener(listener.clone());
        self.rb_type_large.add_action_listener(listener.clone());
        if let Some(rb_type_extra_large) = &self.rb_type_extra_large {
            rb_type_extra_large.add_action_listener(listener.clone());
        }
        self.rb_type_custom.add_action_listener(listener);
    }

    /// Java public override `actionPerformed(ActionEvent)` (called with null
    /// from inside the class).
    pub fn action_performed(&self, _event: Option<&ActionEvent>) {
        // ((RadioButton.RadioButtonModel) bgType.getSelection()).getEnumeratedType().
        // Java throws NullPointerException on an empty selection, which cannot
        // happen here (rbTypeMedium selects itself on construction and a button
        // group never deselects); None skips the enumerated-type branch.
        let enumerated_type: Option<EnumeratedTypeRef> = self
            .bg_type
            .get_selection()
            .and_then(|button| button.get_model())
            .and_then(|model| {
                model
                    .as_any()
                    .downcast_ref::<RadioButtonModel>()
                    .and_then(|model| model.get_enumerated_type())
            });
        if let Some(enumerated_type) = &enumerated_type {
            // enumeratedType != CombinePatchSize.CUSTOM
            // && enumeratedType instanceof CombinePatchSize
            if let Some(combined_patch_size) = enumerated_type.downcast_ref::<CombinePatchSize>()
                && *combined_patch_size != CombinePatchSize::Custom
            {
                let len = std::cmp::min(self.ltf_xyz.len(), combined_patch_size.get_xyz_len());
                for i in 0..len {
                    self.ltf_xyz[i].set_text_int(combined_patch_size.get_xyz(i));
                }
            }
        }
        self.update_display();
    }

    /// Java private `updateDisplay()`.
    fn update_display(&self) {
        let enabled = self.rb_type_custom.is_enabled();
        let selected = self.rb_type_custom.is_selected();
        for i in 0..self.ltf_xyz.len() {
            self.ltf_xyz[i].set_enabled(enabled);
            self.ltf_xyz[i].set_editable(selected);
        }
    }

    /// Java package-private `setEnabled(boolean)`.
    pub fn set_enabled(&self, enabled: bool) {
        if let Some(rb_type_small) = &self.rb_type_small {
            rb_type_small.set_enabled(enabled);
        }
        self.rb_type_medium.set_enabled(enabled);
        self.rb_type_large.set_enabled(enabled);
        if let Some(rb_type_extra_large) = &self.rb_type_extra_large {
            rb_type_extra_large.set_enabled(enabled);
        }
        self.rb_type_custom.set_enabled(enabled);
        self.update_display();
    }

    /// Java private `isEnabled()`.
    fn is_enabled(&self) -> bool {
        self.rb_type_medium.is_enabled()
    }

    /// Java package-private `setParameters(ConstCombineParams)`.
    pub fn set_parameters_const_combine_params(&self, combine_params: &dyn ConstCombineParams) {
        let mut combine_patch_size: Option<CombinePatchSize>;
        combine_patch_size = combine_params.get_patch_size(self.final_size);
        if combine_patch_size.is_none() && self.final_size {
            combine_patch_size = Some(DEFAULT_FINAL_SIZE);
        }
        if combine_patch_size == Some(CombinePatchSize::Custom) {
            self.rb_type_custom.set_selected_boolean(true);
            let xyz = combine_params.get_patch_size_xyz_array(self.final_size);
            if let Some(xyz) = xyz {
                let len = std::cmp::min(self.ltf_xyz.len(), xyz.len());
                for i in 0..len {
                    self.ltf_xyz[i].set_text_string(Some(&xyz[i]));
                }
            }
        } else {
            self.set_fixed_type(combine_patch_size);
        }
        self.action_performed(None);
    }

    /// Java package-private `setParameters(ConstPatchcrawl3DParam)`.
    pub fn set_parameters_const_patchcrawl3d_param(
        &self,
        patchrawl_param: &ConstPatchcrawl3DParam,
    ) {
        // Assume flipped
        let xyz: [i32; 3] = [
            patchrawl_param.get_x_patch_size(),
            patchrawl_param.get_z_patch_size(),
            patchrawl_param.get_y_patch_size(),
        ];
        let combine_patch_size = CombinePatchSize::get_instance_xyz_ints(Some(&xyz));
        if combine_patch_size == Some(CombinePatchSize::Custom) {
            self.rb_type_custom.set_selected_boolean(true);
            let len = std::cmp::min(self.ltf_xyz.len(), xyz.len());
            for i in 0..len {
                self.ltf_xyz[i].set_text_int(xyz[i]);
            }
        } else {
            self.set_fixed_type(combine_patch_size);
        }
        self.action_performed(None);
    }

    /// Java package-private `setFixedType(CombinePatchSize)`; `None` is Java
    /// null (matches no branch).
    pub fn set_fixed_type(&self, combine_patch_size: Option<CombinePatchSize>) {
        if combine_patch_size == Some(CombinePatchSize::Small) {
            if let Some(rb_type_small) = &self.rb_type_small {
                rb_type_small.set_selected_boolean(true);
            } else {
                self.select_type_xyz(CombinePatchSize::Small);
            }
        } else if combine_patch_size == Some(CombinePatchSize::Medium) {
            self.rb_type_medium.set_selected_boolean(true);
        } else if combine_patch_size == Some(CombinePatchSize::Large) {
            self.rb_type_large.set_selected_boolean(true);
        } else if combine_patch_size == Some(CombinePatchSize::ExtraLarge) {
            if let Some(rb_type_extra_large) = &self.rb_type_extra_large {
                rb_type_extra_large.set_selected_boolean(true);
            } else {
                self.select_type_xyz(CombinePatchSize::ExtraLarge);
            }
        }
    }

    /// Java package-private `getParameters(CombineParams, boolean)`.
    pub fn get_parameters(&self, combine_params: &mut CombineParams, do_validation: bool) -> bool {
        if self.is_enabled() {
            // ((RadioButton.RadioButtonModel) bgType.getSelection()).getEnumeratedType().
            // Java throws NullPointerException on an empty selection, which cannot
            // happen here (rbTypeMedium selects itself on construction and a button
            // group never deselects); None skips the enumerated-type branch.
            let combined_patch_size: Option<EnumeratedTypeRef> = self
                .bg_type
                .get_selection()
                .and_then(|button| button.get_model())
                .and_then(|model| {
                    model
                        .as_any()
                        .downcast_ref::<RadioButtonModel>()
                        .and_then(|model| model.get_enumerated_type())
                });
            // combinedPatchSize instanceof CombinePatchSize
            let patch_size = combined_patch_size
                .as_ref()
                .and_then(|combined_patch_size| {
                    combined_patch_size
                        .downcast_ref::<CombinePatchSize>()
                        .copied()
                });
            if let Some(patch_size) = patch_size {
                combine_params.set_patch_size(self.final_size, Some(patch_size));
            }
            if patch_size == Some(CombinePatchSize::Custom) {
                let mut xyz: Vec<String> = Vec::with_capacity(self.ltf_xyz.len());
                // try { ... } catch (FieldValidationFailedException e) { return false; }
                for i in 0..self.ltf_xyz.len() {
                    match self.ltf_xyz[i].get_text_boolean(do_validation) {
                        Ok(text) => xyz.push(text.unwrap_or_default()),
                        Err(_) => return false,
                    }
                }
                let xyz: Vec<&str> = xyz.iter().map(String::as_str).collect();
                combine_params.set_patch_size_xyz(self.final_size, Some(&xyz));
            }
        } else {
            combine_params.reset_patch_size(self.final_size);
        }
        true
    }

    /// Java private `selectTypeXyz(CombinePatchSize)`.
    fn select_type_xyz(&self, combine_patch_size: CombinePatchSize) {
        self.rb_type_custom.set_selected_boolean(true);
        let len = std::cmp::min(self.ltf_xyz.len(), combine_patch_size.get_xyz_len());
        for i in 0..len {
            self.ltf_xyz[i].set_text_int(combine_patch_size.get_xyz(i));
        }
    }

    /// Java package-private `setSmallTooltip(String)`.
    pub fn set_small_tooltip(&self, tooltip: &str) {
        if let Some(rb_type_small) = &self.rb_type_small {
            rb_type_small.set_tool_tip_text_string(Some(tooltip));
        }
    }

    /// Java package-private `setMediumTooltip(String)`.
    pub fn set_medium_tooltip(&self, tooltip: &str) {
        self.rb_type_medium.set_tool_tip_text_string(Some(tooltip));
    }

    /// Java package-private `setLargeTooltip(String)`.
    pub fn set_large_tooltip(&self, tooltip: &str) {
        self.rb_type_large.set_tool_tip_text_string(Some(tooltip));
    }

    /// Java package-private `setExtraLargeTooltip(String)`.
    pub fn set_extra_large_tooltip(&self, tooltip: &str) {
        if let Some(rb_type_extra_large) = &self.rb_type_extra_large {
            rb_type_extra_large.set_tool_tip_text_string(Some(tooltip));
        }
    }

    /// Java package-private `setCustomTooltip(String)`.
    pub fn set_custom_tooltip(&self, tooltip: &str) {
        self.rb_type_custom.set_tool_tip_text_string(Some(tooltip));
    }

    /// Java package-private `setXTooltip(String)`.
    pub fn set_x_tooltip(&self, tooltip: &str) {
        self.ltf_xyz[X_INDEX].set_tool_tip_text(Some(tooltip));
    }

    /// Java package-private `setYTooltip(String)`.
    pub fn set_y_tooltip(&self, tooltip: &str) {
        self.ltf_xyz[Y_INDEX].set_tool_tip_text(Some(tooltip));
    }

    /// Java package-private `setZTooltip(String)`.
    pub fn set_z_tooltip(&self, tooltip: &str) {
        self.ltf_xyz[Z_INDEX].set_tool_tip_text(Some(tooltip));
    }
}
