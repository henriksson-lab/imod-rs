//! `IMOD/Etomo/src/etomo/ui/swing/NewstackAndBlendmontParamPanel.java`.
//!
//! Java `final class NewstackAndBlendmontParamPanel implements
//! FiducialessParams`.  An EDT object: created as `Rc<Self>` by
//! [`NewstackAndBlendmontParamPanel::get_instance`], every method takes
//! `&self`, mutable state lives in `RefCell`.  The two listener classes
//! (`NewstackAndBlendmontParamPanelActionListener`,
//! `NewstackAndBlendmontBinningChangeListener`) are closures holding a weak
//! reference to the panel.

use crate::imod::etomo::ui::field::Field;
use std::cell::RefCell;
use std::rc::{Rc, Weak};

use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::blendmont_param::{BlendmontParam, ConvertError};
use crate::imod::etomo::comscript::const_newst_param::ConstNewstParam;
use crate::imod::etomo::comscript::fortran_input_syntax_exception::FortranInputSyntaxException;
use crate::imod::etomo::comscript::newst_param::{self, NewstParam, SetSizeToOutputInXandYError};
use crate::imod::etomo::jdk::{ActionListener, ChangeListener, JComponent};
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::storage::autodoc::autodoc_factory;
use crate::imod::etomo::storage::autodoc::read_only_autodoc::ReadOnlyAutodoc;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{ConstEtomoNumber, Number};
use crate::imod::etomo::r#type::const_meta_data::ConstMetaData;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::etomo_autodoc;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::meta_data::MetaData;
use crate::imod::etomo::r#type::view_type::ViewType;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;
use crate::imod::etomo::util::invalid_parameter_exception::InvalidParameterException;

use super::blendmont_display::BlendmontDisplayException;
use super::check_box::CheckBox;
use super::deferred_3dmod_button::Deferred3dmodButton;
use super::fiducialess_params::FiducialessParams;
use super::label::Label;
use super::labeled_spinner::LabeledSpinner;
use super::labeled_text_field::LabeledTextField;
use super::newstack_display::NewstackDisplayException;
use super::process_control_panel;
use super::spaced_panel::SpacedPanel;

/// Java private static final `SIZE_TO_OUTPUT_IN_X_AND_Y_LABEL`.
pub const SIZE_TO_OUTPUT_IN_X_AND_Y_LABEL: &str = "Size to output";
/// Java package-private static final `BINNING_LABEL`.
pub const BINNING_LABEL: &str = "Aligned image stack binning";

/// Java `final class NewstackAndBlendmontParamPanel implements FiducialessParams`.
pub struct NewstackAndBlendmontParamPanel {
    /// Java private final `pnlRoot`.
    pnl_root: Rc<SpacedPanel>,
    /// Java private final `spinBinning`.
    spin_binning: Rc<LabeledSpinner>,
    /// Java private final `ltfSizeToOutputInXandY`.
    ltf_size_to_output_in_xand_y: Rc<LabeledTextField>,
    /// Java private final `ltfRotation`.
    ltf_rotation: Rc<LabeledTextField>,
    /// Java private final `cbFiducialess`.
    cb_fiducialess: Rc<CheckBox>,
    /// Java private final `cbUseLinearInterpolation`.
    cb_use_linear_interpolation: Rc<CheckBox>,
    /// Java private final `lCtf3d1` (a `Label`).
    l_ctf3d1: Rc<Label>,
    /// Java private final `lCtf3d2` (a `Label`).
    l_ctf3d2: Rc<Label>,

    /// Java private final `axisID`.
    axis_id: AxisID,
    /// Java private final `manager`.
    manager: &'static ApplicationManager,
    /// Java private final `dialogType`.
    dialog_type: DialogType,
    /// Java private final `cbAntialiasFilter`; null (`None`) for a montage.
    cb_antialias_filter: Option<Rc<CheckBox>>,
    /// Java private final `antialiasFilterValue`; null (`None`) for a montage.
    /// The Java object is final but mutable (`set`), hence the `RefCell`.
    antialias_filter_value: Option<RefCell<EtomoNumber>>,
}

impl NewstackAndBlendmontParamPanel {
    /// Java private constructor `NewstackAndBlendmontParamPanel(ApplicationManager,
    /// AxisID, DialogType)` (NewstackAndBlendmontParamPanel.java:71).
    fn new(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        dialog_type: DialogType,
    ) -> Rc<NewstackAndBlendmontParamPanel> {
        // Field initializers, in declaration order.
        let pnl_root = SpacedPanel::get_instance_boolean(true);
        let spin_binning = LabeledSpinner::get_instance_string_int_int_int_int(
            Some(&format!("{BINNING_LABEL}: ")),
            1,
            1,
            8,
            1,
        );
        let ltf_size_to_output_in_xand_y = LabeledTextField::new_field_type_string(
            FieldType::IntegerPair,
            Some(&format!(
                "{SIZE_TO_OUTPUT_IN_X_AND_Y_LABEL} (X,Y - unbinned): "
            )),
        );
        let ltf_rotation = LabeledTextField::new_field_type_string(
            FieldType::FloatingPoint,
            Some("Tilt axis rotation: "),
        );
        let cb_fiducialess = CheckBox::new_string(Some("Coarse alignment only"));
        let cb_use_linear_interpolation = CheckBox::new_string(Some("Use linear interpolation"));
        let l_ctf3d1 = Label::new_string(Some(
            "No need to make stack if doing 3D CTF and using raw images - ",
        ));
        let l_ctf3d2 = Label::new_string(Some(
            "unless you want to check gold erasing with an aligned stack.",
        ));
        // Constructor body.
        let (cb_antialias_filter, antialias_filter_value) =
            if manager.get_meta_data().get_view_type() != ViewType::Montage {
                (
                    Some(CheckBox::new_string(Some(
                        "Reduce size with antialiasing filter",
                    ))),
                    Some(RefCell::new(EtomoNumber::new())),
                )
            } else {
                (None, None)
            };
        Rc::new(NewstackAndBlendmontParamPanel {
            pnl_root,
            spin_binning,
            ltf_size_to_output_in_xand_y,
            ltf_rotation,
            cb_fiducialess,
            cb_use_linear_interpolation,
            l_ctf3d1,
            l_ctf3d2,
            axis_id,
            manager,
            dialog_type,
            cb_antialias_filter,
            antialias_filter_value,
        })
    }

    /// Java static `getInstance(ApplicationManager, AxisID, DialogType)`
    /// (NewstackAndBlendmontParamPanel.java:86).
    pub fn get_instance(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        dialog_type: DialogType,
    ) -> Rc<NewstackAndBlendmontParamPanel> {
        let instance = NewstackAndBlendmontParamPanel::new(manager, axis_id, dialog_type);
        instance.create_panel();
        instance.add_listeners();
        instance.set_tool_tip_text();
        instance
    }

    /// Java private `addListeners()` (NewstackAndBlendmontParamPanel.java:96).
    fn add_listeners(self: &Rc<Self>) {
        // NewstackAndBlendmontParamPanelActionListener
        let adaptee: Weak<NewstackAndBlendmontParamPanel> = Rc::downgrade(self);
        let action_listener: ActionListener = Rc::new(move |event| {
            if let Some(adaptee) = adaptee.upgrade() {
                adaptee.action(event.get_action_command().unwrap_or(""), None, None);
            }
        });
        self.cb_fiducialess
            .add_action_listener(Some(action_listener));
        // NewstackAndBlendmontBinningChangeListener
        let panel: Weak<NewstackAndBlendmontParamPanel> = Rc::downgrade(self);
        let change_listener: ChangeListener = Rc::new(move |_event| {
            if let Some(panel) = panel.upgrade() {
                panel.update_enabled();
            }
        });
        self.spin_binning.add_change_listener(change_listener);
    }

    /// Java `getComponent()` (NewstackAndBlendmontParamPanel.java:102).
    pub fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.get_container()
    }

    /// Java private `createPanel()` (NewstackAndBlendmontParamPanel.java:106).
    fn create_panel(&self) {
        // init
        self.l_ctf3d1
            .get_component()
            .set_foreground(Some(process_control_panel::COLOR_COMPLETE));
        self.l_ctf3d2
            .get_component()
            .set_foreground(Some(process_control_panel::COLOR_COMPLETE));
        self.l_ctf3d1.set_visible(false);
        self.l_ctf3d2.set_visible(false);
        // Root panel
        // Swing layout: pnlRoot BoxLayout Y_AXIS, component alignment LEFT.
        self.pnl_root
            .add_check_box(&self.cb_use_linear_interpolation);
        let pnl_binning = JComponent::new_panel();
        // Swing layout: pnlBinning BoxLayout X_AXIS, CENTER_ALIGNMENT.
        pnl_binning.add(&self.spin_binning.get_container());
        // Swing layout: horizontal glue.
        self.pnl_root.add_j_panel(&pnl_binning);
        if let Some(cb_antialias_filter) = &self.cb_antialias_filter {
            self.pnl_root.add_check_box(cb_antialias_filter);
        }
        self.pnl_root.add_check_box(&self.cb_fiducialess);
        self.pnl_root.add_labeled_text_field(&self.ltf_rotation);
        self.pnl_root
            .add_labeled_text_field(&self.ltf_size_to_output_in_xand_y);
        // Swing layout: Box.createVerticalStrut(2).
        self.pnl_root.add_j_label(&self.l_ctf3d1.get_component());
        self.pnl_root.add_j_label(&self.l_ctf3d2.get_component());
        // Swing layout: Box.createVerticalStrut(3).
        // update
        self.update_fiducialess();
    }

    /// Java `setVisible(boolean)` (NewstackAndBlendmontParamPanel.java:136).
    pub fn set_visible(&self, visible: bool) {
        self.pnl_root.set_visible(visible);
    }

    /// Java `setParameters(BlendmontParam)` (NewstackAndBlendmontParamPanel.java:140).
    pub fn set_parameters_blendmont_param(&self, blendmont_param: &BlendmontParam) {
        self.cb_use_linear_interpolation
            .set_selected_boolean(blendmont_param.is_linear_interpolation());
    }

    /// Java `setParameters(ConstNewstParam)` (NewstackAndBlendmontParamPanel.java:144).
    pub fn set_parameters_const_newst_param(&self, newst_param: &dyn ConstNewstParam) {
        self.cb_use_linear_interpolation
            .set_selected_boolean(newst_param.is_linear_interpolation());
        if let Some(cb_antialias_filter) = &self.cb_antialias_filter {
            let antialias_filter = !newst_param.is_antialias_filter_null();
            cb_antialias_filter.set_selected_boolean(antialias_filter);
            if antialias_filter {
                // antialiasFilterValue is non-null whenever cbAntialiasFilter is.
                if let Some(antialias_filter_value) = &self.antialias_filter_value {
                    antialias_filter_value
                        .borrow_mut()
                        .set_string(Some(&newst_param.get_antialias_filter()));
                }
            }
        }
    }

    /// Java `getParameters(MetaData) throws FortranInputSyntaxException`
    /// (NewstackAndBlendmontParamPanel.java:162).  The Metadata values that
    /// are from the setup dialog should not be overrided by this dialog unless
    /// the Metadata values are empty.  Must save data from the two instances
    /// under separate keys.
    pub fn get_parameters_meta_data(
        &self,
        meta_data: &MetaData,
    ) -> Result<(), FortranInputSyntaxException> {
        meta_data.set_size_to_output_in_x_and_y(
            self.axis_id,
            self.ltf_size_to_output_in_xand_y.get_text_void().as_deref(),
        )?;
        meta_data.set_stack_binning_int(self.axis_id, self.get_binning());
        let antialias_filter_value = self
            .antialias_filter_value
            .as_ref()
            .map(|value| value.borrow());
        meta_data.set_antialias_filter(
            self.dialog_type,
            self.axis_id,
            antialias_filter_value.as_deref().map(|value| &**value),
        );
        Ok(())
    }

    /// Java `getParameters(BlendmontParam, boolean) throws
    /// FortranInputSyntaxException, InvalidParameterException, IOException`
    /// (NewstackAndBlendmontParamPanel.java:168).
    pub fn get_parameters_blendmont_param_boolean(
        &self,
        blendmont_param: &mut BlendmontParam,
        do_validation: bool,
    ) -> Result<bool, BlendmontDisplayException> {
        // try { ... } catch (FieldValidationFailedException e) { return false; }
        blendmont_param.set_bin_by_factor_int(self.get_binning());
        blendmont_param.set_linear_interpolation(self.cb_use_linear_interpolation.is_selected());
        let size_to_output_in_xand_y = match self
            .ltf_size_to_output_in_xand_y
            .get_text_boolean(do_validation)
        {
            Ok(text) => text,
            Err(_) => return Ok(false),
        };
        match blendmont_param.convert_to_starting_and_ending_xand_y(
            size_to_output_in_xand_y.as_deref().unwrap_or(""),
            self.manager
                .get_meta_data()
                .get_image_rotation(self.axis_id)
                .get_double(),
            Some(&self.ltf_size_to_output_in_xand_y.get_label()),
        ) {
            Ok(true) => {}
            Ok(false) => return Ok(false),
            Err(ConvertError::FortranInputSyntax(e)) => {
                // catch (FortranInputSyntaxException e): printStackTrace, then
                // rethrow with the label prefixed.
                e.print_stack_trace();
                return Err(BlendmontDisplayException::FortranInputSyntaxException(
                    FortranInputSyntaxException::new(&format!(
                        "{SIZE_TO_OUTPUT_IN_X_AND_Y_LABEL}:  {e}"
                    )),
                ));
            }
            // InvalidParameterException / IOException from reading the
            // montage size propagate unchanged.
            Err(ConvertError::MontagesizeRead(message)) => {
                return Err(BlendmontDisplayException::InvalidParameterException(
                    InvalidParameterException::new(&message),
                ));
            }
        }
        blendmont_param.set_fiducialess(self.cb_fiducialess.is_selected());
        Ok(true)
    }

    /// Java `getParameters(NewstParam, boolean) throws
    /// FortranInputSyntaxException, InvalidParameterException, IOException`
    /// (NewstackAndBlendmontParamPanel.java:196).  Copy the newstack
    /// parameters from the GUI to the NewstParam object.
    pub fn get_parameters_newst_param_boolean(
        &self,
        newst_param: &mut NewstParam,
        do_validation: bool,
    ) -> Result<bool, NewstackDisplayException> {
        // try { ... } catch (FieldValidationFailedException e) { return false; }
        let binning = self.get_binning();
        // Only explicitly write out the binning if its value is something other than
        // the default of 1 to keep from cluttering up the com script
        if binning > 1 {
            newst_param.set_bin_by_factor(Some(Number::Integer(binning)));
        } else {
            newst_param.set_bin_by_factor(Some(Number::Integer(i32::MIN)));
        }
        // Save when field is disabled
        // Upstream bug fixed in translation: NewstackAndBlendmontParamPanel.java:209
        // dereferences `cbAntialiasFilter`, which the constructor leaves null
        // for a montage, so this method throws a NullPointerException there.
        // A missing checkbox reads as unchecked here (no antialias filter),
        // which is what the montage panel displays.
        let antialias_filter = self
            .cb_antialias_filter
            .as_ref()
            .is_some_and(|cb_antialias_filter| cb_antialias_filter.is_selected());
        newst_param.set_antialias_filter(antialias_filter);
        if antialias_filter {
            if let Some(antialias_filter_value) = &self.antialias_filter_value {
                newst_param.set_antialias_filter_value(&antialias_filter_value.borrow());
            }
        }
        newst_param.set_linear_interpolation(self.cb_use_linear_interpolation.is_selected());
        let size_to_output_in_xand_y = match self
            .ltf_size_to_output_in_xand_y
            .get_text_boolean(do_validation)
        {
            Ok(text) => text,
            Err(_) => return Ok(false),
        };
        match newst_param.set_size_to_output_in_xand_y(
            size_to_output_in_xand_y.as_deref().unwrap_or(""),
            self.get_binning(),
            self.manager
                .get_meta_data()
                .get_image_rotation(self.axis_id)
                .get_double(),
            Some(&self.ltf_size_to_output_in_xand_y.get_label()),
        ) {
            Ok(result) => Ok(result),
            Err(SetSizeToOutputInXandYError::FortranInputSyntax(e)) => {
                Err(NewstackDisplayException::FortranInputSyntaxException(e))
            }
            // InvalidParameterException / IOException from reading the header.
            Err(SetSizeToOutputInXandYError::HeaderRead(message)) => {
                Err(NewstackDisplayException::InvalidParameterException(
                    InvalidParameterException::new(&message),
                ))
            }
        }
    }

    /// Java `setParameters(ConstMetaData)` (NewstackAndBlendmontParamPanel.java:225).
    pub fn set_parameters_const_meta_data(&self, meta_data: &dyn ConstMetaData) {
        let ctf3d = meta_data.is_ctf_3d_setup_slab_thickness_in_nm_set();
        self.l_ctf3d1.set_visible(ctf3d);
        self.l_ctf3d2.set_visible(ctf3d);
        self.spin_binning
            .set_value_int(meta_data.get_stack_binning(self.axis_id));
        if !meta_data.is_antialias_filter_null(self.dialog_type, self.axis_id) {
            // Upstream bug fixed in translation: NewstackAndBlendmontParamPanel.java:231
            // calls `antialiasFilterValue.set(...)` unguarded, but the
            // constructor leaves `antialiasFilterValue` null for a montage, so
            // a montage dataset whose metadata carries an antialias value
            // throws a NullPointerException.  The value is only stored when
            // the field exists (the montage panel has nothing to show it in).
            if let Some(antialias_filter_value) = &self.antialias_filter_value {
                let value = meta_data.get_antialias_filter(self.dialog_type, self.axis_id);
                antialias_filter_value
                    .borrow_mut()
                    .set_const_etomo_number(value.as_deref());
            }
        }
        self.ltf_size_to_output_in_xand_y.set_text_string(Some(
            &meta_data
                .get_size_to_output_in_x_and_y(self.axis_id)
                .to_string_default_is_blank(true),
        ));
        self.update_fiducialess();
        self.update_enabled();
    }

    /// Java `setFiducialessAlignment(boolean)` (NewstackAndBlendmontParamPanel.java:239).
    pub fn set_fiducialess_alignment(&self, input: bool) {
        self.cb_fiducialess.set_selected_boolean(input);
        self.update_fiducialess();
    }

    /// Java `setImageRotation(String)` (NewstackAndBlendmontParamPanel.java:244).
    pub fn set_image_rotation(&self, input: Option<&str>) {
        self.ltf_rotation.set_text_string(input);
    }

    /// Java `setBinning(ConstEtomoNumber)` (NewstackAndBlendmontParamPanel.java:248).
    pub fn set_binning(&self, binning: &ConstEtomoNumber) {
        self.spin_binning.set_value_const_etomo_number(binning);
        self.update_enabled();
    }

    /// Java private `getBinning()` (NewstackAndBlendmontParamPanel.java:253):
    /// `((Integer) spinBinning.getValue()).intValue()`.
    fn get_binning(&self) -> i32 {
        self.spin_binning.get_value().int_value()
    }

    /// Java private `updateEnabled()` (NewstackAndBlendmontParamPanel.java:257).
    fn update_enabled(&self) {
        if let Some(cb_antialias_filter) = &self.cb_antialias_filter {
            // `Number value = spinBinning.getValue(); value != null && ...`:
            // the spinner's model always holds a value.
            let value = self.spin_binning.get_value();
            cb_antialias_filter.set_enabled(value.int_value() > 1);
        }
    }

    /// Java private `updateFiducialess()` (NewstackAndBlendmontParamPanel.java:269).
    fn update_fiducialess(&self) {
        self.ltf_rotation
            .set_enabled(self.cb_fiducialess.is_selected());
    }

    /// Java `updateAdvanced(boolean)` (NewstackAndBlendmontParamPanel.java:279).
    pub fn update_advanced(&self, advanced: bool) {
        self.ltf_size_to_output_in_xand_y.set_visible(advanced);
    }

    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`
    /// (NewstackAndBlendmontParamPanel.java:292).  Executes the action
    /// associated with command.  Deferred3dmodButton is null if it comes from
    /// the dialog's ActionListener.
    pub fn action(
        &self,
        command: &str,
        _deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        _run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        if self.cb_fiducialess.get_action_command().as_deref() == Some(command) {
            self.update_fiducialess();
        }
    }

    /// Java private `setToolTipText()` (NewstackAndBlendmontParamPanel.java:299).
    fn set_tool_tip_text(&self) {
        let manager: &'static dyn BaseManager = self.manager;
        let autodoc: *mut crate::imod::etomo::storage::autodoc::autodoc::Autodoc = match unsafe {
            autodoc_factory::get_instance(
                Some(manager),
                Some(autodoc_factory::NEWSTACK),
                self.axis_id,
                false,
            )
        } {
            Ok(autodoc) => autodoc,
            // catch (final LockException except) {}
            Err(LogFileError::Lock(_)) => std::ptr::null_mut(),
            // catch (final LogFileException | IOException except)
            Err(except) => {
                eprintln!("{except}");
                std::ptr::null_mut()
            }
        };
        if !autodoc.is_null() {
            // `ltfSizeToOutputInXandY != null` is always true (final field).
            let autodoc: *const dyn ReadOnlyAutodoc = autodoc;
            self.ltf_size_to_output_in_xand_y.set_tool_tip_text(
                unsafe {
                    etomo_autodoc::get_tooltip(
                        Some(&*autodoc),
                        Some(newst_param::SIZE_TO_OUTPUT_IN_X_AND_Y),
                    )
                }
                .as_deref(),
            );
        }
        // `cbUseLinearInterpolation != null` is always true (final field).
        self.cb_use_linear_interpolation
            .set_tool_tip_text_string(Some(
                "Make aligned stack with linear instead of cubic interpolation to  reduce noise.",
            ));
        self.spin_binning.set_tool_tip_text(Some(
            "Set the binning for the aligned image stack and tomogram.  With a binned \
             tomogram, all of the thickness, position, and size parameters in Tomogram \
             Generation are still entered in unbinned pixels.",
        ));
        self.cb_fiducialess
            .set_tool_tip_text_string(Some("Use cross-correlation alignment only."));
        self.ltf_rotation.set_tool_tip_text(Some(
            "Rotation angle of tilt axis for generating aligned stack from \
             cross-correlation alignment only.",
        ));
        if let Some(cb_antialias_filter) = &self.cb_antialias_filter {
            cb_antialias_filter.set_tool_tip_text_string(Some(
                "Use antialiased image reduction instead binning with the default filter \
                 in Newstack; useful for data from direct detection cameras.",
            ));
        }
    }
}

impl FiducialessParams for NewstackAndBlendmontParamPanel {
    /// Java `isFiducialess()` (NewstackAndBlendmontParamPanel.java:265).
    fn is_fiducialess(&self) -> bool {
        self.cb_fiducialess.is_selected()
    }

    /// Java `getImageRotation(boolean) throws FieldValidationFailedException`
    /// (NewstackAndBlendmontParamPanel.java:274).
    fn get_image_rotation(
        &self,
        do_validation: bool,
    ) -> Result<String, FieldValidationFailedException> {
        // The trait returns a String; a text field's text is never null.
        Ok(self
            .ltf_rotation
            .get_text_boolean(do_validation)?
            .unwrap_or_default())
    }
}
