//! `IMOD/Etomo/src/etomo/ui/swing/CcdEraserBeadsPanel.java`.
//!
//! Java `final class CcdEraserBeadsPanel implements Run3dmodButtonContainer,
//! CcdEraserDisplay, FocusListener`: the "Erase Beads" section of the Erase
//! Gold tab of the final aligned stack dialog.  Its `getParameters(CCDEraserParam,
//! boolean)` decides the contents of golderaser.com.
//!
//! An EDT object (`Rc<Self>`, `&self` methods).  The inner listener class
//! `CcdEraserPanelActionListener` is a closure holding a weak reference to the
//! panel; the inner `Runnable` `CallFocusLost` is a job posted with
//! `event_queue::invoke_later`; the private static inner enumerated type
//! `PolynomialOrder` is [`PolynomialOrder`].

use std::cell::Cell;
use std::fmt;
use std::rc::{Rc, Weak};
use std::sync::Arc;

use super::abstract_radio_button_model::AbstractRadioButtonModel;
use super::ccd_eraser_display::CcdEraserDisplay;
use super::check_box_spinner::CheckBoxSpinner;
use super::deferred_3dmod_button::Deferred3dmodButton;
use super::etched_border::EtchedBorder;
use super::etomo_button_group::EtomoButtonGroup;
use super::label::Label;
use super::labeled_text_field::LabeledTextField;
use super::process_control_panel;
use super::process_display::ProcessDisplay;
use super::radio_button::{RadioButton, RadioButtonModel};
use super::radio_button_interface::{EnumeratedTypeRef, RadioButtonInterface};
use super::run_3dmod_button::Run3dmodButton;
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use super::spaced_panel::SpacedPanel;
use super::ui_utilities;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::ccd_eraser_param::CCDEraserParam;
use crate::imod::etomo::comscript::const_ccd_eraser_param::{self, ConstCCDEraserParam};
use crate::imod::etomo::comscript::fortran_input_syntax_exception::FortranInputSyntaxException;
use crate::imod::etomo::comscript::makecomfile_param::MakecomfileParam;
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, FocusEvent, JComponent};
use crate::imod::etomo::logic::converter;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::storage::autodoc::autodoc::Autodoc;
use crate::imod::etomo::storage::autodoc::autodoc_factory;
use crate::imod::etomo::storage::autodoc::read_only_autodoc::ReadOnlyAutodoc;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{
    ConstEtomoNumber, Type, java_lang_double_to_string,
};
use crate::imod::etomo::r#type::const_meta_data::ConstMetaData;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::enumerated_type::EnumeratedType;
use crate::imod::etomo::r#type::etomo_autodoc;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::meta_data::MetaData;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::r#type::recon_screen_state::ReconScreenState;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::util::event_queue::{self, EdtRef};
use crate::imod::etomo::util::utilities;

/// Java package-private static final `CCD_ERASER_LABEL`.
pub const CCD_ERASER_LABEL: &str = "Erase Beads";
/// Java package-private static final `USE_ERASED_STACK_LABEL`.
pub const USE_ERASED_STACK_LABEL: &str = "Use Erased Stack";
/// Java package-private static final `FIDUCIAL_DIAMETER_LABEL`.
pub const FIDUCIAL_DIAMETER_LABEL: &str = "Diameter to erase";

/// Java `BoxLayout.Y_AXIS` / `BoxLayout.X_AXIS` (for `SpacedPanel.setBoxLayout`).
const Y_AXIS: i32 = 1;
const X_AXIS: i32 = 0;

/// Java `final class CcdEraserBeadsPanel implements Run3dmodButtonContainer,
/// CcdEraserDisplay, FocusListener`.
pub struct CcdEraserBeadsPanel {
    /// Java `this`.
    this: Weak<CcdEraserBeadsPanel>,
    /// Java private final `actionListener` (`CcdEraserPanelActionListener`).
    action_listener: ActionListener,
    /// Java private final `pnlRoot = SpacedPanel.getInstance()`.
    pnl_root: Rc<SpacedPanel>,
    /// Java private final `ltfFiducialDiameter`.
    ltf_fiducial_diameter: Rc<LabeledTextField>,
    /// Java private final `lFiducialDiameter1` (a `Label`, which is a `JLabel`).
    l_fiducial_diameter1: Rc<Label>,
    /// Java private final `lFiducialDiameter2`.
    l_fiducial_diameter2: Rc<Label>,
    /// Java private final `cbspExpandCircleIterations`.
    cbsp_expand_circle_iterations: Rc<CheckBoxSpinner>,
    /// Java private final `bgPolynomialOrder`.
    bg_polynomial_order: Rc<EtomoButtonGroup>,
    /// Java private final `rbPolynomialOrderUseMean`.
    rb_polynomial_order_use_mean: Rc<RadioButton>,
    /// Java private final `rbPolynomialOrderFillNoise`.
    rb_polynomial_order_fill_noise: Rc<RadioButton>,
    /// Java private final `rbPolynomialOrderFitAPlane`.
    rb_polynomial_order_fit_a_plane: Rc<RadioButton>,
    /// Java private final `btn3dmodCcdEraser`.
    btn_3dmod_ccd_eraser: Rc<Run3dmodButton>,
    /// Java private final `lCtf3d1`.
    l_ctf3d1: Rc<Label>,
    /// Java private final `lCtf3d2`.
    l_ctf3d2: Rc<Label>,

    /// Java private final `btnCcdEraser`.
    btn_ccd_eraser: Rc<Run3dmodButton>,
    /// Java private final `btnUseCcdEraser` (a `MultiLineButton`; the factory
    /// hands out the concrete `Run3dmodButton`, which derefs to one).
    btn_use_ccd_eraser: Rc<Run3dmodButton>,
    /// Java private final `axisID`.
    axis_id: AxisID,
    /// Java private final `manager`.
    manager: &'static ApplicationManager,
    /// Java private final `dialogType`.
    dialog_type: DialogType,

    /// Java private `alignedStackBinning` (`Integer`, null is `None`).
    aligned_stack_binning: Cell<Option<i32>>,
}

impl CcdEraserBeadsPanel {
    /// Java private constructor `CcdEraserBeadsPanel(ApplicationManager, AxisID,
    /// DialogType)`.
    fn new(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        dialog_type: DialogType,
    ) -> Rc<CcdEraserBeadsPanel> {
        let instance = Rc::new_cyclic(|this: &Weak<CcdEraserBeadsPanel>| {
            // Field initializers, in declaration order.
            // Java `new CcdEraserPanelActionListener(this)`.
            let adaptee = this.clone();
            let action_listener: ActionListener = Rc::new(move |event: &ActionEvent| {
                let Some(adaptee) = adaptee.upgrade() else {
                    return;
                };
                adaptee.action_performed(event);
            });
            let pnl_root = SpacedPanel::get_instance_void();
            let ltf_fiducial_diameter = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some(&format!("{FIDUCIAL_DIAMETER_LABEL} (binned pixels): ")),
            );
            let l_fiducial_diameter1 = Label::new_string(Some(FIDUCIAL_DIAMETER_LABEL));
            let l_fiducial_diameter2 = Label::new_string(Some(FIDUCIAL_DIAMETER_LABEL));
            let cbsp_expand_circle_iterations = CheckBoxSpinner::get_instance_string_int_int_int(
                Some("Iterations to grow circular areas:"),
                2,
                1,
                5,
            );
            let bg_polynomial_order = EtomoButtonGroup::new();
            let rb_polynomial_order_use_mean = Self::new_radio_button(
                "Use mean of surrounding points",
                PolynomialOrder::UseMean,
                &bg_polynomial_order,
            );
            let rb_polynomial_order_fill_noise = Self::new_radio_button(
                "Fill pixels with noise values",
                PolynomialOrder::FillWithNoise,
                &bg_polynomial_order,
            );
            let rb_polynomial_order_fit_a_plane = Self::new_radio_button(
                "Fit a plane to surrounding points",
                PolynomialOrder::FitAPlane,
                &bg_polynomial_order,
            );
            let container: Weak<dyn Run3dmodButtonContainer> = this.clone();
            let btn_3dmod_ccd_eraser =
                Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                    Some("View Erased Stack"),
                    Some(container),
                );
            let l_ctf3d1 = Label::new_string(Some(
                "If doing 3D CTF, finalize model and set erasing options; no need",
            ));
            let l_ctf3d2 = Label::new_string(Some(&format!(
                "to run {}",
                "\"Erase Beads\" if \"Erase gold\" option set in 3D CTF."
            )));
            // Constructor body.
            let display_factory = manager.get_process_result_display_factory(axis_id);
            // Java casts `(Run3dmodButton) displayFactory.getCcdEraserBeads()` and
            // `(MultiLineButton) displayFactory.getUseCcdEraserBeads()`; the factory
            // returns the concrete buttons.
            let btn_ccd_eraser = display_factory.get_ccd_eraser_beads();
            let btn_use_ccd_eraser = display_factory.get_use_ccd_eraser_beads();
            CcdEraserBeadsPanel {
                this: this.clone(),
                action_listener,
                pnl_root,
                ltf_fiducial_diameter,
                l_fiducial_diameter1,
                l_fiducial_diameter2,
                cbsp_expand_circle_iterations,
                bg_polynomial_order,
                rb_polynomial_order_use_mean,
                rb_polynomial_order_fill_noise,
                rb_polynomial_order_fit_a_plane,
                btn_3dmod_ccd_eraser,
                l_ctf3d1,
                l_ctf3d2,
                btn_ccd_eraser,
                btn_use_ccd_eraser,
                axis_id,
                manager,
                dialog_type,
                aligned_stack_binning: Cell::new(None),
            }
        });
        instance.update_aligned_stack_binning();
        instance
    }

    /// Java field initializer `new RadioButton(String, EnumeratedType,
    /// ButtonGroup)` with an `EtomoButtonGroup`.  Java's `RadioButton`
    /// constructor calls `group.add(radioButton)`, which dispatches to
    /// `EtomoButtonGroup.add` and records the button's `RadioButtonModel` under
    /// its enumerated type.  The Rust `RadioButton` takes the plain
    /// `ButtonGroup`, so the button is built without one and then added to the
    /// `EtomoButtonGroup` with a `RadioButtonModel` for the same button (the
    /// model only answers the button's enumerated type).  Rust-only
    /// construction plumbing, not a Java method.
    fn new_radio_button(
        text: &str,
        polynomial_order: PolynomialOrder,
        group: &Rc<EtomoButtonGroup>,
    ) -> Rc<RadioButton> {
        let radio_button = RadioButton::new_string_enumerated_type_button_group(
            Some(text),
            Some(EnumeratedTypeRef::new(polynomial_order)),
            None,
        );
        let button: Weak<dyn RadioButtonInterface> =
            Rc::downgrade(&radio_button) as Weak<dyn RadioButtonInterface>;
        let model: Rc<dyn AbstractRadioButtonModel> = RadioButtonModel::new(Some(button));
        group.add(&radio_button.get_component(), Some(model));
        radio_button
    }

    /// Java static `getInstance(ApplicationManager, AxisID, DialogType)`.
    pub fn get_instance(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        dialog_type: DialogType,
    ) -> Rc<CcdEraserBeadsPanel> {
        let instance = CcdEraserBeadsPanel::new(manager, axis_id, dialog_type);
        instance.create_panel();
        instance.add_listeners();
        instance.set_tool_tip_text();
        instance
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self) {
        self.btn_ccd_eraser
            .add_action_listener(self.action_listener.clone());
        self.btn_3dmod_ccd_eraser
            .add_action_listener(self.action_listener.clone());
        self.btn_use_ccd_eraser
            .add_action_listener(self.action_listener.clone());
        let this = self.this.clone();
        self.ltf_fiducial_diameter
            .add_focus_listener(Rc::new(move |event: &FocusEvent| {
                if let Some(this) = this.upgrade() {
                    if event.gained {
                        this.focus_gained();
                    } else {
                        this.focus_lost();
                    }
                }
            }));
    }

    /// Java package-private `initialize()`.
    pub fn initialize(&self) {
        if self.ltf_fiducial_diameter.is_empty() {
            self.ltf_fiducial_diameter.set_text_double(
                self.manager.calc_binned_bead_diameter_pixels(
                    self.axis_id,
                    &file_type::CLASS.aligned_stack,
                    1,
                ),
            );
        }
    }

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        // local panels
        let ccd_eraser_parameter_panel = SpacedPanel::get_instance_void();
        let polynomial_order_panel = JComponent::new_panel();
        let ccd_eraser_button_panel = SpacedPanel::get_instance_void();
        let fiducial_diameter_panel = JComponent::new_panel();
        let pnl_ctf3d = JComponent::new_panel();
        let l_fiducial_diameter1_panel = JComponent::new_panel();
        let l_fiducial_diameter2_panel = JComponent::new_panel();
        // initalization
        let container: Weak<dyn Run3dmodButtonContainer> = self.this.clone();
        self.btn_ccd_eraser.set_container(Some(container));
        let deferred: Rc<dyn Deferred3dmodButton> = self.btn_3dmod_ccd_eraser.clone();
        self.btn_ccd_eraser
            .set_deferred_3dmod_button_deferred_3dmod_button(Some(deferred));
        self.l_ctf3d1
            .get_component()
            .set_foreground(Some(process_control_panel::COLOR_COMPLETE));
        self.l_ctf3d2
            .get_component()
            .set_foreground(Some(process_control_panel::COLOR_COMPLETE));
        self.l_ctf3d1.set_visible(false);
        self.l_ctf3d2.set_visible(false);
        self.l_fiducial_diameter1
            .get_component()
            .set_foreground(Some(process_control_panel::COLOR_COMPLETE));
        self.l_fiducial_diameter2
            .get_component()
            .set_foreground(Some(process_control_panel::COLOR_COMPLETE));
        // Root
        self.pnl_root.set_box_layout(Y_AXIS);
        self.pnl_root
            .set_border(&EtchedBorder::new(Some("Erase Beads")).get_border());
        self.pnl_root.add_spaced_panel(&ccd_eraser_parameter_panel);
        // Swing layout: pnlRoot.add(Box.createVerticalStrut(3)).
        self.pnl_root.add_j_panel(&pnl_ctf3d);
        // Swing layout: pnlRoot.add(Box.createVerticalStrut(5)).
        self.pnl_root.add_spaced_panel(&ccd_eraser_button_panel);
        // Ctf3d
        // Swing layout: pnlCtf3d BoxLayout Y_AXIS.
        pnl_ctf3d.add(&self.l_ctf3d1.get_component());
        pnl_ctf3d.add(&self.l_ctf3d2.get_component());
        // Component.CENTER_ALIGNMENT
        ui_utilities::align_components_x(&pnl_ctf3d, 0.5);
        // ccderaser parameters
        ccd_eraser_parameter_panel.set_box_layout(X_AXIS);
        ccd_eraser_parameter_panel.add_j_panel(&fiducial_diameter_panel);
        ccd_eraser_parameter_panel.add_j_panel(&polynomial_order_panel);
        // Fiducial diameter
        // Swing layout: fiducialDiameterPanel BoxLayout Y_AXIS.
        fiducial_diameter_panel.add(&self.ltf_fiducial_diameter.get_container());
        fiducial_diameter_panel.add(&l_fiducial_diameter1_panel);
        fiducial_diameter_panel.add(&l_fiducial_diameter2_panel);
        // Swing layout: fiducialDiameterPanel.add(Box.createVerticalGlue()).
        fiducial_diameter_panel.add(&self.cbsp_expand_circle_iterations.get_container());
        // Swing layout: fiducialDiameterPanel.add(Box.createVerticalGlue()).
        // lFiducialDiameter1Panel
        // Swing layout: lFiducialDiameter1Panel BoxLayout X_AXIS, horizontal glue.
        l_fiducial_diameter1_panel.add(&self.l_fiducial_diameter1.get_component());
        // lFiducialDiameter2Panel
        // Swing layout: lFiducialDiameter2Panel BoxLayout X_AXIS, horizontal glue.
        l_fiducial_diameter2_panel.add(&self.l_fiducial_diameter2.get_component());
        // polynomial order
        // Swing layout: polynomialOrderPanel BoxLayout Y_AXIS.
        polynomial_order_panel.set_border_title(
            EtchedBorder::new(Some("Pixel Filling Method"))
                .get_border()
                .get_title()
                .as_deref(),
        );
        polynomial_order_panel.add(&self.rb_polynomial_order_use_mean.get_component());
        polynomial_order_panel.add(&self.rb_polynomial_order_fill_noise.get_component());
        polynomial_order_panel.add(&self.rb_polynomial_order_fit_a_plane.get_component());
        // buttons
        ccd_eraser_button_panel.set_box_layout(X_AXIS);
        ccd_eraser_button_panel.add_component(&self.btn_ccd_eraser.get_component());
        ccd_eraser_button_panel.add_component(&self.btn_3dmod_ccd_eraser.get_component());
        ccd_eraser_button_panel.add_component(&self.btn_use_ccd_eraser.get_component());
    }

    /// Java package-private `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.get_container()
    }

    /// Java package-private `getPolynomialOrder()`:
    /// `((RadioButton.RadioButtonModel) bgPolynomialOrder.getSelection())
    /// .getEnumeratedType().toString()`.
    ///
    /// Upstream NullPointerException fixed in translation: with no selection
    /// (or a model without an enumerated type) the Java throws; here the
    /// result is null.  The group always holds a selection in practice (the
    /// USE_MEAN button selects itself as the default).
    pub fn get_polynomial_order(&self) -> Option<String> {
        let selection = self.bg_polynomial_order.get_selection()?;
        let model = selection.get_model()?;
        let model = model.as_any().downcast_ref::<RadioButtonModel>()?;
        let enumerated_type = AbstractRadioButtonModel::get_enumerated_type(model)?;
        Some(enumerated_type.to_string())
    }

    /// Java package-private `setPolynomialOrder(EnumeratedType)`.
    pub fn set_polynomial_order(&self, enumerated_type: &EnumeratedTypeRef) {
        if self
            .rb_polynomial_order_use_mean
            .get_enumerated_type()
            .as_ref()
            == Some(enumerated_type)
        {
            self.rb_polynomial_order_use_mean.set_selected_boolean(true);
        } else if self
            .rb_polynomial_order_fill_noise
            .get_enumerated_type()
            .as_ref()
            == Some(enumerated_type)
        {
            self.rb_polynomial_order_fill_noise
                .set_selected_boolean(true);
        } else if self
            .rb_polynomial_order_fit_a_plane
            .get_enumerated_type()
            .as_ref()
            == Some(enumerated_type)
        {
            self.rb_polynomial_order_fit_a_plane
                .set_selected_boolean(true);
        }
    }

    /// Java package-private `getParameters(MetaData) throws
    /// FortranInputSyntaxException`.  The Metadata values that are from the
    /// setup dialog should not be overrided by this dialog unless the Metadata
    /// values are empty.
    pub fn get_parameters_meta_data(
        &self,
        meta_data: &MetaData,
    ) -> Result<(), FortranInputSyntaxException> {
        meta_data.set_final_stack_fiducial_diameter(
            self.axis_id,
            self.ltf_fiducial_diameter.get_text_void().as_deref(),
        );
        meta_data.set_final_stack_expand_circle_iterations_object(
            self.axis_id,
            Some(self.cbsp_expand_circle_iterations.get_value()),
        );
        meta_data.set_use_final_stack_expand_circle_iterations(
            self.axis_id,
            self.cbsp_expand_circle_iterations.is_selected(),
        );
        meta_data
            .set_final_stack_polynomial_order(self.axis_id, self.get_polynomial_order().as_deref());
        Ok(())
    }

    /// Java package-private `setParameters(ConstMetaData)`.
    pub fn set_parameters_const_meta_data(&self, meta_data: &dyn ConstMetaData) {
        let ctf3d = meta_data.is_ctf_3d_setup_slab_thickness_in_nm_set();
        self.l_ctf3d1.set_visible(ctf3d);
        self.l_ctf3d2.set_visible(ctf3d);
        if !meta_data.is_final_stack_fiducial_diameter_null(self.axis_id) {
            self.ltf_fiducial_diameter.set_text_string(Some(
                &meta_data.get_final_stack_fiducial_diameter(self.axis_id),
            ));
        } else if !meta_data.is_final_stack_better_radius_empty(self.axis_id) {
            // backwards compatibility - used to save better radius, convert it to
            // fiducial diameter in pixels
            let mut better_radius = EtomoNumber::new_with_type(Some(Type::Double));
            better_radius.set_string(Some(&meta_data.get_final_stack_better_radius(self.axis_id)));
            // Math.round(double) is a long; long / 10.0 is a double.
            self.ltf_fiducial_diameter.set_text_double(
                utilities::java_lang_math_round(better_radius.get_double() * 2.0 * 10.0) as f64
                    / 10.0,
            );
        } else {
            // Currently not allowing an empty value to be saved.
            // Default fiducialDiameter to fiducialDiameter from setup / pixel size
            // (convert to pixels). Round to 1 decimal place.
            self.ltf_fiducial_diameter.set_text_double(
                utilities::java_lang_math_round(
                    meta_data.get_fiducial_diameter() / meta_data.get_pixel_size() * 10.0,
                ) as f64
                    / 10.0,
            );
        }
        self.set_polynomial_order(&EnumeratedTypeRef::new(PolynomialOrder::get_instance_int(
            meta_data.get_final_stack_polynomial_order(self.axis_id),
        )));
        self.cbsp_expand_circle_iterations
            .set_selected(meta_data.is_use_final_stack_expand_circle_iterations(self.axis_id));
        if meta_data.is_final_stack_expand_circle_iterations_set(self.axis_id) {
            self.cbsp_expand_circle_iterations
                .set_value_int(meta_data.get_final_stack_expand_circle_iterations(self.axis_id));
        }
        self.focus_lost();
    }

    /// Java public `setParameters(ConstCCDEraserParam)`.
    pub fn set_parameters_const_ccd_eraser_param(&self, param: &ConstCCDEraserParam) {
        if param.is_better_radius_set() {
            let mut fiducial_diameter = EtomoNumber::new_with_type(Some(Type::Double));
            fiducial_diameter.set_double(param.get_better_radius() * 2.0);
            self.ltf_fiducial_diameter
                .set_text_const_etomo_number(Some(&*fiducial_diameter));
        }
        self.cbsp_expand_circle_iterations
            .set_selected(param.is_expand_circle_iterations_set());
        if self.cbsp_expand_circle_iterations.is_selected() {
            self.cbsp_expand_circle_iterations
                .set_value_string(param.get_expand_circle_iterations().as_deref());
        }
        let polynomial_order =
            PolynomialOrder::get_instance_string(param.get_polynomial_order().as_deref());
        self.bg_polynomial_order
            .set_selected(Some(&EnumeratedTypeRef::new(polynomial_order)));
        self.focus_lost();
    }

    /// Java public synchronized `updateAlignedStackBinning()`.  Only called on
    /// the EDT here (the manager posts its call), so `synchronized` has no
    /// counterpart.
    pub fn update_aligned_stack_binning(&self) {
        let aligned_stack: &Arc<FileType> = &file_type::CLASS.aligned_stack;
        let manager: &'static dyn BaseManager = self.manager;
        if aligned_stack.exists(Some(manager), Some(self.axis_id)) {
            match utilities::get_stack_binning_for_file_type_boolean(
                manager,
                self.axis_id,
                aligned_stack,
                true,
            ) {
                Ok(binning) => self.aligned_stack_binning.set(Some(binning)),
                Err(_) => {
                    // Most likely it's in the middle of creating the file. There will be
                    // another update after newst is complete.
                    return;
                }
            }
        } else {
            self.aligned_stack_binning.set(None);
        }
        // SwingUtilities.invokeLater(new CallFocusLost()).
        if let Some(this) = self.this.upgrade() {
            let call_focus_lost = EdtRef::new(this);
            // Java inner class `CallFocusLost.run()`: focusLost(null).
            event_queue::invoke_later(move || {
                call_focus_lost.get().focus_lost();
            });
        }
    }

    /// Java `focusGained(FocusEvent)`: empty.
    pub fn focus_gained(&self) {}

    /// Java synchronized `focusLost(FocusEvent)`.  The event is never read
    /// (every caller in the source passes null), so the Rust method takes none.
    pub fn focus_lost(&self) {
        // Not looking at event.getComponent because only ltfFiducialDiameter has a focus
        // listener.
        let l_fiducial_diameter1 = self.l_fiducial_diameter1.get_component();
        let l_fiducial_diameter2 = self.l_fiducial_diameter2.get_component();
        if let Some(aligned_stack_binning) = self.aligned_stack_binning.get() {
            l_fiducial_diameter1.set_text("");
            l_fiducial_diameter2.set_text("");
            let fiducial_diameter = self.ltf_fiducial_diameter.get_text_void();
            if !utilities::is_empty(fiducial_diameter.as_deref()) {
                let d_fiducial_diameter = converter::to_double(fiducial_diameter.as_deref());
                if let Some(d_fiducial_diameter) = d_fiducial_diameter {
                    // Create a message containing the unbinned diameter rounded to one
                    // decimal place.
                    let unbinned_diameter = utilities::java_lang_math_round(
                        d_fiducial_diameter * aligned_stack_binning as f64 * 10.0,
                    ) as f64
                        / 10.0;
                    l_fiducial_diameter1.set_text(&format!(
                        "(corresponds to unbinned diameter of {}",
                        // Java string concatenation of a `double`: `Double.toString`.
                        java_lang_double_to_string(unbinned_diameter)
                    ));
                    l_fiducial_diameter2
                        .set_text(&format!("with current binning of {aligned_stack_binning})"));
                }
            }
        } else {
            l_fiducial_diameter1.set_text("Enter diameter in unbinned pixels");
            l_fiducial_diameter2.set_text("");
        }
    }

    /// Java package-private `done()`.
    pub fn done(&self) {
        self.btn_ccd_eraser
            .remove_action_listener(&self.action_listener);
        self.btn_use_ccd_eraser
            .remove_action_listener(&self.action_listener);
    }

    /// Java final package-private `setParameters(ReconScreenState)`.
    pub fn set_parameters_recon_screen_state(&self, screen_state: &ReconScreenState) {
        self.btn_ccd_eraser.set_button_state(
            screen_state.get_button_state(self.btn_ccd_eraser.get_button_state_key().as_deref()),
        );
        self.btn_use_ccd_eraser.set_button_state(
            screen_state
                .get_button_state(self.btn_use_ccd_eraser.get_button_state_key().as_deref()),
        );
    }

    /// Java private `setToolTipText()`.
    fn set_tool_tip_text(&self) {
        // Java `ReadOnlyAutodoc autodoc = null;` then the try/catch.
        let mut autodoc: *const dyn ReadOnlyAutodoc = std::ptr::null::<Autodoc>();
        let manager: &'static dyn BaseManager = self.manager;
        match unsafe {
            autodoc_factory::get_instance(
                Some(manager),
                Some(autodoc_factory::CCDERASER),
                self.axis_id,
                false,
            )
        } {
            Ok(instance) => autodoc = instance as *const Autodoc,
            // `catch (final LockException except) {}`.
            Err(LogFileError::Lock(_)) => {}
            // `catch (final LogFileException | IOException except)`:
            // `except.printStackTrace()`.
            Err(except) => eprintln!("{}", except),
        }
        // SAFETY: `autodoc` is null or an autodoc the factory keeps for the life of
        // the process.
        let autodoc: Option<&dyn ReadOnlyAutodoc> = if autodoc.is_null() {
            None
        } else {
            Some(unsafe { &*autodoc })
        };
        self.ltf_fiducial_diameter.set_tool_tip_text(Some(
            "The diameter, in pixels of the aligned stack, that will be erased around each point.",
        ));
        self.cbsp_expand_circle_iterations.set_tool_tip_text(
            etomo_autodoc::get_tooltip(
                autodoc,
                Some(const_ccd_eraser_param::EXPAND_CIRCLE_ITERATIONS_KEY),
            )
            .as_deref(),
        );
        self.rb_polynomial_order_use_mean
            .set_tool_tip_text_string(Some(
                "Fill the erased pixels with the mean of surrounding pixels.",
            ));
        self.rb_polynomial_order_fill_noise
            .set_tool_tip_text_string(Some(
                "Fill the erased pixels with noise matching the SD of surrounding pixels.",
            ));
        self.rb_polynomial_order_fit_a_plane
            .set_tool_tip_text_string(Some(
                "Fill the erased pixels with a gradient based on plane fit to surrounding pixels.",
            ));
        self.btn_ccd_eraser.set_tool_tip_text(Some(
            "Run Ccderaser on the aligned stack to erase around model points.",
        ));
        self.btn_3dmod_ccd_eraser.set_tool_tip_text(Some(
            "View the results of running Ccderaser on the aligned stack along with the \
             _erase.fid model.",
        ));
        self.btn_use_ccd_eraser.set_tool_tip_text(Some(
            "Replace the full aligned stack (.ali) with the erased stack (_erase.ali).",
        ));
    }

    /// Java inner class `CcdEraserPanelActionListener.actionPerformed(ActionEvent)`:
    /// `adaptee.action(event.getActionCommand(), null, null)`.
    fn action_performed(&self, event: &ActionEvent) {
        self.action(event.get_action_command().unwrap_or(""), None, None);
    }
}

impl Run3dmodButtonContainer for CcdEraserBeadsPanel {
    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    fn action(
        &self,
        command: &str,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        if Some(command) == self.btn_ccd_eraser.get_action_command().as_deref() {
            let display: ProcessResultDisplayHandle = self.btn_ccd_eraser.clone();
            self.manager
                .gold_eraser_process_result_display_process_series_deferred3dmod_button_run3dmod_menu_options_axis_id_dialog_type_ccd_eraser_display(
                    Some(display),
                    None,
                    deferred_3dmod_button,
                    run_3dmod_menu_options,
                    self.axis_id,
                    self.dialog_type,
                    self,
                );
        } else if Some(command) == self.btn_3dmod_ccd_eraser.get_action_command().as_deref() {
            self.manager
                .imod_erased_fiducials(run_3dmod_menu_options, self.axis_id);
        } else if Some(command) == self.btn_use_ccd_eraser.get_action_command().as_deref() {
            let display: ProcessResultDisplayHandle = self.btn_use_ccd_eraser.clone();
            self.manager.use_ccd_eraser(
                Some(display),
                self.axis_id,
                self.dialog_type,
                CCD_ERASER_LABEL,
            );
        }
    }
}

impl ProcessDisplay for CcdEraserBeadsPanel {
    fn as_ccd_eraser_display(&self) -> Option<&dyn CcdEraserDisplay> {
        Some(self)
    }
}

impl CcdEraserDisplay for CcdEraserBeadsPanel {
    /// Java public `getParameters(CCDEraserParam, boolean)`: fills in
    /// golderaser.com.
    fn get_parameters(&self, param: &mut CCDEraserParam, do_validation: bool) -> bool {
        // try { ... } catch (FieldValidationFailedException e) { return false; }
        let manager: &'static dyn BaseManager = self.manager;
        param.set_input_file(
            file_type::CLASS
                .aligned_stack
                .get_file_name(Some(manager), Some(self.axis_id))
                .as_deref(),
        );
        param.set_model_file(
            file_type::CLASS
                .ccd_eraser_beads_input_model
                .get_file_name(Some(manager), Some(self.axis_id))
                .as_deref(),
        );
        param.set_output_file_type(&file_type::CLASS.erased_beads_stack);
        let mut fiducial_diameter = EtomoNumber::new_with_type(Some(Type::Double));
        let Ok(text) = self.ltf_fiducial_diameter.get_text_boolean(do_validation) else {
            return false;
        };
        fiducial_diameter.set_string(text.as_deref());
        param.set_better_radius(fiducial_diameter.get_double() / 2.0);
        if self.cbsp_expand_circle_iterations.is_selected() {
            param.set_expand_circle_iterations(&self.cbsp_expand_circle_iterations.get_value());
        } else {
            param.reset_expand_circle_iterations();
        }
        param.set_polynomial_order(self.get_polynomial_order().as_deref());
        param.validate()
    }

    /// Java public `getParameters(MakecomfileParam, boolean)`.
    fn get_parameters_makecomfile(
        &self,
        param: &mut MakecomfileParam,
        do_validation: bool,
    ) -> bool {
        match self.ltf_fiducial_diameter.get_text_boolean(do_validation) {
            Ok(bead_size) => param.set_bead_size(bead_size.as_deref()),
            Err(_) => return false,
        }
        true
    }
}

/// Java `private static final class PolynomialOrder implements EnumeratedType`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PolynomialOrder {
    /// Java `USE_MEAN = new PolynomialOrder(0)`.
    UseMean,
    /// Java `FIT_A_PLANE = new PolynomialOrder(1)`.
    FitAPlane,
    /// Java `FILL_WITH_NOISE = new PolynomialOrder(-1)`.
    FillWithNoise,
}

impl PolynomialOrder {
    /// Java private static final `DEFAULT = USE_MEAN`.
    const DEFAULT: PolynomialOrder = PolynomialOrder::UseMean;

    /// Java field `value`, set by the constructor.
    fn value(self) -> EtomoNumber {
        let mut value = EtomoNumber::new();
        value.set_int(match self {
            Self::UseMean => 0,
            Self::FitAPlane => 1,
            Self::FillWithNoise => -1,
        });
        value
    }

    /// Java private static `getInstance(int)`.
    fn get_instance_int(value: i32) -> PolynomialOrder {
        if Self::UseMean.value().equals_int(value) {
            return Self::UseMean;
        }
        if Self::FitAPlane.value().equals_int(value) {
            return Self::FitAPlane;
        }
        if Self::FillWithNoise.value().equals_int(value) {
            return Self::FillWithNoise;
        }
        Self::DEFAULT
    }

    /// Java private static `getInstance(String)`.
    fn get_instance_string(value: Option<&str>) -> PolynomialOrder {
        if Self::UseMean.value().equals_string(value) {
            return Self::UseMean;
        }
        if Self::FitAPlane.value().equals_string(value) {
            return Self::FitAPlane;
        }
        if Self::FillWithNoise.value().equals_string(value) {
            return Self::FillWithNoise;
        }
        Self::DEFAULT
    }
}

impl EnumeratedType for PolynomialOrder {
    /// Java `isDefault()`.
    fn is_default(&self) -> bool {
        *self == Self::DEFAULT
    }

    /// Java `getValue()`.
    fn get_value(&self) -> ConstEtomoNumber {
        self.value().base
    }

    /// Java `getLabel()`: null.
    fn get_label(&self) -> Option<String> {
        None
    }
}

/// Java `toString()`: `value.toString()`.
impl fmt::Display for PolynomialOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polynomial_order_lookup() {
        assert_eq!(PolynomialOrder::FillWithNoise.to_string(), "-1");
        assert_eq!(
            PolynomialOrder::get_instance_int(1),
            PolynomialOrder::FitAPlane
        );
        assert_eq!(
            PolynomialOrder::get_instance_string(Some("-1")),
            PolynomialOrder::FillWithNoise
        );
        assert_eq!(
            PolynomialOrder::get_instance_string(None),
            PolynomialOrder::UseMean
        );
        assert!(PolynomialOrder::UseMean.is_default());
    }
}
