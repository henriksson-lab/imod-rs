//! `IMOD/Etomo/src/etomo/ui/swing/Ctf3dPanel.java`.
//!
//! Java `final class Ctf3dPanel implements Ctf3dSetupDisplay, FocusListener,
//! Expandable, ActionListener, Run3dmodButtonContainer, UIComponent,
//! SwingComponent`: the "3D CTF" method panel of the Tomogram Generation
//! dialog.  An EDT object created as `Rc<Self>` by [`Ctf3dPanel::get_instance`];
//! every method takes `&self`.
//!
//! The panel is its own `ActionListener` in the Java; here that listener is one
//! closure ([`Ctf3dPanel::action_listener`]) holding a weak reference, created
//! once so the `remove*ActionListener` calls in `done()` get the same `Rc`.
//! The `FocusListener` methods are [`Ctf3dPanel::focus_gained`] /
//! [`Ctf3dPanel::focus_lost`] without an event; the registered listener
//! (`focus_listener`) holds this panel weakly and calls the one the event's kind
//! selects.
//!
//! The parent dialog is held weakly (it owns this panel); the dialog is built
//! before its panels, so it can be reached from this constructor.

use std::cell::Cell;
use std::rc::{Rc, Weak};

use super::button_control_text_efield::ButtonControlTextEfield;
use super::check_box_efield::CheckBoxEfield;
use super::ctf3d_setup_display::Ctf3dSetupDisplay;
use super::deferred_3dmod_button::Deferred3dmodButton;
use super::etched_border::EtchedBorder;
use super::etomo_button_group::EtomoButtonGroup;
use super::expand_button::ExpandButton;
use super::expandable::Expandable;
use super::file_chooser;
use super::global_expand_button::GlobalExpandButton;
use super::label::Label;
use super::multi_line_button::MultiLineButton;
use super::panel_header::PanelHeader;
use super::process_control_panel;
use super::radio_ebutton::RadioEbutton;
use super::run_3dmod_button::Run3dmodButton;
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use super::spinner::Spinner;
use super::swing_component::SwingComponent;
use super::text_efield::TextEfield;
use super::tomogram_generation_dialog::TomogramGenerationDialog;
use super::tomogram_generation_parent::TomogramGenerationParent;
use super::ui_harness;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::ctf3d_setup_param::{self, Ctf3dSetupParam};
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, FocusEvent, FocusListener, JComponent};
use crate::imod::etomo::logic::converter;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::storage::autodoc::autodoc::Autodoc;
use crate::imod::etomo::storage::autodoc::autodoc_factory;
use crate::imod::etomo::storage::autodoc::read_only_autodoc::ReadOnlyAutodoc;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_meta_data::ConstMetaData;
use crate::imod::etomo::r#type::const_panel_header_settings::ConstPanelHeaderSettings;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::etomo_autodoc;
use crate::imod::etomo::r#type::meta_data::MetaData;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::r#type::recon_screen_state::ReconScreenState;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::shared_strings;
use crate::imod::etomo::ui::ui_component::UIComponent;

/// Java private static final `NUMBER_OF_SLABS_LABEL1`.
const NUMBER_OF_SLABS_LABEL1: &str = " (";
/// Java private static final `NUMBER_OF_SLABS_EMPTY`.
const NUMBER_OF_SLABS_EMPTY: &str = "?";
/// Java private static final `NUMBER_OF_SLABS_LABEL2`.
const NUMBER_OF_SLABS_LABEL2: &str = " slabs) ";
/// Java package-private static final `RUN_BUTTON_LABEL`.
pub const RUN_BUTTON_LABEL: &str = "Generate CTF-corrected Tomogram";
/// Java package-private static final `USE_BUTTON_LABEL`.
pub const USE_BUTTON_LABEL: &str = "Use CTF-corrected Tomogram";
/// Java package-private static final `SLAB_THICKNESS_IN_NM_LABEL`.
pub const SLAB_THICKNESS_IN_NM_LABEL: &str = "Thickness of slab for each CTF correction (nm): ";
/// Java private static final `ERASE_FIDUCIALS_LABEL`.
const ERASE_FIDUCIALS_LABEL: &str = "Erase gold";
/// Java private static final `FILTER_IN_2D_LABEL`.
const FILTER_IN_2D_LABEL: &str = "Apply 2D filter";

/// Java `final class Ctf3dPanel`.
pub struct Ctf3dPanel {
    /// Rust-only: Java `this`.
    self_ref: Weak<Ctf3dPanel>,
    /// Java private final `pnlRoot = new JPanel()`.
    pnl_root: Rc<JComponent>,
    /// Java private final `tfSlabThicknessInNm`.
    tf_slab_thickness_in_nm: Rc<TextEfield>,
    /// Java private final `lNumberOfSlabs` (a `Label`).
    l_number_of_slabs: Rc<Label>,
    /// Java private final `cbRunSlabsInParallel`.
    cb_run_slabs_in_parallel: Rc<CheckBoxEfield>,
    /// Java private final `cbEraseFiducials`.
    cb_erase_fiducials: Rc<CheckBoxEfield>,
    /// Java private final `cbFilterIn2D`.
    cb_filter_in_2d: Rc<CheckBoxEfield>,
    /// Java private final `cbUseUnalignedImages`.
    cb_use_unaligned_images: Rc<CheckBoxEfield>,
    /// Java private final `spFourierReduceByFactor`.
    sp_fourier_reduce_by_factor: Rc<Spinner>,
    /// Java private final `pnlXAxisTiltedSlices = new JPanel()`.
    pnl_x_axis_tilted_slices: Rc<JComponent>,
    /// Java private final `bgXAxisTiltedSlices`.
    bg_x_axis_tilted_slices: Rc<EtomoButtonGroup>,
    /// Java private final `rbXAxisTiltedSlices`.
    rb_x_axis_tilted_slices: Rc<RadioEbutton>,
    /// Java private final `rbVerticalSlices`.
    rb_vertical_slices: Rc<RadioEbutton>,
    /// Java private final `rbOldStyleXtilting`.
    rb_old_style_xtilting: Rc<RadioEbutton>,
    /// Java private final `bctfTemporaryDirectory`.
    bctf_temporary_directory: Rc<ButtonControlTextEfield>,
    /// Java private final `btn3dmodCtf3d`.
    btn_3dmod_ctf3d: Rc<Run3dmodButton>,
    /// Java private final `pnlCtf3dSetupBody = new JPanel()`.
    pnl_ctf3d_setup_body: Rc<JComponent>,
    /// Java private final `lEraseFiducials` (a `Label`).
    l_erase_fiducials: Rc<Label>,
    /// Java private final `lFilterIn2D` (a `Label`).
    l_filter_in_2d: Rc<Label>,
    /// Java private final `lctf3d` (a `JLabel`).
    lctf3d: Rc<JComponent>,
    /// Java private final `cbAdjustForAlignZShift`.
    cb_adjust_for_align_z_shift: Rc<CheckBoxEfield>,

    /// Java private final `phCtf3dSetup`.
    ph_ctf3d_setup: Rc<PanelHeader>,
    /// Java private final `btnCtf3dSetup`.
    btn_ctf3d_setup: Rc<Run3dmodButton>,
    /// Java private final `btnUseCtf3d`.
    btn_use_ctf3d: Rc<MultiLineButton>,
    /// Java private final `manager`.
    manager: &'static ApplicationManager,
    /// Java private final `axisID`.
    axis_id: AxisID,
    /// Java private final `parent` (held weakly: the dialog owns this panel).
    parent: Weak<TomogramGenerationDialog>,
    /// Java private final `metaData`.
    meta_data: &'static MetaData,

    /// Java private `numberOfSlabs` (a `Long`, null is `None`).
    number_of_slabs: Cell<Option<i64>>,

    /// Rust-only: Java `this` as the `ActionListener` it registers.
    action_listener: ActionListener,
    /// Rust-only: Java `this` as the `FocusListener` it registers.
    focus_listener: FocusListener,
}

impl Ctf3dPanel {
    /// Java private constructor `Ctf3dPanel(ApplicationManager, AxisID,
    /// TomogramGenerationDialog)`.
    fn new(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        parent: Weak<TomogramGenerationDialog>,
    ) -> Rc<Ctf3dPanel> {
        Rc::new_cyclic(|self_ref: &Weak<Ctf3dPanel>| {
            let container: Weak<dyn Run3dmodButtonContainer> = self_ref.clone();
            // Field initializers, in declaration order.
            let pnl_root = JComponent::new_panel();
            let tf_slab_thickness_in_nm = TextEfield::get_labeled_instance(
                Some(SLAB_THICKNESS_IN_NM_LABEL),
                Some(FieldType::Integer),
            );
            let l_number_of_slabs = Label::get_named_instance(Some(SLAB_THICKNESS_IN_NM_LABEL));
            let cb_run_slabs_in_parallel =
                CheckBoxEfield::get_instance(Some("Compute slabs in parallel"));
            let cb_erase_fiducials = CheckBoxEfield::get_instance(Some(ERASE_FIDUCIALS_LABEL));
            let cb_filter_in_2d = CheckBoxEfield::get_instance(Some(FILTER_IN_2D_LABEL));
            let cb_use_unaligned_images =
                CheckBoxEfield::get_instance(Some("Reconstruct from raw images"));
            let sp_fourier_reduce_by_factor = Spinner::get_labeled_instance_string_int_int_int(
                Some("Reduced by: "),
                ctf3d_setup_param::FOURIER_REDUCE_BY_FACTOR_DEFAULT,
                ctf3d_setup_param::FOURIER_REDUCE_BY_FACTOR_MIN,
                ctf3d_setup_param::FOURIER_REDUCE_BY_FACTOR_MAX,
            );
            let pnl_x_axis_tilted_slices = JComponent::new_panel();
            let bg_x_axis_tilted_slices = EtomoButtonGroup::new();
            let rb_x_axis_tilted_slices = RadioEbutton::get_instance(
                Some("Let programs decide whether to interpolate from vertical slices"),
                Some(&*bg_x_axis_tilted_slices),
            );
            let rb_vertical_slices = RadioEbutton::get_instance(
                Some("Always use direct backprojection into X-tilted output slice"),
                Some(&*bg_x_axis_tilted_slices),
            );
            let rb_old_style_xtilting = RadioEbutton::get_instance(
                Some("Always make vertical slices and interpolate to get X-tilted output"),
                Some(&*bg_x_axis_tilted_slices),
            );
            let bctf_temporary_directory =
                ButtonControlTextEfield::get_labeled_file_instance_string_boolean_boolean(
                    Some("Temporary directory: "),
                    true,
                    false,
                );
            let btn_3dmod_ctf3d =
                Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                    Some("View Tomogram In 3dmod"),
                    Some(container),
                );
            let pnl_ctf3d_setup_body = JComponent::new_panel();
            let l_erase_fiducials = Label::new_string_string(
                Some(ERASE_FIDUCIALS_LABEL),
                Some("   Parameters must already be set for gold erasing"),
            );
            let l_filter_in_2d = Label::new_string_string(
                Some(FILTER_IN_2D_LABEL),
                Some("   Parameters must already be set for 2-D filtering"),
            );
            let lctf3d = JComponent::new_label(&format!(
                "Parameters must already be set in Final Aligned Stack {}",
                shared_strings::CTF_CORRECTION_LABEL
            ));
            let cb_adjust_for_align_z_shift = CheckBoxEfield::get_instance(Some(
                "Adjust for Z shift in fine alignment and positioning",
            ));
            // Java `this` as the ActionListener.
            let adaptee = self_ref.clone();
            let action_listener: ActionListener = Rc::new(move |event: &ActionEvent| {
                if let Some(adaptee) = adaptee.upgrade() {
                    adaptee.action_performed(Some(event));
                }
            });
            // Java `this` as the FocusListener.
            let adaptee = self_ref.clone();
            let focus_listener: FocusListener = Rc::new(move |event: &FocusEvent| {
                if let Some(adaptee) = adaptee.upgrade() {
                    if event.gained {
                        adaptee.focus_gained();
                    } else {
                        adaptee.focus_lost();
                    }
                }
            });

            // Constructor body.
            let meta_data = manager.get_meta_data();
            let expandable: Weak<dyn Expandable> = self_ref.clone();
            let ph_ctf3d_setup = PanelHeader::get_advanced_basic_only_instance(
                Some("3D CTF Correction Parameters"),
                Some(expandable),
                Some(DialogType::TomogramGeneration),
                parent.upgrade().map(|parent| parent.get_advanced_button()),
                false,
            );
            let display_factory = manager.get_process_result_display_factory(axis_id);
            // Java casts `(Run3dmodButton) displayFactory.getCtf3dSetup()` and
            // `(MultiLineButton) displayFactory.getUseCtf3d()`; the factory returns
            // the concrete buttons.
            let btn_ctf3d_setup = display_factory.get_ctf3d_setup();
            let btn_use_ctf3d = display_factory.get_use_ctf3d();
            Ctf3dPanel {
                self_ref: self_ref.clone(),
                pnl_root,
                tf_slab_thickness_in_nm,
                l_number_of_slabs,
                cb_run_slabs_in_parallel,
                cb_erase_fiducials,
                cb_filter_in_2d,
                cb_use_unaligned_images,
                sp_fourier_reduce_by_factor,
                pnl_x_axis_tilted_slices,
                bg_x_axis_tilted_slices,
                rb_x_axis_tilted_slices,
                rb_vertical_slices,
                rb_old_style_xtilting,
                bctf_temporary_directory,
                btn_3dmod_ctf3d,
                pnl_ctf3d_setup_body,
                l_erase_fiducials,
                l_filter_in_2d,
                lctf3d,
                cb_adjust_for_align_z_shift,
                ph_ctf3d_setup,
                btn_ctf3d_setup,
                btn_use_ctf3d,
                manager,
                axis_id,
                parent,
                meta_data,
                number_of_slabs: Cell::new(None),
                action_listener,
                focus_listener,
            }
        })
    }

    /// Java static `getInstance(ApplicationManager, AxisID,
    /// TomogramGenerationDialog)`.
    pub fn get_instance(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        parent: Weak<TomogramGenerationDialog>,
    ) -> Rc<Ctf3dPanel> {
        let instance = Ctf3dPanel::new(manager, axis_id, parent);
        instance.create_panel();
        instance.set_tooltips();
        instance.add_listeners();
        instance
    }

    /// The parent dialog (Java field `parent`, never null in the Java).
    fn parent(&self) -> Option<Rc<TomogramGenerationDialog>> {
        self.parent.upgrade()
    }

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        // panels
        let pnl_slab_thickness_in_nm = JComponent::new_panel();
        let pnl_run_slabs_in_parallel = JComponent::new_panel();
        let pnl_buttons = JComponent::new_panel();
        let pnl_use_unaligned_images = JComponent::new_panel();
        let pnl_erase_fiducials = JComponent::new_panel();
        let pnl_filter_in_2d = JComponent::new_panel();
        // init
        self.tf_slab_thickness_in_nm.set_required(true);
        self.tf_slab_thickness_in_nm.set_columns();
        self.tf_slab_thickness_in_nm
            .set_text_int(ctf3d_setup_param::SLAB_THICKNESS_IN_NM_DEFAULT);
        let container: Weak<dyn Run3dmodButtonContainer> = self.self_ref.clone();
        self.btn_ctf3d_setup.set_container(Some(container));
        let deferred: Rc<dyn Deferred3dmodButton> = self.btn_3dmod_ctf3d.clone();
        self.btn_ctf3d_setup
            .set_deferred_3dmod_button_deferred_3dmod_button(Some(deferred));
        self.bctf_temporary_directory
            .set_select_file_dir(self.manager.get_property_user_dir().as_deref());
        self.bctf_temporary_directory
            .set_file_selection_mode(file_chooser::DIRECTORIES_ONLY);
        self.bctf_temporary_directory
            .set_limit_displayed_file_path(45);
        self.lctf3d
            .set_foreground(Some(process_control_panel::COLOR_COMPLETE));
        self.l_filter_in_2d
            .get_component()
            .set_foreground(Some(process_control_panel::COLOR_COMPLETE));
        self.l_erase_fiducials
            .get_component()
            .set_foreground(Some(process_control_panel::COLOR_COMPLETE));
        // Root
        // Swing layout: pnlRoot BoxLayout Y_AXIS, etched border.
        self.pnl_root.add(&self.ph_ctf3d_setup.get_component());
        self.pnl_root.add(&self.pnl_ctf3d_setup_body);
        // Ctf3dSetupBody
        // Swing layout: pnlCtf3dSetupBody BoxLayout Y_AXIS; rigid areas x0_y2
        // before lctf3d, x0_y10 after it, x0_y5 around the X-tilted slices
        // panel, the temporary directory and the buttons.
        self.pnl_ctf3d_setup_body.add(&self.lctf3d);
        self.pnl_ctf3d_setup_body.add(&pnl_slab_thickness_in_nm);
        self.pnl_ctf3d_setup_body.add(&pnl_run_slabs_in_parallel);
        self.pnl_ctf3d_setup_body.add(&pnl_erase_fiducials);
        self.pnl_ctf3d_setup_body.add(&pnl_filter_in_2d);
        self.pnl_ctf3d_setup_body.add(&pnl_use_unaligned_images);
        self.pnl_ctf3d_setup_body
            .add(&self.cb_adjust_for_align_z_shift.get_component());
        self.pnl_ctf3d_setup_body
            .add(&self.pnl_x_axis_tilted_slices);
        self.pnl_ctf3d_setup_body
            .add(&SwingComponent::get_component(
                &*self.bctf_temporary_directory,
            ));
        self.pnl_ctf3d_setup_body.add(&pnl_buttons);
        // SlabThicknessInNm
        // Swing layout: BoxLayout X_AXIS; horizontal glue at the end.
        pnl_slab_thickness_in_nm.add(&self.tf_slab_thickness_in_nm.get_component());
        pnl_slab_thickness_in_nm.add(&self.l_number_of_slabs.get_component());
        // RunSlabsInParallel
        // Swing layout: BoxLayout X_AXIS (the Java adds its horizontal glue to
        // pnlSlabThicknessInNm instead).
        pnl_run_slabs_in_parallel.add(&self.cb_run_slabs_in_parallel.get_component());
        // EraseFiducials
        // Swing layout: BoxLayout X_AXIS.
        pnl_erase_fiducials.add(&self.cb_erase_fiducials.get_component());
        pnl_erase_fiducials.add(&self.l_erase_fiducials.get_component());
        // FilterIn2D
        // Swing layout: BoxLayout X_AXIS.
        pnl_filter_in_2d.add(&self.cb_filter_in_2d.get_component());
        pnl_filter_in_2d.add(&self.l_filter_in_2d.get_component());
        // UseUnalignedImages
        // Swing layout: BoxLayout X_AXIS; rigid areas x15_y0 and x1_y0.
        pnl_use_unaligned_images.add(&self.cb_use_unaligned_images.get_component());
        pnl_use_unaligned_images.add(&self.sp_fourier_reduce_by_factor.get_component());
        // XAxisTiltedSlices
        // Swing layout: BoxLayout Y_AXIS.
        self.pnl_x_axis_tilted_slices.set_border_title(
            EtchedBorder::new(Some("Computing X-axis Tilted Slices"))
                .get_border()
                .get_title()
                .as_deref(),
        );
        self.pnl_x_axis_tilted_slices
            .add(&self.rb_x_axis_tilted_slices.get_component());
        self.pnl_x_axis_tilted_slices
            .add(&self.rb_vertical_slices.get_component());
        self.pnl_x_axis_tilted_slices
            .add(&self.rb_old_style_xtilting.get_component());
        // Buttons
        // Swing layout: BoxLayout X_AXIS.
        pnl_buttons.add(&SwingComponent::get_component(&*self.btn_ctf3d_setup));
        pnl_buttons.add(&SwingComponent::get_component(&*self.btn_3dmod_ctf3d));
        pnl_buttons.add(&self.btn_use_ctf3d.get_component());
        // align
        // Swing layout: pnlRoot.setAlignmentX(LEFT_ALIGNMENT);
        // UIUtilities.alignComponentsX(pnlRoot / pnlCtf3dSetupBody, LEFT_ALIGNMENT).
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self) {
        self.tf_slab_thickness_in_nm
            .add_focus_listener(self.focus_listener.clone());
        let Some(parent) = self.parent() else {
            return;
        };
        let expandable: Weak<dyn Expandable> = self.self_ref.clone();
        parent.get_advanced_button().register_expandable(expandable);
        parent.add_axis_tilt_focus_listener(self.focus_listener.clone());
        parent.add_use_local_alignment_action_listener(self.action_listener.clone());
        parent.add_use_z_factors_action_listener(self.action_listener.clone());
        self.cb_use_unaligned_images
            .add_action_listener(self.action_listener.clone());
        self.btn_ctf3d_setup
            .add_action_listener(self.action_listener.clone());
        self.btn_3dmod_ctf3d
            .add_action_listener(self.action_listener.clone());
        self.btn_use_ctf3d
            .add_action_listener(self.action_listener.clone());
        self.cb_erase_fiducials
            .add_action_listener(self.action_listener.clone());
        self.cb_filter_in_2d
            .add_action_listener(self.action_listener.clone());
    }

    /// Java `focusGained(FocusEvent)`: empty.
    pub fn focus_gained(&self) {}

    /// Java `focusLost(FocusEvent)`.  The Java also calls it directly with a
    /// null event.
    pub fn focus_lost(&self) {
        // numberOfSlabs
        let tomo_thickness = self.parent().and_then(|parent| parent.get_tomo_thickness());
        self.number_of_slabs
            .set(Ctf3dSetupParam::calc_number_of_slabs(
                tomo_thickness,
                self.meta_data.get_pixel_size(),
                converter::to_long(self.tf_slab_thickness_in_nm.get_text_void().as_deref()),
            ));
        if let Some(number_of_slabs) = self.number_of_slabs.get() {
            self.l_number_of_slabs.get_component().set_text(&format!(
                "{NUMBER_OF_SLABS_LABEL1}{number_of_slabs}{NUMBER_OF_SLABS_LABEL2}"
            ));
        } else {
            self.l_number_of_slabs.get_component().set_text(&format!(
                "{NUMBER_OF_SLABS_LABEL1}{NUMBER_OF_SLABS_EMPTY}{NUMBER_OF_SLABS_LABEL2}"
            ));
        }
        // XAxisTiltedSlices
        self.update_display();
    }

    /// Java `actionPerformed(ActionEvent)` (the panel is its own
    /// `ActionListener`).
    pub fn action_performed(&self, event: Option<&ActionEvent>) {
        // Java `action(event != null ? event.getActionCommand() : null, null,
        // null)`; `action` then dereferences the command.  A null command is
        // treated as matching no button (the final `else` branch).
        let action_command = event.and_then(|event| event.get_action_command().map(str::to_owned));
        self.action_option(action_command.as_deref(), None, None);
    }

    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)` with a
    /// possibly-null command (see [`Ctf3dPanel::action_performed`]).
    fn action_option(
        &self,
        action_command: Option<&str>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        if action_command.is_some()
            && action_command == self.btn_ctf3d_setup.get_action_command().as_deref()
        {
            let Some(parent) = self.parent() else {
                return;
            };
            let display: ProcessResultDisplayHandle = self.btn_ctf3d_setup.clone();
            self.manager.ctf3d_setup(
                Some(display),
                None,
                deferred_3dmod_button,
                run_3dmod_menu_options,
                self,
                self.axis_id,
                DialogType::TomogramGeneration,
                Some(parent.get_processing_method()),
            );
        } else if action_command.is_some()
            && action_command == self.btn_3dmod_ctf3d.get_action_command().as_deref()
        {
            self.manager
                .open_ctf3d(self.axis_id, run_3dmod_menu_options);
        } else if action_command.is_some()
            && action_command == self.btn_use_ctf3d.get_action_command().as_deref()
        {
            let display: ProcessResultDisplayHandle = self.btn_use_ctf3d.clone();
            self.manager.use_ctf3d(
                Some(display),
                self.btn_ctf3d_setup
                    .get_unformatted_label()
                    .as_deref()
                    .unwrap_or("null"),
                self.axis_id,
                DialogType::TomogramGeneration,
            );
        } else {
            self.update_display();
        }
    }

    /// Java package-private `updateDisplay()`.
    pub fn update_display(&self) {
        let advanced = self.ph_ctf3d_setup.is_advanced();
        self.pnl_x_axis_tilted_slices.set_visible(advanced);
        self.sp_fourier_reduce_by_factor
            .set_enabled(self.cb_use_unaligned_images.is_selected());
        self.cb_adjust_for_align_z_shift.set_visible(advanced);
        self.l_erase_fiducials
            .get_component()
            .set_enabled(self.cb_erase_fiducials.is_selected());
        self.l_filter_in_2d
            .get_component()
            .set_enabled(self.cb_filter_in_2d.is_selected());
        // XAxisTiltedSlices
        let parent = self.parent();
        let x_axis_tilt = converter::to_double(
            parent
                .as_ref()
                .and_then(|parent| parent.get_x_axis_tilt())
                .as_deref(),
        );
        let x_axis_tilted_slices = x_axis_tilt.is_some_and(|x_axis_tilt| x_axis_tilt != 0.0)
            && parent.as_ref().is_some_and(|parent| {
                !parent.is_use_local_alignment() && !parent.is_use_z_factors()
            });
        self.pnl_x_axis_tilted_slices
            .set_enabled(x_axis_tilted_slices);
        self.rb_x_axis_tilted_slices
            .set_enabled(x_axis_tilted_slices);
        self.rb_vertical_slices.set_enabled(x_axis_tilted_slices);
        self.rb_old_style_xtilting.set_enabled(x_axis_tilted_slices);
        let manager: &'static dyn BaseManager = self.manager;
        ui_harness::with(|harness| {
            harness.pack_axis_id_base_manager(Some(self.axis_id), Some(manager))
        });
    }

    /// Java `getComponent()` (implements `SwingComponent`).
    pub fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.clone()
    }

    /// Java package-private `msgMethodChanged()`.
    pub fn msg_method_changed(&self) {
        self.pnl_root
            .set_visible(self.parent().is_some_and(|parent| parent.is_ctf3d()));
    }

    /// Java package-private `done()`.
    pub fn done(&self) {
        self.tf_slab_thickness_in_nm
            .remove_focus_listener(&self.focus_listener);
        if let Some(parent) = self.parent() {
            let expandable: Weak<dyn Expandable> = self.self_ref.clone();
            parent.get_advanced_button().deregister(&expandable);
            parent.remove_axis_tilt_focus_listener(&self.focus_listener);
            parent.remove_use_local_alignment_action_listener(&self.action_listener);
            parent.remove_use_z_factors_action_listener(&self.action_listener);
        }
        self.btn_ctf3d_setup
            .remove_action_listener(&self.action_listener);
        self.btn_3dmod_ctf3d
            .remove_action_listener(&self.action_listener);
        self.btn_use_ctf3d
            .remove_action_listener(&self.action_listener);
    }

    /// Java package-private `setParameters(ConstMetaData)`.
    pub fn set_parameters_const_meta_data(&self, meta_data: &dyn ConstMetaData) {
        self.sp_fourier_reduce_by_factor.set_value_string(Some(
            &meta_data.get_gen_ctf_3d_fourier_reduce_by_factor(self.axis_id),
        ));
        if meta_data.is_gen_ctf_3d_vertical_slices(self.axis_id) {
            self.rb_vertical_slices.set_selected(true);
        } else if meta_data.is_gen_ctf_3d_old_style_xtilting(self.axis_id) {
            self.rb_old_style_xtilting.set_selected(true);
        } else {
            self.rb_x_axis_tilted_slices.set_selected(true);
        }
    }

    /// Java package-private `getParameters(MetaData)`.
    pub fn get_parameters_meta_data(&self, meta_data: &MetaData) {
        meta_data.set_gen_ctf_3d_fourier_reduce_by_factor(
            self.axis_id,
            Some(self.sp_fourier_reduce_by_factor.get_value()),
        );
        meta_data
            .set_gen_ctf_3d_vertical_slices(self.axis_id, self.rb_vertical_slices.is_selected());
        meta_data.set_gen_ctf_3d_old_style_xtilting(
            self.axis_id,
            self.rb_old_style_xtilting.is_selected(),
        );
    }

    /// Java package-private `setParameters(Ctf3dSetupParam)`.
    pub fn set_parameters_ctf3d_setup_param(&self, param: &Ctf3dSetupParam) {
        if !param.is_slab_thickness_in_nm_null() {
            self.tf_slab_thickness_in_nm
                .set_text_string(Some(&param.get_slab_thickness_in_nm()));
        }
        self.cb_run_slabs_in_parallel
            .set_selected(param.is_run_slabs_in_parallel());
        self.cb_erase_fiducials
            .set_selected(param.is_erase_fiducials());
        self.cb_filter_in_2d.set_selected(param.is_filter_in_2d());
        self.cb_use_unaligned_images
            .set_selected(param.is_use_unaligned_images());
        self.cb_adjust_for_align_z_shift
            .set_selected(param.is_adjust_for_align_z_shift());
        self.sp_fourier_reduce_by_factor
            .set_value_string(Some(&param.get_fourier_reduce_by_factor()));
        if param.is_vertical_slices() {
            self.rb_vertical_slices.set_selected(true);
        } else if param.is_old_style_xtilting() {
            self.rb_old_style_xtilting.set_selected(true);
        } else {
            self.rb_x_axis_tilted_slices.set_selected(true);
        }
        self.bctf_temporary_directory
            .set_text_string(Some(&param.get_temporary_directory()));
        self.focus_lost();
        self.update_display();
    }

    /// Java package-private `getParameters(ReconScreenState)`.
    pub fn get_parameters_recon_screen_state(&self, screen_state: &ReconScreenState) {
        self.btn_ctf3d_setup.set_button_state(
            screen_state.get_button_state(self.btn_ctf3d_setup.get_button_state_key().as_deref()),
        );
    }

    /// Java package-private `setParameters(ReconScreenState)`.
    pub fn set_parameters_recon_screen_state(&self, screen_state: &ReconScreenState) {
        self.btn_ctf3d_setup.set_button_state(
            screen_state.get_button_state(self.btn_ctf3d_setup.get_button_state_key().as_deref()),
        );
    }

    /// Java private `setTooltips()`.
    fn set_tooltips(&self) {
        // Java `ReadOnlyAutodoc autodoc = null;` then the try/catch.
        let mut autodoc: *const dyn ReadOnlyAutodoc = std::ptr::null::<Autodoc>();
        // SAFETY: the factory keeps every autodoc it returns for the life of the
        // process.
        match unsafe {
            autodoc_factory::get_instance(
                Some(self.manager),
                Some(autodoc_factory::CTF_3D_SETUP),
                self.axis_id,
                false,
            )
        } {
            Ok(instance) => autodoc = instance as *const Autodoc,
            // catch (final LockException except) {}
            Err(LogFileError::Lock(_)) => {}
            // catch (final LogFileException | IOException except):
            // except.printStackTrace().
            Err(except) => eprintln!("{except}"),
        }
        // SAFETY: `autodoc` is null or an autodoc the factory keeps alive.
        let autodoc: Option<&dyn ReadOnlyAutodoc> = if autodoc.is_null() {
            None
        } else {
            Some(unsafe { &*autodoc })
        };
        self.tf_slab_thickness_in_nm.set_tooltip(
            etomo_autodoc::get_tooltip(autodoc, Some(ctf3d_setup_param::SLAB_THICKNESS_IN_NM_KEY))
                .as_deref(),
        );
        self.cb_run_slabs_in_parallel.set_tooltip(
            etomo_autodoc::get_tooltip(autodoc, Some(ctf3d_setup_param::RUN_SLABS_IN_PARALLEL_KEY))
                .as_deref(),
        );
        self.cb_erase_fiducials.set_tooltip(
            etomo_autodoc::get_tooltip(autodoc, Some(ctf3d_setup_param::ERASE_FIDUCIALS_KEY))
                .as_deref(),
        );
        self.cb_filter_in_2d.set_tooltip(
            etomo_autodoc::get_tooltip(autodoc, Some(ctf3d_setup_param::FILTER_IN_2D_KEY))
                .as_deref(),
        );
        self.cb_use_unaligned_images.set_tooltip(
            etomo_autodoc::get_tooltip(autodoc, Some(ctf3d_setup_param::USE_UNALIGNED_IMAGES_KEY))
                .as_deref(),
        );
        self.sp_fourier_reduce_by_factor.set_tool_tip_text(
            etomo_autodoc::get_tooltip(
                autodoc,
                Some(ctf3d_setup_param::FOURIER_REDUCE_BY_FACTOR_KEY),
            )
            .as_deref(),
        );
        self.rb_vertical_slices.set_tooltip_string(
            etomo_autodoc::get_tooltip(autodoc, Some(ctf3d_setup_param::VERTICAL_SLICES_KEY))
                .as_deref(),
        );
        self.rb_old_style_xtilting.set_tooltip_string(
            etomo_autodoc::get_tooltip(autodoc, Some(ctf3d_setup_param::OLD_STYLE_X_TILTING_KEY))
                .as_deref(),
        );
        self.bctf_temporary_directory.set_tooltip_string(
            etomo_autodoc::get_tooltip(autodoc, Some(ctf3d_setup_param::TEMPORARY_DIRECTORY_KEY))
                .as_deref(),
        );

        self.cb_adjust_for_align_z_shift.set_tooltip(Some(
            "Adjust for Z shift of tomogram away from cross-correlation alignment; use this if \
             the average material determining the CTF is not centered in the tomogram but \
             would probably have been centered with the cross-correlation alignment.",
        ));
    }
}

impl Ctf3dSetupDisplay for Ctf3dPanel {
    /// Java override `getParameters(Ctf3dSetupParam, boolean)`.
    fn get_parameters(&self, param: &mut Ctf3dSetupParam, do_validation: bool) -> bool {
        // try { ... } catch (final FieldValidationFailedException e) { return false; }
        let Ok(slab_thickness_in_nm) = self.tf_slab_thickness_in_nm.get_text_boolean(do_validation)
        else {
            return false;
        };
        param.set_slab_thickness_in_nm(slab_thickness_in_nm.as_deref());
        let Ok(temporary_directory) = self
            .bctf_temporary_directory
            .get_text_boolean(do_validation)
        else {
            return false;
        };
        param.set_temporary_directory(temporary_directory.as_deref());
        param.set_run_slabs_in_parallel(self.cb_run_slabs_in_parallel.is_selected());
        param.set_erase_fiducials(self.cb_erase_fiducials.is_selected());
        param.set_filter_in_2d(self.cb_filter_in_2d.is_selected());
        param.set_use_unaligned_images(self.cb_use_unaligned_images.is_selected());
        param.set_adjust_for_align_z_shift(self.cb_adjust_for_align_z_shift.is_selected());
        param.set_fourier_reduce_by_factor(Some(self.sp_fourier_reduce_by_factor.get_value()));
        param.set_vertical_slices(self.rb_vertical_slices.is_selected());
        param.set_old_style_xtilting(self.rb_old_style_xtilting.is_selected());
        if do_validation {
            self.focus_lost();
            let number_of_slabs = self.number_of_slabs.get();
            if number_of_slabs.is_none_or(|number_of_slabs| {
                number_of_slabs < ctf3d_setup_param::NUMBER_OF_SLABS_MIN as i64
            }) {
                let manager: &'static dyn BaseManager = self.manager;
                let ui_component: &dyn UIComponent = &*self.tf_slab_thickness_in_nm;
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_ui_component_string_string(
                        Some(manager),
                        Some(ui_component),
                        &format!(
                            "Requires at least {} slabs:  decrease thickness.",
                            ctf3d_setup_param::NUMBER_OF_SLABS_MIN
                        ),
                        "Not Enough Slabs",
                    )
                });
                return false;
            }
        }
        true
    }

    /// Java override `isRunSlabsInParallel()`.
    fn is_run_slabs_in_parallel(&self) -> bool {
        self.cb_run_slabs_in_parallel.is_selected()
    }

    /// Java override `getCtfCorrectionUIComponent()`: `this`.
    fn get_ctf_correction_ui_component(&self) -> Option<&dyn UIComponent> {
        Some(self)
    }

    /// Java override `getEraseFiducialsUIComponent()`.
    fn get_erase_fiducials_ui_component(&self) -> Option<&dyn UIComponent> {
        Some(&*self.cb_erase_fiducials)
    }

    /// Java override `isEraseFiducials()`.
    fn is_erase_fiducials(&self) -> bool {
        self.cb_erase_fiducials.is_selected()
    }

    /// Java override `getFilterIn2DUIComponent()`.
    fn get_filter_in_2d_ui_component(&self) -> Option<&dyn UIComponent> {
        Some(&*self.cb_filter_in_2d)
    }

    /// Java override `isFilterIn2D()`.
    fn is_filter_in_2d(&self) -> bool {
        self.cb_filter_in_2d.is_selected()
    }

    /// Java override `isUseUnalignedImages()`.
    fn is_use_unaligned_images(&self) -> bool {
        self.cb_use_unaligned_images.is_selected()
    }

    /// Java override `getUseUnalignedImagesUIComponent()`.
    fn get_use_unaligned_images_ui_component(&self) -> Option<&dyn UIComponent> {
        Some(&*self.cb_use_unaligned_images)
    }
}

impl Expandable for Ctf3dPanel {
    /// Java final `expand(ExpandButton)`.
    fn expand_expand_button(&self, button: &Rc<ExpandButton>) {
        if self.ph_ctf3d_setup.equals_advanced_basic(button) {
            self.update_display();
        }
    }

    /// Java final `expand(GlobalExpandButton)`.
    fn expand_global_expand_button(&self, _button: &Rc<GlobalExpandButton>) {
        self.update_display();
        let manager: &'static dyn BaseManager = self.manager;
        ui_harness::with(|harness| {
            harness.pack_axis_id_base_manager(Some(self.axis_id), Some(manager))
        });
    }
}

impl Run3dmodButtonContainer for Ctf3dPanel {
    /// Java override `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    fn action(
        &self,
        action_command: &str,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        self.action_option(
            Some(action_command),
            deferred_3dmod_button,
            run_3dmod_menu_options,
        );
    }
}

impl UIComponent for Ctf3dPanel {
    /// Java override `getUIComponent()`: `this`.
    fn get_ui_component(&self) -> &dyn SwingComponent {
        self
    }

    /// Java override `getComponent()`.
    fn get_component(&self) -> Rc<JComponent> {
        Ctf3dPanel::get_component(self)
    }
}

impl SwingComponent for Ctf3dPanel {
    /// Java override `getComponent()`.
    fn get_component(&self) -> Rc<JComponent> {
        Ctf3dPanel::get_component(self)
    }
}
