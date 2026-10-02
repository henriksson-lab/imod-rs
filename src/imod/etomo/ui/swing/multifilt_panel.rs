//! `IMOD/Etomo/src/etomo/ui/swing/MultifiltPanel.java`.
//!
//! Java `final class MultifiltPanel implements ActionListener,
//! MultifiltSetupDisplay, Run3dmodButtonContainer, Expandable, FilterType`:
//! the Filter Trials part of the Tomogram Generation dialog.  An EDT object
//! created as `Rc<Self>` by [`MultifiltPanel::get_instance`]; every method
//! takes `&self`.  The panel's own `ActionListener` role (`this` registered on
//! its buttons) is the closure in field `action_listener`, which calls
//! [`MultifiltPanel::action_performed`].  The parent dialog is held weakly (it
//! owns this panel).
//!
//! The private static inner class `FilterType` (an `EnumeratedType`) is the
//! private [`FilterType`] enum of this module; the package interface of the
//! same simple name that the panel implements is
//! `super::filter_type::FilterType`, referred to by path.

use std::cell::RefCell;
use std::rc::{Rc, Weak};
use std::sync::Arc;

use super::abstract_radio_button_model::AbstractRadioButtonModel;
use super::deferred_3dmod_button::Deferred3dmodButton;
use super::etched_border::EtchedBorder;
use super::expand_button::ExpandButton;
use super::expandable::Expandable;
use super::file_chooser::{self, FileChooser};
use super::global_expand_button::GlobalExpandButton;
use super::labeled_text_field::LabeledTextField;
use super::multifilt_setup_display::MultifiltSetupDisplay;
use super::panel_header::PanelHeader;
use super::radio_button::{RadioButton, RadioButtonModel};
use super::radio_button_interface::EnumeratedTypeRef;
use super::run_3dmod_button::Run3dmodButton;
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use super::text_field::TextField;
use super::tomogram_generation_dialog::TomogramGenerationDialog;
use super::tomogram_generation_parent::TomogramGenerationParent;
use super::ui_harness;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::multifilt_setup_param::{self, MultifiltSetupParam};
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, ButtonGroup, FileFilter, JComponent};
use crate::imod::etomo::process::imod_manager;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::storage::autodoc::autodoc::Autodoc;
use crate::imod::etomo::storage::autodoc::autodoc_factory;
use crate::imod::etomo::storage::autodoc::read_only_autodoc::ReadOnlyAutodoc;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::storage::multifilt_output_file_filter::MultifiltOutputFileFilter;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::ConstEtomoNumber;
use crate::imod::etomo::r#type::const_meta_data::ConstMetaData;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::enumerated_type::EnumeratedType;
use crate::imod::etomo::r#type::etomo_autodoc;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::image_filename_style::ImageFilenameStyle;
use crate::imod::etomo::r#type::meta_data::MetaData;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::r#type::recon_screen_state::ReconScreenState;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::shared_strings;

/// Java `final class MultifiltPanel`.
pub struct MultifiltPanel {
    /// Java private final `pnlRoot = new JPanel()`.
    pnl_root: Rc<JComponent>,
    /// Java private final `pnlParametersBody = new JPanel()`.
    pnl_parameters_body: Rc<JComponent>,
    /// Java private final `bgFilter`.
    bg_filter: Rc<ButtonGroup>,
    /// Java private final `rbFakeSIRTiterations`.
    rb_fake_sirt_iterations: Rc<RadioButton>,
    /// Java private final `rbExactObjectSizes`.
    rb_exact_object_sizes: Rc<RadioButton>,
    /// Java private final `rbGaussian`.
    rb_gaussian: Rc<RadioButton>,
    /// Java private final `rbHammingLikeStarts`.
    rb_hamming_like_starts: Rc<RadioButton>,
    /// Java private final `tfFakeSIRTiterations`.
    tf_fake_sirt_iterations: Rc<TextField>,
    /// Java private final `tfExactObjectSizes`.
    tf_exact_object_sizes: Rc<TextField>,
    /// Java private final `tfGaussianCutoffs`.
    tf_gaussian_cutoffs: Rc<TextField>,
    /// Java private final `ltfGaussianFalloffs`.
    ltf_gaussian_falloffs: Rc<LabeledTextField>,
    /// Java private final `tfHammingLikeStarts`.
    tf_hamming_like_starts: Rc<TextField>,
    /// Java private final `btn3dmodMultifiltSetup`.
    btn_3dmod_multifilt_setup: Rc<Run3dmodButton>,
    //
    /// Java private final `ltfWidthInX`.
    ltf_width_in_x: Rc<LabeledTextField>,
    /// Java private final `ltfShiftInX`.
    ltf_shift_in_x: Rc<LabeledTextField>,
    /// Java private final `ltfSizeInY`.
    ltf_size_in_y: Rc<LabeledTextField>,
    /// Java private final `ltfShiftInY`.
    ltf_shift_in_y: Rc<LabeledTextField>,
    /// Java private final `ltfThicknessInZ`.
    ltf_thickness_in_z: Rc<LabeledTextField>,
    /// Java private final `ltfShiftInDepth`.
    ltf_shift_in_depth: Rc<LabeledTextField>,

    /// Java private final `axisID`.
    axis_id: AxisID,
    /// Java private final `manager`.
    manager: &'static ApplicationManager,
    /// Java private final `parent` (held weakly; the dialog owns this panel).
    parent: Weak<TomogramGenerationDialog>,
    /// Java private final `dialogType`.
    dialog_type: DialogType,
    /// Java private final `btnMultifiltSetup`.
    btn_multifilt_setup: Rc<Run3dmodButton>,
    /// Java private final `header`.
    header: Rc<PanelHeader>,
    /// Java private final `fileFilterArray = new
    /// MultifiltOutputFileFilter[FilterType.ARRAY_SIZE]`.
    file_filter_array: RefCell<Vec<Option<Rc<MultifiltOutputFileFilter>>>>,
    /// Java private final `imageFilenameStyle`.  `None` is Java null (no base
    /// metadata).
    image_filename_style: Option<ImageFilenameStyle>,

    /// Java private `listenerList = null`.
    listener_list: RefCell<Option<Vec<ActionListener>>>,
    /// Java `this` in its `ActionListener` role (`actionPerformed`).
    action_listener: ActionListener,
    /// Java `this`, for the containers handed out after construction.
    this: Weak<MultifiltPanel>,
}

impl MultifiltPanel {
    /// Java private constructor `MultifiltPanel(ApplicationManager, AxisID,
    /// TomogramGenerationDialog, DialogType)`.
    fn new(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        parent: Weak<TomogramGenerationDialog>,
        dialog_type: DialogType,
    ) -> Rc<MultifiltPanel> {
        Rc::new_cyclic(|this: &Weak<MultifiltPanel>| {
            // Field initializers, in declaration order.
            let pnl_root = JComponent::new_panel();
            let pnl_parameters_body = JComponent::new_panel();
            let bg_filter = ButtonGroup::new();
            let rb_fake_sirt_iterations = RadioButton::new_enumerated_type_button_group(
                EnumeratedTypeRef::new(FilterType::FakeSirtIterations),
                Some(&bg_filter),
            );
            let rb_exact_object_sizes = RadioButton::new_enumerated_type_button_group(
                EnumeratedTypeRef::new(FilterType::ExactObjectSizes),
                Some(&bg_filter),
            );
            let rb_gaussian = RadioButton::new_enumerated_type_button_group(
                EnumeratedTypeRef::new(FilterType::Gaussian),
                Some(&bg_filter),
            );
            let rb_hamming_like_starts = RadioButton::new_enumerated_type_button_group(
                EnumeratedTypeRef::new(FilterType::HammingLikeStarts),
                Some(&bg_filter),
            );
            let tf_fake_sirt_iterations = TextField::new(
                FieldType::IntegerList,
                Some(FilterType::FakeSirtIterations.label()),
                None,
            );
            let tf_exact_object_sizes = TextField::new(
                FieldType::IntegerList,
                Some(FilterType::ExactObjectSizes.label()),
                None,
            );
            let tf_gaussian_cutoffs = TextField::new(
                FieldType::FloatingPointArray,
                Some(FilterType::Gaussian.label()),
                None,
            );
            let ltf_gaussian_falloffs = LabeledTextField::new_field_type_string(
                FieldType::FloatingPointArray,
                Some(" falloffs:"),
            );
            let tf_hamming_like_starts = TextField::new(
                FieldType::FloatingPointArray,
                Some(FilterType::HammingLikeStarts.label()),
                None,
            );
            let container: Weak<dyn Run3dmodButtonContainer> = this.clone();
            let btn_3dmod_multifilt_setup =
                Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                    Some("View Tomogram(s) In 3dmod"),
                    Some(container),
                );
            //
            let ltf_width_in_x = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some("Tomogram width in X: "),
            );
            let ltf_shift_in_x = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("X shift: "),
            );
            let ltf_size_in_y = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some("Tomogram height in Y: "),
            );
            let ltf_shift_in_y =
                LabeledTextField::new_field_type_string(FieldType::Integer, Some(" Y shift: "));
            let ltf_thickness_in_z = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some("Tomogram thickness in Z: "),
            );
            let ltf_shift_in_depth = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some(" Z shift: "),
            );
            let file_filter_array = RefCell::new(vec![None; FilterType::ARRAY_SIZE]);
            // Java `this` as an ActionListener: `actionPerformed(ActionEvent)`.
            let adaptee = this.clone();
            let action_listener: ActionListener = Rc::new(move |event: &ActionEvent| {
                if let Some(adaptee) = adaptee.upgrade() {
                    adaptee.action_performed(Some(event));
                }
            });

            // Constructor body.
            let base_manager: &'static dyn BaseManager = manager;
            // Java dereferences getBaseMetaData() unguarded; a missing one leaves
            // the style null.
            let image_filename_style = base_manager
                .get_base_meta_data()
                .map(|meta_data| meta_data.base().get_image_filename_style());
            // Java casts `(Run3dmodButton) ...getMultifiltSetup()`; the factory
            // returns the concrete button.
            let btn_multifilt_setup = manager
                .get_process_result_display_factory(axis_id)
                .get_multifilt_setup();
            let expandable: Weak<dyn Expandable> = this.clone();
            let header = PanelHeader::get_instance(
                Some("Filter Trials"),
                Some(expandable),
                Some(dialog_type),
            );
            MultifiltPanel {
                pnl_root,
                pnl_parameters_body,
                bg_filter,
                rb_fake_sirt_iterations,
                rb_exact_object_sizes,
                rb_gaussian,
                rb_hamming_like_starts,
                tf_fake_sirt_iterations,
                tf_exact_object_sizes,
                tf_gaussian_cutoffs,
                ltf_gaussian_falloffs,
                tf_hamming_like_starts,
                btn_3dmod_multifilt_setup,
                ltf_width_in_x,
                ltf_shift_in_x,
                ltf_size_in_y,
                ltf_shift_in_y,
                ltf_thickness_in_z,
                ltf_shift_in_depth,
                axis_id,
                manager,
                parent,
                dialog_type,
                btn_multifilt_setup,
                header,
                file_filter_array,
                image_filename_style,
                listener_list: RefCell::new(None),
                action_listener,
                this: this.clone(),
            }
        })
    }

    /// Java static `getInstance(ApplicationManager, AxisID,
    /// TomogramGenerationDialog, DialogType)`.
    pub fn get_instance(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        parent: Weak<TomogramGenerationDialog>,
        dialog_type: DialogType,
    ) -> Rc<MultifiltPanel> {
        let instance = MultifiltPanel::new(manager, axis_id, parent, dialog_type);
        instance.create_panel();
        instance.set_tooltips();
        instance.add_listeners();
        instance
    }

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        // init
        self.rb_fake_sirt_iterations.set_selected_boolean(true);
        self.tf_fake_sirt_iterations.set_required(true);
        self.tf_exact_object_sizes.set_required(true);
        self.tf_hamming_like_starts.set_required(true);
        self.ltf_width_in_x.set_preferred_width(163);
        self.ltf_size_in_y.set_preferred_width(159);
        let container: Weak<dyn Run3dmodButtonContainer> = self.this.clone();
        self.btn_multifilt_setup.set_container(Some(container));
        let deferred: Rc<dyn Deferred3dmodButton> = self.btn_3dmod_multifilt_setup.clone();
        self.btn_multifilt_setup
            .set_deferred_3dmod_button_deferred_3dmod_button(Some(deferred));
        self.tf_gaussian_cutoffs.set_preferred_width(108);
        self.ltf_gaussian_falloffs.set_preferred_width(109);
        self.create_file_filter(Some(FilterType::FakeSirtIterations));
        self.create_file_filter(Some(FilterType::ExactObjectSizes));
        self.create_file_filter(Some(FilterType::Gaussian));
        self.create_file_filter(Some(FilterType::HammingLikeStarts));
        self.create_file_filter(None);
        // panels
        let pnl_parameters = JComponent::new_panel();
        let pnl_filter_to_try = JComponent::new_panel();
        let pnl_fake_sirt_iterations = JComponent::new_panel();
        let pnl_exact_object_sizes = JComponent::new_panel();
        let pnl_gaussian = JComponent::new_panel();
        let pnl_hamming_like_starts = JComponent::new_panel();
        let pnl_subarea = JComponent::new_panel();
        let pnl_x = JComponent::new_panel();
        let pnl_y = JComponent::new_panel();
        let pnl_z = JComponent::new_panel();
        let pnl_buttons = JComponent::new_panel();
        //
        // Root
        // Swing layout: pnlRoot BoxLayout Y_AXIS.
        self.pnl_root.add(&pnl_parameters);
        self.pnl_root.add(&pnl_buttons);
        //
        // Parameters
        // Swing layout: pnlParameters BoxLayout Y_AXIS, etched border.
        pnl_parameters.add(&self.header.get_component());
        pnl_parameters.add(&self.pnl_parameters_body);
        // ParametersBody
        // Swing layout: pnlParametersBody BoxLayout Y_AXIS.
        self.pnl_parameters_body.add(&pnl_filter_to_try);
        self.pnl_parameters_body.add(&pnl_subarea);
        //
        // FilterToTry
        // Swing layout: pnlFilterToTry BoxLayout Y_AXIS.
        pnl_filter_to_try.set_border_title(
            EtchedBorder::new(Some("Filter To Try"))
                .get_border()
                .get_title()
                .as_deref(),
        );
        pnl_filter_to_try.add(&pnl_fake_sirt_iterations);
        pnl_filter_to_try.add(&pnl_exact_object_sizes);
        pnl_filter_to_try.add(&pnl_gaussian);
        pnl_filter_to_try.add(&pnl_hamming_like_starts);
        // FakeSIRTiterations
        // Swing layout: BoxLayout X_AXIS.
        pnl_fake_sirt_iterations.add(&self.rb_fake_sirt_iterations.get_component());
        pnl_fake_sirt_iterations.add(&self.tf_fake_sirt_iterations.get_component());
        // ExactObjectSizes
        // Swing layout: BoxLayout X_AXIS.
        pnl_exact_object_sizes.add(&self.rb_exact_object_sizes.get_component());
        pnl_exact_object_sizes.add(&self.tf_exact_object_sizes.get_component());
        // Gaussian
        // Swing layout: BoxLayout X_AXIS.
        pnl_gaussian.add(&self.rb_gaussian.get_component());
        pnl_gaussian.add(&self.tf_gaussian_cutoffs.get_component());
        pnl_gaussian.add(&self.ltf_gaussian_falloffs.get_component());
        // HammingLikeStarts
        // Swing layout: BoxLayout X_AXIS.
        pnl_hamming_like_starts.add(&self.rb_hamming_like_starts.get_component());
        pnl_hamming_like_starts.add(&self.tf_hamming_like_starts.get_component());
        //
        // Subarea
        // Swing layout: pnlSubarea BoxLayout Y_AXIS; rigid areas x0_y5 / x0_y3
        // between the rows.
        pnl_subarea.set_border_title(
            EtchedBorder::new(Some("Subarea"))
                .get_border()
                .get_title()
                .as_deref(),
        );
        pnl_subarea.add(&pnl_x);
        pnl_subarea.add(&pnl_y);
        pnl_subarea.add(&pnl_z);
        // X panel
        // Swing layout: BoxLayout X_AXIS; rigid area x5_y0 between the fields.
        pnl_x.add(&self.ltf_width_in_x.get_container());
        pnl_x.add(&self.ltf_shift_in_x.get_container());
        // Y panel
        // Swing layout: BoxLayout X_AXIS; rigid area x5_y0 between the fields.
        pnl_y.add(&self.ltf_size_in_y.get_container());
        pnl_y.add(&self.ltf_shift_in_y.get_container());
        // Z panel
        // Swing layout: BoxLayout X_AXIS; rigid area x5_y0 between the fields.
        pnl_z.add(&self.ltf_thickness_in_z.get_container());
        pnl_z.add(&self.ltf_shift_in_depth.get_container());
        //
        // Buttons
        // Swing layout: pnlButtons BoxLayout X_AXIS.
        pnl_buttons.add(&self.btn_multifilt_setup.get_component());
        pnl_buttons.add(&self.btn_3dmod_multifilt_setup.get_component());
        self.update_display();
    }

    /// Java private `createFileFilter(FilterType)`.
    fn create_file_filter(&self, filter_type: Option<FilterType>) {
        let base_manager: &'static dyn BaseManager = self.manager;
        if let Some(filter_type) = filter_type {
            self.file_filter_array.borrow_mut()[filter_type.index()] =
                Some(MultifiltOutputFileFilter::get_instance(
                    base_manager,
                    self.image_filename_style,
                    self.axis_id,
                    Some(filter_type.file_type()),
                ));
        } else {
            // This is the all-filters element
            self.file_filter_array.borrow_mut()[FilterType::ARRAY_SIZE - 1] =
                Some(MultifiltOutputFileFilter::get_instance(
                    base_manager,
                    self.image_filename_style,
                    self.axis_id,
                    None,
                ));
        }
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self) {
        self.rb_fake_sirt_iterations
            .add_action_listener(self.action_listener.clone());
        self.rb_exact_object_sizes
            .add_action_listener(self.action_listener.clone());
        self.rb_gaussian
            .add_action_listener(self.action_listener.clone());
        self.rb_hamming_like_starts
            .add_action_listener(self.action_listener.clone());
        self.btn_multifilt_setup
            .add_action_listener(self.action_listener.clone());
        self.btn_3dmod_multifilt_setup
            .add_action_listener(self.action_listener.clone());
    }

    /// Java public `addActionListener(ActionListener)`.  (Java returns on a
    /// null listener; a Rust listener is never null.)
    pub fn add_action_listener(&self, listener: ActionListener) {
        // In order to signal the listener when fields values have been set.
        {
            let mut listener_list = self.listener_list.borrow_mut();
            if listener_list.is_none() {
                *listener_list = Some(Vec::new());
            }
            listener_list.as_mut().unwrap().push(listener.clone());
        }
        self.rb_fake_sirt_iterations
            .add_action_listener(listener.clone());
        self.rb_exact_object_sizes
            .add_action_listener(listener.clone());
        self.rb_gaussian.add_action_listener(listener.clone());
        self.rb_hamming_like_starts.add_action_listener(listener);
    }

    /// Java package-private `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.clone()
    }

    /// Java public `actionPerformed(ActionEvent)` (implements
    /// `ActionListener`).
    pub fn action_performed(&self, event: Option<&ActionEvent>) {
        // Upstream bug fixed in translation (MultifiltPanel.java:271-282): Java
        // passes a null command on to `action`, whose first
        // `actionCommand.equals` then throws a NullPointerException.  A missing
        // command here matches no button, so it takes `action`'s final
        // `updateDisplay()` branch.
        let action_command = event.and_then(|event| event.get_action_command());
        Run3dmodButtonContainer::action(self, action_command.unwrap_or(""), None, None);
    }

    /// Java private `openFilesInImod(Run3dmodMenuOptions)`.
    fn open_files_in_imod(&self, run_3dmod_menu_options: Option<Run3dmodMenuOptions>) {
        let base_manager: &'static dyn BaseManager = self.manager;
        // Don't open the file chooser if there is only one file to choose
        // (Java builds `fileFilter` here and never reads it.)
        let _file_filter = MultifiltOutputFileFilter::get_instance(
            base_manager,
            self.image_filename_style,
            self.axis_id,
            None,
        );
        let chooser = FileChooser::new_base_manager(Some(base_manager));
        // Swing layout: chooser.setPreferredSize(UIParameters.getInstance()
        // .getFileChooserDimension()).
        chooser.set_file_selection_mode(file_chooser::FILES_ONLY);
        chooser.set_multi_selection_enabled(true);
        let file_filter_array = self.file_filter_array.borrow().clone();
        for filter in file_filter_array.iter().flatten() {
            let filter: Rc<dyn FileFilter> = filter.clone();
            chooser.add_choosable_file_filter(filter);
        }
        // Java `(FilterType) ((RadioButton.RadioButtonModel) bgFilter.getSelection())
        // .getEnumeratedType()`.
        let filter_type = self
            .bg_filter
            .get_selection()
            .and_then(|button| button.get_model())
            .and_then(|model| {
                model
                    .as_any()
                    .downcast_ref::<RadioButtonModel>()
                    .and_then(|model| AbstractRadioButtonModel::get_enumerated_type(model))
            })
            .and_then(|enumerated_type| enumerated_type.downcast_ref::<FilterType>().copied());
        // Upstream bug fixed in translation (MultifiltPanel.java:333-336): with no
        // radio button selected Java throws a NullPointerException here; the
        // chooser is left on its default filter instead.
        if let Some(filter_type) = filter_type {
            let filter: Option<Rc<dyn FileFilter>> = file_filter_array[filter_type.index()]
                .clone()
                .map(|filter| filter as Rc<dyn FileFilter>);
            chooser.set_file_filter(filter);
        }
        let return_val = chooser.show_open_dialog(Some(&self.pnl_root));
        if return_val != file_chooser::APPROVE_OPTION {
            return;
        }
        let file_list = chooser.get_selected_files();
        if file_list.is_empty() {
            return;
        }
        // Java passes the (possibly null) options through; a null is read as
        // the default options.
        self.manager
            .open_files_in_imod_axis_id_string_file_array_run3dmod_menu_options(
                self.axis_id,
                imod_manager::MULTIFILT_KEY,
                &file_list,
                run_3dmod_menu_options.unwrap_or_default(),
            );
    }

    /// Java private `updateDisplay()`.
    fn update_display(&self) {
        self.tf_fake_sirt_iterations
            .set_enabled(self.rb_fake_sirt_iterations.is_selected());
        self.tf_exact_object_sizes
            .set_enabled(self.rb_exact_object_sizes.is_selected());
        let enabled = self.rb_gaussian.is_selected();
        self.tf_gaussian_cutoffs.set_enabled(enabled);
        self.ltf_gaussian_falloffs.set_enabled(enabled);
        self.tf_hamming_like_starts
            .set_enabled(self.rb_hamming_like_starts.is_selected());
    }

    /// Java `msgMethodChanged()`.  A parent that is gone answers false (Java's
    /// parent outlives the panel).
    pub fn msg_method_changed(&self) {
        self.pnl_root.set_visible(
            self.parent
                .upgrade()
                .is_some_and(|parent| parent.is_multifilt()),
        );
    }

    /// Java `done()`.
    pub fn done(&self) {
        self.rb_fake_sirt_iterations
            .remove_action_listener(&self.action_listener);
        self.rb_exact_object_sizes
            .remove_action_listener(&self.action_listener);
        self.rb_gaussian
            .remove_action_listener(&self.action_listener);
        self.rb_hamming_like_starts
            .remove_action_listener(&self.action_listener);
        self.btn_multifilt_setup
            .remove_action_listener(&self.action_listener);
    }

    /// Java `getParameters(ReconScreenState)`.
    pub fn get_parameters_recon_screen_state(&self, screen_state: &ReconScreenState) {
        self.btn_multifilt_setup.set_button_state(
            screen_state
                .get_button_state(self.btn_multifilt_setup.get_button_state_key().as_deref()),
        );
    }

    /// Java `setParameters(ReconScreenState)`.
    pub fn set_parameters_recon_screen_state(&self, screen_state: &ReconScreenState) {
        self.btn_multifilt_setup.set_button_state(
            screen_state
                .get_button_state(self.btn_multifilt_setup.get_button_state_key().as_deref()),
        );
    }

    /// Java `getParameters(MetaData)`.
    pub fn get_parameters_meta_data(&self, meta_data: &MetaData) {
        meta_data.set_gen_filter_trials_fake_sirt_iterations(
            self.axis_id,
            self.tf_fake_sirt_iterations.get_text_void().as_deref(),
        );
        meta_data.set_gen_filter_trials_exact_object_sizes(
            self.axis_id,
            self.tf_exact_object_sizes.get_text_void().as_deref(),
        );
        meta_data.set_gen_filter_trials_gaussian_cutoffs(
            self.axis_id,
            self.tf_gaussian_cutoffs.get_text_void().as_deref(),
        );
        meta_data.set_gen_filter_trials_gaussian_falloffs(
            self.axis_id,
            self.ltf_gaussian_falloffs.get_text_void().as_deref(),
        );
        meta_data.set_gen_filter_trials_hamming_like_starts(
            self.axis_id,
            self.tf_hamming_like_starts.get_text_void().as_deref(),
        );
    }

    /// Java `setParameters(ConstMetaData)`.
    pub fn set_parameters_const_meta_data(&self, meta_data: &dyn ConstMetaData) {
        self.tf_fake_sirt_iterations.set_text_string(Some(
            &meta_data.get_gen_filter_trials_fake_sirt_iterations(self.axis_id),
        ));
        self.tf_exact_object_sizes.set_text_string(Some(
            &meta_data.get_gen_filter_trials_exact_object_sizes(self.axis_id),
        ));
        self.tf_gaussian_cutoffs.set_text_string(Some(
            &meta_data.get_gen_filter_trials_gaussian_cutoffs(self.axis_id),
        ));
        self.ltf_gaussian_falloffs.set_text_string(Some(
            &meta_data.get_gen_filter_trials_gaussian_falloffs(self.axis_id),
        ));
        self.tf_hamming_like_starts.set_text_string(Some(
            &meta_data.get_gen_filter_trials_hamming_like_starts(self.axis_id),
        ));
        self.signal_listeners();
    }

    /// Java private `signalListeners()`.
    fn signal_listeners(&self) {
        let listener_list = self.listener_list.borrow().clone();
        if let Some(listener_list) = listener_list {
            // Java calls `actionPerformed(null)`.  A Rust `ActionListener` takes
            // an event, so an event from this panel's root with no action
            // command stands in for the null (the only listener registered, the
            // radial panel's, does not read it).
            let event = ActionEvent {
                source: self.pnl_root.clone(),
                action_command: None,
            };
            for listener in listener_list.iter() {
                listener(&event);
            }
        }
    }

    /// Java `setParameters(MultifiltSetupParam)`.
    pub fn set_parameters_multifilt_setup_param(&self, param: &MultifiltSetupParam) {
        if param.is_fake_sirt_iterations() {
            self.rb_fake_sirt_iterations.set_selected_boolean(true);
            self.tf_fake_sirt_iterations
                .set_text_string(Some(&param.get_fake_sirt_iterations()));
        } else if param.is_exact_object_sizes() {
            self.rb_exact_object_sizes.set_selected_boolean(true);
            self.tf_exact_object_sizes
                .set_text_string(Some(&param.get_exact_object_sizes()));
        } else if param.is_gaussian_cutoffs() || param.is_gaussian_falloffs() {
            self.rb_gaussian.set_selected_boolean(true);
            self.tf_gaussian_cutoffs
                .set_text_string(Some(&param.get_gaussian_cutoffs()));
            self.ltf_gaussian_falloffs
                .set_text_string(Some(&param.get_gaussian_falloffs()));
        } else if param.is_hamming_like_starts() {
            self.rb_hamming_like_starts.set_selected_boolean(true);
            self.tf_hamming_like_starts
                .set_text_string(Some(&param.get_hamming_like_starts()));
        }
        self.ltf_width_in_x
            .set_text_string(Some(&param.get_width_in_x()));
        self.ltf_shift_in_x
            .set_text_string(Some(&param.get_shift_in_x()));
        self.ltf_size_in_y
            .set_text_string(Some(&param.get_size_in_y()));
        self.ltf_shift_in_y
            .set_text_string(Some(&param.get_shift_in_y()));
        self.ltf_thickness_in_z
            .set_text_string(Some(&param.get_thickness_in_z()));
        self.ltf_shift_in_depth
            .set_text_string(Some(&param.get_shift_in_depth()));
        self.update_display();
        self.signal_listeners();
    }

    /// Java `getParameters(MultifiltSetupParam, boolean)` (implements
    /// `MultifiltSetupDisplay`).
    pub fn get_parameters_multifilt_setup_param_boolean(
        &self,
        param: &mut MultifiltSetupParam,
        do_validation: bool,
    ) -> bool {
        // try { ... } catch (FieldValidationFailedException e) { return false; }
        let result = (|| -> Result<(), ()> {
            if self.rb_fake_sirt_iterations.is_selected() {
                let text = self
                    .tf_fake_sirt_iterations
                    .get_text_boolean(do_validation)
                    .map_err(|_| ())?;
                param.set_fake_sirt_iterations(text.as_deref());
            } else {
                param.reset_fake_sirt_iterations();
            }
            if self.rb_exact_object_sizes.is_selected() {
                let text = self
                    .tf_exact_object_sizes
                    .get_text_boolean(do_validation)
                    .map_err(|_| ())?;
                param.set_exact_object_sizes(text.as_deref());
            } else {
                param.reset_exact_object_sizes();
            }
            if self.rb_gaussian.is_selected() {
                let text = self
                    .tf_gaussian_cutoffs
                    .get_text_boolean(do_validation)
                    .map_err(|_| ())?;
                param.set_gaussian_cutoffs(text.as_deref());
                let text = self
                    .ltf_gaussian_falloffs
                    .get_text_boolean(do_validation)
                    .map_err(|_| ())?;
                param.set_gaussian_falloffs(text.as_deref());
                if do_validation {
                    let falloffs: &dyn Field = &*self.ltf_gaussian_falloffs;
                    self.tf_gaussian_cutoffs
                        .validate_paired_arrays(Some(falloffs))
                        .map_err(|_| ())?;
                }
            } else {
                param.reset_gaussian_cutoffs();
                param.reset_gaussian_falloffs();
            }
            if self.rb_hamming_like_starts.is_selected() {
                let text = self
                    .tf_hamming_like_starts
                    .get_text_boolean(do_validation)
                    .map_err(|_| ())?;
                param.set_hamming_like_starts(text.as_deref());
            } else {
                param.reset_hamming_like_starts();
            }
            let text = self
                .ltf_width_in_x
                .get_text_boolean(do_validation)
                .map_err(|_| ())?;
            param.set_width_in_x(text.as_deref());
            let text = self
                .ltf_shift_in_x
                .get_text_boolean(do_validation)
                .map_err(|_| ())?;
            param.set_shift_in_x(text.as_deref());
            let text = self
                .ltf_size_in_y
                .get_text_boolean(do_validation)
                .map_err(|_| ())?;
            param.set_size_in_y(text.as_deref());
            let text = self
                .ltf_shift_in_y
                .get_text_boolean(do_validation)
                .map_err(|_| ())?;
            param.set_shift_in_y(text.as_deref());
            let text = self
                .ltf_thickness_in_z
                .get_text_boolean(do_validation)
                .map_err(|_| ())?;
            param.set_thickness_in_z(text.as_deref());
            let text = self
                .ltf_shift_in_depth
                .get_text_boolean(do_validation)
                .map_err(|_| ())?;
            param.set_shift_in_depth(text.as_deref());
            Ok(())
        })();
        if result.is_err() {
            return false;
        }
        true
    }

    /// Java private `setTooltips()`.
    fn set_tooltips(&self) {
        // Java `ReadOnlyAutodoc autodoc = null;` then the try/catch.
        let mut autodoc: *const dyn ReadOnlyAutodoc = std::ptr::null::<Autodoc>();
        let base_manager: &'static dyn BaseManager = self.manager;
        // SAFETY: the factory returns an autodoc it keeps for the life of the
        // process.
        match unsafe {
            autodoc_factory::get_instance(
                Some(base_manager),
                Some(autodoc_factory::MULTIFILT_SETUP),
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
        // SAFETY: `autodoc` is null or an autodoc the factory keeps for the life
        // of the process.
        let autodoc: Option<&dyn ReadOnlyAutodoc> = if autodoc.is_null() {
            None
        } else {
            Some(unsafe { &*autodoc })
        };
        self.rb_fake_sirt_iterations
            .set_tool_tip_text_string(Some(shared_strings::SIRT_LIKE_FILTER_RADIO_BUTTON_TOOLTIP));
        self.tf_fake_sirt_iterations.set_tool_tip_text(
            etomo_autodoc::get_tooltip(autodoc, Some(multifilt_setup_param::FAKE_SIRT_ITERATIONS))
                .as_deref(),
        );
        self.rb_exact_object_sizes
            .set_tool_tip_text_string(Some(shared_strings::EXACT_FILTER_RADIO_BUTTON_TOOLTIP));
        self.tf_exact_object_sizes.set_tool_tip_text(
            etomo_autodoc::get_tooltip(autodoc, Some(multifilt_setup_param::EXACT_OBJECT_SIZES))
                .as_deref(),
        );
        self.rb_gaussian
            .set_tool_tip_text_string(Some(shared_strings::GAUSSIAN_FILTER_RADIO_BUTTON_TOOLTIP));
        self.tf_gaussian_cutoffs.set_tool_tip_text(
            etomo_autodoc::get_tooltip(autodoc, Some(multifilt_setup_param::GAUSSIAN_CUTOFFS))
                .as_deref(),
        );
        self.ltf_gaussian_falloffs.set_tool_tip_text(
            etomo_autodoc::get_tooltip(autodoc, Some(multifilt_setup_param::GAUSSIAN_FALLOFFS))
                .as_deref(),
        );
        self.rb_hamming_like_starts.set_tool_tip_text_string(Some(
            shared_strings::HAMMING_LIKE_FILTER_RADIO_BUTTON_TOOLTIP,
        ));
        self.tf_hamming_like_starts.set_tool_tip_text(
            etomo_autodoc::get_tooltip(autodoc, Some(multifilt_setup_param::HAMMING_LIKE_STARTS))
                .as_deref(),
        );
        //
        self.btn_multifilt_setup.set_tool_tip_text(Some(
            "Run multifiltsetup, and then run the resulting .com files with processchunks.",
        ));
        self.btn_3dmod_multifilt_setup.set_tool_tip_text(Some(
            "Opens a file chooser for picking filter trial output to open together in 3dmod",
        ));
        //
        self.ltf_width_in_x.set_tool_tip_text(Some(
            "This entry specifies the width, in unbinned pixels, of the output \
             image; the default is the width of the input image.",
        ));
        self.ltf_shift_in_x.set_tool_tip_text(Some(
            "Amount, in unbinned pixels, to shift the reconstructed slices in \
             X before output.  A positive value will shift the slice to the right, and \
             the output will contain the left part of the whole potentially \
             reconstructable area.  The default is no shift.",
        ));
        self.ltf_size_in_y.set_tool_tip_text(Some(
            "This entry specifies the Y extent of the output tomogram, in \
             unbinned pixels; the default is the height of the aligned stack.",
        ));
        self.ltf_shift_in_y.set_tool_tip_text(Some(
            "Amount to shift the reconstructed region in Y, in unbinned pixels.  \
             A positive value will shift the region upward and reconstruct an area lower \
             in Y.  The default is no shift.",
        ));
        self.ltf_thickness_in_z.set_tool_tip_text(Some(
            "Thickness, in unbinned pixels, along the z-axis of the \
             reconstructed volume.  The default is the value on the Back Projection \
             panel.",
        ));
        self.ltf_shift_in_depth.set_tool_tip_text(Some(
            "Amount, in unbinned pixels, to shift the reconstructed slices in \
             Z before output.  A positive value will shift the slice upward.  The default \
             is the value on the Back Projection panel.",
        ));
    }
}

impl MultifiltSetupDisplay for MultifiltPanel {
    /// Java `getParameters(MultifiltSetupParam, boolean)`.
    fn get_parameters(&self, param: &mut MultifiltSetupParam, do_validation: bool) -> bool {
        self.get_parameters_multifilt_setup_param_boolean(param, do_validation)
    }
}

impl Run3dmodButtonContainer for MultifiltPanel {
    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    fn action(
        &self,
        action_command: &str,
        _deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        if Some(action_command) == self.btn_multifilt_setup.get_action_command().as_deref() {
            let display: ProcessResultDisplayHandle = self.btn_multifilt_setup.clone();
            let processing_method = self
                .parent
                .upgrade()
                .map(|parent| parent.get_processing_method());
            self.manager.multifilt_setup(
                self.axis_id,
                Some(display),
                None,
                self.dialog_type,
                processing_method,
                self,
            );
        } else if Some(action_command)
            == self
                .btn_3dmod_multifilt_setup
                .get_action_command()
                .as_deref()
        {
            self.open_files_in_imod(run_3dmod_menu_options);
        } else {
            self.update_display();
        }
    }
}

impl super::filter_type::FilterType for MultifiltPanel {
    /// Java `isRadialFilter()`.
    fn is_radial_filter(&self) -> bool {
        self.rb_fake_sirt_iterations.is_selected() || self.rb_exact_object_sizes.is_selected()
    }

    /// Java `isHighFrequencyFilter()`.
    fn is_high_frequency_filter(&self) -> bool {
        self.rb_gaussian.is_selected() || self.rb_hamming_like_starts.is_selected()
    }
}

impl Expandable for MultifiltPanel {
    /// Java public final `expand(ExpandButton)`.
    fn expand_expand_button(&self, button: &Rc<ExpandButton>) {
        // Java `if (header != null)`: final and always set.
        if self.header.equals_open_close(button) {
            self.pnl_parameters_body.set_visible(button.is_expanded());
        }
        let base_manager: &'static dyn BaseManager = self.manager;
        ui_harness::with(|harness| {
            harness.pack_axis_id_base_manager(Some(self.axis_id), Some(base_manager))
        });
    }

    /// Java public final `expand(GlobalExpandButton)`; empty.
    fn expand_global_expand_button(&self, _button: &Rc<GlobalExpandButton>) {}
}

/// Java `private static final class FilterType implements EnumeratedType`.
/// The four static instances, indexed 0..3 in declaration order
/// (`indexValue++`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FilterType {
    /// Java `FAKE_SIRT_ITERATIONS` (the default instance).
    FakeSirtIterations,
    /// Java `EXACT_OBJECT_SIZES`.
    ExactObjectSizes,
    /// Java `GAUSSIAN`.
    Gaussian,
    /// Java `HAMMING_LIKE_STARTS`.
    HammingLikeStarts,
}

impl FilterType {
    /// Java private static final `ARRAY_SIZE = indexValue + 1`: make room in
    /// the array for the all-filter file filter.
    const ARRAY_SIZE: usize = 4 + 1;

    /// Java field `label`.
    fn label(self) -> &'static str {
        match self {
            FilterType::FakeSirtIterations => "SIRT-like filter with iterations:",
            FilterType::ExactObjectSizes => "'Exact filter' function with 'object sizes':",
            FilterType::Gaussian => "Standard Gaussian with cutoffs:",
            FilterType::HammingLikeStarts => "Hamming-like filter with start frequencies:",
        }
    }

    /// Java field `index`.
    fn index(self) -> usize {
        match self {
            FilterType::FakeSirtIterations => 0,
            FilterType::ExactObjectSizes => 1,
            FilterType::Gaussian => 2,
            FilterType::HammingLikeStarts => 3,
        }
    }

    /// Java field `fileType`.
    fn file_type(self) -> Arc<FileType> {
        match self {
            FilterType::FakeSirtIterations => file_type::CLASS
                .mutlifilt_fake_sirt_iterations_output_template
                .clone(),
            FilterType::ExactObjectSizes => file_type::CLASS
                .mutlifilt_exact_object_sizes_output_template
                .clone(),
            FilterType::Gaussian => file_type::CLASS.mutlifilt_gaussian_output_template.clone(),
            FilterType::HammingLikeStarts => file_type::CLASS
                .mutlifilt_hamming_like_starts_output_template
                .clone(),
        }
    }
}

impl EnumeratedType for FilterType {
    /// Java `isDefault()`: field `defaultInstance` (true only for
    /// `FAKE_SIRT_ITERATIONS`).
    fn is_default(&self) -> bool {
        *self == FilterType::FakeSirtIterations
    }

    /// Java `getValue()` returns null; the Rust trait returns a value, so a
    /// null (unset) number stands in.
    fn get_value(&self) -> ConstEtomoNumber {
        EtomoNumber::new().base
    }

    /// Java `getLabel()`.
    fn get_label(&self) -> Option<String> {
        Some(self.label().to_string())
    }
}

/// Java `toString()` is not overridden (Object's `Class@hash`); nothing reads
/// it.  The label is written.
impl std::fmt::Display for FilterType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}
