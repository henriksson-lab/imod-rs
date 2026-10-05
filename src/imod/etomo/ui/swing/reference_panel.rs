//! `IMOD/Etomo/src/etomo/ui/swing/ReferencePanel.java`.
//!
//! The PEET dialog's "Reference" box: a particle in a volume, a user supplied file, or
//! a multiparticle reference.  An event dispatch thread object, created as
//! `Rc<Self>` by [`ReferencePanel::get_instance`]; it keeps a weak reference to its
//! parent.

use std::rc::{Rc, Weak};

use super::combo_box::ComboBox;
use super::etched_border::EtchedBorder;
use super::etomo_panel::EtomoPanel;
use super::file_text_field2::FileTextField2;
use super::labeled_text_field::LabeledTextField;
use super::peet_dialog;
use super::radio_button::RadioButton;
use super::radio_text_field::RadioTextField;
use super::reference_parent::ReferenceParent;
use super::spinner::Spinner;
use super::swing_component::SwingComponent;
use super::ui_harness;
use crate::imod::etomo::base_manager::{BaseManager, ManagerBrowsingDirectory};
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, ButtonGroup, FileFilter, JComponent};
use crate::imod::etomo::logic::multiparticle_reference;
use crate::imod::etomo::storage::autodoc::autodoc_factory;
use crate::imod::etomo::storage::autodoc::read_only_autodoc::ReadOnlyAutodoc;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::storage::matlab_param::{self, MatlabParam};
use crate::imod::etomo::storage::volume_file_filter::VolumeFileFilter;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_peet_meta_data::ConstPeetMetaData;
use crate::imod::etomo::r#type::etomo_autodoc;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::peet_meta_data::PeetMetaData;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::shared_strings;
use crate::imod::etomo::ui::ui_component::UIComponent;
use crate::imod::etomo::util::file_path::FilePath;

/// Java private static final `TITLE`.
const TITLE: &str = "Reference";
/// Java private static final `REFERENCE_FILE_LABEL`.
const REFERENCE_FILE_LABEL: &str = "User supplied file: ";
/// Java private static final `VOLUME_LABEL`.
const VOLUME_LABEL: &str = "In Volume";

/// Java private static final `MULTIPARTICLE_BUTTON_LABEL`.
fn multiparticle_button_label() -> String {
    format!("{} with", shared_strings::FLG_FAIR_REFERENCE_LABEL)
}

/// Java package-private `final class ReferencePanel implements UIComponent,
/// SwingComponent`.
pub struct ReferencePanel {
    /// Java private final `pnlRoot`.
    pnl_root: Rc<EtomoPanel>,
    /// Java private final `bgReference`.
    bg_reference: Rc<ButtonGroup>,
    /// Java private final `rtfParticle`.
    rtf_particle: Rc<RadioTextField>,
    /// Java private final `sVolume`.
    s_volume: Rc<Spinner>,
    /// Java private final `rbFile`.
    rb_file: Rc<RadioButton>,
    /// Java private final `ftfFile`.
    ftf_file: Rc<FileTextField2>,
    /// Java private final `rbMultiparticle`.
    rb_multiparticle: Rc<RadioButton>,
    /// Java private final `cmbMultiparticle`.
    cmb_multiparticle: Rc<ComboBox>,
    /// Java private final `lMultiparticle`.
    l_multiparticle: Rc<JComponent>,
    /// Java private final `ltfVolume`.
    ltf_volume: Rc<LabeledTextField>,
    /// Java private final `parent`.
    parent: Weak<dyn ReferenceParent>,
    /// Java private final `manager`.
    manager: &'static dyn BaseManager,
    /// Java `this`.
    self_ref: Weak<ReferencePanel>,
}

impl ReferencePanel {
    /// Java private `ReferencePanel(ReferenceParent, BaseManager)`, with the field
    /// initializers.
    fn new(
        parent: Weak<dyn ReferenceParent>,
        manager: &'static dyn BaseManager,
    ) -> Rc<ReferencePanel> {
        let bg_reference = ButtonGroup::new();
        let multiparticle_label = multiparticle_button_label();
        Rc::new_cyclic(|self_ref: &Weak<ReferencePanel>| ReferencePanel {
            pnl_root: EtomoPanel::new(),
            rtf_particle: RadioTextField::get_instance_field_type_string_button_group_string(
                FieldType::Integer,
                Some("Particle "),
                Some(&bg_reference),
                Some(peet_dialog::SETUP_LOCATION_DESCR),
            ),
            s_volume: Spinner::get_labeled_instance_string(Some(&format!("{VOLUME_LABEL}: "))),
            rb_file: RadioButton::new_string_button_group(
                Some(REFERENCE_FILE_LABEL),
                Some(&bg_reference),
            ),
            // unlabeled
            ftf_file: FileTextField2::get_unlabeled_peet_instance(
                Some(manager),
                Some(REFERENCE_FILE_LABEL),
            ),
            rb_multiparticle: RadioButton::new_string_button_group(
                Some(&multiparticle_label),
                Some(&bg_reference),
            ),
            cmb_multiparticle: ComboBox::get_unlabeled_instance(Some(&multiparticle_label)),
            l_multiparticle: JComponent::new_label("particles"),
            ltf_volume: LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some(&format!("{VOLUME_LABEL}: ")),
            ),
            bg_reference,
            parent,
            manager,
            self_ref: self_ref.clone(),
        })
    }

    /// Java static `getInstance(ReferenceParent, BaseManager)`.
    pub fn get_instance(
        parent: Weak<dyn ReferenceParent>,
        manager: &'static dyn BaseManager,
    ) -> Rc<ReferencePanel> {
        let instance = ReferencePanel::new(parent, manager);
        instance.create_panel();
        instance.set_tooltips();
        instance.add_listeners();
        instance
    }

    /// Java field read `parent`.
    fn parent(&self) -> Rc<dyn ReferenceParent> {
        self.parent
            .upgrade()
            .expect("the PEET dialog owns its reference panel")
    }

    /// Java private `addListeners()` with `ReferenceActionListener`.
    fn add_listeners(&self) {
        let adaptee = self.self_ref.clone();
        let action_listener: ActionListener = Rc::new(move |event: &ActionEvent| {
            if let Some(reference_panel) = adaptee.upgrade() {
                reference_panel.action(event.get_action_command().unwrap_or(""));
            }
        });
        self.rtf_particle
            .add_action_listener(action_listener.clone());
        self.rb_file.add_action_listener(action_listener.clone());
        self.rb_multiparticle.add_action_listener(action_listener);
    }

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        // Init
        let n_entries = multiparticle_reference::get_num_entries();
        for i in 0..n_entries {
            self.cmb_multiparticle.add_item(Some(
                &multiparticle_reference::get_particle_count(i).to_string(),
            ));
        }
        self.cmb_multiparticle
            .set_selected_index(multiparticle_reference::get_default_index());
        self.ftf_file.set_adjusted_field_width(225.0);
        self.ftf_file
            .set_file_filter(Some(
                Rc::new(VolumeFileFilter::get_instance(Some(self.manager))) as Rc<dyn FileFilter>,
            ));
        self.ftf_file.set_use_text_as_file_chooser_dir(true);
        self.ftf_file
            .set_browsing_directory(Some(Rc::new(ManagerBrowsingDirectory(self.manager))));
        // local panels
        let pnl_particle = JComponent::new_panel();
        let pnl_file = JComponent::new_panel();
        let pnl_multiparticle = JComponent::new_panel();
        let pnl_border = JComponent::new_panel();
        // Root (BoxLayout Y_AXIS, rigid area 0x8 after the border)
        self.pnl_root.get_component().add(&pnl_border);
        // border (BoxLayout Y_AXIS, rigid area 0x34 at the end)
        pnl_border.set_border_title(
            EtchedBorder::new(Some(TITLE))
                .get_border()
                .get_title()
                .as_deref(),
        );
        pnl_border.add(&pnl_particle);
        pnl_border.add(&pnl_file);
        pnl_border.add(&pnl_multiparticle);
        // particle panel (BoxLayout X_AXIS)
        pnl_particle.add(&self.rtf_particle.get_container());
        pnl_particle.add(&self.s_volume.get_container());
        pnl_particle.add(&self.ltf_volume.get_container());
        // file panel (BoxLayout X_AXIS)
        pnl_file.add(&self.rb_file.get_component());
        pnl_file.add(&self.ftf_file.get_root_panel());
        // multiparticle panel (BoxLayout X_AXIS)
        pnl_multiparticle.add(&self.rb_multiparticle.get_component());
        pnl_multiparticle.add(&self.cmb_multiparticle.get_component());
        pnl_multiparticle.add(&self.l_multiparticle);
    }

    /// Java package-private `convertCopiedPaths(String)`.  Make the copied path
    /// relative to this dataset, preserving the location of the files that the old
    /// dataset was using.  If the file path is absolute, don't change it.
    pub fn convert_copied_paths(&self, orig_dataset_dir: &str) {
        let property_user_dir = self.manager.get_property_user_dir();
        if !self.ftf_file.is_empty() {
            self.ftf_file.set_text_string(
                FilePath::get_rerooted_relative_path(
                    Some(orig_dataset_dir),
                    property_user_dir.as_deref(),
                    self.ftf_file.get_text_void().as_deref(),
                )
                .as_deref(),
            );
        }
    }

    /// Java package-private `isIncorrectPaths()`.
    pub fn is_incorrect_paths(&self) -> bool {
        self.is_reference_file_selected() && !self.ftf_file.is_empty() && !self.ftf_file.exists()
    }

    /// Java package-private `fixIncorrectPaths(boolean)`.  If ftfReferenceFile has an
    /// invalid path, call ReferenceParent.fixIncorrectPath(FileTextField,boolean).
    /// Returns true to keep fixing paths.  Returns false to stop fixing paths.
    pub fn fix_incorrect_paths(&self, choose_path_every_row: bool) -> bool {
        if self.is_incorrect_paths() {
            return self
                .parent()
                .fix_incorrect_path(&*self.ftf_file, choose_path_every_row);
        }
        true
    }

    /// Java package-private `getParameters(PeetMetaData)`.  Send values to
    /// PeetMetaData.
    pub fn get_parameters_meta_data(&self, meta_data: &PeetMetaData) {
        if self.s_volume.is_visible() {
            meta_data.set_reference_volume_number(Some(self.s_volume.get_value()));
        } else {
            meta_data.set_reference_volume_string(self.ltf_volume.get_text_void().as_deref());
        }
        meta_data.set_reference_particle(self.rtf_particle.get_text_void().as_deref());
        meta_data.set_reference_file(self.ftf_file.get_text_void().as_deref());
        meta_data.set_reference_multiparticle_level(Some(
            &multiparticle_reference::convert_index_to_level(
                self.cmb_multiparticle.get_selected_index(),
            ),
        ));
    }

    /// Java package-private `setParameters(ConstPeetMetaData)`.  Load data from
    /// ConstPeetMetaData.
    pub fn set_parameters_meta_data(&self, meta_data: &dyn ConstPeetMetaData) {
        self.ftf_file
            .set_text_string(meta_data.get_reference_file().as_deref());
        self.rtf_particle
            .set_text_const_etomo_number(&meta_data.get_reference_particle());
        self.cmb_multiparticle.set_selected_index(
            multiparticle_reference::convert_level_to_index_int(
                meta_data.get_reference_multiparticle_level(),
            ),
        );
        if self.s_volume.is_visible() {
            self.s_volume
                .set_value_const_etomo_number(&meta_data.get_reference_volume());
        } else {
            self.ltf_volume
                .set_text_const_etomo_number(Some(&meta_data.get_reference_volume()));
        }
    }

    /// Java package-private `setParameters(MatlabParam)`.  Load active data from
    /// MatlabParam.
    pub fn set_parameters_matlab_param(&self, matlab_param: &MatlabParam) {
        if matlab_param.use_reference_file() {
            self.rb_file.set_selected_boolean(true);
            self.ftf_file
                .set_text_string(matlab_param.get_reference_file().as_deref());
        } else if matlab_param.is_flg_fair_reference() {
            self.rb_multiparticle.set_selected_boolean(true);
            let level = matlab_param.get_reference_level();
            let mut index = EtomoNumber::new();
            if !multiparticle_reference::convert_level_to_index_string(level.as_deref(), &mut index)
            {
                ui_harness::with(|harness| {
                    harness.open_problem_value_message_dialog(
                        Some(self.manager),
                        Some(self as &dyn UIComponent),
                        "Incorrect",
                        Some(matlab_param::REFERENCE_KEY),
                        Some("level"),
                        Some(shared_strings::FLG_FAIR_REFERENCE_LABEL),
                        level.as_deref(),
                        Some(&multiparticle_reference::convert_index_to_level(
                            index.get_int(),
                        )),
                        None,
                    )
                });
            }
            self.cmb_multiparticle.set_selected_index(index.get_int());
        } else {
            self.rtf_particle.set_selected_boolean(true);
            self.rtf_particle
                .set_text_string(matlab_param.get_reference_particle().as_deref());
            if self.s_volume.is_visible() {
                self.s_volume
                    .set_value_parsed_element(matlab_param.get_reference_volume());
            } else {
                self.ltf_volume
                    .set_text_string(matlab_param.get_reference_volume_string().as_deref());
            }
        }
    }

    /// Java package-private `getParameters(MatlabParam, boolean)`.  Send active data
    /// to MatlabParam.
    pub fn get_parameters_matlab_param(
        &self,
        matlab_param: &mut MatlabParam,
        do_validation: bool,
    ) -> bool {
        if self.rtf_particle.is_selected() {
            if self.s_volume.is_visible() {
                matlab_param.set_reference_volume_number(self.s_volume.get_value());
            } else {
                matlab_param
                    .set_reference_volume_string(self.ltf_volume.get_text_void().as_deref());
            }
            let Ok(particle) = self.rtf_particle.get_text_boolean(do_validation) else {
                // catch (FieldValidationFailedException e) { return false; }
                return false;
            };
            matlab_param.set_reference_particle_string(particle.as_deref());
        } else if self.rb_file.is_selected() {
            matlab_param.set_reference_file(self.ftf_file.get_text_void().as_deref());
        } else if self.rb_multiparticle.is_selected() {
            matlab_param.set_flg_fair_reference(true);
            matlab_param.set_reference_level(Some(
                &multiparticle_reference::convert_index_to_level(
                    self.cmb_multiparticle.get_selected_index(),
                ),
            ));
        }
        true
    }

    /// Java package-private `isReferenceFileSelected()`.
    pub fn is_reference_file_selected(&self) -> bool {
        self.rb_file.is_selected()
    }

    /// Java package-private `isReferenceParticleSelected()`.
    pub fn is_reference_particle_selected(&self) -> bool {
        self.rtf_particle.is_selected()
    }

    /// Java package-private `msgFlgVolNamesAreTemplates(boolean, boolean)`.
    pub fn msg_flg_vol_names_are_templates(&self, init: bool, on: bool) {
        if on && !init && self.ltf_volume.is_empty() {
            self.ltf_volume
                .set_text_number(Some(self.s_volume.get_value()));
        }
        self.s_volume.set_visible(!on);
        self.ltf_volume.set_visible(on);
    }

    /// Java private `action(String)`.
    fn action(&self, action_command: &str) {
        if Some(action_command) == self.rtf_particle.get_action_command().as_deref()
            || Some(action_command) == self.rb_file.get_action_command().as_deref()
            || Some(action_command) == self.rb_multiparticle.get_action_command().as_deref()
        {
            self.parent().update_display(false);
        }
    }

    /// Java package-private `validateRun()`.  Validatation for fields to be used when
    /// prmParser is run.  Returns an error string if invalid.
    pub fn validate_run(&self) -> Option<String> {
        // Must either have a volume and particle or a reference file.
        // Must have particle number if volume is selected
        if self.rtf_particle.is_selected() && self.rtf_particle.is_empty() {
            return Some(format!(
                "In {TITLE}, {} is required when {} is selected.",
                self.rtf_particle
                    .get_label()
                    .unwrap_or_else(|| "null".to_owned()),
                self.s_volume
                    .get_label()
                    .unwrap_or_else(|| "null".to_owned())
            ));
        }
        // Must have a reference file if reference file is selected
        if self.rb_file.is_selected() && self.ftf_file.is_empty() {
            return Some(format!(
                "In {TITLE}, a file is required when {} is selected.",
                self.rb_file.get_component().get_text()
            ));
        }
        None
    }

    /// Java package-private `setDefaults()`.
    pub fn set_defaults(&self) {
        self.rtf_particle.set_selected_boolean(true);
    }

    /// Java package-private `updateDisplay(boolean)`.
    pub fn update_display(&self, init: bool) {
        let parent = self.parent();
        self.rtf_particle
            .set_enabled(parent.get_volume_table_size() > 0);
        self.s_volume.set_enabled(self.rtf_particle.is_selected());
        self.ltf_volume.set_enabled(self.rtf_particle.is_selected());
        self.s_volume.set_max(parent.get_volume_table_size());
        self.ftf_file.set_enabled(self.rb_file.is_selected());
        self.cmb_multiparticle
            .set_enabled(self.rb_multiparticle.is_selected());
        self.l_multiparticle
            .set_enabled(self.rb_multiparticle.is_selected());
        self.msg_flg_vol_names_are_templates(init, parent.is_flg_vol_names_are_templates());
    }

    /// Java private `setTooltips()`.
    fn set_tooltips(&self) {
        let autodoc = match unsafe {
            autodoc_factory::get_instance(
                Some(self.manager),
                Some(autodoc_factory::PEET_PRM),
                AxisID::Only,
                false,
            )
        } {
            Ok(autodoc) => autodoc,
            Err(LogFileError::Lock(_)) => std::ptr::null_mut(),
            Err(e) => {
                eprintln!("{e}");
                std::ptr::null_mut()
            }
        };
        let autodoc = unsafe { autodoc.as_ref() }.map(|autodoc| autodoc as &dyn ReadOnlyAutodoc);
        let text = "The number of the volume containing the reference.";
        self.s_volume.set_tool_tip_text(Some(text));
        self.ltf_volume.set_tool_tip_text(Some(text));
        self.rtf_particle.set_radio_button_tool_tip_text(Some(
            "Specify the reference by volume and particle numbers.",
        ));
        self.rtf_particle.set_text_field_tool_tip_text(Some(
            "The number of the particle to use as the reference.",
        ));
        self.rb_file
            .set_tool_tip_text_string(Some("Specify the reference by filename."));
        self.ftf_file.set_tool_tip_text(Some(
            "The name of the file containing the MRC volume to use as the reference.",
        ));
        self.rb_multiparticle.set_tool_tip_text_string(
            etomo_autodoc::get_tooltip_autodoc_add_source(
                autodoc,
                Some(matlab_param::FLG_FAIR_REFERENCE_KEY),
                false,
            )
            .as_deref(),
        );
        self.cmb_multiparticle.set_tool_tip_text(Some(
            "Number of particles to be used to generate a multi-particle reference.",
        ));
    }
}

impl SwingComponent for ReferencePanel {
    /// Java `getComponent()`.
    fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.get_component()
    }
}

impl UIComponent for ReferencePanel {
    /// Java `getUIComponent()`.
    fn get_ui_component(&self) -> &dyn SwingComponent {
        self
    }

    fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.get_component()
    }
}
