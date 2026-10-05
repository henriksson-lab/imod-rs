//! `IMOD/Etomo/src/etomo/ui/swing/FixPathsPanel.java`.
//!
//! Panel for fixing incorrect file paths: hidden until the PEET dialog finds a file
//! that does not exist.  An event dispatch thread object, created as `Rc<Self>` by
//! [`FixPathsPanel::get_instance`]; it keeps a weak reference to its file container.

use std::rc::{Rc, Weak};

use super::check_box::CheckBox;
use super::etomo_panel::EtomoPanel;
use super::expand_button::ExpandButton;
use super::expandable::Expandable;
use super::file_container::FileContainer;
use super::global_expand_button::GlobalExpandButton;
use super::multi_line_button::MultiLineButton;
use super::panel_header::PanelHeader;
use super::ui_harness;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::jdk::{ActionEvent, JComponent};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::dialog_type::DialogType;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id$";

/// Java package-private `final class FixPathsPanel implements Expandable`.
pub struct FixPathsPanel {
    /// Java private final `pnlRoot`.
    pnl_root: Rc<JComponent>,
    /// Java private final `pnlMain`.
    pnl_main: Rc<EtomoPanel>,
    /// Java private final `pnlBody`.
    pnl_body: Rc<JComponent>,
    /// Java private final `lblwarning`.
    lblwarning: Rc<JComponent>,
    /// Java private final `cbChoosePathEveryRow`.
    cb_choose_path_every_row: Rc<CheckBox>,
    /// Java private final `bnFixPaths`.
    bn_fix_paths: Rc<MultiLineButton>,
    /// Java private final `header`.
    header: Rc<PanelHeader>,
    /// Java private final `fileContainer`.
    file_container: Weak<dyn FileContainer>,
    /// Java private final `manager`.
    manager: &'static dyn BaseManager,
    /// Java private final `axisID`.
    axis_id: AxisID,
    /// Java `this`.
    self_ref: Weak<FixPathsPanel>,
}

impl FixPathsPanel {
    /// Java private `setTooltips()`.
    fn set_tooltips(&self) {
        self.cb_choose_path_every_row.set_tool_tip_text_string(Some(
            "Causes a file chooser to be brought up for each row in the Volume Table.  Otherwise the file chooser will only be brought up when a file cannot be found in either the original path or the most recent new path.",
        ));
        self.bn_fix_paths.set_tool_tip_text(Some(
            "Brings up file chooser(s) so that the new location(s) of any files that cannot be found can be specified.",
        ));
    }

    /// Java private `FixPathsPanel(FileContainer, BaseManager, AxisID, DialogType)`.
    fn new(
        file_container: Weak<dyn FileContainer>,
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        dialog_type: Option<DialogType>,
    ) -> Rc<FixPathsPanel> {
        let this = Rc::new_cyclic(|self_ref: &Weak<FixPathsPanel>| {
            let expandable: Weak<dyn Expandable> = self_ref.clone();
            FixPathsPanel {
                pnl_root: JComponent::new_panel(),
                pnl_main: EtomoPanel::new(),
                pnl_body: JComponent::new_panel(),
                lblwarning: JComponent::new_label("Files cannot be found.  PEET may not run."),
                cb_choose_path_every_row: CheckBox::new_string(Some(
                    "Files may be in separate directories",
                )),
                bn_fix_paths: MultiLineButton::new_string(Some("Fix Incorrect Paths")),
                header: PanelHeader::get_instance(
                    Some("Fix File Paths"),
                    Some(expandable),
                    dialog_type,
                ),
                file_container,
                manager,
                axis_id,
                self_ref: self_ref.clone(),
            }
        });
        // root panel (BoxLayout X_AXIS)
        this.pnl_root.set_visible(false);
        this.pnl_root.add(&this.pnl_main.get_component());
        // main panel (BoxLayout Y_AXIS, an untitled etched border)
        this.pnl_main.add(&this.header);
        this.pnl_main.get_component().add(&this.pnl_body);
        // body panel (BoxLayout Y_AXIS)
        // Swing painting: lblwarning.setForeground(ProcessControlPanel.colorNotStarted).
        this.pnl_body.add(&this.lblwarning);
        this.pnl_body
            .add(&this.cb_choose_path_every_row.get_component());
        let pnl_button = JComponent::new_panel();
        // pnlButton (BoxLayout X_AXIS, horizontal glue on both sides)
        pnl_button.add(&this.bn_fix_paths.get_component());
        this.pnl_body.add(&pnl_button);
        // Swing layout: minimize the size once the panels contains all their
        // elements (pnlMain.setMaximumSize(boxLayout.preferredLayoutSize(pnlMain))).
        this
    }

    /// Java static `getInstance(FileContainer, BaseManager, AxisID, DialogType)`.
    pub fn get_instance(
        file_container: Weak<dyn FileContainer>,
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        dialog_type: Option<DialogType>,
    ) -> Rc<FixPathsPanel> {
        let instance = FixPathsPanel::new(file_container, manager, axis_id, dialog_type);
        instance.set_tooltips();
        instance.add_listeners();
        instance
    }

    /// Java private `addListeners()` with `FixPathsPanelListener`.
    fn add_listeners(&self) {
        let adaptee = self.self_ref.clone();
        self.bn_fix_paths
            .add_action_listener(Rc::new(move |_event: &ActionEvent| {
                if let Some(adaptee) = adaptee.upgrade() {
                    adaptee.action();
                }
            }));
    }

    /// Java package-private `getRootComponent()`.
    pub fn get_root_component(&self) -> Rc<JComponent> {
        self.pnl_root.clone()
    }

    /// Java package-private `setIncorrectPaths(boolean)`.
    pub fn set_incorrect_paths(&self, incorrect_paths: bool) {
        if incorrect_paths {
            self.pnl_root.set_visible(true);
            self.lblwarning.set_visible(true);
        } else {
            self.lblwarning.set_visible(false);
        }
    }

    /// Java private `action()`.
    fn action(&self) {
        if let Some(file_container) = self.file_container.upgrade() {
            file_container.fix_incorrect_paths(self.cb_choose_path_every_row.is_selected());
        }
    }
}

impl Expandable for FixPathsPanel {
    /// Java `expand(ExpandButton)`.
    fn expand_expand_button(&self, button: &Rc<ExpandButton>) {
        if self.header.equals_open_close(button) {
            self.pnl_body.set_visible(button.is_expanded());
        }
        ui_harness::with(|harness| {
            harness.pack_axis_id_base_manager(Some(self.axis_id), Some(self.manager))
        });
    }

    /// Java `expand(GlobalExpandButton)`: empty.
    fn expand_global_expand_button(&self, _button: &Rc<GlobalExpandButton>) {}
}
