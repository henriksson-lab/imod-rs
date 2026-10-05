//! `IMOD/Etomo/src/etomo/ui/swing/DirectiveSectionPanel.java`.
//!
//! One section of `directives.csv` in the directive editor: an etched-border panel
//! titled with the section header, an "Include:" label and a `DirectivePanel` per
//! directive, plus the "show" check box that the dialog places at the top of its
//! column.  An event dispatch thread object, created as `Rc<Self>` by
//! [`DirectiveSectionPanel::get_instance`].

use std::cell::{Cell, RefCell};
use std::rc::{Rc, Weak};
use std::sync::Arc;

use super::check_box::CheckBox;
use super::directive_panel::DirectivePanel;
use super::etched_border::EtchedBorder;
use super::ui_harness;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::jdk::{ActionEvent, JComponent};
use crate::imod::etomo::logic::directive_tool::DirectiveTool;
use crate::imod::etomo::storage::directive::Directive;
use crate::imod::etomo::storage::directive_descr_section::DirectiveDescrSection;
use crate::imod::etomo::storage::directive_map::DirectiveMap;
use crate::imod::etomo::r#type::axis_type::AxisType;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java package-private `final class DirectiveSectionPanel`.
pub struct DirectiveSectionPanel {
    /// Java private final `pnlRoot`.
    pnl_root: Rc<JComponent>,
    /// Java private final `pnlBody`.
    pnl_body: Rc<JComponent>,
    /// Java private final `directivePanelArray`.
    directive_panel_array: RefCell<Vec<Rc<DirectivePanel>>>,
    /// Java private final `pnlDirectives`.
    pnl_directives: Rc<JComponent>,
    /// Java private final `cbShow`.
    cb_show: Rc<CheckBox>,
    /// Java private final `manager`.
    manager: &'static dyn BaseManager,
    /// Java private final `sourceAxisType`.
    source_axis_type: AxisType,
    /// Java private final `tool`.
    tool: Rc<DirectiveTool>,
    /// Java private final `descrSection`.
    descr_section: Rc<DirectiveDescrSection>,
    /// Java private `debug`, initialised to false.  Never read in the source.
    debug: Cell<bool>,
    /// Java `this`.
    self_ref: Weak<DirectiveSectionPanel>,
}

impl DirectiveSectionPanel {
    /// Java private `DirectiveSectionPanel(BaseManager, AxisType, DirectiveTool,
    /// DirectiveDescrSection, DirectiveMap)`.  The map is only read by `createPanel`,
    /// so it is passed there rather than kept.
    fn new(
        manager: &'static dyn BaseManager,
        source_axis_type: AxisType,
        tool: Rc<DirectiveTool>,
        descr_section: Rc<DirectiveDescrSection>,
    ) -> Rc<DirectiveSectionPanel> {
        let cb_show = CheckBox::new_string(Some(&descr_section.to_string()));
        Rc::new_cyclic(|self_ref| DirectiveSectionPanel {
            pnl_root: JComponent::new_panel(),
            pnl_body: JComponent::new_panel(),
            directive_panel_array: RefCell::new(Vec::new()),
            pnl_directives: JComponent::new_panel(),
            cb_show,
            manager,
            source_axis_type,
            tool,
            descr_section,
            debug: Cell::new(false),
            self_ref: self_ref.clone(),
        })
    }

    /// Java package-private static `getInstance(BaseManager, DirectiveDescrSection,
    /// DirectiveMap, AxisType, DirectiveTool)`.
    pub fn get_instance(
        manager: &'static dyn BaseManager,
        descr_section: Rc<DirectiveDescrSection>,
        directive_map: &DirectiveMap,
        source_axis_type: AxisType,
        tool: Rc<DirectiveTool>,
    ) -> Rc<DirectiveSectionPanel> {
        let instance = DirectiveSectionPanel::new(manager, source_axis_type, tool, descr_section);
        instance.create_panel(directive_map);
        instance.add_listeners();
        instance.set_tooltips();
        instance
    }

    /// Java private `createPanel()`.
    fn create_panel(&self, directive_map: &DirectiveMap) {
        // root panel (BoxLayout Y_AXIS)
        self.pnl_root.set_border_title(
            EtchedBorder::new(Some(&self.descr_section.to_string()))
                .get_title()
                .as_deref(),
        );
        self.pnl_root.add(&self.pnl_body);
        // body panel (BoxLayout Y_AXIS)
        // Swing layout: pnlBody.add(Box.createRigidArea(FixedDim.x0_y3)).
        let pnl = JComponent::new_panel();
        // pnl (BoxLayout X_AXIS)
        pnl.add(&JComponent::new_label("Include:"));
        // Swing layout: pnl.add(Box.createHorizontalGlue()).
        self.pnl_body.add(&pnl);
        self.pnl_body.add(&self.pnl_directives);
        // directives panel (BoxLayout Y_AXIS)
        for name in self.descr_section.name_iterator() {
            let directive: Option<Arc<Directive>> = directive_map.get_directive_string(Some(&name));
            let Some(directive) = directive else {
                continue;
            };
            // directive set panels
            let directive_panel = DirectivePanel::get_instance(
                self.manager,
                directive,
                self.tool.clone(),
                self.source_axis_type,
            );
            self.directive_panel_array
                .borrow_mut()
                .push(directive_panel.clone());
            self.pnl_directives.add(&directive_panel.get_component());
        }
        self.msg_control_changed(false, true, true);
    }

    /// Java package-private `msgControlChanged(boolean, boolean, boolean)`.
    pub fn msg_control_changed(
        &self,
        include_change: bool,
        show_change: bool,
        expand_change: bool,
    ) {
        let mut visible_directives = false;
        let mut include = false;
        let directive_panel_array = self.directive_panel_array.borrow().clone();
        for directive_panel in &directive_panel_array {
            if directive_panel.msg_control_changed(include_change, expand_change)
                && !visible_directives
            {
                visible_directives = true;
            }
            if show_change && !include && directive_panel.is_include() {
                include = true;
            }
        }
        if include {
            self.cb_show.set_selected_boolean(true);
        }
        self.cb_show.set_enabled(visible_directives);
        self.pnl_root
            .set_visible(self.cb_show.is_selected() && self.cb_show.is_enabled());
    }

    /// Java package-private `isDifferentFromCheckpoint(boolean)`.
    pub fn is_different_from_checkpoint(&self, check_include: bool) -> bool {
        self.directive_panel_array
            .borrow()
            .iter()
            .any(|directive_panel| directive_panel.is_different_from_checkpoint(check_include))
    }

    /// Java package-private `close()`.
    pub fn close(&self) {
        self.cb_show.set_selected_boolean(false);
        self.pnl_root.set_visible(false);
    }

    /// Java package-private `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.clone()
    }

    /// Java package-private `getShowCheckBox()`.
    pub fn get_show_check_box(&self) -> Rc<JComponent> {
        self.cb_show.get_component()
    }

    /// Java package-private `addListeners()`.
    pub fn add_listeners(&self) {
        let adaptee = self.self_ref.clone();
        self.cb_show
            .add_action_listener(Some(Rc::new(move |_event: &ActionEvent| {
                if let Some(adaptee) = adaptee.upgrade() {
                    adaptee.action();
                }
            })));
    }

    /// Java package-private `getIncludeDirectiveList()`.
    pub fn get_include_directive_list(&self) -> Vec<Arc<Directive>> {
        let mut directive_list = Vec::new();
        for directive_set in self.directive_panel_array.borrow().iter() {
            if directive_set.is_include() {
                directive_list.push(directive_set.get_state());
            }
        }
        directive_list
    }

    /// Java package-private `checkpoint()`.
    pub fn checkpoint(&self) {
        for directive_panel in self.directive_panel_array.borrow().iter() {
            directive_panel.checkpoint();
        }
    }

    /// Java private `action()`.
    fn action(&self) {
        self.pnl_root
            .set_visible(self.cb_show.is_enabled() && self.cb_show.is_selected());
        let manager = self.manager;
        ui_harness::with(|harness| harness.pack_base_manager(Some(manager)));
    }

    /// Java private `setTooltips()`.
    fn set_tooltips(&self) {
        self.cb_show
            .set_tool_tip_text_string(Some("Show directives."));
    }
}
