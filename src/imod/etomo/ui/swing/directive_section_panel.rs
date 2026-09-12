//! `IMOD/Etomo/src/etomo/ui/swing/DirectiveSectionPanel.java`.
//!
//! Swing's `JPanel`, `BoxLayout`, `JLabel`, `Box`, and `ActionListener` are
//! retained as presentation-boundary state.  `Directive`, `DirectiveMap`,
//! `DirectiveDescrSection`, `DirectiveTool`, and `DirectivePanel` are separate
//! Etomo source units; the interfaces below are deliberately narrow boundaries
//! for precisely the calls made by this source unit.
#![allow(dead_code)]

use std::collections::HashMap;

use super::check_box::CheckBox;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_type::AxisType;

/// Java `Directive`, at the storage-unit boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Directive {
    pub name: String,
}

/// Java `DirectiveDescrSection`, at the storage-unit boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectiveDescrSection {
    pub title: String,
    pub names: Vec<String>,
}

impl std::fmt::Display for DirectiveDescrSection {
    /// Java `toString()` as used by this source unit.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.title)
    }
}

/// Java `DirectiveMap`, at the storage-unit boundary.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DirectiveMap {
    pub directives: HashMap<String, Directive>,
}

impl DirectiveMap {
    /// Java `getDirective(String)`.
    pub fn get_directive(&self, name: &str) -> Option<&Directive> {
        self.directives.get(name)
    }
}

/// Java `DirectiveTool`, retained as the logic-unit identity passed to every
/// `DirectivePanel.getInstance` call.
#[derive(Default)]
pub struct DirectiveTool;

/// The source-facing portion of Java `DirectivePanel` used by this unit.
///
/// Its actual widget/value implementation belongs to `DirectivePanel.java`;
/// this trait prevents this section unit from replacing that logic.
pub trait DirectivePanel {
    /// Java static `DirectivePanel.getInstance(BaseManager, Directive,
    /// DirectiveTool, AxisType)`.
    fn get_instance(
        manager: &'static dyn BaseManager,
        directive: &Directive,
        tool: &'static DirectiveTool,
        source_axis_type: AxisType,
    ) -> Self;

    /// Java `msgControlChanged(boolean, boolean)`.
    fn msg_control_changed(&mut self, include_change: bool, expand_change: bool) -> bool;

    /// Java `isInclude()`.
    fn is_include(&self) -> bool;

    /// Java `isDifferentFromCheckpoint(boolean)`.
    fn is_different_from_checkpoint(&self, check_include: bool) -> bool;

    /// Java `getState()`.
    fn get_state(&mut self) -> Directive;

    /// Java `checkpoint()`.
    fn checkpoint(&mut self);
}

/// Swing `JPanel` state touched by this source unit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JPanelBoundary {
    pub visible: bool,
    pub layout: Option<&'static str>,
    pub border_title: Option<String>,
    pub rigid_area_x0_y3: bool,
    pub include_label: bool,
    pub horizontal_glue: bool,
    pub directive_component_count: usize,
}

impl Default for JPanelBoundary {
    fn default() -> Self {
        Self {
            visible: true,
            layout: None,
            border_title: None,
            rigid_area_x0_y3: false,
            include_label: false,
            horizontal_glue: false,
            directive_component_count: 0,
        }
    }
}

/// Java final `DirectiveSectionPanel`.
pub struct DirectiveSectionPanel<P: DirectivePanel> {
    /// Java `rcsid`.
    pub rcsid: &'static str,
    /// Java final `pnlRoot`.
    pub pnl_root: JPanelBoundary,
    /// Java final `pnlBody`.
    pub pnl_body: JPanelBoundary,
    /// Java final `directivePanelArray`.
    pub directive_panel_array: Vec<P>,
    /// Java final `pnlDirectives`.
    pub pnl_directives: JPanelBoundary,
    /// Java final `cbShow`.
    pub cb_show: CheckBox,
    /// Java final `manager`.
    pub manager: &'static dyn BaseManager,
    /// Java final `sourceAxisType`.
    pub source_axis_type: AxisType,
    /// Java final `tool`.
    pub tool: &'static DirectiveTool,
    /// Java final `descrSection`.
    pub descr_section: DirectiveDescrSection,
    /// Java final `directiveMap`.
    pub directive_map: DirectiveMap,
    /// Java `debug`, initialized to false and otherwise unused in the source.
    pub debug: bool,
    /// Native Swing `UIHarness.INSTANCE.pack(manager)` boundary invoked by `action`.
    pub pack_requested: bool,
    /// Native `DirectiveSectionListener` attachment boundary.
    pub listener_added: bool,
}

impl<P: DirectivePanel> DirectiveSectionPanel<P> {
    /// Java private `DirectiveSectionPanel(BaseManager, AxisType, DirectiveTool,
    /// DirectiveDescrSection, DirectiveMap)`.
    fn new(
        manager: &'static dyn BaseManager,
        source_axis_type: AxisType,
        tool: &'static DirectiveTool,
        descr_section: DirectiveDescrSection,
        directive_map: DirectiveMap,
    ) -> Self {
        let cb_show = CheckBox::new_with_text(&descr_section.to_string());
        Self {
            rcsid: "$Id:$",
            pnl_root: JPanelBoundary::default(),
            pnl_body: JPanelBoundary::default(),
            directive_panel_array: Vec::new(),
            pnl_directives: JPanelBoundary::default(),
            cb_show,
            manager,
            source_axis_type,
            tool,
            descr_section,
            directive_map,
            debug: false,
            pack_requested: false,
            listener_added: false,
        }
    }

    /// Java static `getInstance(BaseManager, DirectiveDescrSection, DirectiveMap,
    /// AxisType, DirectiveTool)`.
    pub fn get_instance(
        manager: &'static dyn BaseManager,
        descr_section: DirectiveDescrSection,
        directive_map: DirectiveMap,
        source_axis_type: AxisType,
        tool: &'static DirectiveTool,
    ) -> Self {
        let mut instance = Self::new(
            manager,
            source_axis_type,
            tool,
            descr_section,
            directive_map,
        );
        instance.create_panel();
        instance.add_listeners();
        instance.set_tooltips();
        instance
    }

    /// Java private `createPanel()`.
    fn create_panel(&mut self) {
        self.pnl_root.layout = Some("BoxLayout.Y_AXIS");
        self.pnl_root.border_title = Some(self.descr_section.to_string());

        self.pnl_body.layout = Some("BoxLayout.Y_AXIS");
        self.pnl_body.rigid_area_x0_y3 = true;
        self.pnl_body.include_label = true;
        self.pnl_body.horizontal_glue = true;

        self.pnl_directives.layout = Some("BoxLayout.Y_AXIS");
        for name in &self.descr_section.names {
            let Some(directive) = self.directive_map.get_directive(name) else {
                continue;
            };
            let directive_panel =
                P::get_instance(self.manager, directive, self.tool, self.source_axis_type);
            self.directive_panel_array.push(directive_panel);
            self.pnl_directives.directive_component_count += 1;
        }
        self.msg_control_changed(false, true, true);
    }

    /// Java `msgControlChanged(boolean, boolean, boolean)`.
    pub fn msg_control_changed(
        &mut self,
        include_change: bool,
        show_change: bool,
        expand_change: bool,
    ) {
        let mut visible_directives = false;
        let mut include = false;
        for directive_panel in &mut self.directive_panel_array {
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
            self.cb_show.set_selected(true);
        }
        self.cb_show.set_enabled(visible_directives);
        self.pnl_root.visible = self.cb_show.is_selected() && self.cb_show.is_enabled();
    }

    /// Java `isDifferentFromCheckpoint(boolean)`.
    pub fn is_different_from_checkpoint(&self, check_include: bool) -> bool {
        self.directive_panel_array
            .iter()
            .any(|directive_panel| directive_panel.is_different_from_checkpoint(check_include))
    }

    /// Java `close()`.
    pub fn close(&mut self) {
        self.cb_show.set_selected(false);
        self.pnl_root.visible = false;
    }

    /// Java `getComponent()` at the Swing component boundary.
    pub fn get_component(&self) -> &JPanelBoundary {
        &self.pnl_root
    }

    /// Java `getShowCheckBox()` at the Swing component boundary.
    pub fn get_show_check_box(&self) -> &CheckBox {
        &self.cb_show
    }

    /// Java `addListeners()`.
    pub fn add_listeners(&mut self) {
        self.cb_show.add_action_listener();
        self.listener_added = true;
    }

    /// Java `getIncludeDirectiveList()`.
    pub fn get_include_directive_list(&mut self) -> Vec<Directive> {
        let mut directive_list = Vec::new();
        for directive_set in &mut self.directive_panel_array {
            if directive_set.is_include() {
                directive_list.push(directive_set.get_state());
            }
        }
        directive_list
    }

    /// Java `checkpoint()`.
    pub fn checkpoint(&mut self) {
        for directive_panel in &mut self.directive_panel_array {
            directive_panel.checkpoint();
        }
    }

    /// Java private `action()`.
    fn action(&mut self) {
        self.pnl_root.visible = self.cb_show.is_enabled() && self.cb_show.is_selected();
        self.pack_requested = true;
    }

    /// Java private `setTooltips()`.
    fn set_tooltips(&mut self) {
        self.cb_show.set_tool_tip_text(Some("Show directives."));
    }

    /// Swing listener dispatch boundary for Java `DirectiveSectionListener`.
    pub fn action_performed(&mut self) {
        DirectiveSectionListener::action_performed(self);
    }
}

/// Java private static final `DirectiveSectionListener`.
pub struct DirectiveSectionListener;

impl DirectiveSectionListener {
    /// Java `actionPerformed(ActionEvent)`.
    pub fn action_performed<P: DirectivePanel>(panel: &mut DirectiveSectionPanel<P>) {
        panel.action();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::directive_editor_manager::DirectiveEditorManager;

    #[derive(Clone, Debug)]
    struct TestDirectivePanel {
        directive: Directive,
        include: bool,
        visible: bool,
        different: bool,
        checkpoint_count: usize,
        include_change: Option<bool>,
        expand_change: Option<bool>,
    }

    impl DirectivePanel for TestDirectivePanel {
        fn get_instance(
            _manager: &'static dyn BaseManager,
            directive: &Directive,
            _tool: &'static DirectiveTool,
            _source_axis_type: AxisType,
        ) -> Self {
            Self {
                include: directive.name == "included",
                visible: directive.name != "hidden",
                different: directive.name == "changed",
                directive: directive.clone(),
                checkpoint_count: 0,
                include_change: None,
                expand_change: None,
            }
        }

        fn msg_control_changed(&mut self, include_change: bool, expand_change: bool) -> bool {
            self.include_change = Some(include_change);
            self.expand_change = Some(expand_change);
            self.visible
        }

        fn is_include(&self) -> bool {
            self.include
        }

        fn is_different_from_checkpoint(&self, _check_include: bool) -> bool {
            self.different
        }

        fn get_state(&mut self) -> Directive {
            self.directive.clone()
        }

        fn checkpoint(&mut self) {
            self.checkpoint_count += 1;
        }
    }

    fn panel() -> DirectiveSectionPanel<TestDirectivePanel> {
        let manager: &'static dyn BaseManager = DirectiveEditorManager::new(None, None, None, None);
        let tool = Box::leak(Box::new(DirectiveTool));
        DirectiveSectionPanel::get_instance(
            manager,
            DirectiveDescrSection {
                title: "Test section".to_owned(),
                names: vec![
                    "hidden".to_owned(),
                    "missing".to_owned(),
                    "included".to_owned(),
                ],
            },
            DirectiveMap {
                directives: HashMap::from([
                    (
                        "hidden".to_owned(),
                        Directive {
                            name: "hidden".to_owned(),
                        },
                    ),
                    (
                        "included".to_owned(),
                        Directive {
                            name: "included".to_owned(),
                        },
                    ),
                ]),
            },
            AxisType::SingleAxis,
            tool,
        )
    }

    #[test]
    fn factory_builds_source_panel_hierarchy_skips_missing_directives_and_sets_tooltip() {
        let panel = panel();
        assert_eq!(panel.pnl_root.layout, Some("BoxLayout.Y_AXIS"));
        assert_eq!(panel.pnl_root.border_title.as_deref(), Some("Test section"));
        assert!(panel.pnl_body.rigid_area_x0_y3);
        assert!(panel.pnl_body.include_label);
        assert!(panel.pnl_body.horizontal_glue);
        assert_eq!(panel.pnl_directives.directive_component_count, 2);
        assert!(panel.listener_added);
        assert_eq!(
            panel.cb_show.check_box.tooltip.as_deref(),
            Some("<html>Show directives.")
        );
        assert!(panel.cb_show.is_selected());
        assert!(panel.get_component().visible);
    }

    #[test]
    fn control_changes_checkpoint_and_include_collection_follow_every_directive_panel() {
        let mut panel = panel();
        panel.msg_control_changed(true, false, true);
        assert_eq!(panel.directive_panel_array[0].include_change, Some(true));
        assert_eq!(panel.directive_panel_array[0].expand_change, Some(true));
        assert!(!panel.is_different_from_checkpoint(false));
        panel.directive_panel_array[0].different = true;
        assert!(panel.is_different_from_checkpoint(true));
        assert_eq!(
            panel.get_include_directive_list(),
            vec![Directive {
                name: "included".to_owned()
            }]
        );
        panel.checkpoint();
        assert_eq!(panel.directive_panel_array[0].checkpoint_count, 1);
        assert_eq!(panel.directive_panel_array[1].checkpoint_count, 1);
    }

    #[test]
    fn close_and_listener_action_use_checkbox_selection_enabled_state_and_pack_boundary() {
        let mut panel = panel();
        panel.close();
        assert!(!panel.cb_show.is_selected());
        assert!(!panel.get_component().visible);
        panel.cb_show.set_selected(true);
        panel.action_performed();
        assert!(panel.get_component().visible);
        assert!(panel.pack_requested);
        panel.cb_show.set_enabled(false);
        panel.action_performed();
        assert!(!panel.get_component().visible);
    }
}
