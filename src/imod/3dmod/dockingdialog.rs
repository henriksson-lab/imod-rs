//! Translation of `IMOD/3dmod/dockingdialog.cpp` and `dockingdialog.h`.
//!
//! `DockingDialog` is a Qt container rather than a dialog policy class.  The
//! widget/layout/icon/event calls remain in `DockingDialogNativeBoundary`; the
//! source-owned state transitions and the calls into `DialogManager` are kept
//! here one-for-one.
#![allow(dead_code)]

use crate::imod::three_dmod::control::{DialogManager, DialogManagerNativeBoundary};

pub const TOOLBUT_SIZE: i32 = 16;
pub const BM_WIDTH: i32 = 12;
pub const BM_HEIGHT: i32 = 12;

/// The Qt event kinds inspected by `DockingDialog::event` and
/// `DockingDialog::changeEvent`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DockingEventType {
    ActivationChange,
    WindowStateChange,
    Hide,
    Show,
    FontChange,
    Other,
}

/// The portion of a Qt window state used by `DockingDialog::event`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DockingWindowState {
    pub minimized: bool,
    pub no_state: bool,
    pub maximized: bool,
}

/// Qt/application operations performed by `DockingDialog`.
///
/// Widget arguments are opaque native widget identities.  A frontend which
/// owns Qt can use its `QWidget *` cast to `usize`; a non-Qt frontend can use
/// stable application handles.  Manager operations intentionally use the
/// already-existing `DialogManagerNativeBoundary` passed to the methods.
pub trait DockingDialogNativeBoundary {
    fn create_layouts_and_stack(&mut self, _docker: usize) {}
    fn add_icon_file(&mut self, _icon: &str, _file: &str, _width: i32, _height: i32) {}
    fn configure_open_close_button(
        &mut self,
        _docker: usize,
        _size: i32,
        _icon: &str,
        _tool_tip: &str,
    ) {
    }
    fn configure_title(&mut self, _title: &str) {}
    fn configure_help_button(&mut self, _size: i32, _icon: &str, _tool_tip: &str) {}
    fn set_open_close_icon_and_tool_tip(&mut self, _icon: &str, _tool_tip: &str) {}
    fn show_stack(&mut self, _show: bool) {}
    /// `diaShowWidget`.
    fn dia_show_widget(&mut self, _show: bool) {}
    fn process_events(&mut self) {}
    fn resize(&mut self, _width: i32, _height: i32) {}
    fn adjust_size(&mut self) {}
    fn frame_height(&self) -> i32 {
        0
    }
    fn frame_width(&self) -> i32 {
        0
    }
    fn widget_width(&self) -> i32 {
        0
    }
    fn widget_height(&self) -> i32 {
        0
    }
    fn size_hint_width(&self) -> i32 {
        0
    }
    fn size_hint_height(&self) -> i32 {
        0
    }
    fn minimum_width(&self) -> i32 {
        0
    }
    fn set_minimum_width(&mut self, _width: i32) {}
    fn dialog_layout_margins(&self, _dialog: usize) -> Option<(i32, i32, i32, i32)> {
        None
    }
    fn set_dialog_layout_margins(
        &mut self,
        _dialog: usize,
        _left: i32,
        _top: i32,
        _right: i32,
        _bottom: i32,
    ) {
    }
    fn set_dialog_layout_spacing(&mut self, _dialog: usize, _spacing: i32) {}
    fn stack_add_widget(&mut self, _dialog: usize) {}
    fn initialize_screen_change(&mut self, _docker: usize, _current_dpr: f32) {}
    fn show_help_page(&mut self, _page: &str) {}
    fn docker_close_event(&mut self) {}
    fn docker_change_event(&mut self) {}
    /// `QWidget::event(e)`.
    fn widget_event(&mut self, _event: DockingEventType) -> bool {
        true
    }
}

/// `DockingDialog`.
#[derive(Clone, Debug)]
pub struct DockingDialog {
    pub m_help_page: Option<String>,
    pub m_key_letter: char,
    pub m_dialog: Option<usize>,
    pub m_opened: bool,
    pub m_frame_width: i32,
    pub m_open_height: i32,
    pub m_closed_height: i32,
    pub m_hinted_width: i32,
    pub m_hinted_height: i32,
    pub m_minimized: bool,
    pub m_cur_dev_pix_ratio: f32,
    pub m_screen_changed: bool,
    pub widget: usize,
    pub inner_title: String,
}

impl DockingDialog {
    /// `DockingDialog()` source constructor.
    pub fn new(
        widget: usize,
        inner_title: &str,
        help_page: Option<&str>,
        key_letter: char,
        native: &mut dyn DockingDialogNativeBoundary,
    ) -> Self {
        let dialog = Self {
            m_help_page: help_page.map(str::to_owned),
            m_key_letter: key_letter,
            m_dialog: None,
            m_opened: false,
            m_frame_width: 0,
            m_open_height: 0,
            m_closed_height: 0,
            m_hinted_width: 0,
            m_hinted_height: 0,
            m_minimized: false,
            m_cur_dev_pix_ratio: 0.,
            m_screen_changed: false,
            widget,
            inner_title: inner_title.to_owned(),
        };
        native.create_layouts_and_stack(widget);
        if dialog.m_help_page.is_some() {
            native.add_icon_file("help", ":/images/questionMark.png", BM_WIDTH, BM_HEIGHT);
        }
        native.add_icon_file("open", ":/images/plusOpen.png", BM_WIDTH, BM_HEIGHT);
        native.add_icon_file("close", ":/images/minusClose.png", BM_WIDTH, BM_HEIGHT);
        native.configure_open_close_button(widget, TOOLBUT_SIZE, "open", "");
        native.configure_title(inner_title);
        if dialog.m_help_page.is_some() {
            native.configure_help_button(TOOLBUT_SIZE, "help", "Open Help page for this dialog");
        }
        dialog
    }

    /// `DockingDialog::~DockingDialog`.  Rust drops the optional `String`.
    pub fn destroy(&mut self) {
        self.m_help_page = None;
    }

    /// `DockingDialog::openClosePressed`.
    pub fn open_close_pressed(&mut self, native: &mut dyn DockingDialogNativeBoundary) {
        self.set_dialog_state(!self.m_opened, native)
    }

    /// `DockingDialog::setDialogState`.
    pub fn set_dialog_state(&mut self, state: bool, native: &mut dyn DockingDialogNativeBoundary) {
        if state {
            native.set_open_close_icon_and_tool_tip(
                "close",
                "Close up the control section in this dialog",
            );
            native.show_stack(true);
        } else {
            native.set_open_close_icon_and_tool_tip("open", "Show the controls in this dialog");
            native.show_stack(false);
        }
        native.dia_show_widget(state);
        self.m_opened = state;
        if state && self.m_hinted_width != 0 {
            native.process_events();
            native.resize(self.m_hinted_width, self.m_hinted_height);
            native.process_events();
        } else {
            native.adjust_size();
        }
        if !state && self.m_closed_height == 0 {
            native.process_events();
            self.m_closed_height = native.frame_height();
        }
    }

    /// `DockingDialog::helpPressed`.
    pub fn help_pressed(&self, native: &mut dyn DockingDialogNativeBoundary) {
        if let Some(page) = &self.m_help_page {
            native.show_help_page(page);
        }
    }

    /// `DockingDialog::addWidgetToStack`.
    pub fn add_widget_to_stack(
        &mut self,
        dialog: usize,
        mut initial_state: bool,
        manager: &mut DialogManager,
        native: &mut dyn DockingDialogNativeBoundary,
    ) {
        if let Some((left, _top, right, bottom)) = native.dialog_layout_margins(dialog) {
            native.set_dialog_layout_margins(dialog, left, 0, right, (bottom + 1) / 2);
            native.set_dialog_layout_spacing(dialog, 3);
        }
        let left = manager.next_docker_state;
        if left >= 0 {
            initial_state = left > 0;
        }
        self.m_dialog = Some(dialog);
        native.stack_add_widget(dialog);
        self.set_dialog_state(true, native);
        native.process_events();
        native.set_minimum_width(native.widget_width());
        self.m_frame_width = native.frame_width();
        self.m_open_height = native.frame_height();
        if !initial_state {
            self.set_dialog_state(false, native);
        }
        native.initialize_screen_change(self.widget, self.m_cur_dev_pix_ratio);
    }

    /// `DockingDialog::resizeToHintWidthAfterShow`.
    pub fn resize_to_hint_width_after_show(
        &mut self,
        native: &mut dyn DockingDialogNativeBoundary,
    ) {
        native.process_events();
        let hwidth = native.size_hint_width();
        if native.minimum_width() > hwidth {
            native.set_minimum_width(hwidth);
        }
        self.m_hinted_width = native.widget_width().min(hwidth);
        self.m_hinted_height = native.size_hint_height().min(native.widget_height());
        native.resize(self.m_hinted_width, self.m_hinted_height);
        native.process_events();
        self.m_frame_width = native.frame_width();
    }

    /// `DockingDialog::getDialogKeyAndState`.
    pub fn get_dialog_key_and_state(&self) -> (char, bool) {
        (self.m_key_letter, self.m_opened)
    }

    /// `DockingDialog::getScreenChanged`.
    pub fn get_screen_changed(&self) -> bool {
        self.m_screen_changed
    }

    /// `DockingDialog::resetScreenChanged`.
    pub fn reset_screen_changed(&mut self) {
        self.m_screen_changed = false;
    }

    /// `DockingDialog::getCurDevPixRatio`.
    pub fn get_cur_dev_pix_ratio(&self) -> f32 {
        self.m_cur_dev_pix_ratio
    }

    /// `DockingDialog::setCurDevPixRatio`.
    pub fn set_cur_dev_pix_ratio(&mut self, new_dpr: f32) {
        self.m_cur_dev_pix_ratio = new_dpr;
    }

    /// `DockingDialog::closeEvent`.
    pub fn close_event(&self, native: &mut dyn DockingDialogNativeBoundary) {
        native.docker_close_event();
    }

    /// `DockingDialog::changeEvent`.
    pub fn change_event(
        &mut self,
        event: DockingEventType,
        native: &mut dyn DockingDialogNativeBoundary,
    ) {
        native.docker_change_event();
        if event == DockingEventType::FontChange {
            native.adjust_size();
        }
    }

    /// `DockingDialog::event`.
    pub fn event(
        &mut self,
        event: DockingEventType,
        window_state: DockingWindowState,
        active_window: bool,
        manager: &mut DialogManager,
        dialog_native: &mut dyn DockingDialogNativeBoundary,
        manager_native: &mut dyn DialogManagerNativeBoundary,
    ) -> bool {
        let minimized = event == DockingEventType::WindowStateChange && window_state.minimized;
        let normal = event == DockingEventType::WindowStateChange
            && (window_state.no_state || window_state.maximized);
        if event == DockingEventType::ActivationChange && active_window {
            manager.docker_was_activated(Some(self.widget), manager_native);
        }
        if (minimized || event == DockingEventType::Hide) && !self.m_minimized {
            self.m_minimized = true;
            manager.docker_was_hidden(self.widget, manager_native);
        } else if (normal || event == DockingEventType::Show) && self.m_minimized {
            self.m_minimized = false;
            manager.docker_was_unhidden(self.widget, manager_native);
        }
        dialog_native.widget_event(event)
    }

    /// `DockingDialog::resizeEvent`.
    pub fn resize_event(
        &self,
        manager: &mut DialogManager,
        manager_native: &mut dyn DialogManagerNativeBoundary,
    ) {
        manager.docker_has_resized(self.widget, manager_native);
    }

    /// `DockingDialog::moveEvent`.
    pub fn move_event(
        &self,
        manager: &mut DialogManager,
        manager_native: &mut dyn DialogManagerNativeBoundary,
    ) {
        manager.docker_has_moved(self.widget, manager_native);
    }

    /// `DockingDialog::screenChanged` (`WATCH_DPI_CHANGE`).
    pub fn screen_changed(
        &mut self,
        manager: &mut DialogManager,
        manager_native: &mut dyn DialogManagerNativeBoundary,
    ) {
        self.m_screen_changed = true;
        manager.docker_changed_screen(self.widget, manager_native);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::three_dmod::control::{
        DOCKING_DIALOG_TYPE, DialogManagerNativeBoundary, IMOD_DIALOG,
    };

    #[derive(Default)]
    struct Native {
        width: i32,
        height: i32,
        hint_width: i32,
        hint_height: i32,
        min_width: i32,
        frame_width: i32,
        frame_height: i32,
        state: Vec<bool>,
        help: String,
        timers: Vec<(i32, i32)>,
    }
    impl DockingDialogNativeBoundary for Native {
        fn show_stack(&mut self, show: bool) {
            self.state.push(show);
        }
        fn dia_show_widget(&mut self, _show: bool) {}
        fn frame_height(&self) -> i32 {
            self.frame_height
        }
        fn frame_width(&self) -> i32 {
            self.frame_width
        }
        fn widget_width(&self) -> i32 {
            self.width
        }
        fn widget_height(&self) -> i32 {
            self.height
        }
        fn size_hint_width(&self) -> i32 {
            self.hint_width
        }
        fn size_hint_height(&self) -> i32 {
            self.hint_height
        }
        fn minimum_width(&self) -> i32 {
            self.min_width
        }
        fn set_minimum_width(&mut self, width: i32) {
            self.min_width = width;
        }
        fn show_help_page(&mut self, page: &str) {
            self.help = page.into();
        }
    }
    impl DialogManagerNativeBoundary for Native {
        fn start_dock_timer(&mut self, class: i32, timeout: i32) {
            self.timers.push((class, timeout));
        }
    }

    #[derive(Default)]
    struct ManagerNative {
        timers: Vec<(i32, i32)>,
    }
    impl DialogManagerNativeBoundary for ManagerNative {
        fn start_dock_timer(&mut self, class: i32, timeout: i32) {
            self.timers.push((class, timeout));
        }
    }

    #[test]
    fn open_close_and_hint_follow_source_state() {
        let mut n = Native {
            width: 120,
            height: 80,
            hint_width: 100,
            hint_height: 60,
            frame_width: 124,
            frame_height: 84,
            ..Default::default()
        };
        let mut d = DockingDialog::new(9, "Title", Some("help.html"), 's', &mut n);
        d.set_dialog_state(true, &mut n);
        d.resize_to_hint_width_after_show(&mut n);
        d.open_close_pressed(&mut n);
        d.help_pressed(&mut n);
        assert_eq!(d.get_dialog_key_and_state(), ('s', false));
        assert_eq!(d.m_hinted_width, 100);
        assert_eq!(d.m_closed_height, 84);
        assert_eq!(n.help, "help.html");
    }

    #[test]
    fn widget_events_call_manager_docker_paths() {
        let mut n = Native::default();
        let mut manager_native = ManagerNative::default();
        let mut manager = DialogManager::new(IMOD_DIALOG);
        manager.add(9, IMOD_DIALOG, DOCKING_DIALOG_TYPE, 0, &mut manager_native);
        let mut d = DockingDialog::new(9, "Title", None, 'x', &mut n);
        d.event(
            DockingEventType::Hide,
            DockingWindowState::default(),
            false,
            &mut manager,
            &mut n,
            &mut manager_native,
        );
        d.event(
            DockingEventType::Show,
            DockingWindowState::default(),
            false,
            &mut manager,
            &mut n,
            &mut manager_native,
        );
        d.screen_changed(&mut manager, &mut manager_native);
        assert!(!manager_native.timers.is_empty());
        assert!(d.get_screen_changed());
    }
}
