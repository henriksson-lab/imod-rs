//! Translation of `IMOD/3dmod/formv_depthcue.cpp` together with
//! `formv_depthcue.h`.
//!
//! The upstream class is a Qt Designer form.  Widget construction, signal
//! connection, native close handling, keyboard grabbing, and presentation
//! belong to the active native GUI backend.  They are deliberately exposed as
//! `DepthcueNativeBoundary` calls, rather than replaced by a synthetic dialog.
#![allow(dead_code)]

use crate::imod::three_dmod::mv_window::{Key, KeyEvent};

pub const DEPTHCUE_MIN: i32 = 0;
pub const DEPTHCUE_MAX: i32 = 100;

/// `QEvent` kinds inspected by `imodvDepthcueForm::changeEvent`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DepthcueChangeEvent {
    FontChange,
    Other,
}

/// Direct native-widget and paired-unit operations used by this source unit.
///
/// A GUI implementation must bind these calls to its actual widget/event
/// system.  No in-memory dialog is used as a rendering substitute.
pub trait DepthcueNativeBoundary {
    fn setup_ui(&mut self);
    fn set_delete_on_close(&mut self);
    fn set_always_show_tool_tips(&mut self);
    fn connect_depthcue_signals(&mut self);
    fn set_start_slider_range(&mut self, minimum_width: i32, minimum: i32, maximum: i32);
    fn set_end_slider_range(&mut self, minimum_width: i32, minimum: i32, maximum: i32);
    fn set_start_label(&mut self, text: &str);
    fn set_end_label(&mut self, text: &str);
    fn set_depthcue_enabled(&mut self, enabled: bool);
    fn set_start_slider(&mut self, value: i32);
    fn set_end_slider(&mut self, value: i32);
    fn close_native_dialog(&mut self);
    fn accept_close_event(&mut self);
    fn grab_keyboard(&mut self);
    fn release_keyboard(&mut self);
    fn retranslate_ui(&mut self);
    fn check_and_set_mac_menu(&mut self);
    fn widget_change_event(&mut self);
    fn hot_slider_active(&self, control_pressed: bool) -> bool;
    fn hot_slider_enabled(&self) -> bool;
    fn hot_slider_key(&self, key: Key) -> bool;
    fn close_key(&self, event: KeyEvent) -> bool;
    fn imodv_depthcue_toggle(&mut self, state: i32);
    fn imodv_depthcue_start_end(&mut self, value: i32, end: bool, dragging: bool);
    fn imodv_depthcue_done(&mut self);
    fn imodv_depthcue_help(&mut self);
    fn imodv_depthcue_closing(&mut self);
    fn imodv_key_press(&mut self, event: KeyEvent);
    fn imodv_key_release(&mut self, event: KeyEvent);
}

/// `imodvDepthcueForm` (`formv_depthcue.h`), including source widget state.
#[derive(Clone, Debug)]
pub struct ImodvDepthcueForm {
    pub m_end_pressed: bool,
    pub m_str: String,
    pub m_start_pressed: bool,
    pub m_start_displayed: i32,
    pub m_end_displayed: i32,
    pub m_ctrl_pressed: bool,
}

impl ImodvDepthcueForm {
    /// `imodvDepthcueForm()` source constructor.
    pub fn new(native: &mut dyn DepthcueNativeBoundary) -> Self {
        native.setup_ui();
        let mut form = Self {
            m_end_pressed: false,
            m_str: String::new(),
            m_start_pressed: false,
            m_start_displayed: 0,
            m_end_displayed: 0,
            m_ctrl_pressed: false,
        };
        form.init(native);
        form
    }

    /// `imodvDepthcueForm::~imodvDepthcueForm`.
    pub fn destroy(&mut self) {}

    /// `imodvDepthcueForm::languageChange`.
    pub fn language_change(&mut self, native: &mut dyn DepthcueNativeBoundary) {
        native.retranslate_ui();
    }

    /// `imodvDepthcueForm::init`.
    pub fn init(&mut self, native: &mut dyn DepthcueNativeBoundary) {
        native.set_delete_on_close();
        native.set_always_show_tool_tips();
        self.m_ctrl_pressed = false;
        self.m_start_pressed = false;
        self.m_end_pressed = false;
        native.connect_depthcue_signals();
        let width = DEPTHCUE_MAX - DEPTHCUE_MIN + 16;
        native.set_start_slider_range(width, DEPTHCUE_MIN, DEPTHCUE_MAX);
        native.set_end_slider_range(width, DEPTHCUE_MIN, DEPTHCUE_MAX);
    }

    /// `imodvDepthcueForm::depthcueToggled`.
    pub fn depthcue_toggled(&mut self, state: bool, native: &mut dyn DepthcueNativeBoundary) {
        native.imodv_depthcue_toggle(state as i32);
    }

    /// `imodvDepthcueForm::displayStartLabel`.
    pub fn display_start_label(&mut self, value: i32, native: &mut dyn DepthcueNativeBoundary) {
        self.m_str = value.to_string();
        native.set_start_label(&self.m_str);
        self.m_start_displayed = value;
    }

    /// `imodvDepthcueForm::startChanged`.
    pub fn start_changed(&mut self, value: i32, native: &mut dyn DepthcueNativeBoundary) {
        if !self.m_start_pressed || native.hot_slider_active(self.m_ctrl_pressed) {
            native.imodv_depthcue_start_end(value, false, self.m_start_pressed);
        }
    }

    /// `imodvDepthcueForm::startPressed`.
    pub fn start_pressed(&mut self) {
        self.m_start_pressed = true;
    }

    /// `imodvDepthcueForm::startReleased`.
    pub fn start_released(&mut self, native: &mut dyn DepthcueNativeBoundary) {
        self.m_start_pressed = false;
        self.start_changed(self.m_start_displayed, native);
    }

    /// `imodvDepthcueForm::displayEndLabel`.
    pub fn display_end_label(&mut self, value: i32, native: &mut dyn DepthcueNativeBoundary) {
        self.m_str = value.to_string();
        native.set_end_label(&self.m_str);
        self.m_end_displayed = value;
    }

    /// `imodvDepthcueForm::endChanged`.
    pub fn end_changed(&mut self, value: i32, native: &mut dyn DepthcueNativeBoundary) {
        if !self.m_end_pressed || native.hot_slider_active(self.m_ctrl_pressed) {
            native.imodv_depthcue_start_end(value, true, self.m_end_pressed);
        }
    }

    /// `imodvDepthcueForm::endPressed`.
    pub fn end_pressed(&mut self) {
        self.m_end_pressed = true;
    }

    /// `imodvDepthcueForm::endReleased`.
    pub fn end_released(&mut self, native: &mut dyn DepthcueNativeBoundary) {
        self.m_end_pressed = false;
        self.end_changed(self.m_end_displayed, native);
    }

    /// `imodvDepthcueForm::donePressed`.
    pub fn done_pressed(&mut self, native: &mut dyn DepthcueNativeBoundary) {
        native.imodv_depthcue_done();
    }

    /// `imodvDepthcueForm::helpPressed`.
    pub fn help_pressed(&mut self, native: &mut dyn DepthcueNativeBoundary) {
        native.imodv_depthcue_help();
    }

    /// `imodvDepthcueForm::setStates`.
    pub fn set_states(
        &mut self,
        enabled: i32,
        start: i32,
        end: i32,
        native: &mut dyn DepthcueNativeBoundary,
    ) {
        native.set_depthcue_enabled(enabled != 0);
        self.display_start_label(start, native);
        native.set_start_slider(start);
        self.display_end_label(end, native);
        native.set_end_slider(end);
    }

    /// `imodvDepthcueForm::closeEvent`.
    pub fn close_event(&mut self, native: &mut dyn DepthcueNativeBoundary) {
        native.imodv_depthcue_closing();
        native.accept_close_event();
    }

    /// `imodvDepthcueForm::keyPressEvent`.
    pub fn key_press_event(&mut self, event: KeyEvent, native: &mut dyn DepthcueNativeBoundary) {
        if native.close_key(event) {
            native.imodv_depthcue_done();
        } else {
            if native.hot_slider_enabled() && native.hot_slider_key(event.key) {
                self.m_ctrl_pressed = true;
                native.grab_keyboard();
            }
            native.imodv_key_press(event);
        }
    }

    /// `imodvDepthcueForm::keyReleaseEvent`.
    pub fn key_release_event(&mut self, event: KeyEvent, native: &mut dyn DepthcueNativeBoundary) {
        if native.hot_slider_key(event.key) {
            self.m_ctrl_pressed = false;
            native.release_keyboard();
        }
        native.imodv_key_release(event);
    }

    /// `imodvDepthcueForm::changeEvent`.
    pub fn change_event(
        &mut self,
        event: DepthcueChangeEvent,
        native: &mut dyn DepthcueNativeBoundary,
    ) {
        native.widget_change_event();
        native.check_and_set_mac_menu();
        if event != DepthcueChangeEvent::FontChange {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Native {
        calls: Vec<String>,
        hot: bool,
    }
    impl DepthcueNativeBoundary for Native {
        fn setup_ui(&mut self) {
            self.calls.push("setup".into())
        }
        fn set_delete_on_close(&mut self) {
            self.calls.push("delete".into())
        }
        fn set_always_show_tool_tips(&mut self) {
            self.calls.push("tips".into())
        }
        fn connect_depthcue_signals(&mut self) {
            self.calls.push("connect".into())
        }
        fn set_start_slider_range(&mut self, width: i32, minimum: i32, maximum: i32) {
            self.calls
                .push(format!("start-range:{width}:{minimum}:{maximum}"))
        }
        fn set_end_slider_range(&mut self, width: i32, minimum: i32, maximum: i32) {
            self.calls
                .push(format!("end-range:{width}:{minimum}:{maximum}"))
        }
        fn set_start_label(&mut self, text: &str) {
            self.calls.push(format!("start-label:{text}"))
        }
        fn set_end_label(&mut self, text: &str) {
            self.calls.push(format!("end-label:{text}"))
        }
        fn set_depthcue_enabled(&mut self, enabled: bool) {
            self.calls.push(format!("enabled:{enabled}"))
        }
        fn set_start_slider(&mut self, value: i32) {
            self.calls.push(format!("start-slider:{value}"))
        }
        fn set_end_slider(&mut self, value: i32) {
            self.calls.push(format!("end-slider:{value}"))
        }
        fn close_native_dialog(&mut self) {
            self.calls.push("close".into())
        }
        fn accept_close_event(&mut self) {
            self.calls.push("accept".into())
        }
        fn grab_keyboard(&mut self) {
            self.calls.push("grab".into())
        }
        fn release_keyboard(&mut self) {
            self.calls.push("release".into())
        }
        fn retranslate_ui(&mut self) {
            self.calls.push("retranslate".into())
        }
        fn check_and_set_mac_menu(&mut self) {
            self.calls.push("mac-menu".into())
        }
        fn widget_change_event(&mut self) {
            self.calls.push("change".into())
        }
        fn hot_slider_active(&self, _: bool) -> bool {
            self.hot
        }
        fn hot_slider_enabled(&self) -> bool {
            true
        }
        fn hot_slider_key(&self, key: Key) -> bool {
            key == Key::Character('C')
        }
        fn close_key(&self, event: KeyEvent) -> bool {
            event.key == Key::Character('Q')
        }
        fn imodv_depthcue_toggle(&mut self, state: i32) {
            self.calls.push(format!("toggle:{state}"))
        }
        fn imodv_depthcue_start_end(&mut self, value: i32, end: bool, dragging: bool) {
            self.calls.push(format!("set:{value}:{end}:{dragging}"))
        }
        fn imodv_depthcue_done(&mut self) {
            self.calls.push("done".into())
        }
        fn imodv_depthcue_help(&mut self) {
            self.calls.push("help".into())
        }
        fn imodv_depthcue_closing(&mut self) {
            self.calls.push("closing".into())
        }
        fn imodv_key_press(&mut self, event: KeyEvent) {
            self.calls.push(format!("key-down:{:?}", event.key))
        }
        fn imodv_key_release(&mut self, event: KeyEvent) {
            self.calls.push(format!("key-up:{:?}", event.key))
        }
    }

    #[test]
    fn source_slider_drag_defers_until_release_without_hot_slider() {
        let mut native = Native::default();
        let mut form = ImodvDepthcueForm::new(&mut native);
        form.start_pressed();
        form.start_changed(25, &mut native);
        assert!(!native.calls.iter().any(|call| call == "set:25:false:true"));
        form.display_start_label(25, &mut native);
        form.start_released(&mut native);
        assert!(native.calls.iter().any(|call| call == "set:25:false:false"));
    }

    #[test]
    fn source_hot_slider_updates_during_drag_and_grabs_keyboard() {
        let mut native = Native {
            hot: true,
            ..Default::default()
        };
        let mut form = ImodvDepthcueForm::new(&mut native);
        form.start_pressed();
        form.key_press_event(
            KeyEvent {
                key: Key::Character('C'),
                ..KeyEvent::default()
            },
            &mut native,
        );
        form.start_changed(42, &mut native);
        assert!(native.calls.iter().any(|call| call == "grab"));
        assert!(native.calls.iter().any(|call| call == "set:42:false:true"));
    }

    #[test]
    fn set_states_keeps_upstream_widget_order() {
        let mut native = Native::default();
        let mut form = ImodvDepthcueForm::new(&mut native);
        form.set_states(1, 10, 90, &mut native);
        let tail = &native.calls[native.calls.len() - 5..];
        assert_eq!(
            tail,
            [
                "enabled:true",
                "start-label:10",
                "start-slider:10",
                "end-label:90",
                "end-slider:90"
            ]
        );
    }

    #[test]
    fn change_event_calls_base_before_mac_menu() {
        let mut native = Native::default();
        let mut form = ImodvDepthcueForm::new(&mut native);
        form.change_event(DepthcueChangeEvent::Other, &mut native);
        assert!(
            native
                .calls
                .windows(2)
                .any(|calls| calls == ["change", "mac-menu"])
        );
    }
}
