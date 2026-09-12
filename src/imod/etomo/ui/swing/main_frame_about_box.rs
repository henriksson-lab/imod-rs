//! `IMOD/Etomo/src/etomo/ui/swing/MainFrame_AboutBox.java`.
//!
//! Swing `JDialog`/`JPanel`/`JLabel` construction, packing, disposal, and native
//! event delivery remain GUI boundaries.  `VersionControl` is not yet a concrete
//! Rust source unit, so its three source calls are represented by the exact
//! boundary below; this unit retains the source-owned layout and event behavior.
#![allow(dead_code)]

use crate::imod::etomo::r#type::{axis_id::AxisID, imod_version::CURRENT_VERSION};

use super::fixed_dim::FixedDim;

/// Direct static `VersionControl` calls made by `MainFrame_AboutBox`.
pub trait MainFrameAboutBoxVersionControl {
    /// Java static `VersionControl.TIME_STAMP`.
    fn time_stamp(&self) -> &str;

    /// Java static `VersionControl.getImodInfo(AxisID)`.
    fn get_imod_info(&self, axis_id: Option<AxisID>) -> Option<Vec<String>>;

    /// Java static `VersionControl.getPeetVersion()`.
    fn get_peet_version(&self) -> Option<String>;
}

/// Java `WindowEvent` IDs used by this source unit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AboutWindowEventId {
    /// Java `WindowEvent.WINDOW_CLOSING`.
    WindowClosing,
    /// Any event for which Java's equality check is false.
    Other,
}

/// Direct native `WindowEvent` boundary for `processWindowEvent`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AboutWindowEvent {
    pub id: AboutWindowEventId,
}

impl AboutWindowEvent {
    /// Java `WindowEvent` with the source-used ID.
    pub fn new(id: AboutWindowEventId) -> Self {
        Self { id }
    }
}

/// Java `ActionEvent.getSource()` values read by `buttonAction`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AboutActionEventSource {
    /// Java field `btnOK`.
    OkButton,
    /// A source object other than `btnOK`.
    Other,
}

/// Direct native `ActionEvent` boundary for `buttonAction`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AboutActionEvent {
    pub source: AboutActionEventSource,
}

impl AboutActionEvent {
    /// Java `ActionEvent` with the source-used component identity.
    pub fn new(source: AboutActionEventSource) -> Self {
        Self { source }
    }
}

/// Source-visible `JPanel`/`JLabel`/`Box` hierarchy sent to the Swing boundary.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MainFrameAboutBoxLayout {
    pub root_layout: Option<&'static str>,
    pub about_layout: Option<&'static str>,
    pub text_layout: Option<&'static str>,
    pub button_layout: Option<&'static str>,
    pub about_component_order: Vec<String>,
    pub text_component_order: Vec<String>,
    pub button_component_order: Vec<String>,
    pub root_component_order: Vec<String>,
}

/// Java `MainFrame_AboutBox` fields and inherited `JDialog` presentation state.
pub struct MainFrameAboutBox {
    /// Java `rcsid`.
    pub rcsid: &'static str,
    /// Java `pnlAbout`.
    pub pnl_about: MainFrameAboutBoxLayout,
    /// Java `btnOK`.
    pub btn_ok_text: String,
    pub btn_ok_action_listener_present: bool,
    pub title: String,
    pub resizable: bool,
    pub packed: bool,
    pub disposed: bool,
    pub super_process_window_event_count: usize,
    pub last_super_window_event: Option<AboutWindowEventId>,
}

impl MainFrameAboutBox {
    /// Java `MainFrame_AboutBox(Frame, AxisID)`.
    pub fn new<V: MainFrameAboutBoxVersionControl>(
        _parent: Option<()>,
        axis_id: Option<AxisID>,
        version_control: &V,
    ) -> Self {
        let mut pnl_about = MainFrameAboutBoxLayout {
            root_layout: Some("BorderLayout"),
            about_layout: Some("BoxLayout.Y_AXIS"),
            text_layout: Some("BoxLayout.Y_AXIS"),
            button_layout: None,
            ..Default::default()
        };
        let btn_ok_text = "OK".to_string();
        pnl_about
            .about_component_order
            .push(format!("Box.createRigidArea({:?})", FixedDim::x0_y10));
        pnl_about
            .text_component_order
            .push("JLabel(Etomo: The IMOD Tomography GUI)".to_string());
        pnl_about
            .text_component_order
            .push(format!("Box.createRigidArea({:?})", FixedDim::x0_y5));
        pnl_about.text_component_order.push(format!(
            "JLabel(Version {CURRENT_VERSION} {})",
            version_control.time_stamp()
        ));
        let imod_info = version_control.get_imod_info(axis_id);
        if let Some(imod_info) = imod_info.as_ref() {
            if imod_info.len() > 2 {
                pnl_about
                    .text_component_order
                    .push(format!("Box.createRigidArea({:?})", FixedDim::x0_y10));
                pnl_about
                    .text_component_order
                    .push(format!("JLabel({})", imod_info[1]));
                pnl_about
                    .text_component_order
                    .push(format!("Box.createRigidArea({:?})", FixedDim::x0_y5));
                pnl_about
                    .text_component_order
                    .push(format!("JLabel({})", imod_info[2]));
            }
        }
        pnl_about
            .text_component_order
            .push(format!("Box.createRigidArea({:?})", FixedDim::x0_y10));
        pnl_about
            .text_component_order
            .push("JLabel(Written by: Rick Gaudette & Sue Held)".to_string());
        if let Some(imod_info) = imod_info.as_ref() {
            if !imod_info.is_empty() {
                pnl_about
                    .text_component_order
                    .push(format!("Box.createRigidArea({:?})", FixedDim::x0_y20));
                pnl_about
                    .text_component_order
                    .push(format!("JLabel(IMOD Version: {})", imod_info[0]));
            }
        }
        let version = version_control.get_peet_version();
        if version.is_some() {
            pnl_about
                .text_component_order
                .push(format!("Box.createRigidArea({:?})", FixedDim::x0_y5));
            let peet_version = version_control.get_peet_version();
            pnl_about.text_component_order.push(format!(
                "JLabel(PEET Version: {})",
                peet_version.as_deref().unwrap_or("null")
            ));
        }
        pnl_about
            .text_component_order
            .push(format!("Box.createRigidArea({:?})", FixedDim::x0_y10));
        pnl_about.button_component_order.push("btnOK".to_string());
        pnl_about.about_component_order.push("pnlText".to_string());
        pnl_about
            .about_component_order
            .push("pnlButton".to_string());
        pnl_about.root_component_order = vec![
            "pnlAbout:BorderLayout.CENTER".to_string(),
            format!(
                "Box.createRigidArea({:?}):BorderLayout.WEST",
                FixedDim::x20_y0
            ),
            format!(
                "Box.createRigidArea({:?}):BorderLayout.EAST",
                FixedDim::x20_y0
            ),
        ];
        Self {
            rcsid: "$Id$",
            pnl_about,
            btn_ok_text,
            btn_ok_action_listener_present: true,
            title: "About".to_string(),
            resizable: false,
            packed: true,
            disposed: false,
            super_process_window_event_count: 0,
            last_super_window_event: None,
        }
    }

    /// Java override `processWindowEvent(WindowEvent)`.
    pub fn process_window_event(&mut self, event: AboutWindowEvent) {
        if event.id == AboutWindowEventId::WindowClosing {
            self.cancel();
        }
        self.super_process_window_event_count += 1;
        self.last_super_window_event = Some(event.id);
    }

    /// Java `cancel()`.
    pub fn cancel(&mut self) {
        self.disposed = true;
    }

    /// Java `buttonAction(ActionEvent)`.
    pub fn button_action(&mut self, event: AboutActionEvent) {
        if event.source == AboutActionEventSource::OkButton {
            self.cancel();
        }
    }
}

/// Java non-static inner `AboutActionListener`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AboutActionListener;

impl AboutActionListener {
    /// Java `AboutActionListener(MainFrame_AboutBox)`.
    pub fn new() -> Self {
        Self
    }

    /// Java `actionPerformed(ActionEvent)`.
    pub fn action_performed(&self, adaptee: &mut MainFrameAboutBox, event: AboutActionEvent) {
        adaptee.button_action(event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct VersionControl {
        imod_info: Option<Vec<String>>,
        peet_versions: std::cell::RefCell<Vec<Option<String>>>,
    }

    impl MainFrameAboutBoxVersionControl for VersionControl {
        fn time_stamp(&self) -> &str {
            "8/14/2026 16:32"
        }

        fn get_imod_info(&self, _axis_id: Option<AxisID>) -> Option<Vec<String>> {
            self.imod_info.clone()
        }

        fn get_peet_version(&self) -> Option<String> {
            self.peet_versions.borrow_mut().remove(0)
        }
    }

    #[test]
    fn constructor_preserves_source_label_and_component_order() {
        let version_control = VersionControl {
            imod_info: Some(vec![
                "4.13.1".into(),
                "revision line".into(),
                "date line".into(),
            ]),
            peet_versions: std::cell::RefCell::new(vec![Some("1.0".into()), Some("1.0".into())]),
        };
        let dialog = MainFrameAboutBox::new(None, Some(AxisID::First), &version_control);
        assert_eq!(dialog.title, "About");
        assert!(!dialog.resizable);
        assert!(dialog.packed);
        assert_eq!(dialog.btn_ok_text, "OK");
        assert!(dialog.btn_ok_action_listener_present);
        assert_eq!(
            dialog.pnl_about.text_component_order,
            vec![
                "JLabel(Etomo: The IMOD Tomography GUI)",
                "Box.createRigidArea(Dimension { width: 0, height: 5 })",
                "JLabel(Version 5.2.17 8/14/2026 16:32)",
                "Box.createRigidArea(Dimension { width: 0, height: 10 })",
                "JLabel(revision line)",
                "Box.createRigidArea(Dimension { width: 0, height: 5 })",
                "JLabel(date line)",
                "Box.createRigidArea(Dimension { width: 0, height: 10 })",
                "JLabel(Written by: Rick Gaudette & Sue Held)",
                "Box.createRigidArea(Dimension { width: 0, height: 20 })",
                "JLabel(IMOD Version: 4.13.1)",
                "Box.createRigidArea(Dimension { width: 0, height: 5 })",
                "JLabel(PEET Version: 1.0)",
                "Box.createRigidArea(Dimension { width: 0, height: 10 })",
            ]
        );
        assert_eq!(
            dialog.pnl_about.root_component_order,
            vec![
                "pnlAbout:BorderLayout.CENTER",
                "Box.createRigidArea(Dimension { width: 20, height: 0 }):BorderLayout.WEST",
                "Box.createRigidArea(Dimension { width: 20, height: 0 }):BorderLayout.EAST",
            ]
        );
    }

    #[test]
    fn source_close_and_ok_event_cancel_but_other_events_do_not() {
        let version_control = VersionControl {
            imod_info: None,
            peet_versions: std::cell::RefCell::new(vec![None]),
        };
        let mut dialog = MainFrameAboutBox::new(None, None, &version_control);
        dialog.process_window_event(AboutWindowEvent::new(AboutWindowEventId::Other));
        assert!(!dialog.disposed);
        assert_eq!(dialog.super_process_window_event_count, 1);
        dialog.button_action(AboutActionEvent::new(AboutActionEventSource::Other));
        assert!(!dialog.disposed);
        AboutActionListener::new().action_performed(
            &mut dialog,
            AboutActionEvent::new(AboutActionEventSource::OkButton),
        );
        assert!(dialog.disposed);
        dialog.process_window_event(AboutWindowEvent::new(AboutWindowEventId::WindowClosing));
        assert_eq!(dialog.super_process_window_event_count, 2);
        assert_eq!(
            dialog.last_super_window_event,
            Some(AboutWindowEventId::WindowClosing)
        );
    }

    #[test]
    fn source_uses_a_second_peet_version_call_for_the_label() {
        let version_control = VersionControl {
            imod_info: None,
            peet_versions: std::cell::RefCell::new(vec![Some("1.0".into()), None]),
        };
        let dialog = MainFrameAboutBox::new(None, None, &version_control);
        assert!(
            dialog
                .pnl_about
                .text_component_order
                .contains(&"JLabel(PEET Version: null)".to_string())
        );
    }
}
