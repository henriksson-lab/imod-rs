//! `IMOD/Etomo/src/etomo/ui/swing/HTMLPageWindow.java`.
//!
//! Swing's `JFrame`/`JEditorPane` and the URL page loader are direct GUI and
//! browser boundaries.  This source unit retains the frame, editor, scroll
//! pane, title, and page-selection state that Java owns around those calls.
#![allow(dead_code)]

/// Java `WindowConstants.DISPOSE_ON_CLOSE` selected by the constructor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HtmlPageWindowCloseOperation {
    DisposeOnClose,
}

/// Java's `HTMLEditorKit` field selection on the `JEditorPane`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HtmlPageWindowEditorKit {
    Html,
}

/// The source-used state of Java's `JEditorPane`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HtmlPageWindowEditorPane {
    pub editor_kit: HtmlPageWindowEditorKit,
    pub page: Option<String>,
    pub editable: bool,
    pub hyperlink_listener_count: usize,
}

/// Java's `JScrollPane(editorPane)` relationship.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HtmlPageWindowScrollPane {
    pub view_is_editor_pane: bool,
}

/// The source-used child relationship in Java's `mainPanel` container.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct HtmlPageWindowMainPanel {
    pub scroll_pane_in_border_layout_center: bool,
}

/// Java `HyperlinkEvent.EventType` values relevant to the listener boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HtmlPageWindowHyperlinkEventType {
    Activated,
    Entered,
    Exited,
}

/// Java's already-parsed `URL` returned by `HyperlinkEvent.getURL()`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HtmlPageWindowUrl {
    pub value: String,
    pub path: String,
}

/// Java's `HyperlinkEvent` values read by `hyperlinkUpdate`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HtmlPageWindowHyperlinkEvent {
    pub event_type: HtmlPageWindowHyperlinkEventType,
    pub url: Option<HtmlPageWindowUrl>,
}

/// The `JEditorPane.setPage` browser/network operation called by this source
/// unit.  A native GUI backend supplies it; this translation does not invent
/// a browser implementation.
pub trait HtmlPageWindowBrowser {
    fn set_page(&mut self, url: &str) -> Result<(), String>;
}

/// Java package-private `HTMLPageWindow`, including its inherited JFrame
/// state touched by this source unit.
pub struct HtmlPageWindow {
    pub rcsid: &'static str,
    pub main_panel: HtmlPageWindowMainPanel,
    pub url: Option<String>,
    pub editor_pane: HtmlPageWindowEditorPane,
    pub scroll_pane: HtmlPageWindowScrollPane,
    pub size: (i32, i32),
    pub title: String,
    pub default_close_operation: HtmlPageWindowCloseOperation,
}

impl HtmlPageWindow {
    /// `HTMLPageWindow()`.
    pub fn new() -> Self {
        Self {
            rcsid: "$Id$",
            main_panel: HtmlPageWindowMainPanel {
                scroll_pane_in_border_layout_center: true,
            },
            url: None,
            editor_pane: HtmlPageWindowEditorPane {
                editor_kit: HtmlPageWindowEditorKit::Html,
                page: None,
                editable: true,
                hyperlink_listener_count: 1,
            },
            scroll_pane: HtmlPageWindowScrollPane {
                view_is_editor_pane: true,
            },
            size: (625, 800),
            title: String::new(),
            default_close_operation: HtmlPageWindowCloseOperation::DisposeOnClose,
        }
    }

    /// `openURL(String)`.
    pub fn open_url<B: HtmlPageWindowBrowser>(&mut self, new_url: String, browser: &mut B) {
        self.url = Some(new_url.clone());
        match browser.set_page(&new_url) {
            Ok(()) => {
                self.editor_pane.page = Some(new_url.clone());
                self.title = new_url;
                self.editor_pane.editable = false;
            }
            Err(except) => {
                eprintln!("Cannot open URL:");
                eprintln!("{new_url}");
                eprintln!("{except}");
            }
        }
    }

    /// `hyperlinkUpdate(HyperlinkEvent)`.
    pub fn hyperlink_update<B: HtmlPageWindowBrowser>(
        &mut self,
        event: &HtmlPageWindowHyperlinkEvent,
        browser: &mut B,
    ) {
        if event.event_type == HtmlPageWindowHyperlinkEventType::Activated {
            match event.url.as_ref() {
                Some(url) => match browser.set_page(&url.value) {
                    Ok(()) => {
                        self.editor_pane.page = Some(url.value.clone());
                        self.title = url.path.clone();
                    }
                    Err(except) => eprintln!("{except}"),
                },
                None => eprintln!("HyperlinkEvent.getURL() returned null"),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Browser {
        loaded: Vec<String>,
        fail: bool,
    }

    impl HtmlPageWindowBrowser for Browser {
        fn set_page(&mut self, url: &str) -> Result<(), String> {
            self.loaded.push(url.into());
            if self.fail {
                Err("load failed".into())
            } else {
                Ok(())
            }
        }
    }

    #[test]
    fn constructor_matches_swing_setup() {
        let window = HtmlPageWindow::new();
        assert_eq!(window.rcsid, "$Id$");
        assert_eq!(window.size, (625, 800));
        assert_eq!(
            window.default_close_operation,
            HtmlPageWindowCloseOperation::DisposeOnClose
        );
        assert_eq!(window.editor_pane.editor_kit, HtmlPageWindowEditorKit::Html);
        assert!(window.editor_pane.editable);
        assert_eq!(window.editor_pane.hyperlink_listener_count, 1);
        assert!(window.main_panel.scroll_pane_in_border_layout_center);
        assert!(window.scroll_pane.view_is_editor_pane);
    }

    #[test]
    fn open_url_sets_url_page_title_and_editability_only_after_loading() {
        let mut window = HtmlPageWindow::new();
        let mut browser = Browser::default();
        window.open_url("https://example.org/page.html".into(), &mut browser);
        assert_eq!(window.url.as_deref(), Some("https://example.org/page.html"));
        assert_eq!(
            window.editor_pane.page.as_deref(),
            Some("https://example.org/page.html")
        );
        assert_eq!(window.title, "https://example.org/page.html");
        assert!(!window.editor_pane.editable);

        browser.fail = true;
        window.open_url("https://example.org/fails.html".into(), &mut browser);
        assert_eq!(
            window.url.as_deref(),
            Some("https://example.org/fails.html")
        );
        assert_eq!(
            window.editor_pane.page.as_deref(),
            Some("https://example.org/page.html")
        );
        assert_eq!(window.title, "https://example.org/page.html");
    }

    #[test]
    fn activated_hyperlink_uses_url_path_as_title_and_other_events_do_nothing() {
        let mut window = HtmlPageWindow::new();
        let mut browser = Browser::default();
        window.hyperlink_update(
            &HtmlPageWindowHyperlinkEvent {
                event_type: HtmlPageWindowHyperlinkEventType::Activated,
                url: Some(HtmlPageWindowUrl {
                    value: "https://example.org/help/intro.html?x=1".into(),
                    path: "/help/intro.html".into(),
                }),
            },
            &mut browser,
        );
        assert_eq!(
            window.editor_pane.page.as_deref(),
            Some("https://example.org/help/intro.html?x=1")
        );
        assert_eq!(window.title, "/help/intro.html");
        assert_eq!(window.url, None);

        window.hyperlink_update(
            &HtmlPageWindowHyperlinkEvent {
                event_type: HtmlPageWindowHyperlinkEventType::Entered,
                url: None,
            },
            &mut browser,
        );
        assert_eq!(browser.loaded.len(), 1);
    }
}
