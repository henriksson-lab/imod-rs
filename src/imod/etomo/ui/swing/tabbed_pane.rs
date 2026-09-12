//! `IMOD/Etomo/src/etomo/ui/swing/TabbedPane.java`.
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::SEPARATOR_CHAR;
use crate::imod::etomo::util::utilities;
/// Java final `TabbedPane` source-visible JTabbedPane state.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TabbedPane {
    pub tabs: Vec<(String, String)>,
    pub name: Option<String>,
    pub printed_names: Vec<String>,
}
impl TabbedPane {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn add_tab(&mut self, title: impl Into<String>, component: impl Into<String>) {
        self.tabs.push((title.into(), component.into()));
        if self.tabs.len() == 1 {
            let t = self.tabs[0].0.clone();
            self.set_name(&t, false)
        }
    }
    pub fn add_tab_spaced_panel(&mut self, title: impl Into<String>, component: impl Into<String>) {
        self.add_tab(title, component)
    }
    pub fn set_title_at(&mut self, index: usize, title: impl Into<String>) {
        let title = title.into();
        self.tabs[index].0 = title.clone();
        if index == 0 {
            self.set_name(&title, false)
        }
    }
    pub fn set_name(&mut self, text: &str, print_names: bool) {
        let value = utilities::convert_label_to_name(Some(text), false).unwrap_or_default();
        let name = format!("tab{SEPARATOR_CHAR}{value}");
        self.name = Some(name.clone());
        if print_names {
            self.printed_names.push(format!("{name} | "));
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn first_tab_and_title_name_pane() {
        let mut p = TabbedPane::new();
        p.add_tab("Setup", "p");
        assert_eq!(p.name.as_deref(), Some("tab.setup"));
        p.set_title_at(0, "Start");
        assert_eq!(p.name.as_deref(), Some("tab.start"));
    }
}
