//! `IMOD/Etomo/src/etomo/ui/swing/SeriesWatcherParent.java`.
#![allow(dead_code)]

/// Java package-private `SeriesWatcherParent`.
pub trait SeriesWatcherParent {
    /// Java `isSeriesWatcherOn()`.
    fn is_series_watcher_on(&self) -> bool;
    /// Java `equalsSeriesWatcherActionCommand(String)`.
    fn equals_series_watcher_action_command(&self, action_command: &str) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Parent;
    impl SeriesWatcherParent for Parent {
        fn is_series_watcher_on(&self) -> bool {
            true
        }
        fn equals_series_watcher_action_command(&self, action_command: &str) -> bool {
            action_command == "watch"
        }
    }
    #[test]
    fn source_contract_includes_action_command_comparison() {
        let parent = Parent;
        assert!(parent.is_series_watcher_on());
        assert!(parent.equals_series_watcher_action_command("watch"));
    }
}
