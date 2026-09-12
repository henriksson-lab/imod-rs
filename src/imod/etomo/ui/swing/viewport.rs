//! `IMOD/Etomo/src/etomo/ui/swing/Viewport.java`.
#![allow(dead_code)]
use super::paging_panel::PagingViewport;
use super::viewable::Viewable;
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PagingPanelBoundary {
    pub visible: bool,
    pub up_enabled: bool,
    pub down_enabled: bool,
}
/// Java final `Viewport`.
pub struct Viewport<V: Viewable<FocusableParent = String>> {
    pub viewable: V,
    pub size: usize,
    pub unique_key: String,
    pub start: isize,
    pub end: isize,
    pub paging_panel: Option<PagingPanelBoundary>,
}
impl<V: Viewable<FocusableParent = String>> Viewport<V> {
    pub const MINUMUM_SIZE: usize = 5;
    pub fn new(viewable: V, size: isize, unique_key: impl Into<String>) -> Self {
        Self {
            viewable,
            size: if size < 5 { 5 } else { size as usize },
            unique_key: unique_key.into(),
            start: 0,
            end: -1,
            paging_panel: None,
        }
    }
    pub fn init_paging(&mut self) {
        self.paging_panel = Some(PagingPanelBoundary::default());
    }
    pub fn get_focusable_parents(&self) -> Option<Vec<String>> {
        Some(self.viewable.get_focusable_parents())
    }
    pub fn home_button_action(&mut self) {
        self.page_viewport(0)
    }
    pub fn page_up_button_action(&mut self) {
        self.page_viewport(self.start - self.size as isize)
    }
    pub fn up_button_action(&mut self) {
        self.page_viewport(self.start - 1)
    }
    pub fn down_button_action(&mut self) {
        self.page_viewport(self.start + 1)
    }
    pub fn page_down_button_action(&mut self) {
        self.page_viewport(self.start + self.size as isize)
    }
    pub fn end_button_action(&mut self) {
        self.page_viewport(self.viewable.size() as isize - self.size as isize)
    }
    pub fn page_viewport(&mut self, new_start: isize) {
        let (a, b) = (self.start, self.end);
        if (self.reset_viewport(new_start) && self.start != a) || self.end != b {
            self.viewable.msg_viewport_paged()
        }
    }
    pub fn reset_viewport(&mut self, new_start: isize) -> bool {
        let (n, size) = (self.viewable.size() as isize, self.size as isize);
        if let Some(p) = &mut self.paging_panel {
            p.visible = n > size
        }
        let changed = !(n <= size && self.start == 0 && self.end == n - 1);
        if changed {
            self.start = new_start;
            if self.start > n - size {
                self.start = n - size
            }
            if self.start < 0 {
                self.start = 0
            }
            self.end = self.start + size - 1;
            if self.end >= n {
                self.end = n - 1
            }
        }
        if let Some(p) = &mut self.paging_panel {
            p.up_enabled = self.start > 0;
            p.down_enabled = self.end < n - 1
        }
        changed
    }
    pub fn get_paging_panel(&self) -> Option<&PagingPanelBoundary> {
        self.paging_panel.as_ref()
    }
    pub fn msg_viewable_changed(&mut self) {
        self.reset_viewport(self.start);
    }
    pub fn adjust_viewport(&mut self, index: isize) -> bool {
        if index > self.end {
            self.reset_viewport(index - self.size as isize + 1)
        } else {
            self.reset_viewport(index)
        }
    }
    pub fn in_viewport(&mut self, index: isize) -> bool {
        self.reset_viewport(self.start);
        index >= self.start && index <= self.end
    }
}
impl<V: Viewable<FocusableParent = String>> PagingViewport for Viewport<V> {
    fn get_focusable_parents(&self) -> Option<Vec<String>> {
        Viewport::get_focusable_parents(self)
    }
    fn home_button_action(&mut self) {
        Viewport::home_button_action(self)
    }
    fn page_up_button_action(&mut self) {
        Viewport::page_up_button_action(self)
    }
    fn up_button_action(&mut self) {
        Viewport::up_button_action(self)
    }
    fn down_button_action(&mut self) {
        Viewport::down_button_action(self)
    }
    fn page_down_button_action(&mut self) {
        Viewport::page_down_button_action(self)
    }
    fn end_button_action(&mut self) {
        Viewport::end_button_action(self)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    struct T {
        n: usize,
        p: usize,
    }
    impl Viewable for T {
        type FocusableParent = String;
        fn msg_viewport_paged(&mut self) {
            self.p += 1
        }
        fn size(&self) -> usize {
            self.n
        }
        fn get_focusable_parents(&self) -> Vec<String> {
            vec![]
        }
    }
    #[test]
    fn clamps() {
        let mut v = Viewport::new(T { n: 11, p: 0 }, 5, "x");
        v.init_paging();
        v.end_button_action();
        assert_eq!((v.start, v.end), (6, 10));
        assert_eq!(v.viewable.p, 1)
    }
}
