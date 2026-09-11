//! Translation of `IMOD/3dmod/formv_views.cpp` and `formv_views.h`.
//! `mv_views.cpp` is not yet a translated source unit, therefore its calls are
//! explicit `ViewsOperations` boundary methods rather than simulated views.
#![allow(dead_code)]
pub const VIEW_STRSIZE: usize = 128;
/// Direct counterparts of the calls in `mv_views.h`.
pub trait ViewsOperations {
    fn store(&mut self, item: i32);
    fn goto(&mut self, item: i32, revert: bool);
    fn new_view(&mut self, label: &str);
    fn delete(&mut self, item: i32, current: i32);
    fn save(&mut self);
    fn autostore(&mut self, state: i32);
    fn label(&mut self, label: &str, item: i32);
    fn closing(&mut self);
    fn done(&mut self);
    fn key_press(&mut self, key: i32);
    fn key_release(&mut self, key: i32);
}
/// Original `imodvViewsForm`.
#[derive(Clone, Debug, Default)]
pub struct ImodvViewsForm {
    pub top_window_open: bool,
    pub delete_on_close: bool,
    pub always_show_tooltips: bool,
    pub autostore: bool,
    pub items: Vec<String>,
    pub current_item: i32,
    pub label_edit: String,
    pub delete_enabled: bool,
    pub list_width: i32,
    pub font_height: i32,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct ViewsKeyEvent {
    pub key: i32,
    pub close: bool,
    pub keypad: bool,
    pub list_height: i32,
}
pub const KEY_UP: i32 = 1;
pub const KEY_DOWN: i32 = 2;
pub const KEY_PAGE_UP: i32 = 3;
pub const KEY_PAGE_DOWN: i32 = 4;
/// Original constructor `imodvViewsForm::imodvViewsForm`.
pub fn imodv_views_form_new() -> ImodvViewsForm {
    let mut form = ImodvViewsForm {
        top_window_open: true,
        ..Default::default()
    };
    form.init();
    form
}
impl ImodvViewsForm {
    pub fn destroy(&mut self) {
        self.top_window_open = false;
    }
    pub fn language_change(&mut self) {}
    /// Original `init`.
    pub fn init(&mut self) {
        self.delete_on_close = true;
        self.always_show_tooltips = true;
        self.font_height = 1;
        self.set_font_dependent_widths();
    }
    pub fn set_font_dependent_widths(&mut self) {
        self.list_width = "A very long string fits here".len() as i32;
    }
    pub fn manage_list_width(&mut self) {
        let maximum = self
            .items
            .iter()
            .map(|x| x.len() as i32 + 30)
            .max()
            .unwrap_or(124)
            .max(124);
        if maximum < self.list_width - 40 || maximum > self.list_width {
            self.list_width = maximum;
        }
    }
    pub fn store_pressed(&mut self, ops: &mut dyn ViewsOperations) {
        if self.current_item >= 0 {
            ops.store(self.current_item)
        }
    }
    pub fn revert_pressed(&mut self, ops: &mut dyn ViewsOperations) {
        if self.current_item >= 0 {
            ops.goto(self.current_item, true)
        }
    }
    pub fn new_pressed(&mut self, ops: &mut dyn ViewsOperations) {
        for i in 1..1000 {
            let label = format!("view {i}");
            if !self.items.contains(&label) {
                ops.new_view(&label);
                return;
            }
        }
    }
    pub fn delete_pressed(&mut self, ops: &mut dyn ViewsOperations) {
        let item = self.current_item;
        if item < 0 || item as usize >= self.items.len() {
            return;
        }
        self.items.remove(item as usize);
        if self.current_item >= self.items.len() as i32 {
            self.current_item = self.items.len() as i32 - 1;
        }
        ops.delete(item, self.current_item);
        self.delete_enabled = self.items.len() > 1;
        self.manage_list_width();
    }
    pub fn save_pressed(&mut self, ops: &mut dyn ViewsOperations) {
        ops.save()
    }
    pub fn autostore_toggled(&mut self, ops: &mut dyn ViewsOperations, state: bool) {
        self.autostore = state;
        ops.autostore(state as i32)
    }
    pub fn view_selected(&mut self, ops: &mut dyn ViewsOperations, item: i32) {
        if item >= 0 && (item as usize) < self.items.len() {
            self.current_item = item;
            self.label_edit = self.items[item as usize].clone();
            ops.goto(item, true)
        }
    }
    pub fn new_label_entered(&mut self, ops: &mut dyn ViewsOperations) {
        let label = self
            .label_edit
            .chars()
            .take(VIEW_STRSIZE - 1)
            .collect::<String>();
        self.label_edit = label.clone();
        let item = self.current_item;
        if item < 0 || item as usize >= self.items.len() {
            return;
        }
        self.items[item as usize] = label.clone();
        ops.label(&label, item);
        self.manage_list_width();
    }
    pub fn set_autostore(&mut self, state: i32) {
        self.autostore = state != 0;
        self.manage_list_width()
    }
    pub fn add_item(&mut self, label: &str) {
        self.items.push(label.into());
        self.delete_enabled = self.items.len() > 1;
    }
    pub fn select_item(&mut self, item: i32, block: bool) {
        if self.items.is_empty() {
            self.current_item = -1;
            return;
        }
        self.current_item = item.clamp(0, self.items.len() as i32 - 1);
        self.label_edit = self.items[self.current_item as usize].clone();
    }
    pub fn remove_all_items(&mut self) {
        self.items.clear();
        self.current_item = -1;
        self.label_edit.clear();
        self.delete_enabled = false;
    }
    pub fn top_close_event(&mut self, ops: &mut dyn ViewsOperations) {
        ops.closing();
        self.top_window_open = false;
    }
    pub fn key_press_event(&mut self, ops: &mut dyn ViewsOperations, event: ViewsKeyEvent) {
        if event.close {
            ops.done();
            self.top_window_open = false;
            return;
        }
        let jump = (event.list_height / self.font_height.max(1) - 1).max(1);
        if event.keypad {
            ops.key_press(event.key);
            return;
        }
        match event.key {
            KEY_UP => self.select_item(self.current_item - 1, false),
            KEY_DOWN => self.select_item(self.current_item + 1, false),
            KEY_PAGE_UP => self.select_item(self.current_item - jump, false),
            KEY_PAGE_DOWN => self.select_item(self.current_item + jump, false),
            _ => ops.key_press(event.key),
        }
    }
    pub fn key_release_event(&mut self, ops: &mut dyn ViewsOperations, event: ViewsKeyEvent) {
        ops.key_release(event.key)
    }
    pub fn top_change_event(&mut self, font_changed: bool) {
        if font_changed {
            self.set_font_dependent_widths()
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Ops {
        calls: Vec<String>,
    }
    impl ViewsOperations for Ops {
        fn store(&mut self, i: i32) {
            self.calls.push(format!("store{i}"))
        }
        fn goto(&mut self, i: i32, _: bool) {
            self.calls.push(format!("goto{i}"))
        }
        fn new_view(&mut self, x: &str) {
            self.calls.push(x.into())
        }
        fn delete(&mut self, _: i32, _: i32) {}
        fn save(&mut self) {}
        fn autostore(&mut self, _: i32) {}
        fn label(&mut self, _: &str, _: i32) {}
        fn closing(&mut self) {}
        fn done(&mut self) {}
        fn key_press(&mut self, _: i32) {}
        fn key_release(&mut self, _: i32) {}
    }
    #[test]
    fn new_and_navigation_use_direct_boundary() {
        let mut f = imodv_views_form_new();
        let mut o = Ops::default();
        f.add_item("view 1");
        f.new_pressed(&mut o);
        assert_eq!(o.calls, ["view 2"]);
        f.view_selected(&mut o, 0);
        assert_eq!(o.calls[1], "goto0");
    }
}
