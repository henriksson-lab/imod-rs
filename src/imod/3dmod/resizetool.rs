//! Translation of `IMOD/3dmod/resizetool.cpp` and `resizetool.h`.
#![allow(dead_code)]
/// The legacy Qt/DialogFrame, preferences, and window-device conversion calls
/// remain named GUI boundaries; no WGPU path is introduced by this unit.
pub trait ResizeToolBoundary {
    fn maximum_window_size(&self) -> (i32, i32);
    fn pixel_to_device(&self, width: i32, height: i32) -> (i32, i32);
    fn rounded_style(&self) -> bool;
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResizeTool {
    pub x_spin_box: i32,
    pub y_spin_box: i32,
    pub x_maximum: i32,
    pub y_maximum: i32,
    pub step: i32,
    pub rounded_style: bool,
    pub focused: bool,
    pub closed: bool,
    pub resize_signal: Option<(i32, i32)>,
    pub closing_signal: bool,
    pub key_press_signal: bool,
    pub key_release_signal: bool,
}

/// `ResizeTool()`: construct the size controller through the native window
/// sizing and DPI boundary.
pub fn resize_tool(
    boundary: &dyn ResizeToolBoundary,
    x_size: i32,
    y_size: i32,
    step: i32,
) -> ResizeTool {
    ResizeTool::new(boundary, x_size, y_size, step)
}

impl ResizeTool {
    pub fn new(boundary: &dyn ResizeToolBoundary, x_size: i32, y_size: i32, step: i32) -> Self {
        let (w, h) = boundary.maximum_window_size();
        let (w, h) = boundary.pixel_to_device(w, h);
        let mut value = Self {
            x_spin_box: 0,
            y_spin_box: 0,
            x_maximum: w,
            y_maximum: h,
            step,
            rounded_style: boundary.rounded_style(),
            focused: false,
            closed: false,
            resize_signal: None,
            closing_signal: false,
            key_press_signal: false,
            key_release_signal: false,
        };
        value.new_size(x_size, y_size);
        value
    }
    pub fn new_size(&mut self, x: i32, y: i32) {
        self.x_spin_box = x;
        self.y_spin_box = y
    }
    pub fn x_size_changed(&mut self, value: i32) {
        self.focused = true;
        self.x_spin_box = value;
        self.resize_signal = Some((value, self.y_spin_box))
    }
    pub fn y_size_changed(&mut self, value: i32) {
        self.focused = true;
        self.y_spin_box = value;
        self.resize_signal = Some((self.x_spin_box, value))
    }
    pub fn button_clicked(&mut self, _which: i32) {
        self.closed = true
    }
    pub fn close_event(&mut self) {
        self.closing_signal = true
    }
    pub fn key_press_event(&mut self) {
        self.key_press_signal = true
    }
    pub fn key_release_event(&mut self) {
        self.key_release_signal = true
    }
    pub fn change_event(&mut self, boundary: &dyn ResizeToolBoundary, font_change: bool) {
        self.rounded_style = boundary.rounded_style();
        if !font_change {
            return;
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    struct B;
    impl ResizeToolBoundary for B {
        fn maximum_window_size(&self) -> (i32, i32) {
            (100, 80)
        }
        fn pixel_to_device(&self, x: i32, y: i32) -> (i32, i32) {
            (x * 2, y * 2)
        }
        fn rounded_style(&self) -> bool {
            true
        }
    }
    #[test]
    fn resize_slots_preserve_other_spin_value() {
        let mut x = ResizeTool::new(&B, 20, 30, 5);
        assert_eq!((x.x_maximum, x.y_maximum), (200, 160));
        x.x_size_changed(40);
        assert_eq!(x.resize_signal, Some((40, 30)));
        x.y_size_changed(50);
        assert_eq!(x.resize_signal, Some((40, 50)));
    }
    #[test]
    fn source_constructor_facade_uses_device_size_boundary() {
        let tool = resize_tool(&B, 10, 12, 2);
        assert_eq!((tool.x_spin_box, tool.y_spin_box, tool.step), (10, 12, 2));
    }
}
