//! Translation of `IMOD/3dmod/pixelview.cpp` and `pixelview.h`.
//!
//! The Qt widgets, image-file reads, display controls, Zap/XYZ/Slicer state,
//! and key dispatch remain direct viewer boundaries.  The source-owned pixel
//! grid, labels, minimum/maximum highlighting, and callbacks are represented
//! here without introducing a replacement image viewer.
#![allow(dead_code)]

pub const PV_ROWS: usize = 7;
pub const PV_COLS: usize = 7;
pub const IMOD_DRAW_IMAGE: i32 = 1;
pub const IMOD_DRAW_XYZ: i32 = 1 << 1;
pub const MRC_MODE_BYTE: i32 = 0;
pub const MRC_MODE_SHORT: i32 = 1;
pub const MRC_MODE_USHORT: i32 = 6;

/// Qt-independent `QKeyEvent` fields used by `PixelView`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PixelViewKeyEvent {
    pub key: i32,
    pub keypad_modifier: bool,
    pub close_key: bool,
}

/// `QColor` values used by the source widget palette.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PixelViewColor(pub u8, pub u8, pub u8);

/// `ViewInfo` fields read by this unit.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PixelViewImageState {
    pub xsize: i32,
    pub ysize: i32,
    pub zsize: i32,
    pub xmouse: f32,
    pub ymouse: f32,
    pub zmouse: f32,
    pub image_mode: i32,
    pub rgb_store: bool,
    pub image_pyramid: bool,
    pub no_readable_image: bool,
    pub multi_file_z: i32,
    pub image_file_is_jpeg: bool,
}

/// Native Qt/image/viewer boundary called by the paired source unit.
pub trait PixelViewNativeBoundary {
    fn image_state(&self) -> PixelViewImageState;
    fn image_list_file_is_jpeg(&self, iz: i32) -> bool;
    fn file_value(&mut self, x: i32, y: i32, z: i32) -> f32;
    fn memory_value(&mut self, x: i32, y: i32, z: i32) -> f32;
    fn rgb_value(&mut self, x: i32, y: i32, z: i32) -> [u8; 3];
    fn load_tiles_containing_area(&mut self, x: i32, y: i32, width: i32, height: i32, z: i32);
    fn raise_window(&mut self);
    fn new_control(&mut self) -> i32;
    fn remove_control(&mut self, control: i32);
    fn control_priority(&mut self, control: i32);
    fn bind_mouse(&mut self, x: f32, y: f32);
    fn draw(&mut self, flags: i32);
    fn show_help_page(&mut self, page: &str);
    fn pixel_view_state(&mut self, state: bool);
    fn adjust_geometry_and_show(&mut self);
    fn close_window(&mut self);
    fn input_default_arrow_key(&mut self, event: PixelViewKeyEvent);
    fn control_key(&mut self, released: bool, event: PixelViewKeyEvent);
    fn check_and_set_mac_menu(&mut self);
}

/// Private `getAndConvertRGB`.
pub fn get_and_convert_rgb(data: [u8; 3]) -> (i32, i32, i32, i32) {
    let red = data[0] as i32;
    let green = data[1] as i32;
    let blue = data[2] as i32;
    (
        (0.3 * red as f64 + 0.59 * green as f64 + 0.11 * blue as f64).round() as i32,
        red,
        green,
        blue,
    )
}

/// Private `fileReadable`.
pub fn file_readable(native: &dyn PixelViewNativeBoundary, iz: i32) -> bool {
    let state = native.image_state();
    let jpeg = if state.multi_file_z != 0 && iz >= 0 && iz < state.multi_file_z {
        native.image_list_file_is_jpeg(iz)
    } else {
        state.image_file_is_jpeg
    };
    !state.rgb_store && !state.no_readable_image && !jpeg
}

/// A source-visible button/label state, standing in for the corresponding Qt widget.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PixelViewButton {
    pub text: String,
    pub visible: bool,
    pub enabled: bool,
    pub minimum_width: i32,
    pub color: PixelViewColor,
}
impl Default for PixelViewButton {
    fn default() -> Self {
        Self {
            text: "8".into(),
            visible: true,
            enabled: true,
            minimum_width: 0,
            color: PixelViewColor(239, 239, 239),
        }
    }
}

/// Source file statics, made owned so multiple Rust viewers can be tested safely.
#[derive(Clone, Debug)]
pub struct PixelViewRuntime {
    pub pixel_view_dialog: Option<PixelView>,
    pub ctrl: i32,
    pub from_file: bool,
    pub last_readable: bool,
    pub last_grid_readable: bool,
    pub grid_from_file: bool,
    pub last_mouse_x: f32,
    pub last_mouse_y: f32,
    pub show_buts: bool,
    pub convert_rgb: bool,
}
impl Default for PixelViewRuntime {
    fn default() -> Self {
        Self {
            pixel_view_dialog: None,
            ctrl: 0,
            from_file: false,
            last_readable: false,
            last_grid_readable: false,
            grid_from_file: true,
            last_mouse_x: 0.,
            last_mouse_y: 0.,
            show_buts: true,
            convert_rgb: false,
        }
    }
}

/// `PixelView` (`pixelview.h`), including all widget state owned by the form.
#[derive(Clone, Debug)]
pub struct PixelView {
    pub m_mouse_label: PixelViewButton,
    pub m_file_val_box: PixelViewButton,
    pub m_bot_labels: [PixelViewButton; PV_COLS],
    pub m_left_labels: [PixelViewButton; PV_ROWS],
    pub m_lab_xy: PixelViewButton,
    pub m_buttons: [[PixelViewButton; PV_COLS]; PV_ROWS],
    pub m_gray_color: PixelViewColor,
    pub m_min_row: i32,
    pub m_min_col: i32,
    pub m_max_row: i32,
    pub m_max_col: i32,
    pub m_grid_val_box: PixelViewButton,
    pub m_convert_box: Option<PixelViewButton>,
    pub m_help_button: PixelViewButton,
    pub window_title: String,
    pub width: i32,
    pub height: i32,
}
impl PixelView {
    /// `PixelView::PixelView`.
    pub fn new(state: PixelViewImageState, runtime: &PixelViewRuntime, readable: bool) -> Self {
        let gray = PixelViewColor(239, 239, 239);
        let label = PixelViewButton {
            text: "88888".into(),
            color: gray,
            ..Default::default()
        };
        let mut out = Self {
            m_mouse_label: PixelViewButton {
                text: " ".into(),
                color: gray,
                ..Default::default()
            },
            m_file_val_box: PixelViewButton {
                text: "File value".into(),
                enabled: readable,
                color: gray,
                ..Default::default()
            },
            m_bot_labels: core::array::from_fn(|_| PixelViewButton {
                text: "8".into(),
                color: gray,
                ..Default::default()
            }),
            m_left_labels: core::array::from_fn(|_| label.clone()),
            m_lab_xy: PixelViewButton {
                text: "Y/X".into(),
                color: gray,
                ..Default::default()
            },
            m_buttons: core::array::from_fn(|_| {
                core::array::from_fn(|_| PixelViewButton {
                    color: gray,
                    ..Default::default()
                })
            }),
            m_gray_color: gray,
            m_min_row: -1,
            m_min_col: 0,
            m_max_row: -1,
            m_max_col: 0,
            m_grid_val_box: PixelViewButton {
                text: "Grid value from file".into(),
                enabled: readable,
                color: gray,
                ..Default::default()
            },
            m_convert_box: state.rgb_store.then(|| PixelViewButton {
                text: "Convert RGB to gray scale".into(),
                color: gray,
                ..Default::default()
            }),
            m_help_button: PixelViewButton {
                text: "Help".into(),
                color: gray,
                ..Default::default()
            },
            window_title: "3dmod Pixel View".into(),
            width: 0,
            height: 0,
        };
        out.set_button_widths();
        out
    }

    /// `PixelView::~PixelView`.
    pub fn destroy(&mut self) {}

    /// `PixelView::setButtonWidths`.
    pub fn set_button_widths(&mut self) {
        for row in &mut self.m_buttons {
            for button in row {
                button.minimum_width = 50;
            }
        }
        self.m_help_button.minimum_width = 38;
    }

    /// `PixelView::changeEvent`.
    pub fn change_event(&mut self, font_change: bool, native: &mut dyn PixelViewNativeBoundary) {
        native.check_and_set_mac_menu();
        if font_change {
            self.set_button_widths()
        }
    }

    /// `PixelView::update`.
    pub fn update(
        &mut self,
        runtime: &mut PixelViewRuntime,
        native: &mut dyn PixelViewNativeBoundary,
    ) {
        let state = native.image_state();
        native.raise_window();
        if !runtime.show_buts {
            return;
        }
        if self.m_min_row >= 0 {
            self.m_buttons[self.m_min_row as usize][self.m_min_col as usize].color =
                self.m_gray_color;
        }
        if self.m_max_row >= 0 {
            self.m_buttons[self.m_max_row as usize][self.m_max_col as usize].color =
                self.m_gray_color;
        }
        self.m_min_row = -1;
        self.m_max_row = -1;
        let iz = state.zmouse.round() as i32;
        let readable = file_readable(native, iz);
        if readable != runtime.last_grid_readable {
            self.m_grid_val_box.enabled = readable;
            runtime.last_grid_readable = readable;
        }
        if state.image_pyramid {
            let x = ((state.xmouse.floor() as i32) - PV_COLS as i32 / 2).max(0);
            let y = ((state.ymouse.floor() as i32) - PV_ROWS as i32 / 2).max(0);
            native.load_tiles_containing_area(
                x,
                y,
                (state.xsize - x).min(PV_COLS as i32),
                (state.ysize - y).min(PV_ROWS as i32),
                iz,
            );
        }
        let mut min_val = 1e38_f32;
        let mut max_val = -1e38_f32;
        let mut floats = -1;
        for i in 0..PV_COLS {
            let x = state.xmouse.floor() as i32 + i as i32 - PV_COLS as i32 / 2;
            self.m_bot_labels[i].text = if i == PV_COLS / 2 {
                format!("{:5}*", x + 1)
            } else {
                format!("{:5} ", x + 1)
            };
            for j in 0..PV_ROWS {
                let y = state.ymouse.floor() as i32 + j as i32 - PV_ROWS as i32 / 2;
                let text;
                if x < 0 || y < 0 || x >= state.xsize || y >= state.ysize {
                    text = "     x".into();
                } else {
                    let (pixel, red, green, blue) = if readable && runtime.grid_from_file {
                        (native.file_value(x, y, iz), 0, 0, 0)
                    } else if state.rgb_store {
                        let (p, r, g, b) = get_and_convert_rgb(native.rgb_value(x, y, iz));
                        (p as f32, r, g, b)
                    } else {
                        (native.memory_value(x, y, iz), 0, 0, 0)
                    };
                    if floats < 0 && readable {
                        floats = i32::from(
                            !(state.image_mode == MRC_MODE_BYTE
                                || state.image_mode == MRC_MODE_SHORT
                                || state.image_mode == MRC_MODE_USHORT),
                        );
                    }
                    text = if floats > 0 {
                        format!("{pixel:9}")
                    } else if state.rgb_store && !runtime.convert_rgb {
                        format!("{:3},{:3},{:3}", red, green, blue)
                    } else {
                        format!("{:6}", pixel as i32)
                    };
                    if pixel < min_val {
                        min_val = pixel;
                        self.m_min_col = i as i32;
                        self.m_min_row = j as i32;
                    }
                    if pixel > max_val {
                        max_val = pixel;
                        self.m_max_col = i as i32;
                        self.m_max_row = j as i32;
                    }
                }
                self.m_buttons[j][i].text = text;
                if i == 0 {
                    self.m_left_labels[j].text = if j == PV_COLS / 2 {
                        format!("{:5}*", y + 1)
                    } else {
                        format!("{:5} ", y + 1)
                    };
                }
            }
        }
        if self.m_min_row >= 0 {
            self.m_buttons[self.m_min_row as usize][self.m_min_col as usize].color =
                PixelViewColor(0, 255, 255);
        }
        if self.m_max_row >= 0 {
            self.m_buttons[self.m_max_row as usize][self.m_max_col as usize].color =
                PixelViewColor(255, 0, 128);
        }
    }

    /// `PixelView::buttonPressed`.
    pub fn button_pressed(
        &mut self,
        pos: i32,
        runtime: &PixelViewRuntime,
        native: &mut dyn PixelViewNativeBoundary,
    ) {
        native.control_priority(runtime.ctrl);
        let y = pos / PV_COLS as i32 - PV_COLS as i32 / 2;
        let x = pos % PV_COLS as i32 - PV_ROWS as i32 / 2;
        let state = native.image_state();
        native.bind_mouse(state.xmouse + x as f32, state.ymouse + y as f32);
        native.draw(IMOD_DRAW_XYZ);
    }
    /// `PixelView::fromFileToggled`.
    pub fn from_file_toggled(&mut self, state: bool, runtime: &mut PixelViewRuntime) {
        runtime.from_file = state
    }
    /// `PixelView::gridFileToggled`.
    pub fn grid_file_toggled(
        &mut self,
        state: bool,
        runtime: &mut PixelViewRuntime,
        native: &mut dyn PixelViewNativeBoundary,
    ) {
        runtime.grid_from_file = state;
        self.update(runtime, native)
    }
    /// `PixelView::convertToggled`.
    pub fn convert_toggled(
        &mut self,
        state: bool,
        runtime: &mut PixelViewRuntime,
        native: &mut dyn PixelViewNativeBoundary,
    ) {
        runtime.convert_rgb = state;
        self.update(runtime, native);
        self.adjust_dialog_size(runtime, native)
    }
    /// `PixelView::helpClicked`.
    pub fn help_clicked(&mut self, native: &mut dyn PixelViewNativeBoundary) {
        native.show_help_page("pixelview.html#TOP")
    }
    /// `PixelView::showButsToggled`.
    pub fn show_buts_toggled(
        &mut self,
        state: bool,
        runtime: &mut PixelViewRuntime,
        native: &mut dyn PixelViewNativeBoundary,
    ) {
        runtime.show_buts = state;
        for i in 0..PV_ROWS {
            for j in 0..PV_COLS {
                self.m_buttons[i][j].visible = state;
            }
            self.m_left_labels[i].visible = state;
        }
        for label in &mut self.m_bot_labels {
            label.visible = state;
        }
        self.m_lab_xy.visible = state;
        self.m_grid_val_box.visible = state;
        if let Some(box_) = &mut self.m_convert_box {
            box_.visible = state;
        }
        self.m_help_button.visible = state;
        self.adjust_dialog_size(runtime, native);
        self.update(runtime, native);
    }
    /// `PixelView::adjustDialogSize`.
    pub fn adjust_dialog_size(
        &mut self,
        runtime: &PixelViewRuntime,
        native: &mut dyn PixelViewNativeBoundary,
    ) {
        let state = native.image_state();
        if runtime.show_buts
            && (state.image_mode == MRC_MODE_BYTE
                || state.image_mode == MRC_MODE_SHORT
                || state.image_mode == MRC_MODE_USHORT
                || (state.rgb_store && runtime.convert_rgb))
        {
            self.width = 70;
            self.height = 0;
        } else {
            self.width = 0;
            self.height = 0;
        }
    }
    /// `PixelView::closeEvent`.
    pub fn close_event(
        &mut self,
        runtime: &mut PixelViewRuntime,
        native: &mut dyn PixelViewNativeBoundary,
    ) {
        native.remove_control(runtime.ctrl);
        native.pixel_view_state(false);
    }
    /// `PixelView::keyPressEvent`.
    pub fn key_press_event(
        &mut self,
        event: PixelViewKeyEvent,
        native: &mut dyn PixelViewNativeBoundary,
    ) {
        if event.close_key {
            native.close_window();
            return;
        }
        if !event.keypad_modifier
            && matches!(
                event.key,
                0x0100_0012 | 0x0100_0013 | 0x0100_0014 | 0x0100_0015
            )
        {
            native.input_default_arrow_key(event)
        } else {
            native.control_key(false, event)
        }
    }
    /// `PixelView::keyReleaseEvent`.
    pub fn key_release_event(
        &mut self,
        event: PixelViewKeyEvent,
        native: &mut dyn PixelViewNativeBoundary,
    ) {
        native.control_key(true, event)
    }
}

/// Private `pviewClose_cb`.
pub fn pview_close_cb(runtime: &mut PixelViewRuntime, native: &mut dyn PixelViewNativeBoundary) {
    if let Some(mut view) = runtime.pixel_view_dialog.take() {
        view.close_event(runtime, native)
    }
}
/// Private `pviewDraw_cb`.
pub fn pview_draw_cb(
    runtime: &mut PixelViewRuntime,
    drawflag: i32,
    native: &mut dyn PixelViewNativeBoundary,
) {
    if drawflag & (IMOD_DRAW_XYZ | IMOD_DRAW_IMAGE) != 0 {
        if let Some(mut view) = runtime.pixel_view_dialog.take() {
            view.update(runtime, native);
            runtime.pixel_view_dialog = Some(view);
        }
    }
}
/// `open_pixelview`.
pub fn open_pixelview(
    runtime: &mut PixelViewRuntime,
    native: &mut dyn PixelViewNativeBoundary,
) -> i32 {
    if runtime.pixel_view_dialog.is_some() {
        native.raise_window();
        return -1;
    }
    let state = native.image_state();
    runtime.pixel_view_dialog = Some(PixelView::new(
        state,
        runtime,
        file_readable(native, state.zmouse.round() as i32),
    ));
    runtime.ctrl = native.new_control();
    native.adjust_geometry_and_show();
    pv_new_mouse_position(
        runtime,
        state.xmouse,
        state.ymouse,
        state.zmouse.round() as i32,
        native,
    );
    if let Some(mut view) = runtime.pixel_view_dialog.take() {
        view.show_buts_toggled(runtime.show_buts, runtime, native);
        runtime.pixel_view_dialog = Some(view);
    }
    native.pixel_view_state(true);
    0
}
/// `pvNewMousePosition`.
pub fn pv_new_mouse_position(
    runtime: &mut PixelViewRuntime,
    x: f32,
    y: f32,
    iz: i32,
    native: &mut dyn PixelViewNativeBoundary,
) {
    runtime.last_mouse_x = x;
    runtime.last_mouse_y = y;
    let state = native.image_state();
    let ix = x as i32;
    let iy = y as i32;
    if runtime.pixel_view_dialog.is_none()
        || ix < 0
        || iy < 0
        || ix >= state.xsize
        || iy >= state.ysize
        || iz < 0
        || iz >= state.zsize
    {
        return;
    }
    let readable = file_readable(native, iz);
    if readable != runtime.last_readable {
        if let Some(view) = &mut runtime.pixel_view_dialog {
            view.m_file_val_box.enabled = readable;
        }
        runtime.last_readable = readable;
    }
    let (value, is_float) = if runtime.from_file && readable {
        (
            native.file_value(ix, iy, iz),
            !(state.image_mode == MRC_MODE_BYTE
                || state.image_mode == MRC_MODE_SHORT
                || state.image_mode == MRC_MODE_USHORT),
        )
    } else if state.rgb_store {
        (
            get_and_convert_rgb(native.rgb_value(ix, iy, iz)).0 as f32,
            false,
        )
    } else {
        (native.memory_value(ix, iy, iz), false)
    };
    if let Some(view) = &mut runtime.pixel_view_dialog {
        view.m_mouse_label.text = if is_float {
            format!(
                "Mouse: {:5}, {:5}, {:4}  Value: {:9}",
                ix + 1,
                iy + 1,
                iz + 1,
                value
            )
        } else {
            format!(
                "Mouse: {:5}, {:5}, {:4}  Value: {:3}",
                ix + 1,
                iy + 1,
                iz + 1,
                value as i32
            )
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct N {
        state: PixelViewImageState,
        draws: Vec<i32>,
        mouse: (f32, f32),
        tiles: usize,
    }
    impl PixelViewNativeBoundary for N {
        fn image_state(&self) -> PixelViewImageState {
            self.state
        }
        fn image_list_file_is_jpeg(&self, _: i32) -> bool {
            false
        }
        fn file_value(&mut self, x: i32, y: i32, _: i32) -> f32 {
            (x + y * 10) as f32
        }
        fn memory_value(&mut self, x: i32, y: i32, _: i32) -> f32 {
            (x + y * 10) as f32
        }
        fn rgb_value(&mut self, _: i32, _: i32, _: i32) -> [u8; 3] {
            [10, 20, 30]
        }
        fn load_tiles_containing_area(&mut self, _: i32, _: i32, _: i32, _: i32, _: i32) {
            self.tiles += 1
        }
        fn raise_window(&mut self) {}
        fn new_control(&mut self) -> i32 {
            3
        }
        fn remove_control(&mut self, _: i32) {}
        fn control_priority(&mut self, _: i32) {}
        fn bind_mouse(&mut self, x: f32, y: f32) {
            self.mouse = (x, y)
        }
        fn draw(&mut self, f: i32) {
            self.draws.push(f)
        }
        fn show_help_page(&mut self, _: &str) {}
        fn pixel_view_state(&mut self, _: bool) {}
        fn adjust_geometry_and_show(&mut self) {}
        fn close_window(&mut self) {}
        fn input_default_arrow_key(&mut self, _: PixelViewKeyEvent) {}
        fn control_key(&mut self, _: bool, _: PixelViewKeyEvent) {}
        fn check_and_set_mac_menu(&mut self) {}
    }
    #[test]
    fn rgb_luminance_matches_source_weights() {
        assert_eq!(get_and_convert_rgb([10, 20, 30]), (18, 10, 20, 30));
    }
    #[test]
    fn update_highlights_grid_extrema_and_button_moves_mouse() {
        let mut n = N {
            state: PixelViewImageState {
                xsize: 20,
                ysize: 20,
                zsize: 1,
                xmouse: 10.,
                ymouse: 10.,
                zmouse: 0.,
                image_mode: MRC_MODE_BYTE,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut r = PixelViewRuntime::default();
        open_pixelview(&mut r, &mut n);
        let mut v = r.pixel_view_dialog.take().unwrap();
        assert_eq!(v.m_buttons[0][0].text.trim(), "77");
        assert_eq!(v.m_buttons[6][6].color, PixelViewColor(255, 0, 128));
        v.button_pressed(48, &r, &mut n);
        assert_eq!(n.mouse, (13., 13.));
        assert_eq!(n.draws, vec![IMOD_DRAW_XYZ]);
    }
}
