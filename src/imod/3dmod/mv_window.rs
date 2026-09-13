//! Translation of `IMOD/3dmod/mv_window.cpp` together with `mv_window.h`.
//!
//! Qt owns the native widgets in the original.  The Rust unit keeps the same
//! window, menu, OpenGL-widget, timer, and event state.  `ImodvWindowSink` is
//! the direct counterpart of calls into the paired `mv_menu`, `mv_input`,
//! `mv_gfx`, and `mv_control` source units; it deliberately contains no
//! substitute renderer or synthetic input behaviour.
#![allow(dead_code)]

use crate::imod::three_dmod::imodv::ImodvApp;

/// `enum {... LAST_VMENU_ID}` in `mv_window.h`.
pub const VFILE_MENU_LOAD: usize = 0;
pub const VFILE_MENU_SAVE: usize = 1;
pub const VFILE_MENU_SAVEAS: usize = 2;
pub const VFILE_MENU_SNAPRGB: usize = 3;
pub const VFILE_MENU_SNAPTIFF: usize = 4;
pub const VFILE_MENU_ZEROSNAP: usize = 5;
pub const VFILE_MENU_SNAPDIR: usize = 6;
pub const VFILE_MENU_MOVIE: usize = 7;
pub const VFILE_MENU_SEQUENCE: usize = 8;
pub const VFILE_MENU_QUIT: usize = 9;
pub const VEDIT_MENU_OBJECTS: usize = 10;
pub const VEDIT_MENU_CONTROLS: usize = 11;
pub const VEDIT_MENU_ROTATION: usize = 12;
pub const VEDIT_MENU_OBJLIST: usize = 13;
pub const VEDIT_MENU_BKG: usize = 14;
pub const VEDIT_MENU_MODELS: usize = 15;
pub const VEDIT_MENU_VIEWS: usize = 16;
pub const VEDIT_MENU_IMAGE: usize = 17;
pub const VEDIT_MENU_ISOSURFACE: usize = 18;
pub const VEDIT_MENU_SAVE_DOCK: usize = 19;
pub const VEDIT_MENU_REOPEN_DOCK: usize = 20;
pub const VVIEW_MENU_DB: usize = 21;
pub const VVIEW_MENU_BOUNDBOX: usize = 22;
pub const VVIEW_MENU_OBJBOUND: usize = 23;
pub const VVIEW_MENU_CURPNT: usize = 24;
pub const VVIEW_MENU_INVERTZ: usize = 25;
pub const VVIEW_MENU_TRANSBKGD: usize = 26;
pub const VVIEW_MENU_LIGHTING: usize = 27;
pub const VVIEW_MENU_WIREFRAME: usize = 28;
pub const VVIEW_MENU_LOWRES: usize = 29;
pub const VVIEW_MENU_STEREO: usize = 30;
pub const VVIEW_MENU_DEPTH: usize = 31;
pub const VVIEW_MENU_SCALEBAR: usize = 32;
pub const VVIEW_MENU_RESIZE: usize = 33;
pub const VVIEW_MENU_LABELS: usize = 34;
pub const VHELP_MENU_MENUS: usize = 35;
pub const VHELP_MENU_KEYBOARD: usize = 36;
pub const VHELP_MENU_MOUSE: usize = 37;
pub const VHELP_MENU_ABOUT: usize = 38;
pub const LAST_VMENU_ID: usize = 39;

/// `QAction` state retained by `ImodvWindow::mActions`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Action {
    pub text: String,
    pub enabled: bool,
    pub checkable: bool,
    pub checked: bool,
    pub shortcut: Option<Key>,
}

/// Portable form of the Qt key/modifier payload used by the source handlers.
///
/// `ImodvGL::mousePressEvent`, `mouseReleaseEvent`, `mouseMoveEvent` and
/// `wheelEvent` are handed a `QMouseEvent`/`QWheelEvent` rather than a
/// `QKeyEvent`, and the `mv_input` handlers read its `x()`, `y()`, `button()`
/// and `delta()`, so those travel in the same payload.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KeyEvent {
    pub key: Key,
    pub control: bool,
    pub shift: bool,
    pub alt: bool,
    /// `QMouseEvent::x()`.
    pub x: i32,
    /// `QMouseEvent::y()`.
    pub y: i32,
    /// `QMouseEvent::button()`, in `mv_input`'s `INPUT_*` bits.
    pub button: u32,
    /// `QMouseEvent::buttons()`, the whole button state, in the same bits.
    pub buttons: u32,
    /// `QWheelEvent::delta()`.
    pub delta: i32,
    /// `Qt::KeypadModifier` on this event.
    pub keypad: bool,
    /// `QKeyEvent::key()` as the plain Qt code.  `Key` above is the widget
    /// vocabulary the form units match on; `mv_input.cpp` and
    /// `imod_input.cpp` switch on the integer, and `imod_input.rs` already
    /// carries those codes, so the same value travels here for them.
    pub qt_key: i32,
}

/// The one Qt event `mv_window.cpp` hands straight to `mv_input.cpp`.
///
/// The source passes the `QKeyEvent *`/`QMouseEvent *`/`QWheelEvent *` itself,
/// so this conversion exists only because the translation gives each unit its
/// own payload type; it moves the same fields and adds nothing.
impl From<KeyEvent> for crate::imod::three_dmod::mv_input::InputEvent {
    fn from(event: KeyEvent) -> Self {
        Self {
            key: event.qt_key,
            x: event.x,
            y: event.y,
            button: event.button,
            buttons: event.buttons,
            modifiers: (if event.control {
                crate::imod::three_dmod::mv_input::INPUT_CTRL
            } else {
                0
            }) | (if event.shift {
                crate::imod::three_dmod::mv_input::INPUT_SHIFT
            } else {
                0
            }) | (if event.keypad {
                crate::imod::three_dmod::mv_input::INPUT_KEYPAD
            } else {
                0
            }),
            delta: event.delta,
        }
    }
}

/// Values needed by `mv_window.cpp`; Qt key codes are intentionally not
/// exposed as platform integers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Key {
    #[default]
    Unknown,
    Character(char),
    Minus,
    Equal,
    Underscore,
    Plus,
    Comma,
    Period,
    Delete,
    LeftBracket,
    RightBracket,
    LeftBrace,
    RightBrace,
    LeftParen,
    RightParen,
    F(u8),
}

/// `QEvent` cases inspected by `ImodvWindow::event` and `changeEvent`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowEvent {
    ActivationChange { active: bool },
    WindowStateChange { minimized: bool, maximized: bool },
    Hide,
    Show,
    Resize { width: i32, height: i32 },
    Move,
    Close,
}

/// The QGL/QOpenGL surface attributes selected in `addGLWidgetToStack`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GlFormat {
    pub double_buffer: bool,
    pub depth: bool,
    pub stereo: bool,
    pub alpha: bool,
    pub samples: bool,
}

/// Exact data ownership carried by the `ImodvGL` C++ widget.
#[derive(Clone, Debug)]
pub struct ImodvGl {
    pub format: GlFormat,
    pub mouse_pressed: bool,
    pub first_draw: i32,
    pub timer_id: i32,
    pub sched_width: i32,
    pub sched_height: i32,
    pub scheduled_resize: bool,
    pub scheduled_bump: i32,
}

impl ImodvGl {
    /// `ImodvGL::ImodvGL`.
    pub fn new(format: GlFormat) -> Self {
        Self {
            format,
            mouse_pressed: false,
            first_draw: 3,
            timer_id: 0,
            sched_width: 0,
            sched_height: 0,
            scheduled_resize: false,
            scheduled_bump: 2,
        }
    }

    /// `ImodvGL::~ImodvGL`.
    pub fn destroy(&mut self) {}

    /// `ImodvGL::initializeGL`.
    pub fn initialize_gl(&mut self, sink: &mut dyn ImodvWindowSink) {
        sink.imodv_initialize_gl();
    }

    /// `ImodvGL::paintGL`.
    pub fn paint_gl(&mut self, window: &mut ImodvWindow, sink: &mut dyn ImodvWindowSink) {
        if self.first_draw >= 3 || window.app().need_new_qglinit > 0 {
            self.timer_id = sink.start_timer(10);
            window.app_mut().need_new_qglinit *= -1;
            self.first_draw -= 1;
        }
        if window.init_width == 0 && window.init_height == 0 {
            return;
        }
        sink.set_current_device_pixel_ratio(window.device_pixel_ratio);
        if window.app().need_new_qglinit == 0 {
            sink.imodv_paint_gl();
        }
    }

    /// `ImodvGL::timerEvent`.
    pub fn timer_event(&mut self, window: &mut ImodvWindow, sink: &mut dyn ImodvWindowSink) {
        if (window.init_width == 0 && window.init_height == 0) || window.app().need_new_qglinit != 0
        {
            return;
        }
        if self.scheduled_resize {
            self.cancel_resize(sink);
            sink.resize_window(self.sched_width + self.scheduled_bump, self.sched_height);
            self.scheduled_bump = -self.scheduled_bump;
            return;
        }
        if self.first_draw < 2 && self.timer_id != 0 {
            sink.kill_timer(self.timer_id);
            self.timer_id = 0;
        }
        if self.first_draw > 0 {
            sink.resize_window(window.init_width + self.first_draw - 1, window.init_height);
        } else {
            sink.update_gl();
        }
        self.first_draw = (self.first_draw - 1).max(0);
    }

    /// `ImodvGL::scheduleResize`.
    pub fn schedule_resize(
        &mut self,
        width: i32,
        height: i32,
        interval: i32,
        sink: &mut dyn ImodvWindowSink,
    ) {
        if self.first_draw > 0 {
            return;
        }
        if self.timer_id == 0 {
            self.timer_id = sink.start_timer(interval);
        }
        self.sched_width = width;
        self.sched_height = height;
        self.scheduled_resize = true;
    }
    /// `ImodvGL::cancelResize`.
    pub fn cancel_resize(&mut self, sink: &mut dyn ImodvWindowSink) {
        if self.timer_id != 0 {
            sink.kill_timer(self.timer_id);
        }
        self.timer_id = 0;
        self.scheduled_resize = false;
    }
    /// `ImodvGL::resizeGL`.
    pub fn resize_gl(&mut self, width: i32, height: i32, sink: &mut dyn ImodvWindowSink) {
        sink.imodv_resize_gl(width, height);
    }
    /// `ImodvGL::mousePressEvent`.
    pub fn mouse_press_event(&mut self, event: KeyEvent, sink: &mut dyn ImodvWindowSink) {
        self.mouse_pressed = true;
        sink.imodv_mouse_press(event);
    }
    /// `ImodvGL::mouseReleaseEvent`.
    pub fn mouse_release_event(&mut self, event: KeyEvent, sink: &mut dyn ImodvWindowSink) {
        self.mouse_pressed = false;
        sink.imodv_mouse_release(event);
    }
    /// `ImodvGL::mouseMoveEvent`.
    pub fn mouse_move_event(&mut self, event: KeyEvent, sink: &mut dyn ImodvWindowSink) {
        if self.mouse_pressed {
            sink.imodv_mouse_move(event);
        }
    }
    /// `ImodvGL::wheelEvent`.
    pub fn wheel_event(&mut self, event: KeyEvent, sink: &mut dyn ImodvWindowSink) {
        sink.imodv_scroll_wheel(event);
    }
}

/// Calls from this unit into its paired translation units.  The traits map one
/// for one to the C++ calls, so the OpenGL/event implementation can be driven
/// by the real winit/glutin backend without changing source semantics.
pub trait ImodvWindowSink {
    fn imodv_file_menu(&mut self, which: i32);
    fn imodv_edit_menu(&mut self, which: i32);
    fn imodv_view_menu(&mut self, which: i32);
    fn imodv_help_menu(&mut self, which: i32);
    fn imodv_key_press(&mut self, event: KeyEvent);
    fn imodv_key_release(&mut self, event: KeyEvent);
    fn imodv_mouse_press(&mut self, event: KeyEvent);
    fn imodv_mouse_release(&mut self, event: KeyEvent);
    fn imodv_mouse_move(&mut self, event: KeyEvent);
    fn imodv_scroll_wheel(&mut self, event: KeyEvent);
    fn imodv_movie_timeout(&mut self);
    fn imodv_initialize_gl(&mut self);
    fn imodv_paint_gl(&mut self);
    fn imodv_resize_gl(&mut self, width: i32, height: i32);
    fn imodv_quit(&mut self);
    fn imodv_rotate_model(&mut self, x: f32, y: f32, z: f32);
    fn imodv_control_change_steps(&mut self, delta: i32);
    fn imodv_control_start(&mut self);
    fn start_timer(&mut self, interval: i32) -> i32;
    fn kill_timer(&mut self, timer: i32);
    fn update_gl(&mut self);
    fn resize_window(&mut self, width: i32, height: i32);
    fn set_current_device_pixel_ratio(&mut self, ratio: f32);
    fn dialogs_hide(&mut self);
    fn dialogs_show(&mut self);
    fn dialogs_master_changed(&mut self);
    fn dialogs_activated(&mut self);
    fn app_lost_focus(&mut self);
}

/// Callback-free sink useful until the paired units are attached.
#[derive(Default)]
pub struct NullImodvWindowSink {
    next_timer: i32,
}
impl ImodvWindowSink for NullImodvWindowSink {
    fn imodv_file_menu(&mut self, _: i32) {}
    fn imodv_edit_menu(&mut self, _: i32) {}
    fn imodv_view_menu(&mut self, _: i32) {}
    fn imodv_help_menu(&mut self, _: i32) {}
    fn imodv_key_press(&mut self, _: KeyEvent) {}
    fn imodv_key_release(&mut self, _: KeyEvent) {}
    fn imodv_mouse_press(&mut self, _: KeyEvent) {}
    fn imodv_mouse_release(&mut self, _: KeyEvent) {}
    fn imodv_mouse_move(&mut self, _: KeyEvent) {}
    fn imodv_scroll_wheel(&mut self, _: KeyEvent) {}
    fn imodv_movie_timeout(&mut self) {}
    fn imodv_initialize_gl(&mut self) {}
    fn imodv_paint_gl(&mut self) {}
    fn imodv_resize_gl(&mut self, _: i32, _: i32) {}
    fn imodv_quit(&mut self) {}
    fn imodv_rotate_model(&mut self, _: f32, _: f32, _: f32) {}
    fn imodv_control_change_steps(&mut self, _: i32) {}
    fn imodv_control_start(&mut self) {}
    fn start_timer(&mut self, _: i32) -> i32 {
        self.next_timer += 1;
        self.next_timer
    }
    fn kill_timer(&mut self, _: i32) {}
    fn update_gl(&mut self) {}
    fn resize_window(&mut self, _: i32, _: i32) {}
    fn set_current_device_pixel_ratio(&mut self, _: f32) {}
    fn dialogs_hide(&mut self) {}
    fn dialogs_show(&mut self) {}
    fn dialogs_master_changed(&mut self) {}
    fn dialogs_activated(&mut self) {}
    fn app_lost_focus(&mut self) {}
}

/// `ImodvWindow` (`mv_window.h`), including the six possible GL widgets.
pub struct ImodvWindow {
    app: *mut ImodvApp,
    pub dbw: Option<ImodvGl>,
    pub dbalw: Option<ImodvGl>,
    pub sbw: Option<ImodvGl>,
    pub dbstw: Option<ImodvGl>,
    pub dbst_alw: Option<ImodvGl>,
    pub sbstw: Option<ImodvGl>,
    pub cur_glw: Option<usize>,
    pub actions: [Action; LAST_VMENU_ID],
    pub minimized: bool,
    pub device_pixel_ratio: f32,
    pub init_width: i32,
    pub init_height: i32,
    pub rotation_tool_open: bool,
    pub resize_tool_open: bool,
    pub num_key_entries: i32,
}

impl ImodvWindow {
    /// `ImodvWindow::ImodvWindow`; native construction is handled by the
    /// winit/glutin driving layer while this reproduces source state selection.
    pub fn new(app: &mut ImodvApp) -> Self {
        let mut window = Self {
            app,
            dbw: None,
            dbalw: None,
            sbw: None,
            dbstw: None,
            dbst_alw: None,
            sbstw: None,
            cur_glw: None,
            actions: std::array::from_fn(|_| Action {
                enabled: true,
                ..Action::default()
            }),
            minimized: false,
            device_pixel_ratio: 0.,
            init_width: 0,
            init_height: 0,
            rotation_tool_open: false,
            resize_tool_open: false,
            num_key_entries: 0,
        };
        window.actions[VFILE_MENU_LOAD].enabled = app.standalone != 0;
        window.actions[VFILE_MENU_SAVE].enabled = app.standalone != 0;
        window.actions[VFILE_MENU_SAVEAS].enabled = app.standalone != 0;
        window.actions[VVIEW_MENU_LOWRES] = Action {
            checkable: true,
            checked: app.lowres != 0,
            enabled: true,
            text: "Low Resolution".into(),
            shortcut: Some(Key::Character('R')),
        };
        window.actions[VVIEW_MENU_LABELS] = Action {
            checkable: true,
            checked: app.draw_labels != 0,
            enabled: true,
            text: "Point Labels".into(),
            shortcut: None,
        };
        window.actions[VVIEW_MENU_INVERTZ] = Action {
            checkable: true,
            checked: app.invert_z != 0,
            enabled: true,
            text: "Invert Z".into(),
            shortcut: None,
        };
        window.actions[VVIEW_MENU_LIGHTING] = Action {
            checkable: true,
            checked: app.lighting != 0,
            enabled: true,
            text: "Lighting".into(),
            shortcut: None,
        };
        window.actions[VVIEW_MENU_WIREFRAME] = Action {
            checkable: true,
            checked: app.wireframe != 0,
            enabled: true,
            text: "Wireframe".into(),
            shortcut: None,
        };
        // `numWidg` and `newOpenGLset` of the source constructor: the first
        // widget added is the current one, and on the `NEW_QTOPENGL` build the
        // single maximal request is the only widget made.
        let new_qt_open_gl = app.need_new_qglinit != 0;
        let mut num_widg = 0;
        let mut new_opengl_set = false;
        if app.enable_depth_db >= 0 {
            window.dbw =
                Some(window.add_gl_widget_to_stack(true, app.enable_depth_db, false, false));
            window.cur_glw = Some(0);
            num_widg += 1;
            new_opengl_set = new_qt_open_gl;
        }
        if app.enable_depth_dbal >= 0 {
            // The source passes `a->enableDepthDB` here, not `enableDepthDBal`.
            let gl = window.add_gl_widget_to_stack(true, app.enable_depth_db, false, true);
            if num_widg == 0 {
                window.cur_glw = Some(1);
                app.alpha_visual = 1;
            }
            window.dbalw = Some(gl);
            num_widg += 1;
            new_opengl_set = new_qt_open_gl;
        }
        if num_widg == 0 && app.enable_depth_dbst >= 0 {
            window.dbstw =
                Some(window.add_gl_widget_to_stack(true, app.enable_depth_dbst, true, false));
            window.cur_glw = Some(3);
            num_widg += 1;
            new_opengl_set = new_qt_open_gl;
        }
        // `|| a->needNewQGLinit`: on the `NEW_QTOPENGL` build every recorded
        // depth is -1 and this maximal stereo/alpha request is the one widget
        // made, whose granted format `initializeForNewQGL` then records.
        if num_widg == 0 && (app.enable_depth_dbst_al >= 0 || app.need_new_qglinit != 0) {
            window.dbst_alw =
                Some(window.add_gl_widget_to_stack(true, app.enable_depth_dbst_al, true, true));
            window.cur_glw = Some(4);
            app.alpha_visual = 1;
            num_widg += 1;
            new_opengl_set = new_qt_open_gl;
        }
        if app.enable_depth_sb >= 0 && !new_opengl_set {
            let gl = window.add_gl_widget_to_stack(false, app.enable_depth_sb, false, false);
            if num_widg == 0 {
                window.cur_glw = Some(2);
                app.dbl_buf = 0;
            }
            window.sbw = Some(gl);
            num_widg += 1;
            new_opengl_set = new_qt_open_gl;
        }
        if num_widg == 0 && app.enable_depth_sbst >= 0 {
            window.sbstw =
                Some(window.add_gl_widget_to_stack(true, app.enable_depth_sbst, true, false));
            window.cur_glw = Some(5);
            app.dbl_buf = 0;
            num_widg += 1;
            new_opengl_set = new_qt_open_gl;
        }
        let _ = (num_widg, new_opengl_set);
        app.db_possible = app.dbl_buf;
        if app.trans_bkgd != 0 {
            if app.dbl_buf != 0 && app.enable_depth_dbal >= 0 {
                window.cur_glw = Some(1);
                app.alpha_visual = 1;
            } else {
                app.trans_bkgd = 0;
            }
        }
        window.actions[VVIEW_MENU_DB].checked = app.dbl_buf > 0;
        window.actions[VVIEW_MENU_DB].checkable = true;
        window.actions[VVIEW_MENU_DB].enabled =
            app.dbl_buf > 0 && app.enable_depth_sb >= 0 && app.trans_bkgd == 0;
        window.actions[VVIEW_MENU_TRANSBKGD].checked = app.trans_bkgd > 0;
        window.actions[VVIEW_MENU_TRANSBKGD].checkable = true;
        window.actions[VVIEW_MENU_TRANSBKGD].enabled =
            app.dbl_buf > 0 && (app.enable_depth_dbal >= 0 || app.enable_depth_dbst_al >= 0);
        window
    }
    pub fn app(&self) -> &ImodvApp {
        unsafe { &*self.app }
    }
    pub fn app_mut(&mut self) -> &mut ImodvApp {
        unsafe { &mut *self.app }
    }
    /// `ImodvWindow::~ImodvWindow`.
    pub fn destroy(&mut self) {}
    /// `ImodvWindow::addGLWidgetToStack`.
    pub fn add_gl_widget_to_stack(
        &self,
        db: bool,
        enable_depth: i32,
        stereo: bool,
        alpha: bool,
    ) -> ImodvGl {
        ImodvGl::new(GlFormat {
            double_buffer: db,
            depth: enable_depth > 0,
            stereo,
            alpha,
            samples: true,
        })
    }
    /// `ImodvWindow::initializeForNewQGL`.
    ///
    /// The first maximally requested surface determines which visual was
    /// actually granted; the source then records it and creates the two
    /// non-stereo fallback visuals.
    pub fn initialize_for_new_qgl(&mut self) {
        let current = match self.cur_glw {
            Some(0) => self.dbw.as_ref(),
            Some(1) => self.dbalw.as_ref(),
            Some(2) => self.sbw.as_ref(),
            Some(3) => self.dbstw.as_ref(),
            Some(4) => self.dbst_alw.as_ref(),
            Some(5) => self.sbstw.as_ref(),
            _ => None,
        }
        .map(|widget| widget.format);
        let Some(format) = current else {
            return;
        };
        let mut current_widget = match self.cur_glw {
            Some(0) => self.dbw.take(),
            Some(1) => self.dbalw.take(),
            Some(2) => self.sbw.take(),
            Some(3) => self.dbstw.take(),
            Some(4) => self.dbst_alw.take(),
            Some(5) => self.sbstw.take(),
            _ => None,
        };
        self.dbst_alw = None;
        if format.stereo {
            if !format.double_buffer {
                self.app_mut().enable_depth_sbst = 1;
                self.sbstw = current_widget.take();
            } else if format.alpha {
                self.app_mut().enable_depth_dbst_al = 1;
                self.dbst_alw = current_widget.take();
            } else {
                self.app_mut().enable_depth_dbst = 1;
                self.dbstw = current_widget.take();
            }
        } else if format.alpha {
            self.app_mut().enable_depth_dbal = 1;
            self.dbalw = current_widget.take();
        } else {
            self.app_mut().enable_depth_db = 1;
            self.dbw = current_widget.take();
        }
        if self.app().enable_depth_dbal < 0 {
            self.dbalw = Some(self.add_gl_widget_to_stack(true, 1, false, true));
            self.cur_glw = Some(1);
            if self
                .dbalw
                .as_ref()
                .is_some_and(|widget| widget.format.alpha)
            {
                self.app_mut().enable_depth_dbal = 1;
            } else {
                self.app_mut().enable_depth_db = 1;
                self.dbw = self.dbalw.take();
                self.cur_glw = Some(0);
            }
        }
        if self.app().enable_depth_db < 0 {
            let gl = self.add_gl_widget_to_stack(true, 1, false, false);
            self.dbw = Some(gl);
            self.cur_glw = Some(0);
            self.app_mut().enable_depth_db = 1;
        }
        let alpha_visual = (self.app().enable_depth_db <= 0) as i32;
        let (enable_depth_sb, trans_bkgd, enable_depth_dbal, enable_depth_dbst_al) = {
            let app = self.app_mut();
            app.alpha_visual = alpha_visual;
            app.need_new_qglinit = 0;
            app.dbl_buf = 1;
            (
                app.enable_depth_sb,
                app.trans_bkgd,
                app.enable_depth_dbal,
                app.enable_depth_dbst_al,
            )
        };
        self.actions[VVIEW_MENU_DB].checked = true;
        self.actions[VVIEW_MENU_DB].enabled = enable_depth_sb >= 0 && trans_bkgd == 0;
        self.actions[VVIEW_MENU_TRANSBKGD].checked = trans_bkgd > 0;
        self.actions[VVIEW_MENU_TRANSBKGD].enabled =
            enable_depth_dbal >= 0 || enable_depth_dbst_al >= 0;
    }

    /// `ImodvWindow::fileMenuSlot`.
    pub fn file_menu_slot(&mut self, which: i32, sink: &mut dyn ImodvWindowSink) {
        sink.imodv_file_menu(which);
    }
    pub fn edit_menu_slot(&mut self, which: i32, sink: &mut dyn ImodvWindowSink) {
        sink.imodv_edit_menu(which);
    }
    pub fn view_menu_slot(&mut self, which: i32, sink: &mut dyn ImodvWindowSink) {
        sink.imodv_view_menu(which);
    }
    pub fn key_menu_slot(&mut self, event: KeyEvent, sink: &mut dyn ImodvWindowSink) {
        sink.imodv_key_press(event);
    }
    pub fn help_menu_slot(&mut self, which: i32, sink: &mut dyn ImodvWindowSink) {
        sink.imodv_help_menu(which);
    }
    pub fn set_checkable_item(&mut self, id: usize, state: bool) {
        if id > 0 && id < LAST_VMENU_ID {
            self.actions[id].checked = state;
        }
    }
    pub fn set_enabled_menu_item(&mut self, id: usize, state: bool) {
        if id > 0 && id < LAST_VMENU_ID {
            self.actions[id].enabled = state;
        }
    }
    /// `ImodvWindow::setGLWidget`.
    pub fn set_gl_widget(&mut self, db: bool, stereo: bool, alpha: bool) -> i32 {
        if stereo {
            let (depth_dbst, depth_dbst_al, depth_sbst) = {
                let app = self.app();
                (
                    app.enable_depth_dbst,
                    app.enable_depth_dbst_al,
                    app.enable_depth_sbst,
                )
            };
            if depth_dbst >= 0 && self.dbstw.is_none() {
                self.dbstw = Some(self.add_gl_widget_to_stack(true, depth_dbst, true, false));
            }
            if depth_dbst_al >= 0 && self.dbst_alw.is_none() {
                self.dbst_alw = Some(self.add_gl_widget_to_stack(true, depth_dbst_al, true, true));
            }
            if depth_sbst >= 0 && self.sbstw.is_none() {
                self.sbstw = Some(self.add_gl_widget_to_stack(false, depth_sbst, true, false));
            }
        }
        self.cur_glw = match (db, stereo, alpha) {
            (true, false, false) if self.dbw.is_some() => Some(0),
            (true, false, true) if self.dbalw.is_some() => Some(1),
            (false, false, _) if self.sbw.is_some() => Some(2),
            (true, true, false) if self.dbstw.is_some() => Some(3),
            (true, true, true) if self.dbst_alw.is_some() => Some(4),
            (false, true, _) if self.sbstw.is_some() => Some(5),
            _ => None,
        };
        if self.cur_glw.is_none() {
            return 1;
        }
        if !stereo {
            if self.dbw.is_some() || self.dbalw.is_some() {
                self.dbstw = None;
            }
            if self.dbw.is_some() || self.dbalw.is_some() || self.sbw.is_some() {
                self.sbstw = None;
            }
        }
        0
    }
    pub fn fake_resize(&mut self, width: i32, height: i32, sink: &mut dyn ImodvWindowSink) {
        sink.imodv_resize_gl(width, height);
    }
    pub fn app_focus_changed(&mut self, focused: bool, sink: &mut dyn ImodvWindowSink) {
        if !focused {
            sink.app_lost_focus();
        }
    }
    pub fn rotation_key_press(&mut self, event: KeyEvent, sink: &mut dyn ImodvWindowSink) {
        sink.imodv_key_press(event);
    }
    pub fn rotation_key_release(&mut self, event: KeyEvent, sink: &mut dyn ImodvWindowSink) {
        sink.imodv_key_release(event);
    }
    pub fn resizer_key_press(&mut self, event: KeyEvent, sink: &mut dyn ImodvWindowSink) {
        sink.imodv_key_press(event);
    }
    pub fn key_press_event(&mut self, event: KeyEvent, sink: &mut dyn ImodvWindowSink) {
        sink.imodv_key_press(event);
    }
    pub fn key_release_event(&mut self, event: KeyEvent, sink: &mut dyn ImodvWindowSink) {
        sink.imodv_key_release(event);
    }
    pub fn close_event(&mut self, sink: &mut dyn ImodvWindowSink) {
        sink.imodv_quit();
    }
    pub fn resize_event(&mut self, sink: &mut dyn ImodvWindowSink) {
        sink.dialogs_master_changed();
    }
    pub fn move_event(&mut self, sink: &mut dyn ImodvWindowSink) {
        sink.dialogs_master_changed();
    }
    pub fn change_event(&mut self, event: WindowEvent, sink: &mut dyn ImodvWindowSink) {
        if matches!(event, WindowEvent::ActivationChange { active: true }) {
            sink.dialogs_activated();
        }
    }
    /// `ImodvWindow::event`.
    pub fn event(&mut self, event: WindowEvent, sink: &mut dyn ImodvWindowSink) -> bool {
        match event {
            WindowEvent::ActivationChange { active: true } => sink.dialogs_activated(),
            WindowEvent::WindowStateChange {
                minimized: true, ..
            }
            | WindowEvent::Hide
                if !self.minimized =>
            {
                self.minimized = true;
                sink.dialogs_hide();
            }
            WindowEvent::WindowStateChange {
                minimized: false,
                maximized: true,
            }
            | WindowEvent::Show
                if self.minimized =>
            {
                self.minimized = false;
                sink.dialogs_show();
            }
            WindowEvent::Resize { width, height } => {
                self.init_width = width;
                self.init_height = height;
                self.resize_event(sink);
            }
            WindowEvent::Move => self.move_event(sink),
            WindowEvent::Close => self.close_event(sink),
            _ => {}
        }
        true
    }
    pub fn timeout_slot(&mut self, sink: &mut dyn ImodvWindowSink) {
        sink.imodv_movie_timeout();
    }
    pub fn open_rotation_tool(&mut self) {
        self.rotation_tool_open = true;
    }
    pub fn rotate_clicked(&mut self, dx: i32, dy: i32, dz: i32, sink: &mut dyn ImodvWindowSink) {
        let d = self.app().delta_rot;
        sink.imodv_rotate_model(-d * dx as f32, d * dy as f32, d * dz as f32);
    }
    pub fn rot_step_changed(&mut self, delta: i32, sink: &mut dyn ImodvWindowSink) {
        sink.imodv_control_change_steps(delta);
    }
    pub fn movie_but_toggled(&mut self, _state: bool, sink: &mut dyn ImodvWindowSink) {
        sink.imodv_control_start();
    }
    pub fn rotation_closing(&mut self) {
        self.rotation_tool_open = false;
    }
    pub fn open_resize_tool(&mut self) {
        self.resize_tool_open = true;
    }
    pub fn new_resizer_size(&mut self, size_x: i32, size_y: i32, sink: &mut dyn ImodvWindowSink) {
        let a = self.app_mut();
        let old = a.winx.min(a.winy).max(1) as f32;
        unsafe {
            if !a.imod.is_null() {
                let imod = &mut *a.imod;
                if let Some(view) = imod.view.get_mut(0) {
                    view.rad *= size_x.min(size_y) as f32 / old;
                }
            }
        }
        a.winx = size_x;
        a.winy = size_y;
        sink.resize_window(size_x, size_y);
    }
    pub fn resizer_closing(&mut self) {
        self.resize_tool_open = false;
    }
}

/// The compatibility-profile entry points that `glow` does not expose.
///
/// `glow` carries no fixed-function OpenGL at all: there is no `glPushName`,
/// `glRenderMode`, `glOrtho`, `glTranslatef` or `glMatrixMode` in it.  Those
/// are precisely the selection-buffer picking mechanism and the matrix stack
/// that `mv_ogl.cpp` and `mv_gfx.cpp` are written against, so they are loaded
/// straight from the GL display and called through their C signatures.  This
/// is the platform boundary Qt used to hold, not a second renderer.
#[cfg(feature = "three-dmod-gl")]
#[derive(Clone, Copy)]
pub struct MvGlCompatEntries {
    pub gl_push_name: unsafe extern "C" fn(u32),
    pub gl_pop_name: unsafe extern "C" fn(),
    pub gl_load_name: unsafe extern "C" fn(u32),
    pub gl_init_names: unsafe extern "C" fn(),
    pub gl_render_mode: unsafe extern "C" fn(u32) -> i32,
    pub gl_matrix_mode: unsafe extern "C" fn(u32),
    pub gl_load_identity: unsafe extern "C" fn(),
    pub gl_ortho: unsafe extern "C" fn(f64, f64, f64, f64, f64, f64),
    pub gl_frustum: unsafe extern "C" fn(f64, f64, f64, f64, f64, f64),
    pub gl_translatef: unsafe extern "C" fn(f32, f32, f32),
    pub gl_rotatef: unsafe extern "C" fn(f32, f32, f32, f32),
    pub gl_scalef: unsafe extern "C" fn(f32, f32, f32),
    pub gl_begin: unsafe extern "C" fn(u32),
    pub gl_end: unsafe extern "C" fn(),
    pub gl_vertex2f: unsafe extern "C" fn(f32, f32),
    pub gl_vertex2i: unsafe extern "C" fn(i32, i32),
    pub gl_color3ub: unsafe extern "C" fn(u8, u8, u8),
    pub gl_color4ub: unsafe extern "C" fn(u8, u8, u8, u8),
    pub gl_indexi: unsafe extern "C" fn(i32),
    pub gl_fogi: unsafe extern "C" fn(u32, i32),
    pub gl_fogf: unsafe extern "C" fn(u32, f32),
    pub gl_point_size: unsafe extern "C" fn(f32),
    pub gl_line_stipple: unsafe extern "C" fn(i32, u16),
    pub gl_raster_pos2f: unsafe extern "C" fn(f32, f32),
    pub gl_pixel_zoom: unsafe extern "C" fn(f32, f32),
    pub gl_draw_pixels: unsafe extern "C" fn(i32, i32, u32, u32, *const std::ffi::c_void),
    pub gl_get_floatv: unsafe extern "C" fn(u32, *mut f32),
    /// `glGetDoublev`, which `imodvUnprojectPickedPoint` reads the model-view
    /// and projection matrices with; `glow` has no double-precision getter.
    pub gl_get_doublev: unsafe extern "C" fn(u32, *mut f64),
    pub gl_get_integerv: unsafe extern "C" fn(u32, *mut i32),
    pub gl_read_pixels: unsafe extern "C" fn(i32, i32, i32, i32, u32, u32, *mut std::ffi::c_void),
    pub gl_vertex3f: unsafe extern "C" fn(f32, f32, f32),
    /// `glVertex3fv`, which is what the GLU tessellator's `GLU_VERTEX`
    /// callback is bound to; `glVertex3f` cannot stand in for it because GLU
    /// hands the callback the vertex pointer it was given.
    pub gl_vertex3fv: unsafe extern "C" fn(*const f32),
    pub gl_normal3f: unsafe extern "C" fn(f32, f32, f32),
    pub gl_color3f: unsafe extern "C" fn(f32, f32, f32),
    pub gl_color4f: unsafe extern "C" fn(f32, f32, f32, f32),
    pub gl_polygon_mode: unsafe extern "C" fn(u32, u32),
    pub gl_front_face: unsafe extern "C" fn(u32),
    pub gl_light_modeli: unsafe extern "C" fn(u32, i32),
    pub gl_light_modelfv: unsafe extern "C" fn(u32, *const f32),
    pub gl_lightf: unsafe extern "C" fn(u32, u32, f32),
    pub gl_lightfv: unsafe extern "C" fn(u32, u32, *const f32),
    pub gl_materialf: unsafe extern "C" fn(u32, u32, f32),
    pub gl_materialfv: unsafe extern "C" fn(u32, u32, *const f32),
    pub gl_push_matrix: unsafe extern "C" fn(),
    pub gl_pop_matrix: unsafe extern "C" fn(),
    pub gl_gen_lists: unsafe extern "C" fn(i32) -> u32,
    pub gl_new_list: unsafe extern "C" fn(u32, u32),
    pub gl_end_list: unsafe extern "C" fn(),
    pub gl_call_list: unsafe extern "C" fn(u32),
    pub gl_delete_lists: unsafe extern "C" fn(u32, i32),
    pub gl_clip_plane: unsafe extern "C" fn(u32, *const f64),
    /// `glPrimitiveRestartIndex` is OpenGL 3.1; `b3dgfx.cpp` only calls it when
    /// `B3DGLEXT_PRIM_RESTART` was detected, so a compatibility 2.1 context
    /// that lacks it is not an error at load time.
    pub gl_primitive_restart_index: Option<unsafe extern "C" fn(u32)>,
}

/// OpenGL enumerants used by this boundary, with their `GL/gl.h` values.  They
/// are written out because `glow` has no compatibility-profile constants.
#[cfg(feature = "three-dmod-gl")]
mod gl_enum {
    pub const FRONT_LEFT: u32 = 0x0400;
    pub const BACK_LEFT: u32 = 0x0402;
    pub const BACK_RIGHT: u32 = 0x0403;
    pub const FRONT: u32 = 0x0404;
    pub const BACK: u32 = 0x0405;
    pub const RIGHT: u32 = 0x0407;
    pub const DEPTH_TEST: u32 = 0x0B71;
    pub const NORMALIZE: u32 = 0x0BA1;
    pub const LINE_SMOOTH: u32 = 0x0B20;
    pub const LINE_STIPPLE: u32 = 0x0B24;
    pub const LINE_WIDTH: u32 = 0x0B21;
    pub const CURRENT_INDEX: u32 = 0x0B01;
    pub const MULTISAMPLE: u32 = 0x809D;
    pub const FOG_MODE: u32 = 0x0B65;
    pub const FOG_START: u32 = 0x0B63;
    pub const FOG_END: u32 = 0x0B64;
    pub const LINEAR: i32 = 0x2601;
    pub const SELECT: u32 = 0x1C02;
    pub const RENDER: u32 = 0x1C00;
    pub const PROJECTION: u32 = 0x1701;
    pub const MODELVIEW: u32 = 0x1700;
    pub const COLOR_BUFFER_BIT: u32 = 0x0000_4000;
    pub const DEPTH_BUFFER_BIT: u32 = 0x0000_0100;
    pub const RGB: u32 = 0x1907;
    pub const RGBA: u32 = 0x1908;
    pub const UNSIGNED_BYTE: u32 = 0x1401;
    pub const UNPACK_ALIGNMENT: u32 = 0x0CF5;
    pub const VERSION: u32 = 0x1F02;
    pub const LINES: u32 = 0x0001;
    pub const LINE_STRIP: u32 = 0x0003;
    pub const FRONT_AND_BACK: u32 = 0x0408;
    pub const LINE: u32 = 0x1B01;
    pub const FILL: u32 = 0x1B02;
    pub const CW: u32 = 0x0900;
    pub const CCW: u32 = 0x0901;
    pub const BLEND: u32 = 0x0BE2;
    pub const CULL_FACE: u32 = 0x0B44;
    pub const LIGHTING: u32 = 0x0B50;
    pub const LIGHT0: u32 = 0x4000;
    pub const COLOR_MATERIAL: u32 = 0x0B57;
    pub const LIGHT_MODEL_LOCAL_VIEWER: u32 = 0x0B51;
    pub const LIGHT_MODEL_TWO_SIDE: u32 = 0x0B52;
    pub const LIGHT_MODEL_AMBIENT: u32 = 0x0B53;
    pub const AMBIENT: u32 = 0x1200;
    pub const DIFFUSE: u32 = 0x1201;
    pub const SPECULAR: u32 = 0x1202;
    pub const POSITION: u32 = 0x1203;
    pub const SHININESS: u32 = 0x1601;
    pub const CONSTANT_ATTENUATION: u32 = 0x1207;
    pub const LINEAR_ATTENUATION: u32 = 0x1208;
    pub const QUADRATIC_ATTENUATION: u32 = 0x1209;
    pub const COMPILE: u32 = 0x1300;
    pub const MAX_CLIP_PLANES: u32 = 0x0D32;
    pub const DEPTH_COMPONENT: u32 = 0x1902;
    pub const FLOAT: u32 = 0x1406;
    pub const MODELVIEW_MATRIX: u32 = 0x0BA6;
    pub const PROJECTION_MATRIX: u32 = 0x0BA7;
    pub const VIEWPORT: u32 = 0x0BA2;
    pub const CLIP_PLANE0: u32 = 0x3000;
}

/// The GLU entry points `mv_ogl.cpp` and `utilities.cpp` draw spheres and
/// tessellated polygons with.  GLU is a separate shared library from the GL the
/// display's `get_proc_address` serves, so it is opened by name.
#[cfg(feature = "three-dmod-gl")]
mod glu_enum {
    pub const POINT: u32 = 100010;
    pub const LINE: u32 = 100011;
    pub const FILL: u32 = 100012;
    pub const TESS_BEGIN: u32 = 100100;
    pub const TESS_VERTEX: u32 = 100101;
    pub const TESS_END: u32 = 100102;
}

/// `gluNewQuadric`, `gluSphere` and the polygon tessellator, resolved from
/// `libGLU`.
#[cfg(feature = "three-dmod-gl")]
#[derive(Clone, Copy)]
pub struct MvGluEntries {
    pub glu_new_quadric: unsafe extern "C" fn() -> *mut std::ffi::c_void,
    pub glu_quadric_draw_style: unsafe extern "C" fn(*mut std::ffi::c_void, u32),
    pub glu_sphere: unsafe extern "C" fn(*mut std::ffi::c_void, f64, i32, i32),
    pub glu_new_tess: unsafe extern "C" fn() -> *mut std::ffi::c_void,
    pub glu_delete_tess: unsafe extern "C" fn(*mut std::ffi::c_void),
    pub glu_tess_callback:
        unsafe extern "C" fn(*mut std::ffi::c_void, u32, Option<unsafe extern "C" fn()>),
    pub glu_begin_polygon: unsafe extern "C" fn(*mut std::ffi::c_void),
    pub glu_end_polygon: unsafe extern "C" fn(*mut std::ffi::c_void),
    pub glu_tess_begin_polygon: unsafe extern "C" fn(*mut std::ffi::c_void, *mut std::ffi::c_void),
    pub glu_tess_begin_contour: unsafe extern "C" fn(*mut std::ffi::c_void),
    pub glu_tess_vertex:
        unsafe extern "C" fn(*mut std::ffi::c_void, *mut f64, *mut std::ffi::c_void),
    pub glu_tess_end_contour: unsafe extern "C" fn(*mut std::ffi::c_void),
    pub glu_tess_end_polygon: unsafe extern "C" fn(*mut std::ffi::c_void),
    /// `gluUnProject`, the unprojection `imodvUnprojectPickedPoint` turns the
    /// clicked window pixel and its depth into model coordinates with.
    pub glu_un_project: unsafe extern "C" fn(
        f64,
        f64,
        f64,
        *const f64,
        *const f64,
        *const i32,
        *mut f64,
        *mut f64,
        *mut f64,
    ) -> i32,
}

/// The concrete compatibility-OpenGL boundary for `mv_gfx.cpp`, `mv_ogl.cpp`
/// and `b3dgfx.cpp`.
///
/// In the original, `ImodvGL` (a `QGLWidget`/`QOpenGLWidget`) owns the context
/// and every `gl*` call in those units runs on whatever context Qt has made
/// current.  There is no Qt here, so this type owns the glutin surface and
/// context and *is* what "the current context" means: `imodv_winset` ->
/// `make_current` and `imodv_swapbuffers` -> `swap_buffers` are its methods,
/// exactly as they are widget methods in the source.
#[cfg(feature = "three-dmod-gl")]
pub struct ImodvNativeGl {
    /// Everything `glow` does cover goes through here.
    pub gl: glow::Context,
    /// Everything it does not.
    pub compat: MvGlCompatEntries,
    pub surface: glutin::surface::Surface<glutin::surface::WindowSurface>,
    pub context: glutin::context::PossiblyCurrentContext,
    /// `mv_ogl.cpp` keeps these as file statics; the translated unit passes the
    /// struct explicitly, so the boundary that calls `imodvDraw_models` holds
    /// it for the same lifetime the statics have.
    pub state: crate::imod::three_dmod::mv_ogl::MvOglState,
    /// Everything `libGLU` covers: `gluSphere` and the tessellator.
    pub glu: MvGluEntries,
    /// The file-static `qobj` of `imodvDraw_spheres`.
    quadric: *mut std::ffi::c_void,
    /// The `sTessel` of `utilities.cpp`, created by `setupFilledContTesselator`.
    filled_tessel: *mut std::ffi::c_void,
    /// `mv_light.cpp`'s `Imodv_light_*` statics.
    pub light: crate::imod::three_dmod::mv_light::LightState,
    /// `a->mainWin->mDevicePixelRatio` and `a->mainWin->fontMetrics().height()`.
    pub device_pixel_ratio: f32,
    pub font_height: i32,
    version: String,
    /// `QObject::startTimer`/`killTimer` on the `ImodvGL` widget and the
    /// `QTimer` on `ImodvWindow`.  Qt's event dispatcher owns this table; the
    /// loop in `run_native_opengl` is that dispatcher here, so it fires the
    /// entries that have come due.  Each entry is the timer id, its interval,
    /// the last time it fired, and which object it belongs to: `false` for the
    /// GL widget's `timerEvent`, `true` for `ImodvWindow::timeoutSlot`.
    pub timers: Vec<(i32, std::time::Duration, std::time::Instant, bool)>,
    pub next_timer_id: i32,
    /// The window the surface was created for.  In the source this is the
    /// `QWidget` the `ImodvGL` widget lives in, and `updateGL`, `resize` and
    /// `setWindowTitle` are its methods.
    pub window: std::rc::Rc<winit::window::Window>,
}

#[cfg(feature = "three-dmod-gl")]
impl ImodvNativeGl {
    /// Resolves the fixed-function entry points against a current context.
    ///
    /// This constructor has no counterpart in the C++: Qt linked the GL library
    /// directly, and the loader is the language/platform boundary that replaces
    /// that link step.  It must be called with the context current.
    pub fn new(
        gl: glow::Context,
        surface: glutin::surface::Surface<glutin::surface::WindowSurface>,
        context: glutin::context::PossiblyCurrentContext,
        window: std::rc::Rc<winit::window::Window>,
        loader: &dyn Fn(&std::ffi::CStr) -> *const std::ffi::c_void,
    ) -> Result<Self, String> {
        macro_rules! entry {
            ($name:literal, $signature:ty) => {{
                let symbol = std::ffi::CString::new($name).expect("literal entry-point name");
                let address = loader(&symbol);
                if address.is_null() {
                    return Err(format!(
                        "3dmodv: the OpenGL driver does not provide {}, which the fixed-function \
                         drawing in mv_ogl.cpp requires; a compatibility profile context is needed",
                        $name
                    ));
                }
                unsafe { std::mem::transmute::<*const std::ffi::c_void, $signature>(address) }
            }};
        }
        let optional_restart = {
            let symbol = std::ffi::CString::new("glPrimitiveRestartIndex").expect("literal");
            let address = loader(&symbol);
            if address.is_null() {
                None
            } else {
                Some(unsafe {
                    std::mem::transmute::<*const std::ffi::c_void, unsafe extern "C" fn(u32)>(
                        address,
                    )
                })
            }
        };
        let compat = MvGlCompatEntries {
            gl_push_name: entry!("glPushName", unsafe extern "C" fn(u32)),
            gl_pop_name: entry!("glPopName", unsafe extern "C" fn()),
            gl_load_name: entry!("glLoadName", unsafe extern "C" fn(u32)),
            gl_init_names: entry!("glInitNames", unsafe extern "C" fn()),
            gl_render_mode: entry!("glRenderMode", unsafe extern "C" fn(u32) -> i32),
            gl_matrix_mode: entry!("glMatrixMode", unsafe extern "C" fn(u32)),
            gl_load_identity: entry!("glLoadIdentity", unsafe extern "C" fn()),
            gl_ortho: entry!(
                "glOrtho",
                unsafe extern "C" fn(f64, f64, f64, f64, f64, f64)
            ),
            gl_frustum: entry!(
                "glFrustum",
                unsafe extern "C" fn(f64, f64, f64, f64, f64, f64)
            ),
            gl_translatef: entry!("glTranslatef", unsafe extern "C" fn(f32, f32, f32)),
            gl_rotatef: entry!("glRotatef", unsafe extern "C" fn(f32, f32, f32, f32)),
            gl_scalef: entry!("glScalef", unsafe extern "C" fn(f32, f32, f32)),
            gl_begin: entry!("glBegin", unsafe extern "C" fn(u32)),
            gl_end: entry!("glEnd", unsafe extern "C" fn()),
            gl_vertex2f: entry!("glVertex2f", unsafe extern "C" fn(f32, f32)),
            gl_vertex2i: entry!("glVertex2i", unsafe extern "C" fn(i32, i32)),
            gl_color3ub: entry!("glColor3ub", unsafe extern "C" fn(u8, u8, u8)),
            gl_color4ub: entry!("glColor4ub", unsafe extern "C" fn(u8, u8, u8, u8)),
            gl_indexi: entry!("glIndexi", unsafe extern "C" fn(i32)),
            gl_fogi: entry!("glFogi", unsafe extern "C" fn(u32, i32)),
            gl_fogf: entry!("glFogf", unsafe extern "C" fn(u32, f32)),
            gl_point_size: entry!("glPointSize", unsafe extern "C" fn(f32)),
            gl_line_stipple: entry!("glLineStipple", unsafe extern "C" fn(i32, u16)),
            gl_raster_pos2f: entry!("glRasterPos2f", unsafe extern "C" fn(f32, f32)),
            gl_pixel_zoom: entry!("glPixelZoom", unsafe extern "C" fn(f32, f32)),
            gl_draw_pixels: entry!(
                "glDrawPixels",
                unsafe extern "C" fn(i32, i32, u32, u32, *const std::ffi::c_void)
            ),
            gl_get_floatv: entry!("glGetFloatv", unsafe extern "C" fn(u32, *mut f32)),
            gl_get_doublev: entry!("glGetDoublev", unsafe extern "C" fn(u32, *mut f64)),
            gl_get_integerv: entry!("glGetIntegerv", unsafe extern "C" fn(u32, *mut i32)),
            gl_read_pixels: entry!(
                "glReadPixels",
                unsafe extern "C" fn(i32, i32, i32, i32, u32, u32, *mut std::ffi::c_void)
            ),
            gl_vertex3f: entry!("glVertex3f", unsafe extern "C" fn(f32, f32, f32)),
            gl_vertex3fv: entry!("glVertex3fv", unsafe extern "C" fn(*const f32)),
            gl_normal3f: entry!("glNormal3f", unsafe extern "C" fn(f32, f32, f32)),
            gl_color3f: entry!("glColor3f", unsafe extern "C" fn(f32, f32, f32)),
            gl_color4f: entry!("glColor4f", unsafe extern "C" fn(f32, f32, f32, f32)),
            gl_polygon_mode: entry!("glPolygonMode", unsafe extern "C" fn(u32, u32)),
            gl_front_face: entry!("glFrontFace", unsafe extern "C" fn(u32)),
            gl_light_modeli: entry!("glLightModeli", unsafe extern "C" fn(u32, i32)),
            gl_light_modelfv: entry!("glLightModelfv", unsafe extern "C" fn(u32, *const f32)),
            gl_lightf: entry!("glLightf", unsafe extern "C" fn(u32, u32, f32)),
            gl_lightfv: entry!("glLightfv", unsafe extern "C" fn(u32, u32, *const f32)),
            gl_materialf: entry!("glMaterialf", unsafe extern "C" fn(u32, u32, f32)),
            gl_materialfv: entry!("glMaterialfv", unsafe extern "C" fn(u32, u32, *const f32)),
            gl_push_matrix: entry!("glPushMatrix", unsafe extern "C" fn()),
            gl_pop_matrix: entry!("glPopMatrix", unsafe extern "C" fn()),
            gl_gen_lists: entry!("glGenLists", unsafe extern "C" fn(i32) -> u32),
            gl_new_list: entry!("glNewList", unsafe extern "C" fn(u32, u32)),
            gl_end_list: entry!("glEndList", unsafe extern "C" fn()),
            gl_call_list: entry!("glCallList", unsafe extern "C" fn(u32)),
            gl_delete_lists: entry!("glDeleteLists", unsafe extern "C" fn(u32, i32)),
            gl_clip_plane: entry!("glClipPlane", unsafe extern "C" fn(u32, *const f64)),
            gl_primitive_restart_index: optional_restart,
        };
        // GLU lives in its own shared library, which the GL display's
        // `get_proc_address` does not serve, so it is opened by name.  The
        // fixed-function sphere and polygon tessellation of `mv_ogl.cpp` have
        // no other implementation, so a missing entry point is an error.
        let glu_handle =
            unsafe { libc::dlopen(c"libGLU.so.1".as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        if glu_handle.is_null() {
            return Err(
                "3dmodv: libGLU.so.1 could not be opened, and gluSphere and the polygon \
                 tessellator in mv_ogl.cpp require it"
                    .to_string(),
            );
        }
        macro_rules! glu_entry {
            ($name:literal, $signature:ty) => {{
                let symbol = std::ffi::CString::new($name).expect("literal entry-point name");
                let address = unsafe { libc::dlsym(glu_handle, symbol.as_ptr()) };
                if address.is_null() {
                    return Err(format!(
                        "3dmodv: libGLU does not provide {}, which the fixed-function drawing \
                         in mv_ogl.cpp requires",
                        $name
                    ));
                }
                unsafe { std::mem::transmute::<*mut std::ffi::c_void, $signature>(address) }
            }};
        }
        let glu = MvGluEntries {
            glu_new_quadric: glu_entry!(
                "gluNewQuadric",
                unsafe extern "C" fn() -> *mut std::ffi::c_void
            ),
            glu_quadric_draw_style: glu_entry!(
                "gluQuadricDrawStyle",
                unsafe extern "C" fn(*mut std::ffi::c_void, u32)
            ),
            glu_sphere: glu_entry!(
                "gluSphere",
                unsafe extern "C" fn(*mut std::ffi::c_void, f64, i32, i32)
            ),
            glu_new_tess: glu_entry!(
                "gluNewTess",
                unsafe extern "C" fn() -> *mut std::ffi::c_void
            ),
            glu_delete_tess: glu_entry!(
                "gluDeleteTess",
                unsafe extern "C" fn(*mut std::ffi::c_void)
            ),
            glu_tess_callback: glu_entry!(
                "gluTessCallback",
                unsafe extern "C" fn(*mut std::ffi::c_void, u32, Option<unsafe extern "C" fn()>)
            ),
            glu_begin_polygon: glu_entry!(
                "gluBeginPolygon",
                unsafe extern "C" fn(*mut std::ffi::c_void)
            ),
            glu_end_polygon: glu_entry!(
                "gluEndPolygon",
                unsafe extern "C" fn(*mut std::ffi::c_void)
            ),
            glu_tess_begin_polygon: glu_entry!(
                "gluTessBeginPolygon",
                unsafe extern "C" fn(*mut std::ffi::c_void, *mut std::ffi::c_void)
            ),
            glu_tess_begin_contour: glu_entry!(
                "gluTessBeginContour",
                unsafe extern "C" fn(*mut std::ffi::c_void)
            ),
            glu_tess_vertex: glu_entry!(
                "gluTessVertex",
                unsafe extern "C" fn(*mut std::ffi::c_void, *mut f64, *mut std::ffi::c_void)
            ),
            glu_tess_end_contour: glu_entry!(
                "gluTessEndContour",
                unsafe extern "C" fn(*mut std::ffi::c_void)
            ),
            glu_tess_end_polygon: glu_entry!(
                "gluTessEndPolygon",
                unsafe extern "C" fn(*mut std::ffi::c_void)
            ),
            glu_un_project: glu_entry!(
                "gluUnProject",
                unsafe extern "C" fn(
                    f64,
                    f64,
                    f64,
                    *const f64,
                    *const f64,
                    *const i32,
                    *mut f64,
                    *mut f64,
                    *mut f64,
                ) -> i32
            ),
        };
        Ok(Self {
            gl,
            compat,
            surface,
            context,
            state: crate::imod::three_dmod::mv_ogl::MvOglState::default(),
            glu,
            quadric: std::ptr::null_mut(),
            filled_tessel: std::ptr::null_mut(),
            light: crate::imod::three_dmod::mv_light::LightState::default(),
            device_pixel_ratio: 1.,
            font_height: 0,
            version: String::new(),
            timers: Vec::new(),
            next_timer_id: 0,
            window,
        })
    }
}

#[cfg(feature = "three-dmod-gl")]
impl crate::imod::three_dmod::mv_gfx::ImodvGfxGl for ImodvNativeGl {
    /// `b3dInitializeGL()` (`b3dgfx.cpp:135`), for `imodvPaintGL`'s first-time
    /// probe (`mv_gfx.cpp:234`).  The state is transient, as it is at the
    /// `b3d_line_width`/`b3d_point_size` routes in this file: the source keeps the
    /// answer in `a->glExtFlags`, which is what the caller stores.
    fn initialize_gl_extensions(&mut self) -> i32 {
        let mut state = crate::imod::three_dmod::b3dgfx::B3dGfxState::default();
        crate::imod::three_dmod::b3dgfx::b3d_initialize_gl(&mut state, self)
    }
    /// `a->mainWin->mCurGLw->makeCurrent()`.  Qt's `makeCurrent` has no error
    /// return, so a failure here is reported once and drawing proceeds as the
    /// source does.
    fn make_current(&mut self) {
        use glutin::prelude::PossiblyCurrentGlContext;
        if self.context.make_current(&self.surface).is_err() {
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| eprintln!("3dmodv: could not make the OpenGL context current"));
        }
    }
    /// `a->mainWin->mCurGLw->swapBuffers()`.
    fn swap_buffers(&mut self) {
        use glutin::surface::GlSurface;
        if self.surface.swap_buffers(&self.context).is_err() {
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| eprintln!("3dmodv: could not swap the OpenGL buffers"));
        }
    }
    fn flush(&mut self) {
        unsafe { glow::HasContext::flush(&self.gl) }
    }
    fn finish(&mut self) {
        unsafe { glow::HasContext::finish(&self.gl) }
    }
    /// `glClearColor` followed by `glClear`.
    fn clear(&mut self, red: f32, green: f32, blue: f32, alpha: f32, depth: bool) {
        unsafe {
            glow::HasContext::clear_color(&self.gl, red, green, blue, alpha);
            glow::HasContext::clear(
                &self.gl,
                gl_enum::COLOR_BUFFER_BIT | if depth { gl_enum::DEPTH_BUFFER_BIT } else { 0 },
            );
        }
    }
    /// `glDrawBuffer(a->dblBuf ? GL_BACK_RIGHT : GL_RIGHT)` and its partner
    /// `glDrawBuffer(a->dblBuf ? GL_BACK : GL_FRONT)`.
    fn draw_buffer(&mut self, right: bool, back: bool) {
        let buffer = match (right, back) {
            (true, true) => gl_enum::BACK_RIGHT,
            (true, false) => gl_enum::RIGHT,
            (false, true) => gl_enum::BACK,
            (false, false) => gl_enum::FRONT,
        };
        unsafe { glow::HasContext::draw_buffer(&self.gl, buffer) }
    }
    /// `glViewport(0, 0, a->winx, a->winy)`.
    fn viewport(&mut self, width: i32, height: i32) {
        unsafe { glow::HasContext::viewport(&self.gl, 0, 0, width, height) }
    }
    /// The body of `imodvInitializeGL`.
    fn initialize(&mut self, fog_end: f32, double_buffer: bool) {
        unsafe {
            glow::HasContext::clear_color(&self.gl, 0.0, 0.0, 0.0, 0.0);
            glow::HasContext::clear(
                &self.gl,
                gl_enum::COLOR_BUFFER_BIT | gl_enum::DEPTH_BUFFER_BIT,
            );
            glow::HasContext::enable(&self.gl, gl_enum::NORMALIZE);
            glow::HasContext::enable(&self.gl, gl_enum::DEPTH_TEST);
            (self.compat.gl_fogi)(gl_enum::FOG_MODE, gl_enum::LINEAR);
            (self.compat.gl_fogf)(gl_enum::FOG_START, 0.0);
            (self.compat.gl_fogf)(gl_enum::FOG_END, fog_end);
            if !double_buffer {
                glow::HasContext::enable(&self.gl, gl_enum::LINE_SMOOTH);
            }
        }
        // `imodvInitializeGL` ends with `light_init()`.  The translated
        // `imodv_initialize_gl` does not carry that call, and `light_init`
        // needs the `LightState` and `Iview` that `mv_light.rs` takes; it
        // belongs in `imodv_initialize_gl`, not in this boundary.
    }
    /// `glRenderMode(GL_SELECT)`.
    fn render_mode_select(&mut self) -> i32 {
        unsafe { (self.compat.gl_render_mode)(gl_enum::SELECT) }
    }
    /// `glRenderMode(GL_RENDER)`.
    fn render_mode_render(&mut self) -> i32 {
        unsafe { (self.compat.gl_render_mode)(gl_enum::RENDER) }
    }
    /// `glInitNames`.
    fn init_names(&mut self) {
        unsafe { (self.compat.gl_init_names)() }
    }
    /// `glEnable(GL_MULTISAMPLE)`.
    fn multisample(&mut self, enabled: bool) {
        unsafe {
            if enabled {
                glow::HasContext::enable(&self.gl, gl_enum::MULTISAMPLE)
            } else {
                glow::HasContext::disable(&self.gl, gl_enum::MULTISAMPLE)
            }
        }
    }
    /// `imodvDraw_models(a)`.
    fn draw_models(&mut self, app: &mut crate::imod::three_dmod::imodv::ImodvApp) {
        // The static drawing state is moved out for the call because the
        // translated entry point takes it beside the boundary it drives.
        let mut state = std::mem::take(&mut self.state);
        crate::imod::three_dmod::mv_ogl::imodv_draw_models(&mut state, app, self);
        self.state = state;
    }
    /// `a->vbManager->clearTempArrays()`.
    fn clear_temp_arrays(&mut self, app: &mut crate::imod::three_dmod::imodv::ImodvApp) {
        // `ImodvApp::vb_manager` is declared in `imodv.rs` as the opaque
        // ownership-boundary pointer the C++ `VertBufManager *` stands for, so
        // it is cast back to the translated manager exactly as the source
        // dereferences it.
        let manager = app.vb_manager as *mut crate::imod::three_dmod::vertexbuffer::VertBufManager;
        if let Some(manager) = unsafe { manager.as_mut() } {
            manager.clear_temp_arrays();
        }
    }
    /// `b3dResizeViewportXY(a->winx, a->winy)`.
    fn resize_viewport_xy(&mut self, width: i32, height: i32) {
        crate::imod::three_dmod::b3dgfx::b3d_resize_viewport_xy(self, width, height);
    }
    /// Static `drawLightVector` (`mv_gfx.cpp:335`).
    fn draw_light_vector(
        &mut self,
        app: &crate::imod::three_dmod::imodv::ImodvApp,
        light: crate::imod::libimod::imodel::Ipoint,
    ) {
        use crate::imod::libimod::imat::{B3D_X, B3D_Y, imod_mat_rot, imod_mat_transform3d};
        use crate::imod::libimod::imodel::Ipoint;
        use crate::imod::libimod::ipoint::imod_point_normalize;
        let winhalf = 0.5 * app.winx.min(app.winy) as f32;
        let radfrac = winhalf * 0.9;
        let del = 10.0f32;
        let small_val = 1.0e-4f32;
        /* RADIANS_PER_DEGREE (`b3dutil.h:68`) */
        let rpd = 0.01745329252f64;
        let Some(mut mat) = crate::imod::libimod::imat::imod_mat_new(3) else {
            return;
        };
        let depth_enabled = unsafe { glow::HasContext::is_enabled(&self.gl, gl_enum::DEPTH_TEST) };
        if depth_enabled {
            unsafe { glow::HasContext::disable(&self.gl, gl_enum::DEPTH_TEST) };
        }
        let mut pnt = light;
        imod_point_normalize(&mut pnt);
        let mut cen = Ipoint::default();
        unsafe {
            (self.compat.gl_color4ub)(255, 0, 0, 255);
            (self.compat.gl_begin)(gl_enum::LINE_STRIP);
            (self.compat.gl_vertex2f)(
                (0.5 * app.winx as f64) as f32,
                (0.5 * app.winy as f64) as f32,
            );
            cen.x = ((pnt.x * radfrac) as f64 + 0.5 * app.winx as f64) as f32;
            cen.y = ((pnt.y * radfrac) as f64 + 0.5 * app.winy as f64) as f32;
            (self.compat.gl_vertex2f)(cen.x, cen.y);
            (self.compat.gl_end)();
        }
        let mut an = Ipoint::default();
        an.y = 0.0;
        if pnt.x > small_val || pnt.z > small_val || pnt.x < -small_val || pnt.z < -small_val {
            an.y = -(pnt.x as f64).atan2(pnt.z as f64) as f32;
        }
        let mut val = an.y as f64;
        val = pnt.z as f64 * val.cos() - pnt.x as f64 * val.sin();
        an.x = (90. * rpd - val.atan2(pnt.y as f64)) as f32;
        imod_mat_rot(&mut mat, -(an.x as f64) / rpd, B3D_X);
        imod_mat_rot(&mut mat, -(an.y as f64) / rpd, B3D_Y);
        let mut ar = Ipoint::default();
        an.z = 0.;
        an.x = del;
        an.y = del;
        imod_mat_transform3d(&mat, &an, &mut ar);
        unsafe {
            (self.compat.gl_begin)(gl_enum::LINE_STRIP);
            (self.compat.gl_vertex2f)(cen.x - ar.x, cen.y - ar.y);
            (self.compat.gl_vertex2f)(cen.x + ar.x, cen.y + ar.y);
            (self.compat.gl_end)();
        }
        an.y = -del;
        imod_mat_transform3d(&mat, &an, &mut ar);
        unsafe {
            (self.compat.gl_begin)(gl_enum::LINE_STRIP);
            (self.compat.gl_vertex2f)(cen.x - ar.x, cen.y - ar.y);
            (self.compat.gl_vertex2f)(cen.x + ar.x, cen.y + ar.y);
            (self.compat.gl_end)();
        }
        if depth_enabled {
            unsafe { glow::HasContext::enable(&self.gl, gl_enum::DEPTH_TEST) };
        }
    }
    /// `scaleBarDraw(a->winx, a->winy, scale, color, ...)`.
    ///
    /// Not implementable from here: the translated `scale_bar_draw` runs
    /// against `ScaleBarNativeBoundary`, whose members are the scale-bar
    /// dialog, the image pixel size and units, and the other windows' bar
    /// sizes.  That is the display host of handover steps 5 and 6, not a GL
    /// entry point.  Returns the source's own "nothing drawn" value.
    fn draw_scale_bar(
        &mut self,
        _app: &crate::imod::three_dmod::imodv::ImodvApp,
        _scale: f32,
        _color: i32,
    ) -> f32 {
        static WARNED: std::sync::Once = std::sync::Once::new();
        WARNED.call_once(|| {
            eprintln!(
                "3dmodv: no scale bar drawn: scalebar.cpp needs a ScaleBarNativeBoundary host"
            )
        });
        -1.
    }
    /// `glReadPixels(xoffset, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, ...)`
    /// followed by `glFlush`.
    fn read_rgb_pixels(&mut self, x: i32, width: i32, height: i32) -> Vec<u8> {
        let mut pixels = vec![0u8; (width.max(0) as usize) * (height.max(0) as usize) * 3];
        unsafe {
            glow::HasContext::read_pixels(
                &self.gl,
                x,
                0,
                width,
                height,
                gl_enum::RGB,
                gl_enum::UNSIGNED_BYTE,
                glow::PixelPackData::Slice(Some(&mut pixels)),
            );
            glow::HasContext::flush(&self.gl);
        }
        pixels
    }
}

#[cfg(feature = "three-dmod-gl")]
impl crate::imod::three_dmod::finegrain::FinegrainRenderBoundary for ImodvNativeGl {
    /// `glColor3f`.
    fn color3f(&mut self, red: f32, green: f32, blue: f32) {
        unsafe { (self.compat.gl_color3f)(red, green, blue) }
    }
    /// `glColor4f`.
    fn color4f(&mut self, red: f32, green: f32, blue: f32, alpha: f32) {
        unsafe { (self.compat.gl_color4f)(red, green, blue, alpha) }
    }
    /// `b3dLineWidth(width, obj)`.
    fn line_width(&mut self, width: i32, object: &crate::imod::libimod::imodel::Iobj) {
        let scale = object.flags & crate::imod::libimod::iobj::IMOD_OBJFLAG_SCALE_WDTH != 0;
        let state = crate::imod::three_dmod::b3dgfx::B3dGfxState::default();
        crate::imod::three_dmod::b3dgfx::b3d_line_width(&state, self, width, scale);
    }
    /// `b3dPointSize(size, obj)`.
    fn point_size(&mut self, size: i32, object: &crate::imod::libimod::imodel::Iobj) {
        let scale = object.flags & crate::imod::libimod::iobj::IMOD_OBJFLAG_SCALE_WDTH != 0;
        let state = crate::imod::three_dmod::b3dgfx::B3dGfxState::default();
        crate::imod::three_dmod::b3dgfx::b3d_point_size(&state, self, size, scale);
    }
    /// `light_adjust(obj, red, green, blue, trans)` (`mv_light.cpp:225`).
    fn light_adjust(
        &mut self,
        object: &crate::imod::libimod::imodel::Iobj,
        red: f32,
        green: f32,
        blue: f32,
        trans: i32,
    ) {
        crate::imod::three_dmod::mv_light::light_adjust(object, red, green, blue, trans, self);
    }
    /// `App->rgba`, which the model view always is.
    fn rgba(&self) -> bool {
        true
    }
}

/// `mv_light.cpp`'s compatibility-profile calls.
#[cfg(feature = "three-dmod-gl")]
impl crate::imod::three_dmod::mv_light::LightGl for ImodvNativeGl {
    /// The three `glLightf(GL_LIGHT0 + light, GL_*_ATTENUATION, ...)` calls.
    fn light_attenuation(&mut self, light: i32, constant: f32, linear: f32, quadratic: f32) {
        unsafe {
            let name = gl_enum::LIGHT0 + light.max(0) as u32;
            (self.compat.gl_lightf)(name, gl_enum::CONSTANT_ATTENUATION, constant);
            (self.compat.gl_lightf)(name, gl_enum::LINEAR_ATTENUATION, linear);
            (self.compat.gl_lightf)(name, gl_enum::QUADRATIC_ATTENUATION, quadratic);
        }
    }
    /// `glLightfv(GL_LIGHT0 + light, GL_AMBIENT, ambient)`.
    fn light_ambient(&mut self, light: i32, ambient: [f32; 4]) {
        unsafe {
            (self.compat.gl_lightfv)(
                gl_enum::LIGHT0 + light.max(0) as u32,
                gl_enum::AMBIENT,
                ambient.as_ptr(),
            )
        }
    }
    /// `glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, ...)` then
    /// `glLightfv(GL_LIGHT0 + light, GL_POSITION, position)`.
    fn light_position(&mut self, light: i32, position: [f32; 4], local_viewer: bool) {
        unsafe {
            (self.compat.gl_light_modeli)(
                gl_enum::LIGHT_MODEL_LOCAL_VIEWER,
                if local_viewer { 1 } else { 0 },
            );
            (self.compat.gl_lightfv)(
                gl_enum::LIGHT0 + light.max(0) as u32,
                gl_enum::POSITION,
                position.as_ptr(),
            );
        }
    }
    /// `glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, ...)` and
    /// `glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambient)`.
    fn light_model(&mut self, ambient: [f32; 4], local_viewer: bool) {
        unsafe {
            (self.compat.gl_light_modeli)(
                gl_enum::LIGHT_MODEL_LOCAL_VIEWER,
                if local_viewer { 1 } else { 0 },
            );
            (self.compat.gl_light_modelfv)(gl_enum::LIGHT_MODEL_AMBIENT, ambient.as_ptr());
        }
    }
    /// The `glMaterialfv(GL_FRONT_AND_BACK, ...)` triple, plus
    /// `glMaterialf(GL_SHININESS, shine)`.  `light_adjust` passes no shininess,
    /// which the source expresses by not making that call at all.
    fn material(
        &mut self,
        ambient: [f32; 4],
        diffuse: [f32; 4],
        specular: [f32; 4],
        shininess: f32,
    ) {
        unsafe {
            let face = gl_enum::FRONT_AND_BACK;
            (self.compat.gl_materialfv)(face, gl_enum::AMBIENT, ambient.as_ptr());
            (self.compat.gl_materialfv)(face, gl_enum::DIFFUSE, diffuse.as_ptr());
            (self.compat.gl_materialfv)(face, gl_enum::SPECULAR, specular.as_ptr());
            if shininess != 0. {
                (self.compat.gl_materialf)(face, gl_enum::SHININESS, shininess);
            }
        }
    }
    /// `glEnable`/`glDisable` of `GL_LIGHTING` and `GL_LIGHT0`.
    fn lighting(&mut self, enabled: bool) {
        unsafe {
            if enabled {
                glow::HasContext::enable(&self.gl, gl_enum::LIGHTING);
                glow::HasContext::enable(&self.gl, gl_enum::LIGHT0);
            } else {
                glow::HasContext::disable(&self.gl, gl_enum::LIGHT0);
                glow::HasContext::disable(&self.gl, gl_enum::LIGHTING);
            }
        }
    }
    /// `glPushMatrix(); glLoadIdentity();`.
    fn push_load_identity(&mut self) {
        unsafe {
            (self.compat.gl_push_matrix)();
            (self.compat.gl_load_identity)();
        }
    }
    /// `glScalef(vw->scale.x, vw->scale.y, vw->scale.z * zscale)`.
    fn model_scale(&mut self, x: f32, y: f32, z: f32) {
        unsafe { (self.compat.gl_scalef)(x, y, z) }
    }
    /// `glPopMatrix`.
    fn pop_matrix(&mut self) {
        unsafe { (self.compat.gl_pop_matrix)() }
    }
    /// `glEnable`/`glDisable(GL_COLOR_MATERIAL)`.
    fn color_material(&mut self, enabled: bool) {
        unsafe {
            if enabled {
                glow::HasContext::enable(&self.gl, gl_enum::COLOR_MATERIAL)
            } else {
                glow::HasContext::disable(&self.gl, gl_enum::COLOR_MATERIAL)
            }
        }
    }
}

#[cfg(feature = "three-dmod-gl")]
impl crate::imod::three_dmod::mv_ogl::MvOglBoundary for ImodvNativeGl {
    /// `glPushName`.
    fn push_name(&mut self, name: u32) {
        unsafe { (self.compat.gl_push_name)(name) }
    }
    /// `glPopName`.
    fn pop_name(&mut self) {
        unsafe { (self.compat.gl_pop_name)() }
    }
    /// `glLoadName`.
    fn load_name(&mut self, name: u32) {
        unsafe { (self.compat.gl_load_name)(name) }
    }
    /// `glFinish`.
    fn finish(&mut self) {
        unsafe { glow::HasContext::finish(&self.gl) }
    }
    /// `glMatrixMode(GL_PROJECTION); glLoadIdentity();`.
    fn projection_identity(&mut self) {
        unsafe {
            (self.compat.gl_matrix_mode)(gl_enum::PROJECTION);
            (self.compat.gl_load_identity)();
        }
    }
    /// `glMatrixMode(GL_PROJECTION)`.
    fn projection_mode(&mut self) {
        unsafe { (self.compat.gl_matrix_mode)(gl_enum::PROJECTION) }
    }
    /// `glMatrixMode(GL_MODELVIEW); glLoadIdentity();`.
    fn modelview_identity(&mut self) {
        unsafe {
            (self.compat.gl_matrix_mode)(gl_enum::MODELVIEW);
            (self.compat.gl_load_identity)();
        }
    }
    /// `glMatrixMode(GL_MODELVIEW)`.
    fn modelview_mode(&mut self) {
        unsafe { (self.compat.gl_matrix_mode)(gl_enum::MODELVIEW) }
    }
    /// `glOrtho(-xs, xs, -ys, ys, znear, zfar)`.
    fn ortho(&mut self, x: f64, y: f64, near: f64, far: f64) {
        unsafe { (self.compat.gl_ortho)(-x, x, -y, y, near, far) }
    }
    /// `glFrustum(-xs, xs, -ys, ys, zn, zf)`.
    fn frustum(&mut self, x: f64, y: f64, near: f64, far: f64) {
        unsafe { (self.compat.gl_frustum)(-x, x, -y, y, near, far) }
    }
    /// `glTranslatef`.
    fn translate(&mut self, x: f32, y: f32, z: f32) {
        unsafe { (self.compat.gl_translatef)(x, y, z) }
    }
    /// `glRotatef`.
    fn rotate(&mut self, degrees: f32, x: f32, y: f32, z: f32) {
        unsafe { (self.compat.gl_rotatef)(degrees, x, y, z) }
    }
    /// `glScalef`.
    fn scale(&mut self, x: f32, y: f32, z: f32) {
        unsafe { (self.compat.gl_scalef)(x, y, z) }
    }
    /// `glPushMatrix`.
    fn push_matrix(&mut self) {
        unsafe { (self.compat.gl_push_matrix)() }
    }
    /// `glPopMatrix`.
    fn pop_matrix(&mut self) {
        unsafe { (self.compat.gl_pop_matrix)() }
    }
    /// `glViewport`.
    fn viewport(&mut self, x: i32, y: i32, width: i32, height: i32) {
        unsafe { glow::HasContext::viewport(&self.gl, x, y, width, height) }
    }
    /// `glDepthMask`.
    fn depth_mask(&mut self, enabled: bool) {
        unsafe { glow::HasContext::depth_mask(&self.gl, enabled) }
    }
    /// `glBegin`.
    fn begin(&mut self, mode: u32) {
        unsafe { (self.compat.gl_begin)(mode) }
    }
    /// `glEnd`.
    fn end(&mut self) {
        unsafe { (self.compat.gl_end)() }
    }
    /// `glVertex3f`.
    fn vertex3f(&mut self, x: f32, y: f32, z: f32) {
        unsafe { (self.compat.gl_vertex3f)(x, y, z) }
    }
    /// `glNormal3f`.
    fn normal3f(&mut self, x: f32, y: f32, z: f32) {
        unsafe { (self.compat.gl_normal3f)(x, y, z) }
    }
    /// `glColor4ub`.
    fn color4ub(&mut self, red: u8, green: u8, blue: u8, alpha: u8) {
        unsafe { (self.compat.gl_color4ub)(red, green, blue, alpha) }
    }
    /// `glPolygonMode(GL_FRONT_AND_BACK, GL_LINE | GL_FILL)`.
    fn polygon_mode_line(&mut self, line: bool) {
        unsafe {
            (self.compat.gl_polygon_mode)(
                gl_enum::FRONT_AND_BACK,
                if line { gl_enum::LINE } else { gl_enum::FILL },
            )
        }
    }
    /// `glFrontFace(GL_CW | GL_CCW)`.
    fn front_face_cw(&mut self, cw: bool) {
        unsafe { (self.compat.gl_front_face)(if cw { gl_enum::CW } else { gl_enum::CCW }) }
    }
    /// `glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, value)`.
    fn light_model_two_side(&mut self, value: i32) {
        unsafe { (self.compat.gl_light_modeli)(gl_enum::LIGHT_MODEL_TWO_SIDE, value) }
    }
    /// `glEnable`/`glDisable(GL_BLEND)`.
    fn blend(&mut self, enabled: bool) {
        unsafe {
            if enabled {
                glow::HasContext::enable(&self.gl, gl_enum::BLEND)
            } else {
                glow::HasContext::disable(&self.gl, gl_enum::BLEND)
            }
        }
    }
    /// `glBlendFunc`.
    fn blend_func(&mut self, source: u32, destination: u32) {
        unsafe { glow::HasContext::blend_func(&self.gl, source, destination) }
    }
    /// `glEnable`/`glDisable(GL_CULL_FACE)`.
    fn cull_face(&mut self, enabled: bool) {
        unsafe {
            if enabled {
                glow::HasContext::enable(&self.gl, gl_enum::CULL_FACE)
            } else {
                glow::HasContext::disable(&self.gl, gl_enum::CULL_FACE)
            }
        }
    }
    /// `glEnable`/`glDisable(GL_LINE_SMOOTH)`.
    fn line_smooth(&mut self, enabled: bool) {
        unsafe {
            if enabled {
                glow::HasContext::enable(&self.gl, gl_enum::LINE_SMOOTH)
            } else {
                glow::HasContext::disable(&self.gl, gl_enum::LINE_SMOOTH)
            }
        }
    }
    /// `glEnable`/`glDisable(GL_NORMALIZE)`.
    fn normalize(&mut self, enabled: bool) {
        unsafe {
            if enabled {
                glow::HasContext::enable(&self.gl, gl_enum::NORMALIZE)
            } else {
                glow::HasContext::disable(&self.gl, gl_enum::NORMALIZE)
            }
        }
    }
    /// `light_on(obj, sModBeingDrawn)` (`mv_light.cpp:270`).  The source looks
    /// the view up as `Imodv->mod[modind]->view`, which is the view of the
    /// model `imodvSetObject` was given, so it is passed in directly.
    fn light_on(
        &mut self,
        object: &crate::imod::libimod::imodel::Iobj,
        view: &crate::imod::libimod::imodel::Iview,
        model_zscale: f32,
    ) {
        let mut light = std::mem::take(&mut self.light);
        crate::imod::three_dmod::mv_light::light_on(&mut light, object, view, model_zscale, self);
        self.light = light;
    }
    /// `light_off()` (`mv_light.cpp:322`).
    fn light_off(&mut self) {
        crate::imod::three_dmod::mv_light::light_off(self);
    }
    /// `gluQuadricDrawStyle(qobj, style)`, creating `qobj` on first use as the
    /// source's file-static does.
    fn quadric_draw_style(&mut self, style: u32) {
        unsafe {
            if self.quadric.is_null() {
                self.quadric = (self.glu.glu_new_quadric)();
            }
            (self.glu.glu_quadric_draw_style)(self.quadric, style);
        }
    }
    /// `gluSphere(qobj, radius, slices, stacks)`.
    fn sphere(&mut self, radius: f64, slices: i32, stacks: i32) {
        unsafe {
            if self.quadric.is_null() {
                self.quadric = (self.glu.glu_new_quadric)();
            }
            (self.glu.glu_sphere)(self.quadric, radius, slices, stacks);
        }
    }
    /// `glGenLists`.
    fn gen_lists(&mut self, range: i32) -> u32 {
        unsafe { (self.compat.gl_gen_lists)(range) }
    }
    /// `glNewList(list, GL_COMPILE)`.
    fn new_list(&mut self, list: u32) {
        unsafe { (self.compat.gl_new_list)(list, gl_enum::COMPILE) }
    }
    /// `glEndList`.
    fn end_list(&mut self) {
        unsafe { (self.compat.gl_end_list)() }
    }
    /// `glCallList`.
    fn call_list(&mut self, list: u32) {
        unsafe { (self.compat.gl_call_list)(list) }
    }
    /// `glDeleteLists`.
    fn delete_lists(&mut self, list: u32, range: i32) {
        unsafe { (self.compat.gl_delete_lists)(list, range) }
    }
    /// The `IMOD_MESH_BGNBIGPOLY` tessellation: `gluNewTess`, the `GLU_BEGIN`,
    /// `GLU_VERTEX` and `GLU_END` callbacks bound to `glBegin`, `glVertex3fv`
    /// and `glEnd`, then `gluBeginPolygon`/`gluTessVertex`/`gluEndPolygon` and
    /// `gluDeleteTess`.
    fn tess_polygon(&mut self, vertices: &[crate::imod::libimod::imodel::Ipoint]) {
        unsafe {
            let tobj = (self.glu.glu_new_tess)();
            if tobj.is_null() {
                return;
            }
            (self.glu.glu_tess_callback)(
                tobj,
                glu_enum::TESS_BEGIN,
                Some(std::mem::transmute::<
                    unsafe extern "C" fn(u32),
                    unsafe extern "C" fn(),
                >(self.compat.gl_begin)),
            );
            (self.glu.glu_tess_callback)(
                tobj,
                glu_enum::TESS_VERTEX,
                Some(std::mem::transmute::<
                    unsafe extern "C" fn(*const f32),
                    unsafe extern "C" fn(),
                >(self.compat.gl_vertex3fv)),
            );
            (self.glu.glu_tess_callback)(tobj, glu_enum::TESS_END, Some(self.compat.gl_end));
            (self.glu.glu_begin_polygon)(tobj);
            // The source reuses one `GLdouble v[3]` for every vertex, and
            // passes the `Ipoint *` itself as the vertex data the callback
            // receives.
            let mut v = [0f64; 3];
            for point in vertices {
                v[0] = point.x as f64;
                v[1] = point.y as f64;
                v[2] = point.z as f64;
                (self.glu.glu_tess_vertex)(
                    tobj,
                    v.as_mut_ptr(),
                    point as *const crate::imod::libimod::imodel::Ipoint as *mut std::ffi::c_void,
                );
            }
            (self.glu.glu_end_polygon)(tobj);
            (self.glu.glu_delete_tess)(tobj);
        }
    }
    /// `setupFilledContTesselator` (`utilities.cpp:1182`).
    fn setup_filled_cont_tesselator(&mut self) {
        unsafe {
            if !self.filled_tessel.is_null() {
                return;
            }
            self.filled_tessel = (self.glu.glu_new_tess)();
            if self.filled_tessel.is_null() {
                return;
            }
            (self.glu.glu_tess_callback)(
                self.filled_tessel,
                glu_enum::TESS_BEGIN,
                Some(std::mem::transmute::<
                    unsafe extern "C" fn(u32),
                    unsafe extern "C" fn(),
                >(self.compat.gl_begin)),
            );
            (self.glu.glu_tess_callback)(
                self.filled_tessel,
                glu_enum::TESS_VERTEX,
                Some(std::mem::transmute::<
                    unsafe extern "C" fn(*const f32),
                    unsafe extern "C" fn(),
                >(self.compat.gl_vertex3fv)),
            );
            (self.glu.glu_tess_callback)(
                self.filled_tessel,
                glu_enum::TESS_END,
                Some(self.compat.gl_end),
            );
        }
    }
    /// `drawFilledPolygon(cont)` (`utilities.cpp:1194`).
    ///
    /// The source unit is `utilities.cpp`, which is translated in
    /// `utilities.rs` only as the `UtilitiesBoundary::draw_filled_polygon`
    /// signature; the GLU sequence is issued here because there is no
    /// tessellator elsewhere in the tree, and it belongs beside
    /// `setupFilledContTesselator` when that unit lands.  The retry on a
    /// tessellation error that the source drives from `sTessError` needs the
    /// `GLU_ERROR` callback, which cannot be a plain C function pointer here,
    /// so the first, whole-contour pass is what is drawn.
    fn draw_filled_polygon(&mut self, contour: &crate::imod::libimod::imodel::Icont) {
        unsafe {
            if self.filled_tessel.is_null() {
                return;
            }
            if contour.pts.len() < 3 {
                return;
            }
            (self.glu.glu_tess_begin_polygon)(self.filled_tessel, std::ptr::null_mut());
            (self.glu.glu_tess_begin_contour)(self.filled_tessel);
            let mut v = [0f64; 3];
            for point in &contour.pts {
                v[0] = point.x as f64;
                v[1] = point.y as f64;
                v[2] = point.z as f64;
                (self.glu.glu_tess_vertex)(
                    self.filled_tessel,
                    v.as_mut_ptr(),
                    point as *const crate::imod::libimod::imodel::Ipoint as *mut std::ffi::c_void,
                );
            }
            (self.glu.glu_tess_end_contour)(self.filled_tessel);
            (self.glu.glu_tess_end_polygon)(self.filled_tessel);
        }
    }
    /// `utilManagePairedMeshes(obj, obNum)` (`utilities.cpp:1255`).
    ///
    /// Not translated: it calls `imodMeshMakePairs` and rewrites `obj->mesh`,
    /// which needs `utilities.rs` plus the mesh-pair maker.  0 is the source's
    /// "nothing changed, carry on drawing" return.
    fn manage_paired_meshes(
        &mut self,
        object: &crate::imod::libimod::imodel::Iobj,
        _object_number: i32,
    ) -> i32 {
        if object.mesh_thickness != 0 {
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| {
                eprintln!(
                    "3dmod: mesh thickness ignored: utilManagePairedMeshes (utilities.cpp:1255) \
                     is not translated"
                )
            });
        }
        0
    }
    /// `glGetIntegerv(GL_MAX_CLIP_PLANES, &maxPlanes)`.
    fn max_clip_planes(&mut self) -> i32 {
        unsafe { glow::HasContext::get_parameter_i32(&self.gl, gl_enum::MAX_CLIP_PLANES) }
    }
    /// `glClipPlane(GL_CLIP_PLANE0 + index, params)`.
    fn clip_plane(&mut self, index: i32, params: [f64; 4]) {
        unsafe {
            (self.compat.gl_clip_plane)(gl_enum::CLIP_PLANE0 + index.max(0) as u32, params.as_ptr())
        }
    }
    /// `glEnable(GL_CLIP_PLANE0 + index)`.
    fn enable_clip_plane(&mut self, index: i32) {
        unsafe { glow::HasContext::enable(&self.gl, gl_enum::CLIP_PLANE0 + index.max(0) as u32) }
    }
    /// `glDisable(GL_CLIP_PLANE0 + index)`.
    fn disable_clip_plane(&mut self, index: i32) {
        unsafe { glow::HasContext::disable(&self.gl, gl_enum::CLIP_PLANE0 + index.max(0) as u32) }
    }
    /// `mvImageAnyClipping()` (`mv_image.cpp:328`).
    ///
    /// The translated call needs `MvImageState`, which the model-view image
    /// host owns and which does not exist yet; false is the source's answer
    /// whenever no image is loaded, which is the 3dmodv case.
    fn image_any_clipping(&mut self) -> bool {
        false
    }
    /// `mvImageGetClipPlanes()` (`mv_image.cpp:342`).  Reachable only when
    /// `image_any_clipping` is true, which it is not here.
    fn image_clip_planes(&mut self) -> Option<crate::imod::libimod::imodel::Iclip_planes> {
        None
    }
    /// `imodvSetLight(imod->view)` (`mv_light.cpp:52`).
    fn set_light(&mut self, view: &mut crate::imod::libimod::imodel::Iview) {
        let mut light = std::mem::take(&mut self.light);
        crate::imod::three_dmod::mv_light::imodv_set_light(&mut light, view, self);
        self.light = light;
    }
    /// `utilEnableStipple(vi, cont)` (`utilities.cpp:242`).
    fn enable_stipple(&mut self, draw_stipple: i32, contour: &crate::imod::libimod::imodel::Icont) {
        if draw_stipple != 0 && contour.flags & crate::imod::libimod::icont::ICONT_STIPPLED != 0 {
            unsafe {
                (self.compat.gl_line_stipple)(3, 0x5555);
                glow::HasContext::enable(&self.gl, gl_enum::LINE_STIPPLE);
            }
        }
    }
    /// `utilDisableStipple(vi, cont)` (`utilities.cpp:253`).
    fn disable_stipple(
        &mut self,
        draw_stipple: i32,
        contour: &crate::imod::libimod::imodel::Icont,
    ) {
        if draw_stipple != 0 && contour.flags & crate::imod::libimod::icont::ICONT_STIPPLED != 0 {
            unsafe { glow::HasContext::disable(&self.gl, gl_enum::LINE_STIPPLE) }
        }
    }
    /// `getTopSlicer`, `getTopSlicerAngles` and `getNormalToPlane`
    /// (`sslice.cpp`).
    ///
    /// Not available: the slicer window list is owned by the normal 3dmod
    /// display host, which does not exist yet.  Non-zero is the source's "no
    /// top slicer" path, which makes `drawCurrentClipPlane` return.
    fn top_slicer_plane(
        &mut self,
        _angles: &mut [f32; 3],
        _center: &mut crate::imod::libimod::imodel::Ipoint,
        _time: &mut i32,
        _normal: &mut crate::imod::libimod::imodel::Ipoint,
        _winx: &mut i32,
        _winy: &mut i32,
        _zoom: &mut f32,
    ) -> i32 {
        static WARNED: std::sync::Once = std::sync::Once::new();
        WARNED.call_once(|| {
            eprintln!(
                "3dmod: no slicer plane drawn: getTopSlicer (sslice.cpp) needs the 3dmod \
                 display host"
            )
        });
        1
    }
    /// Static `drawImageForCurrentModel`.
    ///
    /// The source does nothing at all when `a->standalone`, which is the
    /// 3dmodv case; the image path needs `MvImageState` plus an
    /// `MvImageSource`/`MvImageGl` pair fed by the image cache, which is the
    /// normal 3dmod display host of handover step 5.
    fn draw_image(
        &mut self,
        app: &mut crate::imod::three_dmod::imodv::ImodvApp,
        _transparent: bool,
    ) {
        if app.standalone != 0 {
            return;
        }
        static WARNED: std::sync::Once = std::sync::Once::new();
        WARNED.call_once(|| {
            eprintln!("3dmod: no model-view image drawn: mv_image.cpp needs an MvImageSource host")
        });
    }
    /// `imodvDrawLabels(imod, obj, mat, a->winy, dpr, fontHeight, GLw)`.
    fn draw_labels(
        &mut self,
        app: &crate::imod::three_dmod::imodv::ImodvApp,
        model: &crate::imod::libimod::imodel::Imod,
        object: &crate::imod::libimod::imodel::Iobj,
    ) {
        crate::imod::three_dmod::mv_ogl::imodv_draw_labels(
            model,
            object,
            app.winy,
            self.device_pixel_ratio,
            self.font_height,
        );
    }
    /// `glReadPixels(x, y, w, h, GL_DEPTH_COMPONENT, GL_FLOAT, depths)`.
    fn read_depth_pixels(&mut self, x: i32, y: i32, width: i32, height: i32, depths: &mut [f32]) {
        unsafe {
            (self.compat.gl_read_pixels)(
                x,
                y,
                width,
                height,
                gl_enum::DEPTH_COMPONENT,
                gl_enum::FLOAT,
                depths.as_mut_ptr().cast(),
            )
        }
    }
    /// `glGetDoublev`, `glGetIntegerv` and `gluUnProject`.
    fn un_project(
        &mut self,
        winx: f64,
        winy: f64,
        winz: f64,
        objx: &mut f64,
        objy: &mut f64,
        objz: &mut f64,
    ) {
        let mut modelview = [0.0f64; 16];
        let mut projection = [0.0f64; 16];
        let mut viewport = [0i32; 4];
        unsafe {
            (self.compat.gl_get_doublev)(gl_enum::MODELVIEW_MATRIX, modelview.as_mut_ptr());
            (self.compat.gl_get_doublev)(gl_enum::PROJECTION_MATRIX, projection.as_mut_ptr());
            (self.compat.gl_get_integerv)(gl_enum::VIEWPORT, viewport.as_mut_ptr());
            (self.glu.glu_un_project)(
                winx,
                winy,
                winz,
                modelview.as_ptr(),
                projection.as_ptr(),
                viewport.as_ptr(),
                objx,
                objy,
                objz,
            );
        }
    }
}

#[cfg(feature = "three-dmod-gl")]
impl crate::imod::three_dmod::b3dgfx::B3dGfxGl for ImodvNativeGl {
    /// `glGetString(GL_VERSION)`.
    fn gl_version(&mut self) -> Option<&str> {
        if self.version.is_empty() {
            self.version =
                unsafe { glow::HasContext::get_parameter_string(&self.gl, gl_enum::VERSION) };
        }
        Some(self.version.as_str())
    }
    /// `glPrimitiveRestartIndex`.
    fn primitive_restart_index(&mut self, index: u32) {
        match self.compat.gl_primitive_restart_index {
            Some(entry) => unsafe { entry(index) },
            None => {
                static WARNED: std::sync::Once = std::sync::Once::new();
                WARNED.call_once(|| {
                    eprintln!("3dmod: glPrimitiveRestartIndex is not available on this context")
                });
            }
        }
    }
    /// `glViewport`.
    fn viewport(&mut self, x: i32, y: i32, width: i32, height: i32) {
        unsafe { glow::HasContext::viewport(&self.gl, x, y, width, height) }
    }
    /// `glOrtho`.
    fn ortho(&mut self, left: f64, right: f64, bottom: f64, top: f64, near: f64, far: f64) {
        unsafe { (self.compat.gl_ortho)(left, right, bottom, top, near, far) }
    }
    /// `glMatrixMode(GL_PROJECTION); glLoadIdentity();`.
    fn projection_identity(&mut self) {
        unsafe {
            (self.compat.gl_matrix_mode)(gl_enum::PROJECTION);
            (self.compat.gl_load_identity)();
        }
    }
    /// `glMatrixMode(GL_MODELVIEW); glLoadIdentity();`.
    fn modelview_identity(&mut self) {
        unsafe {
            (self.compat.gl_matrix_mode)(gl_enum::MODELVIEW);
            (self.compat.gl_load_identity)();
        }
    }
    /// `glIndexi`.
    fn color_index(&mut self, index: i32) {
        unsafe { (self.compat.gl_indexi)(index) }
    }
    /// `glGetIntegerv(GL_CURRENT_INDEX, ...)`.
    fn current_color_index(&mut self) -> i32 {
        unsafe { glow::HasContext::get_parameter_i32(&self.gl, gl_enum::CURRENT_INDEX) }
    }
    /// `glColor3ub`.
    fn color_rgb(&mut self, rgb: [u8; 3]) {
        unsafe { (self.compat.gl_color3ub)(rgb[0], rgb[1], rgb[2]) }
    }
    /// `glLineStipple`.
    fn line_stipple(&mut self, factor: i32, pattern: u16) {
        unsafe { (self.compat.gl_line_stipple)(factor, pattern) }
    }
    /// `glLineWidth`.
    fn line_width(&mut self, width: f32) {
        unsafe { glow::HasContext::line_width(&self.gl, width) }
    }
    /// `glGetFloatv(GL_LINE_WIDTH, ...)`.
    fn current_line_width(&mut self) -> f32 {
        let mut width = 0f32;
        unsafe { (self.compat.gl_get_floatv)(gl_enum::LINE_WIDTH, &mut width) };
        width
    }
    /// `glPointSize`.
    fn point_size(&mut self, size: f32) {
        unsafe { (self.compat.gl_point_size)(size) }
    }
    /// `gluDisk` on a `GLUquadricObj`.  GLU is a separate library from the GL
    /// the display loader serves, so this needs `libGLU` bound explicitly;
    /// nothing in the model-view path reaches it.
    fn disk(&mut self, _x: f32, _y: f32, _inner: f64, _outer: f64, _slices: i32, _loops: i32) {
        static WARNED: std::sync::Once = std::sync::Once::new();
        WARNED.call_once(|| {
            eprintln!("3dmod: no disk drawn: b3dgfx.cpp's gluDisk needs libGLU bound")
        });
    }
    /// `glBegin`.
    fn begin(&mut self, mode: u32) {
        unsafe { (self.compat.gl_begin)(mode) }
    }
    /// `glVertex2i`.
    fn vertex_2i(&mut self, x: i32, y: i32) {
        unsafe { (self.compat.gl_vertex2i)(x, y) }
    }
    /// `glEnd`.
    fn end(&mut self) {
        unsafe { (self.compat.gl_end)() }
    }
    /// `glEnable/glDisable(GL_LINE_SMOOTH)`.
    fn line_smooth(&mut self, enabled: bool) {
        unsafe {
            if enabled {
                glow::HasContext::enable(&self.gl, gl_enum::LINE_SMOOTH)
            } else {
                glow::HasContext::disable(&self.gl, gl_enum::LINE_SMOOTH)
            }
        }
    }
    /// `glEnable/glDisable(GL_LINE_STIPPLE)`.
    fn line_stipple_enabled(&mut self, enabled: bool) {
        unsafe {
            if enabled {
                glow::HasContext::enable(&self.gl, gl_enum::LINE_STIPPLE)
            } else {
                glow::HasContext::disable(&self.gl, gl_enum::LINE_STIPPLE)
            }
        }
    }
    /// `glPixelStorei(GL_UNPACK_ALIGNMENT, ...)`.
    fn pixel_store_unpack_alignment(&mut self, alignment: i32) {
        unsafe { glow::HasContext::pixel_store_i32(&self.gl, gl_enum::UNPACK_ALIGNMENT, alignment) }
    }
    /// `glPixelZoom`.
    fn pixel_zoom(&mut self, x: f32, y: f32) {
        unsafe { (self.compat.gl_pixel_zoom)(x, y) }
    }
    /// `glRasterPos2f`.
    fn raster_pos_2f(&mut self, x: f32, y: f32) {
        unsafe { (self.compat.gl_raster_pos2f)(x, y) }
    }
    /// `glDrawPixels`.
    fn draw_pixels(&mut self, width: i32, height: i32, format: u32, typ: u32, data: &[u8]) {
        unsafe {
            (self.compat.gl_draw_pixels)(
                width,
                height,
                format,
                typ,
                data.as_ptr() as *const std::ffi::c_void,
            )
        }
    }
    /// `glReadPixels(..., GL_RGBA, GL_UNSIGNED_BYTE, ...)`.
    fn read_pixels_rgba(&mut self, x: i32, y: i32, width: i32, height: i32, out: &mut [u8]) {
        unsafe {
            glow::HasContext::read_pixels(
                &self.gl,
                x,
                y,
                width,
                height,
                gl_enum::RGBA,
                gl_enum::UNSIGNED_BYTE,
                glow::PixelPackData::Slice(Some(out)),
            )
        }
    }
    /// `glFlush`.
    fn flush(&mut self) {
        unsafe { glow::HasContext::flush(&self.gl) }
    }
}

/// Native replacement for the `QMainWindow`/`QOpenGLWidget` ownership chain.
///
/// The context is compatibility OpenGL 2.1, requested by profile as well as by
/// version: `mv_ogl.cpp` and `mv_gfx.cpp` are written in fixed-function
/// OpenGL - the selection-buffer names, `glRenderMode`, and the projection and
/// model-view matrix stack - and none of those entry points exist on a core
/// profile context.
///
/// Ownership: in the source, `ImodvGL` owns the context and every `gl*` call in
/// the drawing units runs on whatever Qt has made current, so `initializeGL`,
/// `paintGL` and `resizeGL` take no context argument.  The equivalent here is
/// `ImodvNativeGl`, which owns the glutin surface and context; it is built once
/// the context is current and handed to the sink through `make_sink` rather
/// than threaded through every callback, which keeps the sink callbacks the
/// same shape as the Qt ones.  `Rc<RefCell<..>>` because the event loop and the
/// sink run on the one thread and both have to reach it.
#[cfg(feature = "three-dmod-gl")]
pub static IMODV_EXIT_REQUESTED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// `qApp->exit()`/`QWidget::close()` reaching the event loop.
///
/// `imodvQuit` ends in `imod_exit`/`QApplication::closeAllWindows` in the
/// source; here the loop owns the exit, so the quit path raises this flag and
/// `run_native_opengl` leaves on the next pass.
#[cfg(feature = "three-dmod-gl")]
pub fn run_native_opengl(
    model_window: Box<ImodvWindow>,
    make_sink: Box<
        dyn FnOnce(std::rc::Rc<std::cell::RefCell<ImodvNativeGl>>) -> Box<dyn ImodvWindowSink>,
    >,
) -> Result<(), String> {
    let mut model_window = model_window;
    use crate::imod::three_dmod::mv_gfx::ImodvGfxGl;
    use glutin::config::ConfigTemplateBuilder;
    use glutin::context::{ContextApi, ContextAttributesBuilder, GlProfile, Version};
    use glutin::display::{GetGlDisplay, GlDisplay};
    use glutin::prelude::*;
    use glutin::surface::{GlSurface, SurfaceAttributesBuilder, SwapInterval, WindowSurface};
    use glutin_winit::DisplayBuilder;
    use raw_window_handle::HasWindowHandle;
    use std::cell::RefCell;
    use std::num::NonZeroU32;
    use std::rc::Rc;
    use winit::dpi::PhysicalSize;
    use winit::event::{
        DeviceEvent, ElementState, Event, MouseButton, MouseScrollDelta,
        WindowEvent as WinitWindowEvent,
    };
    use winit::event_loop::{DeviceEvents, EventLoop};
    use winit::window::Window;

    let event_loop = EventLoop::new().map_err(|error| error.to_string())?;
    // Raw button events are what separate a wheel press from its release; see
    // the `MouseWheel` arm below.  Qt's xcb backend reads the same X11
    // XInput2 events directly, so this is not an added input source, only the
    // same information winit does not put on `WindowEvent::MouseWheel`.
    event_loop.listen_device_events(DeviceEvents::Always);
    let app = model_window.app();
    let width = app.want_winx.max(app.winx).max(1) as u32;
    let height = app.want_winy.max(app.winy).max(1) as u32;
    let attrs = Window::default_attributes()
        .with_title("3dmod")
        .with_inner_size(PhysicalSize::new(width, height));
    let display_builder = DisplayBuilder::new().with_window_attributes(Some(attrs));
    let template = ConfigTemplateBuilder::new()
        .with_alpha_size(if app.alpha_visual != 0 { 8 } else { 0 })
        .with_depth_size(24)
        .with_transparency(app.trans_bkgd != 0);
    let (window, config) = display_builder
        .build(&event_loop, template, |configs| {
            configs
                .max_by_key(|config| config.num_samples())
                .expect("no GL config")
        })
        .map_err(|error| error.to_string())?;
    let window = window.ok_or_else(|| "winit did not create the 3dmod window".to_owned())?;
    let raw_handle = window
        .window_handle()
        .map_err(|error| error.to_string())?
        .as_raw();
    let display = config.display();
    let context_attributes = ContextAttributesBuilder::new()
        .with_profile(GlProfile::Compatibility)
        .with_context_api(ContextApi::OpenGl(Some(Version::new(2, 1))))
        .build(Some(raw_handle));
    let not_current = unsafe { display.create_context(&config, &context_attributes) }
        .map_err(|error| error.to_string())?;
    let surface_attributes = SurfaceAttributesBuilder::<WindowSurface>::new().build(
        raw_handle,
        NonZeroU32::new(width).expect("positive width"),
        NonZeroU32::new(height).expect("positive height"),
    );
    let surface = unsafe { display.create_window_surface(&config, &surface_attributes) }
        .map_err(|error| error.to_string())?;
    let context = not_current
        .make_current(&surface)
        .map_err(|error| error.to_string())?;
    surface
        .set_swap_interval(
            &context,
            SwapInterval::Wait(NonZeroU32::new(1).expect("nonzero")),
        )
        .map_err(|error| error.to_string())?;
    let gl =
        unsafe { glow::Context::from_loader_function_cstr(|name| display.get_proc_address(name)) };
    let window = Rc::new(window);
    let mut backend = ImodvNativeGl::new(gl, surface, context, Rc::clone(&window), &|name| {
        display.get_proc_address(name)
    })?;
    model_window.init_width = width as i32;
    model_window.init_height = height as i32;
    model_window.device_pixel_ratio = window.scale_factor() as f32;
    backend.device_pixel_ratio = model_window.device_pixel_ratio;
    let backend = Rc::new(RefCell::new(backend));
    let mut sink = make_sink(Rc::clone(&backend));
    // `QMouseEvent::buttons()` and `QInputEvent::modifiers()`: Qt carries
    // both on every event, winit reports the transitions, so the dispatcher
    // keeps them.
    let mut buttons_down = 0u32;
    let mut shift_down = false;
    let mut ctrl_down = false;
    let mut alt_down = false;
    let mut cursor = winit::dpi::PhysicalPosition::new(0.0f64, 0.0f64);
    // Set by the raw release of a legacy wheel button, consumed by the
    // duplicate `MouseWheel` that follows it.  See the `MouseWheel` arm.
    let mut wheel_button_released = false;
    IMODV_EXIT_REQUESTED.store(false, std::sync::atomic::Ordering::Relaxed);
    event_loop
        .run(move |event, target| match event {
            Event::AboutToWait => {
                // Qt's event dispatcher fires the widget's `timerEvent` and
                // the window's `timeoutSlot`; this loop is that dispatcher.
                let due = {
                    let mut shared = backend.borrow_mut();
                    let now = std::time::Instant::now();
                    let mut due = Vec::new();
                    for entry in shared.timers.iter_mut() {
                        if now.duration_since(entry.2) >= entry.1 {
                            entry.2 = now;
                            due.push(entry.3);
                        }
                    }
                    due
                };
                for window_timer in due {
                    if window_timer {
                        model_window.timeout_slot(&mut *sink);
                    } else if let Some(index) = model_window.cur_glw {
                        let mut gl_widget = match index {
                            0 => model_window.dbw.take(),
                            1 => model_window.dbalw.take(),
                            2 => model_window.sbw.take(),
                            3 => model_window.dbstw.take(),
                            4 => model_window.dbst_alw.take(),
                            _ => model_window.sbstw.take(),
                        };
                        if let Some(ref mut gl_widget) = gl_widget {
                            gl_widget.timer_event(&mut model_window, &mut *sink);
                        }
                        match index {
                            0 => model_window.dbw = gl_widget,
                            1 => model_window.dbalw = gl_widget,
                            2 => model_window.sbw = gl_widget,
                            3 => model_window.dbstw = gl_widget,
                            4 => model_window.dbst_alw = gl_widget,
                            _ => model_window.sbstw = gl_widget,
                        }
                    }
                }
                if IMODV_EXIT_REQUESTED.load(std::sync::atomic::Ordering::Relaxed) {
                    target.exit();
                    return;
                }
                window.request_redraw()
            }
            Event::WindowEvent { event, .. } => match event {
                WinitWindowEvent::CloseRequested => {
                    model_window.close_event(&mut *sink);
                    target.exit();
                }
                WinitWindowEvent::Resized(size) => {
                    if let (Some(width), Some(height)) =
                        (NonZeroU32::new(size.width), NonZeroU32::new(size.height))
                    {
                        {
                            let shared = backend.borrow();
                            shared.surface.resize(&shared.context, width, height);
                        }
                        model_window.event(
                            WindowEvent::Resize {
                                width: size.width as i32,
                                height: size.height as i32,
                            },
                            &mut *sink,
                        );
                        // `imodvResizeGL` issues the `glViewport` itself.
                        sink.imodv_resize_gl(size.width as i32, size.height as i32);
                    }
                }
                WinitWindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                    model_window.device_pixel_ratio = scale_factor as f32;
                    backend.borrow_mut().device_pixel_ratio = scale_factor as f32;
                }
                WinitWindowEvent::Focused(focused) => {
                    model_window.app_focus_changed(focused, &mut *sink)
                }
                WinitWindowEvent::RedrawRequested => {
                    // `imodvPaintGL` owns all model fixed-function rendering from
                    // mv_gfx/mv_ogl, and it clears the window itself, so nothing
                    // is drawn here outside the widget's paint callback.
                    if let Some(index) = model_window.cur_glw {
                        let mut gl_widget = match index {
                            0 => model_window.dbw.take(),
                            1 => model_window.dbalw.take(),
                            2 => model_window.sbw.take(),
                            3 => model_window.dbstw.take(),
                            4 => model_window.dbst_alw.take(),
                            _ => model_window.sbstw.take(),
                        };
                        if let Some(ref mut gl_widget) = gl_widget {
                            gl_widget.paint_gl(&mut model_window, &mut *sink);
                        }
                        match index {
                            0 => model_window.dbw = gl_widget,
                            1 => model_window.dbalw = gl_widget,
                            2 => model_window.sbw = gl_widget,
                            3 => model_window.dbstw = gl_widget,
                            4 => model_window.dbst_alw = gl_widget,
                            _ => model_window.sbstw = gl_widget,
                        }
                    }
                    // Qt swaps for the widget after `paintGL` unless
                    // `setBufferSwapAuto(false)` is in force.
                    ImodvGfxGl::swap_buffers(&mut *backend.borrow_mut());
                }
                WinitWindowEvent::MouseInput { state, button, .. } => {
                    let bit = match button {
                        MouseButton::Left => crate::imod::three_dmod::mv_input::INPUT_LEFT,
                        MouseButton::Middle => crate::imod::three_dmod::mv_input::INPUT_MIDDLE,
                        MouseButton::Right => crate::imod::three_dmod::mv_input::INPUT_RIGHT,
                        _ => 0,
                    };
                    let left_pressed = state == ElementState::Pressed;
                    if left_pressed {
                        buttons_down |= bit;
                    } else {
                        buttons_down &= !bit;
                    }
                    let event = KeyEvent {
                        x: cursor.x as i32,
                        y: cursor.y as i32,
                        button: bit,
                        buttons: buttons_down,
                        control: ctrl_down,
                        shift: shift_down,
                        alt: alt_down,
                        ..KeyEvent::default()
                    };
                    if let Some(index) = model_window.cur_glw {
                        let mut gl_widget = match index {
                            0 => model_window.dbw.take(),
                            1 => model_window.dbalw.take(),
                            2 => model_window.sbw.take(),
                            3 => model_window.dbstw.take(),
                            4 => model_window.dbst_alw.take(),
                            _ => model_window.sbstw.take(),
                        };
                        if let Some(ref mut gl_widget) = gl_widget {
                            if left_pressed {
                                gl_widget.mouse_press_event(event, &mut *sink);
                            } else {
                                gl_widget.mouse_release_event(event, &mut *sink);
                            }
                        }
                        match index {
                            0 => model_window.dbw = gl_widget,
                            1 => model_window.dbalw = gl_widget,
                            2 => model_window.sbw = gl_widget,
                            3 => model_window.dbstw = gl_widget,
                            4 => model_window.dbst_alw = gl_widget,
                            _ => model_window.sbstw = gl_widget,
                        }
                    }
                }
                WinitWindowEvent::CursorMoved { position, .. } => {
                    cursor = position;
                    if let Some(index) = model_window.cur_glw {
                        let mut gl_widget = match index {
                            0 => model_window.dbw.take(),
                            1 => model_window.dbalw.take(),
                            2 => model_window.sbw.take(),
                            3 => model_window.dbstw.take(),
                            4 => model_window.dbst_alw.take(),
                            _ => model_window.sbstw.take(),
                        };
                        if let Some(ref mut gl_widget) = gl_widget {
                            gl_widget.mouse_move_event(
                                KeyEvent {
                                    x: position.x as i32,
                                    y: position.y as i32,
                                    button: buttons_down,
                                    buttons: buttons_down,
                                    control: ctrl_down,
                                    shift: shift_down,
                                    alt: alt_down,
                                    ..KeyEvent::default()
                                },
                                &mut *sink,
                            );
                        }
                        match index {
                            0 => model_window.dbw = gl_widget,
                            1 => model_window.dbalw = gl_widget,
                            2 => model_window.sbw = gl_widget,
                            3 => model_window.dbstw = gl_widget,
                            4 => model_window.dbst_alw = gl_widget,
                            _ => model_window.sbstw = gl_widget,
                        }
                    }
                }
                WinitWindowEvent::MouseWheel { delta, .. } => {
                    // `QWheelEvent::delta()` is 120 per notch, the value
                    // `imodvScrollWheel` tests the sign of.
                    //
                    // Where X11 reports the wheel as legacy button 4/5,
                    // winit's `xinput2_button_input`
                    // (`x11/event_processor.rs:1083`) builds this event from
                    // the button number without looking at `state`, so it
                    // fires for the press *and* the release, while Qt's xcb
                    // backend acts on the press only and delivers one
                    // `QWheelEvent` per notch.  The raw button events say
                    // which of the two this is: measured under Xvfb with an
                    // XTEST-injected notch, the order is raw press, wheel,
                    // raw release, wheel.  So the wheel that follows a raw
                    // release is the duplicate, and dropping it leaves one
                    // per notch, as Qt has.
                    //
                    // A device reported as axis motion takes winit's motion
                    // path instead; its emulated buttons carry
                    // `XIPointerEmulated`, which suppresses both the raw
                    // button event (`event_processor.rs:1420`) and the button
                    // one (`:1063`), so the flag is never set and its single
                    // wheel event is delivered.  The flag is cleared by the
                    // one wheel event it suppresses and never outlives it.
                    if wheel_button_released {
                        wheel_button_released = false;
                        return;
                    }
                    let event = KeyEvent {
                        x: cursor.x as i32,
                        y: cursor.y as i32,
                        buttons: buttons_down,
                        control: ctrl_down,
                        shift: shift_down,
                        alt: alt_down,
                        delta: match delta {
                            MouseScrollDelta::LineDelta(_, y) => (y * 120.) as i32,
                            MouseScrollDelta::PixelDelta(position) => position.y as i32,
                        },
                        ..KeyEvent::default()
                    };
                    sink.imodv_scroll_wheel(event);
                }
                WinitWindowEvent::ModifiersChanged(modifiers) => {
                    let state = modifiers.state();
                    shift_down = state.shift_key();
                    ctrl_down = state.control_key();
                    alt_down = state.alt_key();
                }
                WinitWindowEvent::KeyboardInput { event, .. } => {
                    use winit::keyboard::NamedKey;
                    // `QKeyEvent::key()`: the printable keys are their ASCII
                    // codes with letters upper-cased, the rest are the
                    // `0x0100_00xx` values `imod_input.rs` already carries.
                    let (key, qt_key) = match event.logical_key.as_ref() {
                        winit::keyboard::Key::Character(text) => match text.chars().next() {
                            Some(c) => {
                                let upper = c.to_ascii_uppercase();
                                let named = match upper {
                                    '-' => Key::Minus,
                                    '=' => Key::Equal,
                                    '_' => Key::Underscore,
                                    '+' => Key::Plus,
                                    ',' => Key::Comma,
                                    '.' => Key::Period,
                                    '[' => Key::LeftBracket,
                                    ']' => Key::RightBracket,
                                    '{' => Key::LeftBrace,
                                    '}' => Key::RightBrace,
                                    '(' => Key::LeftParen,
                                    ')' => Key::RightParen,
                                    _ => Key::Character(upper),
                                };
                                (named, upper as i32)
                            }
                            None => (Key::Unknown, 0),
                        },
                        winit::keyboard::Key::Named(named) => match named {
                            NamedKey::Delete => (Key::Delete, 0x0100_0007),
                            NamedKey::Escape => (Key::Unknown, 0x0100_0000),
                            // Qt splits the one Enter key by location:
                            // `Key_Return` on the main block, `Key_Enter` on
                            // the keypad.
                            NamedKey::Enter => (
                                Key::Unknown,
                                if event.location == winit::keyboard::KeyLocation::Numpad {
                                    0x0100_0005
                                } else {
                                    0x0100_0004
                                },
                            ),
                            NamedKey::Insert => (Key::Unknown, 0x0100_0006),
                            NamedKey::Home => (Key::Unknown, 0x0100_0010),
                            NamedKey::End => (Key::Unknown, 0x0100_0011),
                            NamedKey::ArrowLeft => (Key::Unknown, 0x0100_0012),
                            NamedKey::ArrowUp => (Key::Unknown, 0x0100_0013),
                            NamedKey::ArrowRight => (Key::Unknown, 0x0100_0014),
                            NamedKey::ArrowDown => (Key::Unknown, 0x0100_0015),
                            NamedKey::PageUp => (Key::Unknown, 0x0100_0016),
                            NamedKey::PageDown => (Key::Unknown, 0x0100_0017),
                            NamedKey::Shift => (Key::Unknown, 0x0100_0020),
                            NamedKey::Control => (Key::Unknown, 0x0100_0021),
                            NamedKey::Alt => (Key::Unknown, 0x0100_0023),
                            NamedKey::F1 => (Key::F(1), 0x0100_0030),
                            NamedKey::F2 => (Key::F(2), 0x0100_0031),
                            NamedKey::F3 => (Key::F(3), 0x0100_0032),
                            NamedKey::F4 => (Key::F(4), 0x0100_0033),
                            NamedKey::F5 => (Key::F(5), 0x0100_0034),
                            NamedKey::F6 => (Key::F(6), 0x0100_0035),
                            NamedKey::F7 => (Key::F(7), 0x0100_0036),
                            NamedKey::F8 => (Key::F(8), 0x0100_0037),
                            NamedKey::F9 => (Key::F(9), 0x0100_0038),
                            NamedKey::F10 => (Key::F(10), 0x0100_0039),
                            NamedKey::F11 => (Key::F(11), 0x0100_003a),
                            NamedKey::F12 => (Key::F(12), 0x0100_003b),
                            _ => (Key::Unknown, 0),
                        },
                        _ => (Key::Unknown, 0),
                    };
                    let payload = KeyEvent {
                        key,
                        qt_key,
                        x: cursor.x as i32,
                        y: cursor.y as i32,
                        buttons: buttons_down,
                        control: ctrl_down,
                        shift: shift_down,
                        alt: alt_down,
                        keypad: matches!(event.location, winit::keyboard::KeyLocation::Numpad),
                        ..KeyEvent::default()
                    };
                    if event.state == ElementState::Pressed {
                        model_window.key_press_event(payload, &mut *sink);
                    } else {
                        model_window.key_release_event(payload, &mut *sink);
                    }
                }
                _ => {}
            },
            Event::DeviceEvent {
                event: DeviceEvent::Button { button, state },
                ..
            } => {
                // X11 legacy wheel buttons.  `xinput2_button_input`
                // (`x11/event_processor.rs:1083`) maps details 4 to 7 to
                // `MouseWheel`, so those are the ones whose release produces
                // a duplicate.
                if (4..=7).contains(&button) {
                    wheel_button_released = state == ElementState::Released;
                }
            }
            _ => {}
        })
        .map_err(|error| error.to_string())
}

/// The real `ImodvWindowSink`: the Qt slot targets of `mv_window.cpp`.
///
/// In the source these are `ImodvWindow`/`ImodvGL` member functions calling
/// straight into `mv_menu.cpp`, `mv_input.cpp`, `mv_gfx.cpp` and
/// `mv_control.cpp`, so every method here is one of those calls and nothing
/// else.  Where the call needs a widget the translation does not yet have —
/// a file picker, a dialog, a snapshot encoder — the method says so on the
/// spot rather than discarding the event.
#[cfg(feature = "three-dmod-gl")]
pub struct ImodvNativeSink {
    /// `Imodv`, the file-scope `ImodvApp *` every handler in the paired units
    /// takes.  It points into the `IMODV_STATE` thread-local of `imodv.rs`,
    /// which outlives the window exactly as `ImodvStruct` outlives
    /// `ImodvWindow` in the source; the sink is dropped when the event loop
    /// returns, before that state goes away.
    app: *mut ImodvApp,
    /// The `ImodvGL` widget: context, surface, window, and the drawing statics
    /// of `mv_ogl.cpp`.
    gl: std::rc::Rc<std::cell::RefCell<ImodvNativeGl>>,
    /// `ImodvBkgColorDialog` (`mv_menu.cpp`).
    bkg_color: crate::imod::three_dmod::mv_menu::ImodvBkgColor,
    /// `sDialog`'s retained values in `mv_control.cpp`.
    control: crate::imod::three_dmod::mv_control::ImodvControlDialogState,
    /// `a->rbgcolor`, the `QColor` `imodvMain` builds from `a->rbgname` with
    /// `QColor::setNamedColor`.  `ImodvApp::rbgcolor` is the opaque Qt
    /// ownership pointer in the translated struct, so the resolved colour is
    /// held by the native host that would own the `QColor`.
    rbgcolor: [u8; 3],
}

#[cfg(feature = "three-dmod-gl")]
impl ImodvNativeSink {
    /// `ImodvWindow`'s slot wiring plus `imodvMain`'s
    /// `a->rbgcolor->setNamedColor(a->rbgname)` (`imodv.cpp:607`).
    pub fn new(app: *mut ImodvApp, gl: std::rc::Rc<std::cell::RefCell<ImodvNativeGl>>) -> Self {
        let name = unsafe { app.as_ref() }.map_or(String::new(), |app| app.rbgname.clone());
        let lower = name.to_ascii_lowercase();
        let rbgcolor = if let Some(digits) = lower.strip_prefix('#') {
            match digits.len() {
                3 => u32::from_str_radix(digits, 16).map_or([0, 0, 0], |value| {
                    [
                        (((value >> 8) & 0xf) * 17) as u8,
                        (((value >> 4) & 0xf) * 17) as u8,
                        ((value & 0xf) * 17) as u8,
                    ]
                }),
                6 => u32::from_str_radix(digits, 16).map_or([0, 0, 0], |value| {
                    [
                        ((value >> 16) & 0xff) as u8,
                        ((value >> 8) & 0xff) as u8,
                        (value & 0xff) as u8,
                    ]
                }),
                _ => [0, 0, 0],
            }
        } else {
            match lower.as_str() {
                "black" => [0, 0, 0],
                "white" => [255, 255, 255],
                "red" => [255, 0, 0],
                "green" => [0, 128, 0],
                "lime" => [0, 255, 0],
                "blue" => [0, 0, 255],
                "cyan" | "aqua" => [0, 255, 255],
                "magenta" | "fuchsia" => [255, 0, 255],
                "yellow" => [255, 255, 0],
                "gray" | "grey" => [128, 128, 128],
                "darkgray" | "darkgrey" => [169, 169, 169],
                "lightgray" | "lightgrey" => [211, 211, 211],
                "silver" => [192, 192, 192],
                "maroon" => [128, 0, 0],
                "olive" => [128, 128, 0],
                "navy" => [0, 0, 128],
                "teal" => [0, 128, 128],
                "purple" => [128, 0, 128],
                "orange" => [255, 165, 0],
                "brown" => [165, 42, 42],
                "pink" => [255, 192, 203],
                "" => [0, 0, 0],
                _ => {
                    // `setNamedColor` resolves the whole SVG colour list; an
                    // unrecognised name there is invalid and the source falls
                    // back to black, which is what happens here too, but the
                    // name may well be a real SVG colour this table lacks.
                    eprintln!(
                        "3dmodv: background colour \"{name}\" is not in this host's colour table \
                         (QColor::setNamedColor resolves the full SVG list); using black"
                    );
                    [0, 0, 0]
                }
            }
        };
        Self {
            app,
            gl,
            bkg_color: crate::imod::three_dmod::mv_menu::imodv_bkg_color_new(),
            control: crate::imod::three_dmod::mv_control::ImodvControlDialogState::default(),
            rbgcolor,
        }
    }
    /// `Imodv`, as every handler in the paired units names it.
    fn app(&mut self) -> &'static mut ImodvApp {
        unsafe { &mut *self.app }
    }
    /// `a->mainWin`.
    fn main_win(&mut self) -> Option<&'static mut ImodvWindow> {
        unsafe { (*self.app).main_win.as_mut() }
    }
    /// `App->newQtOpenGL`; the native driver creates the context with the
    /// window, which is the `QOpenGLWidget` shape the flag selects.
    fn new_qt_opengl(&self) -> bool {
        true
    }
}

#[cfg(feature = "three-dmod-gl")]
impl ImodvWindowSink for ImodvNativeSink {
    /// `imodvFileMenu(which)`.
    fn imodv_file_menu(&mut self, which: i32) {
        let action = crate::imod::three_dmod::mv_menu::imodv_file_menu(which as usize);
        match action {
            Some("imodv_reset_snap") => {
                let app = self.app();
                crate::imod::three_dmod::mv_gfx::imodv_reset_snap(app);
            }
            Some("close") => crate::imod::three_dmod::imodv::imodv_quit(),
            Some(name) => eprintln!(
                "3dmodv: File menu item {which} needs {name}, which has no native host yet \
                 (file picker, snapshot encoder, snapshot directory chooser, and the movie and \
                 sequence dialogs of mv_movie.cpp)"
            ),
            None => {}
        }
    }
    /// `imodvEditMenu(which)`.
    fn imodv_edit_menu(&mut self, which: i32) {
        let rgb = [
            self.rbgcolor[0] as i32,
            self.rbgcolor[1] as i32,
            self.rbgcolor[2] as i32,
        ];
        let mut color = std::mem::take(&mut self.bkg_color);
        let action =
            crate::imod::three_dmod::mv_menu::imodv_edit_menu(which as usize, &mut color, rgb);
        self.bkg_color = color;
        if let Some(name) = action {
            eprintln!(
                "3dmodv: Edit menu item {which} needs {name}, which has no native host yet \
                 (the model-view dialogs of mv_objed.cpp, mv_control.cpp, mv_listobj.cpp, \
                 mv_modeled.cpp, mv_views.cpp, mv_image.cpp and isosurface.cpp)"
            );
        }
    }
    /// `imodvViewMenu(which)`.
    fn imodv_view_menu(&mut self, which: i32) {
        let item = which as usize;
        let app = self.app();
        let window = unsafe { app.main_win.as_mut() };
        let action = crate::imod::three_dmod::mv_menu::imodv_view_menu(app, item, window);
        match action {
            Some("imodv_setbuffer") => {
                // `imodv_setbuffer(a, 1 - a->dblBuf, -1, -1)` and
                // `imodv_setbuffer(a, -1, -1, 1 - a->transBkgd)`
                // (`mv_menu.cpp:389,396`).  The translated `imodvViewMenu`
                // applies the flag before returning, so the value passed here
                // is the one the source's expression produces.
                let (db, stereo, alpha) = if item == VVIEW_MENU_DB {
                    (self.app().dbl_buf, -1, -1)
                } else {
                    (-1, -1, (self.app().trans_bkgd != 0) as i32)
                };
                let rgb = self.rbgcolor;
                let window = unsafe { (*self.app).main_win };
                let mut set_widget = |db: bool, stereo: bool, alpha: bool| -> i32 {
                    match unsafe { window.as_mut() } {
                        Some(window) => window.set_gl_widget(db, stereo, alpha),
                        None => 1,
                    }
                };
                let gl = std::rc::Rc::clone(&self.gl);
                let mut gl = gl.borrow_mut();
                crate::imod::three_dmod::mv_gfx::imodv_setbuffer(
                    self.app(),
                    db,
                    stereo,
                    alpha,
                    rgb,
                    &mut set_widget,
                    &mut *gl,
                );
            }
            Some(name) => eprintln!(
                "3dmodv: View menu item {which} needs {name}, which has no native host yet \
                 (mv_stereo.cpp, mv_depthcue.cpp, scalebar.cpp and the resize tool)"
            ),
            None => {}
        }
    }
    /// `imodvHelpMenu(which)`.
    fn imodv_help_menu(&mut self, which: i32) {
        match crate::imod::three_dmod::mv_menu::imodv_help_menu(which as usize) {
            Some("3dmodv Version") => {
                // `dia_vasmsg` (`mv_menu.cpp`), a modal message box.
                crate::imod::three_dmod::imod::imod_print_info(
                    "3dmodv Version , written by David Mastronarde, James Kremer, and Quanren \
                     Xiong\nCopyright (C) by the Regents of the University of Colorado\n",
                );
            }
            Some(page) => {
                if let Err(message) = crate::imod::three_dmod::imod::imod_show_help_page(page) {
                    eprintln!("{message}");
                }
            }
            None => {}
        }
    }
    /// `imodvKeyPress(event)`.
    fn imodv_key_press(&mut self, event: KeyEvent) {
        let payload: crate::imod::three_dmod::mv_input::InputEvent = event.into();
        let app = self.app();
        crate::imod::three_dmod::mv_input::imodv_key_press(app, payload, self);
    }
    /// `imodvKeyRelease(event)`.
    fn imodv_key_release(&mut self, event: KeyEvent) {
        let payload: crate::imod::three_dmod::mv_input::InputEvent = event.into();
        let app = self.app();
        crate::imod::three_dmod::mv_input::imodv_key_release(app, payload, self);
    }
    /// `imodvMousePress(event)`.
    fn imodv_mouse_press(&mut self, event: KeyEvent) {
        let payload: crate::imod::three_dmod::mv_input::InputEvent = event.into();
        let app = self.app();
        crate::imod::three_dmod::mv_input::imodv_mouse_press(app, payload, self);
    }
    /// `imodvMouseRelease(event)`.
    fn imodv_mouse_release(&mut self, event: KeyEvent) {
        let payload: crate::imod::three_dmod::mv_input::InputEvent = event.into();
        let app = self.app();
        crate::imod::three_dmod::mv_input::imodv_mouse_release(app, payload, self);
    }
    /// `imodvMouseMove(event)`.
    fn imodv_mouse_move(&mut self, event: KeyEvent) {
        let payload: crate::imod::three_dmod::mv_input::InputEvent = event.into();
        let app = self.app();
        crate::imod::three_dmod::mv_input::imodv_mouse_move(app, payload, self);
    }
    /// `imodvScrollWheel(event)`.
    fn imodv_scroll_wheel(&mut self, event: KeyEvent) {
        let payload: crate::imod::three_dmod::mv_input::InputEvent = event.into();
        let app = self.app();
        crate::imod::three_dmod::mv_input::imodv_scroll_wheel(app, payload, self);
    }
    /// `imodvMovieTimeout()`.
    fn imodv_movie_timeout(&mut self) {
        let app = self.app();
        crate::imod::three_dmod::mv_input::imodv_movie_timeout(app, self);
    }
    /// `imodvInitializeGL()`.
    fn imodv_initialize_gl(&mut self) {
        let gl = std::rc::Rc::clone(&self.gl);
        let mut gl = gl.borrow_mut();
        crate::imod::three_dmod::mv_gfx::imodv_initialize_gl(self.app(), &mut *gl);
    }
    /// `imodvPaintGL()`.
    fn imodv_paint_gl(&mut self) {
        let rgb = self.rbgcolor;
        let new_qt_opengl = self.new_qt_opengl();
        let gl = std::rc::Rc::clone(&self.gl);
        let mut gl = gl.borrow_mut();
        crate::imod::three_dmod::mv_gfx::imodv_paint_gl(self.app(), rgb, new_qt_opengl, &mut *gl);
    }
    /// `imodvResizeGL(winx, winy)`.
    fn imodv_resize_gl(&mut self, width: i32, height: i32) {
        let new_qt_opengl = self.new_qt_opengl();
        let gl = std::rc::Rc::clone(&self.gl);
        let mut gl = gl.borrow_mut();
        let device_pixel_ratio = gl.device_pixel_ratio;
        crate::imod::three_dmod::mv_gfx::imodv_resize_gl(
            self.app(),
            width,
            height,
            device_pixel_ratio,
            new_qt_opengl,
            &mut *gl,
        );
    }
    /// `imodvQuit()`.
    fn imodv_quit(&mut self) {
        crate::imod::three_dmod::imodv::imodv_quit();
    }
    /// `imodv_rotate_model(a, x, y, z)`.
    fn imodv_rotate_model(&mut self, x: f32, y: f32, z: f32) {
        let app = self.app();
        crate::imod::three_dmod::mv_input::imodv_rotate_model(
            app, x as i32, y as i32, z as i32, self,
        );
    }
    /// `imodvControlChangeSteps(delta)`.
    fn imodv_control_change_steps(&mut self, delta: i32) {
        let control = self.control.clone();
        let app = unsafe { &mut *self.app };
        crate::imod::three_dmod::mv_control::imodv_control_change_steps(app, &control, self, delta);
    }
    /// `imodvControlStart()`.
    ///
    /// The movie timer is not started here: `imodv_compute_rotation`
    /// (`mv_input.cpp:1382`) calls `imodv_start_movie` on the first rotation
    /// once `a->movie` is set, and `imodvMovieTimeout` stops it.
    fn imodv_control_start(&mut self) {
        let app = unsafe { &mut *self.app };
        crate::imod::three_dmod::mv_control::imodv_control_start(app, self);
    }
    /// `QObject::startTimer(interval)` on the `ImodvGL` widget.
    fn start_timer(&mut self, interval: i32) -> i32 {
        let mut gl = self.gl.borrow_mut();
        let id = gl.next_timer_id + 1;
        gl.next_timer_id = id;
        gl.timers.push((
            id,
            std::time::Duration::from_millis(interval.max(0) as u64),
            std::time::Instant::now(),
            false,
        ));
        id
    }
    /// `QObject::killTimer(timer)`.
    fn kill_timer(&mut self, timer: i32) {
        self.gl.borrow_mut().timers.retain(|entry| entry.0 != timer);
    }
    /// `mCurGLw->updateGL()`.
    fn update_gl(&mut self) {
        self.gl.borrow().window.request_redraw();
    }
    /// `ImodvWindow::resize(width, height)`.
    fn resize_window(&mut self, width: i32, height: i32) {
        let gl = self.gl.borrow();
        let _ = gl.window.request_inner_size(winit::dpi::PhysicalSize::new(
            width.max(1) as u32,
            height.max(1) as u32,
        ));
    }
    /// `setCurrentDevicePixelRatio` (`utilities.cpp`).
    fn set_current_device_pixel_ratio(&mut self, ratio: f32) {
        self.gl.borrow_mut().device_pixel_ratio = ratio;
    }
    /// `imodvDialogManager.hide()`.
    fn dialogs_hide(&mut self) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: imodvDialogManager.hide() does nothing: no model-view dialog windows \
                 are registered with the DialogManager of control.cpp yet"
            )
        });
    }
    /// `imodvDialogManager.show()`.
    fn dialogs_show(&mut self) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: imodvDialogManager.show() does nothing: no model-view dialog windows \
                 are registered with the DialogManager of control.cpp yet"
            )
        });
    }
    /// `imodvDialogManager.masterWinHasChanged()`.
    fn dialogs_master_changed(&mut self) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: imodvDialogManager.masterWinHasChanged() does nothing: no model-view \
                 dialog windows are registered with the DialogManager of control.cpp yet"
            )
        });
    }
    /// `imodvDialogManager.windowActivated()`.
    fn dialogs_activated(&mut self) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: imodvDialogManager.windowActivated() does nothing: no model-view dialog \
                 windows are registered with the DialogManager of control.cpp yet"
            )
        });
    }
    /// `imodvAppLostFocus()`.
    fn app_lost_focus(&mut self) {
        crate::imod::three_dmod::mv_input::imodv_app_lost_focus();
    }
}

/// `imodvControlForm`, the dialog `mv_control.cpp` drives.
///
/// The rotation tool on the window reaches `imodvControlStart` and
/// `imodvControlChangeSteps` whether or not the control dialog is open; with
/// no dialog the source's form calls are on a null `sDialog` and do nothing.
/// The dialog itself is not translated into a widget yet, so each call says
/// what it would have updated.
#[cfg(feature = "three-dmod-gl")]
impl crate::imod::three_dmod::mv_control::ImodvControlNativeBoundary for ImodvNativeSink {
    fn remove_dialog(&mut self) {}
    fn close_dialog(&mut self) {}
    fn model_edit_update(&mut self) {}
    fn rotation_center_state(&mut self, state: bool) {
        let _ = state;
    }
    fn rotation_step_label(&mut self, step: f32) {
        let _ = step;
    }
    fn raise_dialog(&mut self) {}
    fn create_dialog(&mut self) -> bool {
        eprintln!(
            "3dmodv: the model view Controls dialog (imodvControlForm, mv_control.cpp) has no \
             native window yet"
        );
        false
    }
    fn set_axis_text(&mut self, axis: i32, value: f32) {
        let _ = (axis, value);
    }
    fn set_scale_text(&mut self, value: f32) {
        let _ = value;
    }
    fn set_rotation_rate(&mut self, value: f32) {
        let _ = value;
    }
    fn set_speed_text(&mut self, value: f32) {
        let _ = value;
    }
    fn set_view_slider(&mut self, slider: i32, value: i32) {
        let _ = (slider, value);
    }
    fn set_kick_box(&mut self, checked: bool) {
        let _ = checked;
    }
    fn update_slicer_link(&mut self, value: i32) {
        let _ = value;
    }
}

/// `mv_input.cpp` calls straight into `imod_input.cpp`; the whole viewer half
/// of that unit has no host in a model-view-only build, so every one of these
/// keeps the trait's own report.
#[cfg(feature = "three-dmod-gl")]
impl crate::imod::three_dmod::imod_input::InputNativeBoundary for ImodvNativeSink {}

/// The widget, dialog, preferences, undo and OpenGL calls of `mv_input.cpp`.
///
/// The overrides are the ones this host can actually perform; the rest keep
/// the trait's default, which names the unit that is missing.
#[cfg(feature = "three-dmod-gl")]
impl crate::imod::three_dmod::mv_input::MvInputNativeBoundary for ImodvNativeSink {
    /// `a->mainWin->close()`.
    fn main_win_close(&mut self) {
        crate::imod::three_dmod::imodv::imodv_quit();
        IMODV_EXIT_REQUESTED.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    /// `ImodvClosed`; the window exists for as long as this sink does.
    fn imodv_closed(&mut self) -> bool {
        IMODV_EXIT_REQUESTED.load(std::sync::atomic::Ordering::Relaxed)
    }
    /// `imodvMenuBgcolor(state)`.
    fn imodv_menu_bgcolor(&mut self, state: i32) {
        let rgb = [
            self.rbgcolor[0] as i32,
            self.rbgcolor[1] as i32,
            self.rbgcolor[2] as i32,
        ];
        let mut color = std::mem::take(&mut self.bkg_color);
        crate::imod::three_dmod::mv_menu::imodv_menu_bgcolor(state, &mut color, rgb);
        self.bkg_color = color;
    }
    /// `imodv_control(a, state)`.
    fn imodv_control(&mut self, a: &mut crate::imod::three_dmod::imodv::ImodvApp, state: i32) {
        let mut dialog = self.control.clone();
        crate::imod::three_dmod::mv_control::imodv_control(a, &mut dialog, self, state);
        self.control = dialog;
    }
    /// `imodvStereoUpdate()`; `sDialog` of `mv_stereo.cpp` is never opened
    /// here, which is the state the source's own null check covers.
    fn imodv_stereo_update(&mut self, a: &mut crate::imod::three_dmod::imodv::ImodvApp) {
        crate::imod::three_dmod::mv_stereo::imodv_stereo_update(None, a);
    }
    /// `imodvControlLinkUpdate(a)`.
    fn imodv_control_link_update(&mut self, a: &mut crate::imod::three_dmod::imodv::ImodvApp) {
        let dialog = self.control.clone();
        crate::imod::three_dmod::mv_control::imodv_control_link_update(a, &dialog, self);
    }
    /// `imodvControlChangeSteps(a, delta)`.
    fn imodv_control_change_steps(
        &mut self,
        a: &mut crate::imod::three_dmod::imodv::ImodvApp,
        delta: i32,
    ) {
        let dialog = self.control.clone();
        crate::imod::three_dmod::mv_control::imodv_control_change_steps(a, &dialog, self, delta);
    }
    /// `imodvControlStart()`.
    fn imodv_control_start(&mut self, a: &mut crate::imod::three_dmod::imodv::ImodvApp) {
        crate::imod::three_dmod::mv_control::imodv_control_start(a, self);
    }
    /// `a->mainWin->openRotationTool(a)`.
    fn open_rotation_tool(&mut self, a: &mut crate::imod::three_dmod::imodv::ImodvApp) {
        if let Some(window) = unsafe { a.main_win.as_mut() } {
            window.open_rotation_tool();
        }
    }
    /// `imodvViewMenu(which)`.
    fn imodv_view_menu(&mut self, a: &mut crate::imod::three_dmod::imodv::ImodvApp, which: i32) {
        ImodvWindowSink::imodv_view_menu(self, which);
    }
    /// `imodvMenuLowres(value)`.
    fn imodv_menu_lowres(&mut self, a: &mut crate::imod::three_dmod::imodv::ImodvApp, value: i32) {
        if let Some(window) = unsafe { a.main_win.as_mut() } {
            crate::imod::three_dmod::mv_menu::imodv_menu_lowres(window, value);
        }
    }
    /// `a->mainWin->setEnabledMenuItem(id, state)`.
    fn set_enabled_menu_item(
        &mut self,
        a: &mut crate::imod::three_dmod::imodv::ImodvApp,
        id: usize,
        state: bool,
    ) {
        if let Some(window) = unsafe { a.main_win.as_mut() } {
            window.set_enabled_menu_item(id, state);
        }
    }
    /// `imodv_setbuffer(a, db, stereo, alpha)`.
    fn imodv_setbuffer(
        &mut self,
        a: &mut crate::imod::three_dmod::imodv::ImodvApp,
        db: i32,
        stereo: i32,
        alpha: i32,
    ) {
        let rgb = self.rbgcolor;
        let window = a.main_win;
        let mut set_widget = |db: bool, stereo: bool, alpha: bool| -> i32 {
            match unsafe { window.as_mut() } {
                Some(window) => window.set_gl_widget(db, stereo, alpha),
                None => 1,
            }
        };
        let gl = std::rc::Rc::clone(&self.gl);
        let mut gl = gl.borrow_mut();
        crate::imod::three_dmod::mv_gfx::imodv_setbuffer(
            a,
            db,
            stereo,
            alpha,
            rgb,
            &mut set_widget,
            &mut *gl,
        );
    }
    /// `imodv_winset(a)`.
    fn imodv_winset(&mut self, a: &mut crate::imod::three_dmod::imodv::ImodvApp) {
        let gl = std::rc::Rc::clone(&self.gl);
        let mut gl = gl.borrow_mut();
        crate::imod::three_dmod::mv_gfx::imodv_winset(a, &mut *gl);
    }
    /// `light_moveby(a->imod->view, x, y)`.
    fn light_moveby(&mut self, a: &mut crate::imod::three_dmod::imodv::ImodvApp, x: i32, y: i32) {
        let Some(imod) = (unsafe { a.imod.as_mut() }) else {
            return;
        };
        let Some(view) = imod.view.first_mut() else {
            return;
        };
        let gl = std::rc::Rc::clone(&self.gl);
        let mut gl = gl.borrow_mut();
        let mut light = std::mem::take(&mut gl.light);
        crate::imod::three_dmod::mv_light::light_moveby(&mut light, view, x, y, &mut *gl);
        gl.light = light;
    }
    /// `imodvSelectVisibleConts(a, pickedOb, pickedCo)`.
    fn imodv_select_visible_conts(
        &mut self,
        a: &mut crate::imod::three_dmod::imodv::ImodvApp,
        picked_ob: &mut i32,
        picked_co: &mut i32,
    ) {
        let gl = std::rc::Rc::clone(&self.gl);
        let mut gl = gl.borrow_mut();
        let mut state = std::mem::take(&mut gl.state);
        unsafe {
            crate::imod::three_dmod::mv_ogl::imodv_select_visible_conts(
                &mut state, a, picked_ob, picked_co, &mut *gl,
            )
        };
        gl.state = state;
    }
    /// `imodvModelDrawRange(a, mstart, mend)` then
    /// `imodvUnprojectPickedPoint(a, mend)`.
    fn imodv_unproject_picked_point(&mut self, a: &mut crate::imod::three_dmod::imodv::ImodvApp) {
        let mut mstart = 0;
        let mut mend = 0;
        crate::imod::three_dmod::mv_modeled::imodv_model_draw_range(a, &mut mstart, &mut mend);
        let gl = std::rc::Rc::clone(&self.gl);
        let mut gl = gl.borrow_mut();
        crate::imod::three_dmod::mv_ogl::imodv_unproject_picked_point(a, mend, &mut *gl);
    }
    /// `findClickedDrawnElement(a, curObj, moNum, obNum, coNum, ptNum)`.
    fn find_clicked_drawn_element(
        &mut self,
        a: &mut crate::imod::three_dmod::imodv::ImodvApp,
        cur_obj: bool,
        mo_num: &mut i32,
        ob_num: &mut i32,
        co_num: &mut i32,
        pt_num: &mut i32,
    ) {
        let gl = std::rc::Rc::clone(&self.gl);
        let mut gl = gl.borrow_mut();
        let mut state = std::mem::take(&mut gl.state);
        unsafe {
            crate::imod::three_dmod::mv_ogl::find_clicked_drawn_element(
                &mut state, a, cur_obj, mo_num, ob_num, co_num, pt_num,
            )
        };
        gl.state = state;
    }
    /// `a->mainWin->mTimer->start(interval)` then `timerId()`.
    fn movie_timer_start(&mut self, interval: i32) -> i32 {
        let mut gl = self.gl.borrow_mut();
        gl.timers.retain(|entry| !entry.3);
        let id = gl.next_timer_id + 1;
        gl.next_timer_id = id;
        gl.timers.push((
            id,
            std::time::Duration::from_millis(interval.max(1) as u64),
            std::time::Instant::now(),
            true,
        ));
        id
    }
    /// `a->mainWin->mTimer->stop()`.
    fn movie_timer_stop(&mut self) {
        self.gl.borrow_mut().timers.retain(|entry| !entry.3);
    }
    /// `Imodv->mainWin->raise()`.
    fn main_win_raise(&mut self) {
        IMODV_HOST_STATE.with(|state| {
            if let Some(window) = state.borrow().gl_window.as_ref() {
                window.focus_window();
            }
        });
    }
}

/// The `imodv.cpp` file statics and Qt-owned objects `ImodvNativeBoundary`
/// stands for: the `ImodvWindow *` `openWindow` creates, the window title
/// `imodvSetCaption` sets, the `-W` window-id request, and the
/// `Imodv_stereo_data` of `mv_stereo.cpp`.
///
/// They live in their own cell, not inside the host object, because every one
/// of these calls can be re-entered from inside the event loop while
/// `run_application` is on the stack.
#[cfg(feature = "three-dmod-gl")]
#[derive(Default)]
pub struct ImodvNativeHostState {
    /// `a->mainWin`.  Boxed because `ImodvApp::main_win` is the pointer to it.
    pub window: Option<Box<ImodvWindow>>,
    /// The native window, once `run_application` has created it.
    pub gl_window: Option<std::rc::Rc<winit::window::Window>>,
    /// `setWindowTitle`.
    pub title: String,
    /// `-W`: `imodPrintStderr("Window id = %u\n", a->mainWin->winId())`.
    pub print_window_id: bool,
    /// `Imodv_stereo_data` (`mv_stereo.cpp`).
    pub stereo: crate::imod::three_dmod::mv_stereo::ImodvStereoData,
    /// `a->standalone`.  `imodvOpenSelectedWindows` takes it from `Imodv`,
    /// which `imodvMain` is in the middle of writing when it makes that call,
    /// so the value is recorded when `openWindow` has the app in hand.
    pub standalone: i32,
}

#[cfg(feature = "three-dmod-gl")]
thread_local! {
    pub static IMODV_HOST_STATE: std::cell::RefCell<ImodvNativeHostState> =
        std::cell::RefCell::new(ImodvNativeHostState::default());
}

/// The concrete `ImodvNativeBoundary`: what Qt did for `imodv.cpp`.
///
/// It carries no fields of its own: `imodvQuit`, `imodvDraw` and the rest are
/// reached again from inside the event loop this host starts, so its state is
/// in `IMODV_HOST_STATE` and each call borrows that cell only for as long as
/// it is working.
#[cfg(feature = "three-dmod-gl")]
pub struct ImodvNativeHost;

#[cfg(feature = "three-dmod-gl")]
impl crate::imod::three_dmod::imodv::ImodvNativeBoundary for ImodvNativeHost {
    /// `App->cvi`.
    fn current_model_view(&mut self) -> *mut crate::imod::three_dmod::imodview::ImodView {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmod: imodv_open has no current image view: App->cvi is owned by the normal \
                 3dmod display host (imod.cpp/imodview.cpp), which is not built yet"
            )
        });
        std::ptr::null_mut()
    }
    /// `new QPixmap(QPixmap::fromImage(QImage(b3dicon)))`.
    fn model_view_icon(&mut self) -> *mut crate::imod::three_dmod::imodv::QPixmap {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: no window icon is set: the b3dicon QPixmap of imodv.cpp needs an image \
                 decoder and winit icon at this host"
            )
        });
        std::ptr::null_mut()
    }
    /// `a->mainWin->raise()`.
    fn raise_model_view(&mut self, _window: *mut ImodvWindow) {
        IMODV_HOST_STATE.with(|state| {
            if let Some(window) = state.borrow().gl_window.as_ref() {
                window.focus_window();
            }
        });
    }
    /// `getVisuals` (`imodv.cpp:236`).
    ///
    /// The source has two builds: the `#ifndef NEW_QTOPENGL` one probes
    /// `imodFindGLVisual` for each attribute list, and the `NEW_QTOPENGL` one
    /// only resets the depth records and lets `initializeForNewQGL` record
    /// what the created surface actually granted.  This host creates the
    /// context with the window, so it is the second build.
    fn get_visuals(&mut self, a: &mut ImodvApp, once_opened: bool) -> i32 {
        if !once_opened {
            a.enable_depth_sb = -1;
            a.enable_depth_db = -1;
            a.enable_depth_dbal = -1;
            a.enable_depth_sbst = -1;
            a.enable_depth_dbst = -1;
            a.enable_depth_dbst_al = -1;
            a.need_new_qglinit = 1;
        }
        0
    }
    /// `openWindow` (`imodv.cpp:398`).
    fn open_model_view(
        &mut self,
        a: &mut ImodvApp,
        once_opened: bool,
        last_geometry: crate::imod::three_dmod::control::Rect,
    ) -> i32 {
        use crate::imod::three_dmod::mv_views::{
            VIEW_WORLD_INVERT_Z, VIEW_WORLD_LABELS, VIEW_WORLD_LOWRES,
        };
        let Some(world) = (unsafe { a.imod.as_ref() })
            .and_then(|imod| imod.view.first())
            .map(|view| view.world)
        else {
            return 1;
        };
        a.invert_z = (world & VIEW_WORLD_INVERT_Z != 0) as i32;
        a.lighting = (world & crate::imod::libimod::iview::VIEW_WORLD_LIGHT != 0) as i32;
        a.draw_labels = (world & VIEW_WORLD_LABELS != 0) as i32;
        a.lowres = (world & VIEW_WORLD_LOWRES != 0) as i32;
        let mut window = Box::new(ImodvWindow::new(a));
        a.main_win = &mut *window as *mut ImodvWindow;
        // `imodvSetCaption()` is the source's next call.  It reads the
        // `Imodv`/`ImodvClosed` globals, which the caller of `openWindow` is
        // in the middle of writing, so the title it produces is recorded here
        // and applied when the native window exists.
        let title = if a.standalone != 0 {
            "3dmodv:"
        } else {
            "3dmod Model View: "
        };
        if a.need_new_qglinit != 0 {
            window.initialize_for_new_qgl();
        }
        if a.want_winx == 0 {
            a.want_winx = crate::imod::three_dmod::imodv::DEFAULT_XSIZE;
            a.want_winy = crate::imod::three_dmod::imodv::DEFAULT_YSIZE;
        }
        let mut new_width = a.want_winx;
        let mut new_height = a.want_winy;
        if once_opened && last_geometry.width > 0 {
            new_width = last_geometry.width;
            new_height = last_geometry.height;
        }
        window.init_width = new_width;
        window.init_height = new_height;
        a.winx = new_width;
        a.winy = new_height;
        IMODV_HOST_STATE.with(|state| {
            let mut state = state.borrow_mut();
            state.title = title.to_owned();
            state.standalone = a.standalone;
            state.window = Some(window);
        });
        0
    }
    /// `imodDraw(a->vi, flags)`.
    fn imod_draw(&mut self, _view: *mut crate::imod::three_dmod::imodview::ImodView, flags: i32) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmod: imodDraw(flags {flags}) is not issued: the image windows it redraws \
                 (xzap.cpp, slicer.cpp, xyz.cpp) have no native host yet"
            )
        });
    }
    /// `imodvObjedNewView`/`objed` redraw of `mv_objed.cpp`.
    fn object_edit_draw(&mut self) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmod: the object edit dialog is not redrawn: mv_objed.cpp has no native window \
                 yet"
            )
        });
    }
    /// `imodInfoSetObjectColor()`.
    fn info_set_object_color(&mut self) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmod: the info window object colour is not updated: info_cb.cpp has no native \
                 window yet"
            )
        });
    }
    /// `setModvDialogTitle`/`setWindowTitle`.
    fn set_modv_dialog_title(&mut self, _window: *mut ImodvWindow, title: &str) {
        IMODV_HOST_STATE.with(|state| {
            let mut state = state.borrow_mut();
            state.title = title.to_owned();
            if let Some(window) = state.gl_window.as_ref() {
                window.set_title(title);
            }
        });
    }
    /// `a->vi->undo->modelChange()`.
    fn undo_model_change(&mut self, _view: *mut crate::imod::three_dmod::imodview::ImodView) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmod: model changes are not registered for undo: ImodView::undo (undoredo.cpp) \
                 is owned by the normal 3dmod display host"
            )
        });
    }
    /// `a->vi->undo->objectPropChg(object)`.
    fn undo_object_prop_change(
        &mut self,
        _view: *mut crate::imod::three_dmod::imodview::ImodView,
        _object: i32,
    ) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmod: object property changes are not registered for undo: ImodView::undo \
                 (undoredo.cpp) is owned by the normal 3dmod display host"
            )
        });
    }
    /// `a->vi->undo->finishUnit()`.
    fn undo_finish_unit(&mut self, _view: *mut crate::imod::three_dmod::imodview::ImodView) {}
    /// `a->mainWin->close()`.
    fn close_model_view(&mut self, _window: *mut ImodvWindow) {
        IMODV_EXIT_REQUESTED.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    /// `imodvDraw(a)`, which is `a->mainWin->mCurGLw->updateGL()`.
    fn imodv_draw(&mut self, _a: &mut ImodvApp) {
        IMODV_HOST_STATE.with(|state| {
            if let Some(window) = state.borrow().gl_window.as_ref() {
                window.request_redraw();
            }
        });
    }
    /// `ivwRestorableGeometry(a->mainWin)`.
    fn restorable_geometry(
        &mut self,
        _window: *mut ImodvWindow,
    ) -> crate::imod::three_dmod::control::Rect {
        IMODV_HOST_STATE.with(|state| {
            let state = state.borrow();
            let Some(window) = state.gl_window.as_ref() else {
                return crate::imod::three_dmod::control::Rect::default();
            };
            let size = window.inner_size();
            let position = window
                .outer_position()
                .unwrap_or(winit::dpi::PhysicalPosition::new(0, 0));
            crate::imod::three_dmod::control::Rect {
                x: position.x,
                y: position.y,
                width: size.width as i32,
                height: size.height as i32,
            }
        })
    }
    /// `ImodPrefs->setModViewGeometry(geometry)`.
    fn record_mod_view_geometry(&mut self, _geometry: crate::imod::three_dmod::control::Rect) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmod: the model view geometry is not remembered: ImodPrefs (preferences.cpp) \
                 has no settings file host yet"
            )
        });
    }
    /// `vbCleanupVBD(a->imod)`.
    fn vb_cleanup_vbd(&mut self, _imod: *mut crate::imod::libimod::imodel::Imod) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: vertex buffer objects are not deleted on exit: vbCleanupVBD \
                 (vertexbuffer.cpp) needs a VertexBufferGl host, which ImodvNativeGl does not \
                 implement yet"
            )
        });
    }
    /// `mvImageCleanup()`.
    fn mv_image_cleanup(&mut self) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: model view image data is not freed on exit: mvImageCleanup \
                 (mv_image.cpp) needs an MvImageGl host"
            )
        });
    }
    /// `ivwFreeExtraObject(a->vi, object)`.
    fn free_extra_object(
        &mut self,
        view: *mut crate::imod::three_dmod::imodview::ImodView,
        object: i32,
    ) {
        if let Some(view) = unsafe { view.as_mut() } {
            crate::imod::three_dmod::imodview::ivw_free_extra_object(view, object);
        }
    }
    /// `ClipHandler->startDisconnect()`.
    fn start_clip_disconnect(&mut self) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: no clipboard handler to disconnect: ImodClipboard (client_message.cpp) \
                 has no native host yet"
            )
        });
    }
    /// `imodvStereoHWOff()` (`mv_stereo.cpp`).
    fn stereo_hw_off(&mut self) {
        IMODV_HOST_STATE.with(|state| {
            let mut state = state.borrow_mut();
            let mut stereo = state.stereo.clone();
            crate::imod::three_dmod::mv_stereo::stereo_hw_off(&mut stereo);
            state.stereo = stereo;
        });
    }
    /// `imodvDialogManager.close()`.
    fn close_model_view_dialogs(&mut self) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: no model view dialogs to close: none are registered with the \
                 DialogManager of control.cpp yet"
            )
        });
    }
    /// `ImodPrefs->saveSettings(1)`.
    fn save_settings(&mut self) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: settings are not saved on exit: ImodPrefs (preferences.cpp) has no \
                 settings file host yet"
            )
        });
    }
    /// `delete ImodHelp`.
    fn delete_imod_help(&mut self) {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "3dmodv: no help assistant to shut down: imod_assistant.cpp has no native host \
                 yet"
            )
        });
    }
    /// `ClipHandler->waitForDisconnect()`.
    fn wait_for_clip_disconnect(&mut self) {}
    /// `imod_exit(0)`/`qApp->quit()`.
    fn exit_application(&mut self) {
        crate::imod::three_dmod::imod::imod_exit(0);
        IMODV_EXIT_REQUESTED.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    /// `imodvDialogManager.checkForExitOnClose()`.
    fn check_for_exit_on_close(&mut self) {
        IMODV_EXIT_REQUESTED.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    /// `App->exiting = 1`.
    fn set_app_exiting(&mut self) {
        if let Some(app) = crate::imod::three_dmod::imod::APP.lock().unwrap().as_mut() {
            app.exiting = 1;
        }
    }
    /// `return qApp->exec();` (`imodv.cpp:648`).
    fn run_application(&mut self) -> i32 {
        let window = IMODV_HOST_STATE.with(|state| state.borrow_mut().window.take());
        let Some(mut window) = window else {
            eprintln!("3dmodv: openWindow did not leave a model view window to run");
            return 3;
        };
        let app = window.app_mut() as *mut ImodvApp;
        let result = run_native_opengl(
            window,
            Box::new(move |gl| {
                IMODV_HOST_STATE.with(|state| {
                    let mut state = state.borrow_mut();
                    let native = std::rc::Rc::clone(&gl.borrow().window);
                    native.set_title(&state.title);
                    if state.print_window_id {
                        // `imodPrintStderr("Window id = %u\n", winId())`.
                        crate::imod::three_dmod::imod::imod_print_stderr(&format!(
                            "Window id = {}\n",
                            u64::from(native.id())
                        ));
                        state.print_window_id = false;
                    }
                    state.gl_window = Some(native);
                });
                Box::new(ImodvNativeSink::new(app, gl))
            }),
        );
        IMODV_HOST_STATE.with(|state| state.borrow_mut().gl_window = None);
        match result {
            Ok(()) => 0,
            Err(message) => {
                eprintln!("3dmodv: {message}");
                3
            }
        }
    }
    /// `-W`: `imodPrintStderr("Window id = %u\n", a->mainWin->winId())`.
    ///
    /// The native window does not exist until `run_application` creates it, so
    /// the request is recorded and the id printed as soon as it does.
    fn print_window_id(&mut self, _window: *mut ImodvWindow) {
        IMODV_HOST_STATE.with(|state| state.borrow_mut().print_window_id = true);
    }
    /// `ClipHandler = new ImodClipboard(useStdin, false)`.
    fn create_clipboard(&mut self, use_stdin: bool) {
        eprintln!(
            "3dmodv: -W/-L clipboard messaging is not available: ImodClipboard \
             (client_message.cpp, use_stdin {use_stdin}) has no native host yet"
        );
    }
    /// `imodvOpenSelectedWindows(windowKeys)`.
    fn open_selected_windows(&mut self, window_keys: Option<&str>) {
        let standalone = IMODV_HOST_STATE.with(|state| state.borrow().standalone);
        let wanted =
            crate::imod::three_dmod::mv_menu::imodv_open_selected_windows(window_keys, standalone);
        for window in wanted {
            eprintln!("3dmodv: -E requested the {window} window, which has no native host yet");
        }
    }
    /// `usage(argv[0])`.
    fn show_usage(&mut self, usage: &str) {
        crate::imod::three_dmod::imod::imod_print_info(usage);
    }
    /// `imodError(NULL, message)`.
    fn show_error(&mut self, message: &str) {
        crate::imod::three_dmod::imod::imod_error(message);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_widget_selection_prefers_double_buffer() {
        let mut app = ImodvApp {
            enable_depth_db: 1,
            enable_depth_sb: 1,
            dbl_buf: 1,
            ..Default::default()
        };
        let win = ImodvWindow::new(&mut app);
        assert_eq!(win.cur_glw, Some(0));
        assert!(win.actions[VVIEW_MENU_DB].checked);
    }
    #[test]
    fn source_widget_selection_falls_back_to_single_buffer() {
        let mut app = ImodvApp {
            enable_depth_db: -1,
            enable_depth_dbal: -1,
            enable_depth_dbst: -1,
            enable_depth_dbst_al: -1,
            enable_depth_sb: 1,
            enable_depth_sbst: -1,
            dbl_buf: 1,
            ..Default::default()
        };
        let win = ImodvWindow::new(&mut app);
        assert_eq!(win.cur_glw, Some(2));
        assert_eq!(app.dbl_buf, 0);
    }
    /// The concrete native backend must answer every boundary the drawing
    /// units call, or `imodvPaintGL` cannot reach `imodvDraw_models` at all.
    #[cfg(feature = "three-dmod-gl")]
    #[test]
    fn imodv_native_gl_answers_every_source_gl_boundary() {
        let boundaries: fn(&mut ImodvNativeGl) = |backend| {
            let _: &mut dyn crate::imod::three_dmod::mv_gfx::ImodvGfxGl = backend;
            let _: &mut dyn crate::imod::three_dmod::mv_ogl::MvOglBoundary = backend;
            let _: &mut dyn crate::imod::three_dmod::b3dgfx::B3dGfxGl = backend;
        };
        let _ = boundaries;
    }

    /// `run_native_opengl` must be able to hand the context it creates to the
    /// sink; the factory shape is what makes that possible.
    #[cfg(feature = "three-dmod-gl")]
    #[test]
    fn imodv_native_opengl_takes_a_sink_factory_over_the_context() {
        let factory: fn(
            Box<
                dyn FnOnce(
                    std::rc::Rc<std::cell::RefCell<ImodvNativeGl>>,
                ) -> Box<dyn ImodvWindowSink>,
            >,
        ) = |maker| {
            let _ = maker;
        };
        factory(Box::new(|_gl| Box::new(NullImodvWindowSink::default())));
    }

    #[test]
    fn resize_timer_records_source_state() {
        let mut gl = ImodvGl::new(GlFormat::default());
        gl.first_draw = 0;
        let mut sink = NullImodvWindowSink::default();
        gl.schedule_resize(31, 41, 10, &mut sink);
        assert!(gl.scheduled_resize);
        assert_eq!((gl.sched_width, gl.sched_height), (31, 41));
        gl.cancel_resize(&mut sink);
        assert!(!gl.scheduled_resize);
    }

    /// Records which source slot the window routed an event to.
    #[derive(Default)]
    struct RecordingImodvWindowSink {
        calls: Vec<&'static str>,
        events: Vec<KeyEvent>,
        timers: Vec<i32>,
        resizes: Vec<(i32, i32)>,
        next_timer: i32,
    }
    impl ImodvWindowSink for RecordingImodvWindowSink {
        fn imodv_file_menu(&mut self, _: i32) {
            self.calls.push("file_menu")
        }
        fn imodv_edit_menu(&mut self, _: i32) {
            self.calls.push("edit_menu")
        }
        fn imodv_view_menu(&mut self, _: i32) {
            self.calls.push("view_menu")
        }
        fn imodv_help_menu(&mut self, _: i32) {
            self.calls.push("help_menu")
        }
        fn imodv_key_press(&mut self, event: KeyEvent) {
            self.calls.push("key_press");
            self.events.push(event);
        }
        fn imodv_key_release(&mut self, event: KeyEvent) {
            self.calls.push("key_release");
            self.events.push(event);
        }
        fn imodv_mouse_press(&mut self, event: KeyEvent) {
            self.calls.push("mouse_press");
            self.events.push(event);
        }
        fn imodv_mouse_release(&mut self, event: KeyEvent) {
            self.calls.push("mouse_release");
            self.events.push(event);
        }
        fn imodv_mouse_move(&mut self, event: KeyEvent) {
            self.calls.push("mouse_move");
            self.events.push(event);
        }
        fn imodv_scroll_wheel(&mut self, event: KeyEvent) {
            self.calls.push("scroll_wheel");
            self.events.push(event);
        }
        fn imodv_movie_timeout(&mut self) {
            self.calls.push("movie_timeout")
        }
        fn imodv_initialize_gl(&mut self) {
            self.calls.push("initialize_gl")
        }
        fn imodv_paint_gl(&mut self) {
            self.calls.push("paint_gl")
        }
        fn imodv_resize_gl(&mut self, width: i32, height: i32) {
            self.calls.push("resize_gl");
            self.resizes.push((width, height));
        }
        fn imodv_quit(&mut self) {
            self.calls.push("quit")
        }
        fn imodv_rotate_model(&mut self, _: f32, _: f32, _: f32) {
            self.calls.push("rotate_model")
        }
        fn imodv_control_change_steps(&mut self, _: i32) {
            self.calls.push("control_change_steps")
        }
        fn imodv_control_start(&mut self) {
            self.calls.push("control_start")
        }
        fn start_timer(&mut self, _: i32) -> i32 {
            self.calls.push("start_timer");
            self.next_timer += 1;
            self.timers.push(self.next_timer);
            self.next_timer
        }
        fn kill_timer(&mut self, timer: i32) {
            self.calls.push("kill_timer");
            self.timers.retain(|id| *id != timer);
        }
        fn update_gl(&mut self) {
            self.calls.push("update_gl")
        }
        fn resize_window(&mut self, width: i32, height: i32) {
            self.calls.push("resize_window");
            self.resizes.push((width, height));
        }
        fn set_current_device_pixel_ratio(&mut self, _: f32) {
            self.calls.push("device_pixel_ratio")
        }
        fn dialogs_hide(&mut self) {
            self.calls.push("dialogs_hide")
        }
        fn dialogs_show(&mut self) {
            self.calls.push("dialogs_show")
        }
        fn dialogs_master_changed(&mut self) {
            self.calls.push("dialogs_master_changed")
        }
        fn dialogs_activated(&mut self) {
            self.calls.push("dialogs_activated")
        }
        fn app_lost_focus(&mut self) {
            self.calls.push("app_lost_focus")
        }
    }

    /// Every `ImodvWindow`/`ImodvGL` slot the native driver calls must reach
    /// its source counterpart on the sink, carrying the payload.
    #[test]
    fn imodv_window_slots_reach_every_source_sink_call() {
        let mut app = ImodvApp {
            enable_depth_db: 1,
            dbl_buf: 1,
            delta_rot: 10.,
            ..Default::default()
        };
        let mut window = ImodvWindow::new(&mut app);
        let mut sink = RecordingImodvWindowSink::default();
        window.file_menu_slot(VFILE_MENU_QUIT as i32, &mut sink);
        window.edit_menu_slot(VEDIT_MENU_OBJECTS as i32, &mut sink);
        window.view_menu_slot(VVIEW_MENU_DB as i32, &mut sink);
        window.help_menu_slot(VHELP_MENU_MENUS as i32, &mut sink);
        let press = KeyEvent {
            key: Key::Character('R'),
            x: 11,
            y: 22,
            ..KeyEvent::default()
        };
        window.key_press_event(press, &mut sink);
        window.key_release_event(press, &mut sink);
        window.rotate_clicked(1, 0, 0, &mut sink);
        window.rot_step_changed(1, &mut sink);
        window.movie_but_toggled(true, &mut sink);
        window.timeout_slot(&mut sink);
        window.app_focus_changed(false, &mut sink);
        window.close_event(&mut sink);
        assert_eq!(
            sink.calls,
            [
                "file_menu",
                "edit_menu",
                "view_menu",
                "help_menu",
                "key_press",
                "key_release",
                "rotate_model",
                "control_change_steps",
                "control_start",
                "movie_timeout",
                "app_lost_focus",
                "quit",
            ]
        );
        assert_eq!(sink.events[0], press);
    }

    /// `ImodvGL`'s own handlers: the press/move gate, the wheel payload, and
    /// the first-draw resize dance that the timer drives.
    #[test]
    fn imodv_gl_widget_events_follow_the_source_gating() {
        let mut app = ImodvApp {
            enable_depth_db: 1,
            dbl_buf: 1,
            ..Default::default()
        };
        let mut window = ImodvWindow::new(&mut app);
        window.init_width = 640;
        window.init_height = 480;
        let mut gl = ImodvGl::new(GlFormat::default());
        let mut sink = RecordingImodvWindowSink::default();
        let moved = KeyEvent {
            x: 5,
            y: 6,
            ..KeyEvent::default()
        };
        // A move with no button down is dropped, exactly as `mMousePressed`
        // gates it in the source.
        gl.mouse_move_event(moved, &mut sink);
        assert!(sink.calls.is_empty());
        gl.mouse_press_event(moved, &mut sink);
        gl.mouse_move_event(moved, &mut sink);
        gl.mouse_release_event(moved, &mut sink);
        gl.mouse_move_event(moved, &mut sink);
        gl.wheel_event(
            KeyEvent {
                delta: 120,
                ..KeyEvent::default()
            },
            &mut sink,
        );
        assert_eq!(
            sink.calls,
            ["mouse_press", "mouse_move", "mouse_release", "scroll_wheel"]
        );
        assert_eq!(sink.events.last().expect("wheel payload").delta, 120);

        // `paintGL` starts the first-draw timer and `timerEvent` walks the
        // width down by one on each pass until `updateGL` takes over.
        sink.calls.clear();
        gl.paint_gl(&mut window, &mut sink);
        assert_eq!(
            sink.calls,
            ["start_timer", "device_pixel_ratio", "paint_gl"]
        );
        sink.calls.clear();
        gl.timer_event(&mut window, &mut sink);
        assert_eq!(sink.calls, ["resize_window"]);
        assert_eq!(sink.resizes, [(640 + 1, 480)]);
        gl.first_draw = 0;
        sink.calls.clear();
        gl.timer_event(&mut window, &mut sink);
        assert_eq!(sink.calls, ["kill_timer", "update_gl"]);
    }

    /// `getVisuals` on the `NEW_QTOPENGL` build clears the recorded depths and
    /// asks for the post-creation initialization; `openWindow` then builds the
    /// window, points `a->mainWin` at it and records the caption.
    #[cfg(feature = "three-dmod-gl")]
    #[test]
    fn imodv_native_host_opens_a_model_view_window_the_source_way() {
        use crate::imod::three_dmod::imodv::ImodvNativeBoundary;
        let mut model = crate::imod::libimod::imodel::Imod::default();
        model
            .view
            .push(crate::imod::libimod::imodel::Iview::default());
        let mut app = ImodvApp {
            standalone: 1,
            imod: &mut model,
            ..Default::default()
        };
        let mut host = ImodvNativeHost;
        assert_eq!(host.get_visuals(&mut app, false), 0);
        assert_eq!(app.enable_depth_db, -1);
        assert_eq!(app.need_new_qglinit, 1);
        assert_eq!(
            host.open_model_view(
                &mut app,
                false,
                crate::imod::three_dmod::control::Rect::default()
            ),
            0
        );
        assert!(!app.main_win.is_null());
        // `initializeForNewQGL` recorded the granted visual and cleared the
        // pending flag, so `paintGL` will draw.
        assert_eq!(app.need_new_qglinit, 0);
        assert_eq!(app.enable_depth_db, 1);
        assert_eq!(
            (app.want_winx, app.want_winy),
            (
                crate::imod::three_dmod::imodv::DEFAULT_XSIZE,
                crate::imod::three_dmod::imodv::DEFAULT_YSIZE
            )
        );
        IMODV_HOST_STATE.with(|state| {
            let state = state.borrow();
            assert_eq!(state.title, "3dmodv:");
            assert!(state.window.is_some());
        });
        // `a->mainWin->close()` reaches the loop through the exit flag.
        IMODV_EXIT_REQUESTED.store(false, std::sync::atomic::Ordering::Relaxed);
        host.close_model_view(app.main_win);
        assert!(IMODV_EXIT_REQUESTED.load(std::sync::atomic::Ordering::Relaxed));
        IMODV_HOST_STATE.with(|state| {
            let mut state = state.borrow_mut();
            state.window = None;
            state.title.clear();
        });
        app.main_win = std::ptr::null_mut();
    }

    /// Without the window `openWindow` should have left, `qApp->exec()` has
    /// nothing to run and says so rather than returning success.
    #[cfg(feature = "three-dmod-gl")]
    #[test]
    fn imodv_native_host_run_application_refuses_without_a_window() {
        use crate::imod::three_dmod::imodv::ImodvNativeBoundary;
        IMODV_HOST_STATE.with(|state| state.borrow_mut().window = None);
        assert_eq!(ImodvNativeHost.run_application(), 3);
    }
}
