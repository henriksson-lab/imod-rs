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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KeyEvent {
    pub key: Key,
    pub control: bool,
    pub shift: bool,
    pub alt: bool,
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
        if app.enable_depth_db >= 0 {
            window.dbw =
                Some(window.add_gl_widget_to_stack(true, app.enable_depth_db, false, false));
            window.cur_glw = Some(0);
        }
        if app.enable_depth_dbal >= 0 {
            let gl = window.add_gl_widget_to_stack(true, app.enable_depth_dbal, false, true);
            if window.cur_glw.is_none() {
                window.cur_glw = Some(1);
                app.alpha_visual = 1;
            }
            window.dbalw = Some(gl);
        }
        if window.cur_glw.is_none() && app.enable_depth_dbst >= 0 {
            window.dbstw =
                Some(window.add_gl_widget_to_stack(true, app.enable_depth_dbst, true, false));
            window.cur_glw = Some(3);
        }
        if window.cur_glw.is_none() && app.enable_depth_dbst_al >= 0 {
            window.dbst_alw =
                Some(window.add_gl_widget_to_stack(true, app.enable_depth_dbst_al, true, true));
            window.cur_glw = Some(4);
            app.alpha_visual = 1;
        }
        if app.enable_depth_sb >= 0 {
            let gl = window.add_gl_widget_to_stack(false, app.enable_depth_sb, false, false);
            if window.cur_glw.is_none() {
                window.cur_glw = Some(2);
                app.dbl_buf = 0;
            }
            window.sbw = Some(gl);
        }
        if window.cur_glw.is_none() && app.enable_depth_sbst >= 0 {
            window.sbstw =
                Some(window.add_gl_widget_to_stack(false, app.enable_depth_sbst, true, false));
            window.cur_glw = Some(5);
            app.dbl_buf = 0;
        }
        app.db_possible = app.dbl_buf;
        if app.trans_bkgd != 0 {
            if app.dbl_buf != 0 && window.dbalw.is_some() {
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

/// Native replacement for the `QMainWindow`/`QOpenGLWidget` ownership chain.
///
/// The context is compatibility OpenGL 2.1, deliberately matching the fixed
/// OpenGL calls in `mv_ogl.cpp`.  The callback is invoked while this context is
/// current, exactly as `ImodvGL::paintGL` invokes `imodvPaintGL` under Qt.
#[cfg(feature = "three-dmod-gl")]
pub fn run_native_opengl(
    mut model_window: ImodvWindow,
    mut sink: Box<dyn ImodvWindowSink>,
) -> Result<(), String> {
    use glutin::config::ConfigTemplateBuilder;
    use glutin::context::{ContextApi, ContextAttributesBuilder, Version};
    use glutin::display::{GetGlDisplay, GlDisplay};
    use glutin::prelude::*;
    use glutin::surface::{GlSurface, SurfaceAttributesBuilder, SwapInterval, WindowSurface};
    use glutin_winit::DisplayBuilder;
    use raw_window_handle::HasWindowHandle;
    use std::num::NonZeroU32;
    use winit::dpi::PhysicalSize;
    use winit::event::{
        ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent as WinitWindowEvent,
    };
    use winit::event_loop::EventLoop;
    use winit::window::Window;

    let event_loop = EventLoop::new().map_err(|error| error.to_string())?;
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
    model_window.init_width = width as i32;
    model_window.init_height = height as i32;
    model_window.device_pixel_ratio = window.scale_factor() as f32;
    let mut left_pressed = false;
    event_loop
        .run(move |event, target| match event {
            Event::AboutToWait => window.request_redraw(),
            Event::WindowEvent { event, .. } => match event {
                WinitWindowEvent::CloseRequested => {
                    model_window.close_event(&mut *sink);
                    target.exit();
                }
                WinitWindowEvent::Resized(size) => {
                    if let (Some(width), Some(height)) =
                        (NonZeroU32::new(size.width), NonZeroU32::new(size.height))
                    {
                        surface.resize(&context, width, height);
                        unsafe {
                            glow::HasContext::viewport(
                                &gl,
                                0,
                                0,
                                size.width as i32,
                                size.height as i32,
                            );
                        }
                        model_window.event(
                            WindowEvent::Resize {
                                width: size.width as i32,
                                height: size.height as i32,
                            },
                            &mut *sink,
                        );
                        sink.imodv_resize_gl(size.width as i32, size.height as i32);
                    }
                }
                WinitWindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                    model_window.device_pixel_ratio = scale_factor as f32
                }
                WinitWindowEvent::Focused(focused) => {
                    model_window.app_focus_changed(focused, &mut *sink)
                }
                WinitWindowEvent::RedrawRequested => {
                    // `imodvPaintGL` owns all model fixed-function rendering from mv_gfx/mv_ogl.
                    unsafe {
                        glow::HasContext::clear_color(&gl, 0.0, 0.0, 0.0, 1.0);
                        glow::HasContext::clear(
                            &gl,
                            glow::COLOR_BUFFER_BIT | glow::DEPTH_BUFFER_BIT,
                        );
                    }
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
                    let _ = surface.swap_buffers(&context);
                }
                WinitWindowEvent::MouseInput {
                    state,
                    button: MouseButton::Left,
                    ..
                } => {
                    left_pressed = state == ElementState::Pressed;
                    let event = KeyEvent::default();
                    if left_pressed {
                        sink.imodv_mouse_press(event);
                    } else {
                        sink.imodv_mouse_release(event);
                    }
                }
                WinitWindowEvent::CursorMoved { .. } if left_pressed => {
                    sink.imodv_mouse_move(KeyEvent::default())
                }
                WinitWindowEvent::MouseWheel { delta, .. } => {
                    let event = KeyEvent {
                        key: match delta {
                            MouseScrollDelta::LineDelta(_, y) if y > 0.0 => Key::Plus,
                            MouseScrollDelta::LineDelta(_, _) => Key::Minus,
                            _ => Key::Unknown,
                        },
                        ..KeyEvent::default()
                    };
                    sink.imodv_scroll_wheel(event);
                }
                WinitWindowEvent::KeyboardInput { event, .. } => {
                    let key = match event.logical_key {
                        winit::keyboard::Key::Character(text) => text
                            .chars()
                            .next()
                            .map(|c| Key::Character(c.to_ascii_uppercase()))
                            .unwrap_or(Key::Unknown),
                        winit::keyboard::Key::Named(winit::keyboard::NamedKey::Delete) => {
                            Key::Delete
                        }
                        _ => Key::Unknown,
                    };
                    let payload = KeyEvent {
                        key,
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
            _ => {}
        })
        .map_err(|error| error.to_string())
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
}
