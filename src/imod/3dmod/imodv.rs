//! Translation of `IMOD/3dmod/imodv.cpp` together with `IMOD/3dmod/imodv.h`.
//!
//! `imodv.cpp` is the entry point of the OpenGL model viewer.  The model and
//! view state below is deliberately kept separate from the widget boundary:
//! the latter belongs to `mv_window.cpp` and is not represented by a static
//! or simulated viewer here.
#![allow(dead_code, unused_variables)]

use std::cell::RefCell;
use std::ffi::c_void;

use crate::imod::libimod::imat::{Imat, imod_mat_new};
use crate::imod::libimod::imodel::{Imod, Iobj, Ipoint, Iview};
use crate::imod::libimod::imodel_files::imod_read;
use crate::imod::libimod::iview::{imod_view_default, imod_view_default_scale};
use crate::imod::three_dmod::control::Rect;
use crate::imod::three_dmod::imodview::ImodView;

/// Original: `MAX_MOVIE_TIMES` (`imodv.h:27`).
pub const MAX_MOVIE_TIMES: usize = 10;
/// Original: `DEFAULT_XSIZE` (`imodv.cpp:76`).
pub const DEFAULT_XSIZE: i32 = 512;
/// Original: `DEFAULT_YSIZE` (`imodv.cpp:77`).
pub const DEFAULT_YSIZE: i32 = 512;
/// Original: `IMODV_STEREO_OFF` (`mv_stereo.h`).
pub const IMODV_STEREO_OFF: i32 = 0;

/// `ImodvWindow` from the paired `mv_window.cpp` translation.
pub use crate::imod::three_dmod::mv_window::ImodvWindow;
/// Rust representation of the C++ `QPixmap` ownership boundary.
pub type QPixmap = c_void;
/// Rust representation of the C++ `QColor` ownership boundary.
pub type QColor = c_void;
/// Rust representation of the C++ `VertBufManager` ownership boundary.
pub type VertBufManager = c_void;

/// Original: `__imodv_struct` / `ImodvApp` (`imodv.h`).
///
/// The object vectors are owned by `libimod`; `imodv.cpp` only selects current
/// elements.  Model-pick coordinates are owned by the application.
pub struct ImodvApp {
    /// Borrowed cursors used by the viewer and rendering boundaries.  Models
    /// loaded by standalone `3dmodv` are kept alive by `owned_models`; models
    /// supplied by the regular image viewer remain borrowed from that viewer.
    pub mod_: Vec<*mut Imod>,
    pub(crate) owned_models: Vec<Box<Imod>>,
    pub imod: *mut Imod,
    pub num_mods: i32,
    pub cur_mod: i32,
    pub obj: *mut Iobj,
    pub obj_num: i32,
    pub mat: Option<Imat>,
    pub rmat: Option<Imat>,
    pub wpid: i32,
    pub main_win: *mut ImodvWindow,
    pub icon_pixmap: *mut QPixmap,
    pub rbgname: String,
    pub rbgcolor: *mut QColor,
    pub enable_depth_sb: i32,
    pub enable_depth_db: i32,
    pub enable_depth_sbst: i32,
    pub enable_depth_dbst: i32,
    pub enable_depth_dbal: i32,
    pub enable_depth_dbst_al: i32,
    pub need_new_qglinit: i32,
    pub cnear: i32,
    pub cfar: i32,
    pub fovy: i32,
    pub dbl_buf: i32,
    pub db_possible: i32,
    pub winx: i32,
    pub winy: i32,
    pub want_winx: i32,
    pub want_winy: i32,
    pub lastmx: i32,
    pub lastmy: i32,
    pub lightx: i32,
    pub lighty: i32,
    pub mousemove: i32,
    pub stereo: i32,
    pub alpha_visual: i32,
    pub trans_bkgd: i32,
    pub clear_after_stereo: i32,
    pub plax: f32,
    pub image_stereo: i32,
    pub images_per_area: i32,
    pub image_delta_z: i32,
    pub delta_rot: f32,
    pub movie: i32,
    pub xrot_movie: f32,
    pub yrot_movie: f32,
    pub zrot_movie: f32,
    pub drawall: i32,
    pub alpha: i32,
    pub current_subset: i32,
    pub draw_extra_only: i32,
    pub movie_frames: i32,
    pub movie_start: i32,
    pub movie_current: i32,
    pub movie_speed: f32,
    pub throw_factor: f32,
    pub movie_times: [i32; MAX_MOVIE_TIMES],
    pub snap_fileno: i32,
    pub draw_clip: i32,
    pub draw_light: i32,
    pub draw_slicer_plane: i32,
    pub draw_labels: i32,
    pub link_to_slicer: i32,
    pub link_slicer_center: i32,
    pub sync_objed_to_cur_obj: i32,
    pub scale_bar_size: f32,
    pub bound_box_extra_obj: i32,
    pub cur_point_extra_obj: i32,
    pub obj_bound_extra_obj: i32,
    pub moveall: i32,
    pub crosset: i32,
    pub fullscreen: i32,
    pub standalone: i32,
    pub view: Iview,
    pub vert_buf_ok: i32,
    pub prim_restart_ok: i32,
    pub gl_ext_flags: i32,
    pub vb_manager: *mut VertBufManager,
    pub tex_map: i32,
    pub tex_trans: i32,
    /// The active image view is owned by the normal 3dmod viewer, except for
    /// standalone `3dmodv`, whose value is retained in `owned_vi` below.
    pub vi: *mut ImodView,
    pub(crate) owned_vi: Option<Box<ImodView>>,
    pub do_pick: i32,
    pub x_pick: i32,
    pub y_pick: i32,
    pub w_pick: i32,
    pub h_pick: i32,
    pub pick_hits: i32,
    pub read_pix_for_pick: i32,
    pub mod_picks: Vec<Ipoint>,
    pub legacy_pick_mode: i32,
    pub lighting: i32,
    pub depthcue: i32,
    pub wireframe: i32,
    pub lowres: i32,
    pub invert_z: i32,
}

impl Default for ImodvApp {
    fn default() -> Self {
        Self {
            mod_: Vec::new(),
            owned_models: Vec::new(),
            imod: std::ptr::null_mut(),
            num_mods: 0,
            cur_mod: 0,
            obj: std::ptr::null_mut(),
            obj_num: 0,
            mat: None,
            rmat: None,
            wpid: 0,
            main_win: std::ptr::null_mut(),
            icon_pixmap: std::ptr::null_mut(),
            rbgname: String::new(),
            rbgcolor: std::ptr::null_mut(),
            enable_depth_sb: 0,
            enable_depth_db: 0,
            enable_depth_sbst: 0,
            enable_depth_dbst: 0,
            enable_depth_dbal: 0,
            enable_depth_dbst_al: 0,
            need_new_qglinit: 0,
            cnear: 0,
            cfar: 0,
            fovy: 0,
            dbl_buf: 0,
            db_possible: 0,
            winx: 0,
            winy: 0,
            want_winx: 0,
            want_winy: 0,
            lastmx: 0,
            lastmy: 0,
            lightx: 0,
            lighty: 0,
            mousemove: 0,
            stereo: 0,
            alpha_visual: 0,
            trans_bkgd: 0,
            clear_after_stereo: 0,
            plax: 0.,
            image_stereo: 0,
            images_per_area: 0,
            image_delta_z: 0,
            delta_rot: 0.,
            movie: 0,
            xrot_movie: 0.,
            yrot_movie: 0.,
            zrot_movie: 0.,
            drawall: 0,
            alpha: 0,
            current_subset: 0,
            draw_extra_only: 0,
            movie_frames: 0,
            movie_start: 0,
            movie_current: 0,
            movie_speed: 0.,
            throw_factor: 0.,
            movie_times: [0; MAX_MOVIE_TIMES],
            snap_fileno: 0,
            draw_clip: 0,
            draw_light: 0,
            draw_slicer_plane: 0,
            draw_labels: 0,
            link_to_slicer: 0,
            link_slicer_center: 0,
            sync_objed_to_cur_obj: 0,
            scale_bar_size: 0.,
            bound_box_extra_obj: 0,
            cur_point_extra_obj: 0,
            obj_bound_extra_obj: 0,
            moveall: 0,
            crosset: 0,
            fullscreen: 0,
            standalone: 0,
            view: Iview::default(),
            // `imodv.cpp:179`: `a->vertBufOK = -2;`.  The sentinel matters --
            // `imodvPaintGL` probes only while it is `< -1` (`mv_gfx.cpp:234`),
            // so initialising it to 0 skipped the probe permanently.
            vert_buf_ok: -2,
            prim_restart_ok: 0,
            gl_ext_flags: 0,
            vb_manager: std::ptr::null_mut(),
            tex_map: 0,
            tex_trans: 0,
            vi: std::ptr::null_mut(),
            owned_vi: None,
            do_pick: 0,
            x_pick: 0,
            y_pick: 0,
            w_pick: 0,
            h_pick: 0,
            pick_hits: 0,
            read_pix_for_pick: 0,
            mod_picks: Vec::new(),
            legacy_pick_mode: 0,
            lighting: 0,
            depthcue: 0,
            wireframe: 0,
            lowres: 0,
            invert_z: 0,
        }
    }
}

/// Direct native operations called by source `imodv.cpp` entry points.
pub trait ImodvNativeBoundary {
    fn current_model_view(&mut self) -> *mut ImodView;
    fn model_view_icon(&mut self) -> *mut QPixmap;
    fn raise_model_view(&mut self, window: *mut ImodvWindow);
    fn get_visuals(&mut self, app: &mut ImodvApp, once_opened: bool) -> i32;
    fn open_model_view(
        &mut self,
        app: &mut ImodvApp,
        once_opened: bool,
        last_geometry: Rect,
    ) -> i32;
    fn imod_draw(&mut self, view: *mut ImodView, flags: i32);
    fn object_edit_draw(&mut self);
    fn info_set_object_color(&mut self);
    fn set_modv_dialog_title(&mut self, window: *mut ImodvWindow, title: &str);
    fn undo_model_change(&mut self, view: *mut ImodView);
    fn undo_object_prop_change(&mut self, view: *mut ImodView, object: i32);
    fn undo_finish_unit(&mut self, view: *mut ImodView);
    fn close_model_view(&mut self, window: *mut ImodvWindow);
    fn imodv_draw(&mut self, app: &mut ImodvApp);
    fn restorable_geometry(&mut self, window: *mut ImodvWindow) -> Rect;
    fn record_mod_view_geometry(&mut self, geometry: Rect);
    fn vb_cleanup_vbd(&mut self, imod: *mut Imod);
    fn mv_image_cleanup(&mut self);
    fn free_extra_object(&mut self, view: *mut ImodView, object: i32);
    fn start_clip_disconnect(&mut self);
    fn stereo_hw_off(&mut self);
    fn close_model_view_dialogs(&mut self);
    fn save_settings(&mut self);
    fn delete_imod_help(&mut self);
    fn wait_for_clip_disconnect(&mut self);
    fn exit_application(&mut self);
    fn check_for_exit_on_close(&mut self);
    fn set_app_exiting(&mut self);
    fn run_application(&mut self) -> i32;
    fn print_window_id(&mut self, window: *mut ImodvWindow);
    fn create_clipboard(&mut self, use_stdin: bool);
    fn open_selected_windows(&mut self, window_keys: Option<&str>);
    fn show_usage(&mut self, usage: &str);
    fn show_error(&mut self, message: &str);
}

/// Native top-slicer operation reached by `imodvNewModelAngles`.
///
/// This stays a single boundary because `setTopSlicerFromModelView` owns the
/// top-window lookup, center linking, widget updates, synchronization, and
/// drawing in the paired `slicer.cpp` translation.
pub trait ImodvSlicerAngleBoundary {
    fn set_top_slicer_from_model_view(&mut self, rot: &Ipoint);
}

thread_local! {
    /// Original globals: `ImodvStruct`, `Imodv`, and `ImodvClosed`.
    static IMODV_STATE: RefCell<(ImodvApp, i32, bool, Rect)> =
        RefCell::new((ImodvApp::default(), 1, false, Rect::default()));
    /// The C++ call resolves the top slicer through the UI-thread global.
    /// The attached Rust/Qt host supplies the equivalent thread-local route.
    pub static IMODV_SLICER_ANGLE_BOUNDARY: RefCell<Option<Box<dyn ImodvSlicerAngleBoundary>>> =
        RefCell::new(None);
    pub static IMODV_NATIVE_BOUNDARY: RefCell<Option<Box<dyn ImodvNativeBoundary>>> =
        RefCell::new(None);
}

/// Original static: `usage` (`imodv.cpp:90`).
pub fn usage(pname: &str) -> String {
    format!(
        "{pname}\noptions: all Qt options plus:\n\\t-f               Open window to max size.\n\\t-b color_name    Background color for rendering.\n\\t-s width,height  Window size in pixels.\n\\t-E <keys>        Open windows specifed by key letters (= hot keys).\n\\t-D               Debug mode.\n\\t-h               Print this help message.\n"
    )
}

/// Original static: `imodv_init` (`imodv.cpp:110`).
pub fn imodv_init(a: &mut ImodvApp) -> i32 {
    a.num_mods = 0;
    a.cur_mod = 0;
    a.mod_.clear();
    a.owned_models.clear();
    a.imod = std::ptr::null_mut();
    a.vi = std::ptr::null_mut();
    a.owned_vi = None;
    a.mat = imod_mat_new(3);
    a.rmat = imod_mat_new(3);
    a.obj = std::ptr::null_mut();
    a.obj_num = 0;
    a.cnear = 0;
    a.cfar = 1000;
    a.fovy = 0;
    a.movie = 0;
    a.movie_frames = 0;
    a.movie_speed = 36.;
    a.snap_fileno = 0;
    a.wpid = 0;
    a.stereo = IMODV_STEREO_OFF;
    a.clear_after_stereo = 0;
    a.trans_bkgd = 0;
    a.alpha_visual = 0;
    a.plax = 5.;
    a.image_stereo = 0;
    a.images_per_area = 2;
    a.image_delta_z = 1;
    a.lightx = 0;
    a.lighty = 0;
    a.rbgname = "black".into();
    a.want_winx = 0;
    a.want_winy = 0;
    a.delta_rot = 10.;
    a.xrot_movie = 0.;
    a.yrot_movie = 0.;
    a.zrot_movie = 0.;
    a.current_subset = 0;
    a.read_pix_for_pick = 0;
    a.mod_picks.clear();
    a.crosset = 0;
    a.fullscreen = 0;
    a.drawall = 0;
    a.moveall = 1;
    a.alpha = 0;
    imod_view_default(&mut a.view);
    a.view.cnear = 0.;
    a.view.cfar = 1.;
    a.do_pick = 0;
    a.w_pick = 5;
    a.h_pick = 5;
    a.lighting = 1;
    a.depthcue = 0;
    a.wireframe = 0;
    a.draw_labels = 0;
    a.lowres = 0;
    a.invert_z = 0;
    a.draw_clip = 0;
    a.draw_slicer_plane = 0;
    a.draw_light = 0;
    a.link_to_slicer = 0;
    a.link_slicer_center = 1;
    a.bound_box_extra_obj = 0;
    a.obj_bound_extra_obj = 0;
    a.cur_point_extra_obj = 0;
    a.vert_buf_ok = -2;
    a.prim_restart_ok = 0;
    a.main_win = std::ptr::null_mut();
    0
}

/// Original static: `initstruct` (`imodv.cpp:191`).
///
/// This is the state transition used when the model viewer is opened from an
/// existing 3dmod image view, as opposed to standalone `3dmodv`.
pub unsafe fn initstruct(vw: &mut ImodView, a: &mut ImodvApp) {
    imodv_init(a);
    a.num_mods = 1;
    a.mod_.push(vw.imod);
    a.imod = vw.imod;
    if !a.imod.is_null()
        && (*a.imod).cindex.object >= 0
        && ((*a.imod).cindex.object as usize) < (*a.imod).obj.len()
    {
        a.obj_num = (*a.imod).cindex.object;
        a.obj = &mut (&mut (*a.imod).obj)[a.obj_num as usize];
    }
    a.fullscreen = 0;
    a.standalone = 0;
    a.tex_map = 0;
    a.tex_trans = 0;
    a.owned_vi = None;
    a.vi = vw;
    if a.imod.is_null() {
        return;
    }
    let image_max = Ipoint {
        x: vw.xsize as f32,
        y: vw.ysize as f32,
        z: vw.zsize as f32,
    };
    let bin_scale = vw.zbin as f32 / vw.xybin as f32;
    let view_count = (*a.imod).view.len();
    for i in 0..view_count {
        let imod = a.imod as *const Imod;
        let view = &mut (&mut (*a.imod).view)[i];
        imod_view_default_scale(&*imod, view, &image_max, bin_scale);
    }
}

/// Original static: `load_models` (`imodv.cpp:483`).
pub unsafe fn load_models(n: i32, fname: &[Vec<u8>], a: &mut ImodvApp) -> i32 {
    if n < 1 {
        return 0;
    }
    a.mod_.clear();
    a.num_mods = n;
    a.cur_mod = 0;
    for i in 0..n {
        let path = String::from_utf8_lossy(&fname[i as usize]);
        let Ok(model) = imod_read(path.as_ref()) else {
            return -1;
        };
        a.owned_models.push(Box::new(model));
        let model = a
            .owned_models
            .last_mut()
            .expect("model was just pushed")
            .as_mut();
        let model = model as *mut Imod;
        a.mod_.push(model);
        let image_max = Ipoint::default();
        let view_count = (*model).view.len();
        for j in 0..view_count {
            let imod = model as *const Imod;
            let view = &mut (&mut (*model).view)[j];
            imod_view_default_scale(&*imod, view, &image_max, 1.);
        }
    }
    a.imod = a.mod_[0];
    if (*a.imod).cindex.object >= 0 && ((*a.imod).cindex.object as usize) < (*a.imod).obj.len() {
        a.obj_num = (*a.imod).cindex.object;
        a.obj = &mut (&mut (*a.imod).obj)[a.obj_num as usize];
    }
    0
}

/// Original static: `getVisuals` (`imodv.cpp:236`).
///
/// Visual selection needs the actual Qt/OpenGL probe in `mv_window.cpp`.
/// It deliberately reports unsupported instead of claiming a fake visual.
pub fn get_visuals(a: &mut ImodvApp) -> i32 {
    let once_opened = IMODV_STATE.with(|state| state.borrow().2);
    IMODV_NATIVE_BOUNDARY.with(|slot| {
        slot.borrow_mut()
            .as_deref_mut()
            .map_or(1, |boundary| boundary.get_visuals(a, once_opened))
    })
}

/// Original static: `openWindow` (`imodv.cpp:398`).
/// The actual OpenGL/QWidget constructor is retained as an unported boundary.
pub fn open_window(a: &mut ImodvApp) -> i32 {
    let (once_opened, last_geometry) = IMODV_STATE.with(|state| {
        let state = state.borrow();
        (state.2, state.3)
    });
    IMODV_NATIVE_BOUNDARY.with(|slot| {
        slot.borrow_mut().as_deref_mut().map_or(1, |boundary| {
            boundary.open_model_view(a, once_opened, last_geometry)
        })
    })
}

/// Original: `imodvMain` (`imodv.cpp:536`).
pub unsafe fn imodv_main(argc: i32, argv: &[Vec<u8>]) -> i32 {
    /// Not a status the source returns: it marks that everything before
    /// `qApp->exec()` succeeded and the event loop is still to be entered.
    const ENTER_EVENT_LOOP: i32 = -1;
    let status = IMODV_NATIVE_BOUNDARY.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(boundary) = slot.as_deref_mut() else {
            return 3;
        };
        IMODV_STATE.with(|state| {
            let mut state = state.borrow_mut();
            let once_opened = state.2;
            let last_geometry = state.3;
            let a = &mut state.0;
            a.standalone = 1;
            imodv_init(a);
            let mut i = 1;
            let mut use_stdin = false;
            let mut print_id = false;
            let mut window_keys = None;
            while i < argc {
                let argument = argv[i as usize].as_slice();
                if !argument.starts_with(b"-") {
                    break;
                }
                match argument {
                    b"-b" => {
                        i += 1;
                        if i >= argc {
                            return 1;
                        }
                        a.rbgname = String::from_utf8_lossy(&argv[i as usize]).into_owned();
                    }
                    b"-D" => crate::imod::three_dmod::imod::IMOD_DEBUG
                        .store(true, std::sync::atomic::Ordering::Relaxed),
                    b"-f" => a.fullscreen = 1,
                    b"-s" => {
                        i += 1;
                        if i >= argc {
                            return 1;
                        }
                        let text = String::from_utf8_lossy(&argv[i as usize]);
                        let mut values = text.split(|c| c == ',' || c == 'x');
                        a.want_winx = values.next().and_then(|v| v.parse().ok()).unwrap_or(0);
                        a.want_winy = values.next().and_then(|v| v.parse().ok()).unwrap_or(0);
                    }
                    b"-W" => print_id = true,
                    b"-E" => {
                        i += 1;
                        if i >= argc {
                            return 1;
                        }
                        window_keys = Some(String::from_utf8_lossy(&argv[i as usize]).into_owned());
                    }
                    b"-L" => use_stdin = true,
                    b"-h" => {
                        boundary.show_usage(&usage(&String::from_utf8_lossy(&argv[0])));
                        return 1;
                    }
                    b"-modv" | b"-view" => {}
                    _ => {
                        boundary.show_error(&format!(
                            "3dmodv error: illegal option {}\n",
                            String::from_utf8_lossy(argument)
                        ));
                        return 1;
                    }
                }
                i += 1;
            }
            a.dbl_buf = 1;
            a.owned_vi = Some(Box::new(ImodView::default()));
            a.vi = a
                .owned_vi
                .as_deref_mut()
                .expect("standalone view was just created");
            if boundary.get_visuals(a, once_opened) != 0 {
                return 3;
            }
            if argc - i < 1 || load_models(argc - i, &argv[i as usize..], a) != 0 {
                return 3;
            }
            unsafe { (*a.vi).imod = a.imod };
            a.icon_pixmap = boundary.model_view_icon();
            if boundary.open_model_view(a, once_opened, last_geometry) != 0 {
                return 3;
            }
            let main_window = a.main_win;
            state.1 = 0;
            if print_id {
                boundary.print_window_id(main_window);
            }
            if print_id || use_stdin {
                boundary.create_clipboard(use_stdin);
            }
            boundary.open_selected_windows(window_keys.as_deref());
            ENTER_EVENT_LOOP
        })
    });
    if status != ENTER_EVENT_LOOP {
        return status;
    }
    // `return qApp->exec();` (`imodv.cpp:648`).
    //
    // The event loop re-enters this unit for the life of the window —
    // `imodvDraw`, `imodvSetCaption`, `imodvQuit` — and each of those borrows
    // `IMODV_STATE` and `IMODV_NATIVE_BOUNDARY` again, so neither may still be
    // borrowed while the loop runs.  The host object stays installed, exactly
    // as the one `QApplication` stays reachable through `qApp`, and is called
    // through the pointer the cell holds rather than through a live borrow of
    // it.
    let host: Option<*mut dyn ImodvNativeBoundary> = IMODV_NATIVE_BOUNDARY.with(|slot| {
        slot.borrow_mut()
            .as_deref_mut()
            .map(|boundary| boundary as *mut dyn ImodvNativeBoundary)
    });
    let Some(host) = host else {
        return 3;
    };
    unsafe { (*host).run_application() }
}

/// Original: `imodv_open` (`imodv.cpp:653`).
pub fn imodv_open() {
    IMODV_NATIVE_BOUNDARY.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(boundary) = slot.as_deref_mut() else {
            return;
        };
        let vw = boundary.current_model_view();
        if vw.is_null() || unsafe { (*vw).imod.is_null() } {
            return;
        }
        IMODV_STATE.with(|state| {
            let mut state = state.borrow_mut();
            if state.1 == 0 {
                boundary.raise_model_view(state.0.main_win);
                return;
            }
            let once_opened = state.2;
            let old_trans_bkgd = state.0.trans_bkgd;
            unsafe { initstruct(&mut *vw, &mut state.0) };
            if once_opened {
                state.0.trans_bkgd = old_trans_bkgd;
            }
            state.0.icon_pixmap = boundary.model_view_icon();
            if boundary.get_visuals(&mut state.0, once_opened) != 0 {
                state.0.mat = None;
                state.0.rmat = None;
                return;
            }
            let last_geometry = state.3;
            if boundary.open_model_view(&mut state.0, once_opened, last_geometry) == 0 {
                state.1 = 0;
            }
        });
    });
}
/// Original: `imodv_close` (`imodv.cpp:696`).
pub fn imodv_close() {
    IMODV_STATE.with(|state| {
        let state = state.borrow();
        if state.1 != 0 {
            return;
        }
        IMODV_NATIVE_BOUNDARY.with(|slot| {
            if let Some(boundary) = slot.borrow_mut().as_deref_mut() {
                boundary.close_model_view(state.0.main_win);
            }
        });
    });
}
/// Original: `imodv_draw` (`imodv.cpp:703`).
pub unsafe fn imodv_draw() {
    IMODV_STATE.with(|state| {
        let mut state = state.borrow_mut();
        if state.1 != 0 || state.0.imod.is_null() {
            return;
        }
        let imod = &mut *state.0.imod;
        if state.0.sync_objed_to_cur_obj != 0
            && state.0.standalone == 0
            && imod.cindex.object >= 0
            && (imod.cindex.object as usize) < imod.obj.len()
            && state.0.obj_num != imod.cindex.object
        {
            state.0.obj_num = imod.cindex.object;
            state.0.obj = &mut imod.obj[imod.cindex.object as usize];
        }
        IMODV_NATIVE_BOUNDARY.with(|slot| {
            if let Some(boundary) = slot.borrow_mut().as_deref_mut() {
                boundary.imodv_draw(&mut state.0);
            }
        });
    });
}
/// Original: `imodv_new_model` (`imodv.cpp:715`).
pub unsafe fn imodv_new_model(mod_: *mut Imod) {
    IMODV_STATE.with(|state| {
        let mut state = state.borrow_mut();
        if state.1 != 0 {
            return;
        }
        state.0.imod = mod_;
        if !state.0.mod_.is_empty() {
            state.0.mod_[0] = mod_;
        }
        if mod_.is_null() || state.0.vi.is_null() {
            return;
        }
        let vi = &*state.0.vi;
        let image_max = Ipoint {
            x: vi.xsize as f32,
            y: vi.ysize as f32,
            z: vi.zsize as f32,
        };
        let bin_scale = vi.zbin as f32 / vi.xybin as f32;
        let view_count = (*mod_).view.len();
        for i in 0..view_count {
            let imod = mod_ as *const Imod;
            let view = &mut (&mut (*mod_).view)[i];
            imod_view_default_scale(&*imod, view, &image_max, bin_scale);
        }
    });
}
/// Original: `imodvLinkedToSlicer` (`imodv.cpp:741`).
pub fn imodv_linked_to_slicer() -> i32 {
    IMODV_STATE.with(|state| {
        let state = state.borrow();
        if state.1 != 0 {
            0
        } else {
            state.0.link_to_slicer
        }
    })
}
/// Original: `imodvDrawSlicerPlane` (`imodv.cpp:748`).
pub fn imodv_draw_slicer_plane() -> i32 {
    IMODV_STATE.with(|state| {
        let state = state.borrow();
        if state.1 != 0 {
            0
        } else {
            state.0.draw_slicer_plane & 1
        }
    })
}
/// Original: `imodvRotCenterLinked` (`imodv.cpp:755`).
pub fn imodv_rot_center_linked() -> i32 {
    IMODV_STATE.with(|state| {
        let state = state.borrow();
        if state.1 != 0 {
            0
        } else {
            (state.0.link_slicer_center != 0 && state.0.link_to_slicer != 0) as i32
        }
    })
}
/// Original: `imodvStandalone` (`imodv.cpp:762`).
pub fn imodv_standalone() -> i32 {
    IMODV_STATE.with(|state| {
        let state = state.borrow();
        if state.1 != 0 { 0 } else { state.0.standalone }
    })
}
/// Original: `imodvNewModelAngles` (`imodv.cpp:769`).
pub fn imodv_new_model_angles(rot: &Ipoint) {
    let linked = IMODV_STATE.with(|state| state.borrow().0.link_to_slicer != 0);
    if linked {
        IMODV_SLICER_ANGLE_BOUNDARY.with(|slot| {
            if let Some(boundary) = slot.borrow_mut().as_deref_mut() {
                boundary.set_top_slicer_from_model_view(rot);
            }
        });
    }
}
/// Original: `imodvSetCaption` (`imodv.cpp:775`).
pub fn imodv_set_caption() {
    IMODV_STATE.with(|state| {
        let state = state.borrow();
        if state.1 != 0 {
            return;
        }
        let a = &state.0;
        IMODV_NATIVE_BOUNDARY.with(|slot| {
            if let Some(boundary) = slot.borrow_mut().as_deref_mut() {
                boundary.set_modv_dialog_title(
                    a.main_win,
                    if a.standalone != 0 {
                        "3dmodv:"
                    } else {
                        "3dmod Model View: "
                    },
                );
            }
        });
    });
}
/// Original: `imodvDrawImodImages` (`imodv.cpp:786`).
pub fn imodv_draw_imod_images(skip_draw: i32) {
    IMODV_STATE.with(|state| {
        let state = state.borrow();
        let a = &state.0;
        if a.standalone != 0 {
            return;
        }
        IMODV_NATIVE_BOUNDARY.with(|slot| {
            if let Some(boundary) = slot.borrow_mut().as_deref_mut() {
                if skip_draw == 0 {
                    boundary.imod_draw(
                        a.vi,
                        crate::imod::three_dmod::imod::IMOD_DRAW_MOD
                            | crate::imod::three_dmod::imod::IMOD_DRAW_SKIPMODV,
                    );
                }
                boundary.object_edit_draw();
                boundary.info_set_object_color();
            }
        });
    });
}
/// Original: `imodvByteImagesExist` (`imodv.cpp:797`).
pub fn imodv_byte_images_exist() -> i32 {
    IMODV_STATE.with(|state| {
        let state = state.borrow();
        if state.0.standalone != 0 || state.0.vi.is_null() {
            return 0;
        }
        let vi = unsafe { &*state.0.vi };
        (vi.rgb_store == 0 && vi.fake_image == 0) as i32
    })
}
/// Original: `imodvRegisterModelChg` (`imodv.cpp:807`).
pub fn imodv_register_model_chg() {
    IMODV_STATE.with(|state| {
        let state = state.borrow();
        IMODV_NATIVE_BOUNDARY.with(|slot| {
            if let Some(boundary) = slot.borrow_mut().as_deref_mut() {
                boundary.undo_model_change(state.0.vi);
            }
        });
    });
}
/// Original: `imodvRegisterObjectChg` (`imodv.cpp:814`).
pub fn imodv_register_object_chg(object: i32) {
    IMODV_STATE.with(|state| {
        let state = state.borrow();
        if state.0.imod.is_null() || object >= unsafe { (*state.0.imod).obj.len() as i32 } {
            return;
        }
        IMODV_NATIVE_BOUNDARY.with(|slot| {
            if let Some(boundary) = slot.borrow_mut().as_deref_mut() {
                boundary.undo_object_prop_change(state.0.vi, object);
            }
        });
    });
}
/// Original: `imodvFinishChgUnit` (`imodv.cpp:821`).
pub fn imodv_finish_chg_unit() {
    IMODV_STATE.with(|state| {
        let state = state.borrow();
        IMODV_NATIVE_BOUNDARY.with(|slot| {
            if let Some(boundary) = slot.borrow_mut().as_deref_mut() {
                boundary.undo_finish_unit(state.0.vi);
            }
        });
    });
}
/// Original: `imodvQuit` (`imodv.cpp:829`).
pub fn imodv_quit() {
    IMODV_STATE.with(|state| {
        let mut state = state.borrow_mut();
        state.1 = 1;
        state.2 = true;
        IMODV_NATIVE_BOUNDARY.with(|slot| {
            if let Some(boundary) = slot.borrow_mut().as_deref_mut() {
                if state.0.standalone != 0 {
                    boundary.set_app_exiting();
                }
                let geometry = boundary.restorable_geometry(state.0.main_win);
                state.3 = geometry;
                if state.0.standalone == 0 {
                    boundary.record_mod_view_geometry(geometry);
                }
                let a = &mut state.0;
                boundary.vb_cleanup_vbd(a.imod);
                boundary.mv_image_cleanup();
                if a.bound_box_extra_obj > 0 {
                    boundary.free_extra_object(a.vi, a.bound_box_extra_obj);
                }
                if a.cur_point_extra_obj > 0 {
                    boundary.free_extra_object(a.vi, a.cur_point_extra_obj);
                }
                if a.standalone != 0 {
                    boundary.start_clip_disconnect();
                }
                boundary.stereo_hw_off();
                boundary.close_model_view_dialogs();
                if a.standalone != 0 {
                    boundary.save_settings();
                    boundary.delete_imod_help();
                    boundary.wait_for_clip_disconnect();
                    boundary.exit_application();
                } else {
                    boundary.check_for_exit_on_close();
                }
            }
        });
        state.0.mat = None;
        state.0.rmat = None;
        state.0.rbgcolor = std::ptr::null_mut();
        state.0.main_win = std::ptr::null_mut();
        state.0.owned_vi = None;
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;

    struct ImodvHost {
        calls: Rc<RefCell<Vec<&'static str>>>,
        current_view: *mut ImodView,
        visual_status: i32,
        window_status: i32,
    }

    impl ImodvNativeBoundary for ImodvHost {
        fn current_model_view(&mut self) -> *mut ImodView {
            self.current_view
        }
        fn model_view_icon(&mut self) -> *mut QPixmap {
            std::ptr::null_mut()
        }
        fn raise_model_view(&mut self, _: *mut ImodvWindow) {
            self.calls.borrow_mut().push("raise");
        }
        fn get_visuals(&mut self, _: &mut ImodvApp, _: bool) -> i32 {
            self.calls.borrow_mut().push("visuals");
            self.visual_status
        }
        fn open_model_view(&mut self, _: &mut ImodvApp, _: bool, _: Rect) -> i32 {
            self.calls.borrow_mut().push("open");
            self.window_status
        }
        fn imod_draw(&mut self, _: *mut ImodView, _: i32) {
            self.calls.borrow_mut().push("draw");
        }
        fn object_edit_draw(&mut self) {
            self.calls.borrow_mut().push("object_edit");
        }
        fn info_set_object_color(&mut self) {
            self.calls.borrow_mut().push("info_color");
        }
        fn set_modv_dialog_title(&mut self, _: *mut ImodvWindow, title: &str) {
            self.calls.borrow_mut().push(match title {
                "3dmodv:" => "standalone_title",
                "3dmod Model View: " => "model_view_title",
                _ => "unexpected_title",
            });
        }
        fn undo_model_change(&mut self, _: *mut ImodView) {
            self.calls.borrow_mut().push("model_change");
        }
        fn undo_object_prop_change(&mut self, _: *mut ImodView, object: i32) {
            self.calls.borrow_mut().push(match object {
                0 => "object_change_0",
                _ => "unexpected_object",
            });
        }
        fn undo_finish_unit(&mut self, _: *mut ImodView) {
            self.calls.borrow_mut().push("finish_unit");
        }
        fn close_model_view(&mut self, _: *mut ImodvWindow) {
            self.calls.borrow_mut().push("close");
        }
        fn imodv_draw(&mut self, _: &mut ImodvApp) {
            self.calls.borrow_mut().push("draw_model_view");
        }
        fn restorable_geometry(&mut self, _: *mut ImodvWindow) -> Rect {
            self.calls.borrow_mut().push("geometry");
            Rect {
                x: 1,
                y: 2,
                width: 3,
                height: 4,
            }
        }
        fn record_mod_view_geometry(&mut self, _: Rect) {
            self.calls.borrow_mut().push("record_geometry");
        }
        fn vb_cleanup_vbd(&mut self, _: *mut Imod) {
            self.calls.borrow_mut().push("vb_cleanup");
        }
        fn mv_image_cleanup(&mut self) {
            self.calls.borrow_mut().push("image_cleanup");
        }
        fn free_extra_object(&mut self, _: *mut ImodView, object: i32) {
            self.calls.borrow_mut().push(match object {
                1 => "free_box",
                2 => "free_point",
                _ => "unexpected_extra",
            });
        }
        fn start_clip_disconnect(&mut self) {
            self.calls.borrow_mut().push("start_disconnect");
        }
        fn stereo_hw_off(&mut self) {
            self.calls.borrow_mut().push("stereo_off");
        }
        fn close_model_view_dialogs(&mut self) {
            self.calls.borrow_mut().push("close_dialogs");
        }
        fn save_settings(&mut self) {
            self.calls.borrow_mut().push("save_settings");
        }
        fn delete_imod_help(&mut self) {
            self.calls.borrow_mut().push("delete_help");
        }
        fn wait_for_clip_disconnect(&mut self) {
            self.calls.borrow_mut().push("wait_disconnect");
        }
        fn exit_application(&mut self) {
            self.calls.borrow_mut().push("exit");
        }
        fn check_for_exit_on_close(&mut self) {
            self.calls.borrow_mut().push("check_exit");
        }
        fn set_app_exiting(&mut self) {
            self.calls.borrow_mut().push("set_exiting");
        }
        fn run_application(&mut self) -> i32 {
            self.calls.borrow_mut().push("run");
            0
        }
        fn print_window_id(&mut self, _: *mut ImodvWindow) {
            self.calls.borrow_mut().push("window_id");
        }
        fn create_clipboard(&mut self, use_stdin: bool) {
            self.calls.borrow_mut().push(if use_stdin {
                "clipboard_stdin"
            } else {
                "clipboard"
            });
        }
        fn open_selected_windows(&mut self, window_keys: Option<&str>) {
            self.calls.borrow_mut().push(match window_keys {
                Some("ZS") => "selected_zs",
                Some(_) => "unexpected_selected",
                None => "selected_none",
            });
        }
        fn show_usage(&mut self, _: &str) {
            self.calls.borrow_mut().push("usage");
        }
        fn show_error(&mut self, _: &str) {
            self.calls.borrow_mut().push("error");
        }
    }

    #[test]
    fn draw_imod_images_preserves_skip_and_standalone_source_branches() {
        let mut view = ImodView::default();
        IMODV_STATE.with(|state| {
            let mut state = state.borrow_mut();
            state.0.standalone = 0;
            state.0.vi = &mut view;
        });
        let calls = Rc::new(RefCell::new(Vec::new()));
        IMODV_NATIVE_BOUNDARY.with(|slot| {
            *slot.borrow_mut() = Some(Box::new(ImodvHost {
                calls: calls.clone(),
                current_view: std::ptr::null_mut(),
                visual_status: 0,
                window_status: 0,
            }));
        });
        imodv_draw_imod_images(0);
        assert_eq!(
            calls.borrow().as_slice(),
            ["draw", "object_edit", "info_color"]
        );
        calls.borrow_mut().clear();
        imodv_draw_imod_images(1);
        assert_eq!(calls.borrow().as_slice(), ["object_edit", "info_color"]);

        IMODV_STATE.with(|state| {
            let mut state = state.borrow_mut();
            state.0.standalone = 1;
        });
        calls.borrow_mut().clear();
        imodv_draw_imod_images(0);
        assert!(calls.borrow().is_empty());
        IMODV_NATIVE_BOUNDARY.with(|slot| *slot.borrow_mut() = None);
    }

    #[test]
    fn caption_and_change_callbacks_preserve_imodv_source_routes() {
        let mut view = ImodView::default();
        let mut model = Imod::default();
        model.obj.push(Iobj::default());
        let calls = Rc::new(RefCell::new(Vec::new()));
        IMODV_STATE.with(|state| {
            let mut state = state.borrow_mut();
            state.1 = 0;
            state.0.standalone = 0;
            state.0.vi = &mut view;
            state.0.imod = &mut model;
        });
        IMODV_NATIVE_BOUNDARY.with(|slot| {
            *slot.borrow_mut() = Some(Box::new(ImodvHost {
                calls: calls.clone(),
                current_view: std::ptr::null_mut(),
                visual_status: 0,
                window_status: 0,
            }));
        });

        imodv_set_caption();
        imodv_register_model_chg();
        imodv_register_object_chg(0);
        imodv_register_object_chg(1);
        imodv_finish_chg_unit();
        assert_eq!(
            calls.borrow().as_slice(),
            [
                "model_view_title",
                "model_change",
                "object_change_0",
                "finish_unit"
            ]
        );

        calls.borrow_mut().clear();
        IMODV_STATE.with(|state| state.borrow_mut().0.standalone = 1);
        imodv_set_caption();
        assert_eq!(calls.borrow().as_slice(), ["standalone_title"]);
        IMODV_NATIVE_BOUNDARY.with(|slot| *slot.borrow_mut() = None);
    }

    #[test]
    fn close_and_draw_route_to_the_source_model_view_window() {
        let mut view = ImodView::default();
        let mut model = Imod::default();
        model.obj.push(Iobj::default());
        let calls = Rc::new(RefCell::new(Vec::new()));
        IMODV_STATE.with(|state| {
            let mut state = state.borrow_mut();
            state.1 = 0;
            state.0.imod = &mut model;
            state.0.vi = &mut view;
            state.0.sync_objed_to_cur_obj = 0;
        });
        IMODV_NATIVE_BOUNDARY.with(|slot| {
            *slot.borrow_mut() = Some(Box::new(ImodvHost {
                calls: calls.clone(),
                current_view: std::ptr::null_mut(),
                visual_status: 0,
                window_status: 0,
            }));
        });

        imodv_close();
        unsafe { imodv_draw() };
        assert_eq!(calls.borrow().as_slice(), ["close", "draw_model_view"]);

        IMODV_STATE.with(|state| state.borrow_mut().1 = 1);
        calls.borrow_mut().clear();
        imodv_close();
        unsafe { imodv_draw() };
        assert!(calls.borrow().is_empty());
        IMODV_NATIVE_BOUNDARY.with(|slot| *slot.borrow_mut() = None);
    }

    #[test]
    fn open_preserves_model_view_lifecycle_and_existing_window_raise() {
        let mut view = ImodView::default();
        let mut model = Imod::default();
        model.obj.push(Iobj::default());
        view.imod = &mut model;
        view.xsize = 100;
        view.ysize = 80;
        view.zsize = 20;
        view.xybin = 1;
        view.zbin = 1;
        let calls = Rc::new(RefCell::new(Vec::new()));
        IMODV_STATE.with(|state| state.borrow_mut().1 = 1);
        IMODV_NATIVE_BOUNDARY.with(|slot| {
            *slot.borrow_mut() = Some(Box::new(ImodvHost {
                calls: calls.clone(),
                current_view: &mut view,
                visual_status: 0,
                window_status: 0,
            }));
        });

        imodv_open();
        assert_eq!(calls.borrow().as_slice(), ["visuals", "open"]);
        assert_eq!(imodv_standalone(), 0);
        calls.borrow_mut().clear();
        imodv_open();
        assert_eq!(calls.borrow().as_slice(), ["raise"]);
        IMODV_NATIVE_BOUNDARY.with(|slot| *slot.borrow_mut() = None);
        IMODV_STATE.with(|state| state.borrow_mut().1 = 1);
    }

    #[test]
    fn standalone_main_loads_a_real_model_then_enters_the_native_event_loop() {
        let calls = Rc::new(RefCell::new(Vec::new()));
        let arguments = [
            "3dmodv",
            "-W",
            "-L",
            "-E",
            "ZS",
            "fixtures/model-empty-seed.mod",
        ]
        .into_iter()
        .map(|argument| argument.as_bytes().to_vec())
        .collect::<Vec<_>>();
        IMODV_STATE.with(|state| {
            let mut state = state.borrow_mut();
            state.1 = 1;
            state.2 = false;
        });
        IMODV_NATIVE_BOUNDARY.with(|slot| {
            *slot.borrow_mut() = Some(Box::new(ImodvHost {
                calls: calls.clone(),
                current_view: std::ptr::null_mut(),
                visual_status: 0,
                window_status: 0,
            }));
        });
        assert_eq!(unsafe { imodv_main(arguments.len() as i32, &arguments) }, 0);
        assert_eq!(
            calls.borrow().as_slice(),
            [
                "visuals",
                "open",
                "window_id",
                "clipboard_stdin",
                "selected_zs",
                "run"
            ]
        );
        IMODV_NATIVE_BOUNDARY.with(|slot| *slot.borrow_mut() = None);
        IMODV_STATE.with(|state| state.borrow_mut().1 = 1);
    }

    #[test]
    fn quit_preserves_source_cleanup_order_for_embedded_and_standalone_viewers() {
        let calls = Rc::new(RefCell::new(Vec::new()));
        IMODV_STATE.with(|state| {
            let mut state = state.borrow_mut();
            state.0.standalone = 0;
            state.0.bound_box_extra_obj = 1;
            state.0.cur_point_extra_obj = 2;
            state.0.rbgcolor = 1 as *mut QColor;
        });
        IMODV_NATIVE_BOUNDARY.with(|slot| {
            *slot.borrow_mut() = Some(Box::new(ImodvHost {
                calls: calls.clone(),
                current_view: std::ptr::null_mut(),
                visual_status: 0,
                window_status: 0,
            }));
        });
        imodv_quit();
        assert_eq!(
            calls.borrow().as_slice(),
            [
                "geometry",
                "record_geometry",
                "vb_cleanup",
                "image_cleanup",
                "free_box",
                "free_point",
                "stereo_off",
                "close_dialogs",
                "check_exit"
            ]
        );
        calls.borrow_mut().clear();
        IMODV_STATE.with(|state| state.borrow_mut().0.standalone = 1);
        imodv_quit();
        assert_eq!(
            calls.borrow().as_slice(),
            [
                "set_exiting",
                "geometry",
                "vb_cleanup",
                "image_cleanup",
                "free_box",
                "free_point",
                "start_disconnect",
                "stereo_off",
                "close_dialogs",
                "save_settings",
                "delete_help",
                "wait_disconnect",
                "exit"
            ]
        );
        IMODV_NATIVE_BOUNDARY.with(|slot| *slot.borrow_mut() = None);
    }

    #[test]
    fn initstruct_selects_the_existing_current_object() {
        let mut model = Box::new(Imod::default());
        model.obj.push(Iobj::default());
        model.cindex.object = 0;
        let mut view = ImodView::default();
        view.imod = &mut *model;
        view.xsize = 100;
        view.ysize = 80;
        view.zsize = 20;
        view.xybin = 1;
        view.zbin = 1;
        let mut app = ImodvApp::default();
        unsafe { initstruct(&mut view, &mut app) };
        assert_eq!(app.num_mods, 1);
        assert_eq!(app.obj_num, 0);
        assert!(std::ptr::eq(app.imod, &*model));
        assert!(!app.obj.is_null());
    }

    #[derive(Clone)]
    struct SlicerAngleBoundary {
        rotations: Rc<RefCell<Vec<Ipoint>>>,
    }

    impl ImodvSlicerAngleBoundary for SlicerAngleBoundary {
        fn set_top_slicer_from_model_view(&mut self, rot: &Ipoint) {
            self.rotations.borrow_mut().push(*rot);
        }
    }

    #[test]
    fn standalone_debug_option_sets_the_source_global_before_host_startup() {
        crate::imod::three_dmod::imod::IMOD_DEBUG
            .store(false, std::sync::atomic::Ordering::Relaxed);
        let arguments = ["3dmodv", "-D"]
            .into_iter()
            .map(|argument| argument.as_bytes().to_vec())
            .collect::<Vec<_>>();
        let calls = Rc::new(RefCell::new(Vec::new()));
        IMODV_NATIVE_BOUNDARY.with(|slot| {
            *slot.borrow_mut() = Some(Box::new(ImodvHost {
                calls,
                current_view: std::ptr::null_mut(),
                visual_status: 0,
                window_status: 0,
            }));
        });
        assert_eq!(unsafe { imodv_main(arguments.len() as i32, &arguments) }, 3);
        assert!(
            crate::imod::three_dmod::imod::IMOD_DEBUG.load(std::sync::atomic::Ordering::Relaxed)
        );
        crate::imod::three_dmod::imod::IMOD_DEBUG
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }

    #[test]
    fn new_model_angles_only_routes_linked_rotations_to_the_top_slicer() {
        let rotations = Rc::new(RefCell::new(Vec::new()));
        IMODV_SLICER_ANGLE_BOUNDARY.with(|slot| {
            *slot.borrow_mut() = Some(Box::new(SlicerAngleBoundary {
                rotations: rotations.clone(),
            }));
        });
        IMODV_STATE.with(|state| state.borrow_mut().0.link_to_slicer = 1);

        let linked = Ipoint {
            x: 12.5,
            y: -43.,
            z: 90.,
        };
        imodv_new_model_angles(&linked);
        IMODV_STATE.with(|state| state.borrow_mut().0.link_to_slicer = 0);
        imodv_new_model_angles(&Ipoint {
            x: 1.,
            y: 2.,
            z: 3.,
        });

        assert_eq!(&*rotations.borrow(), &[linked]);
        IMODV_SLICER_ANGLE_BOUNDARY.with(|slot| *slot.borrow_mut() = None);
    }
}
