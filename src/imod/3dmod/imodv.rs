//! Translation of `IMOD/3dmod/imodv.cpp` together with `IMOD/3dmod/imodv.h`.
//!
//! `imodv.cpp` is the entry point of the OpenGL model viewer.  The model and
//! view state below is deliberately kept separate from the widget boundary:
//! the latter belongs to `mv_window.cpp` and is not represented by a static
//! or simulated viewer here.
#![allow(dead_code, unused_variables)]

use std::cell::RefCell;
use std::ffi::{CStr, c_char, c_void};

use crate::imod::libimod::imat::{Imat, imod_mat_new};
use crate::imod::libimod::imodel::{Imod, Iobj, Ipoint, Iview};
use crate::imod::libimod::imodel_files::imod_read;
use crate::imod::libimod::iview::{imod_view_default, imod_view_default_scale};
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
/// Pointer fields retain the source ownership relationship.  The object
/// vectors are owned by `libimod`; `imodv.cpp` only selects current elements.
#[repr(C)]
pub struct ImodvApp {
    pub mod_: Vec<*mut Imod>,
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
    pub vi: *mut c_void,
    pub do_pick: i32,
    pub x_pick: i32,
    pub y_pick: i32,
    pub w_pick: i32,
    pub h_pick: i32,
    pub pick_hits: i32,
    pub read_pix_for_pick: i32,
    pub mod_picks: *mut Ipoint,
    pub max_mod_picks: i32,
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
            vert_buf_ok: 0,
            prim_restart_ok: 0,
            gl_ext_flags: 0,
            vb_manager: std::ptr::null_mut(),
            tex_map: 0,
            tex_trans: 0,
            vi: std::ptr::null_mut(),
            do_pick: 0,
            x_pick: 0,
            y_pick: 0,
            w_pick: 0,
            h_pick: 0,
            pick_hits: 0,
            read_pix_for_pick: 0,
            mod_picks: std::ptr::null_mut(),
            max_mod_picks: 0,
            legacy_pick_mode: 0,
            lighting: 0,
            depthcue: 0,
            wireframe: 0,
            lowres: 0,
            invert_z: 0,
        }
    }
}

thread_local! {
    /// Original globals: `ImodvStruct`, `Imodv`, and `ImodvClosed`.
    static IMODV_STATE: RefCell<(ImodvApp, i32, bool)> = RefCell::new((ImodvApp::default(), 1, false));
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
    a.imod = std::ptr::null_mut();
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
    a.mod_picks = std::ptr::null_mut();
    a.max_mod_picks = 0;
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
    a.vi = vw as *mut ImodView as *mut c_void;
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
pub unsafe fn load_models(n: i32, fname: *const *const c_char, a: &mut ImodvApp) -> i32 {
    if n < 1 {
        return 0;
    }
    a.mod_.clear();
    a.num_mods = n;
    a.cur_mod = 0;
    for i in 0..n {
        let name = CStr::from_ptr(*fname.add(i as usize));
        let path = String::from_utf8_lossy(name.to_bytes());
        let Ok(model) = imod_read(path.as_ref()) else {
            return -1;
        };
        let model = Box::into_raw(Box::new(model));
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
pub fn get_visuals(_a: &mut ImodvApp) -> i32 {
    1
}

/// Original static: `openWindow` (`imodv.cpp:398`).
/// The actual OpenGL/QWidget constructor is retained as an unported boundary.
pub fn open_window(_a: &mut ImodvApp) -> i32 {
    1
}

/// Original: `imodvMain` (`imodv.cpp:536`).
pub unsafe fn imodv_main(argc: i32, argv: *const *const c_char) -> i32 {
    IMODV_STATE.with(|state| {
        let mut state = state.borrow_mut();
        let a = &mut state.0;
        a.standalone = 1;
        imodv_init(a);
        let mut i = 1;
        let mut use_stdin = false;
        while i < argc {
            let argument = CStr::from_ptr(*argv.add(i as usize)).to_bytes();
            if !argument.starts_with(b"-") {
                break;
            }
            match argument {
                b"-b" => {
                    i += 1;
                    if i >= argc {
                        return 1;
                    }
                    a.rbgname = CStr::from_ptr(*argv.add(i as usize))
                        .to_string_lossy()
                        .into_owned();
                }
                b"-D" => {}
                b"-f" => a.fullscreen = 1,
                b"-s" => {
                    i += 1;
                    if i >= argc {
                        return 1;
                    }
                    let text = CStr::from_ptr(*argv.add(i as usize)).to_string_lossy();
                    let mut values = text.split(|c| c == ',' || c == 'x');
                    a.want_winx = values.next().and_then(|v| v.parse().ok()).unwrap_or(0);
                    a.want_winy = values.next().and_then(|v| v.parse().ok()).unwrap_or(0);
                }
                b"-W" => {}
                b"-E" => {
                    i += 1;
                }
                b"-L" => use_stdin = true,
                b"-h" => return 1,
                b"-modv" | b"-view" => {}
                _ => return 1,
            }
            i += 1;
        }
        a.dbl_buf = 1;
        if argc - i < 1 || load_models(argc - i, argv.add(i as usize), a) != 0 {
            return 3;
        }
        // The source calls getVisuals/openWindow/qApp->exec here.  Do not replace that
        // OpenGL/QApplication execution with a fabricated static viewer.
        let _ = use_stdin;
        3
    })
}

/// Original: `imodv_open` (`imodv.cpp:653`).
pub fn imodv_open() {}
/// Original: `imodv_close` (`imodv.cpp:696`).
pub fn imodv_close() {
    // The source delegates this to `ImodvWindow::close`; final state changes
    // occur in the close callback, `imodv_quit` below.
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
        // `imodvDraw` is the actual OpenGL rendering boundary in `mv_gfx.cpp`.
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
        let vi = &*(state.0.vi as *const ImodView);
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
pub fn imodv_new_model_angles(_rot: &Ipoint) {}
/// Original: `imodvSetCaption` (`imodv.cpp:775`).
pub fn imodv_set_caption() {}
/// Original: `imodvDrawImodImages` (`imodv.cpp:786`).
pub fn imodv_draw_imod_images(_skip_draw: i32) {}
/// Original: `imodvByteImagesExist` (`imodv.cpp:797`).
pub fn imodv_byte_images_exist() -> i32 {
    IMODV_STATE.with(|state| {
        let state = state.borrow();
        if state.0.standalone != 0 || state.0.vi.is_null() {
            return 0;
        }
        let vi = unsafe { &*(state.0.vi as *const ImodView) };
        (vi.rgb_store == 0 && vi.fake_image == 0) as i32
    })
}
/// Original: `imodvRegisterModelChg` (`imodv.cpp:807`).
pub fn imodv_register_model_chg() {}
/// Original: `imodvRegisterObjectChg` (`imodv.cpp:814`).
pub fn imodv_register_object_chg(_object: i32) {}
/// Original: `imodvFinishChgUnit` (`imodv.cpp:821`).
pub fn imodv_finish_chg_unit() {}
/// Original: `imodvQuit` (`imodv.cpp:829`).
pub fn imodv_quit() {
    IMODV_STATE.with(|state| {
        let mut state = state.borrow_mut();
        state.1 = 1;
        state.2 = true;
        state.0.mat = None;
        state.0.rmat = None;
        state.0.main_win = std::ptr::null_mut();
    });
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
