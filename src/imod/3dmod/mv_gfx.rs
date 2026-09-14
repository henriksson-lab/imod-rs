//! Translation of `IMOD/3dmod/mv_gfx.cpp` together with `mv_gfx.h`.
//!
//! The compatibility OpenGL commands remain commands on the current native
//! context.  `mv_ogl.cpp` owns model primitive drawing; its calls are explicit
//! methods here rather than a replacement renderer.
#![allow(dead_code, unused_variables)]

use std::fs::File;
use std::io::{self, Write};
use std::sync::Mutex;

use crate::imod::libimod::imodel::Ipoint;
use crate::imod::three_dmod::imodv::{IMODV_STEREO_OFF, ImodvApp};
use crate::imod::three_dmod::mv_window::VVIEW_MENU_DB;

pub const IMODV_STEREO_RL: i32 = 1;
pub const IMODV_STEREO_TB: i32 = 2;
pub const IMODV_STEREO_HW: i32 = 3;
pub const SNAP_SHOT_TIF: i32 = 0;
pub const SNAP_SHOT_RGB: i32 = 1;

/// Original global: `ImodvCurModLight`.
static IMODV_CUR_MOD_LIGHT: Mutex<Ipoint> = Mutex::new(Ipoint {
    x: 0.,
    y: 0.,
    z: 0.,
});

/// Direct compatibility-profile boundary for the calls in this source unit.
pub trait ImodvGfxGl {
    fn make_current(&mut self);
    fn swap_buffers(&mut self);
    fn flush(&mut self);
    fn finish(&mut self);
    fn clear(&mut self, red: f32, green: f32, blue: f32, alpha: f32, depth: bool);
    fn draw_buffer(&mut self, right: bool, back: bool);
    fn viewport(&mut self, width: i32, height: i32);
    fn initialize(&mut self, fog_end: f32, double_buffer: bool);
    fn render_mode_select(&mut self) -> i32;
    fn render_mode_render(&mut self) -> i32;
    fn init_names(&mut self);
    fn multisample(&mut self, enabled: bool);
    fn draw_models(&mut self, app: &mut ImodvApp);
    /// `a->vbManager->clearTempArrays()`; the manager is owned by the app, so
    /// it is passed exactly as the source reaches it.
    fn clear_temp_arrays(&mut self, app: &mut ImodvApp);
    fn resize_viewport_xy(&mut self, width: i32, height: i32);
    fn draw_light_vector(&mut self, app: &ImodvApp, light: Ipoint);
    fn draw_scale_bar(&mut self, app: &ImodvApp, scale: f32, color: i32) -> f32;
    fn read_rgb_pixels(&mut self, x: i32, width: i32, height: i32) -> Vec<u8>;
    /// `b3dInitializeGL()` (`b3dgfx.cpp`), called from `imodvPaintGL`'s
    /// first-time probe (`mv_gfx.cpp:234-238`).  Returns the extension flags.
    fn initialize_gl_extensions(&mut self) -> i32;
}

/// `imodv_winset`.
pub fn imodv_winset(a: &ImodvApp, gl: &mut dyn ImodvGfxGl) -> i32 {
    if a.do_pick == 0 {
        gl.make_current();
    }
    1
}
/// `imodv_swapbuffers`.
pub fn imodv_swapbuffers(a: &ImodvApp, gl: &mut dyn ImodvGfxGl) {
    if a.do_pick != 0 {
        return;
    }
    gl.make_current();
    if a.dbl_buf != 0 {
        gl.swap_buffers();
    }
    gl.flush();
}
/// Static `imodv_clear`.
pub fn imodv_clear(a: &mut ImodvApp, rgb: [u8; 3], gl: &mut dyn ImodvGfxGl) {
    imodv_winset(a, gl);
    if a.clear_after_stereo != 0 {
        gl.draw_buffer(true, a.dbl_buf != 0);
        gl.clear(
            rgb[0] as f32 / 256.,
            rgb[1] as f32 / 256.,
            rgb[2] as f32 / 256.,
            if a.trans_bkgd != 0 { 0. } else { 1. },
            true,
        );
        gl.draw_buffer(false, a.dbl_buf != 0);
        a.clear_after_stereo = 0;
    }
    gl.clear(
        rgb[0] as f32 / 256.,
        rgb[1] as f32 / 256.,
        rgb[2] as f32 / 256.,
        if a.trans_bkgd != 0 { 0. } else { 1. },
        true,
    );
    gl.flush();
}
/// `imodv_setbuffer`.  `set_widget` is the paired `ImodvWindow::setGLWidget` call.
pub fn imodv_setbuffer(
    a: &mut ImodvApp,
    db: i32,
    stereo: i32,
    alpha: i32,
    rgb: [u8; 3],
    set_widget: &mut dyn FnMut(bool, bool, bool) -> i32,
    gl: &mut dyn ImodvGfxGl,
) {
    let mut use_stereo = a.stereo == IMODV_STEREO_HW;
    let mut use_db = a.dbl_buf != 0;
    let mut use_alpha = a.alpha_visual != 0;
    let in_stereo = use_stereo;
    if db >= 0 {
        if (db != 0
            && ((!in_stereo && a.enable_depth_sb < 0)
                || (in_stereo && a.enable_depth_sbst < 0)
                || use_alpha))
            || (db == 0
                && ((!in_stereo && a.enable_depth_db < 0 && a.enable_depth_dbal < 0)
                    || (in_stereo && a.enable_depth_dbst < 0 && a.enable_depth_dbst_al < 0)))
        {
            return;
        }
        use_db = db != 0;
    } else if alpha >= 0 {
        if (alpha != 0
            && ((!in_stereo && a.enable_depth_db < 0) || (in_stereo && a.enable_depth_dbst < 0)))
            || (alpha == 0
                && ((!in_stereo && a.enable_depth_dbal < 0)
                    || (in_stereo && a.enable_depth_dbst_al < 0)
                    || !use_db))
        {
            return;
        }
        use_alpha = alpha != 0;
    } else {
        if (a.dbl_buf != 0
            && ((stereo == 0 && a.enable_depth_db < 0 && a.enable_depth_dbal < 0)
                || (stereo != 0 && a.enable_depth_dbst < 0 && a.enable_depth_dbst_al < 0)))
            || (a.dbl_buf == 0
                && ((stereo == 0 && a.enable_depth_sb < 0)
                    || (stereo != 0 && a.enable_depth_sbst < 0)))
        {
            return;
        }
        use_stereo = stereo != 0;
    }
    imodv_clear(a, rgb, gl);
    if set_widget(use_db, use_stereo, use_alpha) != 0 {
        return;
    }
    gl.make_current();
    a.dbl_buf = use_db as i32;
    a.alpha_visual = use_alpha as i32;
    // `setCheckableItem(VVIEW_MENU_DB, useDb)` is the paired window call.
    let _ = VVIEW_MENU_DB;
    gl.viewport(a.winx, a.winy);
    gl.flush();
    gl.finish();
}
/// `imodvInitializeGL`.
pub fn imodv_initialize_gl(a: &mut ImodvApp, gl: &mut dyn ImodvGfxGl) {
    let rad = unsafe { a.imod.as_ref() }
        .and_then(|m| m.view.first())
        .map_or(0., |v| v.rad * 1.5);
    gl.initialize(rad, a.dbl_buf != 0);
}
/// `imodvResizeGL`.
pub fn imodv_resize_gl(
    a: &mut ImodvApp,
    winx: i32,
    winy: i32,
    device_pixel_ratio: f32,
    new_qt_opengl: bool,
    gl: &mut dyn ImodvGfxGl,
) {
    a.winx = if new_qt_opengl && device_pixel_ratio > 1. {
        (winx as f32 * device_pixel_ratio) as i32
    } else {
        winx
    };
    a.winy = if new_qt_opengl && device_pixel_ratio > 1. {
        (winy as f32 * device_pixel_ratio) as i32
    } else {
        winy
    };
    gl.viewport(a.winx, a.winy);
}
/// `imodvDraw`.  Slicer linking is retained in `mv_input`/slicer units; this is the actual repaint dispatch.
pub fn imodv_draw(a: &mut ImodvApp, update_gl: &mut dyn FnMut()) {
    update_gl();
}
/// `imodvPaintGL`.
pub fn imodv_paint_gl(
    a: &mut ImodvApp,
    rgb: [u8; 3],
    new_qt_opengl: bool,
    gl: &mut dyn ImodvGfxGl,
) {
    // `mv_gfx.cpp:233-238`: first time in, find the OpenGL version and set
    // `vertBufOK` to -1 or 1.  This runs before the `!a->imod` return, and the
    // translation had omitted it entirely, so `vertBufOK` stayed at its
    // sentinel and `Ctrl-Shift-V` never reached its toggle.
    if a.vert_buf_ok < -1 {
        a.gl_ext_flags = gl.initialize_gl_extensions();
        a.vert_buf_ok = if a.gl_ext_flags & crate::imod::three_dmod::b3dgfx::B3DGLEXT_VERTBUF != 0 {
            1
        } else {
            -1
        };
        a.prim_restart_ok =
            i32::from(a.gl_ext_flags & crate::imod::three_dmod::b3dgfx::B3DGLEXT_PRIM_RESTART != 0);
    }
    if a.imod.is_null() {
        return;
    }
    imodv_winset(a, gl);
    if new_qt_opengl {
        gl.multisample(true);
    }
    if a.do_pick != 0 {
        gl.render_mode_select();
        gl.init_names();
    }
    imodv_clear(a, rgb, gl);
    match a.stereo {
        IMODV_STEREO_RL => {
            a.winx /= 2;
            a.stereo *= -1;
            gl.draw_models(a);
            a.stereo *= -1;
            gl.draw_models(a);
            a.winx *= 2;
            gl.viewport(a.winx, a.winy);
        }
        IMODV_STEREO_TB => {
            a.winy /= 2;
            a.stereo *= -1;
            gl.draw_models(a);
            a.stereo *= -1;
            gl.draw_models(a);
            a.winy *= 2;
            gl.viewport(a.winx, a.winy);
        }
        IMODV_STEREO_HW => {
            a.stereo *= -1;
            gl.draw_buffer(true, a.dbl_buf != 0);
            gl.clear(0., 0., 0., 1., true);
            gl.draw_models(a);
            a.stereo *= -1;
            gl.draw_buffer(false, a.dbl_buf != 0);
            gl.clear(0., 0., 0., 1., true);
            gl.draw_models(a);
        }
        IMODV_STEREO_OFF => {
            if a.read_pix_for_pick != 0 {
                gl.draw_models(a);
                a.read_pix_for_pick = 0;
                imodv_clear(a, rgb, gl);
            }
            gl.draw_models(a);
        }
        _ => {}
    }
    gl.clear_temp_arrays(a);
    gl.resize_viewport_xy(a.winx, a.winy);
    if a.draw_light != 0 {
        let light = *IMODV_CUR_MOD_LIGHT.lock().unwrap();
        gl.draw_light_vector(a, light);
    }
    let color = if a.tex_map == 0 && rgb == [0, 0, 0] {
        -1
    } else if a.tex_map == 0 && rgb == [255, 255, 255] {
        1
    } else {
        0
    };
    let rad = unsafe { a.imod.as_ref() }
        .and_then(|m| m.view.first())
        .map_or(1., |v| v.rad);
    a.scale_bar_size = gl.draw_scale_bar(a, 0.5 * a.winx.min(a.winy) as f32 / rad, color);
    if a.do_pick != 0 {
        a.pick_hits = gl.render_mode_render();
    }
}
/// `imodvResetSnap`.
pub fn imodv_reset_snap(a: &mut ImodvApp) {
    a.snap_fileno = 0;
}

/// Calls in `imodv_auto_snapshot` owned by `b3dfile.cpp` and preferences.
/// They deliberately remain a lower-source boundary: this is not a second
/// image encoder hidden in the viewer.
pub trait ImodvSnapshotSink {
    fn snapshot_name(&mut self, root: &str, format_type: i32, digits: i32, sequence: i32)
    -> String;
    fn short_snapshot_name(&mut self, name: &str) -> String;
    fn pre_snapshot_changes(&mut self);
    fn restore_snapshot_changes(&mut self);
    fn set_buffer_swap_auto(&mut self, enabled: bool);
    fn draw(&mut self);
    fn set_current_size(&mut self, width: i32, height: i32);
    fn snapshot_tif(&mut self, name: &str) -> i32;
    fn snapshot_non_tif(&mut self, name: &str) -> i32;
    fn preference_snap_format(&self) -> &str;
}
/// Static `imodv_snapshot`; writes the upstream uncompressed SGI RGB layout.
pub fn imodv_snapshot(a: &ImodvApp, fname: &str, pixels: &[u8]) -> io::Result<()> {
    let mut width = a.winx;
    let height = a.winy;
    let xoffset = (width % 4) / 2;
    width = 4 * (width / 4);
    let size = width.max(0) as usize * height.max(0) as usize * 3;
    if pixels.len() < size {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "glReadPixels returned too little RGB data",
        ));
    }
    let mut out = File::create(fname)?;
    out.write_all(&474u16.to_be_bytes())?;
    out.write_all(&[0, 1])?;
    out.write_all(&3u16.to_be_bytes())?;
    out.write_all(&(width as u16).to_be_bytes())?;
    out.write_all(&(height as u16).to_be_bytes())?;
    out.write_all(&3u16.to_be_bytes())?;
    out.write_all(&0u32.to_be_bytes())?;
    out.write_all(&255u32.to_be_bytes())?;
    out.write_all(&0u32.to_be_bytes())?;
    let mut name = [0u8; 80];
    let tail = std::path::Path::new(fname)
        .file_name()
        .unwrap_or_default()
        .to_string_lossy();
    let text = format!("{}, Created by 3dmodv.", &tail[..tail.len().min(59)]);
    name[..text.len().min(80)].copy_from_slice(&text.as_bytes()[..text.len().min(80)]);
    out.write_all(&name)?;
    out.write_all(&0u32.to_be_bytes())?;
    out.write_all(&[0; 404])?;
    let n = width as usize * height as usize;
    for channel in 0..3 {
        for i in 0..n {
            out.write_all(&[pixels[i * 3 + channel]])?;
        }
    }
    let _ = xoffset;
    Ok(())
}
/// `imodv_auto_snapshot`; its `b3dfile.cpp` / preferences calls are carried
/// by the paired direct source boundary.
pub fn imodv_auto_snapshot(
    a: &mut ImodvApp,
    mut fname: String,
    format_type: i32,
    gl: &mut dyn ImodvGfxGl,
    sink: &mut dyn ImodvSnapshotSink,
) -> io::Result<i32> {
    if fname.is_empty() {
        fname = sink.snapshot_name("modv", format_type, 4, a.snap_fileno);
    }
    let _short_name = sink.short_snapshot_name(&fname);
    sink.pre_snapshot_changes();
    if a.dbl_buf != 0 {
        sink.set_buffer_swap_auto(false);
    }
    sink.draw();
    sink.set_current_size(a.winx, a.winy);
    let _error = if format_type == SNAP_SHOT_TIF {
        sink.snapshot_tif(&fname)
    } else if format_type == SNAP_SHOT_RGB && sink.preference_snap_format() != "RGB" {
        sink.snapshot_non_tif(&fname)
    } else {
        let pixels = gl.read_rgb_pixels((a.winx % 4) / 2, 4 * (a.winx / 4), a.winy);
        imodv_snapshot(a, &fname, &pixels).map_or(-1, |_| 0)
    };
    if a.dbl_buf != 0 {
        imodv_swapbuffers(a, gl);
        sink.set_buffer_swap_auto(true);
    }
    sink.restore_snapshot_changes();
    // The C++ source reports `error` to stderr but deliberately returns zero.
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::libimod::imodel::{Imod, Iview};

    /// Records the boundary calls `imodvPaintGL` makes, in order.
    #[derive(Default)]
    struct RecordingGfxGl {
        calls: Vec<&'static str>,
        /// What `b3dInitializeGL()` reports back to the `mv_gfx.cpp:234` probe.
        ext_flags: i32,
    }
    impl ImodvGfxGl for RecordingGfxGl {
        fn initialize_gl_extensions(&mut self) -> i32 {
            self.calls.push("initialize_gl_extensions");
            self.ext_flags
        }
        fn make_current(&mut self) {
            self.calls.push("make_current")
        }
        fn swap_buffers(&mut self) {
            self.calls.push("swap_buffers")
        }
        fn flush(&mut self) {
            self.calls.push("flush")
        }
        fn finish(&mut self) {
            self.calls.push("finish")
        }
        fn clear(&mut self, _: f32, _: f32, _: f32, _: f32, _: bool) {
            self.calls.push("clear")
        }
        fn draw_buffer(&mut self, _: bool, _: bool) {
            self.calls.push("draw_buffer")
        }
        fn viewport(&mut self, _: i32, _: i32) {
            self.calls.push("viewport")
        }
        fn initialize(&mut self, _: f32, _: bool) {
            self.calls.push("initialize")
        }
        fn render_mode_select(&mut self) -> i32 {
            self.calls.push("render_mode_select");
            0
        }
        fn render_mode_render(&mut self) -> i32 {
            self.calls.push("render_mode_render");
            0
        }
        fn init_names(&mut self) {
            self.calls.push("init_names")
        }
        fn multisample(&mut self, _: bool) {
            self.calls.push("multisample")
        }
        fn draw_models(&mut self, _: &mut ImodvApp) {
            self.calls.push("draw_models")
        }
        fn clear_temp_arrays(&mut self, _: &mut ImodvApp) {
            self.calls.push("clear_temp_arrays")
        }
        fn resize_viewport_xy(&mut self, _: i32, _: i32) {
            self.calls.push("resize_viewport_xy")
        }
        fn draw_light_vector(&mut self, _: &ImodvApp, _: Ipoint) {
            self.calls.push("draw_light_vector")
        }
        fn draw_scale_bar(&mut self, _: &ImodvApp, _: f32, _: i32) -> f32 {
            self.calls.push("draw_scale_bar");
            1.
        }
        fn read_rgb_pixels(&mut self, _: i32, _: i32, _: i32) -> Vec<u8> {
            self.calls.push("read_rgb_pixels");
            Vec::new()
        }
    }

    /// `mv_gfx.cpp:234` probes only while `vertBufOK < -1`, so a second paint
    /// must not re-probe, and the flags it derives must follow `:236-237`.
    #[test]
    fn imodv_paint_gl_probes_gl_extensions_once_and_derives_the_flags() {
        use crate::imod::three_dmod::b3dgfx::{B3DGLEXT_PRIM_RESTART, B3DGLEXT_VERTBUF};
        let mut model = Imod::default();
        let mut a = ImodvApp {
            winx: 64,
            winy: 48,
            ..Default::default()
        };
        a.imod = &mut model;
        assert_eq!(a.vert_buf_ok, -2, "imodv.cpp:179 sentinel");
        let mut gl = RecordingGfxGl {
            ext_flags: B3DGLEXT_VERTBUF | B3DGLEXT_PRIM_RESTART,
            ..Default::default()
        };
        imodv_paint_gl(&mut a, [0, 0, 0], false, &mut gl);
        assert_eq!(a.gl_ext_flags, B3DGLEXT_VERTBUF | B3DGLEXT_PRIM_RESTART);
        assert_eq!(a.vert_buf_ok, 1);
        assert_eq!(a.prim_restart_ok, 1);
        assert_eq!(
            gl.calls
                .iter()
                .filter(|c| **c == "initialize_gl_extensions")
                .count(),
            1
        );
        gl.calls.clear();
        imodv_paint_gl(&mut a, [0, 0, 0], false, &mut gl);
        assert!(
            !gl.calls.contains(&"initialize_gl_extensions"),
            "probes once"
        );
        // Without the extension, `:236` gives -1, not 0.
        let mut b = ImodvApp {
            winx: 64,
            winy: 48,
            ..Default::default()
        };
        b.imod = &mut model;
        let mut off = RecordingGfxGl::default();
        imodv_paint_gl(&mut b, [0, 0, 0], false, &mut off);
        assert_eq!(b.vert_buf_ok, -1);
        assert_eq!(b.prim_restart_ok, 0);
    }
    #[test]
    fn imodv_paint_gl_issues_the_source_boundary_sequence() {
        let mut model = Imod {
            view: vec![Iview::default()],
            ..Default::default()
        };
        let mut a = ImodvApp {
            winx: 64,
            winy: 48,
            ..Default::default()
        };
        a.imod = &mut model;
        let mut gl = RecordingGfxGl::default();
        imodv_paint_gl(&mut a, [0, 0, 0], false, &mut gl);
        // `mv_gfx.cpp:233-238` probes the GL version on the first paint, before
        // anything else, while `vertBufOK` is still its `imodv.cpp:179`
        // sentinel of -2.  This expectation previously omitted it because the
        // probe was untranslated.
        assert_eq!(
            gl.calls,
            vec![
                "initialize_gl_extensions",
                "make_current",
                "make_current",
                "clear",
                "flush",
                "draw_models",
                "clear_temp_arrays",
                "resize_viewport_xy",
                "draw_scale_bar",
            ]
        );
        assert_eq!(a.scale_bar_size, 1.);
    }
    #[test]
    fn snapshot_has_sgi_header_and_planar_data() {
        let a = ImodvApp {
            winx: 4,
            winy: 1,
            ..Default::default()
        };
        let p = std::env::temp_dir().join("imod-rs-mv-gfx.rgb");
        imodv_snapshot(
            &a,
            p.to_str().unwrap(),
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        )
        .unwrap();
        let got = std::fs::read(&p).unwrap();
        assert_eq!(&got[..2], &474u16.to_be_bytes());
        assert_eq!(&got[512..516], &[1, 4, 7, 10]);
        std::fs::remove_file(p).unwrap();
    }
}
