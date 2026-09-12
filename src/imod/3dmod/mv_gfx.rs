//! Translation of `IMOD/3dmod/mv_gfx.cpp` together with `mv_gfx.h`.
//!
//! The compatibility OpenGL commands remain commands on the current native
//! context.  `mv_ogl.cpp` owns model primitive drawing; its calls are explicit
//! methods here rather than a replacement renderer.
#![allow(dead_code, unused_variables)]

use std::fs::File;
use std::io::{self, Write};

use crate::imod::libimod::imodel::Ipoint;
use crate::imod::three_dmod::imodv::{IMODV_STEREO_OFF, ImodvApp};
use crate::imod::three_dmod::mv_window::VVIEW_MENU_DB;

pub const IMODV_STEREO_RL: i32 = 1;
pub const IMODV_STEREO_TB: i32 = 2;
pub const IMODV_STEREO_HW: i32 = 3;
pub const SNAP_SHOT_TIF: i32 = 0;
pub const SNAP_SHOT_RGB: i32 = 1;

/// Original global: `ImodvCurModLight`.
pub static mut IMODV_CUR_MOD_LIGHT: Ipoint = Ipoint {
    x: 0.,
    y: 0.,
    z: 0.,
};

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
    fn clear_temp_arrays(&mut self);
    fn resize_viewport_xy(&mut self, width: i32, height: i32);
    fn draw_light_vector(&mut self, app: &ImodvApp, light: Ipoint);
    fn draw_scale_bar(&mut self, app: &ImodvApp, scale: f32, color: i32) -> f32;
    fn read_rgb_pixels(&mut self, x: i32, width: i32, height: i32) -> Vec<u8>;
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
    gl.clear_temp_arrays();
    gl.resize_viewport_xy(a.winx, a.winy);
    if a.draw_light != 0 {
        unsafe {
            gl.draw_light_vector(a, IMODV_CUR_MOD_LIGHT);
        }
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
