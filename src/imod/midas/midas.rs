//! Translation of `IMOD/midas/midas.cpp` and its paired public `midas.h`.
//!
//! `midas.cpp` is both the executable entry and the Qt control-panel unit.
//! Its image loading, transform, GL drawing, and slot endpoints belong to
//! `file_io.cpp`, `transforms.cpp`, `graphics.cpp`, and `slots.cpp`; those
//! source units have not been translated yet.  This unit deliberately stops at
//! that real boundary instead of manufacturing a replacement alignment UI.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::imod::libiimod::mrcfiles::{LoadInfo, MrcHeader};

pub const MIDAS_VIEW_SINGLE: i32 = 0;
pub const MIDAS_VIEW_COLOR: i32 = 1;
pub const MIDAS_VIEW_MULTI: i32 = 2;
pub const XTYPE_XO: i32 = 0;
pub const XTYPE_XF: i32 = 1;
pub const XTYPE_XG: i32 = 2;
pub const XTYPE_XREF: i32 = 3;
pub const XTYPE_MONT: i32 = 4;
pub const MAX_CACHE_MBYTES: i32 = 1024;
pub const MAX_ZOOMIND: i32 = 14;
pub const MAX_INCREMENTS: i32 = 6;
pub const MAX_TOP_ERR: usize = 10;

/// C global `Midas_debug` (`midas.cpp:59`).
pub static MIDAS_DEBUG: AtomicBool = AtomicBool::new(false);

/// C `Midas_transform` (`midas.h`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MidasTransform {
    pub black: i32,
    pub white: i32,
    pub mat: [f32; 9],
}

/// C `Midas_cache` (`midas.h`).  `sec` is retained as an optional source
/// object marker until `Islice` is translated by the Midas closure.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MidasCache {
    pub zval: i32,
    pub xformed: i32,
    pub used: i32,
    pub mat: [f32; 9],
    pub n_control: i32,
    pub nx_grid: i32,
    pub ny_grid: i32,
    pub x_start: f32,
    pub y_start: f32,
    pub x_interval: f32,
    pub y_interval: f32,
    pub mean_sds: [f32; 8],
    pub sec_present: bool,
}

/// C `Midas_chunk` (`midas.h`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MidasChunk {
    pub size: i32,
    pub start: i32,
    pub cur_sec: i32,
    pub ref_sec: i32,
    pub max_cur_sec: i32,
    pub min_ref_sec: i32,
}

/// C `Midas_view` (`midas.h`), preserving source field ownership while Qt and
/// GL pointers become explicit availability flags at their Rust boundary.
pub struct MidasView {
    pub zsize: i32,
    pub xsize: i32,
    pub ysize: i32,
    pub xysize: i32,
    pub binning: i32,
    pub cz: i32,
    pub refz: i32,
    pub xcenter: f32,
    pub ycenter: f32,
    pub xfixed: f32,
    pub yfixed: f32,
    pub use_fixed: i32,
    pub cur_chunk: i32,
    pub sminin: f32,
    pub smaxin: f32,
    pub cachein: i32,
    pub rot_mode: i32,
    pub global_rot: f64,
    pub cos_stretch: i32,
    pub tilt_offset: f32,
    pub num_chunks: i32,
    pub quiet: i32,
    pub image_for_channel: [i32; 3],
    pub chunk: Vec<MidasChunk>,
    pub tilt_angles: Vec<f32>,
    /// C `li`: the source-owned load conversion parameters.
    pub li: Option<LoadInfo>,
    /// C `hin`: the open primary MRC header, including its source FILE pointer.
    pub hin: Option<MrcHeader>,
    pub usecount: i32,
    pub cachesize: i32,
    pub cache: Vec<MidasCache>,
    pub unbinned_buf: Vec<u8>,
    pub tr: Vec<MidasTransform>,
    pub ref_present: bool,
    /// C `vw->ref->mean`, retained with the reference byte image below.
    pub ref_mean: f32,
    /// C `vw->ref->data.b`; Rust owns the byte allocation directly.
    pub ref_data: Vec<u8>,
    pub showref: i32,
    pub sangle: i32,
    pub phi: f32,
    pub sdat: Vec<u32>,
    pub id: Vec<u32>,
    pub zoom: f32,
    pub truezoom: f32,
    pub zoomind: i32,
    pub bin_to_zoom_down: bool,
    pub xtrans: i32,
    pub ytrans: i32,
    pub xoffset: i32,
    pub yoffset: i32,
    pub vmode: i32,
    pub fast_interp: i32,
    pub lastmx: i32,
    pub lastmy: i32,
    pub firstmx: i32,
    pub firstmy: i32,
    pub mx: i32,
    pub my: i32,
    pub mousemoving: i32,
    pub width: i32,
    pub height: i32,
    pub device_pixel_ratio: f32,
    pub dev_pix_varies: i32,
    pub screen_changed: bool,
    pub orig_height: i32,
    pub startup_done: bool,
    pub xtype: i32,
    pub xname: Option<String>,
    pub oname: Option<String>,
    pub refname: Option<String>,
    pub refzsize: i32,
    pub tiltname: Option<String>,
    pub changed: i32,
    pub didsave: i32,
    pub plname: Option<String>,
    pub blended_montage: Option<String>,
    pub blended_coords: Option<String>,
    pub blended_binning: i32,
    pub xsec: i32,
    pub corr_box_size: i32,
    pub corr_shift_limit: i32,
    pub corr_vals_entered: i32,
    pub minxpiece: i32,
    pub minypiece: i32,
    pub minzpiece: i32,
    pub maxzpiece: i32,
    pub nxpieces: i32,
    pub nypieces: i32,
    pub nxoverlap: i32,
    pub nyoverlap: i32,
    pub any_skipped: i32,
    pub robust_fit: i32,
    pub robust_crit: f32,
    pub exclude_skipped: i32,
    pub nedge: [i32; 2],
    pub maxedge: [i32; 2],
    pub xory: i32,
    pub center_xory: i32,
    pub montcz: i32,
    pub curedge: i32,
    pub edgeind: i32,
    pub curleavex: f32,
    pub curleavey: f32,
    pub topind: [i32; MAX_TOP_ERR],
    pub num_top_err: i32,
    pub skip_err: i32,
    pub depth: i32,
    pub exposed: i32,
    pub blackstate: i32,
    pub whitestate: i32,
    pub reversemap: i32,
    pub applytoone: i32,
    pub draw_corr_box: i32,
    pub keepsecdiff: i32,
    pub edit_warps: bool,
    pub warping_ok: bool,
    pub draw_vectors: bool,
    pub cur_control: i32,
    pub cur_warp_file: i32,
    pub warp_nz: i32,
    pub warp_scale: f32,
    pub max_warp_backup: i32,
    pub num_warp_backup: i32,
    pub grid_size: i32,
    pub old_mat: [f32; 9],
    pub last_grid_size: i32,
    pub last_warped_z: i32,
    pub last_nx_grid: i32,
    pub last_ny_grid: i32,
    pub last_xstart: f32,
    pub last_ystart: f32,
    pub last_xinterv: f32,
    pub last_yinterv: f32,
    pub last_mat: [f32; 9],
    pub incindex: [i32; 3],
    pub increment: [f32; 3],
    pub paramstate: [f32; 5],
    pub backup_mat: [f32; 9],
    pub backup_edgedx: f32,
    pub backup_edgedy: f32,
    pub mouse_xonly: i32,
    pub ctrl_pressed: i32,
    pub shift_pressed: i32,
    pub exiting: bool,
    pub midas_window_present: bool,
    pub midas_slots_present: bool,
    pub midas_gl_present: bool,
}

impl Default for MidasView {
    fn default() -> Self {
        new_view()
    }
}

/// Releases the source `hin->fp` ownership acquired by `load_image`.
impl Drop for MidasView {
    fn drop(&mut self) {
        if let Some(header) = self.hin.take() {
            if !header.fp.is_null() {
                unsafe { libc::fclose(header.fp.cast()) };
            }
        }
    }
}

/// C `new_view(MidasView *)` (`transforms.cpp:33`), represented as Rust
/// construction because no caller can observe the C pre-initialization state.
pub fn new_view() -> MidasView {
    MidasView {
        zsize: 0,
        xsize: 0,
        ysize: 0,
        xysize: 0,
        binning: 0,
        cz: 0,
        refz: 0,
        xcenter: 0.,
        ycenter: 0.,
        xfixed: 0.,
        yfixed: 0.,
        use_fixed: 0,
        cur_chunk: 0,
        sminin: 0.,
        smaxin: 0.,
        cachein: 0,
        rot_mode: 0,
        global_rot: 0.,
        cos_stretch: 0,
        tilt_offset: 0.,
        num_chunks: 0,
        quiet: 0,
        image_for_channel: [1, 2, 1],
        chunk: vec![],
        tilt_angles: vec![],
        li: None,
        hin: None,
        usecount: 0,
        cachesize: 0,
        cache: vec![],
        unbinned_buf: vec![],
        tr: vec![],
        ref_present: false,
        ref_mean: 0.,
        ref_data: vec![],
        showref: 0,
        sangle: 0,
        phi: 0.,
        sdat: vec![],
        id: vec![],
        zoom: 1.,
        truezoom: 1.,
        zoomind: 6,
        bin_to_zoom_down: false,
        xtrans: 0,
        ytrans: 0,
        xoffset: 0,
        yoffset: 0,
        vmode: MIDAS_VIEW_COLOR,
        fast_interp: 1,
        lastmx: 0,
        lastmy: 0,
        firstmx: 0,
        firstmy: 0,
        mx: 0,
        my: 0,
        mousemoving: 0,
        width: 0,
        height: 0,
        device_pixel_ratio: 0.,
        dev_pix_varies: 0,
        screen_changed: false,
        orig_height: 0,
        startup_done: false,
        xtype: XTYPE_XF,
        xname: None,
        oname: None,
        refname: None,
        refzsize: 0,
        tiltname: None,
        changed: 0,
        didsave: 0,
        plname: None,
        blended_montage: None,
        blended_coords: None,
        blended_binning: 0,
        xsec: 0,
        corr_box_size: 0,
        corr_shift_limit: 0,
        corr_vals_entered: 0,
        minxpiece: 0,
        minypiece: 0,
        minzpiece: 0,
        maxzpiece: 0,
        nxpieces: 0,
        nypieces: 0,
        nxoverlap: 0,
        nyoverlap: 0,
        any_skipped: 0,
        robust_fit: 0,
        robust_crit: 1.,
        exclude_skipped: 0,
        nedge: [0; 2],
        maxedge: [0; 2],
        xory: 0,
        center_xory: -1,
        montcz: 0,
        curedge: 0,
        edgeind: 0,
        curleavex: 0.,
        curleavey: 0.,
        topind: [0; MAX_TOP_ERR],
        num_top_err: 6,
        skip_err: 0,
        depth: 0,
        exposed: 0,
        blackstate: 0,
        whitestate: 255,
        reversemap: 0,
        applytoone: 0,
        draw_corr_box: 0,
        keepsecdiff: 1,
        edit_warps: false,
        warping_ok: false,
        draw_vectors: false,
        cur_control: 0,
        cur_warp_file: -1,
        warp_nz: 0,
        warp_scale: 0.,
        max_warp_backup: 0,
        num_warp_backup: 0,
        grid_size: 0,
        old_mat: [0.; 9],
        last_grid_size: 0,
        last_warped_z: -1,
        last_nx_grid: 0,
        last_ny_grid: 0,
        last_xstart: 0.,
        last_ystart: 0.,
        last_xinterv: 0.,
        last_yinterv: 0.,
        last_mat: [0.; 9],
        incindex: [4; 3],
        increment: [0.; 3],
        paramstate: [-999.; 5],
        backup_mat: [0.; 9],
        backup_edgedx: 0.,
        backup_edgedy: 0.,
        mouse_xonly: 0,
        ctrl_pressed: 0,
        shift_pressed: 0,
        exiting: false,
        midas_window_present: false,
        midas_slots_present: false,
        midas_gl_present: false,
    }
}

/// C `MenuIDs` (`midas.h`).
#[repr(i32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MenuIds {
    FileMenuLoad,
    FileMenuSave,
    FileMenuSaveAs,
    FileMenuSaveImage,
    FileMenuTransform,
    FileMenuQuit,
    EditMenuStore,
    EditMenuReset,
    EditMenuRevert,
    EditMenuMirror,
    EditMenuDeletept,
    HelpMenuAbout,
    HelpMenuControls,
    HelpMenuHotkeys,
    HelpMenuMouse,
    HelpMenuManpage,
    LastMenuId,
}

/// Rust-side record of the source `MidasWindow` construction; it does not
/// substitute for Qt's QWidget/QOpenGLWidget runtime.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MidasWindow {
    pub double_buffer: bool,
    pub title: String,
    pub file_menu: Vec<String>,
    pub edit_menu: Vec<String>,
    pub help_menu: Vec<String>,
    pub controls: Vec<String>,
}

impl MidasWindow {
    pub fn new(double_buffer: bool, view: &mut MidasView) -> Self {
        let mut window = Self {
            double_buffer,
            ..Self::default()
        };
        window.file_menu = vec![
            "Load transforms",
            "Save transforms",
            "Save transforms as...",
            "Save contrast-scaled image...",
            "Transform model...",
            "Quit",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        window.edit_menu = vec![
            "Store section transform",
            "Reset to unit transform",
            "Revert to stored transform",
            "Mirror around X axis",
            "Delete control point",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        window.help_menu = vec!["Controls", "Hotkeys", "Mouse", "Man Page", "About Midas"]
            .into_iter()
            .map(str::to_owned)
            .collect();
        window.create_section_controls(view);
        window.create_contrast_controls(view);
        window.create_zoom_block(view);
        window.create_view_toggle(view);
        window.create_parameter_display(view);
        window.set_title_bar(view);
        view.midas_window_present = true;
        view.midas_gl_present = true;
        window
    }
    pub fn close_event(&mut self, view: &mut MidasView) {
        view.exiting = true;
    }
    pub fn key_press_event(&mut self, _key: i32) {}
    pub fn key_release_event(&mut self, key: i32, view: &mut MidasView) {
        if key == 0x0100_0021 {
            view.ctrl_pressed = 0;
        }
        if key == 0x0100_0020 {
            view.shift_pressed = 0;
        }
    }
    pub fn move_event(&mut self, view: &MidasView) {
        if view.dev_pix_varies != 0 {
            self.controls.push("extendTimerIfActive(500)".to_owned());
        }
    }
    pub fn make_separator(&mut self, width: i32) {
        self.controls.push(format!("separator:{width}"));
    }
    pub fn make_two_arrows(&mut self, direction: i32, signal: i32, repeat: bool) {
        self.controls
            .push(format!("arrows:{direction}:{signal}:{repeat}"));
    }
    pub fn make_arrow_row(
        &mut self,
        direction: i32,
        signal: i32,
        repeat: bool,
        text: &str,
        value: f32,
    ) {
        self.make_two_arrows(direction, signal, repeat);
        self.controls.push(format!("{text}:{value}"));
    }
    pub fn make_labeled_arrows(&mut self, text: &str, repeat: bool) {
        self.make_two_arrows(1, 1, repeat);
        self.controls.push(text.to_owned());
    }
    pub fn make_spin_box_row(&mut self, label: &str, minimum: i32, maximum: i32) {
        self.controls.push(format!("{label}:{minimum}..{maximum}"));
    }
    pub fn create_parameter_display(&mut self, view: &MidasView) {
        self.controls.extend(
            [
                "Rotation",
                "X translation",
                "Y translation",
                "Magnification",
                "Stretch",
            ]
            .map(str::to_owned),
        );
        if view.rot_mode != 0 {
            self.controls.push("Global rotation".to_owned());
        }
        if view.cos_stretch != 0 {
            self.controls.push("Cosine stretch".to_owned());
        }
    }
    pub fn create_section_controls(&mut self, view: &MidasView) {
        if view.xtype != XTYPE_MONT {
            self.make_spin_box_row(
                "Reference Sec.",
                1,
                if view.xtype == XTYPE_XREF {
                    view.refzsize
                } else {
                    view.zsize
                },
            );
        }
        self.make_spin_box_row(
            "Current Sec.",
            1,
            if view.xtype == XTYPE_MONT {
                view.maxzpiece + 1
            } else {
                view.zsize
            },
        );
        if view.num_chunks != 0 {
            self.make_spin_box_row("Current Chunk", 2, view.num_chunks);
        } else if view.xtype != XTYPE_MONT {
            self.controls.push("Keep Curr - Ref diff = 1".to_owned());
        }
    }
    pub fn create_zoom_block(&mut self, view: &MidasView) {
        self.make_labeled_arrows("Zoom  1.00", false);
        self.controls
            .push(format!("Zoom down by binning:{}", view.bin_to_zoom_down));
        self.controls
            .push(format!("Interpolate:{}", view.fast_interp == 0));
    }
    pub fn create_view_toggle(&mut self, view: &MidasView) {
        self.controls
            .push(format!("Overlay view:{}", view.vmode == MIDAS_VIEW_COLOR));
        self.controls.push("Toggle Ref/Cur".to_owned());
    }
    pub fn create_contrast_controls(&mut self, _view: &MidasView) {
        self.controls.extend(
            [
                "Black",
                "White",
                "Apply to only one sec.",
                "Reverse contrast",
                "Auto Contrast",
            ]
            .map(str::to_owned),
        );
    }
    pub fn set_title_bar(&mut self, view: &MidasView) {
        self.title = match &view.xname {
            Some(name) => format!(
                "midas: {}",
                Path::new(name)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
            ),
            None => "midas".to_owned(),
        };
    }
}

/// C `usage(void)` (`midas.cpp:63`).
pub fn usage() -> String {
    "midas version IMOD-Rust\nUsage: midas [options] <mrc filename> [transform filename]\nOptions:\n   -g\t\t Output global transforms (default is local)\n   -r <filename>\t Load reference image file\n   -rz <section>\t Section # for reference (default 0)\n   -p <filename>\t Load piece list file for fixing montages\n   -mb <filename>\t Name of blended montage for edge-finding in 3dmod\n   -mc <filename>\t Aligned piece coordinate file for edge-finding in 3dmod\n   -c <size list>\t Align chunks of sections\n   -cs <size list>\t Align chunks of sections; list # of sample slices\n   -B <factor>\t Bin images by factor\n   -C <size>\t Set cache size\n   -s <min,max>\t Set intensity scaling\n   -b 0\t\t Turn on interpolation\n   -a <angle>\t Rotate all images\n   -t <filename>\t Load tilt angles and allow cosine stretching\n   -o <filename>\t Output transforms to filename\n   -l <size,limit>\t Initial correlation box size and limit\n   -e <number>\t Show given number of buttons with largest edge errors\n   -O <letters>\t Colors of previous/current overlay\n   -S\t\t Use single-buffered visual\n   -D\t\t Debug mode\n   -q\t\t Suppress montage reminder\n".to_owned()
}

/// C `main(int, char **)` (`midas.cpp:102`) through its first untranslated
/// `load_view`/Qt endpoint.
pub fn midas_main(arguments: &[String]) -> Result<i32, String> {
    let mut view = MidasView::default();
    let mut double_buffer = true;
    let mut sample_slices = -1;
    let mut chunk_list = Vec::<String>::new();
    let mut oarg: Option<String> = None;
    let mut i = 1;
    if arguments.len() < 2 {
        return Err(usage());
    }
    while i < arguments.len() && arguments[i].starts_with('-') {
        let option = &arguments[i];
        if option == "-h" {
            return Err(usage());
        }
        match option.as_str() {
            "-r" => {
                i += 1;
                view.refname = Some(
                    arguments
                        .get(i)
                        .ok_or("midas: -r requires a filename")?
                        .clone(),
                );
            }
            "-rz" => {
                i += 1;
                view.xsec = arguments
                    .get(i)
                    .ok_or("midas: -rz requires a section")?
                    .parse()
                    .unwrap_or(0);
            }
            "-p" => {
                i += 1;
                view.plname = Some(
                    arguments
                        .get(i)
                        .ok_or("midas: -p requires a filename")?
                        .clone(),
                );
            }
            "-g" => view.xtype = XTYPE_XG,
            "-C" => {
                i += 1;
                view.cachein = arguments
                    .get(i)
                    .ok_or("midas: -C requires a size")?
                    .parse()
                    .unwrap_or(0);
            }
            "-b" => {
                i += 1;
                if arguments
                    .get(i)
                    .ok_or("midas: -b requires a value")?
                    .parse::<i32>()
                    .unwrap_or(0)
                    == 0
                {
                    view.fast_interp = 0;
                }
            }
            "-B" => {
                i += 1;
                view.binning = arguments
                    .get(i)
                    .ok_or("midas: -B requires a factor")?
                    .parse()
                    .unwrap_or(0)
                    .clamp(1, 8);
            }
            "-e" => {
                i += 1;
                view.num_top_err = arguments
                    .get(i)
                    .ok_or("midas: -e requires a number")?
                    .parse()
                    .unwrap_or(0);
            }
            "-a" => {
                i += 1;
                view.global_rot = arguments
                    .get(i)
                    .ok_or("midas: -a requires an angle")?
                    .parse()
                    .unwrap_or(0.);
                view.rot_mode = 1;
            }
            "-t" => {
                i += 1;
                view.tiltname = Some(
                    arguments
                        .get(i)
                        .ok_or("midas: -t requires a filename")?
                        .clone(),
                );
                view.cos_stretch = -1;
                view.rot_mode = 1;
            }
            "-s" => {
                i += 1;
                let values = arguments
                    .get(i)
                    .ok_or("midas: -s requires min,max")?
                    .split(',')
                    .collect::<Vec<_>>();
                view.sminin = values.first().and_then(|v| v.parse().ok()).unwrap_or(0.);
                view.smaxin = values.get(1).and_then(|v| v.parse().ok()).unwrap_or(0.);
            }
            "-D" => MIDAS_DEBUG.store(true, Ordering::Relaxed),
            "-q" => view.quiet = 1,
            "-S" => double_buffer = false,
            "-c" | "-cs" => {
                if sample_slices >= 0 {
                    return Err("ERROR: midas - You cannot enter both -c and -cs".to_owned());
                }
                sample_slices = i32::from(option == "-cs");
                i += 1;
                chunk_list = arguments
                    .get(i)
                    .ok_or("midas: chunk option requires a size list")?
                    .split(',')
                    .filter(|v| !v.is_empty())
                    .map(str::to_owned)
                    .collect();
                view.num_chunks = if sample_slices != 0 {
                    (chunk_list.len() / 2 + 1) as i32
                } else {
                    chunk_list.len() as i32
                };
            }
            "-o" => {
                i += 1;
                view.oname = Some(
                    arguments
                        .get(i)
                        .ok_or("midas: -o requires a filename")?
                        .clone(),
                );
            }
            "-O" => {
                i += 1;
                oarg = Some(
                    arguments
                        .get(i)
                        .ok_or("midas: -O requires two letters")?
                        .clone(),
                );
            }
            "-mb" => {
                i += 1;
                view.blended_montage = Some(
                    arguments
                        .get(i)
                        .ok_or("midas: -mb requires a filename")?
                        .clone(),
                );
            }
            "-mc" => {
                i += 1;
                view.blended_coords = Some(
                    arguments
                        .get(i)
                        .ok_or("midas: -mc requires a filename")?
                        .clone(),
                );
            }
            _ => {
                return Err(format!(
                    "ERROR: midas - Illegal option entered: {option}\n{}",
                    usage()
                ));
            }
        };
        i += 1;
    }
    if i < arguments.len().saturating_sub(2) || i == arguments.len() {
        return Err(usage());
    }
    let image_name = arguments.get(i).ok_or_else(usage)?.clone();
    if i == arguments.len() - 2 {
        view.xname = arguments.last().cloned();
        if view
            .xname
            .as_ref()
            .is_some_and(|name| Path::new(name).is_file())
        {
            view.didsave = -1;
        }
    }
    if let Some(colors) = oarg {
        if colors.len() != 2 {
            return Err("Two letters must be entered with -O: two of r g b c m y".to_owned());
        }
        view.image_for_channel = [0; 3];
        for (position, color) in colors.bytes().enumerate() {
            let add = position as i32 + 1;
            match color {
                b'r' => view.image_for_channel[2] += add,
                b'g' => view.image_for_channel[1] += add,
                b'b' => view.image_for_channel[0] += add,
                b'c' => {
                    view.image_for_channel[1] += add;
                    view.image_for_channel[0] += add;
                }
                b'm' => {
                    view.image_for_channel[2] += add;
                    view.image_for_channel[0] += add;
                }
                b'y' => {
                    view.image_for_channel[2] += add;
                    view.image_for_channel[1] += add;
                }
                _ => {
                    return Err("The letters entered with -O must be two of r g b c m y".to_owned());
                }
            }
        }
        if view.image_for_channel.iter().any(|channel| *channel > 2) {
            return Err("The two letters entered with -O must specify different color channels for previous and current images".to_owned());
        }
    }
    if view.plname.is_some() {
        if view.refname.is_some() || view.rot_mode != 0 || view.num_chunks != 0 {
            return Err(
                "You cannot use the -p option with the -r, -a, -t, or -c option.".to_owned(),
            );
        }
        view.xtype = XTYPE_MONT;
        let root = Path::new(&image_name).with_extension("");
        if view.blended_montage.is_none() {
            let candidate = format!("{}_preblend.mrc", root.display());
            if Path::new(&candidate).exists() {
                view.blended_montage = Some(candidate);
            }
        }
        if view.blended_coords.is_none() {
            let candidate = format!("{}.alipl", root.display());
            if Path::new(&candidate).exists() {
                view.blended_coords = Some(candidate);
            }
        }
        if view
            .blended_coords
            .as_ref()
            .is_some_and(|file| !Path::new(file).exists())
        {
            return Err(
                "The aligned piece coordinate file entered with -mc does not exist".to_owned(),
            );
        }
        if view
            .blended_montage
            .as_ref()
            .is_some_and(|file| !Path::new(file).exists())
        {
            return Err("The blended montage file entered with -mb does not exist".to_owned());
        }
    }
    if view.refname.is_some() || view.num_chunks != 0 {
        view.rot_mode = 0;
        view.cos_stretch = 0;
        if view.refname.is_some() {
            view.xtype = XTYPE_XREF;
        }
    }
    if view.cos_stretch != 0 && view.xtype == XTYPE_XG {
        return Err("Global alignment mode cannot be used with cosine stretching".to_owned());
    }
    if view.num_chunks != 0 {
        if view.refname.is_some() {
            return Err("Chunk alignment cannot be done in reference alignment mode".to_owned());
        }
        if sample_slices != 0 && chunk_list.len() % 2 != 0 {
            return Err("A list of sample slices must have an even number of values".to_owned());
        }
        if view.num_chunks < 2
            || chunk_list
                .iter()
                .any(|value| value.parse::<i32>().unwrap_or(0) <= 0)
        {
            return Err(if sample_slices != 0 {
                "The -cs option must be followed by a comma-separated list of the number of slices in each bottom and top sample.".to_owned()
            } else {
                "The -c option must be followed by a comma-separated list of the number of sections in each chunk.".to_owned()
            });
        }
        view.chunk = vec![MidasChunk::default(); view.num_chunks as usize + 1];
        for chunk_index in 0..view.num_chunks as usize {
            let (bottom, top) = if sample_slices != 0 {
                (
                    if chunk_index == 0 {
                        0
                    } else {
                        chunk_list[2 * chunk_index - 1].parse().unwrap_or(0)
                    },
                    if chunk_index == view.num_chunks as usize - 1 {
                        0
                    } else {
                        chunk_list[2 * chunk_index].parse().unwrap_or(0)
                    },
                )
            } else {
                (0, chunk_list[chunk_index].parse().unwrap_or(0))
            };
            view.chunk[chunk_index].size = bottom + top;
            view.chunk[chunk_index + 1].start =
                view.chunk[chunk_index].start + view.chunk[chunk_index].size;
            view.chunk[chunk_index].min_ref_sec = view.chunk[chunk_index].start + bottom;
            view.chunk[chunk_index].max_cur_sec = if sample_slices != 0 {
                view.chunk[chunk_index].min_ref_sec - 1
            } else {
                view.chunk[chunk_index + 1].start - 1
            };
        }
    }
    view.warping_ok = view.xtype != XTYPE_XG && view.xtype != XTYPE_MONT && view.rot_mode == 0;
    let _window = MidasWindow::new(double_buffer, &mut view);
    // C `load_view` begins its image setup with `load_image`; keep this as a
    // direct source-unit call rather than duplicating its ownership boundary
    // in the executable unit.
    crate::imod::midas::file_io::load_image(&mut view, Path::new(&image_name))?;
    if view.cos_stretch != 0 {
        crate::imod::midas::file_io::load_angles(&mut view)?;
    }
    Err(
        "midas: the image-cache/Qt/OpenGL continuation after load_view is not yet translated"
            .into(),
    )
}

/// C `midas_error(const char *, const char *, int)` without C process exit.
pub fn midas_error(top_message: &str, bottom_message: &str, retval: i32) -> Result<(), String> {
    if retval != 0 {
        Err(format!("{top_message} {bottom_message}"))
    } else {
        Ok(())
    }
}

/// C `printStderr(const char *, ...)`; Rust callers provide the already-formatted text.
pub fn print_stderr(message: &str) {
    eprint!("{message}");
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn chunks_follow_the_source_start_boundaries() {
        let input = vec![
            "midas".into(),
            "-c".into(),
            "2,3".into(),
            format!(
                "{}/fixtures/newstack-warp-input.mrc",
                env!("CARGO_MANIFEST_DIR")
            ),
        ];
        let error = midas_main(&input).unwrap_err();
        // `load_image` now owns the real MRC header/load conversion state;
        // the remaining source boundary is the image-cache/GL continuation.
        assert!(error.contains("image-cache/Qt/OpenGL continuation"));
    }
    #[test]
    fn overlay_rejects_channel_overlap() {
        let input = vec!["midas".into(), "-O".into(), "rr".into(), "in.mrc".into()];
        assert!(midas_main(&input).unwrap_err().contains("different color"));
    }
}
