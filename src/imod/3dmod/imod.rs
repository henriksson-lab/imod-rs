//! Translation of `IMOD/3dmod/imod.cpp` together with its public `imod.h` API.
//!
//! This is the 3dmod program's orchestration unit.  Its terminal calls are the
//! separately translated Qt/OpenGL units (`display.cpp`, `imodview.cpp`,
//! `info_setup.cpp`, `xzap.cpp`, and `slicer.cpp`).  It deliberately reports
//! that boundary rather than pretending to be a viewer.

use std::ffi::CString;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::{LazyLock, Mutex};

use crate::imod::libiimod::iimage::ii_add_check_function;

pub const IMOD_DRAW_IMAGE: i32 = 1;
pub const IMOD_DRAW_XYZ: i32 = 1 << 1;
pub const IMOD_DRAW_MOD: i32 = 1 << 2;
pub const IMOD_DRAW_SLICE: i32 = 1 << 3;
pub const IMOD_DRAW_SKIPMODV: i32 = 1 << 5;
pub const IMOD_DRAW_COLORMAP: i32 = 1 << 11;
pub const IMOD_DRAW_NOSYNC: i32 = 1 << 12;
pub const IMOD_DRAW_RETHINK: i32 = 1 << 13;
pub const IMOD_DRAW_ACTIVE: i32 = 1 << 14;
pub const IMOD_DRAW_TOP: i32 = 1 << 15;
pub const IMOD_DRAW_ALL: i32 = IMOD_DRAW_IMAGE | IMOD_DRAW_XYZ | IMOD_DRAW_MOD;
pub const IMOD_PLUG_MENU: i32 = 1;
pub const IMOD_PLUG_TOOL: i32 = 2;
pub const IMOD_PLUG_PROC: i32 = 4;
pub const IMOD_PLUG_VIEW: i32 = 8;
pub const IMOD_PLUG_KEYS: i32 = 16;
pub const IMOD_PLUG_FILE: i32 = 32;
pub const IMOD_PLUG_MESSAGE: i32 = 64;
pub const IMOD_PLUG_MOUSE: i32 = 128;
pub const IMOD_PLUG_EVENT: i32 = 256;
pub const IMOD_PLUG_CHOOSER: i32 = 512;
pub const IMOD_REASON_EXECUTE: i32 = 1;
pub const IMOD_REASON_STARTUP: i32 = 3;
pub const IMOD_REASON_MODUPDATE: i32 = 4;
pub const IMOD_REASON_NEWMODEL: i32 = 5;
pub const SNAP_SHOT_DEFAULT: i32 = 0;
pub const SNAP_SHOT_RGB: i32 = 1;
pub const SNAP_SHOT_TIF: i32 = 2;
pub const SNAP_SHOT_PNG: i32 = 3;
pub const SNAP_SHOT_JPG: i32 = 4;

/// `ImodApp` (`imodP.h`), fields initialized by `imod.cpp::main`.
#[derive(Clone, Debug, Default)]
pub struct ImodApp {
    pub rgba: i32,
    pub exiting: i32,
    pub closing: i32,
    pub base: i32,
    pub convert_snap: i32,
    pub gl_initialized: i32,
    pub chooser_plugin: i32,
    pub info_initial_setup: i32,
    pub new_qt_open_gl: i32,
    pub is_windows: i32,
    pub background: i32,
    pub foreground: i32,
    pub select: i32,
    pub shadow: i32,
    pub endpoint: i32,
    pub bgnpoint: i32,
    pub curpoint: i32,
    pub ghost: i32,
    pub arrow: [i32; 4],
    pub listening: i32,
}

/// Source globals `App`, `Model`, `Imod_imagefile`, `Imod_IFDpath`, and `Imod_cwdpath`.
pub static APP: LazyLock<Mutex<Option<ImodApp>>> = LazyLock::new(|| Mutex::new(None));
pub static MODEL_LOADED: AtomicBool = AtomicBool::new(false);
pub static IMOD_IMAGEFILE: LazyLock<Mutex<Option<String>>> = LazyLock::new(|| Mutex::new(None));
pub static IMOD_IFD_PATH: LazyLock<Mutex<String>> = LazyLock::new(|| Mutex::new(String::new()));
pub static IMOD_CWD_PATH: LazyLock<Mutex<String>> = LazyLock::new(|| Mutex::new(String::new()));
pub static IMOD_DEBUG: AtomicBool = AtomicBool::new(false);
pub static IMOD_TRANS: AtomicBool = AtomicBool::new(true);
pub static RAMP_BASE: AtomicI32 = AtomicI32::new(0);
static LOOP_STARTED: AtomicBool = AtomicBool::new(false);
static DEBUG_KEYS: LazyLock<Mutex<Option<String>>> = LazyLock::new(|| Mutex::new(None));
static WINDOW_KEYS: LazyLock<Mutex<Option<String>>> = LazyLock::new(|| Mutex::new(None));
static INITIAL_ZOOM: Mutex<f32> = Mutex::new(0.);

/// Values from `IloadInfo`/`ImodView` set by `imod.cpp::main` before image loading.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ImodLaunch {
    pub image_files: Vec<String>,
    pub model_file: Option<String>,
    pub xyz_window_open: bool,
    pub slicer_open: bool,
    pub zap_open: bool,
    pub model_view_open: bool,
    pub fill_cache: bool,
    pub startup_dialog: bool,
    pub imodv: bool,
    pub print_window_id: bool,
    pub data_from_stdin: bool,
    pub use_stdin: bool,
    pub no_fork: bool,
    pub x_range: Option<(i32, i32)>,
    pub y_range: Option<(i32, i32)>,
    pub z_range: Option<(i32, i32)>,
    pub scale: Option<(f32, f32)>,
    pub xy_bin: Option<i32>,
    pub z_bin: Option<i32>,
    pub initial_center: Option<(f32, f32, f32)>,
    pub initial_angles: Option<(f32, f32, f32)>,
    pub initial_tilt: Option<f32>,
    pub initial_zoom: Option<f32>,
    pub cache_size: Option<i32>,
    pub tile_cache: bool,
    pub raw_size: Option<(i32, i32, i32)>,
    pub raw_mode: Option<i32>,
    pub raw_header_size: Option<i32>,
    pub raw_inverted: bool,
    pub raw_swap: bool,
    pub raw_image_store: Option<i32>,
    pub integer_option_entered: bool,
    pub scale_scan_type: Option<i32>,
    pub store_scan_in_mrc: bool,
    pub mirror_fft: bool,
    pub rotate_yz: bool,
    pub grayscale_rgb: bool,
    pub two_dimensional: bool,
    pub multi_file_z: bool,
    pub eer_super_resolution: Option<i32>,
    pub eer_z_binning: Option<i32>,
    pub frames: bool,
    pub pyramid: bool,
    pub angle_file: Option<String>,
    pub piece_list_file: Option<String>,
    pub aligned_piece_file: Option<String>,
    pub piece_key_type: Option<i32>,
    pub frame_layout: Option<(i32, i32)>,
    pub frame_overlap: Option<(i32, i32)>,
    pub window_keys: Option<String>,
    pub style: Option<String>,
}

/// `imod_usage(char *)` (`imod.cpp:83`).
pub fn imod_usage(name: &str) -> String {
    let name = name.rsplit('/').next().unwrap_or(name);
    format!(
        "Usage: {name} [options] <Image files> <model file>\nOptions:\n   -xyz  Open xyz window first.\n   -S    Open slicer window first.\n   -V    Open model view window first.\n   -Z    Open Zap window (use with -S, -xyz, or -V).\n   -O    Set options with startup dialog box.\n   -x min,max  Load in sub image.\n   -y min,max  Load in sub image.\n   -z min,max  Load in sub image.\n   -s min,max  Scale input to range [min,max].\n   -ic x,y,z   Set initial center position.\n   -ia x,y,z   Set initial angles.\n   -iz #       Set initial zoom.\n   -it #       Set initial tilt angle.\n   -C #  Set cache size.\n   -F    Fill cache right after starting program.\n   -B #  Bin images in X, Y, and Z.\n   -b nxy,nz  Bin images separately.\n   -E <keys>  Open windows specified by key letters.\n   -h    Print this help message.\n"
    )
}

/// Private `badOption(const char *)` (`imod.cpp:1278`).
pub fn bad_option(option: &str) -> String {
    format!(
        "3dmod: The argument {option} is not a valid option.\nIf it is a filename, put ./ in front of it"
    )
}

/// `main(int, char **)` (`imod.cpp:128`), through its first untranslated display call.
pub fn imod_main(arguments: &[String]) -> Result<i32, String> {
    let program = arguments.first().map_or("3dmod", String::as_str);
    let app = ImodApp {
        rgba: 1,
        base: RAMP_BASE.load(Ordering::Relaxed),
        info_initial_setup: 2,
        new_qt_open_gl: 1,
        is_windows: i32::from(cfg!(windows)),
        background: 1,
        foreground: 2,
        select: 3,
        shadow: 4,
        endpoint: 5,
        bgnpoint: 6,
        curpoint: 7,
        ghost: 8,
        arrow: [9, 10, 11, 12],
        ..Default::default()
    };
    let mut launch = ImodLaunch {
        imodv: program.ends_with('v'),
        mirror_fft: true,
        ..Default::default()
    };
    let mut first_file = false;
    let mut i = 1;
    while i < arguments.len() {
        let arg = &arguments[i];
        if !arg.starts_with('-') || arg == "-" {
            first_file = true;
            launch.image_files.push(arg.clone());
            i += 1;
            continue;
        }
        if first_file {
            return Err(format!(
                "3dmod: invalid to have argument {arg} after first filename"
            ));
        }
        if (arg.starts_with("-x") || arg.starts_with("-y") || arg.starts_with("-z"))
            && arg.len() > 2
            && arg.as_bytes()[2].is_ascii_digit()
        {
            let (a, b) = arg[2..].split_once(',').ok_or_else(|| bad_option(arg))?;
            let pair = (
                a.parse().map_err(|_| bad_option(arg))?,
                b.parse().map_err(|_| bad_option(arg))?,
            );
            if arg.starts_with("-x") {
                launch.x_range = Some(pair);
            } else if arg.starts_with("-y") {
                launch.y_range = Some(pair);
            } else {
                launch.z_range = Some(pair);
            };
            i += 1;
            continue;
        }
        match arg.as_str() {
            "-h" => return Err(imod_usage(program)),
            "-xyz" => launch.xyz_window_open = true,
            "-S" => launch.slicer_open = true,
            "-V" => launch.model_view_open = true,
            "-Z" => launch.zap_open = true,
            "-O" => launch.startup_dialog = true,
            "-F" => launch.fill_cache = true,
            "-f" => launch.frames = true,
            "-py" => launch.pyramid = true,
            "-W" => {
                launch.print_window_id = true;
                launch.no_fork = true;
            }
            "-R" => {
                launch.data_from_stdin = true;
                launch.no_fork = true;
            }
            "-L" => {
                launch.use_stdin = true;
                launch.no_fork = true;
            }
            "-modv" | "-view" => launch.imodv = true,
            "-ci" => {}
            "-m" => IMOD_TRANS.store(false, Ordering::Relaxed),
            "-CT" => launch.tile_cache = true,
            "-ri" => launch.raw_inverted = true,
            "-w" => launch.raw_swap = true,
            "-D" => {
                IMOD_DEBUG.store(true, Ordering::Relaxed);
                launch.no_fork = true;
            }
            "-Y" => launch.rotate_yz = true,
            "-G" => launch.grayscale_rgb = true,
            "-2" => launch.two_dimensional = true,
            "-T" => launch.multi_file_z = true,
            "-K" => launch.store_scan_in_mrc = true,
            "-M" => launch.mirror_fft = false,
            "-C" => {
                i += 1;
                let v = arguments.get(i).ok_or_else(|| bad_option(arg))?;
                let mut n = v
                    .trim_end_matches(['M', 'm', 'G', 'g'])
                    .parse::<i32>()
                    .map_err(|_| bad_option(arg))?;
                if v.ends_with(['M', 'm']) {
                    n = -n
                } else if v.ends_with(['G', 'g']) {
                    n = -n * 1024
                };
                launch.cache_size = Some(if n == 0 { i32::MAX } else { n });
            }
            "-x" | "-y" | "-z" | "-s" | "-b" | "-P" | "-o" | "-r" => {
                i += 1;
                let v = arguments.get(i).ok_or_else(|| bad_option(arg))?;
                let values = v.split(',').collect::<Vec<_>>();
                if arg == "-r" {
                    if values.len() != 3 {
                        return Err(bad_option(arg));
                    };
                    launch.raw_size = Some((
                        values[0].parse().map_err(|_| bad_option(arg))?,
                        values[1].parse().map_err(|_| bad_option(arg))?,
                        values[2].parse().map_err(|_| bad_option(arg))?,
                    ));
                } else if arg == "-b" {
                    launch.xy_bin = Some(values[0].parse().map_err(|_| bad_option(arg))?);
                    launch.z_bin = Some(
                        values
                            .get(1)
                            .unwrap_or(&"1")
                            .parse()
                            .map_err(|_| bad_option(arg))?,
                    );
                } else {
                    if values.len() != 2 {
                        return Err(bad_option(arg));
                    };
                    if arg == "-x" {
                        launch.x_range = Some((
                            values[0].parse().map_err(|_| bad_option(arg))?,
                            values[1].parse().map_err(|_| bad_option(arg))?,
                        ));
                    } else if arg == "-y" {
                        launch.y_range = Some((
                            values[0].parse().map_err(|_| bad_option(arg))?,
                            values[1].parse().map_err(|_| bad_option(arg))?,
                        ));
                    } else if arg == "-z" {
                        launch.z_range = Some((
                            values[0].parse().map_err(|_| bad_option(arg))?,
                            values[1].parse().map_err(|_| bad_option(arg))?,
                        ));
                    } else if arg == "-s" {
                        launch.scale = Some((
                            values[0].parse().map_err(|_| bad_option(arg))?,
                            values[1].parse().map_err(|_| bad_option(arg))?,
                        ));
                    } else if arg == "-P" {
                        launch.frame_layout = Some((
                            values[0].parse().map_err(|_| bad_option(arg))?,
                            values[1].parse().map_err(|_| bad_option(arg))?,
                        ));
                    } else {
                        launch.frame_overlap = Some((
                            values[0].parse().map_err(|_| bad_option(arg))?,
                            values[1].parse().map_err(|_| bad_option(arg))?,
                        ));
                    }
                }
            }
            "-B" | "-iz" | "-it" | "-es" | "-ez" | "-I" | "-J" | "-t" | "-H" | "-A" | "-E"
            | "-a" | "-p" | "-pa" | "-cm" => {
                i += 1;
                let v = arguments.get(i).ok_or_else(|| bad_option(arg))?.clone();
                match arg.as_str() {
                    "-B" => {
                        let n = v.parse().map_err(|_| bad_option(arg))?;
                        launch.xy_bin = Some(n);
                        launch.z_bin = Some(n);
                    }
                    "-iz" => {
                        let n = v.parse().map_err(|_| bad_option(arg))?;
                        launch.initial_zoom = Some(n);
                        *INITIAL_ZOOM.lock().unwrap() = n;
                    }
                    "-it" => launch.initial_tilt = Some(v.parse().map_err(|_| bad_option(arg))?),
                    "-es" => {
                        launch.eer_super_resolution =
                            Some(v.parse::<i32>().map_err(|_| bad_option(arg))?.clamp(-3, 2))
                    }
                    "-ez" => {
                        launch.eer_z_binning =
                            Some(v.parse::<i32>().map_err(|_| bad_option(arg))?.max(1))
                    }
                    "-I" => {
                        launch.raw_image_store =
                            Some(if v.parse::<i32>().map_err(|_| bad_option(arg))? != 0 {
                                6
                            } else {
                                0
                            });
                        launch.integer_option_entered = true;
                    }
                    "-J" => {
                        launch.scale_scan_type =
                            Some(v.parse::<i32>().map_err(|_| bad_option(arg))?.clamp(0, 2))
                    }
                    "-t" => launch.raw_mode = Some(v.parse().map_err(|_| bad_option(arg))?),
                    "-H" => launch.raw_header_size = Some(v.parse().map_err(|_| bad_option(arg))?),
                    "-A" => {
                        launch.piece_key_type =
                            Some(v.parse::<i32>().map_err(|_| bad_option(arg))?.clamp(0, 2))
                    }
                    "-E" => {
                        let mut keys = WINDOW_KEYS.lock().unwrap();
                        keys.get_or_insert_with(String::new).push_str(&v);
                        launch.window_keys = keys.clone();
                    }
                    "-a" => launch.angle_file = Some(v),
                    "-p" => launch.piece_list_file = Some(v),
                    "-pa" => launch.aligned_piece_file = Some(v),
                    _ => {}
                }
            }
            "-ic" | "-ia" => {
                i += 1;
                let v = arguments.get(i).ok_or_else(|| bad_option(arg))?;
                let x = v.split(',').collect::<Vec<_>>();
                if x.len() != 3 {
                    return Err(bad_option(arg));
                };
                let t = (
                    x[0].parse().map_err(|_| bad_option(arg))?,
                    x[1].parse().map_err(|_| bad_option(arg))?,
                    x[2].parse().map_err(|_| bad_option(arg))?,
                );
                if arg == "-ic" {
                    launch.initial_center = Some(t)
                } else {
                    launch.initial_angles = Some(t)
                }
            }
            _ if arg.starts_with("-D") => {
                IMOD_DEBUG.store(true, Ordering::Relaxed);
                launch.no_fork = true;
                *DEBUG_KEYS.lock().unwrap() = Some(arg[2..].to_owned());
            }
            _ if arg.starts_with("-style=") => launch.style = Some(arg[7..].to_owned()),
            "-style" => {
                i += 1;
                launch.style = Some(arguments.get(i).ok_or_else(|| bad_option(arg))?.clone())
            }
            _ => return Err(bad_option(arg)),
        };
        i += 1;
    }
    if launch.image_files.len() > 1 {
        launch.model_file = launch.image_files.pop();
    }
    if launch.data_from_stdin
        && (launch.x_range.is_some()
            || launch.y_range.is_some()
            || launch.z_range.is_some()
            || launch.piece_list_file.is_some()
            || launch.use_stdin
            || launch.raw_size.is_some()
            || launch.raw_mode.is_some()
            || launch.raw_header_size.is_some()
            || launch.raw_inverted
            || launch.raw_swap
            || launch.cache_size.is_some()
            || launch.pyramid
            || !launch.image_files.is_empty())
    {
        return Err("3dmod: You cannot use -C, -L, -p, -py, raw options, subareas, or image files when reading data from stdin".to_owned());
    }
    *APP.lock().unwrap() = Some(app);
    // `imod.cpp:814`: QImage is deliberately appended after the default
    // image checks so registered plugins retain their source precedence.
    unsafe { ii_add_check_function(Some(super::iiqimage::ii_q_image_check)) };
    if launch.imodv {
        // `imod.cpp` calls `imodv_main(argcHere, argv)` here.  `imodv.rs`
        // deliberately retains the source C-compatible entry signature.
        let strings = arguments
            .iter()
            .map(|argument| {
                CString::new(argument.as_str())
                    .map_err(|_| "3dmod: NUL byte in argument".to_owned())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let pointers = strings
            .iter()
            .map(|argument| argument.as_ptr())
            .collect::<Vec<_>>();
        let status = unsafe { super::imodv::imodv_main(pointers.len() as i32, pointers.as_ptr()) };
        return Err(format!(
            "3dmodv boundary after imod.cpp dispatched imodv_main: status {status}; IMOD/3dmod/mv_window.cpp Qt/OpenGL window closure is not translated yet"
        ));
    }
    Err(format!(
        "3dmod viewer boundary after imod.cpp argument processing: display.cpp, info_setup.cpp, imodview.cpp, xzap.cpp, slicer.cpp, imod_io.cpp, preferences.cpp, imodplug.cpp, cachefill.cpp, pyramidcache.cpp, and client_message.cpp require translation; requested images: {:?}; model: {:?}",
        launch.image_files, launch.model_file
    ))
}

pub fn imod_loop_started() -> bool {
    LOOP_STARTED.load(Ordering::Relaxed)
}
pub fn imod_exit(return_code: i32) -> i32 {
    if let Some(app) = APP.lock().unwrap().as_mut() {
        app.closing = 1;
        app.exiting = 1
    };
    return_code
}
pub fn imod_quit() -> i32 {
    imod_exit(0)
}
pub fn imod_debug(key: char) -> bool {
    IMOD_DEBUG.load(Ordering::Relaxed)
        && DEBUG_KEYS
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|keys| keys.contains(key))
}
pub fn window_keys_has(key: char) -> bool {
    WINDOW_KEYS
        .lock()
        .unwrap()
        .as_ref()
        .is_some_and(|keys| keys.contains(key))
}
pub fn imod_initial_zoom() -> f32 {
    *INITIAL_ZOOM.lock().unwrap()
}
pub fn imod_draw_model() -> Result<(), String> {
    Err("3dmod OpenGL boundary: model_draw.cpp is not translated yet".to_owned())
}
pub fn imod_depth() -> i32 {
    APP.lock().unwrap().as_ref().map_or(0, |app| app.rgba * 24)
}
pub fn imod_color_value(color: i32) -> i32 {
    APP.lock().unwrap().as_ref().map_or(0, |app| match color {
        1 => app.background,
        2 => app.foreground,
        3 => app.select,
        4 => app.shadow,
        5 => app.endpoint,
        6 => app.bgnpoint,
        7 => app.curpoint,
        8 => app.ghost,
        9 => app.arrow[0],
        10 => app.arrow[1],
        11 => app.arrow[2],
        12 => app.arrow[3],
        _ => 0,
    })
}
pub fn wprint(message: &str) {
    print!("{message}");
}
pub fn imod_error(message: &str) {
    eprint!("{message}");
}
pub fn imod_print_stderr(message: &str) {
    eprint!("{message}");
}
pub fn imod_trace(key: char, message: &str) {
    if imod_debug(key) {
        imod_print_stderr(message)
    }
}
pub fn imod_puts(message: &str) {
    eprintln!("{message}");
}
pub fn imod_print_info(message: &str) {
    println!("{message}");
}
pub fn imod_default_keys() -> Result<(), String> {
    Err("3dmod input boundary: imod_input.cpp is not translated yet".to_owned())
}
pub fn imod_show_help_page(_page: &str) -> Result<(), String> {
    Err("3dmod help boundary: imod_assistant Qt integration is not translated yet".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parse_and_stop_at_viewer_boundary() {
        let args = [
            "3dmod",
            "-xyz",
            "-x2,8",
            "-b",
            "3,2",
            "-E",
            "ZS",
            "image.mrc",
            "model.mod",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
        assert!(
            imod_main(&args)
                .unwrap_err()
                .starts_with("3dmod viewer boundary")
        );
        assert!(window_keys_has('Z'));
    }
}
