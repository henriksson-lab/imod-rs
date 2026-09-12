//! Translation of `IMOD/3dmod/preferences.cpp` and `preferences.h`.
//!
//! Qt's `QSettings`, widget tree, style factory, and window-manager calls are
//! deliberately represented by `PreferencesNativeBoundary`.  The source-owned
//! preference state and non-Qt policy remain in this translation unit.
#![allow(dead_code)]

use std::collections::BTreeMap;

pub const HOT_SLIDER_KEYUP: i32 = 0;
pub const HOT_SLIDER_KEYDOWN: i32 = 1;
pub const NO_HOT_SLIDER: i32 = 2;
pub const MAX_ZOOMS: usize = 22;
pub const MAX_GEOMETRIES: usize = 10;
pub const MAX_NAMED_COLORS: usize = 12;
pub const MAX_STYLES: usize = 24;

#[derive(Clone, Debug, PartialEq)]
pub struct Triplet<T> {
    pub value: T,
    pub dflt: T,
    pub chgd: bool,
}
impl<T: Default + Clone> Default for Triplet<T> {
    fn default() -> Self {
        Self {
            value: T::default(),
            dflt: T::default(),
            chgd: false,
        }
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NewObjectProps {
    pub flags: u32,
    pub pdrawsize: i32,
    pub symbol: i32,
    pub symsize: i32,
    pub linewidth2: i32,
    pub symflags: i32,
    pub point_limit: i32,
    pub fill_trans: i32,
}

/// `imod_pref_struct`.  Each upstream `TRIPLET` is a `Triplet` here.
#[derive(Clone, Debug, PartialEq)]
pub struct ImodPrefStruct {
    pub hot_slider_key: Triplet<i32>,
    pub hot_slider_flag: Triplet<i32>,
    pub mouse_mapping: Triplet<i32>,
    pub modv_swap_left_mid: Triplet<bool>,
    pub allow_ctrl_on_mac: Triplet<bool>,
    pub silent_beep: Triplet<bool>,
    pub classic_slicer: Triplet<bool>,
    pub start_in_hq: Triplet<bool>,
    pub arrows_scroll_zap: Triplet<bool>,
    pub start_at_mid_z: Triplet<bool>,
    pub auto_con_at_start: Triplet<i32>,
    pub attach_to_on_obj: Triplet<bool>,
    pub slicer_new_surf: Triplet<bool>,
    pub font: String,
    pub font_chgd: bool,
    pub style_key: String,
    pub style_chgd: bool,
    pub bw_step: Triplet<i32>,
    pub page_step: Triplet<i32>,
    pub iconify_imodv_dlg: Triplet<bool>,
    pub iconify_imod_dlg: Triplet<bool>,
    pub iconify_image_win: Triplet<bool>,
    pub stack_imod_dlgs: Triplet<bool>,
    pub stack_imodv_dlgs: Triplet<bool>,
    pub raise_imod_dlg_stack: Triplet<bool>,
    pub raise_imodv_dlg_stack: Triplet<bool>,
    pub keep_dlg_stack_on_top: Triplet<bool>,
    pub dlg_frame_adjustment: Triplet<i32>,
    pub eer_super_res: Triplet<i32>,
    pub eer_zbinning: Triplet<i32>,
    pub min_mod_pt_size: Triplet<i32>,
    pub min_im_pt_size: Triplet<i32>,
    pub boost_model_cursor: Triplet<i32>,
    pub zooms: [f64; MAX_ZOOMS],
    pub zooms_dflt: [f64; MAX_ZOOMS],
    pub zooms_chgd: bool,
    pub autosave_interval: Triplet<i32>,
    pub autosave_on: Triplet<bool>,
    pub autosave_no_cont_mesh: Triplet<bool>,
    pub autosave_no_iso_mesh: Triplet<bool>,
    pub autosave_dir: Triplet<String>,
    pub remember_geom: Triplet<bool>,
    pub auto_target_mean: Triplet<i32>,
    pub auto_target_sd: Triplet<i32>,
    pub named_index: [i32; MAX_NAMED_COLORS],
    pub named_color: [u32; MAX_NAMED_COLORS],
    pub named_color_dflt: [u32; MAX_NAMED_COLORS],
    pub named_color_chgd: [bool; MAX_NAMED_COLORS],
    pub snap_format: Triplet<String>,
    pub snap_quality: Triplet<i32>,
    pub snap_dpi: Triplet<i32>,
    pub scale_snap_dpi: Triplet<bool>,
    pub no_cur_pnt_on_snaps: Triplet<bool>,
    pub tiff_compression: Triplet<i32>,
    pub tiff_jpeg_quality: Triplet<i32>,
    pub jpeg_for_tiff_images: Triplet<bool>,
    pub slicer_pan_kb: Triplet<i32>,
    pub max_slicer_buf_mb: Triplet<i32>,
    pub speedup_slider: Triplet<bool>,
    pub max_linked_slicers: Triplet<i32>,
    pub load_ushorts: Triplet<bool>,
    pub load_int_if_mean_sd: Triplet<bool>,
    pub load_int_if_estimate: Triplet<bool>,
    pub prefer_mean_sd: Triplet<bool>,
    pub num_sds_for_scaling: Triplet<i32>,
    pub scale_scan_type: Triplet<i32>,
    pub change_mrc_stats: Triplet<bool>,
    pub use_ali_piece_coords: Triplet<i32>,
    pub exit_when_all_closed: Triplet<i32>,
    pub iso_high_thresh: Triplet<bool>,
    pub iso_box_initial: Triplet<i32>,
    pub iso_box_limit: Triplet<i32>,
    pub key_sets_hw_stereo: Triplet<bool>,
    pub no_vert_buf_for_cont: Triplet<bool>,
    pub no_vbo_for_sphere: Triplet<bool>,
}
impl Default for ImodPrefStruct {
    fn default() -> Self {
        let z = [0.; MAX_ZOOMS];
        Self {
            hot_slider_key: Default::default(),
            hot_slider_flag: Default::default(),
            mouse_mapping: Default::default(),
            modv_swap_left_mid: Default::default(),
            allow_ctrl_on_mac: Default::default(),
            silent_beep: Default::default(),
            classic_slicer: Default::default(),
            start_in_hq: Default::default(),
            arrows_scroll_zap: Default::default(),
            start_at_mid_z: Default::default(),
            auto_con_at_start: Default::default(),
            attach_to_on_obj: Default::default(),
            slicer_new_surf: Default::default(),
            font: String::new(),
            font_chgd: false,
            style_key: String::new(),
            style_chgd: false,
            bw_step: Default::default(),
            page_step: Default::default(),
            iconify_imodv_dlg: Default::default(),
            iconify_imod_dlg: Default::default(),
            iconify_image_win: Default::default(),
            stack_imod_dlgs: Default::default(),
            stack_imodv_dlgs: Default::default(),
            raise_imod_dlg_stack: Default::default(),
            raise_imodv_dlg_stack: Default::default(),
            keep_dlg_stack_on_top: Default::default(),
            dlg_frame_adjustment: Default::default(),
            eer_super_res: Default::default(),
            eer_zbinning: Default::default(),
            min_mod_pt_size: Default::default(),
            min_im_pt_size: Default::default(),
            boost_model_cursor: Default::default(),
            zooms: z,
            zooms_dflt: z,
            zooms_chgd: false,
            autosave_interval: Default::default(),
            autosave_on: Default::default(),
            autosave_no_cont_mesh: Default::default(),
            autosave_no_iso_mesh: Default::default(),
            autosave_dir: Default::default(),
            remember_geom: Default::default(),
            auto_target_mean: Default::default(),
            auto_target_sd: Default::default(),
            named_index: [0; MAX_NAMED_COLORS],
            named_color: [0; MAX_NAMED_COLORS],
            named_color_dflt: [0; MAX_NAMED_COLORS],
            named_color_chgd: [false; MAX_NAMED_COLORS],
            snap_format: Default::default(),
            snap_quality: Default::default(),
            snap_dpi: Default::default(),
            scale_snap_dpi: Default::default(),
            no_cur_pnt_on_snaps: Default::default(),
            tiff_compression: Default::default(),
            tiff_jpeg_quality: Default::default(),
            jpeg_for_tiff_images: Default::default(),
            slicer_pan_kb: Default::default(),
            max_slicer_buf_mb: Default::default(),
            speedup_slider: Default::default(),
            max_linked_slicers: Default::default(),
            load_ushorts: Default::default(),
            load_int_if_mean_sd: Default::default(),
            load_int_if_estimate: Default::default(),
            prefer_mean_sd: Default::default(),
            num_sds_for_scaling: Default::default(),
            scale_scan_type: Default::default(),
            change_mrc_stats: Default::default(),
            use_ali_piece_coords: Default::default(),
            exit_when_all_closed: Default::default(),
            iso_high_thresh: Default::default(),
            iso_box_initial: Default::default(),
            iso_box_limit: Default::default(),
            key_sets_hw_stereo: Default::default(),
            no_vert_buf_for_cont: Default::default(),
            no_vbo_for_sphere: Default::default(),
        }
    }
}

/// Rust-side representation of the QSettings key/value boundary.
#[derive(Clone, Debug, Default)]
pub struct PreferencesSettings {
    pub values: BTreeMap<String, String>,
}
impl PreferencesSettings {
    pub fn contains(&self, key: &str) -> bool {
        self.values.contains_key(key)
    }
    pub fn value(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(String::as_str)
    }
    pub fn set_value(&mut self, key: &str, value: impl ToString) {
        self.values.insert(key.into(), value.to_string());
    }
}
/// Qt/style/window/application operations reached from this source unit.
pub trait PreferencesNativeBoundary {
    fn style_exists(&self, _key: &str) -> bool {
        true
    }
    fn set_style(&mut self, _key: &str) {}
    fn set_font(&mut self, _font: &str) {}
    fn supported_image_formats(&self) -> Vec<String> {
        vec!["PNG".into(), "JPEG".into()]
    }
    fn start_autosave(&mut self) {}
    fn map_named_colors_and_draw(&mut self) {}
    fn setup_cursor(&mut self) {}
    fn toggle_model_mode(&mut self) {}
    fn restack_dialogs(&mut self) {}
    fn update_dialog(&mut self) {}
    fn update_movie(&mut self) {}
    fn set_current_panel(&mut self, _: i32) {}
    fn cancel_button_exists(&self) -> bool {
        false
    }
    fn defaults_button_exists(&self) -> bool {
        false
    }
    fn rounded_style(&self) -> bool {
        false
    }
    fn set_button_width(&mut self, _: &str, _: bool, _: f32, _: &str) -> i32 {
        0
    }
    fn set_button_fixed_width(&mut self, _: &str, _: i32) {}
    fn manage_list_stack_sizes(&mut self) {}
    fn accept_close_event(&mut self) {}
}

/// `ImodPreferences` (`preferences.h`).
#[derive(Clone, Debug)]
pub struct ImodPreferences {
    pub current_prefs: ImodPrefStruct,
    pub dialog_prefs: ImodPrefStruct,
    pub tab_dialog_open: bool,
    pub current_tab: i32,
    pub timer_id: i32,
    pub geom_image_xsize: [i32; MAX_GEOMETRIES],
    pub geom_image_ysize: [i32; MAX_GEOMETRIES],
    pub geom_info_win: [Rect; MAX_GEOMETRIES],
    pub geom_zap_win: [Rect; MAX_GEOMETRIES],
    pub geom_mod_view: [Rect; MAX_GEOMETRIES],
    pub new_obj_props: NewObjectProps,
    pub new_obj_props_dflt: NewObjectProps,
    pub new_obj_props_chgd: bool,
    pub recorded_zap_geom: Rect,
    pub recorded_mod_view_geom: Rect,
    pub geom_last_saved: i32,
    pub multi_zgeom: Rect,
    pub multi_znum_x: i32,
    pub multi_znum_y: i32,
    pub multi_zstep: i32,
    pub multi_zdraw_cen: i32,
    pub multi_zdraw_others: i32,
    pub xyz_apply_zscale: bool,
    pub ghost_mode: i32,
    pub ghost_dist: i32,
    pub classic_warned: bool,
    pub saved_snap_format: String,
    pub imod_dlgs_in_stack: String,
    pub imod_dlg_stack_states: String,
    pub imodv_dlgs_in_stack: String,
    pub imodv_dlg_stack_states: String,
    pub changing_frame_adj: bool,
    pub prev_browser_dir: String,
    pub use_prev_browser_dir: bool,
    pub generic_list: Vec<(String, Vec<f64>)>,
    pub style_status: [i32; MAX_STYLES],
}
impl ImodPreferences {
    /// `ImodPreferences::ImodPreferences`; QSettings reads are represented by `settings`.
    pub fn new(
        cmd_line_style: Option<&str>,
        settings: &PreferencesSettings,
        native: &mut dyn PreferencesNativeBoundary,
    ) -> Self {
        let mut p = ImodPrefStruct::default();
        macro_rules! n {
            ($field:ident,$key:literal,$d:expr) => {{
                p.$field = Triplet {
                    value: settings
                        .value($key)
                        .and_then(|x| x.parse().ok())
                        .unwrap_or($d),
                    dflt: $d,
                    chgd: settings.contains($key),
                };
            }};
        }
        macro_rules! b {
            ($field:ident,$key:literal,$d:expr) => {{
                p.$field = Triplet {
                    value: settings
                        .value($key)
                        .map(|x| x == "true" || x == "1")
                        .unwrap_or($d),
                    dflt: $d,
                    chgd: settings.contains($key),
                };
            }};
        }
        n!(hot_slider_key, "hotSliderKey", 0);
        n!(hot_slider_flag, "hotSliderFlag", HOT_SLIDER_KEYUP);
        n!(mouse_mapping, "mouseMapping", 0);
        b!(modv_swap_left_mid, "modvSwapLeftMid", false);
        b!(allow_ctrl_on_mac, "allowCtrlOnMac", false);
        b!(silent_beep, "silentBeep", false);
        b!(classic_slicer, "classicSlicer", false);
        b!(start_in_hq, "startInHQ", true);
        b!(arrows_scroll_zap, "arrowsScrollZap", false);
        b!(start_at_mid_z, "startAtMidZ", true);
        n!(auto_con_at_start, "autoConAtStart", 1);
        b!(attach_to_on_obj, "attachToOnObj", true);
        b!(slicer_new_surf, "slicerNewSurf", true);
        n!(bw_step, "bwStep", 3);
        n!(page_step, "pageStep", 10);
        b!(iconify_imodv_dlg, "iconifyImodvDlg", true);
        b!(iconify_imod_dlg, "iconifyImodDlg", true);
        b!(iconify_image_win, "iconifyImageWin", false);
        b!(stack_imod_dlgs, "stackImodDlgs", true);
        b!(stack_imodv_dlgs, "stackImodvDlgs", true);
        b!(raise_imod_dlg_stack, "raiseImodDlgStack", true);
        b!(raise_imodv_dlg_stack, "raiseImodvDlgStack", true);
        b!(keep_dlg_stack_on_top, "keepDlgStackOnTop", true);
        n!(dlg_frame_adjustment, "dlgFrameAdjustment", 0);
        n!(eer_super_res, "eerSuperRes", 1);
        n!(eer_zbinning, "eerZbinning", 10);
        n!(min_mod_pt_size, "minModPtSize", 4);
        n!(min_im_pt_size, "minImPtSize", 4);
        n!(boost_model_cursor, "boostModelCursor", 0);
        n!(autosave_interval, "autosaveInterval", 5);
        b!(autosave_on, "autosaveOn", true);
        b!(autosave_no_cont_mesh, "autosaveNoContMesh", false);
        b!(autosave_no_iso_mesh, "autosaveNoIsoMesh", false);
        p.autosave_dir = Triplet {
            value: settings.value("autosaveDir").unwrap_or("").into(),
            dflt: String::new(),
            chgd: settings.contains("autosaveDir"),
        };
        b!(remember_geom, "rememberGeom", true);
        n!(auto_target_mean, "autoTargetMean", 150);
        n!(auto_target_sd, "autoTargetSD", 40);
        p.named_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        p.named_color_dflt = [
            0xffff00, 0xb9b900, 0xff0000, 0x00ff00, 0xffff80, 0xffff80, 0x404060, 0x101010,
            0xff00a0, 0x00ff00, 0x00aaff, 0xffaa00,
        ];
        p.named_color = p.named_color_dflt;
        p.snap_format = Triplet {
            value: settings.value("snapFormat").unwrap_or("JPEG").into(),
            dflt: "JPEG".into(),
            chgd: settings.contains("snapFormat"),
        };
        n!(snap_quality, "snapQuality", 80);
        n!(snap_dpi, "snapDPI", 0);
        b!(scale_snap_dpi, "scaleSnapDPI", false);
        b!(no_cur_pnt_on_snaps, "noCurPntOnSnaps", false);
        n!(tiff_compression, "tiffCompression", 0);
        n!(tiff_jpeg_quality, "tiffJpegQuality", 80);
        b!(jpeg_for_tiff_images, "jpegForTiffImages", false);
        n!(slicer_pan_kb, "slicerPanKb", 6000);
        n!(max_slicer_buf_mb, "maxSlicerBufMB", 500);
        n!(max_linked_slicers, "maxLinkedSlicers", 20);
        b!(speedup_slider, "speedupSlider", false);
        b!(load_ushorts, "loadUshorts", false);
        b!(load_int_if_mean_sd, "loadIntIfMeanSD", true);
        b!(load_int_if_estimate, "loadIntIfEstimate", true);
        b!(prefer_mean_sd, "preferMeanSD", false);
        n!(num_sds_for_scaling, "numSDsForScaling", 6);
        n!(scale_scan_type, "scaleScanType", 0);
        b!(change_mrc_stats, "changeMRCstats", false);
        n!(use_ali_piece_coords, "useAliPieceCoords", 0);
        n!(exit_when_all_closed, "exitWhenAllClosed", -1);
        b!(iso_high_thresh, "isoHighThresh", false);
        n!(iso_box_initial, "isoBoxInitial", 130);
        n!(iso_box_limit, "isoBoxLimit", 500);
        b!(key_sets_hw_stereo, "keySetsHWstereo", true);
        b!(no_vert_buf_for_cont, "noVertBufForCont", false);
        b!(no_vbo_for_sphere, "noVBOForSphere", false);
        let zooms = [
            0.01, 0.02, 0.04, 0.07, 0.1, 0.1667, 0.25, 0.3333, 0.5, 0.75, 1., 1.5, 2., 3., 4., 5.,
            6., 8., 10., 12., 16., 20.,
        ];
        p.zooms = zooms;
        p.zooms_dflt = zooms;
        for i in 0..MAX_ZOOMS {
            if let Some(v) = settings.value(&format!("zooms/{i}")) {
                p.zooms[i] = v.parse().unwrap_or(zooms[i]);
                p.zooms_chgd = true;
            }
        }
        for i in 0..MAX_NAMED_COLORS {
            let key = format!("namedColors/{i}");
            if let Some(v) = settings.value(&key) {
                p.named_color[i] = v.parse().unwrap_or(p.named_color_dflt[i]);
                p.named_color_chgd[i] = true;
            }
        }
        let style = cmd_line_style
            .filter(|x| Self::style_ok(x, native))
            .or_else(|| {
                settings
                    .value("styleKey")
                    .filter(|x| Self::style_ok(x, native))
            })
            .unwrap_or("Windows");
        p.style_key = style.into();
        p.style_chgd = cmd_line_style.is_none() && settings.contains("styleKey");
        native.set_style(&p.style_key);
        let classic_warned = settings
            .value("classicWarned")
            .map(|x| x == "true" || x == "1")
            .unwrap_or(false)
            || p.classic_slicer.value;
        let current_tab = settings
            .value("currentPrefsTab")
            .and_then(|x| x.parse().ok())
            .unwrap_or(0)
            .clamp(0, 5);
        Self {
            current_prefs: p.clone(),
            dialog_prefs: p,
            tab_dialog_open: false,
            current_tab,
            timer_id: 0,
            geom_image_xsize: [0; MAX_GEOMETRIES],
            geom_image_ysize: [0; MAX_GEOMETRIES],
            geom_info_win: [Rect::default(); MAX_GEOMETRIES],
            geom_zap_win: [Rect::default(); MAX_GEOMETRIES],
            geom_mod_view: [Rect::default(); MAX_GEOMETRIES],
            new_obj_props: NewObjectProps::default(),
            new_obj_props_dflt: NewObjectProps {
                symbol: 0,
                symsize: 3,
                linewidth2: 1,
                ..Default::default()
            },
            new_obj_props_chgd: false,
            recorded_zap_geom: Rect::default(),
            recorded_mod_view_geom: Rect::default(),
            geom_last_saved: settings
                .value("lastGeometrySaved")
                .and_then(|x| x.parse().ok())
                .unwrap_or(-1),
            multi_zgeom: Rect::default(),
            multi_znum_x: 0,
            multi_znum_y: 0,
            multi_zstep: 0,
            multi_zdraw_cen: 0,
            multi_zdraw_others: 0,
            xyz_apply_zscale: false,
            ghost_mode: 0,
            ghost_dist: 0,
            classic_warned,
            saved_snap_format: String::new(),
            imod_dlgs_in_stack: settings.value("imodDlgsInStack").unwrap_or("").into(),
            imod_dlg_stack_states: settings.value("imodDlgStackStates").unwrap_or("").into(),
            imodv_dlgs_in_stack: settings.value("imodvDlgsInStack").unwrap_or("").into(),
            imodv_dlg_stack_states: settings.value("imodvDlgStackStates").unwrap_or("").into(),
            changing_frame_adj: false,
            prev_browser_dir: settings.value("browserDir").unwrap_or("").into(),
            use_prev_browser_dir: settings
                .value("usePrevBrowserDir")
                .map(|x| x == "true" || x == "1")
                .unwrap_or(false),
            generic_list: vec![],
            style_status: [0; MAX_STYLES],
        }
    }
    /// `ImodPreferences::styleOK`.
    pub fn style_ok(key: &str, native: &dyn PreferencesNativeBoundary) -> bool {
        !key.is_empty()
            && [
                "Windows",
                "Motif",
                "CDE",
                "Plastique",
                "Cleanlooks",
                "Fusion",
                "Macintosh (Aqua)",
                "Macintosh",
            ]
            .iter()
            .any(|x| x.eq_ignore_ascii_case(key))
            && native.style_exists(key)
    }
    /// `ImodPreferences::saveSettings`.
    pub fn save_settings(&self, settings: &mut PreferencesSettings, _modv_alone: i32) {
        macro_rules! write {
            ($f:ident,$k:literal) => {
                if self.current_prefs.$f.chgd {
                    settings.set_value($k, &self.current_prefs.$f.value);
                }
            };
        }
        write!(hot_slider_key, "hotSliderKey");
        write!(hot_slider_flag, "hotSliderFlag");
        write!(mouse_mapping, "mouseMapping");
        write!(modv_swap_left_mid, "modvSwapLeftMid");
        write!(autosave_interval, "autosaveInterval");
        write!(autosave_on, "autosaveOn");
        write!(autosave_dir, "autosaveDir");
        write!(snap_format, "snapFormat");
        write!(snap_quality, "snapQuality");
        write!(remember_geom, "rememberGeom");
        write!(exit_when_all_closed, "exitWhenAllClosed");
        settings.set_value("classicWarned", self.classic_warned);
        settings.set_value("currentPrefsTab", self.current_tab);
        settings.set_value("xyzApplyZscale", self.xyz_apply_zscale);
        settings.set_value("browserDir", &self.prev_browser_dir);
        settings.set_value("usePrevBrowserDir", self.use_prev_browser_dir);
        if self.current_prefs.zooms_chgd {
            for i in 0..MAX_ZOOMS {
                settings.set_value(&format!("zooms/{i}"), self.current_prefs.zooms[i]);
            }
        }
        for i in 0..MAX_NAMED_COLORS {
            if self.current_prefs.named_color_chgd[i] {
                settings.set_value(
                    &format!("namedColors/{i}"),
                    self.current_prefs.named_color[i],
                );
            }
        }
        for (key, values) in &self.generic_list {
            for (i, value) in values.iter().enumerate() {
                settings.set_value(&format!("{key}/{i}"), value);
            }
        }
    }
    /// `ImodPreferences::editPrefs`.
    pub fn edit_prefs(&mut self) {
        if !self.tab_dialog_open {
            self.dialog_prefs = self.current_prefs.clone();
            self.tab_dialog_open = true;
        }
    }
    /// `ImodPreferences::donePressed`.  Form unload is a caller-owned Qt boundary.
    pub fn done_pressed(&mut self, native: &mut dyn PreferencesNativeBoundary) {
        let old = self.current_prefs.clone();
        let new = self.dialog_prefs.clone();
        self.current_prefs = new.clone();
        macro_rules! changed {
            ($f:ident) => {
                self.current_prefs.$f.chgd |= new.$f.value != old.$f.value;
            };
        }
        changed!(hot_slider_key);
        changed!(hot_slider_flag);
        changed!(mouse_mapping);
        changed!(modv_swap_left_mid);
        changed!(allow_ctrl_on_mac);
        changed!(silent_beep);
        changed!(classic_slicer);
        changed!(start_in_hq);
        changed!(arrows_scroll_zap);
        changed!(start_at_mid_z);
        changed!(auto_con_at_start);
        changed!(attach_to_on_obj);
        changed!(slicer_new_surf);
        changed!(bw_step);
        changed!(page_step);
        changed!(iconify_imodv_dlg);
        changed!(iconify_imod_dlg);
        changed!(iconify_image_win);
        changed!(stack_imod_dlgs);
        changed!(stack_imodv_dlgs);
        changed!(raise_imod_dlg_stack);
        changed!(raise_imodv_dlg_stack);
        changed!(keep_dlg_stack_on_top);
        changed!(dlg_frame_adjustment);
        changed!(eer_super_res);
        changed!(eer_zbinning);
        changed!(min_mod_pt_size);
        changed!(min_im_pt_size);
        changed!(boost_model_cursor);
        changed!(remember_geom);
        changed!(auto_target_mean);
        changed!(auto_target_sd);
        changed!(snap_format);
        changed!(snap_quality);
        changed!(snap_dpi);
        changed!(scale_snap_dpi);
        changed!(no_cur_pnt_on_snaps);
        changed!(tiff_compression);
        changed!(tiff_jpeg_quality);
        changed!(jpeg_for_tiff_images);
        changed!(slicer_pan_kb);
        changed!(max_slicer_buf_mb);
        changed!(max_linked_slicers);
        changed!(speedup_slider);
        changed!(load_ushorts);
        changed!(load_int_if_mean_sd);
        changed!(load_int_if_estimate);
        changed!(prefer_mean_sd);
        changed!(num_sds_for_scaling);
        changed!(scale_scan_type);
        changed!(change_mrc_stats);
        changed!(use_ali_piece_coords);
        changed!(exit_when_all_closed);
        changed!(iso_high_thresh);
        changed!(iso_box_limit);
        changed!(iso_box_initial);
        changed!(key_sets_hw_stereo);
        changed!(no_vert_buf_for_cont);
        changed!(no_vbo_for_sphere);
        changed!(autosave_interval);
        changed!(autosave_on);
        changed!(autosave_dir);
        changed!(autosave_no_cont_mesh);
        changed!(autosave_no_iso_mesh);
        self.current_prefs.font_chgd |= new.font != old.font;
        self.current_prefs.style_chgd |= !new.style_key.eq_ignore_ascii_case(&old.style_key);
        self.current_prefs.zooms_chgd |= new.zooms != old.zooms;
        for i in 0..MAX_NAMED_COLORS {
            self.current_prefs.named_color_chgd[i] |= new.named_color[i] != old.named_color[i];
        }
        if self.current_prefs.boost_model_cursor.chgd {
            self.cursor_boost_changed(native);
        }
        self.tab_dialog_open = false;
        native.update_dialog();
        native.update_movie();
    }
    /// `ImodPreferences::cancelPressed`.
    pub fn cancel_pressed(&mut self) {
        self.user_canceled()
    }
    /// `ImodPreferences::userCanceled`.
    pub fn user_canceled(&mut self) {
        if self.tab_dialog_open && self.timer_id == 0 {
            self.tab_dialog_open = false;
            self.timer_id = 1;
        }
    }
    /// `ImodPreferences::timerEvent`.
    pub fn timer_event(&mut self, native: &mut dyn PreferencesNativeBoundary) {
        self.timer_id = 0;
        if self.dialog_prefs.style_key != self.current_prefs.style_key {
            self.change_style(&self.current_prefs.style_key.clone(), native);
        }
        if self.dialog_prefs.font != self.current_prefs.font {
            self.change_font(&self.current_prefs.font.clone(), native);
        }
        if self.dialog_prefs.dlg_frame_adjustment.value
            != self.current_prefs.dlg_frame_adjustment.value
        {
            self.dlg_frame_adj_changed(native);
        }
        self.point_size_changed(native);
    }
    /// `ImodPreferences::defaultPressed`; caller supplies current Qt panel index.
    pub fn default_pressed(&mut self, panel: i32, native: &mut dyn PreferencesNativeBoundary) {
        let p = &mut self.dialog_prefs;
        match panel {
            0 => {
                p.min_mod_pt_size.value = p.min_mod_pt_size.dflt;
                p.min_im_pt_size.value = p.min_im_pt_size.dflt;
                p.boost_model_cursor.value = p.boost_model_cursor.dflt;
                p.named_color = p.named_color_dflt;
                p.zooms = p.zooms_dflt;
                p.slicer_pan_kb.value = p.slicer_pan_kb.dflt;
                p.max_slicer_buf_mb.value = p.max_slicer_buf_mb.dflt;
                p.max_linked_slicers.value = p.max_linked_slicers.dflt;
                p.speedup_slider.value = p.speedup_slider.dflt;
                p.iso_high_thresh.value = p.iso_high_thresh.dflt;
                p.iso_box_limit.value = p.iso_box_limit.dflt;
                p.iso_box_initial.value = p.iso_box_initial.dflt;
                native.map_named_colors_and_draw();
            }
            1 => {
                p.load_ushorts.value = p.load_ushorts.dflt;
                p.load_int_if_mean_sd.value = p.load_int_if_mean_sd.dflt;
                p.load_int_if_estimate.value = p.load_int_if_estimate.dflt;
                p.prefer_mean_sd.value = p.prefer_mean_sd.dflt;
                p.num_sds_for_scaling.value = p.num_sds_for_scaling.dflt;
                p.scale_scan_type.value = p.scale_scan_type.dflt;
                p.change_mrc_stats.value = p.change_mrc_stats.dflt;
                p.use_ali_piece_coords.value = p.use_ali_piece_coords.dflt;
                p.auto_target_mean.value = p.auto_target_mean.dflt;
                p.auto_target_sd.value = p.auto_target_sd.dflt;
                p.auto_con_at_start.value = p.auto_con_at_start.dflt;
                p.eer_super_res.value = p.eer_super_res.dflt;
                p.eer_zbinning.value = p.eer_zbinning.dflt;
            }
            2 => {
                p.snap_format.value = p.snap_format.dflt.clone();
                p.snap_quality.value = p.snap_quality.dflt;
                p.snap_dpi.value = p.snap_dpi.dflt;
                p.scale_snap_dpi.value = p.scale_snap_dpi.dflt;
                p.no_cur_pnt_on_snaps.value = p.no_cur_pnt_on_snaps.dflt;
                p.tiff_compression.value = p.tiff_compression.dflt;
                p.tiff_jpeg_quality.value = p.tiff_jpeg_quality.dflt;
                p.jpeg_for_tiff_images.value = p.jpeg_for_tiff_images.dflt;
            }
            3 => {
                p.allow_ctrl_on_mac.value = p.allow_ctrl_on_mac.dflt;
                p.silent_beep.value = p.silent_beep.dflt;
                p.start_at_mid_z.value = p.start_at_mid_z.dflt;
                p.start_in_hq.value = p.start_in_hq.dflt;
                p.arrows_scroll_zap.value = p.arrows_scroll_zap.dflt;
                p.key_sets_hw_stereo.value = p.key_sets_hw_stereo.dflt;
                p.no_vert_buf_for_cont.value = p.no_vert_buf_for_cont.dflt;
                p.no_vbo_for_sphere.value = p.no_vbo_for_sphere.dflt;
                p.attach_to_on_obj.value = p.attach_to_on_obj.dflt;
                p.slicer_new_surf.value = p.slicer_new_surf.dflt;
                p.exit_when_all_closed.value = p.exit_when_all_closed.dflt;
                p.bw_step.value = p.bw_step.dflt;
                p.page_step.value = p.page_step.dflt;
                p.autosave_interval.value = p.autosave_interval.dflt;
                p.autosave_on.value = p.autosave_on.dflt;
                p.autosave_no_cont_mesh.value = p.autosave_no_cont_mesh.dflt;
                p.autosave_no_iso_mesh.value = p.autosave_no_iso_mesh.dflt;
                p.autosave_dir.value = p.autosave_dir.dflt.clone();
            }
            4 => {
                p.iconify_imodv_dlg.value = p.iconify_imodv_dlg.dflt;
                p.iconify_imod_dlg.value = p.iconify_imod_dlg.dflt;
                p.iconify_image_win.value = p.iconify_image_win.dflt;
                p.remember_geom.value = p.remember_geom.dflt;
                p.stack_imod_dlgs.value = p.stack_imod_dlgs.dflt;
                p.raise_imod_dlg_stack.value = p.raise_imod_dlg_stack.dflt;
                p.keep_dlg_stack_on_top.value = p.keep_dlg_stack_on_top.dflt;
                p.stack_imodv_dlgs.value = p.stack_imodv_dlgs.dflt;
                p.raise_imodv_dlg_stack.value = p.raise_imodv_dlg_stack.dflt;
                p.dlg_frame_adjustment.value = p.dlg_frame_adjustment.dflt;
            }
            5 => {
                p.hot_slider_key.value = p.hot_slider_key.dflt;
                p.hot_slider_flag.value = p.hot_slider_flag.dflt;
                p.mouse_mapping.value = p.mouse_mapping.dflt;
                p.modv_swap_left_mid.value = p.modv_swap_left_mid.dflt;
            }
            _ => {}
        }
    }
    /// `ImodPreferences::findCurrentTab`.
    pub fn find_current_tab(&mut self, current_index: i32) {
        self.current_tab = current_index;
    }
    /// `ImodPreferences::changeFont`.
    pub fn change_font(&mut self, font: &str, native: &mut dyn PreferencesNativeBoundary) {
        native.set_font(font);
    }
    /// `ImodPreferences::changeStyle`.
    pub fn change_style(&mut self, key: &str, native: &mut dyn PreferencesNativeBoundary) {
        native.set_style(key);
        self.change_font(
            if self.tab_dialog_open {
                &self.dialog_prefs.font
            } else {
                &self.current_prefs.font
            }
            .clone()
            .as_str(),
            native,
        );
    }
    /// `ImodPreferences::snapFormatList`.
    pub fn snap_format_list(&self, native: &dyn PreferencesNativeBoundary) -> Vec<String> {
        let mut out = Vec::new();
        for mut x in native.supported_image_formats() {
            x = x.to_ascii_uppercase();
            if x == "JPG" {
                x = "JPEG".into();
            }
            if !matches!(x.as_str(), "PBM" | "XBM" | "TIF" | "TIFF") && !out.contains(&x) {
                out.push(x);
            }
        }
        out.push("RGB".into());
        out
    }
    /// `ImodPreferences::hotSliderActive`.
    pub fn hot_slider_active(&self, ctrl_pressed: i32) -> bool {
        (self.current_prefs.hot_slider_flag.value == HOT_SLIDER_KEYDOWN && ctrl_pressed != 0)
            || (self.current_prefs.hot_slider_flag.value == HOT_SLIDER_KEYUP && ctrl_pressed == 0)
    }
    /// `ImodPreferences::snapFormat2`.
    pub fn snap_format2(
        &self,
        current: Option<&str>,
        native: &dyn PreferencesNativeBoundary,
    ) -> String {
        let second = if current.unwrap_or(&self.current_prefs.snap_format.value) == "PNG" {
            "JPEG"
        } else {
            "PNG"
        };
        if self.snap_format_list(native).iter().any(|x| x == second) {
            second.into()
        } else {
            String::new()
        }
    }
    /// `ImodPreferences::set2ndSnapFormat`.
    pub fn set_2nd_snap_format(&mut self, native: &dyn PreferencesNativeBoundary) {
        self.saved_snap_format = self.current_prefs.snap_format.value.clone();
        let f = self.snap_format2(None, native);
        if !f.is_empty() {
            self.current_prefs.snap_format.value = f;
        }
    }
    /// `ImodPreferences::restoreSnapFormat`.
    pub fn restore_snap_format(&mut self) {
        self.current_prefs.snap_format.value = self.saved_snap_format.clone();
    }
    /// `ImodPreferences::setSnapQuality`.
    pub fn set_snap_quality(&mut self, value: i32) {
        self.current_prefs.snap_quality.value = value;
        self.current_prefs.snap_quality.chgd = true;
    }
    /// `ImodPreferences::actualButton` (Qt left/middle/right codes are 1/2/3 boundary-neutral values).
    pub fn actual_button(&self, logical_button: i32) -> i32 {
        let mapping = self.current_prefs.mouse_mapping.value;
        (mapping + (1 - 2 * (mapping / 3)) * (logical_button - 1)).rem_euclid(3) + 1
    }
    /// `ImodPreferences::actualModvButton`.
    pub fn actual_modv_button(&self, logical_button: i32) -> i32 {
        if logical_button < 3 && self.current_prefs.modv_swap_left_mid.value {
            3 - logical_button
        } else {
            logical_button
        }
    }
    /// `ImodPreferences::autosaveDir`; supply the process environment value explicitly.
    pub fn autosave_dir(&self, environment: Option<&str>) -> String {
        if self.current_prefs.autosave_dir.chgd || environment.is_none() {
            self.current_prefs.autosave_dir.value.clone()
        } else {
            environment.unwrap().replace('\\', "/")
        }
    }
    /// `ImodPreferences::autosaveSec`; supply `IMOD_AUTOSAVE` explicitly.
    pub fn autosave_sec(&self, environment: Option<&str>) -> i32 {
        if self.current_prefs.autosave_on.chgd
            || self.current_prefs.autosave_interval.chgd
            || environment.is_none()
        {
            if self.current_prefs.autosave_on.value {
                60 * self.current_prefs.autosave_interval.value
            } else {
                0
            }
        } else {
            let n = environment.unwrap().parse::<i32>().unwrap_or(0);
            if n < 0 { -n } else { 60 * n }
        }
    }
    /// `ImodPreferences::minCurrentImPtSize`.
    pub fn min_current_im_pt_size(&self) -> i32 {
        (if self.tab_dialog_open {
            &self.dialog_prefs
        } else {
            &self.current_prefs
        })
        .min_im_pt_size
        .value
    }
    /// `ImodPreferences::minCurrentModPtSize`.
    pub fn min_current_mod_pt_size(&self) -> i32 {
        (if self.tab_dialog_open {
            &self.dialog_prefs
        } else {
            &self.current_prefs
        })
        .min_mod_pt_size
        .value
    }
    /// `ImodPreferences::pointSizeChanged`.
    pub fn point_size_changed(&self, native: &mut dyn PreferencesNativeBoundary) {
        native.map_named_colors_and_draw();
    }
    /// `ImodPreferences::cursorBoostChanged`.
    pub fn cursor_boost_changed(&self, native: &mut dyn PreferencesNativeBoundary) {
        native.setup_cursor();
        native.toggle_model_mode();
    }
    /// `ImodPreferences::dlgFrameAdjustment`.
    pub fn dlg_frame_adjustment(&self) -> i32 {
        (if self.tab_dialog_open {
            &self.dialog_prefs
        } else {
            &self.current_prefs
        })
        .dlg_frame_adjustment
        .value
    }
    /// `ImodPreferences::dlgFrameAdjChanged`.
    pub fn dlg_frame_adj_changed(&mut self, native: &mut dyn PreferencesNativeBoundary) {
        self.changing_frame_adj = true;
        native.restack_dialogs();
        self.changing_frame_adj = false;
    }
    /// `ImodPreferences::namedColor`.
    pub fn named_color(&self, index: i32) -> u32 {
        let p = if self.tab_dialog_open {
            &self.dialog_prefs
        } else {
            &self.current_prefs
        };
        for i in 0..MAX_NAMED_COLORS {
            if p.named_index[i] == index {
                return p.named_color[i];
            }
        }
        0
    }
    /// `ImodPreferences::getRoundedStyle`.
    pub fn get_rounded_style(&self) -> bool {
        let s = &if self.tab_dialog_open {
            &self.dialog_prefs
        } else {
            &self.current_prefs
        }
        .style_key;
        s.to_ascii_lowercase().contains("aqua") || s.to_ascii_lowercase().contains("macintosh")
    }
    /// `ImodPreferences::setDefaultObjProps`.
    pub fn set_default_obj_props(&mut self, props: Option<NewObjectProps>) {
        if let Some(x) = props {
            self.new_obj_props = x;
            self.new_obj_props_chgd = true;
        }
    }
    /// `ImodPreferences::restoreDefaultObjProps`.
    pub fn restore_default_obj_props(&mut self) {
        self.new_obj_props = self.new_obj_props_dflt;
        self.new_obj_props_chgd = true;
    }
    /// `ImodPreferences::setInfoGeometry`; Qt info-window positioning is an explicit boundary.
    pub fn set_info_geometry(&self, image_x: i32, image_y: i32) -> Option<Rect> {
        if !self.current_prefs.remember_geom.value {
            return None;
        }
        let i = self
            .geom_image_xsize
            .iter()
            .zip(self.geom_image_ysize.iter())
            .position(|(&x, &y)| x == image_x && y == image_y)
            .or_else(|| usize::try_from(self.geom_last_saved).ok());
        i.and_then(|x| {
            let r = self.geom_info_win[x];
            (r.width != 0 && r.height != 0).then_some(r)
        })
    }
    /// `ImodPreferences::getZapGeometry`.
    pub fn get_zap_geometry(&self, image_x: i32, image_y: i32) -> Rect {
        self.get_geometry_index(image_x, image_y)
            .map(|i| self.geom_zap_win[i])
            .unwrap_or_default()
    }
    /// `ImodPreferences::getModViewGeometry`.
    pub fn get_mod_view_geometry(&self, image_x: i32, image_y: i32) -> Rect {
        self.get_geometry_index(image_x, image_y)
            .map(|i| self.geom_mod_view[i])
            .unwrap_or_default()
    }
    /// `ImodPreferences::getGeometryIndex`.
    pub fn get_geometry_index(&self, image_x: i32, image_y: i32) -> Option<usize> {
        if self.current_prefs.remember_geom.value {
            self.geom_image_xsize
                .iter()
                .zip(self.geom_image_ysize.iter())
                .position(|(&x, &y)| x == image_x && y == image_y)
        } else {
            None
        }
    }
    /// `ImodPreferences::recordZapGeometry`.
    pub fn record_zap_geometry(&mut self, geometry: Rect) {
        self.recorded_zap_geom = geometry;
    }
    /// `ImodPreferences::recordModViewGeometry` inline header method.
    pub fn record_mod_view_geometry(&mut self, geometry: Rect) {
        self.recorded_mod_view_geom = geometry;
    }
    /// `ImodPreferences::recordMultiZparams`.
    pub fn record_multi_zparams(
        &mut self,
        geom: Rect,
        numx: i32,
        numy: i32,
        zstep: i32,
        draw_cen: i32,
        draw_other: i32,
    ) {
        self.multi_zgeom = geom;
        self.multi_znum_x = numx;
        self.multi_znum_y = numy;
        self.multi_zstep = zstep;
        self.multi_zdraw_cen = draw_cen;
        self.multi_zdraw_others = draw_other;
    }
    /// `ImodPreferences::getMultiZparams`.
    pub fn get_multi_zparams(&self) -> Option<(Rect, i32, i32, i32, i32, i32)> {
        (self.multi_zgeom.width != 0).then_some((
            self.multi_zgeom,
            self.multi_znum_x,
            self.multi_znum_y,
            self.multi_zstep,
            self.multi_zdraw_cen,
            self.multi_zdraw_others,
        ))
    }
    /// `ImodPreferences::getAutoContrastTargets`.
    pub fn get_auto_contrast_targets(&self) -> (i32, i32) {
        (
            self.current_prefs.auto_target_mean.value,
            self.current_prefs.auto_target_sd.value,
        )
    }
    /// `ImodPreferences::saveGenericSettings`.
    pub fn save_generic_settings(&mut self, key: &str, values: &[f64]) -> i32 {
        if let Some((_, old)) = self.generic_list.iter_mut().find(|(k, _)| k == key) {
            *old = values.into();
            return 0;
        }
        self.generic_list.push((key.into(), values.into()));
        0
    }
    /// `ImodPreferences::getGenericSettings`.
    pub fn get_generic_settings(
        &self,
        key: &str,
        settings: &PreferencesSettings,
        values: &mut [f64],
    ) -> i32 {
        let mut i = 0;
        while i < values.len() {
            let Some(v) = settings.value(&format!("{key}/{i}")) else {
                break;
            };
            values[i] = v.parse().unwrap_or(0.);
            i += 1;
        }
        i as i32
    }
    /// `ImodPreferences::classicWarned`.
    pub fn classic_warned(&mut self) -> bool {
        let old = self.classic_warned;
        self.classic_warned = true;
        old
    }
    /// `ImodPreferences::getStyleList`.
    pub fn get_style_list(&self) -> [&'static str; 8] {
        [
            "Windows",
            "Motif",
            "CDE",
            "Plastique",
            "Cleanlooks",
            "Macintosh (Aqua)",
            "Fusion",
            "",
        ]
    }
    /// `ImodPreferences::getStyleStatus`.
    pub fn get_style_status(&mut self) -> &mut [i32; MAX_STYLES] {
        &mut self.style_status
    }
    /// Header `RETURN_PREF` accessors.
    pub fn get_zooms(&self) -> &[f64; MAX_ZOOMS] {
        &self.current_prefs.zooms
    }
    pub fn get_bw_step(&self) -> i32 {
        self.current_prefs.bw_step.value
    }
    pub fn get_page_step(&self) -> i32 {
        self.current_prefs.page_step.value
    }
    pub fn iconify_imodv_dlg(&self) -> bool {
        self.current_prefs.iconify_imodv_dlg.value
    }
    pub fn iconify_imod_dlg(&self) -> bool {
        self.current_prefs.iconify_imod_dlg.value
    }
    pub fn iconify_image_win(&self) -> bool {
        self.current_prefs.iconify_image_win.value
    }
    pub fn stack_imod_dlgs(&self) -> bool {
        self.current_prefs.stack_imod_dlgs.value
    }
    pub fn stack_imodv_dlgs(&self) -> bool {
        self.current_prefs.stack_imodv_dlgs.value
    }
    pub fn raise_imod_dlg_stack(&self) -> bool {
        self.current_prefs.raise_imod_dlg_stack.value
    }
    pub fn raise_imodv_dlg_stack(&self) -> bool {
        self.current_prefs.raise_imodv_dlg_stack.value
    }
    pub fn keep_dlg_stack_on_top(&self) -> bool {
        self.current_prefs.keep_dlg_stack_on_top.value
    }
    pub fn eer_super_res(&self) -> i32 {
        self.current_prefs.eer_super_res.value
    }
    pub fn eer_zbinning(&self) -> i32 {
        self.current_prefs.eer_zbinning.value
    }
    pub fn hot_slider_key_value(&self) -> i32 {
        self.current_prefs.hot_slider_key.value
    }
    pub fn hot_slider_flag_value(&self) -> i32 {
        self.current_prefs.hot_slider_flag.value
    }
    pub fn auto_con_at_start(&self) -> i32 {
        self.current_prefs.auto_con_at_start.value
    }
    pub fn boost_model_cursor(&self) -> i32 {
        self.current_prefs.boost_model_cursor.value
    }
    pub fn start_at_mid_z(&self) -> bool {
        self.current_prefs.start_at_mid_z.value
    }
    pub fn autosave_no_cont_mesh(&self) -> bool {
        self.current_prefs.autosave_no_cont_mesh.value
    }
    pub fn autosave_no_iso_mesh(&self) -> bool {
        self.current_prefs.autosave_no_iso_mesh.value
    }
    pub fn get_allow_ctrl_on_mac(&self, environment_set: bool) -> bool {
        environment_set || self.current_prefs.allow_ctrl_on_mac.value
    }
    pub fn silent_beep(&self) -> bool {
        self.current_prefs.silent_beep.value
    }
    pub fn classic_slicer(&self) -> bool {
        self.current_prefs.classic_slicer.value
    }
    pub fn start_in_hq(&self) -> bool {
        self.current_prefs.start_in_hq.value
    }
    pub fn arrows_scroll_zap(&self) -> bool {
        self.current_prefs.arrows_scroll_zap.value
    }
    pub fn attach_to_on_obj(&self) -> bool {
        self.current_prefs.attach_to_on_obj.value
    }
    pub fn slicer_new_surf(&self) -> bool {
        self.current_prefs.slicer_new_surf.value
    }
    pub fn new_object_props(&self) -> NewObjectProps {
        self.new_obj_props
    }
    pub fn get_dialog_prefs(&mut self) -> &mut ImodPrefStruct {
        &mut self.dialog_prefs
    }
    pub fn set_exit_when_all_closed(&mut self, value: i32) {
        self.current_prefs.exit_when_all_closed.value = value;
        self.current_prefs.exit_when_all_closed.chgd = true;
    }
    pub fn snap_format(&self) -> &str {
        &self.current_prefs.snap_format.value
    }
    pub fn snap_quality(&self) -> i32 {
        self.current_prefs.snap_quality.value
    }
    pub fn snap_dpi(&self) -> i32 {
        self.current_prefs.snap_dpi.value
    }
    pub fn scale_snap_dpi(&self) -> bool {
        self.current_prefs.scale_snap_dpi.value
    }
    pub fn no_cur_pnt_on_snaps(&self) -> bool {
        self.current_prefs.no_cur_pnt_on_snaps.value
    }
    pub fn tiff_compression(&self) -> i32 {
        self.current_prefs.tiff_compression.value
    }
    pub fn tiff_jpeg_quality(&self) -> i32 {
        self.current_prefs.tiff_jpeg_quality.value
    }
    pub fn jpeg_for_tiff_images(&self) -> bool {
        self.current_prefs.jpeg_for_tiff_images.value
    }
    pub fn slicer_pan_kb(&self) -> i32 {
        self.current_prefs.slicer_pan_kb.value
    }
    pub fn max_slicer_buf_mb(&self) -> i32 {
        self.current_prefs.max_slicer_buf_mb.value
    }
    pub fn max_linked_slicers(&self) -> i32 {
        self.current_prefs.max_linked_slicers.value
    }
    pub fn speedup_slider(&self) -> bool {
        self.current_prefs.speedup_slider.value
    }
    pub fn load_ushorts(&self) -> bool {
        self.current_prefs.load_ushorts.value
    }
    pub fn load_int_if_mean_sd(&self) -> bool {
        self.current_prefs.load_int_if_mean_sd.value
    }
    pub fn load_int_if_estimate(&self) -> bool {
        self.current_prefs.load_int_if_estimate.value
    }
    pub fn prefer_mean_sd(&self) -> bool {
        self.current_prefs.prefer_mean_sd.value
    }
    pub fn num_sds_for_scaling(&self) -> i32 {
        self.current_prefs.num_sds_for_scaling.value
    }
    pub fn scale_scan_type(&self) -> i32 {
        self.current_prefs.scale_scan_type.value
    }
    pub fn change_mrc_stats(&self) -> bool {
        self.current_prefs.change_mrc_stats.value
    }
    pub fn use_ali_piece_coords(&self) -> i32 {
        self.current_prefs.use_ali_piece_coords.value
    }
    pub fn exit_when_all_closed(&self) -> i32 {
        self.current_prefs.exit_when_all_closed.value
    }
    pub fn iso_high_thresh(&self) -> bool {
        self.current_prefs.iso_high_thresh.value
    }
    pub fn iso_box_initial(&self) -> i32 {
        self.current_prefs.iso_box_initial.value
    }
    pub fn iso_box_limit(&self) -> i32 {
        self.current_prefs.iso_box_limit.value
    }
    pub fn key_sets_hw_stereo(&self) -> bool {
        self.current_prefs.key_sets_hw_stereo.value
    }
    pub fn no_vert_buf_for_cont(&self) -> bool {
        self.current_prefs.no_vert_buf_for_cont.value
    }
    pub fn no_vbo_for_sphere(&self) -> bool {
        self.current_prefs.no_vbo_for_sphere.value
    }
    pub fn xyz_apply_zscale(&self) -> bool {
        self.xyz_apply_zscale
    }
    pub fn set_xyz_apply_zscale(&mut self, value: bool) {
        self.xyz_apply_zscale = value;
    }
    pub fn ghost_mode(&self) -> i32 {
        self.ghost_mode
    }
    pub fn ghost_dist(&self) -> i32 {
        self.ghost_dist
    }
    pub fn imod_dlgs_in_stack(&self) -> &str {
        &self.imod_dlgs_in_stack
    }
    pub fn set_imod_dlgs_in_stack(&mut self, value: String) {
        self.imod_dlgs_in_stack = value;
    }
    pub fn imod_dlg_stack_states(&self) -> &str {
        &self.imod_dlg_stack_states
    }
    pub fn set_imod_dlg_stack_states(&mut self, value: String) {
        self.imod_dlg_stack_states = value;
    }
    pub fn imodv_dlgs_in_stack(&self) -> &str {
        &self.imodv_dlgs_in_stack
    }
    pub fn set_imodv_dlgs_in_stack(&mut self, value: String) {
        self.imodv_dlgs_in_stack = value;
    }
    pub fn imodv_dlg_stack_states(&self) -> &str {
        &self.imodv_dlg_stack_states
    }
    pub fn set_imodv_dlg_stack_states(&mut self, value: String) {
        self.imodv_dlg_stack_states = value;
    }
    pub fn changing_frame_adj(&self) -> bool {
        self.changing_frame_adj
    }
    pub fn prev_browser_dir(&self) -> &str {
        &self.prev_browser_dir
    }
    pub fn set_prev_browser_dir(&mut self, value: String) {
        self.prev_browser_dir = value;
    }
    pub fn use_prev_browser_dir(&self) -> bool {
        self.use_prev_browser_dir
    }
    pub fn set_use_prev_browser_dir(&mut self, value: bool) {
        self.use_prev_browser_dir = value;
    }
}

/// `hotSliderFlag`; upstream uses global `ImodPrefs`, Rust keeps ownership explicit.
pub fn hot_slider_flag(preferences: &ImodPreferences) -> i32 {
    preferences.current_prefs.hot_slider_flag.value
}
/// `hotSliderKey`; Qt Control/Shift/Alt numeric key bindings are native boundary constants.
pub fn hot_slider_key(preferences: &ImodPreferences) -> i32 {
    [0, 1, 2][preferences.current_prefs.hot_slider_key.value.clamp(0, 2) as usize]
}

/// Source-owned state of `PrefsDialog`; actual Qt layout and signals are native boundaries.
#[derive(Clone, Debug, Default)]
pub struct PrefsDialog {
    pub selected_panel: i32,
    pub visible: bool,
}
impl PrefsDialog {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn panel_selected(&mut self, which: i32, native: &mut dyn PreferencesNativeBoundary) {
        self.selected_panel = which;
        native.set_current_panel(which);
    }
    pub fn set_font_dependent_widths(&mut self, native: &mut dyn PreferencesNativeBoundary) {
        if !native.cancel_button_exists() || !native.defaults_button_exists() {
            return;
        }
        let rounded = native.rounded_style();
        let width = native.set_button_width("mCancelBut", rounded, 1.8, "Cancel");
        native.set_button_fixed_width("mDoneBut", width);
        native.set_button_width("mDefaultsBut", rounded, 1.2, "Defaults for Panel");
    }
    pub fn close_event(
        &mut self,
        prefs: &mut ImodPreferences,
        native: &mut dyn PreferencesNativeBoundary,
    ) {
        prefs.user_canceled();
        native.accept_close_event();
    }
    pub fn change_event(&mut self, font_change: bool, native: &mut dyn PreferencesNativeBoundary) {
        if font_change {
            native.manage_list_stack_sizes();
            self.set_font_dependent_widths(native);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct N {
        draws: i32,
        styles: Vec<String>,
        dialog: Vec<String>,
    }
    impl PreferencesNativeBoundary for N {
        fn set_style(&mut self, x: &str) {
            self.styles.push(x.into())
        }
        fn map_named_colors_and_draw(&mut self) {
            self.draws += 1
        }
        fn set_current_panel(&mut self, which: i32) {
            self.dialog.push(format!("panel:{which}"));
        }
        fn cancel_button_exists(&self) -> bool {
            true
        }
        fn defaults_button_exists(&self) -> bool {
            true
        }
        fn rounded_style(&self) -> bool {
            true
        }
        fn set_button_width(&mut self, button: &str, _: bool, factor: f32, text: &str) -> i32 {
            self.dialog.push(format!("width:{button}:{factor}:{text}"));
            text.len() as i32
        }
        fn set_button_fixed_width(&mut self, button: &str, width: i32) {
            self.dialog.push(format!("fixed:{button}:{width}"));
        }
        fn manage_list_stack_sizes(&mut self) {
            self.dialog.push("manage".into());
        }
        fn accept_close_event(&mut self) {
            self.dialog.push("accept".into());
        }
    }
    #[test]
    fn upstream_mouse_mapping_and_autosave_policy() {
        let mut n = N::default();
        let mut p = ImodPreferences::new(None, &PreferencesSettings::default(), &mut n);
        p.current_prefs.mouse_mapping.value = 4;
        assert_eq!(p.actual_button(1), 2);
        assert_eq!(p.actual_button(2), 1);
        assert_eq!(p.actual_button(3), 3);
        assert_eq!(p.autosave_sec(Some("-12")), 12);
        p.current_prefs.autosave_on.chgd = true;
        p.current_prefs.autosave_on.value = false;
        assert_eq!(p.autosave_sec(Some("-12")), 0)
    }
    #[test]
    fn preferences_dialog_routes_source_panel_font_and_close_events() {
        let mut native = N::default();
        let mut prefs = ImodPreferences::new(None, &PreferencesSettings::default(), &mut native);
        let mut dialog = PrefsDialog::new();
        dialog.panel_selected(3, &mut native);
        dialog.change_event(false, &mut native);
        dialog.change_event(true, &mut native);
        dialog.close_event(&mut prefs, &mut native);
        assert_eq!(
            native.dialog,
            [
                "panel:3",
                "manage",
                "width:mCancelBut:1.8:Cancel",
                "fixed:mDoneBut:6",
                "width:mDefaultsBut:1.2:Defaults for Panel",
                "accept",
            ]
        );
    }
    #[test]
    fn snapshot_and_generic_settings_follow_source() {
        let mut n = N::default();
        let mut p = ImodPreferences::new(None, &PreferencesSettings::default(), &mut n);
        assert_eq!(p.snap_format2(Some("PNG"), &n), "JPEG");
        p.set_2nd_snap_format(&n);
        assert_eq!(p.snap_format(), "PNG");
        p.restore_snap_format();
        assert_eq!(p.snap_format(), "JPEG");
        assert_eq!(p.save_generic_settings("zap", &[1., 2.]), 0);
        let mut s = PreferencesSettings::default();
        p.save_settings(&mut s, 0);
        let mut v = [0.; 3];
        assert_eq!(p.get_generic_settings("zap", &s, &mut v), 2);
        assert_eq!(&v[..2], &[1., 2.]);
    }
}
