#![allow(dead_code)]

pub const HUGE_CACHE: i32 = 2_000_000_000;

/// The `ImodView` fields read by `StartupForm::setValues`.
///
/// This is deliberately the source method's input projection: `ImodView` has
/// not yet acquired ownership-safe equivalents of every C `ViewInfo` member.
#[derive(Clone, Debug, Default)]
pub struct StartupViewValues {
    pub xmin: i32,
    pub xmax: i32,
    pub ymin: i32,
    pub ymax: i32,
    pub zmin: i32,
    pub zmax: i32,
    pub smin: f32,
    pub smax: f32,
    pub axis: i32,
    pub xybin: i32,
    pub zbin: i32,
    pub scale_scan_type: i32,
    pub store_scan_in_mrc: i32,
    pub vm_entered_as_gb: bool,
    pub vm_size: i32,
    pub strip_or_tile_cache: i32,
    pub image_pyramid: i32,
    pub gray_rgbs: i32,
    pub multi_file_z: i32,
    pub raw_image_store: i32,
    pub eer_super_res: i32,
    pub eer_zbinning: i32,
}

/// Native Qt, `imodplug`, preference, and application-global boundary.
pub trait StartupNativeBoundary {
    fn setup_ui(&mut self) {}
    fn set_modal(&mut self, _: bool) {}
    fn retranslate_ui(&mut self) {}
    fn use_prev_browser_dir(&self) -> bool;
    fn set_use_prev_browser_dir(&mut self, state: bool);
    fn browser_dir(&self) -> String;
    fn set_browser_dir(&mut self, directory: &str);
    fn prev_browser_dir(&self) -> String;
    fn current_dir(&self) -> String;
    fn open_names(&mut self, title: &str, directory: &str, filter: &str) -> Vec<String>;
    fn open_file(&mut self, title: &str, filter: &str) -> String;
    /// `QDir(path, pattern)` enumeration in `addImageFiles`.
    fn matching_files(&self, path: &str, pattern: &str) -> Vec<String>;
    fn manage_browser_dir(&mut self, name: &str, from_command_line: bool);
    fn is_model_file(&self, name: &str) -> i32;
    fn warning(&mut self, message: &str);
    fn show_help_page(&mut self, page: &str);
}

/// Source widget values.  Keeping these named makes the UI-file mapping
/// auditable without inventing an alternate dialog model.
#[derive(Clone, Debug)]
pub struct StartupUi {
    pub image_files_edit: String,
    pub model_file_edit: String,
    pub piece_file_edit: String,
    pub angle_file_edit: String,
    pub open_zap_box: bool,
    pub open_xyz_box: bool,
    pub open_slicer_box: bool,
    pub open_modv_box: bool,
    pub flip_check_box: bool,
    pub fill_cache_box: bool,
    pub show_rgb_gray_box: bool,
    pub load_frames_box: bool,
    pub load_sep_times_box: bool,
    pub load_unscaled_box: bool,
    pub change_mrcstats_box: bool,
    pub load_no_mirror_box: bool,
    pub load_ushort_box: bool,
    pub image_pyramid_box: bool,
    pub tile_cache_box: bool,
    pub show_montage_box: bool,
    pub prev_browser_dir_box: bool,
    pub cache_size_edit: String,
    pub x_from_edit: String,
    pub x_to_edit: String,
    pub y_from_edit: String,
    pub y_to_edit: String,
    pub z_from_edit: String,
    pub z_to_edit: String,
    pub scale_from_edit: String,
    pub scale_to_edit: String,
    pub bin_xy_spin_box: i32,
    pub bin_z_spin_box: i32,
    pub super_res_spin_box: i32,
    pub eer_zbin_spin_box: i32,
    pub x_montage_spin_box: i32,
    pub y_montage_spin_box: i32,
    pub x_overlap_spin_box: i32,
    pub y_overlap_spin_box: i32,
    pub x_size_spin_box: i32,
    pub y_size_spin_box: i32,
    pub image_files_label: String,
    pub image_fields_enabled: bool,
    pub model_fields_enabled: bool,
    pub piece_fields_enabled: bool,
    pub angle_fields_enabled: bool,
    pub montage_fields_enabled: bool,
    pub scanning_fields_enabled: bool,
    pub modv_size_fields_enabled: bool,
    pub buttons_enabled: bool,
}

impl Default for StartupUi {
    fn default() -> Self {
        Self {
            open_zap_box: true,
            bin_xy_spin_box: 1,
            bin_z_spin_box: 1,
            super_res_spin_box: 1,
            eer_zbin_spin_box: 10,
            image_files_label: "Image file(s):".into(),
            image_fields_enabled: true,
            model_fields_enabled: true,
            piece_fields_enabled: true,
            angle_fields_enabled: true,
            montage_fields_enabled: false,
            scanning_fields_enabled: true,
            buttons_enabled: true,
            ..Self::empty()
        }
    }
}

impl StartupUi {
    const fn empty() -> Self {
        Self {
            image_files_edit: String::new(),
            model_file_edit: String::new(),
            piece_file_edit: String::new(),
            angle_file_edit: String::new(),
            open_zap_box: false,
            open_xyz_box: false,
            open_slicer_box: false,
            open_modv_box: false,
            flip_check_box: false,
            fill_cache_box: false,
            show_rgb_gray_box: false,
            load_frames_box: false,
            load_sep_times_box: false,
            load_unscaled_box: false,
            change_mrcstats_box: false,
            load_no_mirror_box: false,
            load_ushort_box: false,
            image_pyramid_box: false,
            tile_cache_box: false,
            show_montage_box: false,
            prev_browser_dir_box: false,
            cache_size_edit: String::new(),
            x_from_edit: String::new(),
            x_to_edit: String::new(),
            y_from_edit: String::new(),
            y_to_edit: String::new(),
            z_from_edit: String::new(),
            z_to_edit: String::new(),
            scale_from_edit: String::new(),
            scale_to_edit: String::new(),
            bin_xy_spin_box: 0,
            bin_z_spin_box: 0,
            super_res_spin_box: 0,
            eer_zbin_spin_box: 0,
            x_montage_spin_box: 0,
            y_montage_spin_box: 0,
            x_overlap_spin_box: 0,
            y_overlap_spin_box: 0,
            x_size_spin_box: 0,
            y_size_spin_box: 0,
            image_files_label: String::new(),
            image_fields_enabled: false,
            model_fields_enabled: false,
            piece_fields_enabled: false,
            angle_fields_enabled: false,
            montage_fields_enabled: false,
            scanning_fields_enabled: false,
            modv_size_fields_enabled: false,
            buttons_enabled: false,
        }
    }
}

/// `StartupForm` (`form_startup.h`).
#[derive(Clone, Debug, Default)]
pub struct StartupForm {
    pub modal: bool,
    pub always_show_tool_tips: bool,
    pub m_show_montage: bool,
    pub m_cache_option: i32,
    pub m_modv_size_option: i32,
    pub m_scan_type: i32,
    pub m_piece_coord_type: i32,
    pub m_modv_mode: bool,
    pub m_image_files: String,
    pub m_piece_files: String,
    pub m_image_file_list: Vec<String>,
    pub m_piece_file_list: Vec<String>,
    pub m_model_file: String,
    pub m_joined_with_space: bool,
    pub m_piece_joinedw_space: bool,
    pub m_files_changed: bool,
    pub m_pieces_changed: bool,
    pub m_browser_set_from_prev: bool,
    pub m_argv: Vec<String>,
    pub m_argc: i32,
    pub m_str: String,
    pub m_angle_file: String,
    pub ui: StartupUi,
}

impl StartupForm {
    /// `StartupForm()` source constructor.
    pub fn new(modal: bool, native: &mut dyn StartupNativeBoundary) -> Self {
        native.setup_ui();
        native.set_modal(modal);
        let mut form = Self {
            modal,
            ..Self::default()
        };
        form.init(native);
        form
    }
    /// `StartupForm::~StartupForm`.
    pub fn destroy(&mut self) {}
    pub fn language_change(&mut self, native: &mut dyn StartupNativeBoundary) {
        native.retranslate_ui()
    }
    /// `StartupForm::init`.
    pub fn init(&mut self, native: &mut dyn StartupNativeBoundary) {
        self.always_show_tool_tips = true;
        self.m_modv_mode = false;
        self.m_show_montage = false;
        self.m_browser_set_from_prev = false;
        self.m_cache_option = 0;
        self.m_scan_type = 0;
        self.m_piece_coord_type = 0;
        self.m_modv_size_option = 0;
        self.ui.open_zap_box = true;
        self.ui.bin_xy_spin_box = 1;
        self.ui.bin_z_spin_box = 1;
        self.ui.prev_browser_dir_box = native.use_prev_browser_dir();
        self.manage_for_mod_view();
        self.m_files_changed = false;
        self.m_pieces_changed = false;
        self.m_argv.clear();
        self.m_argc = 0;
        self.m_joined_with_space = false;
        self.m_piece_joinedw_space = false;
    }
    /// `StartupForm::manageForModView`.
    pub fn manage_for_mod_view(&mut self) {
        self.ui.image_files_label = if self.m_modv_mode {
            "Model file(s)"
        } else {
            "Image file(s):"
        }
        .into();
        // The source leaves image-file editing active in both modes; in model
        // mode its label changes to "Model file(s)".
        self.ui.image_fields_enabled = true;
        self.ui.model_fields_enabled = !self.m_modv_mode;
        self.ui.piece_fields_enabled = !self.m_modv_mode;
        self.ui.angle_fields_enabled = !self.m_modv_mode;
        self.ui.scanning_fields_enabled = !self.m_modv_mode;
        self.ui.modv_size_fields_enabled = self.m_modv_mode;
        self.manage_montage();
        self.manage_modv_size();
    }
    /// `StartupForm::manageMontage`.
    pub fn manage_montage(&mut self) {
        self.ui.montage_fields_enabled = !self.m_modv_mode && self.m_show_montage
    }
    /// `StartupForm::manageModvSize`.
    pub fn manage_modv_size(&mut self) {
        self.ui.modv_size_fields_enabled = self.m_modv_mode && self.m_modv_size_option == 2
    }
    /// `StartupForm::modvSizeClicked`.
    pub fn modv_size_clicked(&mut self, id: i32) {
        self.m_modv_size_option = id;
        self.manage_modv_size()
    }
    /// `StartupForm::cacheTypeClicked`.
    pub fn cache_type_clicked(&mut self, id: i32) {
        self.m_cache_option = id
    }
    /// `StartupForm::scanTypeClicked`.
    pub fn scan_type_clicked(&mut self, value: i32) {
        self.m_scan_type = value
    }
    /// `StartupForm::pieceCoordTypeClicked`.
    pub fn piece_coord_type_clicked(&mut self, value: i32) {
        self.m_piece_coord_type = value
    }
    /// `StartupForm::showMontageToggled`.
    pub fn show_montage_toggled(&mut self, state: bool) {
        self.m_show_montage = state;
        self.ui.show_montage_box = state;
        self.manage_montage()
    }
    /// `StartupForm::useBrowserDirToggled`.
    pub fn use_browser_dir_toggled(&mut self, state: bool, native: &mut dyn StartupNativeBoundary) {
        if !state && self.m_browser_set_from_prev {
            native.set_browser_dir("");
            self.m_browser_set_from_prev = false
        }
        self.ui.prev_browser_dir_box = state;
        native.set_use_prev_browser_dir(state)
    }
    /// `StartupForm::startAsClicked`.
    pub fn start_as_clicked(&mut self, id: i32) {
        self.m_modv_mode = id != 0;
        self.manage_for_mod_view()
    }
    /// `StartupForm::imageChanged`.
    pub fn image_changed(&mut self, images: &str) {
        self.m_files_changed = true;
        self.m_image_files = images.into();
        self.ui.image_files_edit = images.into()
    }
    /// `StartupForm::modelChanged`.
    pub fn model_changed(&mut self, model: &str) {
        self.m_model_file = model.into();
        self.ui.model_file_edit = model.into()
    }
    /// `StartupForm::pfileChanged`.
    pub fn pfile_changed(&mut self, pfile: &str) {
        self.m_pieces_changed = true;
        self.m_piece_files = pfile.into();
        self.ui.piece_file_edit = pfile.into()
    }
    /// `StartupForm::angleFileChanged`.
    pub fn angle_file_changed(&mut self, afile: &str) {
        self.m_angle_file = afile.into();
        self.ui.angle_file_edit = afile.into()
    }
    /// `StartupForm::imageSelectClicked`.
    pub fn image_select_clicked(&mut self, native: &mut dyn StartupNativeBoundary) {
        if self.ui.prev_browser_dir_box && native.browser_dir().is_empty() {
            let p = native.prev_browser_dir();
            native.set_browser_dir(&p);
            self.m_browser_set_from_prev = true;
        }
        self.enable_buttons(false);
        let title = if self.m_modv_mode {
            "Select model file(s) to load"
        } else {
            "Select image file(s) to load"
        };
        let filter = if self.m_modv_mode {
            "Model files (*.*mod)"
        } else {
            "MRC files (*.*st *.*ali *.*rec* *.*mrc *.*join);;TIFF files (*.tif);;JPEG files (*.jpg);;PNG files (*.png);;All files (*)"
        };
        self.m_image_file_list = native.open_names(title, &native.browser_dir(), filter);
        self.enable_buttons(true);
        if let Some(name) = self.m_image_file_list.first().cloned() {
            native.manage_browser_dir(&name, false)
        }
        // `imageSelectClicked` (unlike `loadFileList`) strips the current
        // directory from the displayed text to stay below QLineEdit's limit.
        let cur_dir = format!("{}/", native.current_dir().trim_end_matches('/'));
        self.m_image_files = self.m_image_file_list.join(";").replace(&cur_dir, "");
        self.m_joined_with_space = false;
        if !self.m_image_files.contains(' ') {
            self.m_image_files = self.m_image_file_list.join(" ").replace(&cur_dir, "");
            self.m_joined_with_space = true;
        }
        self.ui.image_files_edit = self.m_image_files.clone();
        if self.m_image_files.len() > 32760 {
            native.warning("WARNING: The file list will be truncated in the Image file(s) edit box.\nAll files should load OK if you do not try to edit the list.")
        }
        self.m_files_changed = false;
    }
    /// `StartupForm::modelSelectClicked`.
    pub fn model_select_clicked(&mut self, native: &mut dyn StartupNativeBoundary) {
        if self.ui.prev_browser_dir_box && native.browser_dir().is_empty() {
            let p = native.prev_browser_dir();
            native.set_browser_dir(&p);
            self.m_browser_set_from_prev = true;
        }
        self.enable_buttons(false);
        self.m_model_file =
            native.open_file("Select model file to load", "Model files (*.*mod *.fid)");
        self.enable_buttons(true);
        self.ui.model_file_edit = self.m_model_file.clone()
    }
    /// `StartupForm::pieceSelectClicked`.
    pub fn piece_select_clicked(&mut self, native: &mut dyn StartupNativeBoundary) {
        self.enable_buttons(false);
        self.m_piece_file_list = native.open_names(
            "Select piece list file(s) to load",
            &native.browser_dir(),
            "Piece list files (*.pl *.mdoc);;All files (*)",
        );
        if let Some(name) = self.m_piece_file_list.first().cloned() {
            native.manage_browser_dir(&name, false)
        }
        self.enable_buttons(true);
        self.load_file_list(&self.m_piece_file_list.clone(), false);
        self.m_pieces_changed = false
    }
    /// `StartupForm::angleSelectClicked`.
    pub fn angle_select_clicked(&mut self, native: &mut dyn StartupNativeBoundary) {
        self.enable_buttons(false);
        self.m_angle_file =
            native.open_file("Select angle file to load", "Tilt angle files (*tlt)");
        self.enable_buttons(true);
        self.ui.angle_file_edit = self.m_angle_file.clone()
    }
    /// `StartupForm::enableButtons`.
    pub fn enable_buttons(&mut self, enable: bool) {
        self.ui.buttons_enabled = enable
    }
    /// `StartupForm::addArg`.
    pub fn add_arg(&mut self, arg: &str) {
        self.m_argv.push(arg.into());
        self.m_argc += 1
    }
    /// `StartupForm::getArguments`.
    pub fn get_arguments(&mut self, native: &dyn StartupNativeBoundary) -> Vec<String> {
        self.m_argv.clear();
        self.m_argc = 0;
        if self.m_modv_mode {
            self.add_arg("3dmodv");
            if self.m_modv_size_option == 1 {
                self.add_arg("-f")
            } else if self.m_modv_size_option == 2 {
                self.add_arg("-s");
                self.add_arg(&format!(
                    "{},{}",
                    self.ui.x_size_spin_box, self.ui.y_size_spin_box
                ));
            }
            self.add_image_files(native);
            return self.m_argv.clone();
        }
        self.add_arg("3dmod");
        for (checked, arg) in [
            (self.ui.open_zap_box, "-Z"),
            (self.ui.open_xyz_box, "-xyz"),
            (self.ui.open_slicer_box, "-S"),
            (self.ui.open_modv_box, "-V"),
            (self.ui.flip_check_box, "-Y"),
            (self.ui.fill_cache_box, "-F"),
            (self.ui.show_rgb_gray_box, "-G"),
            (self.ui.load_frames_box, "-f"),
            (self.ui.load_sep_times_box, "-T"),
            (self.ui.load_unscaled_box, "-m"),
            (self.ui.change_mrcstats_box, "-K"),
            (self.ui.load_no_mirror_box, "-M"),
        ] {
            if checked {
                self.add_arg(arg)
            }
        }
        self.add_arg("-I");
        self.add_arg(if self.ui.load_ushort_box { "1" } else { "0" });
        self.add_arg("-es");
        self.add_arg(&self.ui.super_res_spin_box.to_string());
        self.add_arg("-ez");
        self.add_arg(&self.ui.eer_zbin_spin_box.to_string());
        if self.ui.image_pyramid_box {
            self.add_arg("-py")
        };
        self.add_arg("-J");
        self.add_arg(&self.m_scan_type.to_string());
        self.add_arg("-A");
        self.add_arg(&self.m_piece_coord_type.to_string());
        if self.ui.bin_xy_spin_box * self.ui.bin_z_spin_box != 1 {
            self.add_arg("-b");
            self.add_arg(&format!(
                "{},{}",
                self.ui.bin_xy_spin_box, self.ui.bin_z_spin_box
            ));
        }
        if self.ui.show_montage_box {
            self.add_arg("-P");
            self.add_arg(&format!(
                "{},{}",
                self.ui.x_montage_spin_box, self.ui.y_montage_spin_box
            ));
            self.add_arg("-o");
            self.add_arg(&format!(
                "{},{}",
                self.ui.x_overlap_spin_box, self.ui.y_overlap_spin_box
            ));
        }
        if !self.ui.cache_size_edit.is_empty() || self.ui.tile_cache_box {
            self.add_arg(if self.ui.tile_cache_box { "-CT" } else { "-C" });
            let size = self.ui.cache_size_edit.parse::<i32>().unwrap_or(0);
            let suffix = if self.m_cache_option > 1 {
                "G"
            } else if self.m_cache_option != 0 {
                "M"
            } else {
                ""
            };
            self.add_arg(&format!("{size}{suffix}"));
        }
        if !self.ui.x_from_edit.is_empty() || !self.ui.x_to_edit.is_empty() {
            let lo = self.ui.x_from_edit.parse::<i32>().unwrap_or(-999);
            let hi = self.ui.x_to_edit.parse::<i32>().unwrap_or(65535);
            self.add_arg("-x");
            self.add_arg(&format!("{lo},{hi}"));
        }
        if !self.ui.y_from_edit.is_empty() || !self.ui.y_to_edit.is_empty() {
            let lo = self.ui.y_from_edit.parse::<i32>().unwrap_or(-999);
            let hi = self.ui.y_to_edit.parse::<i32>().unwrap_or(65535);
            self.add_arg("-y");
            self.add_arg(&format!("{lo},{hi}"));
        }
        if !self.ui.z_from_edit.is_empty() || !self.ui.z_to_edit.is_empty() {
            let lo = self.ui.z_from_edit.parse::<i32>().unwrap_or(-999);
            let hi = self.ui.z_to_edit.parse::<i32>().unwrap_or(65535);
            self.add_arg("-z");
            self.add_arg(&format!("{lo},{hi}"));
        }
        if !self.ui.scale_from_edit.is_empty() && !self.ui.scale_to_edit.is_empty() {
            self.add_arg("-s");
            self.add_arg(&format!(
                "{},{}",
                self.ui.scale_from_edit.parse::<i32>().unwrap_or(0),
                self.ui.scale_to_edit.parse::<i32>().unwrap_or(0)
            ));
        }
        if self.m_pieces_changed {
            self.m_piece_file_list =
                if !self.m_piece_files.contains(';') && self.m_piece_joinedw_space {
                    self.m_piece_files
                        .split_whitespace()
                        .map(str::to_owned)
                        .collect()
                } else {
                    self.m_piece_files
                        .split(';')
                        .filter(|x| !x.is_empty())
                        .map(str::to_owned)
                        .collect()
                };
        }
        for file in self.m_piece_file_list.clone() {
            self.add_arg("-p");
            self.add_arg(&file)
        }
        if !self.ui.angle_file_edit.is_empty() {
            self.add_arg("-a");
            self.add_arg(&self.ui.angle_file_edit.clone())
        }
        self.add_image_files(native);
        if !self.ui.model_file_edit.is_empty() {
            self.add_arg(&self.ui.model_file_edit.clone())
        };
        self.m_argv.clone()
    }
    /// `StartupForm::addImageFiles`.
    pub fn add_image_files(&mut self, native: &dyn StartupNativeBoundary) {
        if self.m_files_changed {
            self.m_image_file_list = if !self.m_image_files.contains(';')
                && (self.m_joined_with_space || self.m_image_files.contains(['/', '\\']))
            {
                self.m_image_files
                    .split_whitespace()
                    .map(str::to_owned)
                    .collect()
            } else {
                self.m_image_files
                    .split(';')
                    .filter(|x| !x.is_empty())
                    .map(str::to_owned)
                    .collect()
            };
        }
        for name in self.m_image_file_list.clone() {
            if !name.contains('*') && !name.contains('?') {
                self.add_arg(&name);
                continue;
            }
            // `QDir(path, pattern)` is a native filesystem-directory operation.
            let sep = name.rfind(['/', '\\']);
            let (path, pattern) = match sep {
                Some(index) => (&name[..=index], &name[index + 1..]),
                None => ("./", name.as_str()),
            };
            for file in native.matching_files(path, pattern) {
                self.add_arg(&format!("{path}{file}"));
            }
        }
    }
    /// `StartupForm::setValues`.
    #[allow(clippy::too_many_arguments)]
    pub fn set_values(
        &mut self,
        vi: &StartupViewValues,
        argv: &[String],
        firstfile: usize,
        do_imodv: i32,
        pl_file_names: &[String],
        anglefname: Option<&str>,
        piece_key_type: i32,
        xyzwinopen: i32,
        sliceropen: i32,
        zap_open: i32,
        model_view_open: i32,
        fill_cache: i32,
        imod_trans: i32,
        mirror: i32,
        frames: i32,
        nframex: i32,
        nframey: i32,
        overx: i32,
        overy: i32,
        over_entered: i32,
        native: &mut dyn StartupNativeBoundary,
    ) {
        self.m_modv_mode = do_imodv != 0;
        if firstfile < argv.len() {
            native.manage_browser_dir(&argv[firstfile], true);
            let mut last = argv.len() - 1;
            if native.is_model_file(&argv[last]) <= 0 {
                self.ui.model_file_edit = argv[last].clone();
                last -= 1;
            }
            if last >= firstfile {
                self.m_image_file_list = argv[firstfile..=last].to_vec();
                self.load_file_list(&self.m_image_file_list.clone(), true);
            }
        }
        if !pl_file_names.is_empty() {
            self.m_piece_file_list = pl_file_names.to_vec();
            self.load_file_list(&self.m_piece_file_list.clone(), false)
        }
        if let Some(angle) = anglefname {
            self.ui.angle_file_edit = angle.into()
        }
        if vi.xmin != -1 || vi.xmax != -1 {
            self.ui.x_from_edit = vi.xmin.to_string();
            self.ui.x_to_edit = vi.xmax.to_string()
        }
        if vi.ymin != -1 || vi.ymax != -1 {
            self.ui.y_from_edit = vi.ymin.to_string();
            self.ui.y_to_edit = vi.ymax.to_string()
        }
        if vi.zmin != -1 || vi.zmax != -1 {
            self.ui.z_from_edit = vi.zmin.to_string();
            self.ui.z_to_edit = vi.zmax.to_string()
        }
        self.ui.bin_xy_spin_box = vi.xybin;
        self.ui.bin_z_spin_box = vi.zbin;
        if vi.smin != vi.smax {
            self.ui.scale_from_edit = format!("{}", vi.smin);
            self.ui.scale_to_edit = format!("{}", vi.smax)
        }
        self.m_scan_type = vi.scale_scan_type;
        self.ui.change_mrcstats_box = vi.store_scan_in_mrc > 0;
        self.ui.open_zap_box = zap_open > 0;
        self.ui.open_xyz_box = xyzwinopen > 0;
        self.ui.open_slicer_box = sliceropen > 0;
        self.ui.open_modv_box = model_view_open > 0;
        self.m_cache_option = if vi.vm_entered_as_gb {
            2
        } else if vi.vm_size < 0 {
            1
        } else {
            0
        };
        let vms = if vi.vm_size == HUGE_CACHE {
            0
        } else {
            vi.vm_size
        };
        if vi.vm_size != 0 {
            self.ui.cache_size_edit = (if vi.vm_entered_as_gb {
                -vms / 1024
            } else if vms > 0 {
                vms
            } else {
                -vms
            })
            .to_string()
        }
        self.ui.tile_cache_box = vi.strip_or_tile_cache > 0;
        self.ui.image_pyramid_box = vi.image_pyramid > 0;
        self.ui.flip_check_box = vi.axis == 2;
        self.ui.fill_cache_box = fill_cache > 0;
        self.ui.show_rgb_gray_box = vi.gray_rgbs > 0;
        self.ui.load_sep_times_box = vi.multi_file_z < 0;
        self.ui.load_frames_box = frames > 0;
        self.ui.load_unscaled_box = imod_trans == 0;
        self.ui.load_no_mirror_box = mirror < 0;
        self.ui.load_ushort_box = vi.raw_image_store > 0;
        self.ui.super_res_spin_box = vi.eer_super_res;
        self.ui.eer_zbin_spin_box = vi.eer_zbinning;
        self.m_piece_coord_type = piece_key_type;
        self.m_show_montage = nframex > 0;
        if self.m_show_montage {
            self.ui.show_montage_box = true;
            self.ui.x_montage_spin_box = nframex;
            self.ui.y_montage_spin_box = nframey
        }
        if over_entered != 0 {
            self.ui.x_overlap_spin_box = overx;
            self.ui.y_overlap_spin_box = overy
        }
        self.manage_for_mod_view();
    }
    /// `StartupForm::loadFileList`.
    pub fn load_file_list(&mut self, file_list: &[String], images: bool) {
        let mut files = file_list.join(";");
        let joined = !files.contains(' ');
        if joined {
            files = files.replace(';', " ")
        }
        if images {
            self.m_image_files = files.clone();
            self.m_joined_with_space = joined;
            self.ui.image_files_edit = files
        } else {
            self.m_piece_files = files.clone();
            self.m_piece_joinedw_space = joined;
            self.ui.piece_file_edit = files
        }
    }
    /// `StartupForm::helpClicked`.
    pub fn help_clicked(&mut self, native: &mut dyn StartupNativeBoundary) {
        native.show_help_page("startup.html#TOP")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct N {
        browser: String,
        warned: bool,
    }
    impl StartupNativeBoundary for N {
        fn use_prev_browser_dir(&self) -> bool {
            false
        }
        fn set_use_prev_browser_dir(&mut self, _: bool) {}
        fn browser_dir(&self) -> String {
            self.browser.clone()
        }
        fn set_browser_dir(&mut self, x: &str) {
            self.browser = x.into()
        }
        fn prev_browser_dir(&self) -> String {
            "/tmp".into()
        }
        fn current_dir(&self) -> String {
            ".".into()
        }
        fn open_names(&mut self, _: &str, _: &str, _: &str) -> Vec<String> {
            vec![]
        }
        fn open_file(&mut self, _: &str, _: &str) -> String {
            String::new()
        }
        fn matching_files(&self, _: &str, _: &str) -> Vec<String> {
            vec![]
        }
        fn manage_browser_dir(&mut self, _: &str, _: bool) {}
        fn is_model_file(&self, _: &str) -> i32 {
            0
        }
        fn warning(&mut self, _: &str) {
            self.warned = true
        }
        fn show_help_page(&mut self, _: &str) {}
    }
    #[test]
    fn argument_order_matches_source_slots() {
        let mut n = N::default();
        let mut f = StartupForm::new(false, &mut n);
        f.ui.open_xyz_box = true;
        f.ui.bin_xy_spin_box = 2;
        f.ui.x_from_edit = "3".into();
        f.m_image_file_list = vec!["a.mrc".into()];
        assert_eq!(
            f.get_arguments(&n),
            vec![
                "3dmod", "-Z", "-xyz", "-I", "0", "-es", "1", "-ez", "10", "-J", "0", "-A", "0",
                "-b", "2,1", "-x", "3,65535", "a.mrc"
            ]
        );
    }
    #[test]
    fn modv_mode_and_file_list_follow_source() {
        let mut n = N::default();
        let mut f = StartupForm::new(false, &mut n);
        f.start_as_clicked(1);
        f.modv_size_clicked(2);
        f.ui.x_size_spin_box = 500;
        f.ui.y_size_spin_box = 400;
        f.image_changed("a.mod b.mod");
        f.m_joined_with_space = true;
        assert_eq!(
            f.get_arguments(&n),
            vec!["3dmodv", "-s", "500,400", "a.mod", "b.mod"]
        );
    }
}
