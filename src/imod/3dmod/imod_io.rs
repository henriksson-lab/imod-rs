//! Translation of `IMOD/3dmod/imod_io.cpp` and `imod_io.h`.
//!
//! The upstream unit combines model-file lifetime with 3dmod's global view
//! state.  [`ImodIoState`] owns those former static globals explicitly.  Qt
//! pickers, window redraws, model-view notification, and image-reader calls
//! are retained as calls on [`ImodIoBoundary`], rather than being replaced by
//! a headless viewer.
#![allow(dead_code)]

use std::fs::{File, OpenOptions, remove_file, rename};

use crate::imod::libcfshr::b3dutil::ImodFile;
use crate::imod::libiimod::iimage::IIFILE_MRC;
use crate::imod::libiimod::mrcfiles::mrc_read_byte;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::imod::libimod::imodel::{
    IMOD_MMOVIE, IMOD_UNIT_NM, IMODF_NEW_TO_3DMOD, Imod, imod_checksum, imod_new, imod_new_object,
};
use crate::imod::libimod::imodel_files::{imod_read_file, imod_write, imod_write_skip_mesh};
use crate::imod::three_dmod::imodview::{
    ImodView, ivw_get_pixel_bytes, ivw_read_binned_section, ivw_reopen,
};

pub const IMOD_IO_SUCCESS: i32 = 0;
pub const IMOD_IO_SAVE_ERROR: i32 = 1;
pub const IMOD_IO_SAVE_CANCEL: i32 = 2;
pub const IMOD_IO_DOES_NOT_EXIST: i32 = 3;
pub const IMOD_IO_NO_ACCESS_ERROR: i32 = 4;
pub const IMOD_IO_READ_ERROR: i32 = 5;
pub const IMOD_IO_NO_FILE_SELECTED: i32 = 6;
pub const IMOD_IO_NOMEM: i32 = 7;
pub const IMOD_IO_READ_CANCEL: i32 = 8;
pub const IMOD_IO_UNIMPLEMENTED_ERROR: i32 = 99;
pub const IMOD_FILENAME_SIZE: usize = 1024;

/// The module-static fields in `imod_io.cpp`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ImodIoState {
    pub imod_filename: String,
    pub autosave_filename: String,
    pub saved_filename: String,
    pub last_checksum: i32,
    pub last_error: i32,
}

/// Source `ViewInfo` fields used by this unit.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ImodIoViewState {
    pub black: i32,
    pub white: i32,
    pub xsize: i32,
    pub ysize: i32,
    pub zsize: i32,
    pub x_unbin_size: i32,
    pub y_unbin_size: i32,
    pub z_unbin_size: i32,
    pub xybin: i32,
    pub zbin: i32,
    pub cur_time: i32,
    pub num_times: i32,
    pub fake_image: bool,
    pub reloadable: bool,
    pub doing_initial_load: bool,
    pub did_model_init_in_load: bool,
}

/// Header scaling values from `ImodImageFile` used by `setModelScalesFromImage`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ImodIoImageScale {
    pub xscale: f32,
    pub zscale: f32,
}

/// Dialog selection returned by the source's `dia_choice` calls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SaveChoice {
    Yes,
    No,
    Cancel,
}

/// Native UI/viewer/image boundary of this source pair.
pub trait ImodIoBoundary {
    fn choose_save_name(&mut self) -> Option<PathBuf> {
        None
    }
    fn choose_load_name(&mut self) -> Option<PathBuf> {
        None
    }
    fn ask_save_current(&mut self, _save_as: bool) -> SaveChoice {
        SaveChoice::Cancel
    }
    fn print(&mut self, _message: &str) {}
    fn model_changed(&mut self, _model: Option<&Imod>) {}
    fn draw_window(&mut self) {}
    fn save_view(&mut self, _model: &mut Imod) {}
    fn restore_view_rotation(&mut self, _model: &mut Imod) {}
    fn save_view_rotation(&mut self, _model: &mut Imod) {}
    fn set_black_white_from_model(&mut self, _view: &mut ImodIoViewState, _model: &Imod) {}
    fn set_colormap(&mut self, _model: &Imod) {}
    fn maintain_model_name(&mut self, _model: &mut Imod) {}
    fn transform_model(&mut self, _view: &mut ImodIoViewState, _model: &mut Imod) {}
    fn info_set_bw(&mut self, _black: i32, _white: i32) {}
    fn model_new(&mut self, _model: Option<&Imod>) {}
    fn new_time(&mut self, _redraw: bool) {}
    fn model_edit_new_model(&mut self) {}
    fn plug_new_model(&mut self) {}
    fn image_load(&mut self, _view: &mut ImodIoViewState) -> Option<Vec<Vec<u8>>> {
        None
    }
}

/// Private `datetime` (`imod_io.cpp:67`).
///
/// The source copies eight characters out of `ctime`'s
/// `"Www Mmm dd hh:mm:ss yyyy\n"` at offset 11 into `dummystring`
/// (`imod_io.cpp:66`), which is **nine** blanks plus its terminator, so the
/// `strncpy` leaves the ninth blank in place and the returned string is
/// `"hh:mm:ss "` — with a trailing space that both `wprint` call sites
/// (`:215`, `:357`) print.
///
/// `ctime` formats local civil time.  Chrono supplies the corresponding local
/// clock while retaining the source's trailing blank.
pub fn datetime() -> String {
    chrono::Local::now().format("%H:%M:%S ").to_string()
}

/// `imod_model_changed`.
pub fn imod_model_changed(imodel: Option<&Imod>) -> i32 {
    imodel.is_some_and(|model| imod_checksum(model) != model.csum) as i32
}

/// Private `imod_make_backup`.
pub fn imod_make_backup(state: &mut ImodIoState, filename: &Path) {
    if filename.as_os_str().is_empty() || state.saved_filename == filename.to_string_lossy() {
        return;
    }
    let one = PathBuf::from(format!("{}~", filename.display()));
    let two = PathBuf::from(format!("{}~~", filename.display()));
    let _ = remove_file(&two);
    let _ = rename(&one, &two);
    let _ = rename(filename, &one);
    state.saved_filename = filename.to_string_lossy().into_owned();
}

/// Private `imod_undo_backup`.
pub fn imod_undo_backup(state: &mut ImodIoState) {
    if state.saved_filename.is_empty() {
        return;
    }
    let name = PathBuf::from(&state.saved_filename);
    let one = PathBuf::from(format!("{}~", name.display()));
    let two = PathBuf::from(format!("{}~~", name.display()));
    let _ = remove_file(&name);
    let _ = rename(&one, &name);
    let _ = rename(&two, &one);
    state.saved_filename.clear();
}

/// Private `imod_finish_backup`.
pub fn imod_finish_backup(state: &ImodIoState) {
    if !state.saved_filename.is_empty() {
        let _ = remove_file(format!("{}~~", state.saved_filename));
    }
}

/// `imod_cleanup_autosave`.
pub fn imod_cleanup_autosave(state: &mut ImodIoState) {
    state.last_error = IMOD_IO_SUCCESS;
    if !state.autosave_filename.is_empty() {
        let _ = remove_file(&state.autosave_filename);
    }
    state.last_checksum = -1;
}

/// `imod_autosave`.
pub fn imod_autosave(
    state: &mut ImodIoState,
    model: Option<&mut Imod>,
    autosave_dir: Option<&Path>,
    skip_mesh: i32,
    native: &mut dyn ImodIoBoundary,
) -> i32 {
    state.last_error = IMOD_IO_SUCCESS;
    let Some(model) = model else {
        return -1;
    };
    let checksum = imod_checksum(model);
    if checksum == model.csum || checksum == state.last_checksum {
        return IMOD_IO_SUCCESS;
    }
    imod_cleanup_autosave(state);
    let base = Path::new(&state.imod_filename)
        .file_name()
        .unwrap_or_default();
    let path = autosave_dir.map_or_else(
        || PathBuf::from(format!("{}#autosave#", state.imod_filename)),
        |d| d.join(format!("{}#autosave#", base.to_string_lossy())),
    );
    state.autosave_filename = path.to_string_lossy().into_owned();
    imod_cleanup_autosave(state);
    state.autosave_filename = path.to_string_lossy().into_owned();
    let Some(mut file) = ImodFile::open(&path, "w") else {
        state.autosave_filename.clear();
        return -1;
    };
    native.save_view_rotation(model);
    let result = imod_write_skip_mesh(model, &mut file, skip_mesh);
    native.restore_view_rotation(model);
    if result.is_ok() {
        native.print(&format!("Saved autosave file {}\n", datetime()));
    } else {
        native.print(&format!(
            "Error: Autosave model file not saved. {}\n",
            path.display()
        ));
    }
    state.last_checksum = imod_checksum(model);
    native.draw_window();
    IMOD_IO_SUCCESS
}

/// `currentSavedModelFile`.
pub fn current_saved_model_file<'a>(
    state: &'a mut ImodIoState,
    model: &mut Imod,
    autosave_dir: Option<&Path>,
    skip_mesh: i32,
    native: &mut dyn ImodIoBoundary,
) -> Option<&'a str> {
    let sum = imod_checksum(model);
    if sum == model.csum && !state.imod_filename.is_empty() {
        return Some(&state.imod_filename);
    }
    if sum == state.last_checksum && !state.autosave_filename.is_empty() {
        return Some(&state.autosave_filename);
    }
    (imod_autosave(state, Some(model), autosave_dir, skip_mesh, native) == IMOD_IO_SUCCESS)
        .then_some(state.autosave_filename.as_str())
}

/// `SaveModel`.
pub fn save_model(
    state: &mut ImodIoState,
    model: &mut Imod,
    view: ImodIoViewState,
    native: &mut dyn ImodIoBoundary,
) -> i32 {
    state.last_error = IMOD_IO_SUCCESS;
    if state.imod_filename.is_empty() {
        let filename = native.choose_save_name();
        return saveas_model(state, model, view, filename, native);
    }
    let name = PathBuf::from(&state.imod_filename);
    imod_make_backup(state, &name);
    // `imod_io.cpp:269` is `fopen(..., "wb+")`.
    let Some(file) = ImodFile::open(&name, "wb+") else {
        imod_undo_backup(state);
        let filename = native.choose_save_name();
        return saveas_model(state, model, view, filename, native);
    };
    write_model(state, model, view, file, &name, native)
}

/// `SaveasModel`.
pub fn saveas_model(
    state: &mut ImodIoState,
    model: &mut Imod,
    view: ImodIoViewState,
    filename: Option<PathBuf>,
    native: &mut dyn ImodIoBoundary,
) -> i32 {
    state.last_error = IMOD_IO_SUCCESS;
    let Some(name) = filename else {
        state.last_error = IMOD_IO_SAVE_CANCEL;
        return state.last_error;
    };
    imod_make_backup(state, &name);
    let Some(file) = ImodFile::open(&name, "w") else {
        imod_undo_backup(state);
        state.last_error = map_errno(io::Error::last_os_error());
        return state.last_error;
    };
    let ret = write_model(state, model, view, file, &name, native);
    if ret == 0 {
        set_imod_filename(state, &name.to_string_lossy());
        native.maintain_model_name(model);
    }
    ret
}

/// Private `writeModel`.
pub fn write_model(
    state: &mut ImodIoState,
    model: &mut Imod,
    view: ImodIoViewState,
    mut file: ImodFile,
    name: &Path,
    native: &mut dyn ImodIoBoundary,
) -> i32 {
    native.save_view(model);
    set_saved_model_state(model, view, native);
    let result = imod_write(model, &mut file);
    restore_saved_model_state(model, view, native);
    if result.is_err() {
        state.last_error = IMOD_IO_SAVE_ERROR;
        native.print("Error saving model.");
        return state.last_error;
    }
    native.print(&format!(
        "Done saving model {}\n{}\n",
        datetime(),
        name.display()
    ));
    imod_finish_backup(state);
    model.csum = imod_checksum(model);
    imod_cleanup_autosave(state);
    IMOD_IO_SUCCESS
}

/// Private `setSavedModelState`.
pub fn set_saved_model_state(
    model: &mut Imod,
    view: ImodIoViewState,
    native: &mut dyn ImodIoBoundary,
) {
    native.save_view_rotation(model);
    model.blacklevel = view.black;
    model.whitelevel = view.white;
    model.xmax = view.x_unbin_size;
    model.ymax = view.y_unbin_size;
    model.zmax = view.z_unbin_size;
    model.flags &= !IMODF_NEW_TO_3DMOD;
}

/// Private `restoreSavedModelState`.
pub fn restore_saved_model_state(
    model: &mut Imod,
    view: ImodIoViewState,
    native: &mut dyn ImodIoBoundary,
) {
    model.xmax = view.xsize;
    model.ymax = view.ysize;
    model.zmax = view.zsize;
    native.restore_view_rotation(model);
}

/// `LoadModel`, after `fopen` has supplied the model stream.
pub fn load_model(file: &mut ImodFile) -> Option<Imod> {
    let mut model = imod_new()?;
    imod_read_file(&mut model, file).ok()?;
    Some(model)
}

/// Private `LoadModelFile`, after the source picker boundary.
pub fn load_model_file(
    state: &mut ImodIoState,
    filename: Option<&Path>,
    native: &mut dyn ImodIoBoundary,
) -> Option<Imod> {
    state.last_error = IMOD_IO_SUCCESS;
    let path = filename
        .map(PathBuf::from)
        .or_else(|| native.choose_load_name())?;
    let Some(mut file) = ImodFile::open(&path, "r") else {
        state.last_error = map_errno(io::Error::last_os_error());
        return None;
    };
    native.print("Loading... ");
    let model = load_model(&mut file);
    if model.is_some() {
        set_imod_filename(state, &path.to_string_lossy());
    } else {
        state.last_error = IMOD_IO_READ_ERROR;
    }
    model
}

/// `openModel`.
pub fn open_model(
    state: &mut ImodIoState,
    current: &mut Imod,
    view: &mut ImodIoViewState,
    filename: Option<&Path>,
    keep_bw: bool,
    save_as: bool,
    native: &mut dyn ImodIoBoundary,
) -> i32 {
    if imod_model_changed(Some(current)) != 0 {
        match native.ask_save_current(save_as) {
            SaveChoice::Yes => {
                let err = if save_as {
                    let filename = native.choose_save_name();
                    saveas_model(state, current, *view, filename, native)
                } else {
                    save_model(state, current, *view, native)
                };
                if err != 0 {
                    return err;
                }
            }
            SaveChoice::Cancel => return IMOD_IO_SAVE_CANCEL,
            SaveChoice::No => {}
        }
    }
    imod_cleanup_autosave(state);
    let Some(mut loaded) = load_model_file(state, filename, native) else {
        return state.last_error;
    };
    init_read_in_model_data(&mut loaded, view, keep_bw, native);
    *current = loaded;
    if view.doing_initial_load {
        view.did_model_init_in_load = true;
    }
    IMOD_IO_SUCCESS
}

/// `initReadInModelData`.
pub fn init_read_in_model_data(
    model: &mut Imod,
    view: &mut ImodIoViewState,
    keep_bw: bool,
    native: &mut dyn ImodIoBoundary,
) {
    native.model_new(None);
    if !keep_bw && model.flags & IMODF_NEW_TO_3DMOD == 0 {
        native.set_black_white_from_model(view, model);
    }
    native.set_colormap(model);
    native.maintain_model_name(model);
    model.drawmode = 1;
    native.transform_model(view, model);
    native.restore_view_rotation(model);
    native.model_new(Some(model));
    native.info_set_bw(view.black, view.white);
    model.mousemode = IMOD_MMOVIE;
    model.csum = imod_checksum(model);
    native.new_time(true);
    native.model_edit_new_model();
    if !view.doing_initial_load {
        native.plug_new_model();
    }
}

/// `createNewModel`.
pub fn create_new_model(
    state: &mut ImodIoState,
    current: &mut Imod,
    view: &mut ImodIoViewState,
    filename: Option<&Path>,
    native: &mut dyn ImodIoBoundary,
) -> i32 {
    state.last_error = IMOD_IO_SUCCESS;
    if !view.doing_initial_load && imod_model_changed(Some(current)) != 0 {
        match native.ask_save_current(false) {
            SaveChoice::Yes => {
                let err = save_model(state, current, *view, native);
                if err != 0 {
                    return err;
                }
            }
            SaveChoice::Cancel => return IMOD_IO_SAVE_CANCEL,
            SaveChoice::No => {}
        }
        imod_cleanup_autosave(state);
    }
    let mode = if view.doing_initial_load {
        IMOD_MMOVIE
    } else {
        current.mousemode
    };
    let Some(mut model) = imod_new() else {
        state.last_error = IMOD_IO_NOMEM;
        return state.last_error;
    };
    if let Some(path) = filename.filter(|p| !p.as_os_str().is_empty()) {
        set_imod_filename(state, &path.to_string_lossy());
    } else {
        state.imod_filename.clear();
    }
    init_new_model(&mut model, view, ImodIoImageScale::default());
    view.reloadable = false;
    native.maintain_model_name(&mut model);
    model.mousemode = mode;
    native.set_colormap(&model);
    native.model_edit_new_model();
    native.model_new(Some(&model));
    native.new_time(true);
    if !view.doing_initial_load {
        native.plug_new_model();
    }
    model.csum = imod_checksum(&model);
    *current = model;
    IMOD_IO_SUCCESS
}

/// `initNewModel`.
pub fn init_new_model(model: &mut Imod, view: &ImodIoViewState, image: ImodIoImageScale) {
    let _ = imod_new_object(model);
    set_model_scales_from_image(model, view.fake_image, image, true);
    if view.num_times != 0 {
        if let Some(object) = model.obj.last_mut() {
            object.flags |= 1 << 10;
        }
    }
}

/// `setModelScalesFromImage`.
pub fn set_model_scales_from_image(
    model: &mut Imod,
    fake_image: bool,
    image: ImodIoImageScale,
    do_zscale: bool,
) {
    if !fake_image && image.xscale != 0. {
        if image.xscale != 1. {
            model.pixsize = image.xscale / 10.;
            model.units = IMOD_UNIT_NM;
        }
        if image.zscale != 0. && do_zscale {
            model.zscale = (1000. * image.zscale / image.xscale + 0.5).floor() * 0.001;
        }
    }
}

/// `imod_io_image_load`; load a non-cached image into the Rust-owned version
/// of the source `idata` allocation and return its legacy section-pointer
/// table.  `ImodView::idata_storage` keeps both the pixels and table alive
/// for the image windows that consume this result.
pub unsafe fn imod_io_image_load(vi: *mut ImodView) -> *mut *mut u8 {
    unsafe {
        let Some(view) = vi.as_mut() else {
            return std::ptr::null_mut();
        };
        let Some(image) = view.image.as_mut() else {
            return std::ptr::null_mut();
        };
        let Some(li) = view.li.as_mut() else {
            return std::ptr::null_mut();
        };
        if image.fp.is_none() && ivw_reopen(image) != 0 {
            return std::ptr::null_mut();
        }
        if image.fp.is_none() {
            return std::ptr::null_mut();
        }

        view.idata_storage.clear();
        view.idata_ptrs.clear();
        if image.file == IIFILE_MRC
            && view.raw_image_store == 0
            && view.xybin * view.zbin == 1
            && li.mirror_fft <= 0
        {
            let Some(header) = image.mrc_header.as_mut() else {
                return std::ptr::null_mut();
            };
            let Some(data) = mrc_read_byte(
                image.fp.as_mut().expect("checked above"),
                header,
                Some(li),
                None,
            ) else {
                return std::ptr::null_mut();
            };
            view.idata_storage = data;
        } else {
            let Some(plane_bytes) = (view.xsize as usize)
                .checked_mul(view.ysize as usize)
                .and_then(|size| {
                    size.checked_mul(ivw_get_pixel_bytes(view.raw_image_store as i32) as usize)
                })
            else {
                return std::ptr::null_mut();
            };
            let Some(num_sections) = usize::try_from(view.zsize).ok() else {
                return std::ptr::null_mut();
            };
            if view.idata_storage.try_reserve_exact(num_sections).is_err() {
                return std::ptr::null_mut();
            }
            for section in 0..view.zsize {
                let mut data = vec![0; plane_bytes];
                if ivw_read_binned_section(vi, data.as_mut_ptr(), section + li.zmin) != 0 {
                    return std::ptr::null_mut();
                }
                view.idata_storage.push(data);
            }
        }
        view.idata_ptrs = view.idata_storage.iter_mut().map(Vec::as_mut_ptr).collect();
        view.idata = view.idata_ptrs.as_mut_ptr();
        view.idata
    }
}

/// `setImod_filename`.
pub fn set_imod_filename(state: &mut ImodIoState, name: &str) {
    state.imod_filename = name.chars().take(IMOD_FILENAME_SIZE - 1).collect();
}
/// `imodIOGetError`.
pub fn imod_io_get_error(state: &ImodIoState) -> i32 {
    state.last_error
}
/// `imodIOGetErrorString`.
pub fn imod_io_get_error_string(state: &ImodIoState) -> &'static str {
    match state.last_error {
        IMOD_IO_SAVE_ERROR => "Unable to save existing model",
        IMOD_IO_DOES_NOT_EXIST => "File does not exist",
        IMOD_IO_NO_ACCESS_ERROR => "Unable to access path or file, check permissions",
        IMOD_IO_NO_FILE_SELECTED => "File not selected",
        IMOD_IO_NOMEM => "Insufficient memory, try closing other programs",
        _ => "Unknown error",
    }
}
/// Private `mapErrno`.
pub fn map_errno(error: io::Error) -> i32 {
    match error.kind() {
        io::ErrorKind::OutOfMemory => IMOD_IO_NOMEM,
        io::ErrorKind::PermissionDenied => IMOD_IO_NO_ACCESS_ERROR,
        io::ErrorKind::NotFound => IMOD_IO_DOES_NOT_EXIST,
        _ => IMOD_IO_UNIMPLEMENTED_ERROR,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Native {
        saved: Option<PathBuf>,
    }
    impl ImodIoBoundary for Native {
        fn choose_save_name(&mut self) -> Option<PathBuf> {
            self.saved.clone()
        }
    }
    #[test]
    fn filename_is_truncated_like_source() {
        let mut s = ImodIoState {
            last_checksum: -1,
            ..Default::default()
        };
        set_imod_filename(&mut s, &"x".repeat(IMOD_FILENAME_SIZE + 4));
        assert_eq!(s.imod_filename.len(), IMOD_FILENAME_SIZE - 1);
    }
    #[test]
    fn scales_use_source_rounding() {
        let mut m = Imod::default();
        set_model_scales_from_image(
            &mut m,
            false,
            ImodIoImageScale {
                xscale: 2.,
                zscale: 2.3456,
            },
            true,
        );
        assert_eq!(m.pixsize, 0.2);
        assert!((m.zscale - 1.173).abs() < 0.000_01);
        assert_eq!(m.units, IMOD_UNIT_NM);
    }
    #[test]
    fn datetime_keeps_ctime_time_field_layout() {
        let value = datetime();
        assert_eq!(value.len(), 9);
        assert_eq!(&value[2..3], ":");
        assert_eq!(&value[5..6], ":");
        assert_eq!(&value[8..], " ");
        assert!(
            value[..8]
                .bytes()
                .enumerate()
                .all(|(index, byte)| index == 2 || index == 5 || byte.is_ascii_digit())
        );
    }
    #[test]
    fn io_error_kinds_keep_source_error_mapping() {
        assert_eq!(
            map_errno(io::Error::from(io::ErrorKind::OutOfMemory)),
            IMOD_IO_NOMEM
        );
        assert_eq!(
            map_errno(io::Error::from(io::ErrorKind::PermissionDenied)),
            IMOD_IO_NO_ACCESS_ERROR
        );
        assert_eq!(
            map_errno(io::Error::from(io::ErrorKind::NotFound)),
            IMOD_IO_DOES_NOT_EXIST
        );
    }
    #[test]
    fn save_and_load_use_real_imod_stream() {
        let path = std::env::temp_dir().join(format!(
            "imod-io-{}-{}.mod",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut state = ImodIoState {
            last_checksum: -1,
            ..Default::default()
        };
        set_imod_filename(&mut state, &path.to_string_lossy());
        let mut m = imod_new().unwrap();
        imod_new_object(&mut m);
        let mut n = Native::default();
        assert_eq!(
            save_model(&mut state, &mut m, ImodIoViewState::default(), &mut n),
            0
        );
        assert!(load_model_file(&mut state, Some(&path), &mut n).is_some());
        let _ = remove_file(path);
    }

    #[test]
    fn noncached_mrc_load_keeps_the_source_section_table_alive_on_the_view() {
        use crate::imod::libiimod::iimage::{ii_delete, ii_open};
        use crate::imod::libiimod::mrcfiles::{
            LoadInfo, MRC_MODE_BYTE, MrcHeader, mrc_head_new, mrc_head_write, mrc_init_li,
            mrc_write_slice,
        };
        let path = std::env::temp_dir().join(format!(
            "imod-io-image-{}-{}.mrc",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut header = MrcHeader::default();
        assert_eq!(mrc_head_new(&mut header, 2, 2, 1, MRC_MODE_BYTE), 0);
        header.amin = 1.;
        header.amax = 4.;
        header.amean = 2.5;
        let mut output = ImodFile::open(&path, "wb").unwrap();
        assert_eq!(mrc_head_write(&mut output, &mut header), 0);
        assert_eq!(
            mrc_write_slice(&[1, 2, 3, 4], &mut output, &mut header, 0, b'Z'),
            0
        );
        drop(output);

        let image = unsafe { ii_open(path.as_os_str().as_encoded_bytes(), "rb") };
        assert!(!image.is_null());
        let mut li = LoadInfo::default();
        assert_eq!(
            mrc_init_li(Some(&mut li), unsafe { (*image).mrc_header.as_ref() }),
            0
        );
        let mut view = ImodView {
            image,
            li: &mut li,
            xsize: 2,
            ysize: 2,
            zsize: 1,
            xysize: 4,
            ..Default::default()
        };
        let table = unsafe { imod_io_image_load(&mut view) };
        assert!(!table.is_null());
        assert_eq!(view.idata, table);
        assert_eq!(view.idata_storage, vec![vec![0, 85, 170, 255]]);
        assert_eq!(
            unsafe { std::slice::from_raw_parts(*table, 4) },
            [0, 85, 170, 255]
        );
        unsafe { ii_delete(image) };
        let _ = remove_file(path);
    }
}
