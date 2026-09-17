//! Translation of `IMOD/3dmod/utilities.cpp` and `utilities.h`.
//!
//! The original is a deliberately mixed utility unit.  Its model-independent
//! calculations are implemented below; painting, Qt widgets, GLU tessellation,
//! file dialogs, and application-global services are represented by the explicit
//! [`UtilitiesBoundary`] instead of silently replacing their behaviour.
#![allow(dead_code, unused_variables)]

use crate::imod::libimod::icont::ICONT_STIPPLED;
use crate::imod::libimod::imodel::{IMODF_FLIPYZ, IMODF_ROT90X, Icont, Imod, Iobj};
use crate::imod::libimod::iobj::{
    IOBJ_SYM_CIRCLE, IOBJ_SYM_NONE, IOBJ_SYM_SQUARE, IOBJ_SYM_STAR, IOBJ_SYM_TRIANGLE,
    IOBJ_SYMF_FILL, iobj_scat,
};
use crate::imod::three_dmod::imodview::ImodView;
use crate::imod::three_dmod::scalebar::ScaleBar;

pub const TB_AUTO_RAISE: bool = true;
pub const FLIP_TO_ROTATION: i32 = 0;
pub const ROTATION_TO_FLIP: i32 = 1;

/// Toolkit-independent equivalent of the cursor choices in `utilSetCursor`.
/// The desktop backend maps these to its own cursor API instead of inheriting
/// Qt's numeric `CursorShape` ABI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UtilityCursor {
    SizeAll,
    SizeForwardDiagonal,
    SizeBackwardDiagonal,
    SizeHorizontal,
    SizeVertical,
    Model,
    Default,
}

/// State retained between `utilSetCursor` calls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UtilityCursorState {
    pub mouse_mode: i32,
    /// `None` is the source's `lastShape == -1` non-special state.
    pub last_special: Option<UtilityCursor>,
}

impl Default for UtilityCursorState {
    fn default() -> Self {
        Self {
            mouse_mode: -1,
            last_special: None,
        }
    }
}

/// Rust ownership for the chunked RGBA backing image assembled by
/// `utilStartMontSnap`.  The C code exposes a `unsigned char **` line table;
/// `line_chunk_offsets` preserves that mapping without manufacturing aliased
/// mutable slices.
#[derive(Clone, Debug, PartialEq)]
pub struct MontageSnapshotBuffers {
    pub frame_pixels: Vec<u8>,
    pub chunks: Vec<Vec<u8>>,
    pub line_chunk_offsets: Vec<(usize, usize)>,
    pub width: usize,
    pub height: usize,
}

/// Source `utilStartMontSnap`'s fixed maximum chunk allocation.  Keeping the
/// chunking matters for very tall snapshots and avoids one giant allocation.
pub const MONT_SNAP_CHUNK_MAX: usize = 10_000_000;

/// Allocate the source montage backing image and apply its temporary scale-bar
/// scaling.  Returns `None` on invalid dimensions, overflow, or allocation
/// failure, corresponding to C's `true` error return.
pub fn util_start_mont_snap(
    winx: i32,
    winy: i32,
    full_width: i32,
    full_height: i32,
    factor: f32,
    bar: &mut ScaleBar,
) -> Option<(MontageSnapshotBuffers, ScaleBar)> {
    let (winx, winy, width, height) = (
        usize::try_from(winx).ok()?,
        usize::try_from(winy).ok()?,
        usize::try_from(full_width).ok()?,
        usize::try_from(full_height).ok()?,
    );
    if winx == 0 || winy == 0 || width == 0 || height == 0 || !factor.is_finite() {
        return None;
    }
    let row_bytes = width.checked_mul(4)?;
    let max_lines = (MONT_SNAP_CHUNK_MAX / row_bytes).max(1);
    let chunk_count = height.checked_add(max_lines - 1)? / max_lines;
    let frame_len = winx.checked_mul(winy)?.checked_mul(4)?;
    let mut frame_pixels = Vec::new();
    frame_pixels.try_reserve_exact(frame_len).ok()?;
    frame_pixels.resize(frame_len, 0);
    let mut chunks = Vec::new();
    chunks.try_reserve_exact(chunk_count).ok()?;
    let mut line_chunk_offsets = Vec::new();
    line_chunk_offsets.try_reserve_exact(height).ok()?;
    let mut line = 0;
    for chunk_index in 0..chunk_count {
        let lines = (height - line).min(max_lines);
        let chunk_len = lines.checked_mul(row_bytes)?;
        let mut chunk = Vec::new();
        chunk.try_reserve_exact(chunk_len).ok()?;
        chunk.resize(chunk_len, 0);
        for in_chunk_line in 0..lines {
            line_chunk_offsets.push((chunk_index, in_chunk_line * row_bytes));
        }
        chunks.push(chunk);
        line += lines;
    }
    let saved_bar = bar.clone();
    // Source `B3DNINT` receives non-negative scale-bar dimensions here.
    let scaled = |value: i32| (factor * value as f32 + 0.5) as i32;
    bar.min_length = scaled(bar.min_length);
    bar.thickness = scaled(bar.thickness);
    bar.indent_x = scaled(bar.indent_x);
    bar.indent_y = scaled(bar.indent_y);
    bar.scale_label = factor;
    Some((
        MontageSnapshotBuffers {
            frame_pixels,
            chunks,
            line_chunk_offsets,
            width,
            height,
        },
        saved_bar,
    ))
}

/// `utilMontSnapScaleBar`, excluding the renderer's `scaleBarTestAdjust`.
/// Returns whether the caller should run that host-side adjustment after the
/// source determines that this panel owns the scale bar.
pub fn util_mont_snap_scale_bar(
    bar: &mut ScaleBar,
    ix: i32,
    iy: i32,
    frames: i32,
    saved_draw: bool,
) -> bool {
    let position = bar.position;
    bar.draw = false;
    let x_edge = ((position == 0 || position == 3) && ix == frames - 1)
        || ((position == 1 || position == 2) && ix == 0);
    let y_edge = ((position == 2 || position == 3) && iy == frames - 1)
        || ((position == 0 || position == 1) && iy == 0);
    if x_edge && y_edge {
        bar.draw = saved_draw;
        return true;
    }
    false
}

/// Saved state from `utilPreSnapChanges`, replacing its file-static
/// `sSaveCursor`.  `set_scale_bar_without_dialog` is a request for the host's
/// scale-bar controller; this utility deliberately does not assume Qt widget
/// ownership.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SnapshotPreChanges {
    pub saved_cursor: Option<i32>,
    pub set_scale_bar_without_dialog: bool,
}

/// State mutation portion of `utilPreSnapChanges`.
pub fn util_pre_snap_changes(
    bar: &ScaleBar,
    view: Option<&mut ImodView>,
    hide_current_point: bool,
) -> SnapshotPreChanges {
    let mut changes = SnapshotPreChanges {
        saved_cursor: None,
        set_scale_bar_without_dialog: bar.draw_on_snapshots,
    };
    if hide_current_point {
        if let Some(view) = view {
            changes.saved_cursor = Some(view.drawcursor);
            view.drawcursor = 0;
        }
    }
    changes
}

/// State mutation portion of `utilRestoreSnapChanges`.  Returns whether the
/// host must disable the temporary no-dialog scale bar state.
pub fn util_restore_snap_changes(view: Option<&mut ImodView>, changes: SnapshotPreChanges) -> bool {
    if let (Some(view), Some(cursor)) = (view, changes.saved_cursor) {
        view.drawcursor = cursor;
    }
    changes.set_scale_bar_without_dialog
}

/// `PopupEntry` in `utilities.h`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PopupEntry {
    pub text: &'static str,
    pub key: i32,
    pub ctrl: bool,
    pub shift: bool,
    pub main_index: i16,
}

/// Toolkit-neutral state produced by the Qt toolbar helpers in this source unit.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct UtilityToolButton {
    pub text: Option<String>,
    pub tooltip: Option<String>,
    pub checkable: bool,
    pub checked: bool,
    pub icon_off: Option<String>,
    pub icon_on: Option<String>,
    pub auto_raise: bool,
}

/// Native retained `HotToolBar` descriptor.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct UtilityToolBar {
    pub caption: Option<String>,
    pub spacing: i32,
    pub add_break: bool,
    pub buttons: Vec<UtilityToolButton>,
}

/// Native popup action created from a `PopupEntry`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UtilityPopupAction {
    pub text: String,
    pub key: i32,
    pub ctrl: bool,
    pub shift: bool,
}

/// Java/C++ Qt `utilTBZoomTools` retained configuration.
pub fn util_tb_zoom_tools(tool_bar: &mut UtilityToolBar, zoom_in_tip: &str, zoom_out_tip: &str) {
    tool_bar.buttons.extend([
        UtilityToolButton {
            tooltip: Some(zoom_in_tip.into()),
            auto_raise: TB_AUTO_RAISE,
            ..Default::default()
        },
        UtilityToolButton {
            tooltip: Some(zoom_out_tip.into()),
            auto_raise: TB_AUTO_RAISE,
            ..Default::default()
        },
    ]);
}

/// `utilTBArrowButton` native descriptor.
pub fn util_tb_arrow_button(tooltip: Option<&str>) -> UtilityToolButton {
    UtilityToolButton {
        tooltip: tooltip.map(str::to_owned),
        auto_raise: TB_AUTO_RAISE,
        ..Default::default()
    }
}

/// `utilTBToolEdit` native descriptor.
pub fn util_tb_tool_edit(tooltip: Option<&str>) -> UtilityToolButton {
    util_tb_arrow_button(tooltip)
}

/// `utilTBToolButton` native descriptor.
pub fn util_tb_tool_button(icon_off: Option<&str>, tooltip: Option<&str>) -> UtilityToolButton {
    UtilityToolButton {
        icon_off: icon_off.map(str::to_owned),
        tooltip: tooltip.map(str::to_owned),
        auto_raise: TB_AUTO_RAISE,
        ..Default::default()
    }
}

/// `utilTBPushButton` native descriptor.
pub fn util_tb_push_button(text: &str, tooltip: Option<&str>) -> UtilityToolButton {
    UtilityToolButton {
        text: Some(text.into()),
        tooltip: tooltip.map(str::to_owned),
        ..Default::default()
    }
}

/// `utilFileListsToIcons`: retain off/on asset paths for the native icon backend.
pub fn util_file_lists_to_icons(file_list: &[(String, String)]) -> Vec<UtilityToolButton> {
    file_list
        .iter()
        .map(|(off, on)| UtilityToolButton {
            icon_off: Some(off.clone()),
            icon_on: Some(on.clone()),
            ..Default::default()
        })
        .collect()
}

/// `utilSetupToggleButton`: make the source-default unchecked toggle at `index`.
pub fn util_setup_toggle_button(
    icons: &[UtilityToolButton],
    tips: Option<&[String]>,
    states: &mut [i32],
    index: usize,
) -> Option<UtilityToolButton> {
    let icon = icons.get(index)?;
    if let Some(state) = states.get_mut(index) {
        *state = 0;
    }
    Some(UtilityToolButton {
        icon_off: icon.icon_off.clone(),
        icon_on: icon.icon_on.clone(),
        tooltip: tips.and_then(|tips| tips.get(index)).cloned(),
        checkable: true,
        checked: false,
        auto_raise: TB_AUTO_RAISE,
        ..Default::default()
    })
}

/// `utilFreeMontSnapArrays`: Rust drops the owned snapshot buffers.
pub fn util_free_mont_snap_arrays(buffers: &mut Option<MontageSnapshotBuffers>) {
    *buffers = None;
}

/// Pure result-selection portion of `utilFinishMontSnap`.
pub fn util_finish_mont_snap(
    width: i32,
    height: i32,
    format: i32,
    prefix: &str,
    fileno: i32,
    digits: usize,
) -> String {
    format!(
        "{prefix}{fileno:0digits$}.{}",
        if format == 0 { "tif" } else { "png" }
    ) + &format!(" ({width} x {height})")
}

/// `utilAssignSurfToCont` state mutation; the caller owns undo/reporting.
pub fn util_assign_surf_to_cont(cont: &mut Icont, new_surf: i32) {
    cont.surf = new_surf;
}

/// `utilBuildPopupMenu`: transform specialized and optional default entries into actions.
pub fn util_build_popup_menu(
    specific: &[PopupEntry],
    add_default: bool,
) -> Vec<UtilityPopupAction> {
    specific
        .iter()
        .chain(add_default.then_some(DEFAULT_ACTIONS).into_iter().flatten())
        .filter(|entry| entry.key > 0)
        .map(|entry| UtilityPopupAction {
            text: entry.text.into(),
            key: entry.key,
            ctrl: entry.ctrl,
            shift: entry.shift,
        })
        .collect()
}

/// `utilBuildExecPopupMenu`: host selects an action; return its source key/modifiers.
pub fn util_build_exec_popup_menu(
    actions: &[UtilityPopupAction],
    selected: Option<usize>,
) -> Option<(i32, bool, bool)> {
    selected
        .and_then(|index| actions.get(index))
        .map(|action| (action.key, action.ctrl, action.shift))
}

/// `utilMakeToolBar` native retained toolbar descriptor.
pub fn util_make_tool_bar(add_break: bool, spacing: i32, caption: Option<&str>) -> UtilityToolBar {
    UtilityToolBar {
        caption: caption.map(str::to_owned),
        spacing,
        add_break,
        ..Default::default()
    }
}

/// `utilFinishTimeToolBar` retained controls and labels.
pub fn util_finish_time_tool_bar(tool_bar: &mut UtilityToolBar, big_time_label: &str) {
    tool_bar.buttons.extend([
        UtilityToolButton {
            text: Some("4th D".into()),
            ..Default::default()
        },
        util_tb_arrow_button(Some("Move to previous image file, (hot key 1)")),
        util_tb_arrow_button(Some("Move to next image file (hot key 2)")),
        UtilityToolButton {
            text: Some(" (999)".into()),
            ..Default::default()
        },
        UtilityToolButton {
            text: Some(big_time_label.into()),
            ..Default::default()
        },
    ]);
}
/// Native record for GLU's `tessError` callback.
pub fn tess_error(error: i32) -> i32 {
    error
}
/// `setupFilledContTesselator`; the renderer owns the actual GLU object.
pub fn setup_filled_cont_tesselator(initialized: &mut bool) {
    *initialized = true;
}
/// Native title form of `setModvDialogTitle`.
pub fn set_modv_dialog_title(
    intro: &str,
    model_name: Option<&str>,
    image_name: Option<&str>,
) -> String {
    imodw_either_name(intro, model_name, image_name, true)
        .unwrap_or_else(|| intro.trim_end_matches(':').to_owned())
}
/// `imodwfname`, including its multi-file fallback text.
pub fn imodwfname(
    intro: &str,
    image_name: Option<&str>,
    num_times: i32,
    current_time: i32,
) -> Option<String> {
    image_name
        .map(|name| format!("{intro}{}{}", if intro.is_empty() { "" } else { " " }, name))
        .or_else(|| {
            (num_times > 1).then(|| {
                format!(
                    "{intro}{}{} image files{}",
                    if intro.is_empty() { "" } else { " " },
                    num_times,
                    if intro == "3dmod:" && current_time > 0 {
                        format!(" ({current_time})")
                    } else {
                        String::new()
                    }
                )
            })
        })
}
/// Native formatting forms of the source diagnostics; the host selects the output sink.
pub fn imod_error(message: impl Into<String>) -> String {
    message.into()
}
pub fn imod_print_info(message: impl Into<String>) -> String {
    message.into()
}
pub fn imod_print_stderr(message: impl Into<String>) -> String {
    message.into()
}
pub fn imod_trace(enabled: bool, message: impl Into<String>) -> Option<String> {
    enabled.then(|| message.into())
}
pub fn imod_puts(message: impl Into<String>) -> String {
    format!("{}\n", message.into())
}
pub fn imod_show_help_page(page: &str, show: impl FnOnce(&str) -> bool) -> i32 {
    (!show(page)) as i32
}
/// `utilWprintMeasure` unit conversion and message formatting.
pub fn util_wprint_measure(
    base: &str,
    pixel_size: f32,
    measure: f32,
    area: bool,
    units: &str,
) -> String {
    if units == "pixels" {
        return base.to_owned();
    }
    let value = measure * pixel_size * if area { pixel_size } else { 1.0 };
    format!("{base}, {value} {units}{}", if area { "^2" } else { "" })
}
pub fn util_set_stay_on_top(_current: bool, state: bool) -> bool {
    state
}
/// `utilManageListStackSizes`, returning (list width, list height, stack width, stack height).
pub fn util_manage_list_stack_sizes(
    item_widths: &[i32],
    font_height: i32,
    stack_sizes: &[(i32, i32)],
) -> (i32, i32, i32, i32) {
    (
        item_widths.iter().copied().max().unwrap_or(0) + 12,
        ((font_height as f32 + 1.5) * item_widths.len() as f32) as i32,
        stack_sizes.iter().map(|size| size.0).max().unwrap_or(0),
        stack_sizes.iter().map(|size| size.1).max().unwrap_or(0),
    )
}
/// `utilOpenFileName` native chooser request, preserving Qt's `;;` filter separator.
pub fn util_open_file_name(
    caption: &str,
    filters: &[&str],
    choose: impl FnOnce(&str, &str) -> Option<String>,
) -> Option<String> {
    let filter = filters
        .iter()
        .copied()
        .chain(std::iter::once("All Files (*)"))
        .collect::<Vec<_>>()
        .join(";;");
    choose(caption, &filter)
}

/// Native model-editor form of `utilAutoNewContour`.
///
/// The editor supplies `new_contour`, which performs Java/C++'s current-object and undo
/// updates; this helper preserves the source decision to start a contour on a plane/time
/// mismatch and returns the newly selected contour.
pub fn util_auto_new_contour<F>(
    current: Option<&Icont>,
    not_in_plane: bool,
    time_mismatch: bool,
    mut new_contour: F,
) -> Option<Icont>
where
    F: FnMut() -> Option<Icont>,
{
    if current.is_none() || not_in_plane || time_mismatch {
        new_contour()
    } else {
        current.cloned()
    }
}

/// Native backend boundary for `utilManagePairedMeshes`.
/// The renderer owns pair generation/removal, while this preserves the source decision:
/// a positive object thickness requires paired meshes, zero removes them.  `true` is an
/// allocation/render failure, matching the C return convention.
pub fn util_manage_paired_meshes(obj: &Iobj, mut manage_pairs: impl FnMut(bool) -> bool) -> i32 {
    manage_pairs(obj.mesh_thickness != 0) as i32
}

/// The non-Rust services called by this source unit.
pub trait UtilitiesBoundary {
    fn draw_symbol(&mut self, x: i32, y: i32, symbol: i32, size: i32, filled: bool);
    fn set_stipple(&mut self, enabled: bool);
    fn clear_window(&mut self, color_index: i32);
    fn redraw_model(&mut self);
    fn change_point_size(&mut self);
    fn finish_undo_unit(&mut self);
    fn message(&mut self, text: &str);
    fn flip_yz(&mut self, imod: &mut Imod);
    fn rotate_90_x(&mut self, imod: &mut Imod, inverse: bool);
    fn draw_filled_polygon(&mut self, points: &[crate::imod::libimod::imodel::Ipoint]);
}

/// `utilDrawSymbol`.
pub fn util_draw_symbol(
    boundary: &mut dyn UtilitiesBoundary,
    mx: i32,
    my: i32,
    sym: i32,
    size: i32,
    flags: u32,
) {
    match sym {
        IOBJ_SYM_CIRCLE | IOBJ_SYM_SQUARE | IOBJ_SYM_TRIANGLE => {
            boundary.draw_symbol(mx, my, sym, size, flags & IOBJ_SYMF_FILL != 0)
        }
        IOBJ_SYM_STAR => {}
        IOBJ_SYM_NONE => boundary.draw_symbol(mx, my, sym, 1, true),
        _ => {}
    }
}

/// `utilGetLongestTimeString`; `time_labels` is `ivwGetTimeIndexLabel`.
pub fn util_get_longest_time_string(num_times: i32, time_labels: &[String]) -> String {
    if num_times == 0 {
        return String::new();
    }
    let mut out = " (999)".to_owned();
    let mut max_len = -1_i32;
    let mut tmax = 0usize;
    for time in 1..num_times as usize {
        if let Some(label) = time_labels.get(time) {
            if label.len() as i32 > max_len {
                max_len = label.len() as i32;
                tmax = time;
            }
        }
    }
    if let Some(label) = time_labels.get(tmax) {
        out.push_str(label);
    }
    out
}

/// `utilShrinkFilenameToFit`; `width_of` is QWidget's font metric call.
pub fn util_shrink_filename_to_fit(label: &mut String, width: i32, width_of: impl Fn(&str) -> i32) {
    let len = label.chars().count();
    if width_of(label) > width - 4 {
        for rem in 0..len / 2 - 3 {
            let keep = len / 2 - rem;
            let text = format!(
                "{}...{}",
                label.chars().take(keep).collect::<String>(),
                label.chars().skip(len - keep).collect::<String>()
            );
            if width_of(&text) <= width - 4 {
                *label = text;
                return;
            }
        }
    }
}

/// `utilCurrentPointSize`.
pub fn util_current_point_size(
    obj: Option<&Iobj>,
    min_mod_size: i32,
    min_im_size: i32,
    xybin: i32,
) -> (i32, i32, i32) {
    let (mut mod_size, mut backup_size, mut im_size) =
        (min_mod_size, min_mod_size + 2, min_im_size);
    let Some(obj) = obj else {
        return (mod_size, backup_size, im_size);
    };
    let mut sym_size = if obj.symbol as i32 != IOBJ_SYM_NONE && obj.symsize > 0 {
        obj.symsize as i32
    } else {
        0
    };
    if sym_size == 0 && obj.pdrawsize > 0 {
        sym_size = obj.pdrawsize / xybin;
    }
    if (sym_size - mod_size).abs() < 2 {
        mod_size = sym_size + 2;
    }
    backup_size = mod_size + 2;
    if (sym_size - backup_size).abs() < 2 {
        backup_size = sym_size + 2;
    }
    if (sym_size - im_size).abs() < 2 {
        im_size = sym_size + 2;
    }
    (mod_size, backup_size, im_size)
}

/// `utilEnableStipple`.
pub fn util_enable_stipple(
    boundary: &mut dyn UtilitiesBoundary,
    draw_stipple: i32,
    cont: &Icont,
) -> bool {
    let enabled = draw_stipple != 0 && cont.flags & ICONT_STIPPLED != 0;
    if enabled {
        boundary.set_stipple(true);
    }
    enabled
}
/// `utilDisableStipple`.
pub fn util_disable_stipple(boundary: &mut dyn UtilitiesBoundary, draw_stipple: i32, cont: &Icont) {
    if draw_stipple != 0 && cont.flags & ICONT_STIPPLED != 0 {
        boundary.set_stipple(false);
    }
}
/// `utilCloseKey` (`utilities.cpp:502`); the `Q_OS_MACX` Ctrl-W arm is not
/// compiled on this platform, so only Escape closes.  The Qt key code travels
/// as the plain `int` the rest of the translated input path uses.
pub fn util_close_key(key: i32) -> bool {
    key == 0x0100_0000
}

/// `utilRaiseIfNeeded`.  The source body is compiled only for the legacy
/// macOS Qt path; the Linux winit host has no corresponding raise action.
pub fn util_raise_if_needed() {}

/// `utilNeedToSetCursor`.  This is true only for the legacy macOS Qt path.
pub fn util_need_to_set_cursor() -> bool {
    false
}

/// `utilIgnoreClosing`.  Its warning and ignored close event exist only for
/// the bounded macOS Qt 5.12--5.14 build configuration.
pub fn util_ignore_closing(_closing: bool) -> bool {
    false
}

/// Geometry/state portion of `utilSetCursor`.  An `Some` result is the cursor
/// update that the GUI backend must apply; `None` means native would leave the
/// existing cursor untouched.
pub fn util_set_cursor(
    mode: i32,
    set_anyway: bool,
    need_special: bool,
    need_size_all: bool,
    dragging: [i32; 4],
    need_model: bool,
    state: &mut UtilityCursorState,
) -> Option<UtilityCursor> {
    if need_special {
        let shape = if need_size_all {
            UtilityCursor::SizeAll
        } else if (dragging[0] != 0 && dragging[2] != 0) || (dragging[1] != 0 && dragging[3] != 0) {
            UtilityCursor::SizeForwardDiagonal
        } else if (dragging[1] != 0 && dragging[2] != 0) || (dragging[0] != 0 && dragging[3] != 0) {
            UtilityCursor::SizeBackwardDiagonal
        } else if dragging[0] != 0 || dragging[1] != 0 {
            UtilityCursor::SizeHorizontal
        } else if dragging[2] != 0 || dragging[3] != 0 {
            UtilityCursor::SizeVertical
        } else {
            // C starts with the previously installed shape if no edge is
            // supplied; this can occur while a band is being cleared.
            state.last_special.unwrap_or(UtilityCursor::Default)
        };
        let changed = state.last_special != Some(shape) || set_anyway;
        state.last_special = Some(shape);
        return changed.then_some(shape);
    }
    if state.mouse_mode == mode && state.last_special.is_none() && !set_anyway {
        return None;
    }
    state.mouse_mode = mode;
    state.last_special = None;
    Some(
        if mode == crate::imod::three_dmod::imodview::IMOD_MMODEL || need_model {
            UtilityCursor::Model
        } else {
            UtilityCursor::Default
        },
    )
}

/// `utilClearWindow`.
pub fn util_clear_window(boundary: &mut dyn UtilitiesBoundary, index: i32) {
    boundary.clear_window(index)
}

/// `utilMouseZaxisRotation`.
pub fn util_mouse_zaxis_rotation(
    winx: i32,
    mx: i32,
    lastmx: i32,
    winy: i32,
    my: i32,
    lastmy: i32,
) -> f32 {
    let (xcen, ycen) = (winx / 2, winy / 2);
    let (mut dx, mut dy) = ((lastmx - xcen) as f64, (winy - 1 - lastmy - ycen) as f64);
    if dx.abs() <= 20. && dy.abs() <= 20. {
        return 0.;
    }
    let start = dy.atan2(dx).to_degrees();
    dx = (mx - xcen) as f64;
    dy = (winy - 1 - my - ycen) as f64;
    if dx.abs() <= 20. && dy.abs() <= 20. {
        return 0.;
    }
    let mut rotation = dy.atan2(dx).to_degrees() - start;
    if rotation < -360. {
        rotation += 360.;
    }
    if rotation > 360. {
        rotation -= 360.;
    }
    rotation as f32
}

/// `utilSetObjFlag`.
pub fn util_set_obj_flag(obj: Option<&mut Iobj>, flag_type: i32, state: bool, flag: u32) {
    let Some(obj) = obj else { return };
    if !(0..=1).contains(&flag_type) {
        return;
    }
    let flags = if flag_type == 1 {
        &mut obj.symflags
    } else {
        return if state {
            obj.flags |= flag
        } else {
            obj.flags &= !flag
        };
    };
    if state {
        *flags |= flag as u8
    } else {
        *flags &= !(flag as u8)
    }
}

/// `utilNextSecWithCont`.
pub fn util_next_sec_with_cont(view: &ImodView, obj: Option<&Iobj>, curz: i32, dir: i32) -> i32 {
    let Some(obj) = obj else { return curz };
    let mut newz = -1;
    for cont in &obj.cont {
        if cont.pts.is_empty()
            || (obj.flags & crate::imod::libimod::iobj::IMOD_OBJFLAG_TIME != 0
                && view.num_times > 1
                && cont.time != view.cur_time)
        {
            continue;
        }
        let lim = if iobj_scat(obj.flags) != 0 {
            cont.pts.len()
        } else {
            1
        };
        for point in cont.pts.iter().take(lim) {
            let z = (point.z.round() as i32).clamp(0, view.zsize - 1);
            let diff = dir * (z - curz);
            if diff > 0 && (newz < 0 || diff < dir * (newz - curz)) {
                newz = z;
            }
        }
    }
    if newz >= 0 { newz } else { curz }
}

/// `utilWheelToPointSizeScaling`.
pub fn util_wheel_to_point_size_scaling(zoom: f32) -> f32 {
    let mut scale = 1. / 1200.;
    if zoom < 4. && zoom >= 2. {
        scale *= 2.;
    } else if zoom < 2. && zoom > 1. {
        scale *= 3.;
    } else if zoom == 1. {
        scale *= 4.;
    } else if zoom < 1. {
        scale *= 5.;
    }
    scale
}

/// `utilWheelChangePointSize`, with selected point supplied directly by input code.
pub fn util_wheel_change_point_size(
    boundary: &mut dyn UtilitiesBoundary,
    cont: &mut Icont,
    point: usize,
    zoom: f32,
    delta: i32,
) {
    if point >= cont.pts.len() {
        return;
    }
    let size = cont.sizes.get(point).copied().unwrap_or(-1.);
    if size == 0. || size < 0. {
        return;
    }
    if cont.sizes.len() <= point {
        cont.sizes.resize(point + 1, -1.);
    }
    cont.sizes[point] = (size + delta as f32 * util_wheel_to_point_size_scaling(zoom)).max(0.);
    boundary.change_point_size();
    boundary.finish_undo_unit();
    boundary.redraw_model();
}

/// `utilIsBandCommitted`.
pub fn util_is_band_committed(
    x: i32,
    y: i32,
    win_x: i32,
    win_y: i32,
    bandmin: i32,
    rb: &mut [i32; 4],
    dragging: &mut [i32; 4],
) -> i32 {
    let (mut x, mut y) = (x, y);
    let (mut dx, mut dy) = (x - rb[0], y - rb[2]);
    let (mut ax, mut ay) = (dx.abs(), dy.abs());
    if dy == 0 && ax >= 6 * bandmin {
        y = rb[2] + if y < win_y / 2 { bandmin } else { -bandmin };
        dy = y - rb[2];
        ay = bandmin;
    } else if dx == 0 && ay >= 6 * bandmin {
        x = rb[0] + if x < win_x / 2 { bandmin } else { -bandmin };
        dx = x - rb[0];
        ax = bandmin;
    }
    if !((ax >= bandmin && ay >= bandmin)
        || (ax >= 3 * bandmin && ay >= (bandmin / 2).max(1))
        || (ay >= 3 * bandmin && ax >= (bandmin / 2).max(1)))
    {
        return 0;
    }
    *dragging = [0; 4];
    if x > rb[0] {
        dragging[1] = 1;
        rb[1] = x
    } else {
        dragging[0] = 1;
        rb[1] = rb[0];
        rb[0] = x;
    }
    if y > rb[2] {
        dragging[3] = 1;
        rb[3] = y
    } else {
        dragging[2] = 1;
        rb[3] = rb[2];
        rb[2] = y;
    }
    1
}

/// `utilAnalyzeBandEdge`.
pub fn util_analyze_band_edge(
    ix: i32,
    iy: i32,
    rb: [i32; 4],
    drag_band: &mut i32,
    dragging: &mut [i32; 4],
) {
    let (x0, x1, y0, y1) = (rb[0], rb[1], rb[2], rb[3]);
    *drag_band = 0;
    *dragging = [0; 4];
    let mut best = 100;
    let mut edge = (-1, -1);
    // Preserve the four source tests' order.  These are `<`, not `<=`, so
    // a cursor exactly equidistant from two corners retains the earlier one.
    for (xedge, yedge, xe, ye) in [
        (x0, y0, 0, 2),
        (x1, y0, 1, 2),
        (x0, y1, 0, 3),
        (x1, y1, 1, 3),
    ] {
        let d = (ix - xedge).pow(2) + (iy - yedge).pow(2);
        if d < best {
            best = d;
            edge = (xe, ye)
        }
    }
    if edge.0 >= 0 {
        *drag_band = 1;
        dragging[edge.0 as usize] = 1;
        dragging[edge.1 as usize] = 1;
        return;
    }
    let mut min = 10;
    let mut found = -1;
    if iy > y0 && iy < y1 && (ix - x0).abs() < min {
        min = (ix - x0).abs();
        found = 0
    };
    if iy > y0 && iy < y1 && (ix - x1).abs() < min {
        min = (ix - x1).abs();
        found = 1
    };
    if ix > x0 && ix < x1 && (iy - y0).abs() < min {
        min = (iy - y0).abs();
        found = 2
    };
    if ix > x0 && ix < x1 && (iy - y1).abs() < min {
        found = 3
    };
    if found >= 0 {
        *drag_band = 1;
        dragging[found as usize] = 1
    }
}

/// `utilTestBandMove`.
pub fn util_test_band_move(x: i32, y: i32, rb: [i32; 4]) -> i32 {
    let (dx0, dx1, dy0, dy1) = (x - rb[0], x - rb[1], y - rb[2], y - rb[3]);
    (((dy0 > 0 && dy1 < 0) && ((dx0).abs() < 10 || (dx1).abs() < 10))
        || ((dx0 > 0 && dx1 < 0) && ((dy0).abs() < 10 || (dy1).abs() < 10))) as i32
}

/// `utilExchangeFlipRotation`.
pub fn util_exchange_flip_rotation(
    boundary: &mut dyn UtilitiesBoundary,
    imod: &mut Imod,
    direction: i32,
) {
    if (direction == FLIP_TO_ROTATION && imod.flags & IMODF_FLIPYZ == 0)
        || (direction == ROTATION_TO_FLIP && imod.flags & IMODF_ROT90X == 0)
    {
        return;
    }
    if direction == FLIP_TO_ROTATION {
        boundary.flip_yz(imod);
        boundary.rotate_90_x(imod, false);
        imod.flags |= IMODF_ROT90X;
        imod.flags &= !IMODF_FLIPYZ;
    } else {
        boundary.rotate_90_x(imod, true);
        boundary.flip_yz(imod);
        imod.flags &= !IMODF_ROT90X;
        imod.flags |= IMODF_FLIPYZ;
    }
}

/// `utilLookupPopupHit`.
pub fn util_lookup_popup_hit(
    index: usize,
    specific: &[PopupEntry],
    num_specific: isize,
) -> Option<(i32, bool, bool)> {
    let count = if num_specific < 0 {
        specific.len()
    } else {
        num_specific as usize
    };
    let entry = if index < count {
        specific.get(index)?
    } else {
        DEFAULT_ACTIONS.get(index - count)?
    };
    Some((entry.key, entry.ctrl, entry.shift))
}

/// `utilUnitZoomForDeviceScaling`.
pub fn util_unit_zoom_for_device_scaling(dev_pix_ratio: f32) -> f32 {
    if dev_pix_ratio >= 1.8 {
        2.
    } else if dev_pix_ratio >= 1.3 {
        1.5
    } else {
        1.
    }
}

/// `utilInitializeScreenChange`.  The source uses `App->DevicePixelRatio`
/// unless its Qt-only `WATCH_DPI_CHANGE` build path is selected; callers with
/// a winit window install the screen callback at their native host instead.
pub fn util_initialize_screen_change(app_device_pixel_ratio: f32) -> f32 {
    app_device_pixel_ratio
}

/// `utilGetNewDevPixRatio`.  `WATCH_DPI_CHANGE` is not enabled for this
/// platform; source therefore returns zero and leaves the current ratio alone.
pub fn util_get_new_dev_pix_ratio() -> f32 {
    0.
}

/// `utilSetZoomOnScreenChange`, with elapsed timer values supplied by the caller.
pub fn util_set_zoom_on_screen_change(
    new_size_change: f32,
    last_size_change: &mut f32,
    screen_changed: &mut bool,
    screen_elapsed: i32,
    resize_elapsed: i32,
    zoom: &mut f32,
    dev_pix_varies: bool,
) -> bool {
    let is_screen = new_size_change == 0.;
    let tiny = new_size_change > 0. && new_size_change > 0.99 && new_size_change < 1.01;
    if !dev_pix_varies {
        return false;
    };
    if !is_screen && !(tiny && resize_elapsed < 2000) {
        *last_size_change = new_size_change
    }
    if (*screen_changed && !is_screen && screen_elapsed < 2000)
        || (!*screen_changed && is_screen && resize_elapsed < 2000)
    {
        if is_screen {
            *screen_changed = true
        };
        if *last_size_change < 0.9 || *last_size_change > 1.1 {
            *zoom *= *last_size_change;
            let zlog = zoom.log10();
            let power = zlog.floor();
            *zoom = 10_f32.powf(power - 1.) * (10_f32.powf(zlog - power) * 10.).round();
            *screen_changed = false;
            *last_size_change = 1.;
            return true;
        }
    } else {
        *screen_changed = is_screen
    };
    false
}

/// `imodwGivenName`.
pub fn imodw_given_name(intro: &str, filein: Option<&str>) -> Option<String> {
    let name = filein?.rsplit('/').next()?;
    if name.is_empty() {
        None
    } else {
        Some(format!(
            "{}{}{}",
            intro,
            if intro.is_empty() { "" } else { " " },
            name
        ))
    }
}
/// `imodwEithername`.
pub fn imodw_either_name(
    intro: &str,
    filein: Option<&str>,
    image_name: Option<&str>,
    model_first: bool,
) -> Option<String> {
    if model_first {
        imodw_given_name(intro, filein).or_else(|| imodw_given_name(intro, image_name))
    } else {
        imodw_given_name(intro, image_name).or_else(|| imodw_given_name(intro, filein))
    }
}
/// `imodCaption`.
pub fn imod_caption(intro: &str, image_name: Option<&str>) -> String {
    let prefix = if intro.is_empty() {
        String::new()
    } else {
        format!("{intro}:")
    };
    imodw_given_name(&prefix, image_name).unwrap_or_else(|| intro.into())
}
/// `utilManageBrowserDir`.
pub fn util_manage_browser_dir(browser_dir: &mut String, filename: &str, only_if_empty: bool) {
    if filename.is_empty() || (only_if_empty && !browser_dir.is_empty()) {
        return;
    }
    if let Some((dir, _)) = filename.rsplit_once('/') {
        *browser_dir = dir.to_owned()
    }
}

/// Qt's popup menu is intentionally a boundary.  These are the source's default rows
/// required by menu units; the full specialized table stays owned by its caller.
pub static DEFAULT_ACTIONS: &[PopupEntry] = &[
    PopupEntry {
        text: "Go to previous object",
        key: 'O' as i32,
        ctrl: false,
        shift: false,
        main_index: 0,
    },
    PopupEntry {
        text: "Go to next object",
        key: 'P' as i32,
        ctrl: false,
        shift: false,
        main_index: 0,
    },
    PopupEntry {
        text: "Go to previous contour",
        key: 'C' as i32,
        ctrl: false,
        shift: false,
        main_index: 0,
    },
    PopupEntry {
        text: "Go to next contour",
        key: 'C' as i32,
        ctrl: false,
        shift: true,
        main_index: 0,
    },
    PopupEntry {
        text: "Save model to file",
        key: 'S' as i32,
        ctrl: false,
        shift: false,
        main_index: 0,
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_mouse_rotation_and_band() {
        assert_eq!(util_mouse_zaxis_rotation(200, 100, 100, 200, 50, 50), 0.);
        let mut rb = [10, 0, 10, 0];
        let mut drag = [0; 4];
        assert_eq!(
            util_is_band_committed(30, 40, 100, 100, 5, &mut rb, &mut drag),
            1
        );
        assert_eq!((rb, drag), ([10, 30, 10, 40], [0, 1, 0, 1]));
    }
    #[test]
    fn source_title_and_zoom() {
        assert_eq!(
            imodw_given_name("3dmod:", Some("a/b.mrc")).as_deref(),
            Some("3dmod: b.mrc")
        );
        assert_eq!(util_unit_zoom_for_device_scaling(1.3), 1.5);
    }
    #[test]
    fn source_longest_time_label() {
        assert_eq!(
            util_get_longest_time_string(3, &["".into(), "1".into(), "long".into()]),
            " (999)long"
        );
    }

    #[test]
    fn platform_conditional_cursor_and_dpi_utilities_use_the_linux_source_path() {
        assert!(!util_need_to_set_cursor());
        assert!(!util_ignore_closing(false));
        assert_eq!(util_initialize_screen_change(1.5), 1.5);
        assert_eq!(util_get_new_dev_pix_ratio(), 0.);
    }

    #[test]
    fn band_corner_tie_keeps_the_source_first_corner() {
        let mut drag_band = 0;
        let mut dragging = [0; 4];
        // Equidistant from lower-left and lower-right.  C checks lower-left
        // first and strict comparison retains it.
        util_analyze_band_edge(5, 0, [0, 10, 0, 10], &mut drag_band, &mut dragging);
        assert_eq!(drag_band, 1);
        assert_eq!(dragging, [1, 0, 1, 0]);
    }
    #[test]
    fn montage_snapshot_buffers_chunk_rows_and_scale_bar_like_source() {
        let mut bar = ScaleBar {
            min_length: 5,
            thickness: 3,
            indent_x: 4,
            indent_y: 7,
            ..Default::default()
        };
        let (buffers, saved) = util_start_mont_snap(2, 3, 2, 3, 1.5, &mut bar).unwrap();
        assert_eq!(buffers.frame_pixels.len(), 24);
        assert_eq!(buffers.chunks.len(), 1);
        assert_eq!(buffers.chunks[0].len(), 24);
        assert_eq!(buffers.line_chunk_offsets, [(0, 0), (0, 8), (0, 16)]);
        assert_eq!(
            (
                bar.min_length,
                bar.thickness,
                bar.indent_x,
                bar.indent_y,
                bar.scale_label
            ),
            (8, 5, 6, 11, 1.5)
        );
        assert_eq!(saved.min_length, 5);
    }
    #[test]
    fn montage_scale_bar_selects_exactly_one_source_corner() {
        let mut bar = ScaleBar {
            position: 0,
            draw: true,
            ..Default::default()
        };
        assert!(!util_mont_snap_scale_bar(&mut bar, 0, 0, 2, true));
        assert!(!bar.draw);
        assert!(util_mont_snap_scale_bar(&mut bar, 1, 0, 2, true));
        assert!(bar.draw);
    }
    #[test]
    fn cursor_selection_keeps_source_drag_edge_precedence_and_cache() {
        let mut state = UtilityCursorState::default();
        assert_eq!(
            util_set_cursor(0, false, true, false, [1, 0, 1, 0], false, &mut state),
            Some(UtilityCursor::SizeForwardDiagonal)
        );
        assert_eq!(
            util_set_cursor(0, false, true, false, [1, 0, 1, 0], false, &mut state),
            None
        );
        assert_eq!(
            util_set_cursor(0, false, true, false, [0, 1, 1, 0], false, &mut state),
            Some(UtilityCursor::SizeBackwardDiagonal)
        );
        assert_eq!(
            util_set_cursor(
                crate::imod::three_dmod::imodview::IMOD_MMODEL,
                false,
                false,
                false,
                [0; 4],
                false,
                &mut state
            ),
            Some(UtilityCursor::Model)
        );
    }
    #[test]
    fn snapshot_pre_changes_hide_and_restore_the_current_point() {
        let mut view = ImodView::default();
        view.drawcursor = 1;
        let bar = ScaleBar {
            draw_on_snapshots: true,
            ..Default::default()
        };
        let changes = util_pre_snap_changes(&bar, Some(&mut view), true);
        assert_eq!(view.drawcursor, 0);
        assert_eq!(
            changes,
            SnapshotPreChanges {
                saved_cursor: Some(1),
                set_scale_bar_without_dialog: true
            }
        );
        assert!(util_restore_snap_changes(Some(&mut view), changes));
        assert_eq!(view.drawcursor, 1);
    }
}
