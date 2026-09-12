//! Translation of `IMOD/3dmod/control.cpp`, `control.h`, and `controlP.h`.
//!
//! The source owns two independent mechanisms: a priority-ordered list of
//! drawing controls in every `ViewInfo`, and a manager which groups Qt
//! dialogs.  Qt widgets, timer delivery, desktop geometry, and native window
//! stacking are represented by `DialogManagerNativeBoundary`; no replacement
//! window system is introduced here.
#![allow(dead_code)]

use crate::imod::three_dmod::imodview::ImodView;
use crate::imod::three_dmod::mv_window::KeyEvent;

pub const IMODV_DIALOG: i32 = 0;
pub const IMOD_DIALOG: i32 = 1;
pub const IMOD_IMAGE: i32 = 2;
pub const ZAP_WINDOW_TYPE: i32 = 0;
pub const MULTIZ_WINDOW_TYPE: i32 = 1;
pub const SLICER_WINDOW_TYPE: i32 = 2;
pub const XYZ_WINDOW_TYPE: i32 = 3;
pub const GRAPH_WINDOW_TYPE: i32 = 4;
pub const TUMBLER_WINDOW_TYPE: i32 = 5;
pub const DOCKING_DIALOG_TYPE: i32 = 6;
pub const UNKNOWN_TYPE: i32 = 7;
pub const IMOD_DRAW_TOP: i32 = 1 << 8;
pub const IMOD_DRAW_ACTIVE: i32 = 1 << 9;
pub const DOCKER_CHANGE_TIMEOUT: i32 = 300;
pub const INFO_SHOW_HIDE_TIMEOUT: i32 = 100;

pub type ImodControlProc = fn(&mut ImodView, usize, i32);
pub type ImodControlKey = fn(&mut ImodView, usize, i32, &KeyEvent);

/// `ImodControl` (`controlP.h`).  `user_data` represents the untyped C
/// pointer as an opaque address-sized value.
#[derive(Clone, Copy)]
pub struct ImodControl {
    pub user_data: usize,
    pub draw_cb: ImodControlProc,
    pub close_cb: ImodControlProc,
    pub key_cb: Option<ImodControlKey>,
    pub id: i32,
    pub status: i32,
}

/// `ImodControlList` (`controlP.h`).
#[derive(Default)]
pub struct ImodControlList {
    pub list: Vec<ImodControl>,
    pub active: i32,
    pub top: i32,
    pub reason: i32,
    pub work_id: i32,
    pub current: usize,
    pub control_timer_running: bool,
    pub display_busy: i32,
    pub clear_movie_on_stop: bool,
}

/// `ivwNewControl`.
pub fn ivw_new_control(
    iv: &mut ImodView,
    draw_cb: ImodControlProc,
    quit_cb: ImodControlProc,
    key_cb: Option<ImodControlKey>,
    data: usize,
) -> i32 {
    static NEXT_CONTROL_ID: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);
    let id = NEXT_CONTROL_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    let list = iv.ctrlist.get_or_insert_with(ImodControlList::default);
    list.top = id;
    list.active = id;
    list.list.push(ImodControl {
        user_data: data,
        draw_cb,
        close_cb: quit_cb,
        key_cb,
        id,
        status: 0,
    });
    id
}

/// `removeControl`.
pub fn remove_control(iv: &mut ImodView, in_ctrl_id: i32, call_close: bool) -> i32 {
    ivw_control_list_draw_cancel(iv);
    let Some(list) = iv.ctrlist.as_mut() else {
        return 0;
    };
    if list.list.is_empty() {
        return 0;
    }
    list.active = 0;
    let Some(index) = list
        .list
        .iter()
        .position(|control| control.id == in_ctrl_id)
    else {
        return 1;
    };
    let control = list.list.remove(index);
    if let Some(first) = list.list.first() {
        list.top = first.id;
    }
    if call_close {
        (control.close_cb)(iv, control.user_data, 0);
    }
    0
}

/// `ivwRemoveControl`.
pub fn ivw_remove_control(iv: &mut ImodView, in_ctrl_id: i32) -> i32 {
    remove_control(iv, in_ctrl_id, false)
}
/// `ivwDeleteControl`.
pub fn ivw_delete_control(iv: &mut ImodView, in_ctrl_id: i32) -> i32 {
    remove_control(iv, in_ctrl_id, true)
}

/// `ivwControlPriority`.
pub fn ivw_control_priority(iv: &mut ImodView, in_ctrl_id: i32) -> i32 {
    let Some(list) = iv.ctrlist.as_mut() else {
        return 0;
    };
    list.active = in_ctrl_id;
    if in_ctrl_id == 0 || list.top == in_ctrl_id {
        return if in_ctrl_id == 0 {
            list.top
        } else {
            in_ctrl_id
        };
    }
    list.control_timer_running = false;
    if let Some(index) = list
        .list
        .iter()
        .position(|control| control.id == in_ctrl_id)
    {
        let control = list.list.remove(index);
        list.list.insert(0, control);
        list.top = in_ctrl_id;
        return 0;
    }
    list.top
}
/// `ivwControlActive`.
pub fn ivw_control_active(iv: &mut ImodView, in_ctrl_id: i32) {
    if let Some(list) = &mut iv.ctrlist {
        list.active = in_ctrl_id;
    }
}
/// `ivwControlDraw`.
pub fn ivw_control_draw(iv: &mut ImodView, reason: i32, in_ctrl_id: i32) {
    ivw_control_list_draw_cancel(iv);
    ivw_control_priority(iv, in_ctrl_id);
    ivw_control_list_draw(iv, reason);
}
/// `stopControlTimer`.
pub fn stop_control_timer(iv: &mut ImodView) {
    if let Some(list) = &mut iv.ctrlist {
        list.control_timer_running = false;
        if list.clear_movie_on_stop {
            list.display_busy = (list.display_busy - 1).max(0);
        }
        list.clear_movie_on_stop = false;
    }
}
/// `ivwControlListDrawCancel`.
pub fn ivw_control_list_draw_cancel(iv: &mut ImodView) {
    if iv.ctrlist.is_some() {
        stop_control_timer(iv);
    }
}
/// `ivwWorkProc`.
pub fn ivw_work_proc(iv: &mut ImodView) {
    let control = {
        let Some(list) = iv.ctrlist.as_mut() else {
            return;
        };
        if list.current >= list.list.len() {
            stop_control_timer(iv);
            return;
        }
        let out = list.list[list.current];
        list.current += 1;
        out
    };
    (control.draw_cb)(iv, control.user_data, iv.ctrlist.as_ref().unwrap().reason);
}
/// `ivwControlListDraw`.
pub fn ivw_control_list_draw(iv: &mut ImodView, reason: i32) {
    if iv.ctrlist.as_ref().is_none_or(|list| list.list.is_empty()) {
        return;
    }
    ivw_control_list_draw_cancel(iv);
    let (control, flags) = {
        let list = iv.ctrlist.as_mut().unwrap();
        list.reason = reason;
        list.current = 1;
        let control = list.list[0];
        let flags = reason
            | IMOD_DRAW_TOP
            | if control.id == list.active {
                list.active = 0;
                IMOD_DRAW_ACTIVE
            } else {
                0
            };
        if list.display_busy > 0 {
            list.display_busy += 1;
            list.clear_movie_on_stop = true;
        }
        list.control_timer_running = true;
        (control, flags)
    };
    (control.draw_cb)(iv, control.user_data, flags);
}
/// `ivwControlListDelete`.
pub fn ivw_control_list_delete(iv: &mut ImodView) {
    if let Some(list) = iv.ctrlist.take() {
        for control in list.list {
            (control.close_cb)(iv, control.user_data, 0);
        }
    }
}
/// `ivwControlKey` with the source global `App->cvi` made an explicit view argument.
pub fn ivw_control_key(iv: &mut ImodView, released: i32, event: &KeyEvent) {
    if let Some(control) = iv
        .ctrlist
        .as_ref()
        .and_then(|list| list.list.iter().find(|control| control.key_cb.is_some()))
        .copied()
    {
        (control.key_cb.unwrap())(iv, control.user_data, released, event);
    }
}

/// Source-compatible geometry form of `QRect`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}
impl Rect {
    pub fn right(self) -> i32 {
        self.x + self.width
    }
    pub fn bottom(self) -> i32 {
        self.y + self.height
    }
    pub fn intersects(self, b: Self) -> bool {
        self.x < b.right() && self.right() > b.x && self.y < b.bottom() && self.bottom() > b.y
    }
}

/// Calls crossing from `DialogManager` to Qt/preferences/application globals.
pub trait DialogManagerNativeBoundary {
    fn window_icon(&mut self, _widget: usize, _dialog_class: i32) {}
    fn close(&mut self, _widget: usize) {}
    fn hidden(&self, _widget: usize) -> bool {
        false
    }
    fn minimized(&self, _widget: usize) -> bool {
        false
    }
    fn hide(&mut self, _widget: usize) {}
    fn show_normal(&mut self, _widget: usize) {}
    fn raise(&mut self, _widget: usize) {}
    fn move_window(&mut self, _widget: usize, _x: i32, _y: i32) {}
    fn geometry(&self, _widget: usize) -> Rect {
        Rect::default()
    }
    fn frame_geometry(&self, widget: usize) -> Rect {
        self.geometry(widget)
    }
    fn stack_master_window(&self, _class: i32) -> Option<usize> {
        None
    }
    fn desktop_geometry(&self, _widget: usize) -> Rect {
        Rect {
            x: 0,
            y: 0,
            width: 1920,
            height: 1080,
        }
    }
    fn dialog_frame_adjustment(&self) -> i32 {
        0
    }
    fn changing_frame_adjustment(&self) -> bool {
        false
    }
    fn iconify_together(&self, _class: i32) -> bool {
        true
    }
    fn stack_dialogs(&self, _class: i32) -> bool {
        false
    }
    fn raise_stack(&self, _class: i32) -> bool {
        false
    }
    fn keep_stack_on_top(&self) -> bool {
        false
    }
    fn master_staying_on_top(&self) -> bool {
        false
    }
    fn set_stay_on_top(&mut self, _widget: usize, _keep: bool) {}
    fn start_dock_timer(&mut self, _class: i32, _timeout: i32) {}
    fn info_input(&mut self) {}
    fn exit_when_all_closed(&self) -> i32 {
        0
    }
    fn imodv_closed(&self) -> bool {
        false
    }
    fn app_closing(&self) -> bool {
        false
    }
    fn ask_exit_when_all_closed(&mut self) -> i32 {
        0
    }
    fn set_exit_when_all_closed(&mut self, _value: i32) {}
}

/// Source predicates over `ZapFuncs` and `SlicerFuncs` used only by the
/// third `getTopWindow` overload.
pub trait DialogWindowSelectionBoundary {
    /// Implements `ctrlPtr->id == slicer->mCtrl` plus the source's starting
    /// or present rubberband condition.
    fn slicer_matches(&self, _widget: usize, _control_id: i32, _with_band: bool) -> bool {
        false
    }
    /// Implements the `ZapFuncs` control/rubberband/lasso and image-edge
    /// checks in `control.cpp`.
    fn zap_matches(
        &self,
        _widget: usize,
        _control_id: i32,
        _with_band: bool,
        _with_lasso: bool,
    ) -> bool {
        false
    }
}

/// State which was file-static (`sModelViewActive`, `sNeedModvActive`,
/// `sActivatingWidget`, and `sSwitchBusy`) in `control.cpp`.  Keeping it in
/// the application object avoids a process-global mutable singleton.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MacMenuState {
    pub model_view_active: bool,
    pub need_modv_active: bool,
    pub activating_widget: Option<usize>,
    pub switch_busy: bool,
}
/// Native calls made by `ivwCheckAndSetMacMenu` / `ivwSwitchMacMenus` and
/// `ivwGetRoundedStyle`.
pub trait MacMenuNativeBoundary {
    fn activation_change(&self) -> bool {
        false
    }
    fn active_window(&self, _widget: usize) -> bool {
        false
    }
    fn start_menu_switch_timer(&mut self) {}
    fn activate_model_view(&mut self) {}
    fn activate_info_window(&mut self) {}
    fn imod_info_input(&mut self) {}
    fn activate_window(&mut self, _widget: usize) {}
    fn rounded_style(&self) -> bool {
        false
    }
}
/// `ivwCheckAndSetMacMenu` (its body is active only in the source's Qt/macOS build).
pub fn ivw_check_and_set_mac_menu(
    state: &mut MacMenuState,
    activation_change: bool,
    widget: usize,
    dialog_type: i32,
    n: &mut dyn MacMenuNativeBoundary,
) {
    if !activation_change || state.switch_busy || !n.active_window(widget) {
        return;
    }
    state.need_modv_active = dialog_type == IMODV_DIALOG;
    state.activating_widget = Some(widget);
    if state.model_view_active != state.need_modv_active {
        state.switch_busy = true;
        n.start_menu_switch_timer();
    }
}
/// `ivwSwitchMacMenus`.
pub fn ivw_switch_mac_menus(state: &mut MacMenuState, n: &mut dyn MacMenuNativeBoundary) {
    if state.need_modv_active {
        n.activate_model_view();
    } else {
        n.activate_info_window();
    }
    state.model_view_active = state.need_modv_active;
    n.imod_info_input();
    if let Some(widget) = state.activating_widget {
        n.activate_window(widget);
    }
    state.switch_busy = false;
}
/// `ivwModvMenuIsActive`.
pub fn ivw_modv_menu_is_active(state: &mut MacMenuState, value: bool) {
    state.model_view_active = value;
}
/// `ivwGetRoundedStyle`.
pub fn ivw_get_rounded_style(n: &dyn MacMenuNativeBoundary) -> bool {
    n.rounded_style()
}

/// `imod_dialog`.
#[derive(Clone, Debug)]
pub struct ImodDialog {
    pub widget: usize,
    pub iconified: i32,
    pub dlg_class: i32,
    pub dlg_type: i32,
    pub ctrl_id: i32,
    pub stack_index: i32,
    pub position: (i32, i32),
}
/// `DialogManager`.
#[derive(Debug)]
pub struct DialogManager {
    pub dialog_list: Vec<ImodDialog>,
    pub last_zap_geom: Rect,
    pub docked_dia_ind: Vec<usize>,
    pub stack_col_right: Vec<i32>,
    pub stack_col_left: Vec<i32>,
    pub col_start_ind: Vec<i32>,
    pub new_dlg_indexes: Vec<usize>,
    pub col_is_split: Vec<bool>,
    pub hidden_dlg_list_inds: Vec<usize>,
    pub shown_dlg_list_inds: Vec<usize>,
    pub stack_row_dir: i32,
    pub moved_dlg_index: i32,
    pub dlg_class: i32,
    pub restacking: bool,
    pub restoring_stack: bool,
    pub next_docker_state: i32,
    pub keep_stack_on_top: bool,
    pub hiding_or_showing: bool,
    pub info_hiding: bool,
    pub info_showing: bool,
}
impl DialogManager {
    /// `DialogManager::DialogManager`.
    pub fn new(dlg_class: i32) -> Self {
        Self {
            dialog_list: Vec::new(),
            last_zap_geom: Rect::default(),
            docked_dia_ind: Vec::new(),
            stack_col_right: Vec::new(),
            stack_col_left: Vec::new(),
            col_start_ind: Vec::new(),
            new_dlg_indexes: Vec::new(),
            col_is_split: Vec::new(),
            hidden_dlg_list_inds: Vec::new(),
            shown_dlg_list_inds: Vec::new(),
            stack_row_dir: 0,
            moved_dlg_index: -1,
            dlg_class,
            restacking: false,
            restoring_stack: false,
            next_docker_state: -1,
            keep_stack_on_top: false,
            hiding_or_showing: false,
            info_hiding: false,
            info_showing: false,
        }
    }
    /// `DialogManager::add`.
    pub fn add(
        &mut self,
        widget: usize,
        dlg_class: i32,
        dlg_type: i32,
        ctrl_id: i32,
        n: &mut dyn DialogManagerNativeBoundary,
    ) {
        let mut dia = ImodDialog {
            widget,
            iconified: 0,
            dlg_class,
            dlg_type,
            ctrl_id,
            stack_index: -2,
            position: (0, 0),
        };
        if dlg_type == DOCKING_DIALOG_TYPE && n.stack_dialogs(dlg_class) {
            if self.restoring_stack {
                dia.stack_index = 0;
            } else {
                self.new_dlg_indexes.push(self.dialog_list.len());
                self.start_dock_timer(dlg_class, 10, n);
            }
        }
        self.dialog_list.push(dia);
        n.window_icon(widget, dlg_class);
    }
    /// `DialogManager::remove`.
    pub fn remove(&mut self, widget: usize, n: &mut dyn DialogManagerNativeBoundary) {
        let Some(index) = self.dialog_list.iter().position(|d| d.widget == widget) else {
            return;
        };
        let dia = self.dialog_list.remove(index);
        if dia.dlg_type == ZAP_WINDOW_TYPE {
            self.last_zap_geom = n.geometry(widget);
        }
        if dia.stack_index >= 0 {
            self.docked_dia_ind.remove(dia.stack_index as usize);
            for ind in &mut self.docked_dia_ind {
                if *ind >= index {
                    *ind -= 1;
                }
            }
            self.restack_dialogs(n);
        }
        if dia.dlg_class == IMOD_IMAGE {
            self.check_for_exit_on_close(n);
        }
    }
    /// `DialogManager::close`.
    pub fn close(&mut self, n: &mut dyn DialogManagerNativeBoundary) {
        let list = core::mem::take(&mut self.dialog_list);
        self.docked_dia_ind.clear();
        for dia in list {
            n.close(dia.widget);
        }
    }
    /// `DialogManager::hide`.
    pub fn hide(&mut self, n: &mut dyn DialogManagerNativeBoundary) {
        self.start_dock_timer(self.dlg_class, INFO_SHOW_HIDE_TIMEOUT, n);
        self.info_hiding = true;
        self.info_showing = false;
    }
    /// `DialogManager::delayedHide`.
    pub fn delayed_hide(&mut self, n: &mut dyn DialogManagerNativeBoundary) {
        self.hiding_or_showing = true;
        for dia in &mut self.dialog_list {
            if n.hidden(dia.widget)
                || n.minimized(dia.widget)
                || (!n.iconify_together(dia.dlg_class) && dia.stack_index < 0)
            {
                dia.iconified = 0;
            } else {
                dia.iconified = 1;
                let r = n.geometry(dia.widget);
                dia.position = (r.x, r.y);
                n.hide(dia.widget);
            }
        }
        n.info_input();
        self.hiding_or_showing = false;
    }
    /// `DialogManager::show`.
    pub fn show(&mut self, n: &mut dyn DialogManagerNativeBoundary) {
        self.start_dock_timer(self.dlg_class, INFO_SHOW_HIDE_TIMEOUT, n);
        self.info_showing = true;
        self.info_hiding = false;
    }
    /// `DialogManager::delayedShow`.
    pub fn delayed_show(&mut self, n: &mut dyn DialogManagerNativeBoundary) {
        self.hiding_or_showing = true;
        for dia in &self.dialog_list {
            if dia.iconified != 0 {
                n.move_window(dia.widget, dia.position.0, dia.position.1);
                n.show_normal(dia.widget);
            }
        }
        n.info_input();
        self.hiding_or_showing = false;
    }
    /// `DialogManager::parent`.
    pub fn parent(&self, dlg_class: i32, n: &dyn DialogManagerNativeBoundary) -> Option<usize> {
        n.stack_master_window(dlg_class)
    }
    /// `DialogManager::stackMasterWindow`.
    pub fn stack_master_window(&self, n: &dyn DialogManagerNativeBoundary) -> Option<usize> {
        n.stack_master_window(self.dlg_class)
    }
    /// `DialogManager::raise`.
    pub fn raise(&self, dlg_class: i32, n: &mut dyn DialogManagerNativeBoundary) {
        for dia in &self.dialog_list {
            if dia.dlg_class == dlg_class && !n.hidden(dia.widget) {
                n.raise(dia.widget);
            }
        }
    }
    /// `DialogManager::windowCount`.
    pub fn window_count(&self, dlg_type: i32) -> i32 {
        self.dialog_list
            .iter()
            .filter(|dia| dia.dlg_type == dlg_type)
            .count() as i32
    }
    /// `DialogManager::checkForExitOnClose`.
    pub fn check_for_exit_on_close(&mut self, n: &mut dyn DialogManagerNativeBoundary) -> i32 {
        let exit = n.exit_when_all_closed();
        if self.dlg_class == IMODV_DIALOG
            || exit == 0
            || !n.imodv_closed()
            || n.app_closing()
            || self
                .dialog_list
                .iter()
                .any(|dia| dia.dlg_class == IMOD_IMAGE)
        {
            return 0;
        }
        if exit < 0 {
            let answer = n.ask_exit_when_all_closed();
            if answer == 0 {
                return 0;
            }
            if answer == 2 {
                n.set_exit_when_all_closed(1);
            }
        }
        self.start_dock_timer(IMOD_DIALOG, -10, n);
        1
    }
    /// `DialogManager::biggestGeometry`.
    pub fn biggest_geometry(&self, dlg_type: i32, n: &dyn DialogManagerNativeBoundary) -> Rect {
        let mut biggest = Rect::default();
        for dia in &self.dialog_list {
            if dia.dlg_type == dlg_type {
                let r = n.geometry(dia.widget);
                if r.width * r.height > biggest.width * biggest.height {
                    biggest = r;
                }
            }
        }
        if biggest.width == 0 {
            self.last_zap_geom
        } else {
            biggest
        }
    }
    /// `DialogManager::windowList`.
    pub fn window_list(&self, dlg_class: i32, dlg_type: i32) -> Vec<usize> {
        self.dialog_list
            .iter()
            .filter(|dia| {
                (dlg_type < 0 || dia.dlg_type == dlg_type)
                    && (dlg_class < 0 || dia.dlg_class == dlg_class)
            })
            .map(|dia| dia.widget)
            .collect()
    }
    /// `DialogManager::getTopWindow(int)`; control-priority ordering is supplied explicitly.
    pub fn get_top_window(&self, dlg_type: i32, controls: &ImodControlList) -> Option<usize> {
        self.get_top_window_two_types(dlg_type, dlg_type, controls)
            .0
    }
    /// `DialogManager::getTopWindow(int,int,int&)`.
    pub fn get_top_window_two_types(
        &self,
        dlg_type: i32,
        dlg_type2: i32,
        controls: &ImodControlList,
    ) -> (Option<usize>, i32) {
        for control in &controls.list {
            if let Some(dia) = self.dialog_list.iter().find(|dia| {
                (dia.dlg_type == dlg_type || dia.dlg_type == dlg_type2) && dia.ctrl_id == control.id
            }) {
                return (Some(dia.widget), dia.dlg_type);
            }
        }
        (None, -1)
    }
    /// `DialogManager::getTopWindow(bool,bool,int,int*)`.  Querying the
    /// concrete `ZapWindow`/`SlicerWindow` C++ classes is an explicit native
    /// boundary; control-list scanning and fallback selection remain here.
    pub fn get_top_window_with_band(
        &self,
        with_band: bool,
        with_lasso: bool,
        dlg_type: i32,
        controls: &ImodControlList,
        usable: &dyn DialogWindowSelectionBoundary,
    ) -> Option<(usize, usize)> {
        let windows = self.window_list(-1, dlg_type);
        if windows.is_empty() {
            return None;
        }
        for (control_index, control) in controls.list.iter().enumerate() {
            for &widget in &windows {
                let matched = if dlg_type == SLICER_WINDOW_TYPE {
                    usable.slicer_matches(widget, control.id, with_band)
                } else {
                    usable.zap_matches(widget, control.id, with_band, with_lasso)
                };
                if matched {
                    return Some((widget, control_index));
                }
            }
        }
        if with_band || with_lasso {
            None
        } else {
            Some((windows[0], 0))
        }
    }
    /// `DialogManager::dockerHasMoved`.
    pub fn docker_has_moved(&mut self, widget: usize, n: &mut dyn DialogManagerNativeBoundary) {
        let Some(index) = self.dialog_list.iter().position(|dia| dia.widget == widget) else {
            return;
        };
        let dia = &self.dialog_list[index];
        if dia.dlg_type != DOCKING_DIALOG_TYPE || dia.stack_index < -1 || self.restacking {
            return;
        }
        self.start_dock_timer(dia.dlg_class, DOCKER_CHANGE_TIMEOUT, n);
        self.moved_dlg_index = index as i32;
    }
    /// `DialogManager::dockerHasResized`.
    pub fn docker_has_resized(&mut self, widget: usize, n: &mut dyn DialogManagerNativeBoundary) {
        let Some(dia) = self.dialog_list.iter().find(|dia| dia.widget == widget) else {
            return;
        };
        if dia.dlg_type == DOCKING_DIALOG_TYPE && dia.stack_index >= 0 {
            self.start_dock_timer(dia.dlg_class, DOCKER_CHANGE_TIMEOUT, n);
        }
    }
    /// `DialogManager::masterWinHasChanged`.
    pub fn master_win_has_changed(
        &mut self,
        dlg_class: i32,
        n: &mut dyn DialogManagerNativeBoundary,
    ) {
        self.moved_dlg_index = -1;
        self.start_dock_timer(dlg_class, DOCKER_CHANGE_TIMEOUT, n);
    }
    /// `DialogManager::dockerWasHidden`.
    pub fn docker_was_hidden(&mut self, widget: usize, n: &mut dyn DialogManagerNativeBoundary) {
        if self.hiding_or_showing {
            return;
        }
        if let Some(index) = self
            .dialog_list
            .iter()
            .position(|dia| dia.widget == widget && dia.stack_index >= 0)
        {
            self.hidden_dlg_list_inds.push(index);
            self.start_dock_timer(self.dlg_class, 50, n);
        }
    }
    /// `DialogManager::dockerWasUnhidden`.
    pub fn docker_was_unhidden(&mut self, widget: usize, n: &mut dyn DialogManagerNativeBoundary) {
        if self.hiding_or_showing {
            return;
        }
        if let Some(index) = self.dialog_list.iter().position(|dia| dia.widget == widget) {
            self.shown_dlg_list_inds.push(index);
            self.start_dock_timer(self.dlg_class, 50, n);
        }
    }
    /// `DialogManager::dockerChangedScreen`.
    pub fn docker_changed_screen(
        &mut self,
        widget: usize,
        n: &mut dyn DialogManagerNativeBoundary,
    ) {
        let Some(index) = self.dialog_list.iter().position(|dia| dia.widget == widget) else {
            return;
        };
        let dia = &self.dialog_list[index];
        if dia.dlg_type == DOCKING_DIALOG_TYPE && dia.stack_index >= -1 && !self.restacking {
            self.start_dock_timer(dia.dlg_class, 3 * DOCKER_CHANGE_TIMEOUT, n);
            self.moved_dlg_index = index as i32;
        }
    }
    /// `DialogManager::masterChangedScreen`.
    pub fn master_changed_screen(&mut self, old_dpr: f32, new_dpr: f32) {
        if (old_dpr - new_dpr).abs() < 0.01 {
            return;
        }
        for dia in &mut self.dialog_list {
            if dia.stack_index >= 0 {
                dia.stack_index = -1;
            }
        }
        self.docked_dia_ind.clear();
    }
    /// `DialogManager::startDockTimer`.
    pub fn start_dock_timer(
        &self,
        dlg_class: i32,
        timeout: i32,
        n: &mut dyn DialogManagerNativeBoundary,
    ) {
        n.start_dock_timer(dlg_class, timeout);
    }
    /// `DialogManager::dockerWasActivated`.
    pub fn docker_was_activated(
        &self,
        widget: Option<usize>,
        n: &mut dyn DialogManagerNativeBoundary,
    ) {
        if !n.raise_stack(self.dlg_class) || !self.new_dlg_indexes.is_empty() || self.restacking {
            return;
        }
        if widget.is_some_and(|w| !self.dialog_list.iter().any(|dia| dia.widget == w))
            || (widget.is_none() && self.docked_dia_ind.is_empty())
        {
            return;
        }
        if let Some(master) = self.stack_master_window(n) {
            n.raise(master);
        }
        for &index in &self.docked_dia_ind {
            n.raise(self.dialog_list[index].widget);
        }
    }
    /// `DialogManager::manageStayingOnTop`.
    pub fn manage_staying_on_top(&mut self, n: &mut dyn DialogManagerNativeBoundary) {
        let keep = n.keep_stack_on_top() && n.master_staying_on_top();
        if keep == self.keep_stack_on_top {
            return;
        }
        self.keep_stack_on_top = keep;
        for &index in &self.docked_dia_ind {
            n.set_stay_on_top(self.dialog_list[index].widget, keep);
        }
    }
    /// `DialogManager::stackChangeTimeout`.  Qt screen/DPR-specific branches are
    /// represented by the caller's move notifications; insertion/removal state
    /// follows the source before rebuilding positions.
    pub fn stack_change_timeout(&mut self, n: &mut dyn DialogManagerNativeBoundary) {
        if self.info_hiding || self.info_showing {
            if self.info_hiding {
                self.delayed_hide(n);
            }
            if self.info_showing {
                self.delayed_show(n);
            }
            self.info_hiding = false;
            self.info_showing = false;
            return;
        }
        for index in core::mem::take(&mut self.hidden_dlg_list_inds) {
            if let Some(dia) = self.dialog_list.get_mut(index) {
                if dia.stack_index >= 0 && (dia.stack_index as usize) < self.docked_dia_ind.len() {
                    self.docked_dia_ind.remove(dia.stack_index as usize);
                    dia.stack_index = -1;
                }
            }
        }
        self.new_dlg_indexes.append(&mut self.shown_dlg_list_inds);
        if !self.new_dlg_indexes.is_empty() || self.restoring_stack {
            for index in core::mem::take(&mut self.new_dlg_indexes) {
                if let Some(dia) = self.dialog_list.get_mut(index) {
                    dia.stack_index = self.docked_dia_ind.len() as i32;
                    self.docked_dia_ind.push(index);
                    if self.keep_stack_on_top {
                        n.set_stay_on_top(dia.widget, true);
                    }
                }
            }
            self.restoring_stack = false;
        }
        self.restack_dialogs(n);
    }
    /// `DialogManager::restackDialogs`.
    pub fn restack_dialogs(&mut self, n: &mut dyn DialogManagerNativeBoundary) {
        if self.docked_dia_ind.is_empty() {
            return;
        }
        let Some(master) = self.stack_master_window(n) else {
            return;
        };
        let desktop = n.desktop_geometry(master);
        let master_rect = self.adjusted_geometry(master, n);
        let usable_left = desktop.x;
        let usable_top = desktop.y;
        let usable_right = desktop.right();
        let usable_bottom = desktop.bottom();
        self.stack_col_left.clear();
        self.stack_col_right.clear();
        self.col_is_split.clear();
        self.col_start_ind.clear();
        let mut max_width;
        if master_rect.x - usable_left < usable_right - master_rect.right() {
            self.stack_row_dir = 1;
            self.stack_col_left.push(master_rect.x);
            self.stack_col_right.push(master_rect.x);
            max_width = usable_right - master_rect.x;
        } else {
            self.stack_row_dir = -1;
            self.stack_col_left.push(master_rect.right());
            self.stack_col_right.push(master_rect.right());
            max_width = master_rect.right() - usable_left;
        }
        let mut rects = Vec::with_capacity(self.docked_dia_ind.len());
        let mut stack_ind = 0;
        while stack_ind < self.docked_dia_ind.len() {
            let index = self.docked_dia_ind[stack_ind];
            let rect = self.adjusted_geometry(self.dialog_list[index].widget, n);
            if rect.width > max_width || rect.height > usable_bottom - usable_top {
                self.dialog_list[index].stack_index = -1;
                self.docked_dia_ind.remove(stack_ind);
            } else {
                rects.push(rect);
                stack_ind += 1;
            }
        }
        if self.docked_dia_ind.is_empty() {
            return;
        }
        self.restacking = true;
        let mut col_ind = 0usize;
        self.col_is_split.push(true);
        self.col_start_ind.push(-1);
        let mut col_part = 0;
        let mut free_in_col = master_rect.y - usable_top;
        let mut first_dia_to_place = 0usize;
        stack_ind = 0;
        while stack_ind < self.docked_dia_ind.len() {
            let mut place_col_ind: Option<usize> = None;
            let mut next_dia_to_place = 0usize;
            let mut free_start = 0;
            while rects[stack_ind].height >= free_in_col {
                if self.col_is_split[col_ind] && col_part == 0 {
                    if place_col_ind.is_none() && stack_ind > first_dia_to_place {
                        place_col_ind = Some(col_ind);
                        free_start = usable_top + free_in_col;
                        next_dia_to_place = stack_ind;
                    }
                    col_part = 1;
                    free_in_col = usable_bottom - master_rect.bottom();
                } else if self.stack_col_left[col_ind] == self.stack_col_right[col_ind] {
                    self.stack_col_left[col_ind] = if self.stack_row_dir > 0 {
                        master_rect.right()
                    } else {
                        master_rect.x
                    };
                    self.stack_col_right[col_ind] = self.stack_col_left[col_ind];
                    self.col_is_split[col_ind] = false;
                    free_in_col = usable_bottom - usable_top;
                    col_part = 0;
                    max_width = if self.stack_row_dir > 0 {
                        usable_right - master_rect.right()
                    } else {
                        master_rect.x - usable_left
                    };
                } else {
                    if place_col_ind.is_none() && stack_ind > first_dia_to_place {
                        place_col_ind = Some(col_ind);
                        next_dia_to_place = stack_ind;
                        free_start = if self.col_is_split[col_ind] {
                            master_rect.bottom()
                        } else {
                            usable_top
                        };
                    }
                    let split = if self.col_is_split[col_ind] {
                        if self.stack_row_dir > 0 {
                            master_rect.right() - self.stack_col_right[col_ind]
                                > (master_rect.width as f32 * 0.33) as i32
                                || (stack_ind > first_dia_to_place
                                    && self.stack_col_right[col_ind] + rects[stack_ind].width
                                        - master_rect.right()
                                        < (master_rect.width as f32 * 0.3) as i32)
                        } else {
                            self.stack_col_left[col_ind] - master_rect.x
                                > (master_rect.width as f32 * 0.33) as i32
                                || (stack_ind > first_dia_to_place
                                    && master_rect.x
                                        - (self.stack_col_left[col_ind] - rects[stack_ind].width)
                                        < (master_rect.width as f32 * 0.3) as i32)
                        }
                    } else {
                        false
                    };
                    if !split && self.col_is_split[col_ind] {
                        if self.stack_row_dir > 0 {
                            self.stack_col_right[col_ind] =
                                self.stack_col_right[col_ind].max(master_rect.right());
                        } else {
                            self.stack_col_left[col_ind] =
                                self.stack_col_left[col_ind].min(master_rect.x);
                        }
                    }
                    let col_start = if self.stack_row_dir > 0 {
                        self.stack_col_right[col_ind]
                    } else {
                        self.stack_col_left[col_ind]
                    };
                    max_width = if self.stack_row_dir > 0 {
                        usable_right - col_start
                    } else {
                        col_start - usable_left
                    };
                    self.col_is_split.push(split);
                    self.stack_col_left.push(col_start);
                    self.stack_col_right.push(col_start);
                    self.col_start_ind.push(-1);
                    col_part = 0;
                    free_in_col = if split {
                        master_rect.y - usable_top
                    } else {
                        usable_bottom - usable_top
                    };
                    col_ind += 1;
                }
            }
            if rects[stack_ind].width > max_width {
                let index = self.docked_dia_ind.remove(stack_ind);
                self.dialog_list[index].stack_index = -1;
                rects.remove(stack_ind);
                continue;
            }
            let new_left = if self.stack_row_dir > 0 {
                let out = self.stack_col_left[col_ind];
                self.stack_col_right[col_ind] =
                    self.stack_col_right[col_ind].max(out + rects[stack_ind].width);
                out
            } else {
                let out = self.stack_col_right[col_ind] - rects[stack_ind].width;
                self.stack_col_left[col_ind] = self.stack_col_left[col_ind].min(out);
                out
            };
            if self.col_start_ind[col_ind] < 0 {
                self.col_start_ind[col_ind] = stack_ind as i32;
            }
            free_in_col -= rects[stack_ind].height;
            for final_column in [false, true] {
                if (!final_column && place_col_ind.is_none())
                    || (final_column && stack_ind < self.docked_dia_ind.len() - 1)
                {
                    continue;
                }
                let pcol = if final_column {
                    col_ind
                } else {
                    place_col_ind.unwrap()
                };
                let next = if final_column {
                    self.docked_dia_ind.len()
                } else {
                    next_dia_to_place
                };
                let mut top = if final_column {
                    if self.col_is_split[col_ind] {
                        if col_part > 0 {
                            master_rect.bottom()
                        } else {
                            usable_top + free_in_col
                        }
                    } else {
                        usable_top
                    }
                } else {
                    free_start
                };
                for place in first_dia_to_place..next {
                    let index = self.docked_dia_ind[place];
                    let left = if self.stack_row_dir > 0 {
                        self.stack_col_left[pcol]
                    } else {
                        self.stack_col_right[pcol] - rects[place].width
                    };
                    if top != rects[place].y
                        || left != rects[place].x
                        || n.changing_frame_adjustment()
                    {
                        n.move_window(
                            self.dialog_list[index].widget,
                            left - n.dialog_frame_adjustment(),
                            top,
                        );
                    }
                    self.dialog_list[index].stack_index = place as i32;
                    self.dialog_list[index].position = (left, top);
                    top += rects[place].height;
                }
                first_dia_to_place = next;
            }
            let _ = new_left;
            stack_ind += 1;
        }
        n.info_input();
        self.restacking = false;
    }
    /// `adjustedGeometry`.
    pub fn adjusted_geometry(&self, widget: usize, n: &dyn DialogManagerNativeBoundary) -> Rect {
        let adjust = n.dialog_frame_adjustment();
        let mut rect = n.frame_geometry(widget);
        if adjust != 0 {
            rect.x += adjust;
            rect.width -= 2 * adjust;
            rect.height -= adjust;
        }
        rect
    }
    /// `DialogManager::lookupDialogInList`.
    pub fn lookup_dialog_in_list(&self, widget: usize) -> Option<(usize, &ImodDialog)> {
        self.dialog_list
            .iter()
            .enumerate()
            .find(|(_, dia)| dia.widget == widget)
    }
    /// `DialogManager::summarizeStackState`; the DockingDialog key/state query is a Qt boundary, passed as a closure.
    pub fn summarize_stack_state(
        &self,
        mut key_state: impl FnMut(usize) -> (char, bool),
    ) -> (String, String) {
        let mut keys = String::new();
        let mut states = String::new();
        for &index in &self.docked_dia_ind {
            let (key, state) = key_state(self.dialog_list[index].widget);
            keys.push(key);
            states.push(if state { '1' } else { '0' });
        }
        (keys, states)
    }
    /// `DialogManager::saveStackStateToSettings`; settings storage is intentionally returned to its caller.
    pub fn save_stack_state_to_settings(
        &self,
        key_state: impl FnMut(usize) -> (char, bool),
    ) -> (String, String) {
        self.summarize_stack_state(key_state)
    }
    /// `DialogManager::restoreStackFromSettings`; source opening of missing Qt dialogs remains a caller boundary.
    pub fn restore_stack_from_settings(
        &mut self,
        keys: &str,
        states: &str,
        mut key_state: impl FnMut(usize) -> (char, bool),
        mut open_or_lookup: impl FnMut(char) -> Option<usize>,
        mut set_state: impl FnMut(usize, bool),
    ) {
        self.restoring_stack = true;
        let mut replacement = Vec::new();
        for (i, key) in keys.chars().enumerate() {
            let current = self
                .docked_dia_ind
                .iter()
                .copied()
                .find(|index| key_state(self.dialog_list[*index].widget).0 == key)
                .or_else(|| open_or_lookup(key));
            if let Some(current) = current {
                replacement.push(current);
                if let Some(state) = states.as_bytes().get(i) {
                    set_state(self.dialog_list[current].widget, *state == b'1');
                }
            } else {
                self.next_docker_state = if states.as_bytes().get(i) == Some(&b'1') {
                    1
                } else {
                    0
                };
            }
        }
        for index in &self.docked_dia_ind {
            if !replacement.contains(index) {
                replacement.push(*index);
            }
        }
        self.docked_dia_ind = replacement;
        self.next_docker_state = -1;
    }
}

/// `adjustGeometryAndShow`; Qt sizing/showing is explicit through the boundary.
pub fn adjust_geometry_and_show(
    widget: usize,
    dlg_class: i32,
    do_size: bool,
    manager: &DialogManager,
    n: &mut dyn DialogManagerNativeBoundary,
) {
    let _ = do_size;
    let mut rect = n.frame_geometry(widget);
    let desk = n.desktop_geometry(widget);
    rect.x = rect.x.clamp(desk.x, desk.right() - rect.width);
    rect.y = rect.y.clamp(desk.y, desk.bottom() - rect.height);
    n.move_window(widget, rect.x, rect.y);
    if let Some(parent) = manager.parent(dlg_class, n) {
        let p = n.frame_geometry(parent);
        if rect.intersects(p) {
            if rect.height <= desk.bottom() - p.bottom() {
                rect.y = p.bottom();
            } else if rect.height < p.y - 32 {
                rect.y = p.y - rect.height;
            } else if rect.width <= desk.right() - p.right() {
                rect.x = p.right();
            } else if rect.width < p.x {
                rect.x = p.x - rect.width;
            }
            n.move_window(widget, rect.x, rect.y);
        }
    }
    n.show_normal(widget);
}
/// `ivwRestorableGeometry`.
pub fn ivw_restorable_geometry(widget: usize, n: &dyn DialogManagerNativeBoundary) -> Rect {
    n.geometry(widget)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn draw(v: &mut ImodView, data: usize, flags: i32) {
        v.xmouse = (data as i32 + flags) as f32;
    }
    fn close(v: &mut ImodView, data: usize, _: i32) {
        v.ymouse = data as f32;
    }
    #[test]
    fn controls_follow_source_priority_and_active_flags() {
        let mut view = ImodView::default();
        let a = ivw_new_control(&mut view, draw, close, None, 2);
        let b = ivw_new_control(&mut view, draw, close, None, 3);
        assert_eq!(ivw_control_priority(&mut view, a), 0);
        ivw_control_list_draw(&mut view, 4);
        assert_eq!(
            view.xmouse,
            (2 + 4 + IMOD_DRAW_TOP + IMOD_DRAW_ACTIVE) as f32
        );
        ivw_work_proc(&mut view);
        assert_eq!(view.xmouse, (3 + 4) as f32);
        assert_eq!(ivw_delete_control(&mut view, b), 0);
        assert_eq!(view.ymouse, 3.);
    }
    #[test]
    fn dialog_manager_lists_by_type() {
        let mut manager = DialogManager::new(IMOD_DIALOG);
        let mut native = Native;
        manager.add(10, IMOD_DIALOG, ZAP_WINDOW_TYPE, 3, &mut native);
        manager.add(11, IMOD_IMAGE, SLICER_WINDOW_TYPE, 4, &mut native);
        assert_eq!(manager.window_count(ZAP_WINDOW_TYPE), 1);
        assert_eq!(manager.window_list(-1, SLICER_WINDOW_TYPE), vec![11]);
    }
    struct Native;
    impl DialogManagerNativeBoundary for Native {}
}
