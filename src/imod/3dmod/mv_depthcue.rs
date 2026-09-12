//! Translation of `IMOD/3dmod/mv_depthcue.cpp` together with `mv_depthcue.h`.
//!
//! The paired `formv_depthcue` unit owns the concrete dialog widgets.  Its
//! creation, dialog-manager registration, and presentation stay at the GUI
//! boundary below; all depth-cue/view mutations are retained in this unit.
#![allow(dead_code)]

use crate::imod::three_dmod::imodv::{
    ImodvApp, imodv_draw, imodv_finish_chg_unit, imodv_register_model_chg,
};
use crate::imod::three_dmod::mv_ogl::VIEW_WORLD_DEPTH_CUE;

/// Original `DEPTHCUE_MIN` (`mv_depthcue.h`).
pub const DEPTHCUE_MIN: i32 = 0;
/// Original `DEPTHCUE_MAX` (`mv_depthcue.h`).
pub const DEPTHCUE_MAX: i32 = 100;
/// Original initial `GL_LINEAR` value of `idcData.fmode`.
pub const GL_LINEAR: i32 = 0x2601;

/// Native Qt dialog-manager and paired-form operations used by this source
/// unit.  These are deliberately not replaced with a synthetic dialog.
pub trait DepthcueDialogBoundary {
    /// `imodShowHelpPage`.
    fn show_help_page(&mut self, page: &str);
    /// `imodvDepthcueForm::close`.
    fn close_depthcue_dialog(&mut self);
    /// `imodvDialogManager.remove`.
    fn remove_depthcue_dialog(&mut self);
    /// `imodvDepthcueForm::raise`.
    fn raise_depthcue_dialog(&mut self);
    /// `new imodvDepthcueForm`.
    fn create_depthcue_dialog(&mut self);
    /// `imodvDepthcueForm::setStates`.
    fn set_depthcue_states(&mut self, enabled: i32, start: i32, end: i32);
    /// `imodvDialogManager.add`.
    fn add_depthcue_dialog(&mut self);
    /// `adjustGeometryAndShow`.
    fn adjust_depthcue_geometry_and_show(&mut self);
}

/// Original file-static `idcData` structure.
///
/// `dia` represents the lifetime of the source `imodvDepthcueForm *`; its
/// concrete ownership belongs to the native form boundary.  `a` is retained
/// even though, exactly as upstream, no other function reads it here.
#[derive(Clone, Debug)]
pub struct ImodvDepthcueData {
    pub dia: bool,
    pub a: *mut ImodvApp,
    pub fstart: i32,
    pub fend: i32,
    pub fmode: i32,
    /// Function-static `sliding` from `imodvDepthcueStartEnd`.
    pub sliding: bool,
}

impl Default for ImodvDepthcueData {
    fn default() -> Self {
        Self {
            dia: false,
            a: std::ptr::null_mut(),
            fstart: DEPTHCUE_MIN,
            fend: 50,
            fmode: GL_LINEAR,
            sliding: false,
        }
    }
}

/// `imodvDepthcueHelp`.
pub fn imodv_depthcue_help(boundary: &mut dyn DepthcueDialogBoundary) {
    boundary.show_help_page("depthcue.html#TOP");
}

/// `imodvDepthCueSetWidgets`.
///
/// As in the source, this synchronizes `ImodvApp::depthcue` from the model
/// view even when no dialog is open.  Widget updates occur only while `dia`
/// is live.
pub fn imodv_depth_cue_set_widgets(
    a: &mut ImodvApp,
    data: &mut ImodvDepthcueData,
    boundary: &mut dyn DepthcueDialogBoundary,
) {
    let Some(imod) = (unsafe { a.imod.as_mut() }) else {
        return;
    };
    let Some(view) = imod.view.first() else {
        return;
    };
    a.depthcue = ((view.world & VIEW_WORLD_DEPTH_CUE) != 0) as i32;
    if data.dia {
        let mut fstart = (view.dcstart * 100.0) as i32;
        let mut fend = (view.dcend * 100.0) as i32;
        if fstart < DEPTHCUE_MIN {
            fstart = DEPTHCUE_MIN;
        }
        if fstart > DEPTHCUE_MAX {
            fstart = DEPTHCUE_MAX;
        }
        if fend < DEPTHCUE_MIN {
            fend = DEPTHCUE_MIN;
        }
        if fend > DEPTHCUE_MAX {
            fend = DEPTHCUE_MAX;
        }
        boundary.set_depthcue_states(a.depthcue, fstart, fend);
        data.fstart = fstart;
        data.fend = fend;
    }
}

/// `imodvDepthcueDone`.
pub fn imodv_depthcue_done(data: &ImodvDepthcueData, boundary: &mut dyn DepthcueDialogBoundary) {
    if data.dia {
        boundary.close_depthcue_dialog();
    }
}

/// `imodvDepthcueClosing`.
pub fn imodv_depthcue_closing(
    data: &mut ImodvDepthcueData,
    boundary: &mut dyn DepthcueDialogBoundary,
) {
    if data.dia {
        boundary.remove_depthcue_dialog();
    }
    data.dia = false;
}

/// `imodvDepthCueEditDialog`.
pub fn imodv_depth_cue_edit_dialog(
    a: &mut ImodvApp,
    state: i32,
    data: &mut ImodvDepthcueData,
    boundary: &mut dyn DepthcueDialogBoundary,
) {
    data.a = a;
    if state == 0 {
        if data.dia {
            boundary.close_depthcue_dialog();
        }
        return;
    }
    if data.dia {
        boundary.raise_depthcue_dialog();
        return;
    }
    boundary.create_depthcue_dialog();
    data.dia = true;
    imodv_depth_cue_set_widgets(a, data, boundary);
    boundary.add_depthcue_dialog();
    boundary.adjust_depthcue_geometry_and_show();
}

/// `imodvDepthcueStartEnd`.
pub fn imodv_depthcue_start_end(
    a: &mut ImodvApp,
    data: &mut ImodvDepthcueData,
    value: i32,
    end: bool,
    dragging: bool,
) {
    if !data.sliding {
        imodv_register_model_chg();
        imodv_finish_chg_unit();
    }
    data.sliding = dragging;
    if end {
        data.fend = value;
    } else {
        data.fstart = value;
    }
    if let Some(imod) = unsafe { a.imod.as_mut() } {
        if let Some(view) = imod.view.first_mut() {
            if end {
                view.dcend = value as f32 * 0.01;
            } else {
                view.dcstart = value as f32 * 0.01;
            }
        }
    }
    unsafe { imodv_draw() };
}

/// `imodvDepthcueToggle`.
pub fn imodv_depthcue_toggle(a: &mut ImodvApp, state: i32) {
    imodv_register_model_chg();
    imodv_finish_chg_unit();
    a.depthcue = state;
    if let Some(imod) = unsafe { a.imod.as_mut() } {
        if let Some(view) = imod.view.first_mut() {
            if state == 0 {
                view.world &= !VIEW_WORLD_DEPTH_CUE;
            } else {
                view.world |= VIEW_WORLD_DEPTH_CUE;
            }
        }
    }
    unsafe { imodv_draw() };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::libimod::imodel::{Imod, Iview};

    #[derive(Default)]
    struct Boundary {
        calls: Vec<String>,
        states: Vec<(i32, i32, i32)>,
    }

    impl DepthcueDialogBoundary for Boundary {
        fn show_help_page(&mut self, page: &str) {
            self.calls.push(format!("help:{page}"));
        }
        fn close_depthcue_dialog(&mut self) {
            self.calls.push("close".into());
        }
        fn remove_depthcue_dialog(&mut self) {
            self.calls.push("remove".into());
        }
        fn raise_depthcue_dialog(&mut self) {
            self.calls.push("raise".into());
        }
        fn create_depthcue_dialog(&mut self) {
            self.calls.push("create".into());
        }
        fn set_depthcue_states(&mut self, enabled: i32, start: i32, end: i32) {
            self.states.push((enabled, start, end));
        }
        fn add_depthcue_dialog(&mut self) {
            self.calls.push("add".into());
        }
        fn adjust_depthcue_geometry_and_show(&mut self) {
            self.calls.push("show".into());
        }
    }

    #[test]
    fn edit_dialog_clamps_model_values_and_tracks_the_source_lifetime() {
        let mut model = Box::new(Imod::default());
        let mut view = Iview::default();
        view.world = VIEW_WORLD_DEPTH_CUE;
        view.dcstart = -0.2;
        view.dcend = 1.4;
        model.view = vec![view];
        let mut app = ImodvApp {
            imod: &mut *model,
            ..ImodvApp::default()
        };
        let mut data = ImodvDepthcueData::default();
        let mut boundary = Boundary::default();
        imodv_depth_cue_edit_dialog(&mut app, 1, &mut data, &mut boundary);
        assert!(data.dia);
        assert_eq!(boundary.states, vec![(1, DEPTHCUE_MIN, DEPTHCUE_MAX)]);
        assert_eq!(boundary.calls, ["create", "add", "show"]);
        imodv_depth_cue_edit_dialog(&mut app, 1, &mut data, &mut boundary);
        assert_eq!(boundary.calls.last().unwrap(), "raise");
        imodv_depthcue_closing(&mut data, &mut boundary);
        assert!(!data.dia);
        assert_eq!(boundary.calls.last().unwrap(), "remove");
    }

    #[test]
    fn callbacks_update_depthcue_view_state() {
        let mut model = Box::new(Imod::default());
        model.view = vec![Iview::default()];
        let mut app = ImodvApp {
            imod: &mut *model,
            ..ImodvApp::default()
        };
        let mut data = ImodvDepthcueData::default();
        imodv_depthcue_start_end(&mut app, &mut data, 35, false, true);
        imodv_depthcue_start_end(&mut app, &mut data, 90, true, false);
        assert_eq!(model.view[0].dcstart, 0.35);
        assert_eq!(model.view[0].dcend, 0.9);
        imodv_depthcue_toggle(&mut app, 1);
        assert_ne!(model.view[0].world & VIEW_WORLD_DEPTH_CUE, 0);
        imodv_depthcue_toggle(&mut app, 0);
        assert_eq!(model.view[0].world & VIEW_WORLD_DEPTH_CUE, 0);
    }
}
