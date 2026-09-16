//! Translation of `IMOD/3dmod/slicer.cpp` and `sslice.h`.
//!
//! The Qt event loop, controller list, model editing, image cache and OpenGL
//! drawing APIs in the original are represented by [`SlicerNativeBoundary`].
//! Coordinate transforms, slice geometry, rubber-band coordinates, interpolation
//! and state transitions stay here, in the source translation unit.
#![allow(dead_code)]

use super::form_slicerangle::{SlicerAngleForm, SlicerAngleNativeBoundary};
use super::slicer_classes::{
    SLICER_LIMIT_INVALID, SLICER_LIMIT_TRUNCATE, SLICER_LIMIT_VALID, SLICER_TOGGLE_ARROW,
    SLICER_TOGGLE_BAND, SLICER_TOGGLE_CENTER, SLICER_TOGGLE_FFT, SLICER_TOGGLE_HIGHRES,
    SLICER_TOGGLE_LOCK, SLICER_TOGGLE_SHIFTLOCK, SLICER_TOGGLE_TIMELOCK, SLICER_TOGGLE_ZSCALE,
    SlicerCore, SlicerEvent,
};
use super::utilities::util_set_zoom_on_screen_change;
use crate::imod::libimod::icont::imod_contour_fit_plane;
use crate::imod::libimod::imodel::{Icont, Ipoint as ModelPoint};

pub const SLICE_ZSCALE_OFF: i32 = 0;
pub const SLICE_ZSCALE_BEFORE: i32 = 1;
pub const SLICE_ZSCALE_AFTER: i32 = 2;
pub const S_MAX_ANGLE: [f32; 3] = [90., 180., 180.];
pub const S_VIEW_AXIS_STEPS: [f32; 8] = [0.1, 0.3, 1., 3., 10., 30., 90., 0.];

/// Rust equivalent of IMOD's `Ipoint`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Ipoint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}
impl Ipoint {
    fn normalize(&mut self) {
        let len = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        if len != 0. {
            self.x /= len;
            self.y /= len;
            self.z /= len;
        }
    }
    fn cross(a: Self, b: Self) -> Self {
        Self {
            x: a.y * b.z - a.z * b.y,
            y: a.z * b.x - a.x * b.z,
            z: a.x * b.y - a.y * b.x,
        }
    }
}

/// The part of `ImodView` read or written by `slicer.cpp` proper.
#[derive(Clone, Debug)]
pub struct SlicerView {
    pub xsize: i32,
    pub ysize: i32,
    pub zsize: i32,
    pub xybin: i32,
    pub zbin: i32,
    pub zscale: f32,
    pub xmouse: f32,
    pub ymouse: f32,
    pub zmouse: f32,
    pub cur_time: i32,
    pub num_times: i32,
    pub track_mouse_for_plugs: i32,
    /// `ImodView::{xmovie,ymovie,zmovie}` plus whether this slicer's control
    /// started the current movie (`imcGetStarterID() == mCtrl`).
    pub xmovie: i32,
    pub ymovie: i32,
    pub zmovie: i32,
    pub movie_started_by_slicer: bool,
    /// `mVi->li->axis`; 3 is the ordinary Z-axis input orientation.
    pub image_axis: i32,
}
impl Default for SlicerView {
    fn default() -> Self {
        Self {
            xsize: 1,
            ysize: 1,
            zsize: 1,
            xybin: 1,
            zbin: 1,
            zscale: 1.,
            xmouse: 0.,
            ymouse: 0.,
            zmouse: 0.,
            cur_time: 1,
            num_times: 1,
            track_mouse_for_plugs: 0,
            xmovie: 0,
            ymovie: 0,
            zmovie: 0,
            movie_started_by_slicer: false,
            image_axis: 3,
        }
    }
}

/// Calls which cross from `slicer.cpp` into Qt, OpenGL, controller, cache,
/// model and movie units.  They deliberately remain explicit source boundaries.
pub trait SlicerNativeBoundary {
    fn draw(&mut self, _draw_flag: i32) {}
    fn draw_slicer_plane(&mut self) {}
    fn update_gl(&mut self) {}
    fn cube_draw(&mut self) {}
    fn set_angles(&mut self, _angles: [f32; 3]) {}
    fn set_zoom_text(&mut self, _zoom: f32) {}
    fn set_toggle_state(&mut self, _index: usize, _state: i32) {}
    fn set_thicknesses(&mut self, _image: i32, _model: f32) {}
    fn set_view_axis_position(&mut self, _min: i32, _max: i32, _current: i32) {}
    fn set_low_high_validity(&mut self, _which: usize, _state: i32) {}
    fn enable_low_high_buttons(&mut self, _enabled: bool) {}
    fn manage_band_size(&mut self, _width: i32, _height: i32, _action: i32) {}
    fn set_cursor(&mut self, _mode: i32, _force: bool) {}
    fn fill_cache(&mut self) {}
    fn help(&mut self) {}
    fn close(&mut self) {}
    fn movie(&mut self, _xmovie: i32, _ymovie: i32, _zmovie: i32) {}
    fn set_movie_limits(&mut self, _axis: i32, _start: i32, _end: i32) {}
    fn input_next_time(&mut self) {}
    fn input_prev_time(&mut self) {}
    fn image_draw(&mut self) {}
    fn model_draw(&mut self) {}
    fn cube_paint(&mut self) {}
    /// `SlicerGL::setMouseTracking`.
    fn set_mouse_tracking(&mut self, _enabled: bool) {}
    /// `SlicerFuncs::changeCenterIfLinked`, whose model-view transform is
    /// owned by the linked native model-view host.
    fn change_center_if_linked(&mut self, _slicer: &mut SlicerFuncs) {}
    /// `setAngleToolbarState`'s four Qt control enable/visibility updates.
    fn set_angle_toolbar_state(&mut self, _open: bool) {}
    /// `SlicerAngleForm::newTime`.
    fn slicer_new_time(&mut self, _refresh: bool) {}
    /// `imodv_draw` for a linked slicer plane.
    fn model_view_slicer_update(&mut self) {}
    /// `ivwBindMouse` after a slicer changes the shared image position.
    fn bind_mouse(&mut self) {}
    /// The movie snapshot portion of `externalDraw`, whose framebuffer and
    /// movie-controller state belong to the native host.  A true return means
    /// it consumed the current draw after the slicer updated its position.
    fn movie_snapshot_cycle(&mut self, _slicer: &mut SlicerFuncs) -> bool {
        false
    }
    /// `b3dSetCurSize` for a snapshot issued from this slicer's GL surface.
    fn set_snapshot_size(&mut self, _width: i32, _height: i32) {}
    /// `b3dNamedSnapshot`; the compatibility-GL host owns its framebuffer and
    /// file encoder, while the slicer owns the limits and source call order.
    fn named_snapshot(
        &mut self,
        _name: &str,
        _window_name: &str,
        _format: i32,
        _limits: Option<[i32; 4]>,
        _check_convert: bool,
    ) -> i32 {
        -1
    }
    /// `ivwGetImagePadding`, expressed as the source's lower limit, padding,
    /// and flip flag for each image axis.  Image-file and montage ownership
    /// remains in the normal 3dmod host.
    fn image_padding(
        &mut self,
        _y_center: i32,
        _z_center: i32,
        _time: i32,
    ) -> Option<SlicerImagePadding> {
        None
    }
    /// `imodPlugHandleMouse` after the winit host has normalized its pointer
    /// event.  The return bits retain the source contract: bit 1 means the
    /// plug consumed the event and bit 2 requests a redraw.
    fn plugin_handle_mouse(
        &mut self,
        _x: f32,
        _y: f32,
        _button1: i32,
        _button2: i32,
        _button3: i32,
    ) -> i32 {
        0
    }
    /// Selection/time snapshot for `SlicerFuncs::drawCurrentPoint`.
    fn current_point_state(&mut self) -> SlicerCurrentPointState {
        SlicerCurrentPointState::default()
    }
    /// Render the compatibility-GL overlay primitives computed by the source
    /// slicer logic.
    fn draw_current_point_overlay(&mut self, _primitives: &[SlicerOverlayPrimitive]) {}
    /// Shared model values consumed by the source slicer renderer.
    fn model_render_config(&mut self) -> SlicerModelRenderConfig {
        SlicerModelRenderConfig::default()
    }
    /// Execute a source `drawModel` pass.  The default retains compatibility
    /// with the earlier host boundary for ordinary (single-pass) rendering.
    fn draw_model_plan(&mut self, _plan: SlicerModelRenderPlan) {
        self.model_draw();
    }
    /// Movie-controller preferences read by `montageSnapshot`.
    fn montage_config(&mut self) -> SlicerMontageConfig {
        SlicerMontageConfig::default()
    }
    /// Allocate the framebuffer/montage backing store.  `true` has the
    /// source `utilStartMontSnap` meaning: allocation failed.
    fn montage_start(&mut self, _plan: SlicerMontagePlan) -> bool {
        true
    }
    /// Draw/read/copy one panel.  This is the explicit compatibility-GL
    /// framebuffer boundary corresponding to updateGL/glReadPixels/memLineCpy.
    fn montage_panel(&mut self, _panel: SlicerMontagePanel) -> bool {
        false
    }
    /// Save the assembled image (`utilFinishMontSnap`) and release host
    /// buffers.  Called only after every panel completed.
    fn montage_finish(&mut self, _plan: SlicerMontagePlan) {}
    /// Release host montage buffers after a panel failure.
    fn montage_abort(&mut self) {}
    /// `imodAllObjNearest` plus `imodSelectionNewCurPoint`.  The shared model
    /// controller performs selection and returns its resulting mouse point.
    fn attach_model_point(
        &mut self,
        _point: Ipoint,
        _selection_size: f32,
        _forward_matrix: [[f32; 3]; 3],
        _time: i32,
        _ctrl_down: bool,
    ) -> SlicerAttachResult {
        SlicerAttachResult::default()
    }
    /// `imodDraw(IMOD_DRAW_XYZ | IMOD_DRAW_RETHINK)` after an edit gesture.
    fn model_edit_draw(&mut self, _rethink: bool) {}
    /// Whether shared-model editing is active for insert/modify gestures.
    fn model_edit_mode(&mut self) -> bool {
        false
    }
    /// `ivwGetOrMakeContour`/planar-contour handling followed by
    /// `inputInsertPoint`, implemented against the shared Rust controller.
    fn insert_model_point(&mut self, _request: SlicerInsertRequest) {}
    /// `inputModifyPoint` against the shared Rust controller.
    fn modify_model_point(&mut self, _point: Ipoint) {}
    /// Construct and attach the slicer orientation cube in the Rust-native
    /// winit/compatibility-GL toolbar host (`SlicerFuncs::addCubeToFrame`).
    fn add_cube_to_frame(&mut self) {}
}

/// Result payload of the source `ivwGetImagePadding` call.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SlicerImagePadding {
    pub llx: i32,
    pub xpad: i32,
    pub xflip: i32,
    pub lly: i32,
    pub ypad: i32,
    pub yflip: i32,
    pub llz: i32,
    pub zpad: i32,
    pub zflip: i32,
}

/// Why `rotateVolCommand` returns no command.  The C function reports these
/// through `wprint` and returns an empty QString; keeping the reason makes the
/// Rust-native form able to present the same outcome without a Qt dependency.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RotateVolCommandError {
    StartingBand,
    BandOutsideVolume,
    BandLimitsUnset,
    UnequalBinning,
    ImagePaddingUnavailable,
}

/// Model/controller information consumed by `drawCurrentPoint`.  It is a
/// snapshot because selection and time ownership remain in the shared 3dmod
/// controller, not in the winit GL surface.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SlicerCurrentPointState {
    pub draw_cursor: bool,
    pub movie_mode: bool,
    pub current_point: Option<Ipoint>,
    pub contour_points: Vec<Ipoint>,
    pub contour_time_mismatch: bool,
    pub image_point_size: i32,
    pub model_point_size: i32,
    pub backup_point_size: i32,
    pub model_line_width: i32,
}

/// Compatibility-GL primitives emitted by `drawCurrentPoint`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SlicerOverlayPrimitive {
    Plus {
        x: i32,
        y: i32,
        size: i32,
        color: SlicerOverlayColor,
    },
    Circle {
        x: i32,
        y: i32,
        size: i32,
        color: SlicerOverlayColor,
        line_width: i32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlicerOverlayColor {
    Beginning,
    Endpoint,
    CurrentPoint,
    Shadow,
}

/// Shared-model values used by the slicer-specific part of `drawModel`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlicerModelRenderConfig {
    pub model_z_scale: f32,
    pub extra_object_count: i32,
    pub extra_cursor_in_window: bool,
}
impl Default for SlicerModelRenderConfig {
    fn default() -> Self {
        Self {
            model_z_scale: 1.,
            extra_object_count: 0,
            extra_cursor_in_window: false,
        }
    }
}

/// One source `drawModel` loop after its projection and model-view state have
/// been resolved.  The compatibility-profile renderer owns the GL calls and
/// delegates object traversal to `model_draw.rs`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlicerModelRenderPlan {
    pub projection_depth: f32,
    pub center: Ipoint,
    pub scale: Ipoint,
    pub angles: [f32; 3],
    pub z_scale_before: f32,
    pub inverse_model_scale: Ipoint,
    pub draw_main_model: bool,
    pub extra_object_mode: i32,
}

/// Host-owned controls used by `montageSnapshot`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlicerMontageConfig {
    pub factor: i32,
    pub scale_thicknesses: bool,
    pub thickness_scaling: i32,
}
impl Default for SlicerMontageConfig {
    fn default() -> Self {
        Self {
            factor: 1,
            scale_thicknesses: false,
            thickness_scaling: 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlicerMontagePlan {
    pub factor: i32,
    pub scale_factor: f32,
    pub overlap_x: i32,
    pub overlap_y: i32,
    pub full_width: i32,
    pub full_height: i32,
    pub snap_type: i32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlicerMontagePanel {
    pub x_index: i32,
    pub y_index: i32,
    pub center: Ipoint,
    pub copy_from_x: i32,
    pub copy_from_y: i32,
    pub copy_to_x: i32,
    pub copy_to_y: i32,
    pub copy_width: i32,
    pub copy_height: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlicerMontageError {
    WindowTooSmall,
    HostAllocationFailed,
    HostPanelFailed,
}

/// Result of the shared-controller half of `attachPoint`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SlicerAttachResult {
    pub model_mode: bool,
    pub selected_mouse: Option<Ipoint>,
}

/// Source inputs for the controller-owned contour/new-surface half of
/// `insertPoint`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlicerInsertRequest {
    pub point: Ipoint,
    pub plane_normal: Ipoint,
    pub time_lock: i32,
}

/// `SlicerFuncs` source state.  Field names follow the systematic snake-case
/// mapping from the paired `sslice.h` members.
#[derive(Clone, Debug)]
pub struct SlicerFuncs {
    pub view: SlicerView,
    pub cx: f32,
    pub cy: f32,
    pub cz: f32,
    pub tang: [f32; 3],
    pub lang: [f32; 3],
    pub locked: i32,
    pub draw_mod_view: i32,
    pub already_drew: bool,
    pub time_lock: i32,
    pub continuous: bool,
    pub linked: bool,
    pub auto_link: i32,
    pub classic: i32,
    pub zoom: f32,
    pub winx: i32,
    pub winy: i32,
    pub hq: i32,
    pub scalez: i32,
    pub fft_mode: i32,
    pub nslice: i32,
    pub depth: f32,
    pub rubberband: i32,
    pub starting_band: i32,
    pub closing: i32,
    pub xstep: [f32; 3],
    pub ystep: [f32; 3],
    pub zstep: [f32; 3],
    pub xo: f32,
    pub yo: f32,
    pub zo: f32,
    pub xzoom: f32,
    pub yzoom: f32,
    pub remaining_zoom: f32,
    pub no_pixel_zoom: bool,
    pub pending: i32,
    pub pendx: f32,
    pub pendy: f32,
    pub pendz: f32,
    pub last_axis_pos: i32,
    pub lastangle: usize,
    pub shift_lock: i32,
    pub mousemode: i32,
    pub need_draw: bool,
    pub doing_draw: bool,
    pub arrow_on: bool,
    pub drawing_arrow: bool,
    pub arrow_head: Vec<Ipoint>,
    pub arrow_tail: Vec<Ipoint>,
    pub arrow_angle: [f32; 3],
    pub band_angle: [f32; 3],
    pub rb_image_x0: f32,
    pub rb_image_x1: f32,
    pub rb_image_y0: f32,
    pub rb_image_y1: f32,
    pub rb_image_z0: f32,
    pub rb_image_z1: f32,
    pub rb_mouse_x0: i32,
    pub rb_mouse_x1: i32,
    pub rb_mouse_y0: i32,
    pub rb_mouse_y1: i32,
    pub rb_start_x0: i32,
    pub rb_start_x1: i32,
    pub rb_start_y0: i32,
    pub rb_start_y1: i32,
    pub band_low_high_limits: [f32; 2],
    pub limit_no_value: i32,
    pub view_axis_index: usize,
    pub image_filled: i32,
    pub cur_buf_size: usize,
    pub matrix: [[f32; 3]; 3],
    pub orig_zmouse: f32,
    pub orig_angles: [f32; 3],
    pub last_xmouse: f32,
    pub last_ymouse: f32,
    pub last_zmouse: f32,
    pub cum_page_moves: f32,
    /// `mGlw->mFirstDraw`, provided by the winit GL widget.
    pub first_draw: i32,
    pub last_xsize_change: f32,
    pub screen_changed: bool,
    pub screen_change_time: std::time::Instant,
    pub screen_resize_time: std::time::Instant,
    pub new_screen_zoom: f32,
    pub user_screen_change: i32,
    /// `App->devPixVaries`, `App->isWindows`, and current left-button state,
    /// refreshed by the native winit host.
    pub dev_pix_varies: bool,
    pub is_windows: bool,
    pub left_mouse_down: bool,
    pub ignore_cur_pt_chg: bool,
    pub zslast: f32,
    pub lx: f32,
    pub ly: f32,
    pub lz: f32,
    pub drawn_xmouse: f32,
    pub drawn_ymouse: f32,
    pub drawn_zmouse: f32,
    /// Source static `sDoingMontage`, set during `montageSnapshot`.
    pub doing_montage: bool,
}

/// One-axis output of `SlicerFuncs::getMontageShifts`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MontageShifts {
    pub overlap: i32,
    pub trans_start: f32,
    pub trans_delta: f32,
    pub copy_delta: i32,
    pub full_size: i32,
}

/// Geometry produced by `SlicerFuncs::resizeToFit` before Qt applies its
/// platform-specific size/position constraints.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlicerResizePlan {
    pub width: i32,
    pub height: i32,
    pub x: i32,
    pub y: i32,
}

/// Rust-owned equivalent of the source dialog manager's slicer window list
/// and `sSliceAngDia`/`sPixelViewOpen` statics.  A winit host owns this
/// registry for its full UI-thread lifetime; it deliberately holds slicer
/// logic, not native widget handles.
pub struct SlicerRegistry {
    pub slicers: Vec<Box<SlicerFuncs>>,
    pub angles_open: bool,
    pub pixel_view_open: bool,
    pub view_axis_index: usize,
    pub scale_thick: i32,
    pub link_was_limited: bool,
}

impl Default for SlicerRegistry {
    fn default() -> Self {
        Self {
            slicers: Vec::new(),
            angles_open: false,
            pixel_view_open: false,
            // `sViewAxisIndex = 2` (`slicer.cpp:72`).
            view_axis_index: 2,
            // `sScaleThick = 1` (`slicer.cpp:74`).
            scale_thick: 1,
            link_was_limited: false,
        }
    }
}

impl SlicerFuncs {
    /// `SlicerFuncs::SlicerFuncs` after Qt construction is delegated to its paired unit.
    pub fn new(mut view: SlicerView, auto_link: i32) -> Self {
        if view.xmouse == 0. && view.ymouse == 0. {
            view.xmouse = view.xsize as f32 / 2.;
            view.ymouse = view.ysize as f32 / 2.;
        }
        let original_mouse = (view.xmouse, view.ymouse, view.zmouse);
        let mut s = Self {
            cx: view.xmouse,
            cy: view.ymouse,
            cz: view.zmouse,
            view,
            tang: [0.; 3],
            lang: [0.; 3],
            locked: 0,
            draw_mod_view: 0,
            already_drew: false,
            time_lock: auto_link,
            continuous: false,
            linked: auto_link > 0,
            auto_link: auto_link.min(2),
            classic: 0,
            zoom: 1.,
            winx: 1,
            winy: 1,
            hq: 0,
            scalez: 0,
            fft_mode: 0,
            nslice: 1,
            depth: 1.,
            rubberband: 0,
            starting_band: 0,
            closing: 0,
            xstep: [1., 0., 0.],
            ystep: [0., 1., 0.],
            zstep: [0., 0., 1.],
            xo: 0.,
            yo: 0.,
            zo: 0.,
            xzoom: 1.,
            yzoom: 1.,
            remaining_zoom: 1.,
            no_pixel_zoom: false,
            pending: 0,
            pendx: 0.,
            pendy: 0.,
            pendz: 0.,
            last_axis_pos: 1,
            lastangle: 0,
            shift_lock: 0,
            mousemode: 0,
            need_draw: false,
            doing_draw: false,
            arrow_on: false,
            drawing_arrow: false,
            arrow_head: Vec::new(),
            arrow_tail: Vec::new(),
            arrow_angle: [0.; 3],
            band_angle: [0.; 3],
            rb_image_x0: 0.,
            rb_image_x1: 0.,
            rb_image_y0: 0.,
            rb_image_y1: 0.,
            rb_image_z0: 0.,
            rb_image_z1: 0.,
            rb_mouse_x0: 0,
            rb_mouse_x1: 0,
            rb_mouse_y0: 0,
            rb_mouse_y1: 0,
            rb_start_x0: 0,
            rb_start_x1: 0,
            rb_start_y0: 0,
            rb_start_y1: 0,
            band_low_high_limits: [0.; 2],
            limit_no_value: 0,
            view_axis_index: 2,
            image_filled: 0,
            cur_buf_size: 0,
            matrix: [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
            orig_zmouse: original_mouse.2,
            // `mOrigAngles[0] = -1000.` starts a fresh page-move sequence.
            orig_angles: [-1000., 0., 0.],
            last_xmouse: original_mouse.0,
            last_ymouse: original_mouse.1,
            last_zmouse: original_mouse.2,
            cum_page_moves: 0.,
            first_draw: 0,
            last_xsize_change: 1.,
            screen_changed: false,
            screen_change_time: std::time::Instant::now(),
            screen_resize_time: std::time::Instant::now(),
            new_screen_zoom: 0.,
            user_screen_change: 0,
            dev_pix_varies: false,
            is_windows: false,
            left_mouse_down: false,
            ignore_cur_pt_chg: false,
            zslast: 1.,
            lx: original_mouse.0,
            ly: original_mouse.1,
            lz: original_mouse.2,
            drawn_xmouse: original_mouse.0,
            drawn_ymouse: original_mouse.1,
            drawn_zmouse: original_mouse.2,
            doing_montage: false,
        };
        s.trans_step();
        s
    }

    /// `SlicerFuncs::setInitialZoom`.
    pub fn set_initial_zoom(&mut self, zoom: f32) {
        self.zoom = zoom;
        if zoom > 1.5 {
            self.hq = 1;
        }
    }
    /// `SlicerFuncs::setMouseTracking`.
    pub fn set_mouse_tracking(&mut self, pixel_view_open: bool, n: &mut dyn SlicerNativeBoundary) {
        n.set_mouse_tracking(
            self.rubberband != 0 || pixel_view_open || self.view.track_mouse_for_plugs != 0,
        );
    }
    /// `SlicerFuncs::viewAxisStepSize`.
    pub fn view_axis_step_size(&self) -> f32 {
        S_VIEW_AXIS_STEPS[self.view_axis_index]
    }
    /// `SlicerFuncs::help`.
    pub fn help(&mut self, n: &mut dyn SlicerNativeBoundary) {
        n.help()
    }
    /// `SlicerFuncs::stepZoom`.
    pub fn step_zoom(&mut self, dir: i32, n: &mut dyn SlicerNativeBoundary) {
        self.zoom = if dir > 0 {
            self.zoom * 1.25
        } else {
            self.zoom / 1.25
        };
        self.manage_buffers();
        n.set_zoom_text(self.zoom);
        self.draw_self_and_linked(n);
    }
    /// `SlicerFuncs::enteredZoom`.
    pub fn entered_zoom(&mut self, zoom: f32, n: &mut dyn SlicerNativeBoundary) {
        if self.closing != 0 {
            return;
        }
        self.zoom = zoom.max(0.01);
        self.manage_buffers();
        n.set_zoom_text(self.zoom);
        self.draw_self_and_linked(n);
    }
    /// `SlicerFuncs::stepTime`.
    pub fn step_time(&mut self, step: i32, n: &mut dyn SlicerNativeBoundary) {
        if self.time_lock != 0 {
            self.time_lock = (self.time_lock + step).clamp(1, self.view.num_times);
            self.draw(n);
        } else if step > 0 {
            n.input_next_time()
        } else {
            n.input_prev_time()
        }
    }
    /// `SlicerFuncs::showSlice`.
    pub fn show_slice(&mut self, n: &mut dyn SlicerNativeBoundary) {
        self.trans_step();
        n.draw(0x20 | self.draw_mod_view);
        self.draw_mod_view = 0;
    }
    /// `SlicerFuncs::fillCache`.
    pub fn fill_cache(&mut self, n: &mut dyn SlicerNativeBoundary) {
        n.fill_cache()
    }
    /// `SlicerFuncs::stateToggled`.
    pub fn state_toggled(&mut self, index: usize, state: i32, n: &mut dyn SlicerNativeBoundary) {
        match index {
            SLICER_TOGGLE_LOCK => {
                self.locked = state;
                if state == 0 {
                    self.cx = self.view.xmouse;
                    self.cy = self.view.ymouse;
                    self.cz = self.view.zmouse;
                    self.pending = 0;
                    self.draw(n);
                }
            }
            SLICER_TOGGLE_HIGHRES => {
                self.hq = state;
                self.manage_buffers();
                self.draw_self_and_linked(n);
            }
            SLICER_TOGGLE_CENTER => self.set_classic_mode(state, false, n),
            SLICER_TOGGLE_SHIFTLOCK => self.shift_lock = state,
            SLICER_TOGGLE_BAND => self.toggle_rubberband(true, n),
            SLICER_TOGGLE_ARROW => self.toggle_arrow(true, n),
            SLICER_TOGGLE_FFT => {
                self.fft_mode = state;
                self.draw_self_and_linked(n);
            }
            SLICER_TOGGLE_ZSCALE => {
                self.scalez = state;
                self.draw_self_and_linked(n);
                n.draw_slicer_plane();
            }
            SLICER_TOGGLE_TIMELOCK => {
                self.time_lock = if state != 0 { self.view.cur_time } else { 0 };
                if state == 0 {
                    self.draw(n);
                }
            }
            _ => {}
        }
    }
    /// `SlicerFuncs::toggleArrow`.
    pub fn toggle_arrow(&mut self, draw_win: bool, n: &mut dyn SlicerNativeBoundary) {
        self.arrow_on = !self.arrow_on;
        self.drawing_arrow = self.arrow_on;
        n.set_toggle_state(SLICER_TOGGLE_ARROW, self.arrow_on as i32);
        if self.arrow_on {
            self.arrow_head.push(Ipoint::default());
            self.arrow_tail.push(Ipoint::default());
        } else {
            self.arrow_head.pop();
            self.arrow_tail.pop();
        }
        if self.arrow_on && self.starting_band != 0 {
            self.toggle_rubberband(false, n);
        }
        n.set_cursor(self.mousemode, true);
        if draw_win {
            self.draw(n);
        }
        n.set_cursor(self.mousemode, true);
    }
    /// `SlicerFuncs::clearArrows`.
    pub fn clear_arrows(&mut self, n: &mut dyn SlicerNativeBoundary) {
        if self.closing == 0 {
            if self.arrow_on {
                self.toggle_arrow(false, n);
            }
            self.arrow_head.clear();
            self.arrow_tail.clear();
            self.draw(n);
        }
    }
    /// `SlicerFuncs::startAddedArrow`.
    pub fn start_added_arrow(&mut self, n: &mut dyn SlicerNativeBoundary) {
        if let (Some(h), Some(t)) = (self.arrow_head.last(), self.arrow_tail.last()) {
            if self.drawing_arrow && *h == Ipoint::default() && *t == Ipoint::default() {
                return;
            }
        }
        self.arrow_on = false;
        self.toggle_arrow(false, n);
    }
    /// `SlicerFuncs::setClassicMode`.
    pub fn set_classic_mode(
        &mut self,
        state: i32,
        skip_draw: bool,
        n: &mut dyn SlicerNativeBoundary,
    ) {
        self.classic = state;
        if self.locked == 0 && state != 0 {
            self.cx = self.view.xmouse;
            self.cy = self.view.ymouse;
            self.cz = self.view.zmouse;
        }
        self.pending = 0;
        if !skip_draw {
            self.draw(n);
            n.draw(1);
        }
    }
    /// `SlicerFuncs::angleChanged`.
    pub fn angle_changed(
        &mut self,
        axis: i32,
        value: i32,
        dragging: bool,
        n: &mut dyn SlicerNativeBoundary,
    ) {
        if axis < 3 {
            self.set_forward_matrix();
            self.tang[axis as usize] = value as f32 * 0.1;
            self.lastangle = axis as usize;
        } else {
            let v = self.normal_to_plane();
            let d = (value - self.last_axis_pos) as f32;
            self.cx += d * v.x;
            self.cy += d * v.y;
            self.cz += d * v.z;
        }
        if !dragging {
            self.show_slice(n);
        } else {
            self.trans_step();
            n.cube_draw();
        }
    }
    /// `SlicerFuncs::updateViewAxisPos`.
    pub fn update_view_axis_pos(&mut self, n: &mut dyn SlicerNativeBoundary) {
        let v = self.normal_to_plane();
        let mut nums = [0_i32; 2];
        for (ind, direction) in [-1., 1.].into_iter().enumerate() {
            let mut i = 1;
            loop {
                let x = self.cx + direction * i as f32 * v.x;
                let y = self.cy + direction * i as f32 * v.y;
                let z = self.cz + direction * i as f32 * v.z;
                if x < 0.
                    || x >= self.view.xsize as f32
                    || y < 0.
                    || y >= self.view.ysize as f32
                    || z < 0.
                    || z >= self.view.zsize as f32 - 0.5
                {
                    nums[ind] = i - 1;
                    break;
                }
                i += 1;
            }
        }
        self.last_axis_pos = 1 + nums[0];
        n.set_view_axis_position(1, self.last_axis_pos + nums[1], self.last_axis_pos);
    }
    /// `SlicerFuncs::drawThickControls`.
    pub fn draw_thick_controls(&self, n: &mut dyn SlicerNativeBoundary) {
        n.set_thicknesses(self.nslice, self.depth)
    }
    /// `SlicerFuncs::imageThickness`.
    pub fn image_thickness(&mut self, depth: i32, n: &mut dyn SlicerNativeBoundary) {
        if self.closing == 0 {
            self.nslice = depth.max(1);
            self.draw_thick_controls(n);
            self.draw_self_and_linked(n);
        }
    }
    /// `SlicerFuncs::modelThickness`.
    pub fn model_thickness(&mut self, depth: f32, n: &mut dyn SlicerNativeBoundary) {
        if self.closing == 0 {
            self.depth = if (self.depth - 0.1).abs() < 0.01 && (depth - 1.1).abs() < 0.01 {
                1.
            } else {
                depth.max(0.1)
            };
            self.draw_thick_controls(n);
            self.draw_self_and_linked(n);
        }
    }
    /// `SlicerFuncs::setLinkedState`.
    pub fn set_linked_state(&mut self, state: bool) {
        self.linked = state;
        if !state {
            self.auto_link = 0;
        }
    }
    /// `SlicerFuncs::closing`.
    pub fn closing(&mut self, n: &mut dyn SlicerNativeBoundary) {
        self.closing = 1;
        self.linked = false;
        self.arrow_head.clear();
        self.arrow_tail.clear();
        n.close();
    }
    /// `SlicerFuncs::getSubsetLimits`.
    pub fn get_subset_limits(&self) -> (i32, i32, i32, i32) {
        let xs = (self.cx - 0.7 * self.winx as f32 / self.zoom).max(0.) as i32;
        let xe = (self.cx + 0.7 * self.winx as f32 / self.zoom).min(self.view.xsize as f32) as i32;
        let ys = (self.cy - 0.7 * self.winy as f32 / self.zoom).max(0.) as i32;
        let ye = (self.cy + 0.7 * self.winy as f32 / self.zoom).min(self.view.ysize as f32) as i32;
        (xs, ys, xe - xs, ye - ys)
    }
    /// `SlicerFuncs::setViewAxisRotation`.
    pub fn set_view_axis_rotation(&mut self, x: f32, y: f32, z: f32) {
        self.set_forward_matrix();
        let r = rotation_matrix(x, y, z);
        self.matrix = matrix_mul(self.matrix, r);
        self.tang = natural_angles(self.matrix);
    }
    /// `SlicerFuncs::rotateOnViewAxis`.
    pub fn rotate_on_view_axis(
        &mut self,
        dx: i32,
        dy: i32,
        dz: i32,
        n: &mut dyn SlicerNativeBoundary,
    ) {
        let u = self.view_axis_step_size();
        self.set_view_axis_rotation(dx as f32 * u, dy as f32 * u, dz as f32 * u);
        n.set_angles(self.tang);
        self.show_slice(n);
        self.draw_self_and_linked(n);
    }
    /// `SlicerFuncs::setZmouseForAxisMove`.
    pub fn set_zmouse_for_axis_move(&mut self, zmove: f32, n: &mut dyn SlicerNativeBoundary) {
        if self.tang == self.orig_angles
            && self.view.xmouse == self.last_xmouse
            && self.view.ymouse == self.last_ymouse
            && self.view.zmouse == self.last_zmouse
        {
            self.cum_page_moves += zmove;
        } else {
            self.orig_angles = self.tang;
            self.orig_zmouse = self.view.zmouse;
            self.cum_page_moves = zmove;
        }
        let normal = self.normal_to_plane();
        self.view.zmouse = (self.orig_zmouse + self.cum_page_moves * normal.z).round();
        self.last_xmouse = self.view.xmouse;
        self.last_ymouse = self.view.ymouse;
        self.last_zmouse = self.view.zmouse;
        n.bind_mouse();
    }
    /// `SlicerFuncs::screenChanged`.  Winit supplies the display-change event
    /// and refreshes the host-derived fields above; this preserves the source
    /// zoom/timer state transition after that event reaches the slicer.
    pub fn screen_changed(&mut self, new_dpr: f32, n: &mut dyn SlicerNativeBoundary) {
        if new_dpr == 0. || self.first_draw > 2 {
            return;
        }
        let screen_elapsed = self
            .screen_change_time
            .elapsed()
            .as_millis()
            .min(i32::MAX as u128) as i32;
        let resize_elapsed = self
            .screen_resize_time
            .elapsed()
            .as_millis()
            .min(i32::MAX as u128) as i32;
        if util_set_zoom_on_screen_change(
            0.,
            &mut self.last_xsize_change,
            &mut self.screen_changed,
            screen_elapsed,
            resize_elapsed,
            &mut self.zoom,
            self.dev_pix_varies,
        ) {
            n.set_zoom_text(self.zoom);
        }
        if self.new_screen_zoom > 0. {
            self.zoom = self.zoom.min(self.new_screen_zoom);
            n.set_zoom_text(self.zoom);
            self.new_screen_zoom = 0.;
        } else if self.is_windows && self.dev_pix_varies && self.left_mouse_down {
            self.user_screen_change = 1;
        }
    }
    /// `SlicerFuncs::findMovieAxis`.
    pub fn find_movie_axis(&mut self, direction: i32) -> (i32, i32, i32, usize) {
        let v = self.normal_to_plane();
        if v.x.abs() >= v.y.abs() && v.x.abs() >= v.z.abs() {
            (direction, 0, 0, 0)
        } else if v.y.abs() >= v.z.abs() {
            (0, direction, 0, 1)
        } else {
            (0, 0, direction, 2)
        }
    }
    /// `SlicerFuncs::findAxisLimits`.
    pub fn find_axis_limits(&mut self, axis: usize) -> (f32, i32, i32) {
        let v = self.normal_to_plane();
        let current = [self.cx, self.cy, self.cz][axis];
        let size = [self.view.xsize, self.view.ysize, self.view.zsize][axis];
        let comp = [v.x, v.y, v.z][axis];
        if comp.abs() < f32::EPSILON {
            return (current, -1, -1);
        }
        let mut start = -1;
        let mut end = -1;
        for i in 0..size {
            let d = (i as f32 - current) / comp;
            let x = self.cx + d * v.x;
            let y = self.cy + d * v.y;
            let z = self.cz + d * v.z;
            if x >= 0.
                && x <= self.view.xsize as f32 - 1.
                && y >= 0.
                && y <= self.view.ysize as f32 - 1.
                && z >= 0.
                && z <= self.view.zsize as f32 - 1.
            {
                if start < 0 {
                    start = i;
                }
                end = i;
            }
        }
        (current, start, end)
    }
    /// `SlicerFuncs::setMovieLimits`.
    pub fn set_movie_limits(&mut self, axis: usize, n: &mut dyn SlicerNativeBoundary) {
        let (_, start, end) = self.find_axis_limits(axis);
        n.set_movie_limits(axis as i32, start, end);
    }
    /// `SlicerFuncs::startMovieCheckSnap`.
    pub fn start_movie_check_snap(&mut self, direction: i32, n: &mut dyn SlicerNativeBoundary) {
        let (x, y, z, axis) = self.find_movie_axis(direction);
        n.movie(x, y, z);
        self.view.xmouse = self.cx;
        self.view.ymouse = self.cy;
        self.view.zmouse = self.cz;
        self.set_movie_limits(axis, n);
    }
    /// `SlicerFuncs::checkMovieLimits`.
    pub fn check_movie_limits(&mut self, n: &mut dyn SlicerNativeBoundary) {
        if (self.view.xmovie == 0 && self.view.ymovie == 0 && self.view.zmovie == 0)
            || !self.view.movie_started_by_slicer
        {
            return;
        }
        let normal = self.normal_to_plane();
        let (mut xmovie, mut ymovie, mut zmovie, mut axis) = self.find_movie_axis(1);
        if (self.view.xmovie != 0 && axis != 0)
            || (self.view.ymovie != 0 && axis != 1)
            || (self.view.zmovie != 0 && axis != 2)
        {
            let old_direction = self.view.xmovie as f32 * normal.x
                + self.view.ymovie as f32 * normal.y
                + self.view.zmovie as f32 * normal.z;
            if old_direction / [normal.x, normal.y, normal.z][axis] < 0. {
                (xmovie, ymovie, zmovie, axis) = self.find_movie_axis(-1);
            }
            n.movie(xmovie, ymovie, zmovie);
        }
        self.set_movie_limits(axis, n);
    }
    /// `SlicerFuncs::externalDraw`, excluding only the framebuffer/movie
    /// snapshot transaction carried by `SlicerNativeBoundary::movie_snapshot_cycle`.
    pub fn external_draw(&mut self, mut drawflag: i32, n: &mut dyn SlicerNativeBoundary) {
        if self.closing != 0 {
            return;
        }
        let ignore_change = self.ignore_cur_pt_chg;
        self.ignore_cur_pt_chg = false;
        if self.already_drew {
            self.already_drew = false;
            if drawflag & (1 << 11) == 0 {
                return;
            }
        }
        if drawflag & (1 << 11) != 0 {
            return;
        }
        self.cx = self.cx.clamp(0., self.view.xsize as f32);
        self.cy = self.cy.clamp(0., self.view.ysize as f32);
        self.cz = self.cz.clamp(0., self.view.zsize as f32 - 0.5);
        n.set_cursor(self.mousemode, false);
        if self.zslast != self.view.zscale {
            self.zslast = self.view.zscale;
            drawflag |= 1 << 14;
        }
        if (self.view.xmovie != 0 || self.view.ymovie != 0 || self.view.zmovie != 0)
            && self.view.movie_started_by_slicer
        {
            let normal = self.normal_to_plane();
            let factor = if self.view.xmovie != 0 && normal.x.abs() > 1.0e-6 {
                (self.view.xmouse - self.cx) / normal.x
            } else if self.view.ymovie != 0 && normal.y.abs() > 1.0e-6 {
                (self.view.ymouse - self.cy) / normal.y
            } else if self.view.zmovie != 0 && normal.z.abs() > 1.0e-6 {
                (self.view.zmouse - self.cz) / normal.z
            } else {
                0.
            };
            if factor != 0.
                || ((self.view.xmouse - self.cx).abs() < 1.0e-5
                    && (self.view.ymouse - self.cy).abs() < 1.0e-5
                    && (self.view.zmouse - self.cz).abs() < 1.0e-5)
            {
                self.view.xmouse = self.cx + factor * normal.x;
                self.view.ymouse = self.cy + factor * normal.y;
                self.cz = (self.cz + factor * normal.z).clamp(0., self.view.zsize as f32 - 1.);
                self.view.zmouse = self.cz.round();
                n.bind_mouse();
                self.cx = self.view.xmouse;
                self.cy = self.view.ymouse;
                self.pending = 0;
                n.update_gl();
                if n.movie_snapshot_cycle(self) {
                    return;
                }
                n.cube_draw();
                return;
            }
        }
        if self.need_draw {
            self.draw(n);
            return;
        }
        if drawflag & (1 << 1) != 0 && self.locked == 0 {
            if self.classic != 0 {
                let (x, y, z) = if self.pending != 0 {
                    (self.pendx, self.pendy, self.pendz)
                } else {
                    (self.view.xmouse, self.view.ymouse, self.view.zmouse)
                };
                if (self.lx, self.ly, self.lz) != (x, y, z) {
                    (self.cx, self.cy, self.cz) = (x, y, z);
                    self.pending = 0;
                    self.draw(n);
                    return;
                }
            } else {
                if (self.lx, self.ly, self.lz) == (self.cx, self.cy, self.cz)
                    && self.lang == self.tang
                    && !ignore_change
                {
                    if self.view.xmouse == self.drawn_xmouse
                        && self.view.ymouse == self.drawn_ymouse
                        && self.view.zmouse != self.drawn_zmouse
                    {
                        self.cz = (self.cz + self.view.zmouse - self.drawn_zmouse)
                            .clamp(0., self.view.zsize as f32 - 0.5);
                    } else if self.view.xmouse == self.drawn_xmouse
                        && self.view.ymouse != self.drawn_ymouse
                        && self.view.zmouse == self.drawn_zmouse
                        && (self.view.ymouse - self.drawn_ymouse).abs() < 1.01
                    {
                        self.cy = (self.cy + self.view.ymouse - self.drawn_ymouse)
                            .clamp(0., self.view.ysize as f32 - 0.5);
                    } else if self.view.xmouse != self.drawn_xmouse
                        && self.view.ymouse == self.drawn_ymouse
                        && self.view.zmouse == self.drawn_zmouse
                        && (self.view.xmouse - self.drawn_xmouse).abs() < 1.01
                    {
                        self.cx = (self.cx + self.view.xmouse - self.drawn_xmouse)
                            .clamp(0., self.view.xsize as f32 - 0.5);
                    }
                }
                self.draw(n);
                return;
            }
        }
        if drawflag & ((1 << 14) | 1) != 0 {
            if self.pending != 0 && self.locked == 0 && self.classic != 0 {
                (self.cx, self.cy, self.cz) = (self.pendx, self.pendy, self.pendz);
                self.pending = 0;
            }
            self.draw(n);
        } else if drawflag & ((1 << 2) | (1 << 1)) != 0 {
            self.update_image(n);
        }
    }
    /// `SlicerFuncs::anglesFromContour`.  The caller supplies IMOD's current
    /// contour from the model owner; all fit, angle, and center calculations
    /// remain in this source unit.
    pub fn angles_from_contour(
        &mut self,
        contour: &Icont,
        n: &mut dyn SlicerNativeBoundary,
    ) -> i32 {
        if contour.pts.is_empty() {
            return 1;
        }
        let mut rotated = contour.clone();
        let (sin, cos) = self.tang[2].to_radians().sin_cos();
        let mut sum = ModelPoint::default();
        for (source, target) in contour.pts.iter().zip(&mut rotated.pts) {
            target.x = cos * source.x - sin * source.y;
            target.y = sin * source.x + cos * source.y;
            target.z = source.z;
            sum.x += source.x;
            sum.y += source.y;
            sum.z += source.z;
        }
        let mut normal = ModelPoint::default();
        let mut dval = 0.;
        let mut alpha = 0.;
        let mut beta = 0.;
        let scale = ModelPoint {
            x: 1.,
            y: 1.,
            z: self.get_z_scale_before(),
        };
        if imod_contour_fit_plane(
            &rotated,
            &scale,
            &mut normal,
            &mut dval,
            &mut alpha,
            &mut beta,
        ) != 0
        {
            return 1;
        }
        self.tang[0] = alpha.to_degrees() as f32;
        self.tang[1] = beta.to_degrees() as f32;
        n.set_angles(self.tang);
        let count = contour.pts.len() as f32;
        self.cx = (sum.x / count).clamp(0., self.view.xsize as f32 - 1.);
        self.cy = (sum.y / count).clamp(0., self.view.ysize as f32 - 1.);
        self.cz = (sum.z / count).clamp(0., self.view.zsize as f32 - 1.);
        self.view.xmouse = self.cx;
        self.view.ymouse = self.cy;
        self.view.zmouse = self.cz;
        n.bind_mouse();
        n.draw((1 << 1) | self.draw_mod_view);
        0
    }
    /// `SlicerFuncs::getZScaleBefore`.
    pub fn get_z_scale_before(&self) -> f32 {
        let mut z = self.view.zbin as f32 / self.view.xybin.max(1) as f32;
        if self.scalez == SLICE_ZSCALE_BEFORE && self.view.zscale > 0. {
            z *= self.view.zscale;
        }
        z
    }
    /// `SlicerFuncs::setxyz`.
    pub fn setxyz(&mut self, x: i32, y: i32) -> i32 {
        let (xm, ym, zm, zmouse) = self.getxyz(x as f32, y as f32, true);
        self.pendx = xm;
        self.pendy = ym;
        self.pendz = zm;
        self.pending = 1;
        self.view.xmouse = xm;
        self.view.ymouse = ym;
        self.view.zmouse = zm;
        zmouse
    }
    /// `SlicerFuncs::attachPoint`.  Arrow/band anchoring, cursor updates, and
    /// non-classic plane recentering belong to the slicer; shared-model nearest
    /// selection is explicitly delegated to the controller boundary.
    pub fn attach_point(
        &mut self,
        x: i32,
        y: i32,
        ctrl_down: bool,
        n: &mut dyn SlicerNativeBoundary,
    ) {
        self.view.zmouse = self.setxyz(x, y) as f32;
        if self.drawing_arrow {
            if let Some(index) = self.arrow_head.len().checked_sub(1) {
                let (px, py, pz, _) = self.getxyz(x as f32, y as f32, false);
                self.arrow_tail[index] = Ipoint {
                    x: px,
                    y: py,
                    z: pz,
                };
                self.arrow_head[index] = self.arrow_tail[index];
                self.arrow_angle = self.tang;
            }
        } else if self.starting_band != 0 {
            self.rb_mouse_x0 = x;
            self.rb_mouse_y0 = y;
            self.band_angle = self.tang;
        } else {
            self.set_forward_matrix();
            let result = n.attach_model_point(
                Ipoint {
                    x: self.view.xmouse,
                    y: self.view.ymouse,
                    z: self.view.zmouse,
                },
                10. / self.zoom,
                self.matrix,
                self.time_lock.max(self.view.cur_time),
                ctrl_down,
            );
            if result.model_mode {
                if let Some(mouse) = result.selected_mouse {
                    self.view.xmouse = mouse.x;
                    self.view.ymouse = mouse.y;
                    self.view.zmouse = mouse.z;
                    if self.classic == 0 {
                        let normal = self.normal_to_plane();
                        let delta = normal.x * (mouse.x - self.cx)
                            + normal.y * (mouse.y - self.cy)
                            + normal.z * (mouse.z - self.cz);
                        self.cx += delta * normal.x;
                        self.cy += delta * normal.y;
                        self.cz += delta * normal.z;
                        if self.cx < 0.
                            || self.cx >= self.view.xsize as f32
                            || self.cy < 0.
                            || self.cy >= self.view.ysize as f32
                            || self.cz < 0.
                            || self.cz >= self.view.zsize as f32 - 0.5
                        {
                            self.cx = mouse.x;
                            self.cy = mouse.y;
                            self.cz = mouse.z;
                        }
                    }
                }
            }
            self.pending = 0;
            if self.classic == 0 {
                self.ignore_cur_pt_chg = true;
            }
            n.model_edit_draw(result.model_mode);
            return;
        }
        self.pending = 0;
        if self.classic == 0 {
            self.ignore_cur_pt_chg = true;
        }
        n.model_edit_draw(false);
    }
    /// `SlicerFuncs::insertPoint`, including source Z-section preservation.
    pub fn insert_point(&mut self, x: i32, y: i32, ctrl: bool, n: &mut dyn SlicerNativeBoundary) {
        if n.model_edit_mode() {
            let zmouse = self.setxyz(x, y);
            if self.classic == 0 {
                self.ignore_cur_pt_chg = true;
            }
            n.insert_model_point(SlicerInsertRequest {
                point: Ipoint {
                    x: self.view.xmouse,
                    y: self.view.ymouse,
                    z: self.view.zmouse,
                },
                plane_normal: self.normal_to_plane(),
                time_lock: self.time_lock,
            });
            self.view.zmouse = zmouse as f32;
        } else if ctrl {
            self.start_movie_check_snap(1, n);
        }
    }
    /// `SlicerFuncs::modifyPoint`, including source Z-section preservation.
    pub fn modify_point(&mut self, x: i32, y: i32, ctrl: bool, n: &mut dyn SlicerNativeBoundary) {
        if n.model_edit_mode() {
            let zmouse = self.setxyz(x, y);
            if self.classic == 0 {
                self.ignore_cur_pt_chg = true;
            }
            n.modify_model_point(Ipoint {
                x: self.view.xmouse,
                y: self.view.ymouse,
                z: self.view.zmouse,
            });
            self.view.zmouse = zmouse as f32;
        } else if ctrl {
            self.start_movie_check_snap(-1, n);
        }
    }
    /// `SlicerFuncs::getxyz` (both C++ overloads are represented by `f32` input).
    pub fn getxyz(&self, x: f32, y: f32, clamp: bool) -> (f32, f32, f32, i32) {
        let zs = 1. / self.get_z_scale_before();
        let xo = (self.winx / 2) as f32 - x;
        let yo = self.winy as f32 / 2. - (self.winy - 1) as f32 + y;
        let xo = xo / self.xzoom;
        let yo = yo / self.yzoom;
        let mut xm = self.cx - (self.xstep[0] * xo + self.ystep[0] * yo);
        let mut ym = self.cy - (self.xstep[1] * xo + self.ystep[1] * yo);
        let mut zm = self.cz - (self.xstep[2] * xo * zs + self.ystep[2] * yo * zs);
        if clamp {
            xm = xm.clamp(0., self.view.xsize as f32 - 1.);
            ym = ym.clamp(0., self.view.ysize as f32 - 1.);
            zm = zm.clamp(0., self.view.zsize as f32 - 1.);
        }
        (xm, ym, zm, zm.round() as i32)
    }
    /// `SlicerFuncs::getWindowCoords`.
    pub fn get_window_coords(&mut self, mut x: f32, mut y: f32, mut z: f32) -> (f32, f32, f32) {
        let zs = 1. / self.get_z_scale_before();
        self.set_forward_matrix();
        x -= self.cx;
        y -= self.cy;
        z = (z - self.cz) / zs;
        let xn = matrix_vec(
            self.matrix,
            Ipoint {
                x: 1.,
                y: 0.,
                z: 0.,
            },
        );
        let yn = matrix_vec(
            self.matrix,
            Ipoint {
                x: 0.,
                y: 1.,
                z: 0.,
            },
        );
        let zn = matrix_vec(
            self.matrix,
            Ipoint {
                x: 0.,
                y: 0.,
                z: 1.,
            },
        );
        let xo = xn.x * x + yn.x * y + zn.x * z;
        let yo = xn.y * x + yn.y * y + zn.y * z;
        let zo = xn.z * x + yn.z * y + zn.z * z;
        (
            xo * self.xzoom + self.winx as f32 / 2.,
            yo * self.yzoom + self.winy as f32 / 2.,
            zo,
        )
    }
    /// `SlicerFuncs::bandMouseToImage`.
    pub fn band_mouse_to_image(&mut self) {
        let (a, b, c, _) = self.getxyz(self.rb_mouse_x0 as f32, self.rb_mouse_y0 as f32, false);
        let (d, e, f, _) = self.getxyz(self.rb_mouse_x1 as f32, self.rb_mouse_y1 as f32, false);
        let saved = (self.cx, self.cy, self.cz);
        self.cx = self.view.xsize as f32 / 2.;
        self.cy = self.view.ysize as f32 / 2.;
        self.cz = self.view.zsize as f32 / 2.;
        let (x0, y0, _) = self.get_window_coords(a, b, c);
        let (x1, y1, _) = self.get_window_coords(d, e, f);
        let (rx0, ry0, rz0, _) = self.getxyz(x0, self.winy as f32 - 1. - y0, false);
        let (rx1, ry1, rz1, _) = self.getxyz(x1, self.winy as f32 - 1. - y1, false);
        self.rb_image_x0 = rx0;
        self.rb_image_y0 = ry0;
        self.rb_image_z0 = rz0;
        self.rb_image_x1 = rx1;
        self.rb_image_y1 = ry1;
        self.rb_image_z1 = rz1;
        (self.cx, self.cy, self.cz) = saved;
    }
    /// `SlicerFuncs::bandImageToMouse`.
    pub fn band_image_to_mouse(&mut self) {
        let (x0, y0, _) =
            self.get_window_coords(self.rb_image_x0, self.rb_image_y0, self.rb_image_z0);
        let (x1, y1, _) =
            self.get_window_coords(self.rb_image_x1, self.rb_image_y1, self.rb_image_z1);
        self.rb_mouse_x0 = x0.round() as i32;
        self.rb_mouse_y0 = self.winy - 1 - y0.round() as i32;
        self.rb_mouse_x1 = x1.round() as i32;
        self.rb_mouse_y1 = self.winy - 1 - y1.round() as i32;
    }
    /// `SlicerFuncs::rubberBandImageCoords`.
    pub fn rubber_band_image_coords(&self) -> (f32, f32, f32, f32, f32, f32) {
        (
            self.rb_image_x0.min(self.rb_image_x1),
            self.rb_image_x0.max(self.rb_image_x1),
            self.rb_image_y0.min(self.rb_image_y1),
            self.rb_image_y0.max(self.rb_image_y1),
            self.rb_image_z0.min(self.rb_image_z1),
            self.rb_image_z0.max(self.rb_image_z1),
        )
    }
    /// `SlicerFuncs::setSnapshotLimits`.
    pub fn set_snapshot_limits(&mut self, device_pixel_ratio: f32) -> Option<[i32; 4]> {
        if self.rubberband == 0 {
            return None;
        }
        self.band_image_to_mouse();
        let x0 = (self.rb_mouse_x0 + 1).clamp(0, self.winx - 2);
        let y0 = (self.winy - self.rb_mouse_y1).clamp(0, self.winy - 2);
        let x1 = (self.rb_mouse_x1 - if device_pixel_ratio > 1. { 2 } else { 1 })
            .clamp(x0, self.winx - 1);
        let y1 = (self.winy - self.rb_mouse_y0 - if device_pixel_ratio > 1. { 3 } else { 2 })
            .clamp(y0, self.winy - 1);
        Some([x0, y0, x1 + 1 - x0, y1 + 1 - y0])
    }
    /// `SlicerFuncs::getMontageShifts`.
    pub fn get_montage_shifts(
        &self,
        factor: i32,
        scale_factor: f32,
        win: i32,
        mut overlap: i32,
    ) -> MontageShifts {
        let izoom = (self.zoom * scale_factor) as i32;
        let izoom = izoom.max(1);
        overlap += (win - overlap) % izoom;
        let copy_delta = win - overlap;
        let trans_delta = copy_delta as f32 / (self.zoom * scale_factor);
        MontageShifts {
            overlap,
            trans_start: -trans_delta * (factor as f32 - 1.) / 2.,
            trans_delta,
            copy_delta,
            full_size: win * factor - (factor - 1) * overlap,
        }
    }
    /// `SlicerFuncs::montageSnapshot`.  Panel geometry, transformed centers,
    /// state changes, and restoration remain source-local.  The winit
    /// compatibility-GL host owns framebuffer allocation, GL readback/copy,
    /// and image encoding through the named montage boundary methods.
    pub fn montage_snapshot(
        &mut self,
        snap_type: i32,
        n: &mut dyn SlicerNativeBoundary,
    ) -> Result<(), SlicerMontageError> {
        let config = n.montage_config();
        let factor = config.factor;
        let overlap = (3. * self.zoom * factor as f32).ceil() as i32;
        if overlap > self.winx / 4 || overlap > self.winy / 4 {
            return Err(SlicerMontageError::WindowTooSmall);
        }
        let scale_factor = factor as f32
            - ((factor - 1) as f32 * overlap as f32) / self.winx.max(self.winy) as f32;
        let x = self.get_montage_shifts(factor, scale_factor, self.winx, overlap);
        let y = self.get_montage_shifts(factor, scale_factor, self.winy, overlap);
        let plan = SlicerMontagePlan {
            factor,
            scale_factor,
            overlap_x: x.overlap,
            overlap_y: y.overlap,
            full_width: x.full_size,
            full_height: y.full_size,
            snap_type,
        };
        if n.montage_start(plan) {
            return Err(SlicerMontageError::HostAllocationFailed);
        }
        let saved_center = Ipoint {
            x: self.cx,
            y: self.cy,
            z: self.cz,
        };
        let saved_zoom = self.zoom;
        let saved_hq = self.hq;
        self.zoom *= scale_factor;
        self.hq = 1;
        self.manage_buffers();
        self.doing_montage = true;
        let copy_axis = |index: i32, overlap: i32, copy_delta: i32, window: i32| {
            let half = overlap / 2;
            let mut copy = window - overlap;
            let mut from = half;
            let mut to = index * copy_delta + half;
            if index == 0 {
                copy += half;
                to -= half;
                from -= half;
            }
            if index == factor - 1 {
                copy += overlap - half;
            }
            (from, to, copy)
        };
        let mut failed = false;
        for iy in 0..factor {
            for ix in 0..factor {
                self.cx = saved_center.x;
                self.cy = saved_center.y;
                self.cz = saved_center.z;
                let vector = Ipoint {
                    x: x.trans_start + ix as f32 * x.trans_delta,
                    y: y.trans_start + iy as f32 * y.trans_delta,
                    z: 0.,
                };
                self.translate_by_rotated_vec(vector, true);
                let (from_x, to_x, width) = copy_axis(ix, x.overlap, x.copy_delta, self.winx);
                let (from_y, to_y, height) = copy_axis(iy, y.overlap, y.copy_delta, self.winy);
                if n.montage_panel(SlicerMontagePanel {
                    x_index: ix,
                    y_index: iy,
                    center: Ipoint {
                        x: self.cx,
                        y: self.cy,
                        z: self.cz,
                    },
                    copy_from_x: from_x,
                    copy_from_y: from_y,
                    copy_to_x: to_x,
                    copy_to_y: to_y,
                    copy_width: width,
                    copy_height: height,
                }) {
                    failed = true;
                    break;
                }
            }
            if failed {
                break;
            }
        }
        self.hq = saved_hq;
        self.zoom = saved_zoom;
        self.manage_buffers();
        self.cx = saved_center.x;
        self.cy = saved_center.y;
        self.cz = saved_center.z;
        self.doing_montage = false;
        if failed {
            n.montage_abort();
            return Err(SlicerMontageError::HostPanelFailed);
        }
        n.montage_finish(plan);
        self.draw(n);
        Ok(())
    }
    /// `SlicerFuncs::namedSnapshot`.
    pub fn named_snapshot(
        &mut self,
        name: &str,
        format: i32,
        check_convert: bool,
        full_area: bool,
        device_pixel_ratio: f32,
        native: &mut dyn SlicerNativeBoundary,
    ) -> i32 {
        native.update_gl();
        native.set_snapshot_size(self.winx, self.winy);
        let limits = if full_area {
            None
        } else {
            self.set_snapshot_limits(device_pixel_ratio)
        };
        native.named_snapshot(name, "slicer", format, limits, check_convert)
    }
    /// `SlicerFuncs::toggleRubberband`.
    pub fn toggle_rubberband(&mut self, draw_win: bool, n: &mut dyn SlicerNativeBoundary) {
        if self.rubberband != 0 || self.starting_band != 0 {
            self.rubberband = 0;
            self.starting_band = 0;
        } else {
            if self.drawing_arrow {
                self.toggle_arrow(false, n);
            }
            self.starting_band = 1;
            self.limit_no_value = self.view.xsize.max(self.view.ysize).max(self.view.zsize);
            self.band_low_high_limits = [2. * self.limit_no_value as f32; 2];
        }
        let on = self.rubberband + self.starting_band;
        n.set_toggle_state(SLICER_TOGGLE_BAND, on);
        n.enable_low_high_buttons(on != 0);
        n.set_cursor(self.mousemode, true);
        if draw_win {
            self.draw(n);
        }
    }
    /// `SlicerFuncs::findBandAxisRange`.
    pub fn find_band_axis_range(&mut self) -> Option<(f32, f32, f32)> {
        if self.rubberband == 0 {
            return None;
        }
        let saved = (self.cx, self.cy, self.cz);
        self.cx = (self.rb_image_x0 + self.rb_image_x1) / 2.;
        self.cy = (self.rb_image_y0 + self.rb_image_y1) / 2.;
        self.cz = (self.rb_image_z0 + self.rb_image_z1) / 2.;
        let (_, _, _, axis) = self.find_movie_axis(1);
        let (current, start, end) = self.find_axis_limits(axis);
        let delta = [self.view.xsize, self.view.ysize, self.view.zsize][axis] as f32 / 2.;
        (self.cx, self.cy, self.cz) = saved;
        if start < 0 {
            None
        } else {
            Some((current - delta, start as f32 - delta, end as f32 - delta))
        }
    }
    /// `SlicerFuncs::checkBandLowHighLimits`.
    pub fn check_band_low_high_limits(
        &mut self,
        n: &mut dyn SlicerNativeBoundary,
    ) -> Result<[f32; 2], i32> {
        let Some((current, start, end)) = self.find_band_axis_range() else {
            return Err(2);
        };
        let mut values = [0.; 2];
        let mut invalid = false;
        for i in 0..2 {
            if self.band_low_high_limits[i] > self.limit_no_value as f32 {
                invalid = true;
                n.set_low_high_validity(i, SLICER_LIMIT_INVALID);
            } else {
                values[i] = (current + self.band_low_high_limits[i]).clamp(start, end);
                n.set_low_high_validity(
                    i,
                    if current + self.band_low_high_limits[i] < start
                        || current + self.band_low_high_limits[i] > end
                    {
                        SLICER_LIMIT_TRUNCATE
                    } else {
                        SLICER_LIMIT_VALID
                    },
                );
            }
        }
        if invalid { Err(1) } else { Ok(values) }
    }
    /// `SlicerFuncs::setBandLowHighLimit` (non-Shift half; Shift movement is Qt input state).
    pub fn set_band_low_high_limit(&mut self, which: usize, n: &mut dyn SlicerNativeBoundary) {
        let (_, _, _, axis) = self.find_movie_axis(1);
        self.band_low_high_limits[which] = self.current_main_axis_distance(axis);
        let _ = self.check_band_low_high_limits(n);
    }
    /// `SlicerFuncs::checkPlugUseMouse`.  Qt's `QMouseEvent` is intentionally
    /// consumed by the winit host; source coordinates and return-bit redraw
    /// handling stay in this translation unit.
    pub fn check_plug_use_mouse(
        &mut self,
        x: i32,
        y: i32,
        button1: i32,
        button2: i32,
        button3: i32,
        n: &mut dyn SlicerNativeBoundary,
    ) -> i32 {
        let (xm, ym, _, _) = self.getxyz(x as f32, y as f32, false);
        let ifdraw = n.plugin_handle_mouse(xm, ym, button1, button2, button3);
        if ifdraw & 2 != 0 {
            self.draw(n);
        }
        ifdraw
    }
    /// `SlicerFuncs::rotateVolCommand`.  The command construction and all
    /// slicer-space geometry are source-local; the image-file padding query is
    /// supplied by the normal 3dmod host through `SlicerNativeBoundary`.
    pub fn rotate_vol_command(
        &mut self,
        n: &mut dyn SlicerNativeBoundary,
    ) -> Result<String, RotateVolCommandError> {
        if self.starting_band != 0 {
            return Err(RotateVolCommandError::StartingBand);
        }
        let true_limits = match self.check_band_low_high_limits(n) {
            Ok(limits) => limits,
            Err(2) => return Err(RotateVolCommandError::BandOutsideVolume),
            Err(_) => return Err(RotateVolCommandError::BandLimitsUnset),
        };
        if self.get_z_scale_before() != 1. {
            return Err(RotateVolCommandError::UnequalBinning);
        }

        let (current, _, _) = self
            .find_band_axis_range()
            .ok_or(RotateVolCommandError::BandOutsideVolume)?;
        let (_, normal, axis_component) = self.get_normal_and_main_component();
        if axis_component == 0. {
            return Err(RotateVolCommandError::BandOutsideVolume);
        }
        let first_delta = (true_limits[0] - current) / axis_component;
        let second_delta = (true_limits[1] - current) / axis_component;
        let low_delta = first_delta.min(second_delta);
        let high_delta = first_delta.max(second_delta);
        let bin = self.view.xybin;
        let zsize = bin * ((high_delta - low_delta).round() as i32 + 1);
        let mid_delta = 0.5 * (low_delta + high_delta);
        let mut xcen = 0.5 * (self.rb_image_x0 + self.rb_image_x1) + mid_delta * normal.x;
        let mut ycen = 0.5 * (self.rb_image_y0 + self.rb_image_y1) + mid_delta * normal.y;
        let mut zcen = 0.5 * (self.rb_image_z0 + self.rb_image_z1) + mid_delta * normal.z;
        let padding = n
            .image_padding(ycen.round() as i32, zcen.round() as i32, self.view.cur_time)
            .ok_or(RotateVolCommandError::ImagePaddingUnavailable)?;

        xcen = bin as f32 * (xcen + padding.llx as f32 - padding.xpad as f32);
        let (alpha, beta, gamma);
        if self.view.image_axis == 3 {
            ycen = bin as f32 * (ycen + padding.lly as f32 - padding.ypad as f32);
            zcen = bin as f32 * (zcen + padding.llz as f32 - padding.zpad as f32) + 0.5;
            [alpha, beta, gamma] = self.tang;
        } else {
            let temp = bin as f32
                * ((self.view.zsize as f32 - 1. - zcen) + padding.lly as f32 - padding.ypad as f32)
                + 0.5;
            zcen = bin as f32 * (ycen + padding.llz as f32 - padding.zpad as f32);
            ycen = temp;
            self.set_forward_matrix();
            [alpha, beta, gamma] =
                natural_angles(matrix_mul(rotation_matrix(-90., 0., 0.), self.matrix));
        }

        self.band_image_to_mouse();
        let xsize = bin
            * 2.max(((self.rb_mouse_x1 + 1 - self.rb_mouse_x0) as f32 / self.xzoom).round() as i32);
        let ysize = bin
            * 2.max(((self.rb_mouse_y1 + 1 - self.rb_mouse_y0) as f32 / self.yzoom).round() as i32);
        Ok(format!(
            "-siz {xsize},{ysize},{zsize} -cen {xcen:.2},{ycen:.2},{zcen:.2} -ang {gamma:.2},{beta:.2},{alpha:.2}"
        ))
    }
    /// `SlicerFuncs::currentMainAxisDistance`.
    pub fn current_main_axis_distance(&mut self, axis: usize) -> f32 {
        let saved = (self.cx, self.cy, self.cz);
        self.cx = self.view.xsize as f32 / 2.;
        self.cy = self.view.ysize as f32 / 2.;
        self.cz = self.view.zsize as f32 / 2.;
        let (wx, wy, _) = self.get_window_coords(saved.0, saved.1, saved.2);
        let (x, y, z, _) = self.getxyz(wx, self.winy as f32 - 1. - wy, false);
        (self.cx, self.cy, self.cz) = saved;
        [saved.0 - x, saved.1 - y, saved.2 - z][axis]
    }
    /// `SlicerFuncs::resizeBandToWindow`.
    pub fn resize_band_to_window(&mut self, n: &mut dyn SlicerNativeBoundary) {
        self.band_image_to_mouse();
        let dx = self.rb_mouse_x1 as f32 - 1. - self.rb_mouse_x0 as f32;
        let dy = self.rb_mouse_y1 as f32 - 1. - self.rb_mouse_y0 as f32;
        if dx != 0. && dy != 0. {
            self.zoom *= (self.winx as f32 / dx).min(self.winx as f32 / dy);
        }
        let (axis, normal, axis_component) = self.get_normal_and_main_component();
        if axis_component != 0. {
            let current = self.current_main_axis_distance(axis) / axis_component;
            self.cx = ((self.rb_image_x0 + self.rb_image_x1) / 2. + current * normal.x)
                .clamp(0., self.view.xsize as f32);
            self.cy = ((self.rb_image_y0 + self.rb_image_y1) / 2. + current * normal.y)
                .clamp(0., self.view.ysize as f32);
            self.cz = ((self.rb_image_z0 + self.rb_image_z1) / 2. + current * normal.z)
                .clamp(0., self.view.zsize as f32 - 0.5);
            if self.locked == 0 {
                self.view.zmouse = self.cz;
            }
        }
        n.change_center_if_linked(self);
        self.show_slice(n);
    }
    /// `SlicerFuncs::resizeToFit`, excluding the native dialog manager's
    /// `diaLimitWindowSize`/`diaLimitWindowPos` and actual window move.
    pub fn resize_to_fit(
        &mut self,
        window_pixels: (i32, i32),
        window_position: (i32, i32),
        device_pixel_ratio: f32,
        n: &mut dyn SlicerNativeBoundary,
    ) -> Option<SlicerResizePlan> {
        if device_pixel_ratio <= 0. || self.rubberband == 0 {
            return None;
        }
        let width = (window_pixels.0 as f32 * device_pixel_ratio + 0.5) as i32;
        let height = (window_pixels.1 as f32 * device_pixel_ratio + 0.5) as i32;
        self.band_image_to_mouse();
        let device_width = self.rb_mouse_x1 - 1 - self.rb_mouse_x0 + width - self.winx;
        let device_height = self.rb_mouse_y1 - 1 - self.rb_mouse_y0 + height - self.winy;
        let (axis, normal, axis_component) = self.get_normal_and_main_component();
        if axis_component != 0. {
            let current = self.current_main_axis_distance(axis) / axis_component;
            self.cx = ((self.rb_image_x0 + self.rb_image_x1) / 2. + current * normal.x)
                .clamp(0., self.view.xsize as f32);
            self.cy = ((self.rb_image_y0 + self.rb_image_y1) / 2. + current * normal.y)
                .clamp(0., self.view.ysize as f32);
            self.cz = ((self.rb_image_z0 + self.rb_image_z1) / 2. + current * normal.z)
                .clamp(0., self.view.zsize as f32 - 0.5);
            if self.locked == 0 {
                self.view.zmouse = self.cz;
            }
        }
        self.toggle_rubberband(false, n);
        let new_width = (device_width as f32 / device_pixel_ratio + 0.5) as i32;
        let new_height = (device_height as f32 / device_pixel_ratio + 0.5) as i32;
        let plan = SlicerResizePlan {
            width: new_width,
            height: new_height,
            x: window_position.0 + width / 2 - new_width / 2,
            y: window_position.1 + height / 2 - new_height / 2,
        };
        n.change_center_if_linked(self);
        self.show_slice(n);
        Some(plan)
    }
    /// `SlicerFuncs::fixangle`.
    pub fn fixangle(mut angle: f64) -> f64 {
        let r = std::f64::consts::PI / 180.;
        if angle <= -180. * r {
            angle += 360. * r;
        }
        if angle > 180. * r {
            angle -= 360. * r;
        }
        angle
    }
    /// `SlicerFuncs::setForwardMatrix`.
    pub fn set_forward_matrix(&mut self) {
        self.matrix = rotation_matrix(self.tang[0], self.tang[1], self.tang[2]);
    }
    /// `SlicerFuncs::setInverseMatrix`.
    pub fn set_inverse_matrix(&mut self) {
        self.matrix = rotation_matrix(-self.tang[0], -self.tang[1], -self.tang[2]);
    }
    /// Original: `getNormalToPlane`.  Both upstream overloads use this return form.
    pub fn normal_to_plane(&mut self) -> Ipoint {
        self.set_inverse_matrix();
        let mut n = matrix_vec(
            self.matrix,
            Ipoint {
                x: 0.,
                y: 0.,
                z: 1.,
            },
        );
        n.z /= self.get_z_scale_before();
        n.normalize();
        n
    }
    /// Original: `getNormalAndMainComponent`.
    pub fn get_normal_and_main_component(&mut self) -> (usize, Ipoint, f32) {
        let (_, _, _, axis) = self.find_movie_axis(1);
        let normal = self.normal_to_plane();
        let component = [normal.x, normal.y, normal.z][axis];
        (axis, normal, component)
    }
    /// `SlicerFuncs::translateByRotatedVec`.
    pub fn translate_by_rotated_vec(&mut self, vector: Ipoint, outside_ok: bool) -> bool {
        self.set_inverse_matrix();
        let mut rotated = matrix_vec(self.matrix, vector);
        rotated.z /= self.get_z_scale_before();
        if vector.z != 0. {
            rotated.normalize();
        }
        let x = self.cx + rotated.x;
        let y = self.cy + rotated.y;
        let z = self.cz + rotated.z;
        if outside_ok
            || (x >= 0.
                && x < self.view.xsize as f32
                && y >= 0.
                && y < self.view.ysize as f32
                && z >= 0.
                && z < self.view.zsize as f32 - 0.5)
        {
            self.cx = x;
            self.cy = y;
            self.cz = z;
            true
        } else {
            false
        }
    }
    /// `SlicerFuncs::setAnglesFromPoints`.
    pub fn set_angles_from_points(&mut self, p1: Ipoint, p2: Ipoint, axis: usize) {
        let mut n = Ipoint {
            x: p2.x - p1.x,
            y: p2.y - p1.y,
            z: (p2.z - p1.z) * self.get_z_scale_before(),
        };
        if n == Ipoint::default() {
            return;
        }
        n.normalize();
        let eps = 1.0e-4;
        let r = std::f32::consts::PI / 180.;
        let a: Ipoint;
        if axis == 0 {
            let az = if n.x.abs() > eps || n.y.abs() > eps {
                -n.y.atan2(n.x)
            } else {
                0.
            };
            let val = n.x * az.cos() - n.y * az.sin();
            a = Ipoint {
                x: 0.,
                y: Self::fixangle((90. * r - val.atan2(n.z)) as f64) as f32,
                z: az,
            };
        } else if axis == 1 {
            let az = if n.x.abs() > eps || n.y.abs() > eps {
                Self::fixangle((90. * r - n.y.atan2(n.x)) as f64) as f32
            } else {
                0.
            };
            let val = n.x * az.sin() + n.y * az.cos();
            a = Ipoint {
                x: -n.z.atan2(val),
                y: 0.,
                z: az,
            };
        } else {
            let az = if n.x.abs() > eps || n.y.abs() > eps {
                if n.y >= 0. {
                    n.x.atan2(n.y)
                } else {
                    -n.x.atan2(-n.y)
                }
            } else {
                0.
            };
            let val = n.x * az.sin() + n.y * az.cos();
            a = Ipoint {
                x: if n.z >= 0. {
                    val.atan2(n.z)
                } else {
                    -val.atan2(-n.z)
                },
                y: 0.,
                z: az,
            };
        }
        self.tang = [a.x / r, a.y / r, a.z / r];
    }
    /// `SlicerFuncs::transStep`.
    pub fn trans_step(&mut self) {
        self.set_inverse_matrix();
        let xn = matrix_vec(
            self.matrix,
            Ipoint {
                x: 1.,
                y: 0.,
                z: 0.,
            },
        );
        let yn = matrix_vec(
            self.matrix,
            Ipoint {
                x: 0.,
                y: 1.,
                z: 0.,
            },
        );
        let zn = matrix_vec(
            self.matrix,
            Ipoint {
                x: 0.,
                y: 0.,
                z: 1.,
            },
        );
        self.xstep = [xn.x, xn.y, xn.z];
        self.ystep = [yn.x, yn.y, yn.z];
        self.zstep = [zn.x, zn.y, zn.z];
        let isize = self.winx as f32 / self.zoom;
        let jsize = self.winy as f32 / self.zoom;
        let zs = 1. / self.get_z_scale_before();
        self.xo = self.cx - isize / 2. * xn.x - jsize / 2. * yn.x;
        self.yo = self.cy - isize / 2. * xn.y - jsize / 2. * yn.y;
        self.zo = self.cz - isize / 2. * xn.z * zs - jsize / 2. * yn.z * zs;
    }
    /// `SlicerFuncs::resize`.
    pub fn resize(&mut self, winx: i32, winy: i32) {
        if self.closing == 0 {
            self.winx = winx;
            self.winy = winy;
            self.manage_buffers();
        }
    }
    /// `SlicerFuncs::cubeResize`.
    pub fn cube_resize(&mut self, _winx: i32, _winy: i32) {}
    /// `SlicerFuncs::addCubeToFrame`.  Qt's `SlicerCube` construction and
    /// `QVBoxLayout` margins map to the established winit toolbar host; there
    /// is no source-local geometry or model state beyond this lifecycle call.
    pub fn add_cube_to_frame(&mut self, n: &mut dyn SlicerNativeBoundary) {
        n.add_cube_to_frame();
    }
    /// `SlicerFuncs::manageBuffers`; allocation occurs in this source unit's caller.
    pub fn manage_buffers(&mut self) -> i32 {
        let mut size = (self.winx.max(0) as usize).saturating_mul(self.winy.max(0) as usize);
        if self.hq != 0 && self.zoom < 1. {
            size = ((self.winx as f32 / self.zoom).ceil() as usize + 1)
                * ((self.winy as f32 / self.zoom).ceil() as usize + 1);
        }
        if size <= self.cur_buf_size && (size as f32) > 0.8 * self.cur_buf_size as f32 {
            return 0;
        }
        self.cur_buf_size = size;
        0
    }
    /// `SlicerFuncs::draw`.
    pub fn draw(&mut self, n: &mut dyn SlicerNativeBoundary) {
        if self.doing_draw {
            return;
        }
        self.doing_draw = true;
        n.update_gl();
        n.cube_draw();
        self.need_draw = false;
        self.doing_draw = false;
    }
    /// `SlicerFuncs::drawSelfAndLinked`.
    pub fn draw_self_and_linked(&mut self, n: &mut dyn SlicerNativeBoundary) {
        self.draw(n)
    }
    /// `SlicerFuncs::updateImage`.
    pub fn update_image(&mut self, n: &mut dyn SlicerNativeBoundary) {
        self.image_filled += 1;
        n.update_gl();
        self.image_filled -= 1;
        n.cube_draw();
    }
    /// `SlicerFuncs::paint`.
    pub fn paint(&mut self, n: &mut dyn SlicerNativeBoundary) {
        if self.closing == 0 {
            n.image_draw();
            self.draw_model(n);
            self.draw_current_point(n);
        }
    }
    /// `SlicerFuncs::drawModel`.  The GL projection/model-view sequence is
    /// represented as source-ordered plans, while `model_draw.rs` remains the
    /// Rust implementation of object traversal and the GL host executes the
    /// compatibility-profile calls.
    pub fn draw_model(&mut self, n: &mut dyn SlicerNativeBoundary) {
        let config = n.model_render_config();
        self.set_inverse_matrix();
        let z_scale_before = self.get_z_scale_before();
        let inverse_model_scale = Ipoint {
            x: 1. / self.zoom,
            y: 1. / self.zoom,
            z: 1. / (self.zoom * config.model_z_scale),
        };
        let mut depth = self.depth;
        let pass_count = if config.extra_object_count > 0 { 2 } else { 1 };
        for pass in 0..pass_count {
            depth *= self.zoom;
            depth *= 0.5;
            n.draw_model_plan(SlicerModelRenderPlan {
                projection_depth: depth,
                center: Ipoint {
                    x: self.cx,
                    y: self.cy,
                    z: self.cz,
                },
                scale: Ipoint {
                    x: self.xzoom,
                    y: self.yzoom,
                    z: self.zoom,
                },
                angles: self.tang,
                z_scale_before,
                inverse_model_scale,
                draw_main_model: pass == 0,
                extra_object_mode: pass * if config.extra_cursor_in_window { 2 } else { 1 },
            });
            // The source intentionally expands the second pass's clipping
            // depth after the first model pass.
            depth = 100. * self.winx.max(self.winy) as f32;
        }
    }
    /// `SlicerFuncs::drawCurrentPoint`.  Shared model selection is obtained
    /// as a host snapshot; all plane tests and image-to-window transforms are
    /// retained here before the compatibility-GL host draws the primitives.
    pub fn draw_current_point(&mut self, n: &mut dyn SlicerNativeBoundary) {
        let state = n.current_point_state();
        if !state.draw_cursor {
            return;
        }
        let normal = self.normal_to_plane();
        let distance = |point: Ipoint, center: Ipoint| {
            normal.x * (point.x - center.x)
                + normal.y * (point.y - center.y)
                + normal.z * (point.z - center.z)
        };
        let center = Ipoint {
            x: self.cx,
            y: self.cy,
            z: self.cz,
        };
        let mut primitives = Vec::new();
        if !self.doing_montage {
            primitives.push(SlicerOverlayPrimitive::Plus {
                x: (self.winx as f32 * 0.5) as i32,
                y: (self.winy as f32 * 0.5) as i32,
                size: 5,
                color: SlicerOverlayColor::Endpoint,
            });
        }
        let current = state.current_point;
        if state.movie_mode || current.is_none() || state.contour_points.is_empty() {
            if self.classic == 0 {
                let point = Ipoint {
                    x: self.view.xmouse,
                    y: self.view.ymouse,
                    z: self.view.zmouse,
                };
                let (x, y, _) = self.get_window_coords(point.x, point.y, point.z);
                primitives.push(SlicerOverlayPrimitive::Plus {
                    x: x.round() as i32,
                    y: y.round() as i32,
                    size: state.image_point_size,
                    color: if distance(point, center).abs() < 0.5 {
                        SlicerOverlayColor::CurrentPoint
                    } else {
                        SlicerOverlayColor::Shadow
                    },
                });
            }
        } else if let Some(point) = current {
            let at_end = state.contour_points.len() > 1
                && (point == state.contour_points[0]
                    || point == *state.contour_points.last().unwrap());
            let (x, y, _) = self.get_window_coords(point.x, point.y, point.z);
            primitives.push(SlicerOverlayPrimitive::Circle {
                x: x.round() as i32,
                y: y.round() as i32,
                size: if at_end {
                    state.backup_point_size
                } else {
                    state.model_point_size
                },
                color: if distance(point, center).abs() < 0.5 * self.depth
                    && !state.contour_time_mismatch
                {
                    SlicerOverlayColor::CurrentPoint
                } else {
                    SlicerOverlayColor::Shadow
                },
                line_width: state.model_line_width,
            });
        }
        if !state.contour_time_mismatch && state.contour_points.len() > 1 {
            for (point, color) in [
                (state.contour_points[0], SlicerOverlayColor::Beginning),
                (
                    *state.contour_points.last().unwrap(),
                    SlicerOverlayColor::Endpoint,
                ),
            ] {
                if distance(point, center).abs() < 0.5 * self.depth {
                    let (x, y, _) = self.get_window_coords(point.x, point.y, point.z);
                    primitives.push(SlicerOverlayPrimitive::Circle {
                        x: x.round() as i32,
                        y: y.round() as i32,
                        size: state.model_point_size,
                        color,
                        line_width: state.model_line_width,
                    });
                }
            }
        }
        n.draw_current_point_overlay(&primitives);
    }
    /// `SlicerFuncs::cubePaint`.
    pub fn cube_paint(&mut self, n: &mut dyn SlicerNativeBoundary) {
        if self.closing == 0 {
            n.cube_paint();
        }
    }
    /// `SlicerFuncs::keyInput`: Qt key decoding remains the actual Qt boundary.
    pub fn key_input(&mut self, event: SlicerEvent, n: &mut dyn SlicerNativeBoundary) {
        match event.key {
            43 => self.step_zoom(1, n),
            45 => self.step_zoom(-1, n),
            _ => {}
        }
    }
    /// `SlicerFuncs::keyRelease`.
    pub fn key_release(&mut self, _event: SlicerEvent) {}
    /// `SlicerFuncs::mousePress`.
    pub fn mouse_press(&mut self, _event: SlicerEvent) {}
    /// `SlicerFuncs::mouseRelease`.
    pub fn mouse_release(&mut self, _event: SlicerEvent) {}
    /// `SlicerFuncs::mouseMove`.
    pub fn mouse_move(&mut self, _event: SlicerEvent) {}
    /// `SlicerFuncs::generalEvent`.
    pub fn general_event(&mut self, _event: SlicerEvent) {}
}

/// `slicerOpen`.  Construction is intentionally separate from native window
/// creation: the registry owns the same `SlicerFuncs` lifetime that Qt's
/// `SlicerWindow` owned in the source.
pub fn slicer_open(registry: &mut SlicerRegistry, view: SlicerView, auto_link: i32) -> i32 {
    let mut slicer = SlicerFuncs::new(view, auto_link);
    slicer.view_axis_index = registry.view_axis_index;
    registry.slicers.push(Box::new(slicer));
    0
}

/// `ivwGetTopSlicerMouse`.  `SlicerWindow::mapFromGlobal` and the device
/// pixel conversion are native-window responsibilities; this function owns
/// the source's subsequent slicer-to-image coordinate conversion.  `Err(1)`
/// is the C routine's no-slicer return value.
pub fn ivw_get_top_slicer_mouse(
    registry: &mut SlicerRegistry,
    device_x: i32,
    device_y: i32,
) -> Result<Ipoint, i32> {
    let slicer = registry.slicers.first_mut().ok_or(1)?;
    let (x, y, z, _) = slicer.getxyz(device_x as f32, device_y as f32, true);
    Ok(Ipoint { x, y, z })
}

/// `slicerKey_cb`.  Winit performs key decoding, then passes the stable
/// source-equivalent event and configured hot-slider key through this callback.
pub fn slicer_key_cb(
    slicer: &mut SlicerFuncs,
    released: bool,
    event: SlicerEvent,
    hot_slider_key: i32,
    native: &mut dyn SlicerNativeBoundary,
) {
    if event.key == hot_slider_key {
        return;
    }
    if released {
        slicer.key_release(event);
    } else {
        slicer.key_input(event, native);
    }
}

/// `slicerDraw_cb`.
pub fn slicer_draw_cb(
    slicer: &mut SlicerFuncs,
    drawflag: i32,
    native: &mut dyn SlicerNativeBoundary,
) {
    if slicer.closing == 0 {
        slicer.external_draw(drawflag, native);
    }
}

/// `slicerClose_cb`.  Like `SlicerWindow::close`, this requests native window
/// closure; the host subsequently calls `SlicerFuncs::closing` from its close
/// event to perform source-owned cleanup.
pub fn slicer_close_cb(native: &mut dyn SlicerNativeBoundary) {
    native.close();
}

/// `SlicerFuncs::setCurrentOrNewRow`.  The winit host performs the preceding
/// `ivwControlPriority` call because it owns controller registration; the
/// translated angle form owns the source row/model operation itself.
pub fn set_current_or_new_row(
    slicer: &SlicerFuncs,
    form: &mut SlicerAngleForm,
    new_row: bool,
    native: &mut dyn SlicerAngleNativeBoundary,
) {
    let time = if slicer.time_lock != 0 {
        slicer.time_lock
    } else {
        slicer.view.cur_time
    };
    form.set_current_or_new_row(time, new_row, native);
}

/// `SlicerFuncs::setAnglesFromRow`; see [`set_current_or_new_row`] for the
/// controller-priority boundary.
pub fn set_angles_from_row(
    slicer: &SlicerFuncs,
    form: &mut SlicerAngleForm,
    native: &mut dyn SlicerAngleNativeBoundary,
) {
    let time = if slicer.time_lock != 0 {
        slicer.time_lock
    } else {
        slicer.view.cur_time
    };
    form.set_angles_from_row(time, native);
}

/// `setupLinkedSlicers`, excluding Qt monitor/toolbar placement.  The native
/// winit host lays out the locked windows after this source lifecycle creates
/// them.
pub fn setup_linked_slicers(
    registry: &mut SlicerRegistry,
    view: SlicerView,
    max_linked: i32,
) -> i32 {
    for slicer in &mut registry.slicers {
        slicer.linked = false;
    }
    let num_linked = 3_i32.min(view.num_times).min(max_linked).max(0);
    if num_linked == 0 {
        registry.link_was_limited = false;
        return 0;
    }
    registry.link_was_limited = num_linked < view.num_times;
    let time_offset = if registry.link_was_limited {
        (view.cur_time - 1).min(view.num_times - num_linked).max(0)
    } else {
        0
    };
    for index in 1..=num_linked {
        if slicer_open(registry, view.clone(), index + time_offset) != 0 {
            return -1;
        }
    }
    0
}

/// `SlicerFuncs::synchronizeSlicers`.  The Qt dialog manager in the source is
/// represented by the Rust-owned registry; widget repainting remains at the
/// `SlicerNativeBoundary` owned by the winit host.
pub fn synchronize_slicers(
    registry: &mut SlicerRegistry,
    source_index: usize,
    draw: bool,
    native: &mut dyn SlicerNativeBoundary,
) -> i32 {
    let Some(source) = registry.slicers.get(source_index) else {
        return 0;
    };
    if !source.linked {
        return 0;
    }
    let state = (
        source.tang,
        source.cx,
        source.cy,
        source.cz,
        source.locked,
        source.nslice,
        source.depth,
        source.zoom,
        source.hq,
        source.classic,
        source.fft_mode,
        source.scalez,
    );
    let mut changed = 0;
    for (index, target) in registry.slicers.iter_mut().enumerate() {
        if index == source_index || !target.linked {
            continue;
        }
        let mut need_draw = false;
        if target.tang != state.0 {
            target.tang = state.0;
            native.set_angles(target.tang);
            need_draw = true;
        }
        if state.4 == 0
            && target.locked == 0
            && (target.cx != state.1 || target.cy != state.2 || target.cz != state.3)
        {
            (target.cx, target.cy, target.cz) = (state.1, state.2, state.3);
            need_draw = true;
        }
        if target.nslice != state.5 || target.depth != state.6 {
            (target.nslice, target.depth) = (state.5, state.6);
            target.draw_thick_controls(native);
            need_draw = true;
        }
        if target.zoom != state.7 {
            target.zoom = state.7;
            target.manage_buffers();
            native.set_zoom_text(target.zoom);
            need_draw = true;
        }
        if target.hq != state.8 {
            target.hq = state.8;
            target.manage_buffers();
            native.set_toggle_state(SLICER_TOGGLE_HIGHRES, target.hq);
            need_draw = true;
        }
        if target.classic != state.9 {
            target.classic = state.9;
            target.pending = 0;
            native.set_toggle_state(SLICER_TOGGLE_CENTER, target.classic);
            need_draw = true;
        }
        if target.fft_mode != state.10 {
            target.fft_mode = state.10;
            native.set_toggle_state(SLICER_TOGGLE_FFT, target.fft_mode);
            need_draw = true;
        }
        if target.scalez != state.11 {
            target.scalez = state.11;
            native.set_toggle_state(SLICER_TOGGLE_ZSCALE, target.scalez);
            need_draw = true;
        }
        if need_draw {
            target.need_draw = true;
            if draw {
                target.draw(native);
                target.already_drew = true;
            }
            changed += 1;
        }
    }
    changed
}

/// `slicerAnglesOpen`.
pub fn slicer_angles_open(
    registry: &mut SlicerRegistry,
    native: &mut dyn SlicerNativeBoundary,
) -> i32 {
    if registry.angles_open {
        return 0;
    }
    registry.angles_open = true;
    notify_slicers_of_ang_dia(registry, true, native);
    0
}

/// `slicerAnglesClosing`.
pub fn slicer_angles_closing(registry: &mut SlicerRegistry, native: &mut dyn SlicerNativeBoundary) {
    registry.angles_open = false;
    notify_slicers_of_ang_dia(registry, false, native);
}

/// Private `notifySlicersOfAngDia`.
pub fn notify_slicers_of_ang_dia(
    registry: &mut SlicerRegistry,
    open: bool,
    native: &mut dyn SlicerNativeBoundary,
) {
    for slicer in &mut registry.slicers {
        set_angle_toolbar_state(slicer, open, native);
    }
}

/// Private `setAngleToolbarState`.
pub fn set_angle_toolbar_state(
    slicer: &mut SlicerFuncs,
    open: bool,
    native: &mut dyn SlicerNativeBoundary,
) {
    if !open {
        slicer.continuous = false;
    }
    native.set_angle_toolbar_state(open);
}

/// `slicerPixelViewState`.
pub fn slicer_pixel_view_state(
    registry: &mut SlicerRegistry,
    state: bool,
    native: &mut dyn SlicerNativeBoundary,
) {
    registry.pixel_view_open = state;
    slicer_set_mouse_tracking(registry, native);
}

/// `slicerSetMouseTracking`.
pub fn slicer_set_mouse_tracking(
    registry: &mut SlicerRegistry,
    native: &mut dyn SlicerNativeBoundary,
) {
    let pixel_view_open = registry.pixel_view_open;
    for slicer in &mut registry.slicers {
        slicer.set_mouse_tracking(pixel_view_open, native);
    }
}

/// `slicerReportAngles`.  The native host emits the returned source values
/// with `imodPrintStderr`; an absent top slicer is its source error path.
pub fn slicer_report_angles(registry: &SlicerRegistry) -> Result<[f32; 3], &'static str> {
    registry
        .slicers
        .first()
        .map(|slicer| slicer.tang)
        .ok_or("ERROR: No slicer windows open\n")
}

/// `setTopSlicerAngles`.
pub fn set_top_slicer_angles(
    registry: &mut SlicerRegistry,
    angles: [f32; 3],
    center: Ipoint,
    draw: bool,
    native: &mut dyn SlicerNativeBoundary,
) -> i32 {
    let Some(slicer) = registry.slicers.first_mut() else {
        return 1;
    };
    for axis in 0..3 {
        slicer.tang[axis] = angles[axis].clamp(-S_MAX_ANGLE[axis], S_MAX_ANGLE[axis]);
    }
    slicer.cx = center.x.clamp(0., (slicer.view.xsize - 1) as f32);
    slicer.cy = center.y.clamp(0., (slicer.view.ysize - 1) as f32);
    slicer.cz = center.z.clamp(0., (slicer.view.zsize - 1) as f32);
    native.set_angles(slicer.tang);
    if slicer.locked == 0 {
        slicer.view.xmouse = slicer.cx;
        slicer.view.ymouse = slicer.cy;
        slicer.view.zmouse = slicer.cz;
        if draw {
            // `IMOD_DRAW_XYZ | IMOD_DRAW_SLICE`; model-view coupling belongs
            // to the native host that owns the linked model window.
            native.draw((1 << 1) | (1 << 3));
        }
    } else if draw {
        slicer.draw(native);
        slicer.already_drew = true;
        slicer.show_slice(native);
    }
    0
}

/// `setTopSlicerZoom`.
pub fn set_top_slicer_zoom(
    registry: &mut SlicerRegistry,
    zoom: f32,
    draw: bool,
    native: &mut dyn SlicerNativeBoundary,
) -> i32 {
    let Some(slicer) = registry.slicers.first_mut() else {
        return 1;
    };
    if !(0.005..=200.).contains(&zoom) {
        return 1;
    }
    slicer.set_initial_zoom(zoom);
    if draw {
        slicer.draw(native);
    }
    0
}

/// `setTopSlicerFromModelView`.
pub fn set_top_slicer_from_model_view(
    registry: &mut SlicerRegistry,
    rotation: Ipoint,
    native: &mut dyn SlicerNativeBoundary,
) -> i32 {
    let Some(slicer) = registry.slicers.first_mut() else {
        return 1;
    };
    slicer.set_forward_matrix();
    slicer.tang = [rotation.x, rotation.y, rotation.z];
    native.change_center_if_linked(slicer);
    native.set_angles(slicer.tang);
    slicer.draw(native);
    slicer.already_drew = true;
    // `IMOD_DRAW_XYZ | IMOD_DRAW_SLICE | IMOD_DRAW_SKIPMODV`.
    native.draw((1 << 1) | (1 << 3) | (1 << 5));
    0
}

/// `getTopSlicerAngles`.
pub fn get_top_slicer_angles(registry: &SlicerRegistry) -> Option<([f32; 3], Ipoint, i32)> {
    registry.slicers.first().map(|slicer| {
        (
            slicer.tang,
            Ipoint {
                x: slicer.cx,
                y: slicer.cy,
                z: slicer.cz,
            },
            if slicer.time_lock != 0 {
                slicer.time_lock
            } else {
                slicer.view.cur_time
            },
        )
    })
}

/// `getTopSlicerTime`.
pub fn get_top_slicer_time(registry: &SlicerRegistry) -> Option<(i32, bool)> {
    registry.slicers.first().map(|slicer| {
        (
            if slicer.time_lock != 0 {
                slicer.time_lock
            } else {
                slicer.view.cur_time
            },
            slicer.continuous,
        )
    })
}

/// `slicerViewAxisStepChange`.
pub fn slicer_view_axis_step_change(registry: &mut SlicerRegistry, delta: i32) -> bool {
    let old_index = registry.view_axis_index;
    if delta > 0 && registry.view_axis_index + 1 < S_VIEW_AXIS_STEPS.len() - 1 {
        registry.view_axis_index += 1;
    } else if delta < 0 {
        registry.view_axis_index = registry.view_axis_index.saturating_sub(1);
    }
    if old_index == registry.view_axis_index {
        return false;
    }
    for slicer in &mut registry.slicers {
        slicer.view_axis_index = registry.view_axis_index;
    }
    true
}

/// `slicerNewTime`.
pub fn slicer_new_time(
    registry: &SlicerRegistry,
    refresh: bool,
    model_view_linked: bool,
    model_view_draws_slicer_plane: bool,
    native: &mut dyn SlicerNativeBoundary,
) {
    if registry.angles_open {
        native.slicer_new_time(refresh);
    }
    if model_view_linked || model_view_draws_slicer_plane {
        native.model_view_slicer_update();
    }
}

/// `getSlicerThicknessScaling`.
pub fn get_slicer_thickness_scaling(registry: &SlicerRegistry) -> i32 {
    registry.scale_thick
}

/// `slicerCubicFillin`.  `int_data` chooses the source's `int *` path; in
/// Rust it is explicit to prevent aliasing a `u16` buffer as `i32`.
pub fn slicer_cubic_fillin_u16(
    data: &mut [u16],
    winx: usize,
    winy: usize,
    izoom: usize,
    ilim: usize,
    jlim: usize,
    minval: i32,
    maxval: i32,
) {
    if izoom == 0 || winx == 0 || winy == 0 {
        return;
    }
    for jfill in 0..izoom {
        let dy = jfill as f32 / izoom as f32;
        let dysq = dy * dy;
        let dycub = dy * dysq;
        let fyp = 2. * dysq - dycub - dy;
        let fy = 1. + dycub - 2. * dysq;
        let fyn = dy + dysq - dycub;
        let fyn2 = dycub - dysq;
        let first = if jfill == 0 { 1 } else { 0 };
        for ifill in first..izoom {
            let dx = ifill as f32 / izoom as f32;
            let dxsq = dx * dx;
            let dxcub = dx * dxsq;
            let fxp = 2. * dxsq - dxcub - dx;
            let fx = 1. + dxcub - 2. * dxsq;
            let fxn = dx + dxsq - dxcub;
            let fxn2 = dxcub - dxsq;
            for j in (izoom + jfill..jlim).step_by(izoom) {
                if j < izoom || j + 2 * izoom > winy {
                    continue;
                }
                for i in (izoom + ifill..ilim).step_by(izoom) {
                    if i < izoom || i + 2 * izoom > winx {
                        continue;
                    }
                    let sample = |xx: usize, yy: usize| data[xx + yy * winx] as f32;
                    let row = |yy: usize| {
                        fxp * sample(i - izoom, yy)
                            + fx * sample(i - ifill, yy)
                            + fxn * sample(i - ifill + izoom, yy)
                            + fxn2 * sample(i - ifill + 2 * izoom, yy)
                    };
                    let value = (fyp * row(j - jfill - izoom)
                        + fy * row(j - jfill)
                        + fyn * row(j - jfill + izoom)
                        + fyn2 * row(j - jfill + 2 * izoom))
                    .clamp(minval as f32, maxval as f32);
                    data[i + j * winx] = (value + 0.5) as u16;
                }
            }
        }
    }
}

fn rotation_matrix(x: f32, y: f32, z: f32) -> [[f32; 3]; 3] {
    let (x, y, z) = (x.to_radians(), y.to_radians(), z.to_radians());
    let (sx, cx) = x.sin_cos();
    let (sy, cy) = y.sin_cos();
    let (sz, cz) = z.sin_cos();
    matrix_mul(
        matrix_mul(
            [[cz, -sz, 0.], [sz, cz, 0.], [0., 0., 1.]],
            [[cy, 0., sy], [0., 1., 0.], [-sy, 0., cy]],
        ),
        [[1., 0., 0.], [0., cx, -sx], [0., sx, cx]],
    )
}
fn matrix_mul(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut r = [[0.; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    r
}
fn matrix_vec(m: [[f32; 3]; 3], p: Ipoint) -> Ipoint {
    Ipoint {
        x: m[0][0] * p.x + m[0][1] * p.y + m[0][2] * p.z,
        y: m[1][0] * p.x + m[1][1] * p.y + m[1][2] * p.z,
        z: m[2][0] * p.x + m[2][1] * p.y + m[2][2] * p.z,
    }
}
fn natural_angles(m: [[f32; 3]; 3]) -> [f32; 3] {
    let y = (-m[2][0]).asin();
    let x = m[2][1].atan2(m[2][2]);
    let z = m[1][0].atan2(m[0][0]);
    [x.to_degrees(), y.to_degrees(), z.to_degrees()]
}

/// `printmat` (`slicer.cpp` debug-only path).  The C source prints the 16
/// row-major matrix entries in four `%5.2f` rows and returns zero.
pub fn printmat(matrix: [[f32; 4]; 4]) -> (i32, String) {
    let mut output = String::new();
    for row in matrix {
        output.push_str(&format!(
            "{:5.2} {:5.2} {:5.2} {:5.2}\n",
            row[0], row[1], row[2], row[3]
        ));
    }
    (0, output)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct N;
    impl SlicerNativeBoundary for N {}
    #[test]
    fn source_coordinate_round_trip_at_zero_rotation() {
        let mut s = SlicerFuncs::new(
            SlicerView {
                xsize: 100,
                ysize: 100,
                zsize: 50,
                xmouse: 50.,
                ymouse: 50.,
                zmouse: 25.,
                ..Default::default()
            },
            0,
        );
        s.resize(200, 100);
        let (x, y, z) = s.get_window_coords(50., 50., 25.);
        assert_eq!((x.round(), y.round(), z.round()), (100., 50., 0.));
        let (ix, iy, iz, _) = s.getxyz(x, s.winy as f32 - 1. - y, true);
        assert!((ix - 50.).abs() < 1.);
        assert!((iy - 50.).abs() < 1.);
        assert!((iz - 25.).abs() < 1.);
    }
    #[test]
    fn normal_and_axis_limits_are_source_geometry() {
        let mut s = SlicerFuncs::new(
            SlicerView {
                xsize: 11,
                ysize: 13,
                zsize: 17,
                xmouse: 5.,
                ymouse: 6.,
                zmouse: 8.,
                ..Default::default()
            },
            0,
        );
        assert_eq!(
            s.normal_to_plane(),
            Ipoint {
                x: 0.,
                y: 0.,
                z: 1.
            }
        );
        assert_eq!(s.find_axis_limits(2), (8., 0, 16));
        let (axis, normal, component) = s.get_normal_and_main_component();
        assert_eq!(axis, 2);
        assert_eq!(
            normal,
            Ipoint {
                x: 0.,
                y: 0.,
                z: 1.
            }
        );
        assert_eq!(component, 1.);
    }
    #[test]
    fn cubic_fills_grid_intermediate_values() {
        let mut d = vec![0_u16; 7 * 7];
        for y in (0..7).step_by(2) {
            for x in (0..7).step_by(2) {
                d[x + y * 7] = (x + y * 10) as u16;
            }
        }
        slicer_cubic_fillin_u16(&mut d, 7, 7, 2, 7, 7, 0, 1000);
        assert_ne!(d[3 + 3 * 7], 0);
    }
    #[test]
    fn toggle_uses_source_exclusive_arrow_band_state() {
        let mut s = SlicerFuncs::new(SlicerView::default(), 0);
        let mut n = N;
        s.toggle_arrow(false, &mut n);
        assert!(s.arrow_on);
        s.toggle_rubberband(false, &mut n);
        assert!(!s.arrow_on);
        assert_eq!(s.starting_band, 1);
    }
    #[test]
    fn registry_coordinates_pixel_tracking_and_top_slicer_angles() {
        struct Tracking(Vec<bool>);
        impl SlicerNativeBoundary for Tracking {
            fn set_mouse_tracking(&mut self, enabled: bool) {
                self.0.push(enabled);
            }
        }
        let mut registry = SlicerRegistry::default();
        assert_eq!(slicer_open(&mut registry, SlicerView::default(), 0), 0);
        let mut native = Tracking(Vec::new());
        slicer_pixel_view_state(&mut registry, true, &mut native);
        assert_eq!(native.0, vec![true]);
        assert_eq!(
            set_top_slicer_angles(
                &mut registry,
                [120., -200., 20.],
                Ipoint {
                    x: -2.,
                    y: 8.,
                    z: 9.
                },
                false,
                &mut native,
            ),
            0
        );
        assert_eq!(slicer_report_angles(&registry), Ok([90., -180., 20.]));
        assert_eq!(registry.slicers[0].cx, 0.);
        assert_eq!(registry.slicers[0].cy, 0.);
        assert_eq!(registry.slicers[0].cz, 0.);
        assert_eq!(
            set_top_slicer_zoom(&mut registry, 2., false, &mut native),
            0
        );
        assert_eq!(registry.slicers[0].zoom, 2.);
        assert_eq!(get_top_slicer_time(&registry), Some((1, false)));
        assert_eq!(slicer_view_axis_step_change(&mut registry, 1), true);
        assert_eq!(registry.slicers[0].view_axis_step_size(), 3.);
    }
    #[test]
    fn angle_dialog_lifecycle_updates_every_registered_slicer() {
        struct Toolbar(Vec<bool>);
        impl SlicerNativeBoundary for Toolbar {
            fn set_angle_toolbar_state(&mut self, open: bool) {
                self.0.push(open);
            }
        }
        let mut registry = SlicerRegistry::default();
        slicer_open(&mut registry, SlicerView::default(), 0);
        slicer_open(&mut registry, SlicerView::default(), 0);
        registry.slicers[0].continuous = true;
        registry.slicers[1].continuous = true;
        let mut native = Toolbar(Vec::new());
        assert_eq!(slicer_angles_open(&mut registry, &mut native), 0);
        assert!(registry.angles_open);
        slicer_angles_closing(&mut registry, &mut native);
        assert!(!registry.angles_open);
        assert!(!registry.slicers[0].continuous);
        assert!(!registry.slicers[1].continuous);
        assert_eq!(native.0, vec![true, true, false, false]);
    }
    #[test]
    fn new_time_notifies_only_the_source_selected_consumers() {
        struct Time(Vec<&'static str>);
        impl SlicerNativeBoundary for Time {
            fn slicer_new_time(&mut self, _: bool) {
                self.0.push("angles");
            }
            fn model_view_slicer_update(&mut self) {
                self.0.push("model-view");
            }
        }
        let mut registry = SlicerRegistry::default();
        let mut native = Time(Vec::new());
        slicer_new_time(&registry, true, true, false, &mut native);
        assert_eq!(native.0, vec!["model-view"]);
        registry.angles_open = true;
        slicer_new_time(&registry, false, false, true, &mut native);
        assert_eq!(native.0, vec!["model-view", "angles", "model-view"]);
        assert_eq!(get_slicer_thickness_scaling(&registry), 1);
    }
    #[test]
    fn linked_slicers_follow_source_time_limit_and_current_time_offset() {
        let mut registry = SlicerRegistry::default();
        slicer_open(&mut registry, SlicerView::default(), 0);
        registry.slicers[0].linked = true;
        let view = SlicerView {
            num_times: 8,
            cur_time: 7,
            ..Default::default()
        };
        assert_eq!(setup_linked_slicers(&mut registry, view, 2), 0);
        assert!(registry.link_was_limited);
        assert!(!registry.slicers[0].linked);
        assert_eq!(registry.slicers.len(), 3);
        assert_eq!(registry.slicers[1].time_lock, 7);
        assert_eq!(registry.slicers[2].time_lock, 8);
        assert!(registry.slicers[1].linked && registry.slicers[2].linked);
    }
    #[test]
    fn linked_slicer_synchronization_copies_unlocked_source_state() {
        let mut registry = SlicerRegistry::default();
        slicer_open(&mut registry, SlicerView::default(), 1);
        slicer_open(&mut registry, SlicerView::default(), 2);
        let source = &mut registry.slicers[0];
        source.tang = [15., -20., 30.];
        (source.cx, source.cy, source.cz) = (4., 5., 6.);
        source.nslice = 3;
        source.depth = 2.5;
        source.zoom = 1.75;
        source.hq = 1;
        source.classic = 1;
        source.fft_mode = 1;
        source.scalez = 1;
        let mut native = N;
        assert_eq!(synchronize_slicers(&mut registry, 0, false, &mut native), 1);
        let target = &registry.slicers[1];
        assert_eq!(target.tang, [15., -20., 30.]);
        assert_eq!((target.cx, target.cy, target.cz), (4., 5., 6.));
        assert_eq!((target.nslice, target.depth, target.zoom), (3, 2.5, 1.75));
        assert_eq!(
            (target.hq, target.classic, target.fft_mode, target.scalez),
            (1, 1, 1, 1)
        );
        assert!(target.need_draw);
    }
    #[test]
    fn axis_moves_accumulate_against_the_original_z_mouse_position() {
        struct Bound(usize);
        impl SlicerNativeBoundary for Bound {
            fn bind_mouse(&mut self) {
                self.0 += 1;
            }
        }
        let mut slicer = SlicerFuncs::new(
            SlicerView {
                zmouse: 5.,
                ..Default::default()
            },
            0,
        );
        let mut native = Bound(0);
        slicer.set_zmouse_for_axis_move(1., &mut native);
        assert_eq!(slicer.view.zmouse, 6.);
        slicer.set_zmouse_for_axis_move(1., &mut native);
        assert_eq!(slicer.view.zmouse, 7.);
        assert_eq!(native.0, 2);
    }
    #[test]
    fn top_slicer_mouse_uses_the_slicer_coordinate_transform() {
        let mut registry = SlicerRegistry::default();
        assert_eq!(ivw_get_top_slicer_mouse(&mut registry, 1, 1), Err(1));
        let view = SlicerView {
            xsize: 100,
            ysize: 100,
            zsize: 50,
            xmouse: 50.,
            ymouse: 50.,
            zmouse: 25.,
            ..Default::default()
        };
        slicer_open(&mut registry, view, 0);
        registry.slicers[0].resize(200, 100);
        let point = ivw_get_top_slicer_mouse(&mut registry, 100, 49).unwrap();
        assert!((point.x - 50.).abs() < 1.);
        assert!((point.y - 50.).abs() < 1.);
        assert!((point.z - 25.).abs() < 1.);
    }
    #[test]
    fn slicer_snapshot_sets_size_and_selects_rubberband_limits() {
        #[derive(Default)]
        struct Snapshot {
            updates: usize,
            size: Option<(i32, i32)>,
            call: Option<(String, String, i32, Option<[i32; 4]>, bool)>,
        }
        impl SlicerNativeBoundary for Snapshot {
            fn update_gl(&mut self) {
                self.updates += 1;
            }
            fn set_snapshot_size(&mut self, width: i32, height: i32) {
                self.size = Some((width, height));
            }
            fn named_snapshot(
                &mut self,
                name: &str,
                window_name: &str,
                format: i32,
                limits: Option<[i32; 4]>,
                check_convert: bool,
            ) -> i32 {
                self.call = Some((
                    name.to_owned(),
                    window_name.to_owned(),
                    format,
                    limits,
                    check_convert,
                ));
                7
            }
        }
        let mut slicer = SlicerFuncs::new(SlicerView::default(), 0);
        slicer.resize(100, 80);
        slicer.rubberband = 1;
        slicer.rb_mouse_x0 = 10;
        slicer.rb_mouse_x1 = 40;
        slicer.rb_mouse_y0 = 20;
        slicer.rb_mouse_y1 = 50;
        let mut native = Snapshot::default();
        assert_eq!(
            slicer.named_snapshot("slice.png", 3, true, false, 1., &mut native),
            7
        );
        assert_eq!(native.updates, 1);
        assert_eq!(native.size, Some((100, 80)));
        let call = native.call.unwrap();
        assert_eq!(
            (call.0, call.1, call.2, call.4),
            ("slice.png".into(), "slicer".into(), 3, true)
        );
        assert!(call.3.is_some());
    }
    #[test]
    fn montage_shifts_keep_panel_grid_coordinates_aligned_at_zoom() {
        let mut slicer = SlicerFuncs::new(SlicerView::default(), 0);
        slicer.zoom = 2.5;
        let shifts = slicer.get_montage_shifts(3, 1.6, 100, 11);
        // `izoom = 4`; overlap grows by one so the 88-pixel panel spacing
        // stays on the sampled grid.
        assert_eq!(shifts.overlap, 12);
        assert_eq!(shifts.copy_delta, 88);
        assert_eq!(shifts.full_size, 276);
        assert_eq!(shifts.trans_delta, 22.);
        assert_eq!(shifts.trans_start, -22.);
    }
    #[test]
    fn montage_snapshot_visits_source_grid_and_restores_slicer_state() {
        #[derive(Default)]
        struct Montage {
            panels: Vec<SlicerMontagePanel>,
            finished: Option<SlicerMontagePlan>,
            updates: usize,
        }
        impl SlicerNativeBoundary for Montage {
            fn montage_config(&mut self) -> SlicerMontageConfig {
                SlicerMontageConfig {
                    factor: 2,
                    ..Default::default()
                }
            }
            fn montage_start(&mut self, _plan: SlicerMontagePlan) -> bool {
                false
            }
            fn montage_panel(&mut self, panel: SlicerMontagePanel) -> bool {
                self.panels.push(panel);
                false
            }
            fn montage_finish(&mut self, plan: SlicerMontagePlan) {
                self.finished = Some(plan);
            }
            fn update_gl(&mut self) {
                self.updates += 1;
            }
        }
        let mut slicer = SlicerFuncs::new(
            SlicerView {
                xsize: 100,
                ysize: 100,
                zsize: 20,
                xmouse: 50.,
                ymouse: 50.,
                zmouse: 10.,
                ..Default::default()
            },
            0,
        );
        slicer.resize(100, 100);
        slicer.zoom = 1.;
        let saved = (slicer.cx, slicer.cy, slicer.cz, slicer.zoom, slicer.hq);
        let mut native = Montage::default();
        assert_eq!(slicer.montage_snapshot(2, &mut native), Ok(()));
        assert_eq!(native.panels.len(), 4);
        assert_eq!(
            native
                .panels
                .iter()
                .map(|panel| (panel.x_index, panel.y_index))
                .collect::<Vec<_>>(),
            vec![(0, 0), (1, 0), (0, 1), (1, 1)]
        );
        assert_eq!(native.finished.unwrap().snap_type, 2);
        assert_eq!(
            (slicer.cx, slicer.cy, slicer.cz, slicer.zoom, slicer.hq),
            saved
        );
        assert!(!slicer.doing_montage);
        assert_eq!(native.updates, 1);
    }
    #[test]
    fn movie_limits_restart_on_the_new_dominant_slicer_axis() {
        #[derive(Default)]
        struct Movie {
            moves: Vec<(i32, i32, i32)>,
            limits: Vec<(i32, i32, i32)>,
        }
        impl SlicerNativeBoundary for Movie {
            fn movie(&mut self, x: i32, y: i32, z: i32) {
                self.moves.push((x, y, z));
            }
            fn set_movie_limits(&mut self, axis: i32, start: i32, end: i32) {
                self.limits.push((axis, start, end));
            }
        }
        let mut slicer = SlicerFuncs::new(
            SlicerView {
                xsize: 10,
                ysize: 12,
                zsize: 14,
                xmouse: 4.,
                ymouse: 5.,
                zmouse: 6.,
                xmovie: 1,
                movie_started_by_slicer: true,
                ..Default::default()
            },
            0,
        );
        let mut native = Movie::default();
        slicer.check_movie_limits(&mut native);
        assert_eq!(native.moves, vec![(0, 0, 1)]);
        assert_eq!(native.limits, vec![(2, 0, 13)]);
    }
    #[test]
    fn controller_key_and_close_callbacks_keep_source_dispatch_order() {
        #[derive(Default)]
        struct Events {
            zooms: Vec<f32>,
            closes: usize,
        }
        impl SlicerNativeBoundary for Events {
            fn set_zoom_text(&mut self, zoom: f32) {
                self.zooms.push(zoom);
            }
            fn close(&mut self) {
                self.closes += 1;
            }
        }
        let mut slicer = SlicerFuncs::new(SlicerView::default(), 0);
        let mut native = Events::default();
        slicer_key_cb(
            &mut slicer,
            false,
            SlicerEvent {
                key: 43,
                ..Default::default()
            },
            99,
            &mut native,
        );
        assert_eq!(native.zooms, vec![1.25]);
        slicer_key_cb(
            &mut slicer,
            false,
            SlicerEvent {
                key: 99,
                ..Default::default()
            },
            99,
            &mut native,
        );
        assert_eq!(native.zooms, vec![1.25]);
        slicer_close_cb(&mut native);
        assert_eq!(native.closes, 1);
    }
    #[test]
    fn contour_plane_fit_updates_angles_and_center_before_redraw() {
        #[derive(Default)]
        struct Fit {
            angles: Option<[f32; 3]>,
            bound: usize,
            draw: Option<i32>,
        }
        impl SlicerNativeBoundary for Fit {
            fn set_angles(&mut self, angles: [f32; 3]) {
                self.angles = Some(angles);
            }
            fn bind_mouse(&mut self) {
                self.bound += 1;
            }
            fn draw(&mut self, flags: i32) {
                self.draw = Some(flags);
            }
        }
        let contour = Icont {
            pts: vec![
                ModelPoint {
                    x: 2.,
                    y: 4.,
                    z: 6.,
                },
                ModelPoint {
                    x: 8.,
                    y: 4.,
                    z: 6.,
                },
                ModelPoint {
                    x: 2.,
                    y: 10.,
                    z: 6.,
                },
            ],
            ..Default::default()
        };
        let mut slicer = SlicerFuncs::new(
            SlicerView {
                xsize: 20,
                ysize: 20,
                zsize: 20,
                ..Default::default()
            },
            0,
        );
        let mut native = Fit::default();
        assert_eq!(slicer.angles_from_contour(&contour, &mut native), 0);
        assert!(native.angles.is_some());
        assert_eq!((slicer.cx, slicer.cy, slicer.cz), (4., 6., 6.));
        assert_eq!(native.bound, 1);
        assert_eq!(native.draw, Some(1 << 1));
    }
    #[test]
    fn resize_to_fit_returns_a_host_applicable_geometry_plan() {
        let mut slicer = SlicerFuncs::new(
            SlicerView {
                xsize: 100,
                ysize: 100,
                zsize: 50,
                xmouse: 50.,
                ymouse: 50.,
                zmouse: 25.,
                ..Default::default()
            },
            0,
        );
        slicer.resize(100, 100);
        slicer.rubberband = 1;
        slicer.rb_image_x0 = 25.;
        slicer.rb_image_x1 = 75.;
        slicer.rb_image_y0 = 25.;
        slicer.rb_image_y1 = 75.;
        slicer.rb_image_z0 = 25.;
        slicer.rb_image_z1 = 25.;
        let mut native = N;
        let plan = slicer
            .resize_to_fit((100, 100), (20, 30), 1., &mut native)
            .unwrap();
        // The upstream calculation delegates size clamping to
        // `diaLimitWindowSize`, so this pre-limit plan may be non-positive.
        assert_ne!(
            plan,
            SlicerResizePlan {
                width: 0,
                height: 0,
                x: 0,
                y: 0,
            }
        );
        assert_eq!(slicer.rubberband, 0);
    }
    #[test]
    fn printmat_keeps_source_row_order_and_return_status() {
        let (status, text) = printmat([
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.],
        ]);
        assert_eq!(status, 0);
        assert_eq!(text.lines().count(), 4);
        assert!(text.starts_with(" 1.00  2.00  3.00  4.00\n"));
        assert!(text.ends_with("13.00 14.00 15.00 16.00\n"));
    }
    #[test]
    fn external_draw_routes_classic_xyz_changes_through_the_slicer_draw() {
        #[derive(Default)]
        struct Draw {
            updates: usize,
            cubes: usize,
        }
        impl SlicerNativeBoundary for Draw {
            fn update_gl(&mut self) {
                self.updates += 1;
            }
            fn cube_draw(&mut self) {
                self.cubes += 1;
            }
        }
        let mut slicer = SlicerFuncs::new(
            SlicerView {
                xsize: 20,
                ysize: 20,
                zsize: 20,
                xmouse: 1.,
                ymouse: 2.,
                zmouse: 3.,
                ..Default::default()
            },
            0,
        );
        slicer.classic = 1;
        slicer.view.xmouse = 4.;
        slicer.view.ymouse = 5.;
        slicer.view.zmouse = 6.;
        let mut native = Draw::default();
        slicer_draw_cb(&mut slicer, 1 << 1, &mut native);
        assert_eq!((slicer.cx, slicer.cy, slicer.cz), (4., 5., 6.));
        assert_eq!((native.updates, native.cubes), (1, 1));
    }
    #[test]
    fn plug_mouse_dispatch_uses_slicer_coordinates_and_source_redraw_bit() {
        #[derive(Default)]
        struct Plug {
            point: Option<(f32, f32, i32, i32, i32)>,
            updates: usize,
            cubes: usize,
        }
        impl SlicerNativeBoundary for Plug {
            fn plugin_handle_mouse(
                &mut self,
                x: f32,
                y: f32,
                button1: i32,
                button2: i32,
                button3: i32,
            ) -> i32 {
                self.point = Some((x, y, button1, button2, button3));
                3
            }
            fn update_gl(&mut self) {
                self.updates += 1;
            }
            fn cube_draw(&mut self) {
                self.cubes += 1;
            }
        }
        let mut slicer = SlicerFuncs::new(
            SlicerView {
                xsize: 100,
                ysize: 100,
                zsize: 20,
                xmouse: 50.,
                ymouse: 50.,
                zmouse: 10.,
                ..Default::default()
            },
            0,
        );
        slicer.resize(100, 100);
        let mut native = Plug::default();
        assert_eq!(slicer.check_plug_use_mouse(50, 50, 1, 0, 1, &mut native), 3);
        let (x, y, b1, b2, b3) = native.point.unwrap();
        assert!(x.is_finite() && y.is_finite());
        assert_eq!((b1, b2, b3), (1, 0, 1));
        assert_eq!((native.updates, native.cubes), (1, 1));
    }
    #[test]
    fn current_point_overlay_keeps_source_plane_and_endpoint_rules() {
        struct Overlay(Vec<SlicerOverlayPrimitive>);
        impl SlicerNativeBoundary for Overlay {
            fn current_point_state(&mut self) -> SlicerCurrentPointState {
                SlicerCurrentPointState {
                    draw_cursor: true,
                    current_point: Some(Ipoint {
                        x: 50.,
                        y: 50.,
                        z: 10.,
                    }),
                    contour_points: vec![
                        Ipoint {
                            x: 50.,
                            y: 50.,
                            z: 10.,
                        },
                        Ipoint {
                            x: 55.,
                            y: 50.,
                            z: 10.,
                        },
                    ],
                    image_point_size: 4,
                    model_point_size: 6,
                    backup_point_size: 8,
                    model_line_width: 2,
                    ..Default::default()
                }
            }
            fn draw_current_point_overlay(&mut self, primitives: &[SlicerOverlayPrimitive]) {
                self.0.extend_from_slice(primitives);
            }
        }
        let mut slicer = SlicerFuncs::new(
            SlicerView {
                xsize: 100,
                ysize: 100,
                zsize: 20,
                xmouse: 50.,
                ymouse: 50.,
                zmouse: 10.,
                ..Default::default()
            },
            0,
        );
        slicer.resize(100, 100);
        let mut native = Overlay(Vec::new());
        slicer.draw_current_point(&mut native);
        assert!(matches!(
            native.0[0],
            SlicerOverlayPrimitive::Plus {
                color: SlicerOverlayColor::Endpoint,
                size: 5,
                ..
            }
        ));
        assert!(native.0.iter().any(|primitive| matches!(
            primitive,
            SlicerOverlayPrimitive::Circle {
                size: 8,
                color: SlicerOverlayColor::CurrentPoint,
                ..
            }
        )));
        assert!(native.0.iter().any(|primitive| matches!(
            primitive,
            SlicerOverlayPrimitive::Circle {
                color: SlicerOverlayColor::Beginning,
                ..
            }
        )));
        assert!(native.0.iter().any(|primitive| matches!(
            primitive,
            SlicerOverlayPrimitive::Circle {
                color: SlicerOverlayColor::Endpoint,
                ..
            }
        )));
    }
    #[test]
    fn model_render_plans_keep_source_two_pass_projection_depths() {
        struct Renderer(Vec<SlicerModelRenderPlan>);
        impl SlicerNativeBoundary for Renderer {
            fn model_render_config(&mut self) -> SlicerModelRenderConfig {
                SlicerModelRenderConfig {
                    model_z_scale: 2.,
                    extra_object_count: 1,
                    extra_cursor_in_window: true,
                }
            }
            fn draw_model_plan(&mut self, plan: SlicerModelRenderPlan) {
                self.0.push(plan);
            }
        }
        let mut slicer = SlicerFuncs::new(SlicerView::default(), 0);
        slicer.winx = 100;
        slicer.winy = 80;
        slicer.depth = 4.;
        slicer.zoom = 2.;
        let mut native = Renderer(Vec::new());
        slicer.draw_model(&mut native);
        assert_eq!(native.0.len(), 2);
        assert_eq!(native.0[0].projection_depth, 4.);
        assert_eq!(native.0[1].projection_depth, 10_000.);
        assert!(native.0[0].draw_main_model);
        assert!(!native.0[1].draw_main_model);
        assert_eq!(native.0[1].extra_object_mode, 2);
        assert_eq!(native.0[0].inverse_model_scale.z, 0.25);
    }
    #[test]
    fn rotate_vol_command_keeps_source_band_geometry_and_padding_order() {
        struct Padding;
        impl SlicerNativeBoundary for Padding {
            fn image_padding(
                &mut self,
                y_center: i32,
                z_center: i32,
                time: i32,
            ) -> Option<SlicerImagePadding> {
                assert_eq!((y_center, z_center, time), (50, 11, 1));
                Some(SlicerImagePadding::default())
            }
        }
        let mut slicer = SlicerFuncs::new(
            SlicerView {
                xsize: 100,
                ysize: 100,
                zsize: 20,
                xmouse: 50.,
                ymouse: 50.,
                zmouse: 10.,
                ..Default::default()
            },
            0,
        );
        slicer.resize(100, 100);
        slicer.rubberband = 1;
        slicer.rb_image_x0 = 25.;
        slicer.rb_image_x1 = 75.;
        slicer.rb_image_y0 = 25.;
        slicer.rb_image_y1 = 75.;
        slicer.rb_image_z0 = 10.;
        slicer.rb_image_z1 = 10.;
        slicer.limit_no_value = 100;
        slicer.band_low_high_limits = [0., 1.];
        let mut native = Padding;
        assert_eq!(
            slicer.rotate_vol_command(&mut native).unwrap(),
            "-siz 51,2,2 -cen 50.00,50.00,11.00 -ang 0.00,0.00,0.00"
        );
    }
    #[test]
    fn screen_change_applies_the_source_pending_size_adjustment() {
        struct Zoom(Vec<f32>);
        impl SlicerNativeBoundary for Zoom {
            fn set_zoom_text(&mut self, zoom: f32) {
                self.0.push(zoom);
            }
        }
        let mut slicer = SlicerFuncs::new(SlicerView::default(), 0);
        slicer.first_draw = 2;
        slicer.dev_pix_varies = true;
        slicer.zoom = 1.;
        slicer.last_xsize_change = 2.;
        slicer.screen_resize_time = std::time::Instant::now();
        let mut native = Zoom(Vec::new());
        slicer.screen_changed(2., &mut native);
        assert_eq!(slicer.zoom, 2.);
        assert_eq!(native.0, vec![2.]);
        assert!(!slicer.screen_changed);
        assert_eq!(slicer.last_xsize_change, 1.);
    }
}
