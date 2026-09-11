//! Translation units from `IMOD/3dmod`.
//!
//! The directory name intentionally follows the upstream layout.  The public
//! Rust module is named `three_dmod`, since a Rust identifier cannot begin
//! with a digit.

/// Translation of `IMOD/3dmod/form_appearance.cpp` and `form_appearance.h`.
pub mod form_appearance;
/// Translation of `IMOD/3dmod/form_autox.cpp` and `form_autox.h`.
pub mod form_autox;
/// Translation of `IMOD/3dmod/form_behavior.cpp` and `form_behavior.h`.
pub mod form_behavior;
/// Translation of `IMOD/3dmod/form_cont_edit.cpp` and `form_cont_edit.h`.
pub mod form_cont_edit;
/// Translation of `IMOD/3dmod/form_finegrain.cpp` and `form_finegrain.h`.
pub mod form_finegrain;
/// Translation of `IMOD/3dmod/form_info.cpp` and `form_info.h`.
pub mod form_info;
pub mod form_layout;
/// Translation of `IMOD/3dmod/form_mouse.cpp` and `form_mouse.h`.
pub mod form_mouse;
/// Translation of `IMOD/3dmod/form_moviecon.cpp` and `form_moviecon.h`.
pub mod form_moviecon;
/// Translation of `IMOD/3dmod/form_object_edit.cpp` and `form_object_edit.h`.
pub mod form_object_edit;
/// Translation of `IMOD/3dmod/form_prefscaling.cpp` and `form_prefscaling.h`.
pub mod form_prefscaling;
pub mod form_rawimage;
/// Translation of `IMOD/3dmod/form_scalebar.cpp` and `form_scalebar.h`.
pub mod form_scalebar;
/// Translation of `IMOD/3dmod/form_slicerangle.cpp` and `form_slicerangle.h`.
pub mod form_slicerangle;
/// Translation of `IMOD/3dmod/form_snapshot.cpp` and `form_snapshot.h`.
pub mod form_snapshot;
/// Translation of `IMOD/3dmod/form_startup.cpp` and `form_startup.h`.
pub mod form_startup;
/// Translation of `IMOD/3dmod/formv_control.cpp` and `formv_control.h`.
pub mod formv_control;
/// Translation of `IMOD/3dmod/formv_depthcue.cpp` and `formv_depthcue.h`.
pub mod formv_depthcue;
/// Translation of `IMOD/3dmod/formv_modeled.cpp` and `formv_modeled.h`.
pub mod formv_modeled;
/// Translation of `IMOD/3dmod/formv_movie.cpp` and `formv_movie.h`.
pub mod formv_movie;
/// Translation of `IMOD/3dmod/formv_objed.cpp` and `formv_objed.h`.
pub mod formv_objed;
/// Translation of `IMOD/3dmod/formv_sequence.cpp` and `formv_sequence.h`.
pub mod formv_sequence;
/// Translation of `IMOD/3dmod/formv_views.cpp` and `formv_views.h`.
pub mod formv_views;
pub mod imod;
pub mod imodv;
pub mod imodview;
/// Translation of `IMOD/3dmod/moviecon.cpp` and `moviecon.h`.
pub mod moviecon;
/// Translation of `IMOD/3dmod/mv_control.cpp` and `mv_control.h`.
pub mod mv_control;
/// Translation of `IMOD/3dmod/mv_gfx.cpp` and `mv_gfx.h`.
pub mod mv_gfx;
/// Translation of `IMOD/3dmod/mv_image.cpp` and `mv_image.h`.
pub mod mv_image;
/// Translation of `IMOD/3dmod/mv_input.cpp` and `mv_input.h`.
pub mod mv_input;
pub mod mv_light;
/// Translation of `IMOD/3dmod/mv_listobj.cpp` and `mv_listobj.h`.
pub mod mv_listobj;
/// Translation of `IMOD/3dmod/mv_modeled.cpp` and `mv_modeled.h`.
pub mod mv_modeled;
/// Translation of `IMOD/3dmod/mv_objed.cpp` and `mv_objed.h`.
pub mod mv_objed;
/// Translation of `IMOD/3dmod/mv_ogl.cpp` and `mv_ogl.h`.
pub mod mv_ogl;
/// Translation of `IMOD/3dmod/mv_stereo.cpp` and `mv_stereo.h`.
pub mod mv_stereo;
/// Translation of `IMOD/3dmod/mv_views.cpp` and `mv_views.h`.
pub mod mv_views;
/// Translation of `IMOD/3dmod/mv_window.cpp` and `mv_window.h`.
pub mod mv_window;
/// Translation of `IMOD/3dmod/object_edit.cpp` and `object_edit.h`.
pub mod object_edit;
/// Translation of `IMOD/3dmod/slicer.cpp` and `sslice.h`.
pub mod slicer;
/// Translation of `IMOD/3dmod/slicer_classes.cpp` and `slicer_classes.h`.
pub mod slicer_classes;
/// Translation of `IMOD/3dmod/vertexbuffer.cpp` and `vertexbuffer.h`.
pub mod vertexbuffer;
/// Translation of `IMOD/3dmod/zap_classes.cpp` and `zap_classes.h`.
pub mod zap_classes;
