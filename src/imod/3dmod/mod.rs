//! Translation units from `IMOD/3dmod`.
//!
//! The directory name intentionally follows the upstream layout.  The public
//! Rust module is named `three_dmod`, since a Rust identifier cannot begin
//! with a digit.

/// Translation of `IMOD/3dmod/autox.cpp` and `autox.h`.
pub mod autox;
/// Translation of `IMOD/3dmod/b3dgfx.cpp` and `b3dgfx.h`.
pub mod b3dgfx;
/// Translation of `IMOD/3dmod/beadfix.cpp` and `beadfix.h`.
pub mod beadfix;
/// Translation of `IMOD/3dmod/cachefill.cpp` and `cachefill.h`.
pub mod cachefill;
/// Translation of `IMOD/3dmod/client_message.cpp` and `client_message.h`.
pub mod client_message;
/// Translation of `IMOD/3dmod/cont_edit.cpp` and `cont_edit.h`.
pub mod cont_edit;
/// Translation of `IMOD/3dmod/control.cpp`, `control.h`, and `controlP.h`.
pub mod control;
/// Translation of `IMOD/3dmod/display.cpp` and `display.h`.
pub mod display;
/// Translation of `IMOD/3dmod/dockingdialog.cpp` and `dockingdialog.h`.
pub mod dockingdialog;
/// Translation of `IMOD/3dmod/finegrain.cpp` and `finegrain.h`.
pub mod finegrain;
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
/// Translation of `IMOD/3dmod/histwidget.cpp` and `histwidget.h`.
pub mod histwidget;
/// Translation of `IMOD/3dmod/iiqimage.cpp`.
pub mod iiqimage;
/// Translation of `IMOD/3dmod/iirawimage.cpp` and `iirawimage.h`.
pub mod iirawimage;
pub mod imod;
/// Translation of `IMOD/3dmod/imod_assistant.cpp` and `imod_assistant.h`.
pub mod imod_assistant;
/// Translation of `IMOD/3dmod/imod_edit.cpp` and `imod_edit.h`.
pub mod imod_edit;
/// Translation of `IMOD/3dmod/imod_input.cpp` and `imod_input.h`.
pub mod imod_input;
/// Translation of `IMOD/3dmod/imod_io.cpp` and `imod_io.h`.
pub mod imod_io;
/// Native winit/glutin ownership for the normal image-display route.
pub mod imod_window;
/// Translation of `IMOD/3dmod/imodplug.cpp`, `imodplug.h`, and `imodplugP.h`.
pub mod imodplug;
pub mod imodv;
pub mod imodview;
/// Translation of `IMOD/3dmod/info_cb.cpp` and `info_cb.h`.
pub mod info_cb;
/// Translation of `IMOD/3dmod/info_menu.cpp` and the paired `InfoWindow`
/// declarations in `info_setup.h`.
pub mod info_menu;
/// Translation of `IMOD/3dmod/info_setup.cpp` and `info_setup.h`.
pub mod info_setup;
/// Translation of `IMOD/3dmod/iproc.cpp` and `iproc.h`.
pub mod iproc;
/// Translation of `IMOD/3dmod/isosurface.cpp` and `isosurface.h`.
pub mod isosurface;
/// Translation of `IMOD/3dmod/isothread.cpp` and `isothread.h`.
pub mod isothread;
/// Translation of `IMOD/3dmod/linegui.cpp` and `linegui.h`.
pub mod linegui;
/// Translation of `IMOD/3dmod/locator.cpp` and `locator.h`.
pub mod locator;
/// Translation of `IMOD/3dmod/mappingtable.cpp` and `mappingtable.h`.
pub mod mappingtable;
/// Translation of `IMOD/3dmod/model_draw.cpp`.
pub mod model_draw;
/// Translation of `IMOD/3dmod/model_edit.cpp` and `model_edit.h`.
pub mod model_edit;
/// Translation of `IMOD/3dmod/moviecon.cpp` and `moviecon.h`.
pub mod moviecon;
/// Translation of `IMOD/3dmod/mv_control.cpp` and `mv_control.h`.
pub mod mv_control;
/// Translation of `IMOD/3dmod/mv_depthcue.cpp` and `mv_depthcue.h`.
pub mod mv_depthcue;
/// Translation of `IMOD/3dmod/mv_gfx.cpp` and `mv_gfx.h`.
pub mod mv_gfx;
/// Translation of `IMOD/3dmod/mv_image.cpp` and `mv_image.h`.
pub mod mv_image;
/// Translation of `IMOD/3dmod/mv_input.cpp` and `mv_input.h`.
pub mod mv_input;
pub mod mv_light;
/// Translation of `IMOD/3dmod/mv_listobj.cpp` and `mv_listobj.h`.
pub mod mv_listobj;
/// Translation of `IMOD/3dmod/mv_menu.cpp` and `mv_menu.h`.
pub mod mv_menu;
/// Translation of `IMOD/3dmod/mv_modeled.cpp` and `mv_modeled.h`.
pub mod mv_modeled;
/// Translation of `IMOD/3dmod/mv_movie.cpp` and `mv_movie.h`.
pub mod mv_movie;
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
/// Translation of `IMOD/3dmod/pixelview.cpp` and `pixelview.h`.
pub mod pixelview;
/// Translation of `IMOD/3dmod/preferences.cpp` and `preferences.h`.
pub mod preferences;
/// Translation of `IMOD/3dmod/pyramidcache.cpp` and `pyramidcache.h`.
pub mod pyramidcache;
/// Translation of `IMOD/3dmod/rescale.cpp` and `rescale.h`.
pub mod rescale;
pub mod resizetool;
pub mod rotationtool;
/// Translation of `IMOD/3dmod/scalebar.cpp` and `scalebar.h`.
pub mod scalebar;
/// Translation of `IMOD/3dmod/slicer.cpp` and `sslice.h`.
pub mod slicer;
/// Translation of `IMOD/3dmod/slicer_classes.cpp` and `slicer_classes.h`.
pub mod slicer_classes;
/// Translation of `IMOD/3dmod/slicerthreads.cpp`.
pub mod slicerthreads;
/// Translation of `IMOD/3dmod/surfpieces.cpp` and `surfpieces.h`.
pub mod surfpieces;
/// Translation of `IMOD/3dmod/undoredo.cpp`, `undoredo.h`, and `undoredoP.h`.
pub mod undoredo;
/// Translation of `IMOD/3dmod/utilities.cpp` and `utilities.h`.
pub mod utilities;
/// Translation of `IMOD/3dmod/vertexbuffer.cpp` and `vertexbuffer.h`.
pub mod vertexbuffer;
/// Translation of `IMOD/3dmod/workprocs.cpp` and `workprocs.h`.
pub mod workprocs;
/// Translation of `IMOD/3dmod/xcorr.cpp` and `xcorr.h`.
pub mod xcorr;
/// Translation of `IMOD/3dmod/xcramp.cpp` and `xcramp.h`.
pub mod xcramp;
/// Translation of `IMOD/3dmod/xgraph.cpp` and `xgraph.h`.
pub mod xgraph;
/// Translation of `IMOD/3dmod/xyz.cpp` and `xxyz.h`.
pub mod xyz;
/// Translation of `IMOD/3dmod/xzap.cpp` and `xzap.h`.
pub mod xzap;
/// Translation of `IMOD/3dmod/zap_classes.cpp` and `zap_classes.h`.
pub mod zap_classes;
