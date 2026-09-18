//! Native ownership root for the normal `3dmod` image-display route.
//!
//! `mv_window.rs` owns the distinct model-view (`imodv`) event loop.  Normal
//! 3dmod has a different lifetime graph: an [`ImodView`] plus its image/model
//! state can own Zap, Slicer, XYZ and information windows.  This module is
//! deliberately that separate root, so those windows share one UI thread and
//! one compatibility OpenGL service graph rather than each boundary inventing
//! an unrelated toolkit object.

use crate::imod::libiimod::mrcfiles::LoadInfo;
use crate::imod::libimod::imodel::Imod;
use crate::imod::three_dmod::imodview::ImodView;

/// Owned normal-viewer state handed from `ImodNativeHost` to the platform
/// event-loop implementation.  The source keeps these on `imod.cpp`'s stack
/// for the duration of `QApplication::exec()`.
pub struct ImodImageHost {
    pub view: Box<ImodView>,
    pub model: Box<Imod>,
    pub load_info: Box<LoadInfo>,
    pub title: String,
}

impl ImodImageHost {
    pub fn new(
        view: Box<ImodView>,
        model: Box<Imod>,
        load_info: Box<LoadInfo>,
        title: impl Into<String>,
    ) -> Self {
        Self {
            view,
            model,
            load_info,
            title: title.into(),
        }
    }
}

/// Start the normal image viewer's native event loop.
///
/// This is intentionally feature-gated just like `run_native_opengl`; the
/// no-GL build remains a useful command/library build and must report that it
/// cannot create a platform image window.
#[cfg(not(feature = "three-dmod-gl"))]
pub fn run_native_image_host(_host: ImodImageHost) -> Result<(), String> {
    Err("3dmod image-display host requires the three-dmod-gl feature".to_owned())
}

/// The first concrete normal-host lifecycle: create the native window, retain
/// the source view/model for its lifetime, redraw on expose, and leave only on
/// a close request.  Rendering and child-window dispatch are added here rather
/// than through the model-view `ImodvWindow` path.
#[cfg(feature = "three-dmod-gl")]
pub fn run_native_image_host(mut host: ImodImageHost) -> Result<(), String> {
    use winit::dpi::PhysicalSize;
    use winit::event::{Event, WindowEvent};
    use winit::event_loop::EventLoop;
    use winit::window::Window;

    let event_loop = EventLoop::new().map_err(|error| error.to_string())?;
    let width = host.view.xsize.max(1) as u32;
    let height = host.view.ysize.max(1) as u32;
    let window = event_loop
        .create_window(
            Window::default_attributes()
                .with_title(&host.title)
                .with_inner_size(PhysicalSize::new(width, height)),
        )
        .map_err(|error| error.to_string())?;
    let mut window_size = PhysicalSize::new(width, height);
    event_loop
        .run(move |event, target| match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => target.exit(),
            Event::WindowEvent {
                event: WindowEvent::Resized(size),
                ..
            } => {
                window_size = size;
                window.request_redraw();
            }
            Event::AboutToWait => window.request_redraw(),
            _ => {}
        })
        .map_err(|error| error.to_string())
}
