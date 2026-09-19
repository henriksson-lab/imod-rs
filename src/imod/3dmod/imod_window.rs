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
    /// Source startup switches (`-Z`, `-S`, and `-xyz`) are consumed by the
    /// same native owner that creates their windows, not by the short-lived
    /// command-line loader.
    pub initial_tools: Vec<InitialToolWindow>,
}

/// A normal 3dmod child window requested during startup.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InitialToolWindow {
    Zap,
    Slicer,
    Xyz,
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
            initial_tools: Vec::new(),
        }
    }

    pub fn with_initial_tools(mut self, initial_tools: Vec<InitialToolWindow>) -> Self {
        self.initial_tools = initial_tools;
        self
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

/// Normal 3dmod's OpenGL resources.  This deliberately uses a small core-GL
/// textured-quad pipeline: unlike `imodv`, the normal viewer is a 2-D section
/// viewer and must not inherit model-view camera state just to show its image.
#[cfg(feature = "three-dmod-gl")]
#[derive(Clone, Copy, Debug)]
pub enum NormalHostCommand {
    DrawModel,
}

#[cfg(feature = "three-dmod-gl")]
#[derive(Clone)]
pub struct NormalHostProxy(winit::event_loop::EventLoopProxy<NormalHostCommand>);

#[cfg(feature = "three-dmod-gl")]
impl NormalHostProxy {
    pub fn request_model_draw(&self) -> Result<(), String> {
        self.0
            .send_event(NormalHostCommand::DrawModel)
            .map_err(|error| error.to_string())
    }
}

#[cfg(feature = "three-dmod-gl")]
struct SectionRenderer {
    gl: glow::Context,
    program: glow::Program,
    vao: glow::VertexArray,
    texture: glow::Texture,
    image_size: (i32, i32),
    zoom: f32,
    pan: (f32, f32),
    model_gl: NormalModelGl,
    model_state: crate::imod::three_dmod::model_draw::ModelDrawState,
}

/// The compatibility-profile entry points used by the normal image model
/// overlay.  The image itself deliberately uses a portable textured-quad
/// shader, but `model_draw.cpp` is source-faithfully expressed in immediate
/// primitives and matrix-stack operations.  Keep this narrower than the
/// imodv bridge: normal 3dmod needs image-plane model drawing, not imodv's
/// picking, display-list, and lighting service graph.
#[cfg(feature = "three-dmod-gl")]
#[derive(Clone, Copy)]
struct NormalModelGl {
    matrix_mode: unsafe extern "C" fn(u32),
    load_identity: unsafe extern "C" fn(),
    ortho: unsafe extern "C" fn(f64, f64, f64, f64, f64, f64),
    begin: unsafe extern "C" fn(u32),
    end: unsafe extern "C" fn(),
    vertex3f: unsafe extern "C" fn(f32, f32, f32),
    color3f: unsafe extern "C" fn(f32, f32, f32),
    color4f: unsafe extern "C" fn(f32, f32, f32, f32),
    point_size: unsafe extern "C" fn(f32),
    push_matrix: unsafe extern "C" fn(),
    pop_matrix: unsafe extern "C" fn(),
    translatef: unsafe extern "C" fn(f32, f32, f32),
    scalef: unsafe extern "C" fn(f32, f32, f32),
}

#[cfg(feature = "three-dmod-gl")]
impl NormalModelGl {
    fn load(loader: &dyn Fn(&std::ffi::CStr) -> *const std::ffi::c_void) -> Result<Self, String> {
        macro_rules! entry {
            ($name:literal, $signature:ty) => {{
                let name = std::ffi::CString::new($name).expect("literal GL symbol");
                let address = loader(&name);
                if address.is_null() {
                    return Err(format!(
                        "3dmod normal model overlay requires compatibility OpenGL entry point {}",
                        $name
                    ));
                }
                unsafe { std::mem::transmute::<*const std::ffi::c_void, $signature>(address) }
            }};
        }
        Ok(Self {
            matrix_mode: entry!("glMatrixMode", unsafe extern "C" fn(u32)),
            load_identity: entry!("glLoadIdentity", unsafe extern "C" fn()),
            ortho: entry!(
                "glOrtho",
                unsafe extern "C" fn(f64, f64, f64, f64, f64, f64)
            ),
            begin: entry!("glBegin", unsafe extern "C" fn(u32)),
            end: entry!("glEnd", unsafe extern "C" fn()),
            vertex3f: entry!("glVertex3f", unsafe extern "C" fn(f32, f32, f32)),
            color3f: entry!("glColor3f", unsafe extern "C" fn(f32, f32, f32)),
            color4f: entry!("glColor4f", unsafe extern "C" fn(f32, f32, f32, f32)),
            point_size: entry!("glPointSize", unsafe extern "C" fn(f32)),
            push_matrix: entry!("glPushMatrix", unsafe extern "C" fn()),
            pop_matrix: entry!("glPopMatrix", unsafe extern "C" fn()),
            translatef: entry!("glTranslatef", unsafe extern "C" fn(f32, f32, f32)),
            scalef: entry!("glScalef", unsafe extern "C" fn(f32, f32, f32)),
        })
    }
}

/// One independently drawable normal-viewer tool surface.  Its controller
/// state stays in the host while this owns only the native GL resources.
#[cfg(feature = "three-dmod-gl")]
struct ToolSurface {
    kind: ToolKind,
    controller_index: usize,
    window: winit::window::Window,
    surface: glutin::surface::Surface<glutin::surface::WindowSurface>,
    context: glutin::context::PossiblyCurrentContext,
    renderer: SectionRenderer,
    size: winit::dpi::PhysicalSize<u32>,
    cursor: winit::dpi::PhysicalPosition<f64>,
    dragging: bool,
}

/// The translated owner paired with one native child surface.
#[cfg(feature = "three-dmod-gl")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ToolKind {
    Zap,
    Slicer,
    Xyz,
    Information,
}

/// `SlicerFuncs` keeps the portion of `ImodView` that it consumes as a Rust
/// value, whereas the C++ controller held an `ImodView *`.  Keep that adapter
/// current whenever the native host changes the shared input location.  Zap
/// already holds the view pointer and XYZ receives a fresh `XyzViewState` at
/// dispatch, so neither needs a duplicate update here.
#[cfg(feature = "three-dmod-gl")]
fn synchronize_slicer_input_views(
    slicers: &mut crate::imod::three_dmod::slicer::SlicerRegistry,
    view: &ImodView,
) {
    for slicer in &mut slicers.slicers {
        slicer.view.xmouse = view.xmouse;
        slicer.view.ymouse = view.ymouse;
        slicer.view.zmouse = view.zmouse;
        slicer.view.cur_time = view.cur_time;
        slicer.view.num_times = view.num_times;
        slicer.view.track_mouse_for_plugs = view.track_mouse_for_plugs;
        slicer.view.xmovie = view.xmovie;
        slicer.view.ymovie = view.ymovie;
        slicer.view.zmovie = view.zmovie;
    }
}

#[cfg(feature = "three-dmod-gl")]
fn window_to_image_coordinates(
    image_size: (i32, i32),
    zoom: f32,
    pan: (f32, f32),
    position: winit::dpi::PhysicalPosition<f64>,
    size: winit::dpi::PhysicalSize<u32>,
) -> Option<(f32, f32)> {
    let (image_width, image_height) = image_size;
    if image_width <= 0 || image_height <= 0 || size.width == 0 || size.height == 0 {
        return None;
    }
    let (mut width, mut height) = (size.width as f64, size.height as f64);
    if width * image_height as f64 > height * image_width as f64 {
        width = height * image_width as f64 / image_height as f64;
    } else {
        height = width * image_height as f64 / image_width as f64;
    }
    let left = (size.width as f64 - width) / 2.;
    let top = (size.height as f64 - height) / 2.;
    let clip_x = ((position.x - left) * 2. / width - 1. - pan.0 as f64) / zoom as f64;
    let clip_y = (1. - (position.y - top) * 2. / height - pan.1 as f64) / zoom as f64;
    Some((
        ((clip_x + 1.) * image_width as f64 / 2.).clamp(0., f64::from(image_width - 1)) as f32,
        ((1. - clip_y) * image_height as f64 / 2.).clamp(0., f64::from(image_height - 1)) as f32,
    ))
}

/// Convert the subset of winit logical keys represented by
/// `inputQDefaultKeys` to its source Qt key values.  Tool-opening hotkeys
/// remain host commands; model/navigation keys travel through the shared
/// translated input core.
#[cfg(feature = "three-dmod-gl")]
fn normal_input_key(key: &winit::keyboard::Key) -> Option<i32> {
    use crate::imod::three_dmod::imod_input::{
        KEY_DELETE, KEY_DOWN, KEY_INSERT, KEY_LEFT, KEY_PAGE_DOWN, KEY_PAGE_UP, KEY_RIGHT, KEY_UP,
    };
    use winit::keyboard::{Key, NamedKey};
    match key {
        Key::Named(NamedKey::Insert) => Some(KEY_INSERT),
        Key::Named(NamedKey::Delete) => Some(KEY_DELETE),
        Key::Named(NamedKey::ArrowLeft) => Some(KEY_LEFT),
        Key::Named(NamedKey::ArrowRight) => Some(KEY_RIGHT),
        Key::Named(NamedKey::ArrowUp) => Some(KEY_UP),
        Key::Named(NamedKey::ArrowDown) => Some(KEY_DOWN),
        Key::Named(NamedKey::PageUp) => Some(KEY_PAGE_UP),
        Key::Named(NamedKey::PageDown) => Some(KEY_PAGE_DOWN),
        Key::Character(character) => character
            .chars()
            .next()
            .map(|character| character.to_ascii_uppercase() as i32),
        _ => None,
    }
}

/// Convert winit modifier state to the translated Qt input bits.
#[cfg(feature = "three-dmod-gl")]
fn normal_input_modifiers(state: winit::keyboard::ModifiersState) -> u32 {
    use crate::imod::three_dmod::imod_input::{INPUT_CTRL, INPUT_SHIFT};
    u32::from(state.shift_key()) * INPUT_SHIFT + u32::from(state.control_key()) * INPUT_CTRL
}

/// The host-owned input boundary.  The source input unit mutates the shared
/// `ImodView`; drawing is batched by the event loop immediately afterwards,
/// so no native window callback needs to borrow the view while it is mutated.
#[cfg(feature = "three-dmod-gl")]
#[derive(Default)]
struct NormalInputBoundary {
    redraw: bool,
}

#[cfg(feature = "three-dmod-gl")]
impl crate::imod::three_dmod::imod_input::InputNativeBoundary for NormalInputBoundary {
    fn draw(&mut self, _view: &mut ImodView, _flags: i32) {
        self.redraw = true;
    }

    fn set_xyz_mouse(&mut self, _view: &mut ImodView) {
        self.redraw = true;
    }
}

#[cfg(feature = "three-dmod-gl")]
impl SectionRenderer {
    fn new(
        gl: glow::Context,
        loader: &dyn Fn(&std::ffi::CStr) -> *const std::ffi::c_void,
    ) -> Result<Self, String> {
        use glow::HasContext;
        const VERTEX: &str = "#version 110\nattribute vec2 position; attribute vec2 texcoord; uniform vec2 scale; uniform vec2 pan; varying vec2 uv; void main() { uv = texcoord; gl_Position = vec4(position * scale + pan, 0.0, 1.0); }";
        const FRAGMENT: &str = "#version 110\nvarying vec2 uv; uniform sampler2D image; void main() { gl_FragColor = texture2D(image, uv); }";
        unsafe {
            let vertex = gl
                .create_shader(glow::VERTEX_SHADER)
                .map_err(|e| e.to_string())?;
            gl.shader_source(vertex, VERTEX);
            gl.compile_shader(vertex);
            if !gl.get_shader_compile_status(vertex) {
                return Err(gl.get_shader_info_log(vertex));
            }
            let fragment = gl
                .create_shader(glow::FRAGMENT_SHADER)
                .map_err(|e| e.to_string())?;
            gl.shader_source(fragment, FRAGMENT);
            gl.compile_shader(fragment);
            if !gl.get_shader_compile_status(fragment) {
                return Err(gl.get_shader_info_log(fragment));
            }
            let program = gl.create_program().map_err(|e| e.to_string())?;
            gl.attach_shader(program, vertex);
            gl.attach_shader(program, fragment);
            gl.link_program(program);
            gl.delete_shader(vertex);
            gl.delete_shader(fragment);
            if !gl.get_program_link_status(program) {
                return Err(gl.get_program_info_log(program));
            }
            let vao = gl.create_vertex_array().map_err(|e| e.to_string())?;
            let buffer = gl.create_buffer().map_err(|e| e.to_string())?;
            // x/y position followed by texture position; flip Y because IMOD
            // section rows are top-to-bottom while GL texture coordinates are not.
            let vertices: [f32; 16] = [
                -1., -1., 0., 1., 1., -1., 1., 1., -1., 1., 0., 0., 1., 1., 1., 0.,
            ];
            gl.bind_vertex_array(Some(vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(buffer));
            let vertex_bytes = core::slice::from_raw_parts(
                vertices.as_ptr().cast::<u8>(),
                core::mem::size_of_val(&vertices),
            );
            gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, vertex_bytes, glow::STATIC_DRAW);
            for (name, offset) in [("position", 0), ("texcoord", 2)] {
                let location = gl
                    .get_attrib_location(program, name)
                    .ok_or_else(|| format!("normal image shader has no {name} attribute"))?;
                gl.enable_vertex_attrib_array(location);
                gl.vertex_attrib_pointer_f32(location, 2, glow::FLOAT, false, 16, offset * 4);
            }
            let texture = gl.create_texture().map_err(|e| e.to_string())?;
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::NEAREST as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::NEAREST as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_S,
                glow::CLAMP_TO_EDGE as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_T,
                glow::CLAMP_TO_EDGE as i32,
            );
            gl.use_program(Some(program));
            gl.uniform_1_i32(gl.get_uniform_location(program, "image").as_ref(), 0);
            Ok(Self {
                gl,
                program,
                vao,
                texture,
                image_size: (0, 0),
                zoom: 1.,
                pan: (0., 0.),
                model_gl: NormalModelGl::load(loader)?,
                model_state: crate::imod::three_dmod::model_draw::ModelDrawState::default(),
            })
        }
    }

    fn upload_section(&mut self, view: &mut ImodView) {
        use crate::imod::three_dmod::imodview::{ivw_get_location, ivw_get_z_section};
        use glow::{HasContext, PixelUnpackData};
        let width = view.xsize.max(1);
        let height = view.ysize.max(1);
        let mut x = 0;
        let mut y = 0;
        let mut section = 0;
        ivw_get_location(view, &mut x, &mut y, &mut section);
        let section = section.clamp(0, view.zsize.saturating_sub(1));
        let rows = ivw_get_z_section(view, section);
        let mut rgb = vec![0u8; width as usize * height as usize * 3];
        if !rows.is_null() {
            for row in 0..height as usize {
                let line = unsafe { *rows.add(row) };
                if line.is_null() {
                    continue;
                }
                for col in 0..width as usize {
                    let out = (row * width as usize + col) * 3;
                    unsafe {
                        if view.rgb_store != 0 {
                            let source = line.add(col * 3);
                            rgb[out..out + 3]
                                .copy_from_slice(core::slice::from_raw_parts(source, 3));
                        } else if view.ushort_store != 0 {
                            let source = line.add(col * 2);
                            let value = u16::from_ne_bytes([*source, *source.add(1)]);
                            let value = (value >> 8) as u8;
                            rgb[out..out + 3].fill(value);
                        } else {
                            rgb[out..out + 3].fill(*line.add(col));
                        }
                    }
                }
            }
        }
        unsafe {
            self.gl.bind_texture(glow::TEXTURE_2D, Some(self.texture));
            self.gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGB as i32,
                width,
                height,
                0,
                glow::RGB,
                glow::UNSIGNED_BYTE,
                PixelUnpackData::Slice(Some(&rgb)),
            );
        }
        self.image_size = (width, height);
    }

    fn draw(&self, size: winit::dpi::PhysicalSize<u32>) {
        use glow::HasContext;
        let (image_width, image_height) = self.image_size;
        let (mut width, mut height) = (size.width as i32, size.height as i32);
        if image_width > 0 && image_height > 0 && width > 0 && height > 0 {
            if width as i64 * image_height as i64 > height as i64 * image_width as i64 {
                width = (height as i64 * image_width as i64 / image_height as i64) as i32;
            } else {
                height = (width as i64 * image_height as i64 / image_width as i64) as i32;
            }
        }
        unsafe {
            self.gl.clear_color(0.08, 0.08, 0.08, 1.0);
            self.gl.clear(glow::COLOR_BUFFER_BIT);
            self.gl.viewport(
                ((size.width as i32 - width) / 2).max(0),
                ((size.height as i32 - height) / 2).max(0),
                width.max(1),
                height.max(1),
            );
            self.gl.use_program(Some(self.program));
            self.gl.uniform_2_f32(
                self.gl.get_uniform_location(self.program, "scale").as_ref(),
                self.zoom,
                self.zoom,
            );
            self.gl.uniform_2_f32(
                self.gl.get_uniform_location(self.program, "pan").as_ref(),
                self.pan.0,
                self.pan.1,
            );
            self.gl.bind_vertex_array(Some(self.vao));
            self.gl.bind_texture(glow::TEXTURE_2D, Some(self.texture));
            self.gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
        }
    }

    /// Draw the translated normal-viewer model on top of the current image
    /// section.  This is intentionally separate from the textured-quad pass:
    /// model drawing is compatibility-profile source code, whereas image data
    /// remains in the portable shader path.
    fn draw_model(&mut self, view: &ImodView, model: &crate::imod::libimod::imodel::Imod) {
        use glow::HasContext;
        const GL_PROJECTION: u32 = 0x1701;
        const GL_MODELVIEW: u32 = 0x1700;
        let (width, height) = self.image_size;
        if width <= 0 || height <= 0 || model.drawmode <= 0 {
            return;
        }
        unsafe {
            self.gl.use_program(None);
            (self.model_gl.matrix_mode)(GL_PROJECTION);
            (self.model_gl.load_identity)();
            // Image rows increase downwards.  This mirrors the texture pass's
            // Y flip and keeps source model coordinates in image space.
            (self.model_gl.ortho)(0., width as f64, height as f64, 0., -1., 1.);
            (self.model_gl.matrix_mode)(GL_MODELVIEW);
            (self.model_gl.load_identity)();
        }
        let mut state = std::mem::take(&mut self.model_state);
        crate::imod::three_dmod::model_draw::imod_draw_model(
            view,
            model,
            true,
            model.zscale,
            &mut state,
            self,
        );
        self.model_state = state;
    }

    fn wheel_zoom(&mut self, amount: f32) {
        self.zoom = (self.zoom * if amount > 0. { 1.15 } else { 1. / 1.15 }).clamp(0.1, 32.);
    }

    fn pan_by_pixels(&mut self, dx: f64, dy: f64, size: winit::dpi::PhysicalSize<u32>) {
        if size.width > 0 {
            self.pan.0 += (2. * dx / f64::from(size.width)) as f32;
        }
        if size.height > 0 {
            self.pan.1 -= (2. * dy / f64::from(size.height)) as f32;
        }
    }

    /// Translate native-window pointer coordinates into the shared source
    /// `ImodView` image coordinates.  This uses the same letterboxed viewport,
    /// zoom and pan transform as `draw`, so Zap/Slicer/XYZ controllers observe
    /// a single coherent mouse location even when the event arrived on a tool
    /// surface.
    fn update_view_mouse(
        &self,
        view: &mut ImodView,
        position: winit::dpi::PhysicalPosition<f64>,
        size: winit::dpi::PhysicalSize<u32>,
    ) {
        let Some((x, y)) =
            window_to_image_coordinates(self.image_size, self.zoom, self.pan, position, size)
        else {
            return;
        };
        view.xmouse = x;
        view.ymouse = y;
    }
}

#[cfg(feature = "three-dmod-gl")]
impl crate::imod::three_dmod::finegrain::FinegrainRenderBoundary for SectionRenderer {
    fn color3f(&mut self, red: f32, green: f32, blue: f32) {
        unsafe { (self.model_gl.color3f)(red, green, blue) }
    }

    fn color4f(&mut self, red: f32, green: f32, blue: f32, alpha: f32) {
        unsafe { (self.model_gl.color4f)(red, green, blue, alpha) }
    }

    fn line_width(&mut self, width: i32, object: &crate::imod::libimod::imodel::Iobj) {
        use glow::HasContext;
        unsafe {
            self.gl.line_width(width.max(1) as f32);
            (self.model_gl.color4f)(
                object.red,
                object.green,
                object.blue,
                1. - object.trans as f32 / 255.,
            );
        }
    }

    fn point_size(&mut self, size: i32, _object: &crate::imod::libimod::imodel::Iobj) {
        unsafe { (self.model_gl.point_size)(size.max(1) as f32) }
    }

    fn light_adjust(
        &mut self,
        _object: &crate::imod::libimod::imodel::Iobj,
        red: f32,
        green: f32,
        blue: f32,
        trans: i32,
    ) {
        self.color4f(red, green, blue, 1. - trans.clamp(0, 255) as f32 / 255.);
    }

    fn rgba(&self) -> bool {
        true
    }
}

#[cfg(feature = "three-dmod-gl")]
impl crate::imod::three_dmod::model_draw::ModelDrawBoundary for SectionRenderer {
    fn set_object_color(&mut self, _object: i32) {}

    fn color_index(&mut self, _color: i32) {}

    fn begin(&mut self, mode: u32) {
        unsafe { (self.model_gl.begin)(mode) }
    }

    fn end(&mut self) {
        unsafe { (self.model_gl.end)() }
    }

    fn vertex3(&mut self, point: crate::imod::libimod::imodel::Ipoint) {
        unsafe { (self.model_gl.vertex3f)(point.x, point.y, point.z) }
    }

    fn push_matrix(&mut self) {
        unsafe { (self.model_gl.push_matrix)() }
    }

    fn pop_matrix(&mut self) {
        unsafe { (self.model_gl.pop_matrix)() }
    }

    fn translate(&mut self, point: crate::imod::libimod::imodel::Ipoint) {
        unsafe { (self.model_gl.translatef)(point.x, point.y, point.z) }
    }

    fn scale(&mut self, x: f32, y: f32, z: f32) {
        unsafe { (self.model_gl.scalef)(x, y, z) }
    }

    fn sphere(&mut self, radius: f64, slices: i32, _loops: i32) {
        // The normal image plane renders point symbols as circles.  This is
        // the 2-D source equivalent of the model-view GLU sphere and avoids
        // making normal 3dmod depend on imodv's separate GLU ownership.
        let steps = slices.clamp(8, 64);
        self.begin(crate::imod::three_dmod::model_draw::GL_LINE_LOOP);
        for index in 0..steps {
            let angle = std::f32::consts::TAU * index as f32 / steps as f32;
            self.vertex3(crate::imod::libimod::imodel::Ipoint {
                x: radius as f32 * angle.cos(),
                y: radius as f32 * angle.sin(),
                z: 0.,
            });
        }
        self.end();
    }

    fn disk(&mut self, _inner: f64, outer: f64, slices: i32, loops: i32) {
        self.sphere(outer, slices, loops);
    }

    fn set_stipple(&mut self, _enabled: bool) {}

    fn time_mismatch(
        &self,
        _view: &ImodView,
        _object: &crate::imod::libimod::imodel::Iobj,
        _contour: &crate::imod::libimod::imodel::Icont,
    ) -> bool {
        false
    }

    fn draw_labels(
        &mut self,
        _model: &crate::imod::libimod::imodel::Imod,
        _object: &crate::imod::libimod::imodel::Iobj,
        _win_y: i32,
        _ratio: f32,
        _height: i32,
    ) {
    }

    fn cleanup_label_font(&mut self) {}

    fn manage_paired_meshes(
        &mut self,
        _object: &crate::imod::libimod::imodel::Iobj,
        _object_number: i32,
    ) -> bool {
        false
    }
}

/// Create the normal-viewer's independent compatibility context and render
/// the current image section.  Up/down select sections; Z/S/X/I open the
/// corresponding normal-viewer tool windows and retain their translated
/// controller state on this same UI thread.
#[cfg(feature = "three-dmod-gl")]
pub fn run_native_image_host(
    mut host: ImodImageHost,
    on_ready: impl FnOnce(NormalHostProxy),
) -> Result<(), String> {
    use crate::imod::three_dmod::control::ZAP_WINDOW_TYPE;
    use crate::imod::three_dmod::info_cb::InfoCbState;
    use crate::imod::three_dmod::slicer::{SlicerRegistry, SlicerView, slicer_open};
    use crate::imod::three_dmod::xyz::{XyzViewState, XyzWindow};
    use crate::imod::three_dmod::xzap::ZapFuncs;
    use glutin::config::ConfigTemplateBuilder;
    use glutin::context::{ContextApi, ContextAttributesBuilder, GlProfile, Version};
    use glutin::display::{GetGlDisplay, GlDisplay};
    use glutin::prelude::*;
    use glutin::surface::{GlSurface, SurfaceAttributesBuilder, SwapInterval, WindowSurface};
    use glutin_winit::DisplayBuilder;
    use raw_window_handle::HasWindowHandle;
    use std::num::NonZeroU32;
    use winit::dpi::PhysicalSize;
    use winit::event::{ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent};
    use winit::event_loop::EventLoop;
    use winit::keyboard::{Key, NamedKey};
    use winit::window::Window;

    let event_loop = EventLoop::<NormalHostCommand>::with_user_event()
        .build()
        .map_err(|error| error.to_string())?;
    on_ready(NormalHostProxy(event_loop.create_proxy()));
    let width = host.view.xsize.max(1) as u32;
    let height = host.view.ysize.max(1) as u32;
    let attrs = Window::default_attributes()
        .with_title(&host.title)
        .with_inner_size(PhysicalSize::new(width, height));
    let display_builder = DisplayBuilder::new().with_window_attributes(Some(attrs));
    let template = ConfigTemplateBuilder::new().with_alpha_size(8);
    let (window, config) = display_builder
        .build(&event_loop, template, |configs| {
            configs
                .max_by_key(|config| config.num_samples())
                .expect("no GL config")
        })
        .map_err(|error| error.to_string())?;
    let window = window.ok_or_else(|| "winit did not create the normal 3dmod window".to_owned())?;
    let raw_handle = window
        .window_handle()
        .map_err(|error| error.to_string())?
        .as_raw();
    let display = config.display();
    let context_attributes = ContextAttributesBuilder::new()
        .with_profile(GlProfile::Compatibility)
        .with_context_api(ContextApi::OpenGl(Some(Version::new(2, 1))))
        .build(Some(raw_handle));
    let not_current = unsafe { display.create_context(&config, &context_attributes) }
        .map_err(|error| error.to_string())?;
    let surface_attributes = SurfaceAttributesBuilder::<WindowSurface>::new().build(
        raw_handle,
        NonZeroU32::new(width).expect("positive width"),
        NonZeroU32::new(height).expect("positive height"),
    );
    let surface = unsafe { display.create_window_surface(&config, &surface_attributes) }
        .map_err(|error| error.to_string())?;
    let context = not_current
        .make_current(&surface)
        .map_err(|error| error.to_string())?;
    surface
        .set_swap_interval(
            &context,
            SwapInterval::Wait(NonZeroU32::new(1).expect("nonzero")),
        )
        .map_err(|error| error.to_string())?;
    let gl =
        unsafe { glow::Context::from_loader_function_cstr(|name| display.get_proc_address(name)) };
    let mut renderer = SectionRenderer::new(gl, &|name| display.get_proc_address(name))?;
    renderer.upload_section(&mut host.view);
    let mut window_size = PhysicalSize::new(width, height);
    let mut main_cursor = winit::dpi::PhysicalPosition::new(0., 0.);
    let mut main_dragging = false;
    let mut modifiers = winit::keyboard::ModifiersState::empty();
    let main_window_id = window.id();
    let mut zap_controllers: Vec<Box<ZapFuncs>> = Vec::new();
    let mut slicers = SlicerRegistry::default();
    let mut xyz_controllers: Vec<XyzWindow> = Vec::new();
    let mut info_controllers: Vec<InfoCbState> = Vec::new();
    let mut tool_surfaces: Vec<ToolSurface> = Vec::new();
    // `imod.cpp` opens command-line-requested tools before entering the Qt
    // loop.  Make the equivalent controller/window pairs before winit starts
    // dispatching events, so `-Z`, `-S`, and `-xyz` are not deferred until a
    // user presses a shortcut in the primary image window.
    for requested in std::mem::take(&mut host.initial_tools) {
        let (kind, controller_index, tool) = match requested {
            InitialToolWindow::Zap => {
                zap_controllers.push(ZapFuncs::new(
                    host.view.as_mut() as *mut ImodView,
                    ZAP_WINDOW_TYPE,
                ));
                (ToolKind::Zap, zap_controllers.len() - 1, "Zap")
            }
            InitialToolWindow::Slicer => {
                let axis = unsafe { host.view.li.as_ref().map_or(3, |li| li.axis) };
                slicer_open(
                    &mut slicers,
                    SlicerView {
                        xsize: host.view.xsize,
                        ysize: host.view.ysize,
                        zsize: host.view.zsize,
                        xybin: host.view.xybin,
                        zbin: host.view.zbin,
                        zscale: host.model.zscale,
                        xmouse: host.view.xmouse,
                        ymouse: host.view.ymouse,
                        zmouse: host.view.zmouse,
                        cur_time: host.view.cur_time,
                        num_times: host.view.num_times,
                        track_mouse_for_plugs: host.view.track_mouse_for_plugs,
                        xmovie: host.view.xmovie,
                        ymovie: host.view.ymovie,
                        zmovie: host.view.zmovie,
                        movie_started_by_slicer: false,
                        image_axis: axis,
                    },
                    0,
                );
                (ToolKind::Slicer, slicers.slicers.len() - 1, "Slicer")
            }
            InitialToolWindow::Xyz => {
                xyz_controllers.push(XyzWindow::new(XyzViewState {
                    xsize: host.view.xsize,
                    ysize: host.view.ysize,
                    zsize: host.view.zsize,
                    xmouse: host.view.xmouse,
                    ymouse: host.view.ymouse,
                    zmouse: host.view.zmouse,
                    zscale: host.model.zscale,
                    cur_time: host.view.cur_time,
                    num_times: host.view.num_times,
                    ushort_store: host.view.ushort_store != 0,
                    rgb_store: host.view.rgb_store != 0,
                    has_pyramid_cache: !host.view.pyr_cache.is_null(),
                    ..Default::default()
                }));
                (ToolKind::Xyz, xyz_controllers.len() - 1, "XYZ")
            }
        };
        let tool_window = event_loop
            .create_window(
                Window::default_attributes()
                    .with_title(format!("3dmod {tool}"))
                    .with_inner_size(PhysicalSize::new(440, 260)),
            )
            .map_err(|error| error.to_string())?;
        let tool_size = tool_window.inner_size();
        let tool_handle = tool_window
            .window_handle()
            .map_err(|error| error.to_string())?
            .as_raw();
        let attributes = ContextAttributesBuilder::new()
            .with_profile(GlProfile::Compatibility)
            .with_context_api(ContextApi::OpenGl(Some(Version::new(2, 1))))
            .build(Some(tool_handle));
        let not_current = unsafe { display.create_context(&config, &attributes) }
            .map_err(|error| error.to_string())?;
        let surface_attributes = SurfaceAttributesBuilder::<WindowSurface>::new().build(
            tool_handle,
            NonZeroU32::new(tool_size.width.max(1)).expect("positive width"),
            NonZeroU32::new(tool_size.height.max(1)).expect("positive height"),
        );
        let tool_surface = unsafe { display.create_window_surface(&config, &surface_attributes) }
            .map_err(|error| error.to_string())?;
        let tool_context = not_current
            .make_current(&tool_surface)
            .map_err(|error| error.to_string())?;
        let tool_gl = unsafe {
            glow::Context::from_loader_function_cstr(|name| display.get_proc_address(name))
        };
        let mut tool_renderer =
            SectionRenderer::new(tool_gl, &|name| display.get_proc_address(name))?;
        tool_renderer.upload_section(&mut host.view);
        tool_surfaces.push(ToolSurface {
            kind,
            controller_index,
            window: tool_window,
            surface: tool_surface,
            context: tool_context,
            renderer: tool_renderer,
            size: tool_size,
            cursor: winit::dpi::PhysicalPosition::new(0., 0.),
            dragging: false,
        });
    }
    event_loop
        .run(move |event, target| match event {
            Event::UserEvent(NormalHostCommand::DrawModel) => window.request_redraw(),
            Event::WindowEvent {
                window_id,
                event: WindowEvent::CloseRequested,
            } if window_id == main_window_id => target.exit(),
            Event::WindowEvent {
                window_id,
                event: WindowEvent::CloseRequested,
            } => {
                let Some(surface_index) = tool_surfaces
                    .iter()
                    .position(|tool| tool.window.id() == window_id)
                else {
                    return;
                };
                let closed = tool_surfaces.remove(surface_index);
                match closed.kind {
                    ToolKind::Zap => {
                        zap_controllers.remove(closed.controller_index);
                    }
                    ToolKind::Slicer => {
                        slicers.slicers.remove(closed.controller_index);
                    }
                    ToolKind::Xyz => {
                        xyz_controllers.remove(closed.controller_index);
                    }
                    ToolKind::Information => {
                        info_controllers.remove(closed.controller_index);
                    }
                }
                for surface in &mut tool_surfaces {
                    if surface.kind == closed.kind
                        && surface.controller_index > closed.controller_index
                    {
                        surface.controller_index -= 1;
                    }
                }
            }
            Event::WindowEvent {
                window_id,
                event: WindowEvent::Resized(size),
            } if window_id == main_window_id => {
                window_size = size;
                if let (Some(width), Some(height)) =
                    (NonZeroU32::new(size.width), NonZeroU32::new(size.height))
                {
                    let _ = context.make_current(&surface);
                    surface.resize(&context, width, height);
                }
                window.request_redraw();
            }
            Event::WindowEvent {
                window_id,
                event: WindowEvent::Resized(size),
            } => {
                if let Some(tool) = tool_surfaces
                    .iter_mut()
                    .find(|tool| tool.window.id() == window_id)
                {
                    tool.size = size;
                    if let (Some(width), Some(height)) =
                        (NonZeroU32::new(size.width), NonZeroU32::new(size.height))
                    {
                        let _ = tool.context.make_current(&tool.surface);
                        tool.surface.resize(&tool.context, width, height);
                    }
                    tool.window.request_redraw();
                }
            }
            Event::WindowEvent {
                window_id,
                event: WindowEvent::ModifiersChanged(state),
            } if window_id == main_window_id => {
                modifiers = state.state();
            }
            Event::WindowEvent {
                window_id,
                event: WindowEvent::ModifiersChanged(state),
            } if tool_surfaces
                .iter()
                .any(|tool| tool.window.id() == window_id) =>
            {
                modifiers = state.state();
            }
            Event::WindowEvent {
                window_id,
                event: WindowEvent::KeyboardInput { event, .. },
            } if window_id == main_window_id && event.state == ElementState::Pressed => {
                let mut input_boundary = NormalInputBoundary::default();
                let mut default_key_handled = false;
                if let Some(key) = normal_input_key(&event.logical_key) {
                    let mut input_event = crate::imod::three_dmod::imod_input::InputKeyEvent {
                        key,
                        modifiers: normal_input_modifiers(modifiers),
                        ..Default::default()
                    };
                    crate::imod::three_dmod::imod_input::input_q_default_keys(
                        &mut input_event,
                        &mut host.view,
                        &mut input_boundary,
                    );
                    default_key_handled = input_event.accepted;
                }
                if default_key_handled || input_boundary.redraw {
                    synchronize_slicer_input_views(&mut slicers, &host.view);
                    let _ = context.make_current(&surface);
                    renderer.upload_section(&mut host.view);
                    for tool in &mut tool_surfaces {
                        let _ = tool.context.make_current(&tool.surface);
                        tool.renderer.upload_section(&mut host.view);
                    }
                    window.request_redraw();
                }
                let tool = match event.logical_key.as_ref() {
                    Key::Character("z") | Key::Character("Z") => Some("Zap"),
                    Key::Character("s") | Key::Character("S") => Some("Slicer"),
                    Key::Character("x") | Key::Character("X") => Some("XYZ"),
                    Key::Character("i") | Key::Character("I") => Some("Information"),
                    _ => None,
                };
                if let Some(tool) = tool {
                    let (kind, controller_index) = match tool {
                        "Zap" => {
                            zap_controllers.push(ZapFuncs::new(
                                host.view.as_mut() as *mut ImodView,
                                ZAP_WINDOW_TYPE,
                            ));
                            (ToolKind::Zap, zap_controllers.len() - 1)
                        }
                        "Slicer" => {
                            let axis = unsafe { host.view.li.as_ref().map_or(3, |li| li.axis) };
                            slicer_open(
                                &mut slicers,
                                SlicerView {
                                    xsize: host.view.xsize,
                                    ysize: host.view.ysize,
                                    zsize: host.view.zsize,
                                    xybin: host.view.xybin,
                                    zbin: host.view.zbin,
                                    zscale: host.model.zscale,
                                    xmouse: host.view.xmouse,
                                    ymouse: host.view.ymouse,
                                    zmouse: host.view.zmouse,
                                    cur_time: host.view.cur_time,
                                    num_times: host.view.num_times,
                                    track_mouse_for_plugs: host.view.track_mouse_for_plugs,
                                    xmovie: host.view.xmovie,
                                    ymovie: host.view.ymovie,
                                    zmovie: host.view.zmovie,
                                    movie_started_by_slicer: false,
                                    image_axis: axis,
                                },
                                0,
                            );
                            (ToolKind::Slicer, slicers.slicers.len() - 1)
                        }
                        "XYZ" => {
                            xyz_controllers.push(XyzWindow::new(XyzViewState {
                                xsize: host.view.xsize,
                                ysize: host.view.ysize,
                                zsize: host.view.zsize,
                                xmouse: host.view.xmouse,
                                ymouse: host.view.ymouse,
                                zmouse: host.view.zmouse,
                                zscale: host.model.zscale,
                                cur_time: host.view.cur_time,
                                num_times: host.view.num_times,
                                ushort_store: host.view.ushort_store != 0,
                                rgb_store: host.view.rgb_store != 0,
                                has_pyramid_cache: !host.view.pyr_cache.is_null(),
                                ..Default::default()
                            }));
                            (ToolKind::Xyz, xyz_controllers.len() - 1)
                        }
                        "Information" => {
                            info_controllers.push(InfoCbState::default());
                            (ToolKind::Information, info_controllers.len() - 1)
                        }
                        _ => unreachable!(),
                    };
                    if let Ok(tool_window) = target.create_window(
                        Window::default_attributes()
                            .with_title(format!("3dmod {tool}"))
                            .with_inner_size(PhysicalSize::new(440, 260)),
                    ) {
                        let tool_size = tool_window.inner_size();
                        let tool_handle = tool_window.window_handle().map(|handle| handle.as_raw());
                        if let Ok(tool_handle) = tool_handle {
                            let attributes = ContextAttributesBuilder::new()
                                .with_profile(GlProfile::Compatibility)
                                .with_context_api(ContextApi::OpenGl(Some(Version::new(2, 1))))
                                .build(Some(tool_handle));
                            if let Ok(not_current) =
                                unsafe { display.create_context(&config, &attributes) }
                            {
                                let surface_attributes =
                                    SurfaceAttributesBuilder::<WindowSurface>::new().build(
                                        tool_handle,
                                        NonZeroU32::new(tool_size.width.max(1))
                                            .expect("positive width"),
                                        NonZeroU32::new(tool_size.height.max(1))
                                            .expect("positive height"),
                                    );
                                if let Ok(tool_surface) = unsafe {
                                    display.create_window_surface(&config, &surface_attributes)
                                } && let Ok(tool_context) =
                                    not_current.make_current(&tool_surface)
                                {
                                    let tool_gl = unsafe {
                                        glow::Context::from_loader_function_cstr(|name| {
                                            display.get_proc_address(name)
                                        })
                                    };
                                    if let Ok(mut tool_renderer) =
                                        SectionRenderer::new(tool_gl, &|name| {
                                            display.get_proc_address(name)
                                        })
                                    {
                                        tool_renderer.upload_section(&mut host.view);
                                        tool_surfaces.push(ToolSurface {
                                            kind,
                                            controller_index,
                                            window: tool_window,
                                            surface: tool_surface,
                                            context: tool_context,
                                            renderer: tool_renderer,
                                            size: tool_size,
                                            cursor: winit::dpi::PhysicalPosition::new(0., 0.),
                                            dragging: false,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Event::WindowEvent {
                window_id,
                event: WindowEvent::MouseWheel { delta, .. },
            } if window_id == main_window_id => {
                renderer.wheel_zoom(match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(position) => position.y as f32,
                });
                window.request_redraw();
            }
            Event::WindowEvent {
                window_id,
                event: WindowEvent::MouseWheel { delta, .. },
            } => {
                if let Some(tool) = tool_surfaces
                    .iter_mut()
                    .find(|tool| tool.window.id() == window_id)
                {
                    tool.renderer.wheel_zoom(match delta {
                        MouseScrollDelta::LineDelta(_, y) => y,
                        MouseScrollDelta::PixelDelta(position) => position.y as f32,
                    });
                    tool.window.request_redraw();
                }
            }
            Event::WindowEvent {
                window_id,
                event:
                    WindowEvent::MouseInput {
                        state,
                        button: MouseButton::Left,
                        ..
                    },
            } if window_id == main_window_id => {
                main_dragging = state == ElementState::Pressed;
            }
            Event::WindowEvent {
                window_id,
                event:
                    WindowEvent::MouseInput {
                        state,
                        button: MouseButton::Left,
                        ..
                    },
            } => {
                if let Some(tool) = tool_surfaces
                    .iter_mut()
                    .find(|tool| tool.window.id() == window_id)
                {
                    tool.dragging = state == ElementState::Pressed;
                }
            }
            Event::WindowEvent {
                window_id,
                event: WindowEvent::CursorMoved { position, .. },
            } if window_id == main_window_id => {
                renderer.update_view_mouse(&mut host.view, position, window_size);
                synchronize_slicer_input_views(&mut slicers, &host.view);
                if main_dragging {
                    renderer.pan_by_pixels(
                        position.x - main_cursor.x,
                        position.y - main_cursor.y,
                        window_size,
                    );
                    window.request_redraw();
                }
                main_cursor = position;
            }
            Event::WindowEvent {
                window_id,
                event: WindowEvent::CursorMoved { position, .. },
            } => {
                if let Some(tool) = tool_surfaces
                    .iter_mut()
                    .find(|tool| tool.window.id() == window_id)
                {
                    tool.renderer
                        .update_view_mouse(&mut host.view, position, tool.size);
                    synchronize_slicer_input_views(&mut slicers, &host.view);
                    if tool.dragging {
                        tool.renderer.pan_by_pixels(
                            position.x - tool.cursor.x,
                            position.y - tool.cursor.y,
                            tool.size,
                        );
                        tool.window.request_redraw();
                    }
                    tool.cursor = position;
                }
            }
            Event::WindowEvent {
                window_id,
                event: WindowEvent::KeyboardInput { event, .. },
            } if event.state == ElementState::Pressed
                && tool_surfaces
                    .iter()
                    .any(|tool| tool.window.id() == window_id) =>
            {
                let mut input_boundary = NormalInputBoundary::default();
                if let Some(key) = normal_input_key(&event.logical_key) {
                    let mut input_event = crate::imod::three_dmod::imod_input::InputKeyEvent {
                        key,
                        modifiers: normal_input_modifiers(modifiers),
                        ..Default::default()
                    };
                    crate::imod::three_dmod::imod_input::input_q_default_keys(
                        &mut input_event,
                        &mut host.view,
                        &mut input_boundary,
                    );
                }
                if input_boundary.redraw {
                    synchronize_slicer_input_views(&mut slicers, &host.view);
                    let _ = context.make_current(&surface);
                    renderer.upload_section(&mut host.view);
                    for tool in &mut tool_surfaces {
                        let _ = tool.context.make_current(&tool.surface);
                        tool.renderer.upload_section(&mut host.view);
                        tool.window.request_redraw();
                    }
                    window.request_redraw();
                }
            }
            Event::WindowEvent {
                window_id,
                event: WindowEvent::RedrawRequested,
            } if window_id == main_window_id => {
                let _ = context.make_current(&surface);
                renderer.draw(window_size);
                renderer.draw_model(&host.view, &host.model);
                let _ = surface.swap_buffers(&context);
            }
            Event::WindowEvent {
                window_id,
                event: WindowEvent::RedrawRequested,
            } => {
                if let Some(tool) = tool_surfaces
                    .iter_mut()
                    .find(|tool| tool.window.id() == window_id)
                {
                    let _ = tool.context.make_current(&tool.surface);
                    tool.renderer.draw(tool.size);
                    tool.renderer.draw_model(&host.view, &host.model);
                    let _ = tool.surface.swap_buffers(&tool.context);
                }
            }
            Event::AboutToWait => {
                window.request_redraw();
                for tool in &tool_surfaces {
                    tool.window.request_redraw();
                }
            }
            _ => {}
        })
        .map_err(|error| error.to_string())
}

#[cfg(all(test, feature = "three-dmod-gl"))]
mod tests {
    use super::{
        normal_input_key, normal_input_modifiers, synchronize_slicer_input_views,
        window_to_image_coordinates,
    };
    use crate::imod::three_dmod::imod_input::{INPUT_CTRL, INPUT_SHIFT, KEY_PAGE_UP, KEY_UP};
    use crate::imod::three_dmod::imodview::ImodView;
    use crate::imod::three_dmod::slicer::{SlicerRegistry, SlicerView, slicer_open};
    use winit::dpi::{PhysicalPosition, PhysicalSize};
    use winit::keyboard::{Key, NamedKey};

    #[test]
    fn normal_host_key_mapping_targets_translated_input_key_values() {
        assert_eq!(
            normal_input_key(&Key::Named(NamedKey::ArrowUp)),
            Some(KEY_UP)
        );
        assert_eq!(
            normal_input_key(&Key::Named(NamedKey::PageUp)),
            Some(KEY_PAGE_UP)
        );
        assert_eq!(normal_input_key(&Key::Character("c".into())), Some(67));
    }

    #[test]
    fn modifier_mapping_preserves_shift_and_control_bits() {
        let state =
            winit::keyboard::ModifiersState::SHIFT | winit::keyboard::ModifiersState::CONTROL;
        assert_eq!(normal_input_modifiers(state), INPUT_SHIFT | INPUT_CTRL);
    }

    #[test]
    fn pointer_mapping_matches_letterbox_pan_and_zoom_transform() {
        // A 4:3 image inside a wide 16:9 surface is centered in a 1200x900
        // viewport; its top-left image pixel remains coordinate zero.
        let top_left = window_to_image_coordinates(
            (400, 300),
            1.,
            (0., 0.),
            PhysicalPosition::new(200., 0.),
            PhysicalSize::new(1600, 900),
        )
        .unwrap();
        assert_eq!(top_left, (0., 0.));
        let panned = window_to_image_coordinates(
            (400, 300),
            2.,
            (0.5, -0.5),
            PhysicalPosition::new(800., 450.),
            PhysicalSize::new(1600, 900),
        )
        .unwrap();
        assert!((panned.0 - 150.).abs() < 0.01);
        assert!((panned.1 - 112.5).abs() < 0.01);
    }

    #[test]
    fn shared_view_input_refreshes_open_slicer_without_erasing_local_ownership() {
        let mut registry = SlicerRegistry::default();
        let mut slicer_view = SlicerView {
            xsize: 100,
            ysize: 80,
            zsize: 12,
            movie_started_by_slicer: true,
            ..Default::default()
        };
        slicer_view.xmouse = 2.;
        slicer_open(&mut registry, slicer_view, 0);

        let mut view = ImodView::default();
        view.xmouse = 41.5;
        view.ymouse = 22.25;
        view.zmouse = 7.;
        view.cur_time = 3;
        view.num_times = 9;
        view.track_mouse_for_plugs = 1;
        view.xmovie = 1;
        view.ymovie = 2;
        view.zmovie = 3;
        synchronize_slicer_input_views(&mut registry, &view);

        let refreshed = &registry.slicers[0].view;
        assert_eq!(
            (refreshed.xmouse, refreshed.ymouse, refreshed.zmouse),
            (41.5, 22.25, 7.)
        );
        assert_eq!((refreshed.cur_time, refreshed.num_times), (3, 9));
        assert_eq!(
            (refreshed.xmovie, refreshed.ymovie, refreshed.zmovie),
            (1, 2, 3)
        );
        assert!(refreshed.movie_started_by_slicer);
    }
}
