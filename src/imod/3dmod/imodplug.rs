//! Translation of `IMOD/3dmod/imodplug.cpp`, `imodplug.h`, and `imodplugP.h`.
//!
//! The original unit keeps a process-global list of resident special modules
//! and dynamically loaded shared libraries.  [`ImodPlugState`] is that list.
//! Qt menu/file-dialog work, the active viewer, and ABI-specific dynamic
//! library loading are intentionally explicit boundaries instead of hidden
//! replacement implementations.
#![allow(dead_code)]

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use crate::imod::three_dmod::imodview::ImodView;
use crate::imod::three_dmod::mv_window::KeyEvent;

pub const IMOD_PLUG_MENU: i32 = 1;
pub const IMOD_PLUG_TOOL: i32 = 2;
pub const IMOD_PLUG_PROC: i32 = 4;
pub const IMOD_PLUG_VIEW: i32 = 8;
pub const IMOD_PLUG_KEYS: i32 = 16;
pub const IMOD_PLUG_FILE: i32 = 32;
pub const IMOD_PLUG_MESSAGE: i32 = 64;
pub const IMOD_PLUG_MOUSE: i32 = 128;
pub const IMOD_PLUG_EVENT: i32 = 256;
pub const IMOD_PLUG_CHOOSER: i32 = 512;

pub const IMOD_REASON_EXECUTE: i32 = 1;
pub const IMOD_REASON_STARTUP: i32 = 3;
pub const IMOD_REASON_MODUPDATE: i32 = 4;
pub const IMOD_REASON_NEWMODEL: i32 = 5;

const IP_INFO: usize = 0;
const IP_EXECUTE: usize = 1;
const IP_EXECUTE_TYPE: usize = 2;
const IP_KEYS: usize = 3;
const IP_MOUSE: usize = 4;
const IP_EVENT: usize = 5;
const IP_EXECUTE_MESSAGE: usize = 6;
const IP_OPEN_FILE_NAME: usize = 7;
const IP_OPEN_FILE_NAMES: usize = 8;
const IP_SAVE_FILE_NAME: usize = 9;

/// Portable source projection of `QMouseEvent`.  Its native Qt payload is
/// deliberately retained at the caller/plugin ABI boundary.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MouseEvent {
    pub button: i32,
    pub modifiers: i32,
}

/// Portable source projection of the selected `QEvent` passed to plugins.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PlugEvent {
    pub event_type: i32,
}

/// `SpecialModule` from `special_module.h`, expressed as an object-safe Rust
/// plugin ABI.  Defaults exactly model a NULL C++ function member.
pub trait SpecialModule {
    fn imod_plug_info(&mut self) -> (&str, i32);
    fn has_imod_plug_execute(&self) -> bool {
        false
    }
    fn imod_plug_execute(&mut self, _view: &mut ImodView) {}
    fn has_imod_plug_execute_type(&self) -> bool {
        false
    }
    fn imod_plug_execute_type(&mut self, _view: &mut ImodView, _type_: i32, _reason: i32) {}
    fn imod_plug_keys(&mut self, _view: &mut ImodView, _event: &KeyEvent) -> i32 {
        0
    }
    fn imod_plug_mouse(
        &mut self,
        _view: &mut ImodView,
        _event: &MouseEvent,
        _imx: f32,
        _imy: f32,
        _but1: i32,
        _but2: i32,
        _but3: i32,
    ) -> i32 {
        0
    }
    fn imod_plug_event(
        &mut self,
        _view: &mut ImodView,
        _event: &PlugEvent,
        _imx: f32,
        _imy: f32,
    ) -> i32 {
        0
    }
    fn imod_plug_execute_message(
        &mut self,
        _view: &mut ImodView,
        _strings: &[String],
        _arg: &mut usize,
    ) -> i32 {
        1
    }
    fn imod_plug_open_file_name(
        &mut self,
        _caption: &str,
        _dir: &str,
        _filter: &str,
    ) -> Option<String> {
        None
    }
    fn imod_plug_open_file_names(
        &mut self,
        _caption: &str,
        _dir: &str,
        _filter: &str,
    ) -> Option<Vec<String>> {
        None
    }
    fn imod_plug_save_file_name(&mut self, _caption: &str) -> Option<String> {
        None
    }
}

/// `PlugData` from `imodplug.cpp`.  `library` is the loaded-library identity;
/// resolving C symbols into this trait is the dynamic-loader boundary.
pub struct PlugData {
    pub name: String,
    pub library: Option<PathBuf>,
    pub module: Box<dyn SpecialModule>,
    pub type_: i32,
}

impl core::fmt::Debug for PlugData {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("PlugData")
            .field("name", &self.name)
            .field("library", &self.library)
            .field("type_", &self.type_)
            .finish()
    }
}

/// Process-static `sPlugList`, `sNumInternal`, and `sEventSource`.
pub struct ImodPlugState {
    pub plug_list: Vec<PlugData>,
    pub num_internal: usize,
    pub event_source: i32,
    pub chooser_plugin: bool,
    pub browser_dir: String,
    /// The source creates Bead Fixer and Line Track here.  They are supplied
    /// by their paired translations so plugin loading remains ABI-independent.
    pub internal_modules: Vec<Box<dyn SpecialModule>>,
}

impl ImodPlugState {
    /// `ipAddInternalModules`.
    pub fn ip_add_internal_modules(&mut self, native: &mut dyn ImodPlugNativeBoundary) -> usize {
        for mut module in self.internal_modules.drain(..) {
            let (name, type_) = module.imod_plug_info();
            native.debug(&format!("Added {name} module to Special menu\n"));
            self.plug_list.push(PlugData {
                name: name.into(),
                library: None,
                module,
                type_,
            });
        }
        self.plug_list.len()
    }
    /// `imodPlugLoad`.
    pub fn imod_plug_load(
        &mut self,
        native: &mut dyn ImodPlugNativeBoundary,
        path: PathBuf,
    ) -> i32 {
        let Ok(mut module) = native.load_plugin(&path) else {
            native.debug(&format!(
                "Warning: {} cannot be loaded as a 3dmod plugin\n",
                path.display()
            ));
            return 2;
        };
        let (name, type_) = module.imod_plug_info();
        if self
            .plug_list
            .iter()
            .any(|plug| plug.type_ == type_ && plug.name == name)
        {
            return 3;
        }
        if type_ & IMOD_PLUG_CHOOSER != 0 {
            self.chooser_plugin = true;
        }
        native.debug(&format!("loaded plugin : {} {name}\n", path.display()));
        self.plug_list.push(PlugData {
            name: name.into(),
            library: Some(path),
            module,
            type_,
        });
        0
    }
    /// `imodPlugOpen`.
    pub fn imod_plug_open(&mut self, view: &mut ImodView, item: usize) -> bool {
        let Some(plug) = self.plug_list.get_mut(item) else {
            return false;
        };
        // A NULL `mExecute` in C is represented by `has_execute`; dynamic ABI
        // adapters expose it when they resolve that optional symbol.
        if plug.module.has_imod_plug_execute() {
            plug.module.imod_plug_execute(view);
            return true;
        }
        if !plug.module.has_imod_plug_execute_type() {
            return false;
        }
        plug.module
            .imod_plug_execute_type(view, IMOD_PLUG_MENU, IMOD_REASON_EXECUTE);
        true
    }
    /// `imodPlugLoaded`.
    pub fn imod_plug_loaded(&self, type_: i32) -> usize {
        self.plug_list
            .iter()
            .filter(|plug| plug.type_ & type_ != 0)
            .count()
    }
    /// `imodPlugMenu`.
    pub fn imod_plug_menu(&self) -> Vec<PlugMenuAction> {
        self.plug_list
            .iter()
            .enumerate()
            .filter(|(_, plug)| plug.type_ & IMOD_PLUG_MENU != 0)
            .map(|(item, plug)| PlugMenuAction {
                text: plug.name.clone(),
                item,
            })
            .collect()
    }
    /// `imodPlugCall`.
    pub fn imod_plug_call(&mut self, view: &mut ImodView, type_: i32, reason: i32) -> usize {
        let mut called = 0;
        for plug in &mut self.plug_list {
            if plug.module.has_imod_plug_execute_type() {
                plug.module.imod_plug_execute_type(view, type_, reason);
                called += 1;
            }
        }
        called
    }
    /// `imodPlugHandleKey`.
    pub fn imod_plug_handle_key(
        &mut self,
        view: &mut ImodView,
        event: &KeyEvent,
        source: i32,
    ) -> i32 {
        for plug in &mut self.plug_list {
            if plug.type_ & IMOD_PLUG_KEYS != 0 {
                self.event_source = source;
                let handled = plug.module.imod_plug_keys(view, event);
                self.event_source = -1;
                if handled != 0 {
                    return 1;
                }
            }
        }
        0
    }
    /// `imodPlugHandleMouse`.
    pub fn imod_plug_handle_mouse(
        &mut self,
        view: &mut ImodView,
        event: &MouseEvent,
        imx: f32,
        imy: f32,
        but1: i32,
        but2: i32,
        but3: i32,
        source: i32,
    ) -> i32 {
        let mut need_draw = 0;
        for plug in &mut self.plug_list {
            if plug.type_ & IMOD_PLUG_MOUSE != 0 {
                self.event_source = source;
                let handled = plug
                    .module
                    .imod_plug_mouse(view, event, imx, imy, but1, but2, but3);
                self.event_source = -1;
                if handled & 1 != 0 {
                    return handled;
                }
                need_draw |= handled;
            }
        }
        need_draw
    }
    /// `imodPlugHandleEvent`.
    pub fn imod_plug_handle_event(
        &mut self,
        view: &mut ImodView,
        event: &PlugEvent,
        imx: f32,
        imy: f32,
        source: i32,
    ) -> i32 {
        let mut need_draw = 0;
        for plug in &mut self.plug_list {
            if plug.type_ & IMOD_PLUG_EVENT != 0 {
                self.event_source = source;
                let handled = plug.module.imod_plug_event(view, event, imx, imy);
                self.event_source = -1;
                if handled & 1 != 0 {
                    return handled;
                }
                need_draw |= handled;
            }
        }
        need_draw
    }
    /// `imodPlugMessage`.
    pub fn imod_plug_message(
        &mut self,
        view: &mut ImodView,
        strings: &[String],
        arg: &mut usize,
    ) -> i32 {
        for plug in &mut self.plug_list {
            if plug.type_ & IMOD_PLUG_MESSAGE != 0 {
                let words: Vec<_> = plug.name.split_whitespace().collect();
                if strings.get(*arg..).is_some_and(|message| {
                    message.len() >= words.len()
                        && words
                            .iter()
                            .zip(message)
                            .all(|(word, message_word)| *word == message_word)
                }) {
                    *arg += words.len();
                    return plug.module.imod_plug_execute_message(view, strings, arg);
                }
            }
        }
        1
    }
    /// `imodPlugGetOpenName`.
    pub fn imod_plug_get_open_name(
        &mut self,
        native: &mut dyn ImodPlugNativeBoundary,
        caption: &str,
        dir: &str,
        filter: &str,
    ) -> String {
        if self.chooser_plugin {
            for plug in &mut self.plug_list {
                if plug.type_ & IMOD_PLUG_CHOOSER != 0 {
                    if let Some(name) = plug.module.imod_plug_open_file_name(caption, dir, filter) {
                        return name;
                    }
                }
            }
        }
        let use_dir = if dir.is_empty() {
            native.current_dir()
        } else {
            dir.into()
        };
        native.open_file_name(caption, &use_dir, filter)
    }
    /// `imodPlugGetOpenNames`.
    pub fn imod_plug_get_open_names(
        &mut self,
        native: &mut dyn ImodPlugNativeBoundary,
        caption: &str,
        dir: &str,
        filter: &str,
    ) -> Vec<String> {
        if self.chooser_plugin {
            for plug in &mut self.plug_list {
                if plug.type_ & IMOD_PLUG_CHOOSER != 0 {
                    if let Some(names) = plug.module.imod_plug_open_file_names(caption, dir, filter)
                    {
                        return names;
                    }
                }
            }
        }
        let use_dir = if dir.is_empty() {
            native.current_dir()
        } else {
            dir.into()
        };
        native.open_file_names(caption, &use_dir, filter)
    }
    /// `imodPlugGetSaveName`.
    pub fn imod_plug_get_save_name(
        &mut self,
        native: &mut dyn ImodPlugNativeBoundary,
        caption: &str,
    ) -> String {
        if self.chooser_plugin {
            for plug in &mut self.plug_list {
                if plug.type_ & IMOD_PLUG_CHOOSER != 0 {
                    if let Some(name) = plug.module.imod_plug_save_file_name(caption) {
                        return name;
                    }
                }
            }
        }
        let directory = if self.browser_dir.is_empty() {
            native.current_dir()
        } else {
            self.browser_dir.clone()
        };
        let name = native.save_file_name(caption, &directory);
        native.manage_browser_dir(&name);
        name
    }
    /// `ivwGetPlugEventSource`.
    pub fn ivw_get_plug_event_source(&self) -> i32 {
        self.event_source
    }

    /// `imodPlugLoadDir`.
    pub fn imod_plug_load_dir(
        &mut self,
        native: &mut dyn ImodPlugNativeBoundary,
        directory: &Path,
    ) -> usize {
        let extension = if cfg!(target_os = "windows") {
            "dll"
        } else if cfg!(target_os = "macos") {
            "dylib"
        } else {
            "so"
        };
        let mut loaded = 0;
        if let Ok(entries) = fs::read_dir(directory) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|suffix| suffix == extension)
                    && self.imod_plug_load(native, path) == 0
                {
                    loaded += 1;
                }
            }
        }
        loaded
    }
    /// `imodPlugOpenByName`.
    pub fn imod_plug_open_by_name(&mut self, view: &mut ImodView, name: &str) {
        for item in 0..self.plug_list.len() {
            if self.plug_list[item].name == name {
                let _ = self.imod_plug_open(view, item);
            }
        }
    }
    /// `imodPlugOpenAllExternal`.
    pub fn imod_plug_open_all_external(&mut self, view: &mut ImodView) {
        for item in self.num_internal..self.plug_list.len() {
            let _ = self.imod_plug_open(view, item);
        }
    }

    /// `imodPlugInit`.
    pub fn imod_plug_init(&mut self, native: &mut dyn ImodPlugNativeBoundary) -> usize {
        self.plug_list.clear();
        self.ip_add_internal_modules(native);
        self.num_internal = self.plug_list.len();
        if let Some(directory) = env::var_os("IMOD_PLUGIN_DIR") {
            self.imod_plug_load_dir(native, Path::new(&directory));
        } else {
            let imod_dir = env::var("IMOD_DIR").unwrap_or_else(|_| "/usr/local/IMOD".into());
            self.imod_plug_load_dir(native, &Path::new(&imod_dir).join("lib/imodplug"));
        }
        if let Some(directory) = env::var_os("IMOD_CALIB_DIR") {
            self.imod_plug_load_dir(native, &Path::new(&directory).join("plugins"));
        }
        let imod_dir = env::var("IMOD_DIR").unwrap_or_else(|_| "/usr/local/IMOD".into());
        self.imod_plug_load_dir(native, &Path::new(&imod_dir).join("Plugins"));
        self.imod_plug_load_dir(native, Path::new("/usr/local/IMOD/plugins"));
        #[cfg(target_os = "windows")]
        {
            self.imod_plug_load_dir(native, Path::new("C:/Program Files/IMOD/lib/imodplug"));
            self.imod_plug_load_dir(native, Path::new("C:/Program Files/3dmod/lib/imodplug"));
        }
        self.imod_plug_load_dir(native, Path::new("usr/freeware/lib/imodplugs"));
        self.plug_list.len()
    }
}

impl Default for ImodPlugState {
    fn default() -> Self {
        Self {
            plug_list: Vec::new(),
            num_internal: 0,
            event_source: -1,
            chooser_plugin: false,
            browser_dir: String::new(),
            internal_modules: Vec::new(),
        }
    }
}

/// Dynamic shared-library and Qt file-chooser boundary for this source unit.
pub trait ImodPlugNativeBoundary {
    /// Equivalent to `QLibrary(path).load()` plus resolving `imodPlugInfo`.
    fn load_plugin(&mut self, path: &Path) -> Result<Box<dyn SpecialModule>, String>;
    fn debug(&mut self, _text: &str) {}
    fn current_dir(&self) -> String;
    fn open_file_name(&mut self, caption: &str, dir: &str, filter: &str) -> String;
    fn open_file_names(&mut self, caption: &str, dir: &str, filter: &str) -> Vec<String>;
    fn save_file_name(&mut self, caption: &str, dir: &str) -> String;
    fn manage_browser_dir(&mut self, _name: &str) {}
}

/// Menu `QAction` and `QSignalMapper` association made by `imodPlugMenu`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlugMenuAction {
    pub text: String,
    pub item: usize,
}

/// `ipGetFunction`.  In Rust dynamic symbols are installed as trait methods,
/// so this reports whether the source's dispatch selector is known.
pub fn ip_get_function(_plug: &PlugData, which: usize) -> bool {
    which <= IP_SAVE_FILE_NAME
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Plugin {
        name: String,
        type_: i32,
        response: i32,
        calls: usize,
    }
    impl SpecialModule for Plugin {
        fn imod_plug_info(&mut self) -> (&str, i32) {
            (&self.name, self.type_)
        }
        fn imod_plug_keys(&mut self, _: &mut ImodView, _: &KeyEvent) -> i32 {
            self.calls += 1;
            self.response
        }
        fn imod_plug_mouse(
            &mut self,
            _: &mut ImodView,
            _: &MouseEvent,
            _: f32,
            _: f32,
            _: i32,
            _: i32,
            _: i32,
        ) -> i32 {
            self.response
        }
        fn imod_plug_execute_message(
            &mut self,
            _: &mut ImodView,
            _: &[String],
            _: &mut usize,
        ) -> i32 {
            self.response
        }
    }
    struct Native;
    impl ImodPlugNativeBoundary for Native {
        fn load_plugin(&mut self, _: &Path) -> Result<Box<dyn SpecialModule>, String> {
            Err("test".into())
        }
        fn current_dir(&self) -> String {
            "/cwd".into()
        }
        fn open_file_name(&mut self, _: &str, dir: &str, _: &str) -> String {
            dir.into()
        }
        fn open_file_names(&mut self, _: &str, dir: &str, _: &str) -> Vec<String> {
            vec![dir.into()]
        }
        fn save_file_name(&mut self, _: &str, dir: &str) -> String {
            dir.into()
        }
    }
    #[test]
    fn internal_modules_and_menu_follow_source_order() {
        let mut state = ImodPlugState {
            internal_modules: vec![
                Box::new(Plugin {
                    name: "One".into(),
                    type_: IMOD_PLUG_MENU,
                    ..Plugin::default()
                }),
                Box::new(Plugin {
                    name: "Two".into(),
                    type_: IMOD_PLUG_KEYS,
                    ..Plugin::default()
                }),
            ],
            ..Default::default()
        };
        let mut native = Native;
        state.ip_add_internal_modules(&mut native);
        assert_eq!(
            state.imod_plug_menu(),
            vec![PlugMenuAction {
                text: "One".into(),
                item: 0
            }]
        );
        assert_eq!(state.imod_plug_loaded(IMOD_PLUG_KEYS), 1);
    }
    #[test]
    fn mouse_combines_draw_and_stops_on_handled() {
        let mut state = ImodPlugState {
            plug_list: vec![
                PlugData {
                    name: "A".into(),
                    library: None,
                    module: Box::new(Plugin {
                        type_: IMOD_PLUG_MOUSE,
                        response: 2,
                        ..Plugin::default()
                    }),
                    type_: IMOD_PLUG_MOUSE,
                },
                PlugData {
                    name: "B".into(),
                    library: None,
                    module: Box::new(Plugin {
                        type_: IMOD_PLUG_MOUSE,
                        response: 1,
                        ..Plugin::default()
                    }),
                    type_: IMOD_PLUG_MOUSE,
                },
            ],
            ..Default::default()
        };
        assert_eq!(
            state.imod_plug_handle_mouse(
                &mut ImodView::default(),
                &MouseEvent::default(),
                0.,
                0.,
                0,
                0,
                0,
                9
            ),
            1
        );
        assert_eq!(state.ivw_get_plug_event_source(), -1);
    }
    #[test]
    fn message_advances_past_multiword_plugin_name() {
        let mut state = ImodPlugState {
            plug_list: vec![PlugData {
                name: "Bead Fixer".into(),
                library: None,
                module: Box::new(Plugin {
                    type_: IMOD_PLUG_MESSAGE,
                    response: 0,
                    ..Plugin::default()
                }),
                type_: IMOD_PLUG_MESSAGE,
            }],
            ..Default::default()
        };
        let mut arg = 0;
        assert_eq!(
            state.imod_plug_message(
                &mut ImodView::default(),
                &["Bead".into(), "Fixer".into(), "go".into()],
                &mut arg
            ),
            0
        );
        assert_eq!(arg, 2);
    }
}
