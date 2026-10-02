//! `IMOD/Etomo/src/etomo/process/BaseImodManager.java`.
//!
//! This class manages the opening, closing and sending of messages to 3dmod instances.
//! It stores information about each 3dmod instance that it manages.
//!
//! Inherit this class in order to create a customized 3dmod instances.
//!
//! **Inheritance.**  The class is abstract, and `ImodManager` overrides four of its
//! methods (`newImodState`, `getPrivateKey`, `isDualAxisOnly`, `isPerAxis`).  Those are
//! the [`BaseImodManagerHooks`] trait, whose default bodies are this class's; the
//! subclass hands its implementation to [`BaseImodManager::new`], and every call the
//! source makes on `this` goes through it.
//!
//! **Shape.**  The manager is reached from the event thread and from the request
//! handler thread, so its fields sit behind locks and every method takes `&self`.
//! `imodMap` holds each key's `Vector` of `ImodState`s; the states are `Arc`s, so a
//! state is looked up under the map lock and opened (which waits for 3dmod) without it.
//!
//! **Exceptions.**  Checked `AxisTypeException`, `SystemProcessException` and
//! `IOException` are [`ImodManagerException`]'s variants.  The source's unchecked
//! `IllegalArgumentException` / `UnsupportedOperationException`, which propagate to the
//! Swing event loop and abandon the action, are its `Runtime` variant, so they abandon
//! the action the same way without unwinding.
//!
//! **Upstream bugs fixed in translation** (each is documented where it applies):
//! - the `imodMap` key a new state is stored under is not always the key `getVector`
//!   looks it up by (a missing single-axis FIRST-to-ONLY correction, an unconditional
//!   one, and the axis extension appended for keys that are not per axis), so a state
//!   can be created and then not found - silently doing nothing, or throwing a
//!   `NullPointerException` in the setters that do not test for null.  Stored states are
//!   now keyed exactly as `getVector` looks them up;
//! - `get(String, AxisID, int)` indexes without a bounds check, `get(String)` and
//!   `get(String, AxisID)` call `lastElement()` on a vector `delete` may have emptied,
//!   and `get(String, AxisID, File)` dereferences a null file;
//! - `open(String, AxisID, int, Run3dmodMenuOptions)` dereferences a null `axisID` while
//!   building its exception message.

use super::base_process_manager::SystemProcessException;
use super::continuous_listener_target::ContinuousListenerTarget;
use super::imod_process::{BeadFixerMode, Run3dmodMenuOptions};
use super::imod_request_handler::ImodRequestHandler;
use super::imod_state::ImodState;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::axis_type_exception::AxisTypeException;
use crate::imod::etomo::r#type::file_type::FileType;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::utilities;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

pub const DEFAULT_BEADFIXER_DIAMETER: i32 = 3;

/// The checked exceptions the 3dmod layer throws, plus the unchecked ones it throws on
/// purpose (see the module comment).
#[derive(Debug)]
pub enum ImodManagerException {
    /// `etomo.type.AxisTypeException`.
    AxisType(AxisTypeException),
    /// `etomo.process.SystemProcessException`.
    SystemProcess(SystemProcessException),
    /// `java.io.IOException`.
    Io(std::io::Error),
    /// `IllegalArgumentException` / `UnsupportedOperationException` thrown by the
    /// source; the message is the exception's.
    Runtime(String),
}

impl std::fmt::Display for ImodManagerException {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImodManagerException::AxisType(e) => std::fmt::Display::fmt(e, f),
            ImodManagerException::SystemProcess(e) => std::fmt::Display::fmt(e, f),
            ImodManagerException::Io(e) => std::fmt::Display::fmt(e, f),
            ImodManagerException::Runtime(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for ImodManagerException {}

impl From<AxisTypeException> for ImodManagerException {
    fn from(e: AxisTypeException) -> ImodManagerException {
        ImodManagerException::AxisType(e)
    }
}

impl From<SystemProcessException> for ImodManagerException {
    fn from(e: SystemProcessException) -> ImodManagerException {
        ImodManagerException::SystemProcess(e)
    }
}

impl From<std::io::Error> for ImodManagerException {
    fn from(e: std::io::Error) -> ImodManagerException {
        ImodManagerException::Io(e)
    }
}

/// The names callers use for the same error.
pub type BaseImodManagerException = ImodManagerException;
pub type ImodManagerError = ImodManagerException;
pub type ImodError = ImodManagerException;

/// The `Vector<ImodState>` a key maps to.
type ImodVector = Vec<Arc<ImodState>>;

/// The methods of `BaseImodManager` a subclass overrides.  The default bodies are
/// `BaseImodManager`'s.
pub trait BaseImodManagerHooks: Send + Sync {
    /// Java protected `newImodState(String, String, AxisID, String, File, String[],
    /// String, File[])`.  Return an imodState specific to the key.
    #[allow(clippy::too_many_arguments)]
    fn new_imod_state(
        &self,
        base: &BaseImodManager,
        key: &str,
        file_extension: Option<&str>,
        axis_id: Option<AxisID>,
        dataset_name: Option<&str>,
        file: Option<&Path>,
        file_name_array: Option<&[String]>,
        subdir_name: Option<&str>,
        file_list: Option<&[PathBuf]>,
    ) -> Result<ImodState, ImodManagerException> {
        let _ = (
            key,
            file_extension,
            dataset_name,
            file_name_array,
            subdir_name,
            file_list,
        );
        Ok(ImodState::new_base_manager_file_axis_id(
            base.manager,
            file,
            axis_id,
        ))
    }

    /// Java `getPrivateKey`.  Return the public key, unless there is a corresponding
    /// private key that can have a different value from the public key.
    fn get_private_key(&self, public_key: &str) -> String {
        public_key.to_string()
    }

    /// Java `isDualAxisOnly`.  Return true if the 3dmod instance(s) are associated the
    /// whole dataset, rather then an individual axis.
    fn is_dual_axis_only(&self, key: &str) -> bool {
        let _ = key;
        false
    }

    /// Java `isPerAxis`.  Return true if there is a different 3dmod instance for each
    /// axis.
    fn is_per_axis(&self, key: &str) -> bool {
        let _ = key;
        true
    }
}

/// Java abstract `BaseImodManager`.
pub struct BaseImodManager {
    axis_type: Mutex<Option<AxisType>>,
    use_map: bool,
    debug: Mutex<bool>,

    /// Java `HashMap imodMap`.
    imod_map: Mutex<HashMap<String, ImodVector>>,

    /// Java protected final `manager`.
    pub manager: &'static dyn BaseManager,
    /// Java final `requestHandler`.  See [`BaseImodManager::start_request_handler`].
    request_handler: OnceLock<Option<Arc<ImodRequestHandler>>>,
    /// The subclass's overrides.
    hooks: Arc<dyn BaseImodManagerHooks>,
}

impl BaseImodManager {
    /// Java protected `BaseImodManager(BaseManager)`.
    ///
    /// The constructor's `ImodRequestHandler.getInstance(this)` needs `this` at its
    /// final address, which a Rust constructor does not have; it is
    /// [`BaseImodManager::start_request_handler`], which the owner calls once the
    /// manager is in place.
    pub fn new(
        manager: &'static dyn BaseManager,
        hooks: Arc<dyn BaseImodManagerHooks>,
    ) -> BaseImodManager {
        BaseImodManager {
            axis_type: Mutex::new(Some(AxisType::SingleAxis)),
            use_map: true,
            debug: Mutex::new(false),
            imod_map: Mutex::new(HashMap::new()),
            manager,
            request_handler: OnceLock::new(),
            hooks,
        }
    }

    /// The request-handler half of the Java constructor.  Only run the request handler
    /// when necesary. In Windows when 3dmod is listening to stdin and it wants to exit,
    /// it sends a request to stderr to ask that the stdin receive a stop listening
    /// command. This is because 3dmod in Windows can't exit when it is listening to
    /// stdin.
    pub fn start_request_handler(&'static self) {
        let _ = self.request_handler.get_or_init(|| {
            if utilities::is_windows_os() && etomo_director::ARGUMENTS.lock().unwrap().is_listen() {
                ImodRequestHandler::get_instance(self)
            } else {
                None
            }
        });
    }

    /// Java protected `newImodState(String, String, AxisID, String, File, String[],
    /// String, File[])`, dispatched to the subclass.
    #[allow(clippy::too_many_arguments)]
    fn new_imod_state_string_string_axis_id_string_file_string_array_string_file_array(
        &self,
        key: &str,
        file_extension: Option<&str>,
        axis_id: Option<AxisID>,
        dataset_name: Option<&str>,
        file: Option<&Path>,
        file_name_array: Option<&[String]>,
        subdir_name: Option<&str>,
        file_list: Option<&[PathBuf]>,
    ) -> Result<Arc<ImodState>, ImodManagerException> {
        self.hooks
            .new_imod_state(
                self,
                key,
                file_extension,
                axis_id,
                dataset_name,
                file,
                file_name_array,
                subdir_name,
                file_list,
            )
            .map(Arc::new)
    }

    /// Java `getPrivateKey`, dispatched to the subclass.
    pub fn get_private_key(&self, public_key: &str) -> String {
        self.hooks.get_private_key(public_key)
    }

    /// Java `isDualAxisOnly`, dispatched to the subclass.
    pub fn is_dual_axis_only(&self, key: &str) -> bool {
        self.hooks.is_dual_axis_only(key)
    }

    /// Java `isPerAxis`, dispatched to the subclass.
    pub fn is_per_axis(&self, key: &str) -> bool {
        self.hooks.is_per_axis(key)
    }

    /// Java final `setAxisType`.
    pub fn set_axis_type(&self, axis_type: Option<AxisType>) {
        *self.axis_type.lock().unwrap() = axis_type;
    }

    /// Java final `equalsAxisType`.
    pub fn equals_axis_type(&self, input: AxisType) -> bool {
        *self.axis_type.lock().unwrap() == Some(input)
    }

    /// Java final `getAxisTypeString`.
    pub fn get_axis_type_string(&self) -> String {
        match *self.axis_type.lock().unwrap() {
            None => AxisType::NotSet.to_string(),
            Some(axis_type) => axis_type.to_string(),
        }
    }

    /// `axisType.toString()` where the source dereferences the field.
    fn axis_type_to_string(&self) -> String {
        match *self.axis_type.lock().unwrap() {
            None => "null".to_string(),
            Some(axis_type) => axis_type.to_string(),
        }
    }

    /// The `imodMap` key a state for `key` and `axisID` is stored under.  Java stores
    /// under `key` for a null axis and `key + axisID.getExtension()` otherwise (in one
    /// overload after an unconditional FIRST-to-ONLY correction), while `getVector`
    /// looks up `key` for a key that is not per axis and corrects FIRST to ONLY only in
    /// a single-axis dataset.  Fixed in translation: stored exactly as `getVector`
    /// looks it up, so a state that was just created is found.
    fn map_key(&self, key: &str, axis_id: Option<AxisID>) -> String {
        let Some(mut axis_id) = axis_id else {
            return key.to_string();
        };
        if !self.is_per_axis(key) {
            return key.to_string();
        }
        if self.equals_axis_type(AxisType::SingleAxis) && axis_id == AxisID::First {
            axis_id = AxisID::Only;
        }
        key.to_string() + &axis_id.get_extension()
    }

    /// Java private final `newImod(String)`.
    fn new_imod_string(&self, key: &str) -> Result<i32, ImodManagerException> {
        self.new_imod_string_string_axis_id_string(key, None, None, None)
    }

    /// Java final `newImod(String, AxisID)`.
    pub fn new_imod_string_axis_id(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
    ) -> Result<i32, ImodManagerException> {
        self.new_imod_string_axis_id_string(key, axis_id, None)
    }

    /// Java public final `newImod(String, String, AxisID)`.
    pub fn new_imod_string_string_axis_id(
        &self,
        key: &str,
        file_extension: Option<&str>,
        axis_id: Option<AxisID>,
    ) -> Result<i32, ImodManagerException> {
        self.new_imod_string_string_axis_id_string(key, file_extension, axis_id, None)
    }

    /// Java public final `newImod(String, AxisID, String)`.
    pub fn new_imod_string_axis_id_string(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
        dataset_name: Option<&str>,
    ) -> Result<i32, ImodManagerException> {
        self.new_imod_string_string_axis_id_string(key, None, axis_id, dataset_name)
    }

    /// Java private final `newImod(String, String, AxisID, String)`.
    fn new_imod_string_string_axis_id_string(
        &self,
        key: &str,
        file_extension: Option<&str>,
        axis_id: Option<AxisID>,
        dataset_name: Option<&str>,
    ) -> Result<i32, ImodManagerException> {
        let key = self.get_private_key(key);
        let mut imod_map = self.imod_map.lock().unwrap();
        if self
            .get_vector_string_axis_id(&mut imod_map, &key, axis_id)?
            .is_none()
        {
            let vector = self.new_vector_string_string_axis_id_string(
                &key,
                file_extension,
                axis_id,
                dataset_name,
            )?;
            imod_map.insert(self.map_key(&key, axis_id), vector);
            return Ok(0);
        }
        let imod_state = self.new_imod_state_string_string_axis_id_string(
            &key,
            file_extension,
            axis_id,
            dataset_name,
        )?;
        let vector = self
            .get_vector_string_axis_id(&mut imod_map, &key, axis_id)?
            .unwrap();
        vector.push(Arc::clone(&imod_state));
        Ok(vector
            .iter()
            .rposition(|state| Arc::ptr_eq(state, &imod_state))
            .map_or(-1, |i| i as i32))
    }

    /// Java public final `newImod(String, AxisID, File[])`.
    pub fn new_imod_string_axis_id_file_array(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
        file_list: Option<&[PathBuf]>,
    ) -> Result<i32, ImodManagerException> {
        let key = self.get_private_key(key);
        let mut imod_map = self.imod_map.lock().unwrap();
        if self
            .get_vector_string_axis_id(&mut imod_map, &key, axis_id)?
            .is_none()
        {
            let vector = self.new_vector_string_axis_id_file_array(&key, axis_id, file_list)?;
            imod_map.insert(self.map_key(&key, axis_id), vector);
            return Ok(0);
        }
        let imod_state = self.new_imod_state_string_axis_id_file_array(&key, axis_id, file_list)?;
        let vector = self
            .get_vector_string_axis_id(&mut imod_map, &key, axis_id)?
            .unwrap();
        vector.push(Arc::clone(&imod_state));
        Ok(vector
            .iter()
            .rposition(|state| Arc::ptr_eq(state, &imod_state))
            .map_or(-1, |i| i as i32))
    }

    /// Java public final `newImod(String, File)`.
    pub fn new_imod_string_file(
        &self,
        key: &str,
        file: Option<&Path>,
    ) -> Result<i32, ImodManagerException> {
        let key = self.get_private_key(key);
        let mut imod_map = self.imod_map.lock().unwrap();
        if self.get_vector_string(&mut imod_map, &key)?.is_none() {
            let vector = self.new_vector_string_file(&key, file)?;
            imod_map.insert(key, vector);
            return Ok(0);
        }
        let imod_state = self.new_imod_state_string_file(&key, file)?;
        let vector = self.get_vector_string(&mut imod_map, &key)?.unwrap();
        vector.push(Arc::clone(&imod_state));
        Ok(vector
            .iter()
            .rposition(|state| Arc::ptr_eq(state, &imod_state))
            .map_or(-1, |i| i as i32))
    }

    /// Java public final `updateImod`.  Changes the file in an existing imod, if the imod
    /// exists.
    ///
    /// `ImodState.setFile` dereferences the file; a null one changes nothing here.
    pub fn update_imod(
        &self,
        key: &str,
        index: i32,
        file: Option<&Path>,
    ) -> Result<(), ImodManagerException> {
        let mut imod_map = self.imod_map.lock().unwrap();
        let imod_state = match self.get_vector_string(&mut imod_map, &self.get_private_key(key))? {
            Some(vector) if index >= 0 && vector.len() > index as usize => {
                Some(Arc::clone(&vector[index as usize]))
            }
            _ => None,
        };
        drop(imod_map);
        if let (Some(imod_state), Some(file)) = (imod_state, file) {
            imod_state.set_file(file);
        }
        Ok(())
    }

    /// Java private final `newImod(String, AxisID, File)`.
    fn new_imod_string_axis_id_file(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
        file: Option<&Path>,
    ) -> Result<i32, ImodManagerException> {
        let key = self.get_private_key(key);
        let mut imod_map = self.imod_map.lock().unwrap();
        if self
            .get_vector_string_axis_id(&mut imod_map, &key, axis_id)?
            .is_none()
        {
            let vector = self.new_vector_string_axis_id_file(&key, axis_id, file)?;
            // Correct axis: see `map_key`.
            imod_map.insert(self.map_key(&key, axis_id), vector);
            return Ok(0);
        }
        let imod_state = self.new_imod_state_string_axis_id_file(&key, axis_id, file)?;
        let vector = self
            .get_vector_string_axis_id(&mut imod_map, &key, axis_id)?
            .unwrap();
        vector.push(Arc::clone(&imod_state));
        Ok(vector
            .iter()
            .rposition(|state| Arc::ptr_eq(state, &imod_state))
            .map_or(-1, |i| i as i32))
    }

    /// Java private final `createImod`.
    fn create_imod(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
        file: Option<&Path>,
    ) -> Result<Option<Arc<ImodState>>, ImodManagerException> {
        if let Some(imod_state) = self.get_string_axis_id_file(key, axis_id, file)? {
            // imodState already exists.
            return Ok(Some(imod_state));
        }
        let mut imod_map = self.imod_map.lock().unwrap();
        if self
            .get_vector_string_axis_id(&mut imod_map, key, axis_id)?
            .is_none()
        {
            // create vector and build imod state.
            let vector = self.new_vector_string_axis_id_file(key, axis_id, file)?;
            imod_map.insert(self.map_key(key, axis_id), vector);
            drop(imod_map);
            return self.get_string_axis_id_file(key, axis_id, file);
        }
        let imod_state = self.new_imod_state_string_axis_id_file(key, axis_id, file)?;
        let vector = self
            .get_vector_string_axis_id(&mut imod_map, key, axis_id)?
            .unwrap();
        vector.push(Arc::clone(&imod_state));
        Ok(Some(imod_state))
    }

    /// Java private final `newImod(String, String[])`.
    fn new_imod_string_string_array(
        &self,
        key: &str,
        file_name_array: Option<&[String]>,
    ) -> Result<i32, ImodManagerException> {
        let key = self.get_private_key(key);
        let mut imod_map = self.imod_map.lock().unwrap();
        if self.get_vector_string(&mut imod_map, &key)?.is_none() {
            let vector = self.new_vector_string_string_array(&key, file_name_array)?;
            imod_map.insert(key, vector);
            return Ok(0);
        }
        let imod_state = self.new_imod_state_string_string_array(&key, file_name_array)?;
        let vector = self.get_vector_string(&mut imod_map, &key)?.unwrap();
        vector.push(Arc::clone(&imod_state));
        Ok(vector
            .iter()
            .rposition(|state| Arc::ptr_eq(state, &imod_state))
            .map_or(-1, |i| i as i32))
    }

    /// Java private final `newImod(String, String[], String)`.
    fn new_imod_string_string_array_string(
        &self,
        key: &str,
        file_name_array: Option<&[String]>,
        subdir_name: Option<&str>,
    ) -> Result<i32, ImodManagerException> {
        let key = self.get_private_key(key);
        let mut imod_map = self.imod_map.lock().unwrap();
        if self.get_vector_string(&mut imod_map, &key)?.is_none() {
            let vector =
                self.new_vector_string_string_array_string(&key, file_name_array, subdir_name)?;
            imod_map.insert(key, vector);
            return Ok(0);
        }
        let imod_state =
            self.new_imod_state_string_string_array_string(&key, file_name_array, subdir_name)?;
        let vector = self.get_vector_string(&mut imod_map, &key)?.unwrap();
        vector.push(Arc::clone(&imod_state));
        Ok(vector
            .iter()
            .rposition(|state| Arc::ptr_eq(state, &imod_state))
            .map_or(-1, |i| i as i32))
    }

    /// Java public final `open(String)`.
    pub fn open_string(&self, key: &str) -> Result<(), ImodManagerException> {
        self.open_string_axis_id_string_run3dmod_menu_options(key, None, None, None)
    }

    /// Java public final `open(String, Run3dmodMenuOptions)`.
    pub fn open_string_run3dmod_menu_options(
        &self,
        key: &str,
        menu_options: Option<Run3dmodMenuOptions>,
    ) -> Result<(), ImodManagerException> {
        self.open_string_axis_id_string_run3dmod_menu_options(key, None, None, menu_options)
        // used for:
        // openCombinedTomogram
    }

    /// Java public final `open(String, String, Run3dmodMenuOptions)`.
    pub fn open_string_string_run3dmod_menu_options(
        &self,
        key: &str,
        model: Option<&str>,
        menu_options: Option<Run3dmodMenuOptions>,
    ) -> Result<(), ImodManagerException> {
        self.open_string_axis_id_string_run3dmod_menu_options(key, None, model, menu_options)
        // used for:
        // openCombinedTomogram
    }

    /// Java public final `open(String, AxisID, Run3dmodMenuOptions)`.
    pub fn open_string_axis_id_run3dmod_menu_options(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
        menu_options: Option<Run3dmodMenuOptions>,
    ) -> Result<(), ImodManagerException> {
        self.open_string_axis_id_string_run3dmod_menu_options(key, axis_id, None, menu_options)
    }

    /// Java public final `open(String, AxisID, String)`.
    pub fn open_string_axis_id_string(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
        model: Option<&str>,
    ) -> Result<(), ImodManagerException> {
        self.open_string_axis_id_string_run3dmod_menu_options(
            key,
            axis_id,
            model,
            Some(Run3dmodMenuOptions::new()),
        )
    }

    /// Java public final `open(String, AxisID)`.
    pub fn open_string_axis_id(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
    ) -> Result<(), ImodManagerException> {
        self.open_string_axis_id_string_run3dmod_menu_options(
            key,
            axis_id,
            None,
            Some(Run3dmodMenuOptions::new()),
        )
    }

    /// Java public final `open(String, File, Run3dmodMenuOptions)`.
    pub fn open_string_file_run3dmod_menu_options(
        &self,
        key: &str,
        file: Option<&Path>,
        menu_options: Option<Run3dmodMenuOptions>,
    ) -> Result<(), ImodManagerException> {
        let key = self.get_private_key(key);
        let mut imod_state = self.get_string(&key)?;
        if imod_state.is_none() {
            self.new_imod_string_file(&key, file)?;
            imod_state = self.get_string(&key)?;
        }
        if let Some(imod_state) = imod_state {
            imod_state.open_run3dmod_menu_options(menu_options)?;
        }
        Ok(())
    }

    /// Java public final `open(String, AxisID, File, Run3dmodMenuOptions)`.
    pub fn open_string_axis_id_file_run3dmod_menu_options(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
        file: Option<&Path>,
        menu_options: Option<Run3dmodMenuOptions>,
    ) -> Result<(), ImodManagerException> {
        let key = self.get_private_key(key);
        let mut imod_state = self.get_string_axis_id_file(&key, axis_id, file)?;
        if imod_state.is_none() {
            self.new_imod_string_axis_id_file(&key, axis_id, file)?;
            imod_state = self.get_string_axis_id(&key, axis_id)?;
        }
        if let Some(imod_state) = imod_state {
            imod_state.open_run3dmod_menu_options(menu_options)?;
        }
        Ok(())
    }

    /// Java public final `open(String, File, Run3dmodMenuOptions, boolean)`.
    pub fn open_string_file_run3dmod_menu_options_boolean(
        &self,
        key: &str,
        file: Option<&Path>,
        menu_options: Option<Run3dmodMenuOptions>,
        swap_yz: bool,
    ) -> Result<(), ImodManagerException> {
        let key = self.get_private_key(key);
        let mut imod_state = self.get_string(&key)?;
        if imod_state.is_none() {
            self.new_imod_string_file(&key, file)?;
            imod_state = self.get_string(&key)?;
        }
        if let Some(imod_state) = imod_state {
            imod_state.set_swap_yz(swap_yz);
            imod_state.open_run3dmod_menu_options(menu_options)?;
        }
        Ok(())
    }

    /// Java public final `open(String, AxisID, String, Run3dmodMenuOptions)`.
    pub fn open_string_axis_id_string_run3dmod_menu_options(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
        model: Option<&str>,
        menu_options: Option<Run3dmodMenuOptions>,
    ) -> Result<(), ImodManagerException> {
        let key = self.get_private_key(key);
        let mut imod_state = self.get_string_axis_id(&key, axis_id)?;
        if imod_state.is_none() {
            self.new_imod_string_axis_id(&key, axis_id)?;
            imod_state = self.get_string_axis_id(&key, axis_id)?;
        }
        if let Some(imod_state) = imod_state {
            if model.is_none() {
                imod_state.open_run3dmod_menu_options(menu_options)?;
            } else {
                imod_state.open_string_run3dmod_menu_options(model, menu_options)?;
            }
        }
        Ok(())
    }

    /// Java public final `open(String, AxisID, List<String>, Run3dmodMenuOptions)`.
    pub fn open_string_axis_id_list_run3dmod_menu_options(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
        model_list: Option<Vec<String>>,
        menu_options: Option<Run3dmodMenuOptions>,
    ) -> Result<(), ImodManagerException> {
        let key = self.get_private_key(key);
        let mut imod_state = self.get_string_axis_id(&key, axis_id)?;
        if imod_state.is_none() {
            self.new_imod_string_axis_id(&key, axis_id)?;
            imod_state = self.get_string_axis_id(&key, axis_id)?;
        }
        if let Some(imod_state) = imod_state {
            if model_list.is_none() {
                imod_state.open_run3dmod_menu_options(menu_options)?;
            } else {
                imod_state.open_list_run3dmod_menu_options(model_list, menu_options)?;
            }
        }
        Ok(())
    }

    /// Java public final `setOpenModelView`.
    pub fn set_open_model_view(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
    ) -> Result<(), ImodManagerException> {
        let key = self.get_private_key(key);
        let mut imod_state = self.get_string_axis_id(&key, axis_id)?;
        if imod_state.is_none() {
            self.new_imod_string_axis_id(&key, axis_id)?;
            imod_state = self.get_string_axis_id(&key, axis_id)?;
        }
        if let Some(imod_state) = imod_state {
            imod_state.set_open_model_view()?;
        }
        Ok(())
    }

    /// Java public final `open(String, String[], Run3dmodMenuOptions)`.
    pub fn open_string_string_array_run3dmod_menu_options(
        &self,
        key: &str,
        file_name_array: Option<&[String]>,
        menu_options: Option<Run3dmodMenuOptions>,
    ) -> Result<(), ImodManagerException> {
        let key = self.get_private_key(key);
        let mut imod_state = self.get_string_axis_id(&key, Some(AxisID::Only))?;
        if imod_state
            .as_ref()
            .is_none_or(|imod_state| !imod_state.equals_file_name_array(file_name_array))
        {
            self.new_imod_string_string_array(&key, file_name_array)?;
            imod_state = self.get_string_axis_id(&key, Some(AxisID::Only))?;
        }
        if let Some(imod_state) = imod_state {
            imod_state.open_run3dmod_menu_options(menu_options)?;
        }
        Ok(())
    }

    /// Java public final `open(String, String[], Run3dmodMenuOptions, String, boolean)`.
    pub fn open_string_string_array_run3dmod_menu_options_string_boolean(
        &self,
        key: &str,
        file_name_array: Option<&[String]>,
        menu_options: Option<Run3dmodMenuOptions>,
        subdir_name: Option<&str>,
        swap_yz: bool,
    ) -> Result<(), ImodManagerException> {
        let key = self.get_private_key(key);
        let mut imod_state = self.get_string_axis_id(&key, Some(AxisID::Only))?;
        if imod_state.as_ref().is_none_or(|imod_state| {
            !imod_state.equals_subdir_name(subdir_name)
                || !imod_state.equals_file_name_array(file_name_array)
        }) {
            self.new_imod_string_string_array_string(&key, file_name_array, subdir_name)?;
            imod_state = self.get_string_axis_id(&key, Some(AxisID::Only))?;
        }
        if let Some(imod_state) = imod_state {
            imod_state.set_swap_yz(swap_yz);
            imod_state.open_run3dmod_menu_options(menu_options)?;
        }
        Ok(())
    }

    /// Java public final `open(AxisID, String, String[], Run3dmodMenuOptions, String)`.
    pub fn open_axis_id_string_string_array_run3dmod_menu_options_string(
        &self,
        axis_id: Option<AxisID>,
        key: &str,
        file_name_array: Option<&[String]>,
        menu_options: Option<Run3dmodMenuOptions>,
        subdir_name: Option<&str>,
    ) -> Result<(), ImodManagerException> {
        let _ = axis_id;
        let key = self.get_private_key(key);
        let mut imod_state = self.get_string_axis_id(&key, Some(AxisID::Only))?;
        if imod_state.as_ref().is_none_or(|imod_state| {
            !imod_state.equals_subdir_name(subdir_name)
                || !imod_state.equals_file_name_array(file_name_array)
        }) {
            self.new_imod_string_string_array_string(&key, file_name_array, subdir_name)?;
            imod_state = self.get_string_axis_id(&key, Some(AxisID::Only))?;
        }
        if let Some(imod_state) = imod_state {
            imod_state.open_run3dmod_menu_options(menu_options)?;
        }
        Ok(())
    }

    /// Java public final `open(String, AxisID, String, boolean, Run3dmodMenuOptions)`.
    pub fn open_string_axis_id_string_boolean_run3dmod_menu_options(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
        model: Option<&str>,
        model_mode: bool,
        menu_options: Option<Run3dmodMenuOptions>,
    ) -> Result<(), ImodManagerException> {
        let key = self.get_private_key(key);
        let mut imod_state = self.get_string_axis_id(&key, axis_id)?;
        if imod_state.is_none() {
            self.new_imod_string_axis_id(&key, axis_id)?;
            imod_state = self.get_string_axis_id(&key, axis_id)?;
        }
        if let Some(imod_state) = imod_state {
            imod_state.open_string_boolean_run3dmod_menu_options(
                model,
                model_mode,
                menu_options,
            )?;
        }
        // rawStack.model(modelName, modelMode);
        Ok(())
    }

    /// Java public final `open(String, AxisID, FileType, Run3dmodMenuOptions)`.
    pub fn open_string_axis_id_file_type_run3dmod_menu_options(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
        model: &FileType,
        menu_options: Option<Run3dmodMenuOptions>,
    ) -> Result<(), ImodManagerException> {
        let key = self.get_private_key(key);
        let mut imod_state = self.get_string_axis_id(&key, axis_id)?;
        if imod_state.is_none() {
            self.new_imod_string_axis_id(&key, axis_id)?;
            imod_state = self.get_string_axis_id(&key, axis_id)?;
        }
        if let Some(imod_state) = imod_state {
            imod_state.open_file_type_run3dmod_menu_options(model, menu_options)?;
        }
        // rawStack.model(modelName, modelMode);
        Ok(())
    }

    /// Java public final `open(String, AxisID, File, String, boolean,
    /// Run3dmodMenuOptions)`.
    pub fn open_string_axis_id_file_string_boolean_run3dmod_menu_options(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
        file: Option<&Path>,
        model: Option<&str>,
        model_mode: bool,
        menu_options: Option<Run3dmodMenuOptions>,
    ) -> Result<(), ImodManagerException> {
        let key = self.get_private_key(key);
        let mut imod_state = self.get_string_axis_id_file(&key, axis_id, file)?;
        if imod_state.is_none() {
            self.new_imod_string_axis_id_file(&key, axis_id, file)?;
            imod_state = self.get_string_axis_id_file(&key, axis_id, file)?;
        }
        if let Some(imod_state) = imod_state {
            imod_state.open_string_boolean_run3dmod_menu_options(
                model,
                model_mode,
                menu_options,
            )?;
        }
        Ok(())
    }

    /// Java public final `open(String, File, String, boolean, Run3dmodMenuOptions)`.
    pub fn open_string_file_string_boolean_run3dmod_menu_options(
        &self,
        key: &str,
        file: Option<&Path>,
        model: Option<&str>,
        model_mode: bool,
        menu_options: Option<Run3dmodMenuOptions>,
    ) -> Result<(), ImodManagerException> {
        let key = self.get_private_key(key);
        let mut imod_state = self.get_string(&key)?;
        if imod_state.is_none() {
            self.new_imod_string_file(&key, file)?;
            imod_state = self.get_string(&key)?;
        }
        if let Some(imod_state) = imod_state {
            imod_state.open_string_boolean_run3dmod_menu_options(
                model,
                model_mode,
                menu_options,
            )?;
        }
        Ok(())
    }

    /// Java public final `open(String, File, AxisID, int, String, boolean,
    /// Run3dmodMenuOptions)`.
    #[allow(clippy::too_many_arguments)]
    pub fn open_string_file_axis_id_int_string_boolean_run3dmod_menu_options(
        &self,
        key: &str,
        file: Option<&Path>,
        axis_id: Option<AxisID>,
        mut vector_index: i32,
        model: Option<&str>,
        model_mode: bool,
        menu_options: Option<Run3dmodMenuOptions>,
    ) -> Result<i32, ImodManagerException> {
        let key = self.get_private_key(key);
        let mut imod_state = None;
        if vector_index != -1 {
            imod_state = self.get_string_axis_id_int(&key, axis_id, vector_index)?;
        }
        if imod_state.is_none() {
            vector_index = self.new_imod_string_axis_id_file(&key, axis_id, file)?;
            imod_state = self.get_string_axis_id_int(&key, axis_id, vector_index)?;
        }
        if let Some(imod_state) = imod_state {
            imod_state.open_string_boolean_run3dmod_menu_options(
                model,
                model_mode,
                menu_options,
            )?;
        }
        Ok(vector_index)
    }

    /// Java public final `setFile(String, File, AxisID, int)`.
    pub fn set_file_string_file_axis_id_int(
        &self,
        key: &str,
        file: Option<&Path>,
        axis_id: Option<AxisID>,
        mut vector_index: i32,
    ) -> Result<i32, ImodManagerException> {
        let key = self.get_private_key(key);
        let mut imod_state = None;
        if vector_index != -1 {
            imod_state = self.get_string_axis_id_int(&key, axis_id, vector_index)?;
        }
        if imod_state.is_none() {
            vector_index = self.new_imod_string_axis_id_file(&key, axis_id, file)?;
            imod_state = self.get_string_axis_id_int(&key, axis_id, vector_index)?;
        }
        // `ImodState.setFile` dereferences the file; a null one sets nothing.
        if let (Some(imod_state), Some(file)) = (imod_state, file) {
            imod_state.set_file(file);
        }
        Ok(vector_index)
    }

    /// Java public final `open(String, File, AxisID, int, File, boolean,
    /// Run3dmodMenuOptions)`.
    #[allow(clippy::too_many_arguments)]
    pub fn open_string_file_axis_id_int_file_boolean_run3dmod_menu_options(
        &self,
        key: &str,
        file: Option<&Path>,
        axis_id: Option<AxisID>,
        mut vector_index: i32,
        model_file: Option<&Path>,
        model_mode: bool,
        menu_options: Option<Run3dmodMenuOptions>,
    ) -> Result<i32, ImodManagerException> {
        let key = self.get_private_key(key);
        let mut imod_state = None;
        if vector_index != -1 {
            imod_state = self.get_string_axis_id_int(&key, axis_id, vector_index)?;
        }
        if imod_state.is_none() {
            vector_index = self.new_imod_string_axis_id_file(&key, axis_id, file)?;
            imod_state = self.get_string_axis_id_int(&key, axis_id, vector_index)?;
        }
        if let Some(imod_state) = imod_state {
            if let Some(model_file) = model_file {
                imod_state.open_string_boolean_run3dmod_menu_options(
                    Some(&utilities::java_io_file_get_absolute_path(
                        &model_file.to_string_lossy(),
                    )),
                    model_mode,
                    menu_options,
                )?;
            } else {
                imod_state.open_run3dmod_menu_options(menu_options)?;
            }
        }
        Ok(vector_index)
    }

    /// Java public final `open(String, File, int, Run3dmodMenuOptions)`.
    pub fn open_string_file_int_run3dmod_menu_options(
        &self,
        key: &str,
        file: Option<&Path>,
        mut vector_index: i32,
        menu_options: Option<Run3dmodMenuOptions>,
    ) -> Result<i32, ImodManagerException> {
        let key = self.get_private_key(key);
        let mut imod_state = None;
        if vector_index != -1 {
            imod_state = self.get_string_int(&key, vector_index)?;
        }
        if imod_state.is_none() {
            vector_index = self.new_imod_string_file(&key, file)?;
            imod_state = self.get_string_int(&key, vector_index)?;
        }
        if let Some(imod_state) = imod_state {
            imod_state.open_run3dmod_menu_options(menu_options)?;
        }
        Ok(vector_index)
    }

    /// Java public final `openModel`.
    pub fn open_model(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
        vector_index: i32,
        model: &str,
        model_mode: bool,
    ) -> Result<(), ImodManagerException> {
        let key = self.get_private_key(key);
        let mut imod_state = None;
        if vector_index != -1 {
            imod_state = self.get_string_axis_id_int(&key, axis_id, vector_index)?;
        }
        if let Some(imod_state) = imod_state {
            imod_state.open_model(model, model_mode)?;
        }
        Ok(())
    }

    /// Java public final `open(String, File, AxisID, int, Run3dmodMenuOptions)`.
    pub fn open_string_file_axis_id_int_run3dmod_menu_options(
        &self,
        key: &str,
        file: Option<&Path>,
        axis_id: Option<AxisID>,
        mut vector_index: i32,
        menu_options: Option<Run3dmodMenuOptions>,
    ) -> Result<i32, ImodManagerException> {
        let key = self.get_private_key(key);
        let mut imod_state = None;
        if vector_index != -1 {
            imod_state = self.get_string_axis_id_int(&key, axis_id, vector_index)?;
        }
        if imod_state.is_none() {
            vector_index = self.new_imod_string_axis_id_file(&key, axis_id, file)?;
            imod_state = self.get_string_axis_id_int(&key, axis_id, vector_index)?;
        }
        if let Some(imod_state) = imod_state {
            imod_state.open_run3dmod_menu_options(menu_options)?;
        }
        Ok(vector_index)
    }

    /// Java public final `open(String, AxisID, int, Run3dmodMenuOptions)`.
    pub fn open_string_axis_id_int_run3dmod_menu_options(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
        vector_index: i32,
        menu_options: Option<Run3dmodMenuOptions>,
    ) -> Result<(), ImodManagerException> {
        let key = self.get_private_key(key);
        let Some(imod_state) = self.get_string_axis_id_int(&key, axis_id, vector_index)? else {
            // The source dereferences a null axisID here; "null" is printed instead.
            return Err(ImodManagerException::Runtime(format!(
                "{} was not created in {} with axisID={} at index {}",
                key,
                self.axis_type_to_string(),
                axis_id.map_or_else(|| "null".to_string(), |axis_id| axis_id.get_extension()),
                vector_index
            )));
        };
        imod_state.open_run3dmod_menu_options(menu_options)
    }

    /// Java public final `open(String, int, Run3dmodMenuOptions)`.
    pub fn open_string_int_run3dmod_menu_options(
        &self,
        key: &str,
        vector_index: i32,
        menu_options: Option<Run3dmodMenuOptions>,
    ) -> Result<(), ImodManagerException> {
        let key = self.get_private_key(key);
        let Some(imod_state) = self.get_string_int(&key, vector_index)? else {
            return Err(ImodManagerException::Runtime(format!(
                "{} was not created in {} at index {}",
                key,
                self.axis_type_to_string(),
                vector_index
            )));
        };
        imod_state.open_run3dmod_menu_options(menu_options)
    }

    /// Java public final `open(String, int, String, boolean, Run3dmodMenuOptions)`.
    pub fn open_string_int_string_boolean_run3dmod_menu_options(
        &self,
        key: &str,
        vector_index: i32,
        model: Option<&str>,
        model_mode: bool,
        menu_options: Option<Run3dmodMenuOptions>,
    ) -> Result<(), ImodManagerException> {
        let key = self.get_private_key(key);
        let Some(imod_state) = self.get_string_int(&key, vector_index)? else {
            return Err(ImodManagerException::Runtime(format!(
                "{} was not created in {} at index {}",
                key,
                self.axis_type_to_string(),
                vector_index
            )));
        };
        imod_state.open_string_boolean_run3dmod_menu_options(model, model_mode, menu_options)
    }

    /// Java public final `delete`.
    pub fn delete(&self, key: &str, vector_index: i32) -> Result<(), ImodManagerException> {
        let key = self.get_private_key(key);
        let Some(imod_state) = self.get_string_int(&key, vector_index)? else {
            return Err(ImodManagerException::Runtime(format!(
                "{} was not created in {} at index {}",
                key,
                self.axis_type_to_string(),
                vector_index
            )));
        };
        imod_state.quit()?;
        self.delete_imod_state(&key, vector_index)
    }

    /// Java public final `isOpen(String)`.
    pub fn is_open_string(&self, key: &str) -> Result<bool, ImodManagerException> {
        self.is_open_string_axis_id(key, None)
    }

    /// Java public final `isOpen(String, AxisID)`.
    pub fn is_open_string_axis_id(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
    ) -> Result<bool, ImodManagerException> {
        let key = self.get_private_key(key);
        let Some(imod_state) = self.get_string_axis_id(&key, axis_id)? else {
            return Ok(false);
        };
        Ok(imod_state.is_open())
    }

    /// Java public final `isOpen(String, AxisID, String)`.
    pub fn is_open_string_axis_id_string(
        &self,
        key: Option<&str>,
        axis_id: Option<AxisID>,
        dataset_name: &str,
    ) -> Result<bool, ImodManagerException> {
        let Some(key) = key else {
            return Ok(false);
        };
        let key = self.get_private_key(key);
        let Some(imod_state) = self.get_string_axis_id_string(&key, axis_id, dataset_name)? else {
            return Ok(false);
        };
        Ok(imod_state.is_open())
    }

    /// Java public final `isOpen(String, AxisID, File)`.
    pub fn is_open_string_axis_id_file(
        &self,
        key: Option<&str>,
        axis_id: Option<AxisID>,
        file: Option<&Path>,
    ) -> Result<bool, ImodManagerException> {
        let Some(key) = key else {
            return Ok(false);
        };
        let key = self.get_private_key(key);
        let Some(imod_state) = self.get_string_axis_id_file(&key, axis_id, file)? else {
            return Ok(false);
        };
        Ok(imod_state.is_open())
    }

    /// Java public final `isOpen()`.
    pub fn is_open(&self) -> Result<bool, ImodManagerException> {
        let states: Vec<Arc<ImodState>> = {
            let mut imod_map = self.imod_map.lock().unwrap();
            if imod_map.is_empty() {
                return Ok(false);
            }
            let mut states = Vec::new();
            let keys: Vec<String> = imod_map.keys().cloned().collect();
            for key in keys {
                if let Some(vector) = self.get_vector_string_boolean(&mut imod_map, &key, true)? {
                    states.extend(vector.iter().cloned());
                }
            }
            states
        };
        for imod_state in states {
            if imod_state.is_open() {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Java public final `getModelName`.
    pub fn get_model_name(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
    ) -> Result<Option<String>, ImodManagerException> {
        let key = self.get_private_key(key);
        let Some(imod_state) = self.get_string_axis_id(&key, axis_id)? else {
            return Ok(Some(String::new()));
        };
        Ok(imod_state.get_model_name())
    }

    /// Java public final `getRubberbandCoordinates`.
    pub fn get_rubberband_coordinates(
        &self,
        key: &str,
    ) -> Result<Option<Vec<String>>, ImodManagerException> {
        let key = self.get_private_key(key);
        let Some(imod_state) = self.get_string(&key)? else {
            ui_harness::post_message_dialog(
                Some(self.manager),
                "3dmod is not running.".to_string(),
                "3dmod Warning".to_string(),
                Some(AxisID::Only),
            );
            return Ok(None);
        };
        imod_state.get_rubberband_coordinates()
    }

    /// Java public final `getSlicerAngles`.
    pub fn get_slicer_angles(
        &self,
        key: &str,
        vector_index: i32,
    ) -> Result<Option<Vec<String>>, ImodManagerException> {
        let key = self.get_private_key(key);
        let imod_state = self.get_string_int(&key, vector_index)?;
        let Some(imod_state) = imod_state.filter(|imod_state| imod_state.is_open()) else {
            ui_harness::post_message_dialog(
                Some(self.manager),
                "3dmod is not running.".to_string(),
                "3dmod Warning".to_string(),
                Some(AxisID::Only),
            );
            return Ok(None);
        };
        imod_state.get_slicer_angles()
    }

    /// Java public final `quit(String)`.
    pub fn quit_string(&self, key: &str) -> Result<(), ImodManagerException> {
        self.quit_string_axis_id(key, None)
    }

    /// Java public final `quit(String, AxisID)`.
    pub fn quit_string_axis_id(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
    ) -> Result<(), ImodManagerException> {
        let key = self.get_private_key(key);
        if let Some(imod_state) = self.get_string_axis_id(&key, axis_id)? {
            imod_state.quit()?;
        }
        Ok(())
    }

    /// Java public final `quit(String, AxisID, String)`.
    pub fn quit_string_axis_id_string(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
        dataset_name: &str,
    ) -> Result<(), ImodManagerException> {
        let key = self.get_private_key(key);
        if let Some(imod_state) = self.get_string_axis_id_string(&key, axis_id, dataset_name)? {
            imod_state.quit()?;
        }
        Ok(())
    }

    /// Java public final `quit(String, AxisID, File)`.
    pub fn quit_string_axis_id_file(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
        file: Option<&Path>,
    ) -> Result<(), ImodManagerException> {
        let key = self.get_private_key(key);
        if let Some(imod_state) = self.get_string_axis_id_file(&key, axis_id, file)? {
            imod_state.quit()?;
        }
        Ok(())
    }

    /// Java public final `quitAll`.
    pub fn quit_all(&self, key: &str, axis_id: Option<AxisID>) -> Result<(), ImodManagerException> {
        let imod_state_vector: ImodVector = {
            let mut imod_map = self.imod_map.lock().unwrap();
            match self.get_vector_string_axis_id(
                &mut imod_map,
                &self.get_private_key(key),
                axis_id,
            )? {
                None => return Ok(()),
                Some(vector) => vector.clone(),
            }
        };
        if imod_state_vector.is_empty() {
            return Ok(());
        }
        for imod_state in imod_state_vector {
            if imod_state.is_open() {
                imod_state.quit()?;
                std::thread::sleep(Duration::from_millis(500));
            }
        }
        Ok(())
    }

    /// Java public final `quit()`.
    pub fn quit(&self) -> Result<(), ImodManagerException> {
        let states: Vec<Arc<ImodState>> = {
            let mut imod_map = self.imod_map.lock().unwrap();
            if imod_map.is_empty() {
                return Ok(());
            }
            let mut states = Vec::new();
            let keys: Vec<String> = imod_map.keys().cloned().collect();
            for key in keys {
                if let Some(vector) = self.get_vector_string_boolean(&mut imod_map, &key, true)? {
                    states.extend(vector.iter().cloned());
                }
            }
            states
        };
        for imod_state in states {
            imod_state.quit()?;
        }
        Ok(())
    }

    /// Java final `processRequest`.
    pub fn process_request(&self) -> Result<(), ImodManagerException> {
        let states: Vec<Arc<ImodState>> = {
            let mut imod_map = self.imod_map.lock().unwrap();
            if imod_map.is_empty() {
                return Ok(());
            }
            let mut states = Vec::new();
            let keys: Vec<String> = imod_map.keys().cloned().collect();
            for key in keys {
                if let Some(vector) = self.get_vector_string_boolean(&mut imod_map, &key, true)? {
                    states.extend(vector.iter().cloned());
                }
            }
            states
        };
        for imod_state in states {
            imod_state.process_request();
        }
        Ok(())
    }

    /// Java public final `disconnect`.
    pub fn disconnect(&self) {
        let states: Vec<Arc<ImodState>> = {
            let mut imod_map = self.imod_map.lock().unwrap();
            if imod_map.is_empty() {
                return;
            }
            let mut states = Vec::new();
            let keys: Vec<String> = imod_map.keys().cloned().collect();
            for key in keys {
                if let Some(vector) = self
                    .get_vector_string_boolean(&mut imod_map, &key, true)
                    .unwrap_or(None)
                {
                    states.extend(vector.iter().cloned());
                }
            }
            states
        };
        for imod_state in states {
            if imod_state.is_open()
                && let Err(e) = imod_state.disconnect()
            {
                eprintln!("{e}");
            }
        }
    }

    /// Java public final `setSwapYZ(String, AxisID, boolean)`.
    ///
    /// The source calls `setSwapYZ` on the state without a null test; a state that
    /// could not be found (see the module comment) threw.  Fixed in translation: no
    /// state, no change.  The same holds for `setFile(String, AxisID, File)`,
    /// `setSwapYZ(String, File, boolean)` and `setContinuousListenerTarget`.
    pub fn set_swap_yz_string_axis_id_boolean(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
        swap_yz: bool,
    ) -> Result<(), ImodManagerException> {
        let key = self.get_private_key(key);
        let mut imod_state = self.get_string_axis_id(&key, axis_id)?;
        if imod_state.is_none() {
            self.new_imod_string_axis_id(&key, axis_id)?;
            imod_state = self.get_string_axis_id(&key, axis_id)?;
        }
        if let Some(imod_state) = imod_state {
            imod_state.set_swap_yz(swap_yz);
        }
        Ok(())
    }

    /// Java public final `setSwapYZ(String, AxisID, File, boolean)`.
    pub fn set_swap_yz_string_axis_id_file_boolean(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
        file: Option<&Path>,
        swap_yz: bool,
    ) -> Result<(), ImodManagerException> {
        let key = self.get_private_key(key);
        let mut imod_state = self.get_string_axis_id_file(&key, axis_id, file)?;
        if imod_state.is_none() {
            imod_state = self.create_imod(&key, axis_id, file)?;
        }
        if let Some(imod_state) = imod_state {
            imod_state.set_swap_yz(swap_yz);
        } else {
            eprintln!(
                "Error: Unable to create ImodState for {}, {}, and {}",
                key,
                axis_id.map_or_else(|| "null".to_string(), |axis_id| axis_id.to_string()),
                file.map_or_else(|| "null".to_string(), |file| file.display().to_string())
            );
        }
        Ok(())
    }

    /// Java public final `setFile(String, AxisID, File)`.
    pub fn set_file_string_axis_id_file(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
        file: Option<&Path>,
    ) -> Result<(), ImodManagerException> {
        let key = self.get_private_key(key);
        let mut imod_state = self.get_string_axis_id(&key, axis_id)?;
        if imod_state.is_none() {
            self.new_imod_string_axis_id(&key, axis_id)?;
            imod_state = self.get_string_axis_id(&key, axis_id)?;
        }
        if let (Some(imod_state), Some(file)) = (imod_state, file) {
            imod_state.set_file(file);
        }
        Ok(())
    }

    /// Java public final `setSwapYZ(String, File, boolean)`.
    pub fn set_swap_yz_string_file_boolean(
        &self,
        key: &str,
        file: Option<&Path>,
        swap_yz: bool,
    ) -> Result<(), ImodManagerException> {
        let key = self.get_private_key(key);
        let mut imod_state = self.get_string(&key)?;
        if imod_state.is_none() {
            self.new_imod_string_file(&key, file)?;
            imod_state = self.get_string(&key)?;
        }
        if let Some(imod_state) = imod_state {
            imod_state.set_swap_yz(swap_yz);
        }
        Ok(())
    }

    /// Java public final `setOpenBeadFixer`.
    pub fn set_open_bead_fixer(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
        open_bead_fixer: bool,
    ) -> Result<(), ImodManagerException> {
        let key = self.get_private_key(key);
        let mut imod_state = self.get_string_axis_id(&key, axis_id)?;
        if imod_state.is_none() {
            self.new_imod_string_axis_id(&key, axis_id)?;
            imod_state = self.get_string_axis_id(&key, axis_id)?;
        }
        if let Some(imod_state) = imod_state {
            if imod_state.is_use_modv() {
                return Err(ImodManagerException::Runtime(
                    "The Bead Fixer cannot be opened in 3dmodv".to_string(),
                ));
            }
            imod_state.set_open_bead_fixer(open_bead_fixer);
        }
        Ok(())
    }

    /// Java public final `setOpenSurfContPoint`.
    pub fn set_open_surf_cont_point(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
        open: bool,
    ) -> Result<(), ImodManagerException> {
        let key = self.get_private_key(key);
        let mut imod_state = self.get_string_axis_id(&key, axis_id)?;
        if imod_state.is_none() {
            self.new_imod_string_axis_id(&key, axis_id)?;
            imod_state = self.get_string_axis_id(&key, axis_id)?;
        }
        if let Some(imod_state) = imod_state {
            imod_state.set_open_surf_cont_point(open);
        }
        Ok(())
    }

    /// Java public final `setAutoCenter`.  The source does not translate the key to its
    /// private key here, nor in the other beadfixer setters below.
    pub fn set_auto_center(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
        auto_center: bool,
    ) -> Result<(), ImodManagerException> {
        let mut imod_state = self.get_string_axis_id(key, axis_id)?;
        if imod_state.is_none() {
            self.new_imod_string_axis_id(key, axis_id)?;
            imod_state = self.get_string_axis_id(key, axis_id)?;
        }
        if let Some(imod_state) = imod_state {
            imod_state.set_auto_center(auto_center);
        }
        Ok(())
    }

    /// Java public final `setSkipList`.
    pub fn set_skip_list(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
        skip_list: Option<&str>,
    ) -> Result<(), ImodManagerException> {
        let mut imod_state = self.get_string_axis_id(key, axis_id)?;
        if imod_state.is_none() {
            self.new_imod_string_axis_id(key, axis_id)?;
            imod_state = self.get_string_axis_id(key, axis_id)?;
        }
        if let Some(imod_state) = imod_state {
            imod_state.set_skip_list(skip_list);
        }
        Ok(())
    }

    /// Java public final `setDeleteAllSections`.
    pub fn set_delete_all_sections(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
        on: bool,
    ) -> Result<(), ImodManagerException> {
        let mut imod_state = self.get_string_axis_id(key, axis_id)?;
        if imod_state.is_none() {
            self.new_imod_string_axis_id(key, axis_id)?;
            imod_state = self.get_string_axis_id(key, axis_id)?;
        }
        if let Some(imod_state) = imod_state {
            imod_state.set_delete_all_sections(on);
        }
        Ok(())
    }

    /// Java public final `setBeadfixerMode`.
    pub fn set_beadfixer_mode(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
        mode: Option<BeadFixerMode>,
    ) -> Result<(), ImodManagerException> {
        let mut imod_state = self.get_string_axis_id(key, axis_id)?;
        if imod_state.is_none() {
            self.new_imod_string_axis_id(key, axis_id)?;
            imod_state = self.get_string_axis_id(key, axis_id)?;
        }
        if let Some(imod_state) = imod_state {
            imod_state.set_beadfixer_mode(mode);
        }
        Ok(())
    }

    /// Java public final `setOpenLog`.
    pub fn set_open_log(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
        open_log: bool,
        log_name: Option<&str>,
    ) -> Result<(), ImodManagerException> {
        let mut imod_state = self.get_string_axis_id(key, axis_id)?;
        if imod_state.is_none() {
            self.new_imod_string_axis_id(key, axis_id)?;
            imod_state = self.get_string_axis_id(key, axis_id)?;
        }
        if let Some(imod_state) = imod_state {
            imod_state.set_open_log(open_log, log_name);
        }
        Ok(())
    }

    /// Java public final `reopenLog`.
    pub fn reopen_log(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
    ) -> Result<(), ImodManagerException> {
        let imod_state = self.get_string_axis_id(key, axis_id)?;
        let Some(imod_state) = imod_state.filter(|imod_state| imod_state.is_open()) else {
            return Ok(());
        };
        imod_state.reopen_log()
    }

    /// Java public final `setOpenLogOff`.
    pub fn set_open_log_off(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
    ) -> Result<(), ImodManagerException> {
        let mut imod_state = self.get_string_axis_id(key, axis_id)?;
        if imod_state.is_none() {
            self.new_imod_string_axis_id(key, axis_id)?;
            imod_state = self.get_string_axis_id(key, axis_id)?;
        }
        if let Some(imod_state) = imod_state {
            imod_state.set_open_log_off();
        }
        Ok(())
    }

    /// Java public final `setNewContours`.
    pub fn set_new_contours(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
        new_contours: bool,
    ) -> Result<(), ImodManagerException> {
        let mut imod_state = self.get_string_axis_id(key, axis_id)?;
        if imod_state.is_none() {
            self.new_imod_string_axis_id(key, axis_id)?;
            imod_state = self.get_string_axis_id(key, axis_id)?;
        }
        if let Some(imod_state) = imod_state {
            imod_state.set_new_contours(new_contours);
        }
        Ok(())
    }

    /// Java public final `setBinning(String, AxisID, int)`.
    pub fn set_binning_string_axis_id_int(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
        binning: i32,
    ) -> Result<(), ImodManagerException> {
        let key = self.get_private_key(key);
        let mut imod_state = self.get_string_axis_id(&key, axis_id)?;
        if imod_state.is_none() {
            self.new_imod_string_axis_id(&key, axis_id)?;
            imod_state = self.get_string_axis_id(&key, axis_id)?;
        }
        if let Some(imod_state) = imod_state {
            imod_state.set_binning(binning);
        }
        Ok(())
    }

    /// Java public final `setTiltFile`.
    pub fn set_tilt_file(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
        tilt_file: Option<&str>,
    ) -> Result<(), ImodManagerException> {
        let key = self.get_private_key(key);
        let mut imod_state = self.get_string_axis_id(&key, axis_id)?;
        if imod_state.is_none() {
            self.new_imod_string_axis_id(&key, axis_id)?;
            imod_state = self.get_string_axis_id(&key, axis_id)?;
        }
        if let Some(imod_state) = imod_state {
            imod_state.set_tilt_file(tilt_file);
        }
        Ok(())
    }

    /// Java public final `resetTiltFile`.
    pub fn reset_tilt_file(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
    ) -> Result<(), ImodManagerException> {
        let key = self.get_private_key(key);
        let mut imod_state = self.get_string_axis_id(&key, axis_id)?;
        if imod_state.is_none() {
            self.new_imod_string_axis_id(&key, axis_id)?;
            imod_state = self.get_string_axis_id(&key, axis_id)?;
        }
        if let Some(imod_state) = imod_state {
            imod_state.reset_tilt_file();
        }
        Ok(())
    }

    /// Java public final `setBinning(String, int)`.
    pub fn set_binning_string_int(
        &self,
        key: &str,
        binning: i32,
    ) -> Result<(), ImodManagerException> {
        let key = self.get_private_key(key);
        let mut imod_state = self.get_string(&key)?;
        if imod_state.is_none() {
            self.new_imod_string(&key)?;
            imod_state = self.get_string(&key)?;
        }
        if let Some(imod_state) = imod_state {
            imod_state.set_binning(binning);
        }
        Ok(())
    }

    /// Java public final `setBinning(String, int, int)`.
    pub fn set_binning_string_int_int(
        &self,
        key: &str,
        vector_index: i32,
        binning: i32,
    ) -> Result<(), ImodManagerException> {
        let key = self.get_private_key(key);
        let Some(imod_state) = self.get_string_int(&key, vector_index)? else {
            return Err(ImodManagerException::Runtime(format!(
                "{} was not created in {} at index {}",
                key,
                self.axis_type_to_string(),
                vector_index
            )));
        };
        imod_state.set_binning(binning);
        Ok(())
    }

    /// Java public final `setBinningXY(String, int)`.
    pub fn set_binning_xy_string_int(
        &self,
        key: &str,
        binning: i32,
    ) -> Result<(), ImodManagerException> {
        let key = self.get_private_key(key);
        let mut imod_state = self.get_string(&key)?;
        if imod_state.is_none() {
            self.new_imod_string(&key)?;
            imod_state = self.get_string(&key)?;
        }
        if let Some(imod_state) = imod_state {
            imod_state.set_binning_xy(binning);
        }
        Ok(())
    }

    /// Java public final `setContinuousListenerTarget`.
    pub fn set_continuous_listener_target(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
        continuous_listener_target: Option<Arc<dyn ContinuousListenerTarget>>,
    ) -> Result<(), ImodManagerException> {
        let key = self.get_private_key(key);
        let mut imod_state = self.get_string_axis_id(&key, axis_id)?;
        if imod_state.is_none() {
            self.new_imod_string_axis_id(&key, axis_id)?;
            imod_state = self.get_string_axis_id(&key, axis_id)?;
        }
        if let Some(imod_state) = imod_state {
            imod_state.set_continuous_listener_target(continuous_listener_target);
        }
        Ok(())
    }

    /// Java public final `setBinningXY(String, int, int)`.
    pub fn set_binning_xy_string_int_int(
        &self,
        key: &str,
        vector_index: i32,
        binning: i32,
    ) -> Result<(), ImodManagerException> {
        let key = self.get_private_key(key);
        let Some(imod_state) = self.get_string_int(&key, vector_index)? else {
            return Err(ImodManagerException::Runtime(format!(
                "{} was not created in {} at index {}",
                key,
                self.axis_type_to_string(),
                vector_index
            )));
        };
        imod_state.set_binning_xy(binning);
        Ok(())
    }

    /// Java public final `setOpenContours`.
    pub fn set_open_contours(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
        open_contours: bool,
    ) -> Result<(), ImodManagerException> {
        let key = self.get_private_key(key);
        let mut imod_state = self.get_string_axis_id(&key, axis_id)?;
        if imod_state.is_none() {
            self.new_imod_string_axis_id(&key, axis_id)?;
            imod_state = self.get_string_axis_id(&key, axis_id)?;
        }
        if let Some(imod_state) = imod_state {
            imod_state.set_open_contours(open_contours);
        }
        Ok(())
    }

    /// Java public final `setStartNewContoursAtNewZ`.
    pub fn set_start_new_contours_at_new_z(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
        start_new_contours_at_new_z: bool,
    ) -> Result<(), ImodManagerException> {
        let key = self.get_private_key(key);
        let mut imod_state = self.get_string_axis_id(&key, axis_id)?;
        if imod_state.is_none() {
            self.new_imod_string_axis_id(&key, axis_id)?;
            imod_state = self.get_string_axis_id(&key, axis_id)?;
        }
        if let Some(imod_state) = imod_state {
            imod_state.set_start_new_contours_at_new_z(start_new_contours_at_new_z);
        }
        Ok(())
    }

    /// Java public final `setPointLimit`.
    pub fn set_point_limit(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
        point_limit: i32,
    ) -> Result<(), ImodManagerException> {
        let key = self.get_private_key(key);
        let mut imod_state = self.get_string_axis_id(&key, axis_id)?;
        if imod_state.is_none() {
            self.new_imod_string_axis_id(&key, axis_id)?;
            imod_state = self.get_string_axis_id(&key, axis_id)?;
        }
        if let Some(imod_state) = imod_state {
            imod_state.set_point_limit(point_limit);
        }
        Ok(())
    }

    /// Java public final `setPreserveContrast`.  Use this when opening a model newly
    /// created by software that doesn't have previous contrast settings in it.
    pub fn set_preserve_contrast(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
        preserve_contrast: bool,
    ) -> Result<(), ImodManagerException> {
        let key = self.get_private_key(key);
        let mut imod_state = self.get_string_axis_id(&key, axis_id)?;
        if imod_state.is_none() {
            self.new_imod_string_axis_id(&key, axis_id)?;
            imod_state = self.get_string_axis_id(&key, axis_id)?;
        }
        if let Some(imod_state) = imod_state {
            imod_state.set_preserve_contrast(preserve_contrast);
        }
        Ok(())
    }

    /// Java public final `setFrames`.
    pub fn set_frames(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
        frames: bool,
    ) -> Result<(), ImodManagerException> {
        let key = self.get_private_key(key);
        let mut imod_state = self.get_string_axis_id(&key, axis_id)?;
        if imod_state.is_none() {
            self.new_imod_string_axis_id(&key, axis_id)?;
            imod_state = self.get_string_axis_id(&key, axis_id)?;
        }
        if let Some(imod_state) = imod_state {
            imod_state.set_frames(frames);
        }
        Ok(())
    }

    /// Java public final `setPieceListFileName(String, AxisID, String)`.
    pub fn set_piece_list_file_name_string_axis_id_string(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
        piece_list_file_name: Option<&str>,
    ) -> Result<(), ImodManagerException> {
        let key = self.get_private_key(key);
        let mut imod_state = self.get_string_axis_id(&key, axis_id)?;
        if imod_state.is_none() {
            self.new_imod_string_axis_id(&key, axis_id)?;
            imod_state = self.get_string_axis_id(&key, axis_id)?;
        }
        if let Some(imod_state) = imod_state {
            imod_state.set_piece_list_file_name(piece_list_file_name);
        }
        Ok(())
    }

    /// Java public final `setMontageSeparation`.
    pub fn set_montage_separation(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
    ) -> Result<(), ImodManagerException> {
        let key = self.get_private_key(key);
        let mut imod_state = self.get_string_axis_id(&key, axis_id)?;
        if imod_state.is_none() {
            self.new_imod_string_axis_id(&key, axis_id)?;
            imod_state = self.get_string_axis_id(&key, axis_id)?;
        }
        if let Some(imod_state) = imod_state {
            imod_state.set_montage_separation();
        }
        Ok(())
    }

    /// Java public final `setInterpolation`.
    pub fn set_interpolation(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
        interpolation: bool,
    ) -> Result<(), ImodManagerException> {
        let key = self.get_private_key(key);
        let mut imod_state = self.get_string_axis_id(&key, axis_id)?;
        if imod_state.is_none() {
            self.new_imod_string_axis_id(&key, axis_id)?;
            imod_state = self.get_string_axis_id(&key, axis_id)?;
        }
        if let Some(imod_state) = imod_state {
            imod_state.set_interpolation(interpolation);
        }
        Ok(())
    }

    /// Java public final `setWorkingDirectory`.
    pub fn set_working_directory(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
        vector_index: i32,
        working_directory: Option<PathBuf>,
    ) -> Result<(), ImodManagerException> {
        let key = self.get_private_key(key);
        if let Some(imod_state) = self.get_string_axis_id_int(&key, axis_id, vector_index)? {
            imod_state.set_working_directory(working_directory);
        }
        Ok(())
    }

    /// Java public final `setPieceListFileName(String, AxisID, int, String)`.
    pub fn set_piece_list_file_name_string_axis_id_int_string(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
        vector_index: i32,
        piece_list_file_name: Option<&str>,
    ) -> Result<(), ImodManagerException> {
        let key = self.get_private_key(key);
        let mut imod_state = self.get_string_axis_id_int(&key, axis_id, vector_index)?;
        if imod_state.is_none() {
            self.new_imod_string_axis_id(&key, axis_id)?;
            imod_state = self.get_string_axis_id(&key, axis_id)?;
        }
        if let Some(imod_state) = imod_state {
            imod_state.set_piece_list_file_name(piece_list_file_name);
        }
        Ok(())
    }

    /// Java public final `stopRequestHandler`.
    pub fn stop_request_handler(&self) {
        if let Some(Some(request_handler)) = self.request_handler.get() {
            request_handler.stop();
        }
    }

    /// Java public final `warnStaleFile`.  Used to prevent warnings about a stale file
    /// from popping up over and over.  Returns true if ImodState.isWarnedStaleFile
    /// returns false.  Also turns on ImodState.warnedStaleFile if it is off.
    pub fn warn_stale_file(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
    ) -> Result<bool, ImodManagerException> {
        let key = self.get_private_key(key);
        let imod_state = self.get_string_axis_id(&key, axis_id)?;
        if let Some(imod_state) = imod_state
            && !imod_state.is_warned_stale_file()
            && imod_state.is_open()
        {
            imod_state.set_warned_stale_file(true);
            return Ok(true);
        }
        Ok(false)
    }

    /// Java final `newVector(ImodState)`.
    pub fn new_vector_imod_state(&self, imod_state: Arc<ImodState>) -> Vec<Arc<ImodState>> {
        let mut vector = Vec::with_capacity(1);
        vector.push(imod_state);
        vector
    }

    /// Java private final `newVector(String, String, AxisID, String)`.
    fn new_vector_string_string_axis_id_string(
        &self,
        key: &str,
        file_extension: Option<&str>,
        axis_id: Option<AxisID>,
        dataset_name: Option<&str>,
    ) -> Result<ImodVector, ImodManagerException> {
        Ok(
            self.new_vector_imod_state(self.new_imod_state_string_string_axis_id_string(
                key,
                file_extension,
                axis_id,
                dataset_name,
            )?),
        )
    }

    /// Java private final `newVector(String, File)`.
    fn new_vector_string_file(
        &self,
        key: &str,
        file: Option<&Path>,
    ) -> Result<ImodVector, ImodManagerException> {
        Ok(self.new_vector_imod_state(self.new_imod_state_string_file(key, file)?))
    }

    /// Java private final `newVector(String, AxisID, File)`.
    fn new_vector_string_axis_id_file(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
        file: Option<&Path>,
    ) -> Result<ImodVector, ImodManagerException> {
        Ok(
            self.new_vector_imod_state(
                self.new_imod_state_string_axis_id_file(key, axis_id, file)?,
            ),
        )
    }

    /// Java private final `newVector(String, String[])`.
    fn new_vector_string_string_array(
        &self,
        key: &str,
        file_name_array: Option<&[String]>,
    ) -> Result<ImodVector, ImodManagerException> {
        Ok(self
            .new_vector_imod_state(self.new_imod_state_string_string_array(key, file_name_array)?))
    }

    /// Java private final `newVector(String, AxisID, File[])`.
    fn new_vector_string_axis_id_file_array(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
        file_list: Option<&[PathBuf]>,
    ) -> Result<ImodVector, ImodManagerException> {
        Ok(self.new_vector_imod_state(
            self.new_imod_state_string_axis_id_file_array(key, axis_id, file_list)?,
        ))
    }

    /// Java private final `newVector(String, String[], String)`.
    fn new_vector_string_string_array_string(
        &self,
        key: &str,
        file_name_array: Option<&[String]>,
        subdir_name: Option<&str>,
    ) -> Result<ImodVector, ImodManagerException> {
        Ok(
            self.new_vector_imod_state(self.new_imod_state_string_string_array_string(
                key,
                file_name_array,
                subdir_name,
            )?),
        )
    }

    /// Java private final `newImodState(String)`.
    fn new_imod_state_string(&self, key: &str) -> Result<Arc<ImodState>, ImodManagerException> {
        self.new_imod_state_string_string_axis_id_string_file_string_array_string_file_array(
            key, None, None, None, None, None, None, None,
        )
    }

    /// Java private final `newImodState(String, AxisID)`.
    fn new_imod_state_string_axis_id(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
    ) -> Result<Arc<ImodState>, ImodManagerException> {
        self.new_imod_state_string_string_axis_id_string_file_string_array_string_file_array(
            key, None, axis_id, None, None, None, None, None,
        )
    }

    /// Java private final `newImodState(String, AxisID, File[])`.
    fn new_imod_state_string_axis_id_file_array(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
        file_list: Option<&[PathBuf]>,
    ) -> Result<Arc<ImodState>, ImodManagerException> {
        self.new_imod_state_string_string_axis_id_string_file_string_array_string_file_array(
            key, None, axis_id, None, None, None, None, file_list,
        )
    }

    /// Java private final `newImodState(String, String, AxisID, String)`.
    fn new_imod_state_string_string_axis_id_string(
        &self,
        key: &str,
        file_extension: Option<&str>,
        axis_id: Option<AxisID>,
        dataset_name: Option<&str>,
    ) -> Result<Arc<ImodState>, ImodManagerException> {
        self.new_imod_state_string_string_axis_id_string_file_string_array_string_file_array(
            key,
            file_extension,
            axis_id,
            dataset_name,
            None,
            None,
            None,
            None,
        )
    }

    /// Java private final `newImodState(String, File)`.
    fn new_imod_state_string_file(
        &self,
        key: &str,
        file: Option<&Path>,
    ) -> Result<Arc<ImodState>, ImodManagerException> {
        self.new_imod_state_string_string_axis_id_string_file_string_array_string_file_array(
            key, None, None, None, file, None, None, None,
        )
    }

    /// Java private final `newImodState(String, AxisID, File)`.
    fn new_imod_state_string_axis_id_file(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
        file: Option<&Path>,
    ) -> Result<Arc<ImodState>, ImodManagerException> {
        self.new_imod_state_string_string_axis_id_string_file_string_array_string_file_array(
            key, None, axis_id, None, file, None, None, None,
        )
    }

    /// Java private final `newImodState(String, String[])`.
    fn new_imod_state_string_string_array(
        &self,
        key: &str,
        file_name_array: Option<&[String]>,
    ) -> Result<Arc<ImodState>, ImodManagerException> {
        self.new_imod_state_string_string_axis_id_string_file_string_array_string_file_array(
            key,
            None,
            None,
            None,
            None,
            file_name_array,
            None,
            None,
        )
    }

    /// Java private final `newImodState(String, String[], String)`.
    fn new_imod_state_string_string_array_string(
        &self,
        key: &str,
        file_name_array: Option<&[String]>,
        subdir_name: Option<&str>,
    ) -> Result<Arc<ImodState>, ImodManagerException> {
        self.new_imod_state_string_string_axis_id_string_file_string_array_string_file_array(
            key,
            None,
            None,
            None,
            None,
            file_name_array,
            subdir_name,
            None,
        )
    }

    /// Java final `get(String)`.
    ///
    /// `Vector.lastElement()` throws on the empty vector `delete` can leave behind;
    /// fixed in translation as "no state".  Likewise `get(String, AxisID)`.
    pub fn get_string(&self, key: &str) -> Result<Option<Arc<ImodState>>, ImodManagerException> {
        let mut imod_map = self.imod_map.lock().unwrap();
        let Some(vector) = self.get_vector_string(&mut imod_map, key)? else {
            return Ok(None);
        };
        Ok(vector.last().cloned())
    }

    /// Java final `get(String, AxisID)`.
    pub fn get_string_axis_id(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
    ) -> Result<Option<Arc<ImodState>>, ImodManagerException> {
        let mut imod_map = self.imod_map.lock().unwrap();
        let Some(vector) = self.get_vector_string_axis_id(&mut imod_map, key, axis_id)? else {
            return Ok(None);
        };
        Ok(vector.last().cloned())
    }

    /// Java private final `get(String, AxisID, String)`.
    fn get_string_axis_id_string(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
        dataset_name: &str,
    ) -> Result<Option<Arc<ImodState>>, ImodManagerException> {
        let vector: ImodVector = {
            let mut imod_map = self.imod_map.lock().unwrap();
            let Some(vector) = self.get_vector_string_axis_id(&mut imod_map, key, axis_id)? else {
                return Ok(None);
            };
            vector.clone()
        };
        for imod_state in vector {
            if imod_state.get_dataset_name() == dataset_name {
                return Ok(Some(imod_state));
            }
        }
        Ok(None)
    }

    /// Java private final `get(String, AxisID, File)`.
    ///
    /// A null file is dereferenced by the source (`file.getAbsolutePath()`); fixed in
    /// translation: it matches no state.
    fn get_string_axis_id_file(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
        file: Option<&Path>,
    ) -> Result<Option<Arc<ImodState>>, ImodManagerException> {
        let vector: ImodVector = {
            let mut imod_map = self.imod_map.lock().unwrap();
            let Some(vector) = self.get_vector_string_axis_id(&mut imod_map, key, axis_id)? else {
                return Ok(None);
            };
            vector.clone()
        };
        let Some(file) = file else {
            return Ok(None);
        };
        let absolute_path = utilities::java_io_file_get_absolute_path(&file.to_string_lossy());
        for imod_state in vector {
            if imod_state.get_dataset_name() == absolute_path {
                return Ok(Some(imod_state));
            }
        }
        Ok(None)
    }

    /// Java private final `get(String, AxisID, int)`.
    ///
    /// The source indexes the vector without a bounds check; an index out of range is
    /// "no state" here, as in `get(String, int)` (fixed in translation).
    fn get_string_axis_id_int(
        &self,
        key: &str,
        axis_id: Option<AxisID>,
        vector_index: i32,
    ) -> Result<Option<Arc<ImodState>>, ImodManagerException> {
        let mut imod_map = self.imod_map.lock().unwrap();
        let Some(vector) = self.get_vector_string_axis_id(&mut imod_map, key, axis_id)? else {
            return Ok(None);
        };
        if vector_index < 0 || vector_index as usize >= vector.len() {
            return Ok(None);
        }
        Ok(Some(Arc::clone(&vector[vector_index as usize])))
    }

    /// Java private final `get(String, int)`.
    fn get_string_int(
        &self,
        key: &str,
        vector_index: i32,
    ) -> Result<Option<Arc<ImodState>>, ImodManagerException> {
        let mut imod_map = self.imod_map.lock().unwrap();
        let vector = self.get_vector_string(&mut imod_map, key)?;
        let Some(vector) =
            vector.filter(|vector| !(vector_index as usize >= vector.len() || vector_index < 0))
        else {
            return Ok(None);
        };
        Ok(Some(Arc::clone(&vector[vector_index as usize])))
    }

    /// Java private final `deleteImodState`.
    fn delete_imod_state(&self, key: &str, vector_index: i32) -> Result<(), ImodManagerException> {
        let mut imod_map = self.imod_map.lock().unwrap();
        let Some(vector) = self.get_vector_string(&mut imod_map, key)? else {
            return Ok(());
        };
        vector.remove(vector_index as usize);
        Ok(())
    }

    /// Java private final `getVector(String)`.  The caller holds the `imodMap` lock and
    /// passes the map in.
    fn get_vector_string<'a>(
        &self,
        imod_map: &'a mut HashMap<String, ImodVector>,
        key: &str,
    ) -> Result<Option<&'a mut ImodVector>, ImodManagerException> {
        if !self.use_map {
            return Err(ImodManagerException::Runtime(
                "This operation is not supported when useMap is false".to_string(),
            ));
        }
        if self.equals_axis_type(AxisType::SingleAxis) && self.is_dual_axis_only(key) {
            return Err(ImodManagerException::AxisType(AxisTypeException::new(
                &format!("{} cannot be found in {}", key, self.axis_type_to_string()),
            )));
        }
        if self.is_dual_axis_only(key) && self.is_per_axis(key) {
            return Err(ImodManagerException::Runtime(format!(
                "{key} cannot be found without axisID information"
            )));
        }
        let vector = if self.is_per_axis(key) {
            imod_map.get_mut(&(key.to_string() + &AxisID::Only.get_extension()))
        } else {
            imod_map.get_mut(key)
        };
        Ok(vector)
    }

    /// Java private final `getVector(String, boolean)`.
    fn get_vector_string_boolean<'a>(
        &self,
        imod_map: &'a mut HashMap<String, ImodVector>,
        key: &str,
        axis_id_in_key: bool,
    ) -> Result<Option<&'a mut ImodVector>, ImodManagerException> {
        if !axis_id_in_key {
            return self.get_vector_string(imod_map, key);
        }
        Ok(imod_map.get_mut(key))
    }

    /// Java private final `getVector(String, AxisID)`.
    fn get_vector_string_axis_id<'a>(
        &self,
        imod_map: &'a mut HashMap<String, ImodVector>,
        key: &str,
        axis_id: Option<AxisID>,
    ) -> Result<Option<&'a mut ImodVector>, ImodManagerException> {
        let Some(mut axis_id) = axis_id else {
            return self.get_vector_string(imod_map, key);
        };
        if !self.use_map {
            return Err(ImodManagerException::Runtime(
                "This operation is not supported when useMap is false".to_string(),
            ));
        }
        if self.equals_axis_type(AxisType::SingleAxis) {
            if self.is_dual_axis_only(key) {
                return Err(ImodManagerException::AxisType(AxisTypeException::new(
                    &format!("{} cannot be found in {}", key, self.axis_type_to_string()),
                )));
            }
            // Correct axis
            if axis_id == AxisID::First {
                axis_id = AxisID::Only;
            }
        }
        let vector = if !self.is_per_axis(key) {
            imod_map.get_mut(key)
        } else {
            imod_map.get_mut(&(key.to_string() + &axis_id.get_extension()))
        };
        Ok(vector)
    }
}
