//! `IMOD/Etomo/src/etomo/ui/swing/ValidationExtension.java`.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

/// Java `ValidationExtension`: the validation settings of an efield.
pub struct ValidationExtension {
    /// Java `locationDescr`.
    location_descr: RefCell<Option<String>>,
    /// Java `fileOnly`.
    file_only: Cell<bool>,
    /// Java `fileMustExist`.
    file_must_exist: Cell<bool>,
    /// Java `mustBePositive`.
    must_be_positive: Cell<bool>,
    /// Java `required`.
    required: Cell<bool>,
}

impl ValidationExtension {
    /// Java `ValidationExtension()`.
    pub fn new() -> Rc<ValidationExtension> {
        Rc::new(ValidationExtension {
            location_descr: RefCell::new(None),
            file_only: Cell::new(false),
            file_must_exist: Cell::new(false),
            must_be_positive: Cell::new(false),
            required: Cell::new(false),
        })
    }

    /// Java `setRequired(boolean)`.
    pub fn set_required(&self, required: bool) {
        self.required.set(required);
    }

    /// Java `getLocationAddon()`.
    pub fn get_location_addon(&self) -> String {
        match self.location_descr.borrow().as_deref() {
            Some(location_descr) => " in ".to_string() + location_descr,
            None => String::new(),
        }
    }

    /// Java `setLocationDescr(String)`.
    pub fn set_location_descr(&self, location_descr: Option<&str>) {
        *self.location_descr.borrow_mut() = location_descr.map(str::to_owned);
    }

    /// Java `setFileOnly(boolean)`.
    pub fn set_file_only(&self, file_only: bool) {
        self.file_only.set(file_only);
    }

    /// Java `setFileMustExist(boolean)`.
    pub fn set_file_must_exist(&self, file_must_exist: bool) {
        self.file_must_exist.set(file_must_exist);
    }

    /// Java `setMustBePositive(boolean)`.
    pub fn set_must_be_positive(&self, must_be_positive: bool) {
        self.must_be_positive.set(must_be_positive);
    }

    /// Java `isRequired()`.
    pub fn is_required(&self) -> bool {
        self.required.get()
    }

    /// Java `isFileOnly()`.
    pub fn is_file_only(&self) -> bool {
        self.file_only.get()
    }

    /// Java `isFileMustExist()`.
    pub fn is_file_must_exist(&self) -> bool {
        self.file_must_exist.get()
    }

    /// Java `isMustBePositive()`.
    pub fn is_must_be_positive(&self) -> bool {
        self.must_be_positive.get()
    }
}
