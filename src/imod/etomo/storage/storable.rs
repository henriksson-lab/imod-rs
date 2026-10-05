//! `IMOD/Etomo/src/etomo/storage/Storable.java`.
//!
//! Java objects are shared by reference, and `ParameterStore.load(Storable)` writes into
//! the very object a manager (or a meta data, or a panel) keeps.  In this translation the
//! managers hold their storables as `&'static` shared references, so the Java interface
//! is the `Storable` trait below with **all four methods on `&self`**: an implementor
//! keeps the fields `load` assigns behind interior mutability (`Mutex`/atomics where the
//! object crosses threads, `Cell`/`RefCell` for EDT-only Swing objects).
//!
//! A few Java storables are small value objects that the translation embeds by value in
//! their owner and mutates through `&mut` (`EtomoNumber`, `EtomoVersion`, the comscript
//! parameter objects, `ProcessData`, `UserConfiguration`, ...).  Giving each of those
//! interior mutability would rewrite every one of their setters and getters.  They
//! implement `StorableValue` instead - the same four Java methods, with `load` on
//! `&mut self` - and the `Mutex`/`RefCell` that makes such a value a shared Java object
//! is its `Storable` (blanket impls below).  So `ParameterStore.load(processData)`, where
//! Java's `processData` is shared by the process manager and a process thread, is
//! `parameter_store.load(&*arc_mutex_process_data)` here.  `StorableValue` is Rust-only
//! plumbing: it declares exactly the Java interface's methods and nothing else.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

/// Rust trait equivalent to Java `Storable`: its four source methods operate
/// on a Java-`Properties`-equivalent deterministic string map.  Every method takes the
/// object shared, as Java's does.
pub trait Storable {
    /// Java `store(Properties)`.
    fn store(&self, properties: &mut BTreeMap<String, String>);
    /// Java `store(Properties, String)`.
    fn store_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str);
    /// Java `load(Properties)`.
    fn load(&self, properties: &mut BTreeMap<String, String>);
    /// Java `load(Properties, String)`.
    fn load_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str);
}

/// Java `Storable`, for a value object this translation owns by value and mutates
/// through `&mut` (see the module comment).  `Mutex<T>` and `RefCell<T>` of such a value
/// are its shared `Storable` form.
pub trait StorableValue {
    /// Java `store(Properties)`.
    fn store(&self, properties: &mut BTreeMap<String, String>);
    /// Java `store(Properties, String)`.
    fn store_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str);
    /// Java `load(Properties)`.
    fn load(&mut self, properties: &mut BTreeMap<String, String>);
    /// Java `load(Properties, String)`.
    fn load_with_prepend(&mut self, properties: &mut BTreeMap<String, String>, prepend: &str);
}

/// A shared value object (`Mutex` for one that crosses threads): each call is one Java
/// call on the object, made under its lock.
impl<T: StorableValue + ?Sized> Storable for Mutex<T> {
    fn store(&self, properties: &mut BTreeMap<String, String>) {
        StorableValue::store(&*self.lock().unwrap(), properties);
    }
    fn store_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        StorableValue::store_with_prepend(&*self.lock().unwrap(), properties, prepend);
    }
    fn load(&self, properties: &mut BTreeMap<String, String>) {
        StorableValue::load(&mut *self.lock().unwrap(), properties);
    }
    fn load_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        StorableValue::load_with_prepend(&mut *self.lock().unwrap(), properties, prepend);
    }
}

/// A shared value object confined to one thread (the EDT, or a lock the owner holds).
impl<T: StorableValue + ?Sized> Storable for RefCell<T> {
    fn store(&self, properties: &mut BTreeMap<String, String>) {
        StorableValue::store(&*self.borrow(), properties);
    }
    fn store_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        StorableValue::store_with_prepend(&*self.borrow(), properties, prepend);
    }
    fn load(&self, properties: &mut BTreeMap<String, String>) {
        StorableValue::load(&mut *self.borrow_mut(), properties);
    }
    fn load_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        StorableValue::load_with_prepend(&mut *self.borrow_mut(), properties, prepend);
    }
}

/// A Java reference held through `Arc` (the process data a process manager and a
/// process thread share) is the object it points to.
impl<T: Storable + ?Sized> Storable for Arc<T> {
    fn store(&self, properties: &mut BTreeMap<String, String>) {
        (**self).store(properties);
    }
    fn store_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        (**self).store_with_prepend(properties, prepend);
    }
    fn load(&self, properties: &mut BTreeMap<String, String>) {
        (**self).load(properties);
    }
    fn load_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        (**self).load_with_prepend(properties, prepend);
    }
}

/// A plain shared reference (`&'static dyn Storable` from a manager) is the object it
/// points to.
impl<T: Storable + ?Sized> Storable for &T {
    fn store(&self, properties: &mut BTreeMap<String, String>) {
        (**self).store(properties);
    }
    fn store_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        (**self).store_with_prepend(properties, prepend);
    }
    fn load(&self, properties: &mut BTreeMap<String, String>) {
        (**self).load(properties);
    }
    fn load_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        (**self).load_with_prepend(properties, prepend);
    }
}

// `Rc<T>` is deliberately not given a blanket impl: `ParallelPanel` and
// `ProcessorTable` implement `Storable` on their `Rc` handles themselves.
