//! Stand-in for `java.awt.EventQueue`, the Swing event dispatch thread (EDT).
//!
//! **Not a translated unit.**  eTomo's Java relies on the JDK's EDT: every
//! Swing component is created and changed on it, `SwingUtilities.invokeLater`
//! queues work onto it, and it exists whether or not a display is attached
//! (`-headless` still has one).  eTomo's process threads (`ComScriptProcess`,
//! `BackgroundProcess`, the monitors) call back into managers and `UIHarness`
//! from their own threads; Swing tolerates that loosely, Slint and Rust's
//! `Rc` do not.  This module is the one place that decision lives:
//!
//! * [`invoke_later`] runs a closure on the EDT.  With the `gui` feature and
//!   a Slint main frame installed ([`install_slint_edt`]) that is
//!   `slint::invoke_from_event_loop`; otherwise it is a dedicated
//!   `AWT-EventQueue-0` thread, started on first use, which is what Java's
//!   headless EDT is.
//! * [`EdtRef`] carries an EDT-confined `Rc` value (a dialog, a
//!   `ProcessResultDisplay`, a `ProcessSeries`) through a process thread
//!   without touching it there, the way a Java process thread holds a
//!   reference it only hands back to the manager.  Dereferencing it anywhere
//!   but its owning thread is a translation defect and panics by name.
//!
//! Where the Java calls a UI method straight from a process thread
//! (`manager.processDone(...)`, `uiHarness.openMessageDialog(...)`), the
//! translation posts that call with [`invoke_later`] and says so at the site.

use std::mem::ManuallyDrop;
use std::rc::Rc;
use std::sync::mpsc;
use std::sync::{Mutex, OnceLock};
use std::thread::ThreadId;

type Job = Box<dyn FnOnce() + Send + 'static>;

enum Backend {
    /// The headless `AWT-EventQueue-0` thread and its queue.
    Headless(Mutex<mpsc::Sender<Job>>),
    /// `slint::invoke_from_event_loop` on the thread that owns the Slint
    /// main frame.
    #[cfg(feature = "gui")]
    Slint,
}

struct Edt {
    thread: ThreadId,
    backend: Backend,
}

static EDT: OnceLock<Edt> = OnceLock::new();

fn edt() -> &'static Edt {
    EDT.get_or_init(|| {
        let (sender, receiver) = mpsc::channel::<Job>();
        let (id_sender, id_receiver) = mpsc::channel::<ThreadId>();
        std::thread::Builder::new()
            .name("AWT-EventQueue-0".to_owned())
            .spawn(move || {
                let _ = id_sender.send(std::thread::current().id());
                while let Ok(job) = receiver.recv() {
                    // A failing job must not take the EDT down with it; Java's
                    // EDT prints the exception and carries on.
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job));
                }
            })
            .expect("starting the event dispatch thread");
        Edt {
            thread: id_receiver.recv().expect("event dispatch thread id"),
            backend: Backend::Headless(Mutex::new(sender)),
        }
    })
}

/// Makes the calling thread, which runs the Slint event loop, the EDT.  Must
/// be called before anything is posted (`UIHarness.createMainFrame`); a later
/// call is ignored and returns false.
#[cfg(feature = "gui")]
pub fn install_slint_edt() -> bool {
    EDT.set(Edt {
        thread: std::thread::current().id(),
        backend: Backend::Slint,
    })
    .is_ok()
}

/// `SwingUtilities.isEventDispatchThread()`.
pub fn is_dispatch_thread() -> bool {
    edt().thread == std::thread::current().id()
}

/// `SwingUtilities.invokeLater(Runnable)`.
pub fn invoke_later(job: impl FnOnce() + Send + 'static) {
    match &edt().backend {
        Backend::Headless(sender) => {
            let _ = sender.lock().unwrap().send(Box::new(job));
        }
        #[cfg(feature = "gui")]
        Backend::Slint => {
            let _ = slint::invoke_from_event_loop(job);
        }
    }
}

/// `SwingUtilities.invokeAndWait(Runnable)`: runs `job` on the EDT and waits
/// for its result.  Called on the EDT it runs `job` directly, where Java
/// would throw; nothing in the translation relies on that exception.
pub fn invoke_and_wait<R: Send + 'static>(job: impl FnOnce() -> R + Send + 'static) -> R {
    if is_dispatch_thread() {
        return job();
    }
    let (sender, receiver) = mpsc::channel();
    invoke_later(move || {
        // Java wraps an exception thrown by the job in an InvocationTargetException
        // for the waiting thread; the panic is handed back the same way.
        let _ = sender.send(std::panic::catch_unwind(std::panic::AssertUnwindSafe(job)));
    });
    match receiver
        .recv()
        .expect("event dispatch thread job did not complete")
    {
        Ok(result) => result,
        Err(panic) => std::panic::resume_unwind(panic),
    }
}

/// An `Rc<T>` confined to the thread that created it, which may nevertheless
/// travel through other threads.  See the module comment.
pub struct EdtRef<T: ?Sized + 'static> {
    value: ManuallyDrop<Rc<T>>,
    owner: ThreadId,
}

// SAFETY: the `Rc` is only cloned, dereferenced or dropped on `owner` (every
// accessor checks), so its non-atomic reference count is never touched by two
// threads.  A drop on another thread is sent back to the EDT when the owner is
// the EDT and is otherwise leaked, never performed.
unsafe impl<T: ?Sized + 'static> Send for EdtRef<T> {}
unsafe impl<T: ?Sized + 'static> Sync for EdtRef<T> {}

impl<T: ?Sized + 'static> EdtRef<T> {
    /// Wraps `value`, owned by the calling thread.
    pub fn new(value: Rc<T>) -> EdtRef<T> {
        EdtRef {
            value: ManuallyDrop::new(value),
            owner: std::thread::current().id(),
        }
    }

    /// True on the owning thread.
    pub fn is_owner_thread(&self) -> bool {
        self.owner == std::thread::current().id()
    }

    /// The value.  Panics off the owning thread.
    pub fn get(&self) -> &Rc<T> {
        assert!(
            self.is_owner_thread(),
            "EdtRef dereferenced off its owning (event dispatch) thread"
        );
        &self.value
    }

    /// Java reference identity (`==`), valid on any thread.
    pub fn ptr_eq(&self, other: &EdtRef<T>) -> bool {
        std::ptr::addr_eq(Rc::as_ptr(&self.value), Rc::as_ptr(&other.value))
    }
}

impl<T: ?Sized + 'static> Clone for EdtRef<T> {
    fn clone(&self) -> EdtRef<T> {
        EdtRef {
            value: ManuallyDrop::new(Rc::clone(self.get())),
            owner: self.owner,
        }
    }
}

impl<T: ?Sized + 'static> Drop for EdtRef<T> {
    fn drop(&mut self) {
        // SAFETY: `value` is taken exactly once, here.
        let value = unsafe { ManuallyDrop::take(&mut self.value) };
        if self.is_owner_thread() {
            drop(value);
            return;
        }
        if self.owner == edt().thread {
            struct Carry<T: ?Sized>(Rc<T>);
            // SAFETY: the Rc is only dropped, on the EDT that owns it.
            unsafe impl<T: ?Sized> Send for Carry<T> {}
            let carry = Carry(value);
            invoke_later(move || drop(carry));
        } else {
            std::mem::forget(value);
        }
    }
}

impl<T: ?Sized + 'static> std::fmt::Debug for EdtRef<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "EdtRef@{:p}", Rc::as_ptr(&self.value) as *const ())
    }
}

/// A value owned by the event dispatch thread inside an object other threads
/// share (a manager's dialogs and panels: the manager is `Send + Sync`, the
/// Swing objects it holds are not).  Only the EDT may touch the value;
/// [`EdtCell::with`] checks.
pub struct EdtCell<T: 'static> {
    value: std::cell::RefCell<Option<T>>,
}

// SAFETY: the value is only reached through `with`/`set`/`take`, which panic
// off the event dispatch thread, so it is only ever touched on one thread.
unsafe impl<T: 'static> Send for EdtCell<T> {}
unsafe impl<T: 'static> Sync for EdtCell<T> {}

impl<T: 'static> Default for EdtCell<T> {
    fn default() -> EdtCell<T> {
        EdtCell {
            value: std::cell::RefCell::new(None),
        }
    }
}

impl<T: 'static> EdtCell<T> {
    /// An empty cell (Java's null field).
    pub fn new() -> EdtCell<T> {
        EdtCell::default()
    }

    fn check() {
        assert!(
            is_dispatch_thread(),
            "EdtCell used off the event dispatch thread"
        );
    }

    /// Sets the value.  EDT only.
    pub fn set(&self, value: Option<T>) {
        EdtCell::<T>::check();
        *self.value.borrow_mut() = value;
    }

    /// Takes the value out.  EDT only.
    pub fn take(&self) -> Option<T> {
        EdtCell::<T>::check();
        self.value.borrow_mut().take()
    }

    /// True when the cell holds a value.  EDT only.
    pub fn is_some(&self) -> bool {
        EdtCell::<T>::check();
        self.value.borrow().is_some()
    }

    /// Runs `f` on the value, if any.  EDT only.
    pub fn with<R>(&self, f: impl FnOnce(&mut T) -> R) -> Option<R> {
        EdtCell::<T>::check();
        let mut value = self.value.borrow_mut();
        value.as_mut().map(f)
    }
}

impl<T: Clone + 'static> EdtCell<T> {
    /// A clone of the value (for `Rc` handles).  EDT only.
    pub fn get(&self) -> Option<T> {
        EdtCell::<T>::check();
        self.value.borrow().clone()
    }
}

/// A lock the owning thread may take again, standing in for the *absence* of
/// one in Java: eTomo shares some non-thread-safe objects (the
/// `ComScriptManager`) between the EDT and its monitor threads and simply
/// hopes they do not collide.  Rust's `Rc`/`RefCell` inside those objects make
/// a collision undefined behaviour, so every access holds this lock for the
/// duration of one call; re-entry from the same thread (a manager method that
/// calls back into another) does not deadlock.
pub struct ReentrantLock {
    owner: Mutex<Option<(ThreadId, usize)>>,
    released: std::sync::Condvar,
}

impl Default for ReentrantLock {
    fn default() -> Self {
        Self::new()
    }
}

impl ReentrantLock {
    pub const fn new() -> ReentrantLock {
        ReentrantLock {
            owner: Mutex::new(None),
            released: std::sync::Condvar::new(),
        }
    }

    /// Takes the lock, waiting while another thread holds it.
    pub fn lock(&self) -> ReentrantGuard<'_> {
        let me = std::thread::current().id();
        let mut owner = self.owner.lock().unwrap();
        loop {
            match *owner {
                None => {
                    *owner = Some((me, 1));
                    break;
                }
                Some((thread, count)) if thread == me => {
                    *owner = Some((thread, count + 1));
                    break;
                }
                Some(_) => owner = self.released.wait(owner).unwrap(),
            }
        }
        ReentrantGuard {
            lock: self,
            _not_send: std::marker::PhantomData,
        }
    }
}

/// Held for one access through a [`ReentrantLock`]; not `Send`, so it is
/// released on the thread that took it.
pub struct ReentrantGuard<'a> {
    lock: &'a ReentrantLock,
    _not_send: std::marker::PhantomData<*const ()>,
}

impl Drop for ReentrantGuard<'_> {
    fn drop(&mut self) {
        let mut owner = self.lock.owner.lock().unwrap();
        if let Some((thread, count)) = *owner {
            if count > 1 {
                *owner = Some((thread, count - 1));
            } else {
                *owner = None;
                self.lock.released.notify_all();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invoke_and_wait_runs_on_the_dispatch_thread() {
        assert!(!is_dispatch_thread());
        assert!(invoke_and_wait(is_dispatch_thread));
    }

    #[test]
    fn edt_ref_travels_and_drops_on_its_owner() {
        let made = invoke_and_wait(|| EdtRef::new(Rc::new(7)));
        let copy = invoke_and_wait({
            let made_ptr = &made as *const EdtRef<i32> as usize;
            move || {
                // SAFETY: `made` outlives the wait.
                let made = unsafe { &*(made_ptr as *const EdtRef<i32>) };
                (made.clone(), **made.get())
            }
        });
        assert_eq!(copy.1, 7);
        assert!(copy.0.ptr_eq(&made));
        drop(copy);
        drop(made);
        assert!(invoke_and_wait(|| true));
    }
}
