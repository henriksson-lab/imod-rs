//! Owned Rust translation of `IMOD/raptor/opencv/cxalloc.cpp` and its paired
//! declarations in `cxcore.h`.
//!
//! The C API hands out manually freed, 32-byte-aligned raw pointers.  Rust
//! callers retain the same configurable allocator policy, but an allocation
//! is an owned value instead.  This makes invalid pointers, misalignment, and
//! double frees unrepresentable while preserving the C size limit and the
//! requirement that allocator and deallocator are installed as a pair.

use std::any::Any;
use std::sync::{Mutex, OnceLock};

/// C `CV_MALLOC_ALIGN` from `cxmisc.h`.
pub const CV_MALLOC_ALIGN: usize = 32;

/// C `CV_MAX_ALLOC_SIZE` from `cxmisc.h`.
pub const CV_MAX_ALLOC_SIZE: usize = 1usize << (usize::BITS - 2);

/// Failure statuses reported by the allocation entry points.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CvAllocError {
    /// C `CV_StsNullPtr`: only one of the allocator pair was supplied.
    MissingMemoryManagerPair,
    /// C `CV_StsOutOfRange`: the requested byte count exceeds OpenCV's limit.
    AllocationSizeOutOfRange,
    /// C `CV_StsNoMem`: the backing Rust allocator could not reserve storage.
    OutOfMemory,
    /// The installed deallocator rejected the allocation.
    DeallocationError,
}

/// Safe, owned replacement for one C `cvAlloc` block.
#[derive(Debug, Eq, PartialEq)]
pub struct CvAllocation {
    bytes: Vec<u8>,
}

impl CvAllocation {
    /// Number of bytes requested through `cv_alloc`.
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Whether this allocation has no addressable bytes.
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Immutable storage view replacing C pointer arithmetic.
    pub fn as_slice(&self) -> &[u8] {
        &self.bytes
    }

    /// Mutable storage view replacing C pointer arithmetic.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.bytes
    }
}

/// C `CvAllocFunc`, expressed with an owned allocation and typed user state.
pub type CvAllocFunc = fn(usize, &mut dyn Any) -> Result<CvAllocation, CvAllocError>;

/// C `CvFreeFunc`, expressed by consuming the allocation it releases.
pub type CvFreeFunc = fn(CvAllocation, &mut dyn Any) -> Result<(), CvAllocError>;

struct CvMemoryManager {
    alloc: CvAllocFunc,
    free: CvFreeFunc,
    userdata: Box<dyn Any + Send>,
}

static MEMORY_MANAGER: OnceLock<Mutex<CvMemoryManager>> = OnceLock::new();

fn icv_default_alloc(size: usize, _: &mut dyn Any) -> Result<CvAllocation, CvAllocError> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(size)
        .map_err(|_| CvAllocError::OutOfMemory)?;
    bytes.resize(size, 0);
    Ok(CvAllocation { bytes })
}

fn icv_default_free(allocation: CvAllocation, _: &mut dyn Any) -> Result<(), CvAllocError> {
    drop(allocation);
    Ok(())
}

/// C `cvSetMemoryManager`.
///
/// Supplying neither callback restores the default owned allocator.  As in C,
/// supplying exactly one callback is an error.
pub fn cv_set_memory_manager(
    alloc_func: Option<CvAllocFunc>,
    free_func: Option<CvFreeFunc>,
    userdata: Option<Box<dyn Any + Send>>,
) -> Result<(), CvAllocError> {
    let (alloc, free) = match (alloc_func, free_func) {
        (Some(alloc), Some(free)) => (alloc, free),
        (None, None) => (
            icv_default_alloc as CvAllocFunc,
            icv_default_free as CvFreeFunc,
        ),
        _ => return Err(CvAllocError::MissingMemoryManagerPair),
    };

    let mut manager = MEMORY_MANAGER
        .get_or_init(|| {
            Mutex::new(CvMemoryManager {
                alloc: icv_default_alloc,
                free: icv_default_free,
                userdata: Box::new(()),
            })
        })
        .lock()
        .expect("memory manager lock poisoned");
    manager.alloc = alloc;
    manager.free = free;
    manager.userdata = userdata.unwrap_or_else(|| Box::new(()));
    Ok(())
}

/// C `cvAlloc`.
pub fn cv_alloc(size: usize) -> Result<CvAllocation, CvAllocError> {
    if size > CV_MAX_ALLOC_SIZE {
        return Err(CvAllocError::AllocationSizeOutOfRange);
    }

    let mut manager = MEMORY_MANAGER
        .get_or_init(|| {
            Mutex::new(CvMemoryManager {
                alloc: icv_default_alloc,
                free: icv_default_free,
                userdata: Box::new(()),
            })
        })
        .lock()
        .expect("memory manager lock poisoned");
    (manager.alloc)(size, manager.userdata.as_mut())
}

/// C `cvFree_`; `None` is the C null-pointer no-op.
pub fn cv_free_(allocation: Option<CvAllocation>) -> Result<(), CvAllocError> {
    let Some(allocation) = allocation else {
        return Ok(());
    };
    let mut manager = MEMORY_MANAGER
        .get_or_init(|| {
            Mutex::new(CvMemoryManager {
                alloc: icv_default_alloc,
                free: icv_default_free,
                userdata: Box::new(()),
            })
        })
        .lock()
        .expect("memory manager lock poisoned");
    (manager.free)(allocation, manager.userdata.as_mut())
}

/// Safe equivalent of the `cvFree(ptr)` C macro: free and clear the handle.
pub fn cv_free(allocation: &mut Option<CvAllocation>) -> Result<(), CvAllocError> {
    cv_free_(allocation.take())
}

#[cfg(test)]
mod tests {
    use super::{
        CV_MAX_ALLOC_SIZE, CvAllocError, CvAllocFunc, CvAllocation, CvFreeFunc, cv_alloc, cv_free,
        cv_free_, cv_set_memory_manager,
    };
    use std::any::Any;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn custom_alloc(size: usize, userdata: &mut dyn Any) -> Result<CvAllocation, CvAllocError> {
        userdata
            .downcast_mut::<Arc<AtomicUsize>>()
            .unwrap()
            .fetch_add(1, Ordering::SeqCst);
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(size)
            .map_err(|_| CvAllocError::OutOfMemory)?;
        bytes.resize(size, 0);
        let mut allocation = CvAllocation { bytes };
        allocation.as_mut_slice().fill(7);
        Ok(allocation)
    }

    fn custom_free(_: CvAllocation, userdata: &mut dyn Any) -> Result<(), CvAllocError> {
        userdata
            .downcast_mut::<Arc<AtomicUsize>>()
            .unwrap()
            .fetch_add(10, Ordering::SeqCst);
        Ok(())
    }

    #[test]
    fn default_allocator_owns_zeroed_requested_bytes() {
        let _guard = TEST_LOCK.lock().unwrap();
        cv_set_memory_manager(None, None, None).unwrap();
        let mut allocation = cv_alloc(5).unwrap();
        assert_eq!(allocation.as_slice(), [0; 5]);
        allocation.as_mut_slice()[2] = 9;
        assert_eq!(allocation.as_slice(), [0, 0, 9, 0, 0]);
        cv_free_(Some(allocation)).unwrap();
    }

    #[test]
    fn free_macro_counterpart_clears_owned_handle() {
        let _guard = TEST_LOCK.lock().unwrap();
        cv_set_memory_manager(None, None, None).unwrap();
        let mut allocation = Some(cv_alloc(3).unwrap());
        cv_free(&mut allocation).unwrap();
        assert!(allocation.is_none());
        cv_free(&mut allocation).unwrap();
    }

    #[test]
    fn manager_callbacks_must_be_installed_together() {
        let _guard = TEST_LOCK.lock().unwrap();
        assert_eq!(
            cv_set_memory_manager(Some(custom_alloc as CvAllocFunc), None, None),
            Err(CvAllocError::MissingMemoryManagerPair)
        );
    }

    #[test]
    fn custom_manager_receives_each_allocation_and_free() {
        let _guard = TEST_LOCK.lock().unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        cv_set_memory_manager(
            Some(custom_alloc as CvAllocFunc),
            Some(custom_free as CvFreeFunc),
            Some(Box::new(Arc::clone(&calls))),
        )
        .unwrap();
        let allocation = cv_alloc(2).unwrap();
        assert_eq!(allocation.as_slice(), [7, 7]);
        cv_free_(Some(allocation)).unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 11);
        cv_set_memory_manager(None, None, None).unwrap();
    }

    #[test]
    fn maximum_size_guard_matches_c_source() {
        assert_eq!(
            cv_alloc(CV_MAX_ALLOC_SIZE.saturating_add(1)),
            Err(CvAllocError::AllocationSizeOutOfRange)
        );
    }
}
