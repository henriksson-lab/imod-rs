//! Making an imod model from other file formats, from
//! `IMOD/libimod/imodel_from.c`.
//!
//! The unit compiles to exactly one function.  `Wmod_Colors` (`imodel_from.c:20`)
//! and `imod_from_wmod` (`imodel_from.c:33`) are both inside `#ifdef NEEDWMOD`,
//! and `NEEDWMOD` is defined nowhere in the pinned tree — `grep -rn NEEDWMOD
//! IMOD/` matches only that `#ifdef` line — so neither is compiled.
//! `nm -D --defined-only` on the reference `libimod.so` confirms it: `substr`
//! is the only symbol this unit contributes.
//!
//! The guarded `imod_from_wmod` is also not translatable as written.  It calls
//! `imodNewObject(mod)` and writes `mod->obj[nobj]` in its first loop, roughly
//! sixty lines before `mod = imodNew();` assigns `mod` at `imodel_from.c:104`,
//! so the pointer is indeterminate at every one of those uses; it also contains
//! the statement `mod->cindex.object;` with no effect, and sets
//! `obj->cont[...].psize = points` after `imodPointAdd` has already maintained
//! that count.  A translation would have to invent an execution order the
//! source does not have, so the deviation is recorded here instead.  The live
//! WIMP-to-model path in this crate is `IMOD/imodutil/wmod2imod.c`, translated
//! in `src/imod/imodutil/wmod2imod.rs`, which contains its own copy of the same
//! parsing loop and is the code the shipped `wmod2imod` command actually runs.

/// Original: `MAXLINE` (`imodel_from.c:16`).
pub const MAXLINE: usize = 128;

/// Original: `MAXOBJ` (`imodel_from.c:17`).
pub const MAXOBJ: usize = 256;

/// Original: `substr` (`imodel_from.c:207`).
///
/// Returns 1 when the first `strlen(ls)` bytes of `bs` equal `ls`, 0 otherwise.
/// Both arguments are the C `char[]`, taken here as the NUL-terminated bytes;
/// `ls` is measured up to its terminator exactly as `strlen` does, and `bs` is
/// indexed without a bounds test, so the caller must pass a buffer at least as
/// long as `ls`.
pub fn substr(bs: &[u8], ls: &[u8]) -> i32 {
    let len: i32;

    len = ls.iter().position(|&b| b == 0).unwrap_or(ls.len()) as i32;

    for i in 0..len as usize {
        if bs[i] == ls[i] {
            continue;
        } else {
            return 0;
        }
    }
    1
}
