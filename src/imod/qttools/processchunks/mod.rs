//! Rust translations of `IMOD/qttools/processchunks`.

pub const CHUNK_SYNC: i32 = -1;
pub const CHUNK_NOT_DONE: i32 = 0;
pub const CHUNK_ASSIGNED: i32 = 1;
pub const CHUNK_DONE: i32 = 2;
pub const CHUNK_TO_SKIP: i32 = 3;

pub mod comfilejobs;
pub mod machinehandler;
// `processchunks` is IMOD's distributed job scheduler: it drives chunks over
// **ssh** to other machines and submits to batch queues (`processchunks.cpp`
// carries 19 ssh and 32 queue references), and `processhandler` emulates Qt's
// `QProcess::finished` with `waitid(WEXITED | WNOWAIT)` so a child stays
// reapable.  Both are POSIX process/remote-execution services with no Windows
// counterpart worth inventing (user, 2026-09-23: *"ssh etc is linux only. I
// dont expect windows portability there, and we can feature gate it"*).
pub mod processchunks;
pub mod processhandler;
