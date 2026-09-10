//! Model-format dispatcher from `IMOD/flib/subrs/model/readw_or_imod.f`.

use std::path::Path;

use crate::imod::flib::subrs::model::read_mod::read_mod;
use crate::imod::libimod::imodel::Imod;
use crate::imod::libimod::imodel_files::imod_read;

/// Original: `readw_or_imod` (`readw_or_imod.f:11`).
pub fn readw_or_imod(path: impl AsRef<Path>) -> Result<Imod, ()> {
    // `openImodData` is first in the source.  Its native replacement reads
    // the same binary V1.2 model container.  On failure the Fortran routine
    // falls through to its WIMP reader and eventually `read_mod`.
    match imod_read(path.as_ref()) {
        Ok(imod) => Ok(imod),
        Err(_) => read_mod(path),
    }
}
