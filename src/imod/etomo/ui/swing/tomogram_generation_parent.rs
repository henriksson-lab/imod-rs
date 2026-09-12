//! `IMOD/Etomo/src/etomo/ui/swing/TomogramGenerationParent.java`.
//!
//! This direct super-interface is required by `Tilt3dFindParent.java`.
#![allow(dead_code)]

/// Java public `TomogramGenerationParent`.
pub trait TomogramGenerationParent {
    fn is_ctf3d(&self) -> bool;
    fn is_method_plugin(&self) -> bool;
    fn is_multifilt(&self) -> bool;
    fn is_back_projection(&self) -> bool;
    fn is_sirt(&self) -> bool;
}
