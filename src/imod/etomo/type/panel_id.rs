//! `IMOD/Etomo/src/etomo/type/PanelId.java`.
//!
//! A typesafe enum of panel identifiers (`public static final` singletons compared by
//! identity), mirrored as a Rust enum.

/// Java `PanelId`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PanelId {
    /// Java `POST_FLATTEN_VOLUME`.
    PostFlattenVolume,
    /// Java `TOOLS_FLATTEN_VOLUME`.
    ToolsFlattenVolume,
    /// Java `CROSS_CORRELATION`.
    CrossCorrelation,
    /// Java `PATCH_TRACKING`.
    PatchTracking,
    /// Java `TILT_3D_FIND`.
    Tilt3dFind,
    /// Java `TILT`.
    Tilt,
    /// Java `SIRTSETUP`.
    Sirtsetup,
    /// Java `CTF_PHASE_FLIP`.
    CtfPhaseFlip,
    /// Java `ALIGN_FRAMES`.
    AlignFrames,
    /// Java `POST_ALT_STACK`.
    PostAltStack,
}

impl PanelId {
    /// Java `rcsid`.
    pub const RCSID: &'static str = "$Id$";

    /// Java field `string`, set by the private `PanelId(String)` constructor.
    fn string(self) -> &'static str {
        match self {
            Self::PostFlattenVolume => "POST_FLATTEN_VOLUME",
            Self::ToolsFlattenVolume => "TOOLS_FLATTEN_VOLUME",
            Self::CrossCorrelation => "CROSS_CORRELATION",
            Self::PatchTracking => "PATCH_TRACKING",
            Self::Tilt3dFind => "TILT_3D_FIND",
            Self::Tilt => "TILT",
            Self::Sirtsetup => "SIRTSETUP",
            Self::CtfPhaseFlip => "CTF_PHASE_FLIP",
            Self::AlignFrames => "ALIGN_FRAMES",
            Self::PostAltStack => "POST_ALT_STACK",
        }
    }
}

/// Java `toString`.
impl std::fmt::Display for PanelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_string_is_the_constructor_string() {
        assert_eq!(PanelId::Tilt3dFind.to_string(), "TILT_3D_FIND");
        assert_eq!(PanelId::Sirtsetup.to_string(), "SIRTSETUP");
        assert_ne!(PanelId::Tilt, PanelId::Tilt3dFind);
    }
}
