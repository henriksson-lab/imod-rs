//! `IMOD/Etomo/src/etomo/type/CombineProcessType.java`.

use super::process_name::ProcessName;

/// Java `SOLVEMATCH_DUALVOLMATCH_INDEX`.
pub const SOLVEMATCH_DUALVOLMATCH_INDEX: i32 = 0;
/// Java `MATCHVOL1_INDEX`.
pub const MATCHVOL1_INDEX: i32 = 1;
/// Java `PATCHCORR_INDEX`.
pub const PATCHCORR_INDEX: i32 = 2;
/// Java `MATCHORWARP_INDEX`.
pub const MATCHORWARP_INDEX: i32 = 3;
/// Java `VOLCOMBINE_INDEX`.
pub const VOLCOMBINE_INDEX: i32 = 4;
/// Java `TOTAL`.
pub const TOTAL: i32 = VOLCOMBINE_INDEX + 1;

/// Java final class `CombineProcessType`: its six singletons.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CombineProcessType {
    Solvematch,
    Dualvolmatch,
    Matchvol1,
    Patchcorr,
    Matchorwarp,
    Volcombine,
}

impl CombineProcessType {
    pub const SOLVEMATCH: CombineProcessType = CombineProcessType::Solvematch;
    pub const DUALVOLMATCH: CombineProcessType = CombineProcessType::Dualvolmatch;
    pub const MATCHVOL1: CombineProcessType = CombineProcessType::Matchvol1;
    pub const PATCHCORR: CombineProcessType = CombineProcessType::Patchcorr;
    pub const MATCHORWARP: CombineProcessType = CombineProcessType::Matchorwarp;
    pub const VOLCOMBINE: CombineProcessType = CombineProcessType::Volcombine;

    /// Java `getInstance(String)`.
    pub fn get_instance(process_name: &str) -> Option<CombineProcessType> {
        if ProcessName::SOLVEMATCH.equals(process_name) {
            return Some(CombineProcessType::Solvematch);
        }
        if ProcessName::DUALVOLMATCH.equals(process_name) {
            return Some(CombineProcessType::Dualvolmatch);
        }
        if ProcessName::MATCHVOL1.equals(process_name) {
            return Some(CombineProcessType::Matchvol1);
        }
        if ProcessName::PATCHCORR.equals(process_name) {
            return Some(CombineProcessType::Patchcorr);
        }
        if ProcessName::MATCHORWARP.equals(process_name) {
            return Some(CombineProcessType::Matchorwarp);
        }
        if ProcessName::VOLCOMBINE.equals(process_name) {
            return Some(CombineProcessType::Volcombine);
        }
        None
    }

    /// Java `getInstance(int, boolean)`.
    pub fn get_instance_index(
        process_index: i32,
        initial_volume_matching: bool,
    ) -> Option<CombineProcessType> {
        if process_index == SOLVEMATCH_DUALVOLMATCH_INDEX {
            if !initial_volume_matching {
                return Some(CombineProcessType::Solvematch);
            }
            return Some(CombineProcessType::Dualvolmatch);
        }
        if process_index == MATCHVOL1_INDEX {
            return Some(CombineProcessType::Matchvol1);
        }
        if process_index == PATCHCORR_INDEX {
            return Some(CombineProcessType::Patchcorr);
        }
        if process_index == MATCHORWARP_INDEX {
            return Some(CombineProcessType::Matchorwarp);
        }
        if process_index == VOLCOMBINE_INDEX {
            return Some(CombineProcessType::Volcombine);
        }
        None
    }

    /// Java `getIndex`.
    pub fn get_index(self) -> i32 {
        match self {
            CombineProcessType::Solvematch | CombineProcessType::Dualvolmatch => {
                SOLVEMATCH_DUALVOLMATCH_INDEX
            }
            CombineProcessType::Matchvol1 => MATCHVOL1_INDEX,
            CombineProcessType::Patchcorr => PATCHCORR_INDEX,
            CombineProcessType::Matchorwarp => MATCHORWARP_INDEX,
            CombineProcessType::Volcombine => VOLCOMBINE_INDEX,
        }
    }
}

/// Java `toString`.
impl std::fmt::Display for CombineProcessType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            CombineProcessType::Solvematch => ProcessName::SOLVEMATCH,
            CombineProcessType::Dualvolmatch => ProcessName::DUALVOLMATCH,
            CombineProcessType::Matchvol1 => ProcessName::MATCHVOL1,
            CombineProcessType::Patchcorr => ProcessName::PATCHCORR,
            CombineProcessType::Matchorwarp => ProcessName::MATCHORWARP,
            CombineProcessType::Volcombine => ProcessName::VOLCOMBINE,
        };
        write!(f, "{name}")
    }
}
