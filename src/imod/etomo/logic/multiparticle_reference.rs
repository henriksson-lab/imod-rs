//! `IMOD/Etomo/src/etomo/logic/MultiparticleReference.java`.
//!
//! Converts between the multiparticle reference level stored in the .prm file and the
//! list of particle counts displayed on the screen.  The particle counts are 2^k,
//! where k is the reference level.

use crate::imod::etomo::r#type::etomo_number::EtomoNumber;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java private static final `MIN_LEVEL`.
const MIN_LEVEL: i32 = 2;
/// Java private static final `MAX_LEVEL`.
const MAX_LEVEL: i32 = 10;
/// Java `DEFAULT_LEVEL`.
pub const DEFAULT_LEVEL: i32 = 5;

/// Java static `getNumEntries()`.
pub fn get_num_entries() -> i32 {
    MAX_LEVEL - MIN_LEVEL + 1
}

/// Java static `getParticleCount(int)`.  Converts the (corrected) index to a level and
/// then to a particle count.
pub fn get_particle_count(index: i32) -> i32 {
    let num_entries = get_num_entries();
    let mut index = index;
    if index < 0 {
        index = 0;
    } else if index >= num_entries {
        index = num_entries - 1;
    }
    2f64.powf((index + MIN_LEVEL) as f64) as i32
}

/// Java static `getDefaultIndex()`.  Converts the default level to an index.
pub fn get_default_index() -> i32 {
    DEFAULT_LEVEL - MIN_LEVEL
}

/// Java static `convertLevelToIndex(String, EtomoNumber)`.  Calculates the index of
/// the reference in the list of particle counts.  Returns true for a known value (no
/// warning).
pub fn convert_level_to_index_string(level: Option<&str>, index: &mut EtomoNumber) -> bool {
    let Some(level) = level else {
        index.set_int(get_default_index());
        return true;
    };
    let mut n_level = EtomoNumber::new();
    n_level.set_ceiling(MAX_LEVEL);
    n_level.set_floor(MIN_LEVEL);
    n_level.set_string(Some(level));
    if !n_level.is_valid() {
        index.set_int(get_default_index());
        return true;
    }
    // calculate index
    index.set_int(n_level.get_int() - MIN_LEVEL);
    if n_level.is_value_altered() {
        return false;
    }
    true
}

/// Java static `convertLevelToIndex(int)`.
pub fn convert_level_to_index_int(level: i32) -> i32 {
    let mut n_level = EtomoNumber::new();
    n_level.set_ceiling(MAX_LEVEL);
    n_level.set_floor(MIN_LEVEL);
    n_level.set_int(level);
    // calculate index
    n_level.get_int() - MIN_LEVEL
}

/// Java static `convertIndexToLevel(int)`.  Returns a reference.
pub fn convert_index_to_level(index: i32) -> String {
    let mut n_level = EtomoNumber::new();
    n_level.set_ceiling(MAX_LEVEL);
    n_level.set_floor(MIN_LEVEL);
    n_level.set_int(index + MIN_LEVEL);
    n_level.to_string()
}
