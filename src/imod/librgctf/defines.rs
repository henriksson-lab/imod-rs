//! Translation of `IMOD/librgctf/defines.h` constants.

pub const INTEGER_DATABASE_VERSION: i32 = 1;
pub const CISTEM_VERSION_TEXT: &str = "1.0.0-beta";
pub const START_PORT: u16 = 3000;
pub const END_PORT: u16 = 5000;

/// `#define PI 3.14159265359` (`defines.h:5`).
///
/// The macro expands to a **`double`** literal, so every C++ expression that
/// mentions `PI` is evaluated in double precision and only narrowed where it
/// is stored into a `float`.  Declaring this `f32` would change the value of
/// every CTF phase, azimuth and mask edge in the library, so it is `f64` and
/// each call site narrows exactly where the source does.
pub const PI: f64 = 3.14159265359;

pub const ANSI_COLOR_RED: &str = "\x1b[31m";
pub const ANSI_COLOR_GREEN: &str = "\x1b[32m";
pub const ANSI_COLOR_YELLOW: &str = "\x1b[33m";
pub const ANSI_COLOR_BLUE: &str = "\x1b[34m";
pub const ANSI_COLOR_MAGENTA: &str = "\x1b[35m";
pub const ANSI_COLOR_CYAN: &str = "\x1b[36m";
pub const ANSI_COLOR_RESET: &str = "\x1b[0m";
/// `ANSI_UNDERLINE` is written `"\e[4m"` in the header, a GCC extension for
/// ESC; the byte is the same 0x1b.
pub const ANSI_UNDERLINE: &str = "\x1b[4m";
pub const ANSI_UNDERLINE_OFF: &str = "\x1b[24m";
pub const ANSI_BLINK_SLOW: &str = "\x1b[5m";
pub const ANSI_BLINK_OFF: &str = "\x1b[25m";

pub const SCALED_IMAGE_SIZE: usize = 1200;
