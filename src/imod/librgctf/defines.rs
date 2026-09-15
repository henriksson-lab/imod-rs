//! Translation of `IMOD/librgctf/defines.h` constants.

pub const INTEGER_DATABASE_VERSION: i32 = 1;
pub const CISTEM_VERSION_TEXT: &str = "1.0.0-beta";
pub const START_PORT: u16 = 3000;
pub const END_PORT: u16 = 5000;
pub const PI: f32 = 3.141_592_653_59;
pub const ANSI_COLOR_RED: &str = "\x1b[31m";
pub const ANSI_COLOR_GREEN: &str = "\x1b[32m";
pub const ANSI_COLOR_YELLOW: &str = "\x1b[33m";
pub const ANSI_COLOR_BLUE: &str = "\x1b[34m";
pub const ANSI_COLOR_MAGENTA: &str = "\x1b[35m";
pub const ANSI_COLOR_CYAN: &str = "\x1b[36m";
pub const ANSI_COLOR_RESET: &str = "\x1b[0m";
pub const ANSI_UNDERLINE: &str = "\x1b[4m";
pub const ANSI_UNDERLINE_OFF: &str = "\x1b[24m";
pub const ANSI_BLINK_SLOW: &str = "\x1b[5m";
pub const ANSI_BLINK_OFF: &str = "\x1b[25m";
pub const SCALED_IMAGE_SIZE: usize = 1200;
