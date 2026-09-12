//! `IMOD/Etomo/src/etomo/ui/swing/Colors.java`.
//!
//! `ColorUIResource` is a Swing look-and-feel marker around an RGB value.  The
//! renderer is the GUI boundary; this unit preserves the source RGB values and
//! lazy static cache instead of introducing another theme system.
#![allow(dead_code)]

use std::sync::{LazyLock, Mutex};

use super::ui_utilities::Color;
use crate::imod::etomo::util::utilities;

/// Java `ColorUIResource` at the native renderer boundary.
pub type ColorUiResource = Color;

pub const CELL_FOREGROUND: ColorUiResource = Color {
    red: 0,
    green: 0,
    blue: 0,
};
pub const CELL_NOT_IN_USE_FOREGROUND: ColorUiResource = Color {
    red: 102,
    green: 102,
    blue: 102,
};
pub const CELL_ERROR_BACKGROUND: ColorUiResource = Color {
    red: 255,
    green: 204,
    blue: 204,
};
pub const CELL_ERROR_BACKGROUND_NOT_EDITABLE: ColorUiResource = Color {
    red: 230,
    green: 184,
    blue: 184,
};
pub const BACKGROUND: ColorUiResource = Color {
    red: 255,
    green: 255,
    blue: 255,
};
pub const WARNING_BACKGROUND: ColorUiResource = Color {
    red: 255,
    green: 255,
    blue: 204,
};
pub const WARNING_BACKGROUND_NOT_EDITABLE: ColorUiResource = Color {
    red: 230,
    green: 230,
    blue: 184,
};
pub const HIGHLIGHT_BACKGROUND: ColorUiResource = Color {
    red: 204,
    green: 255,
    blue: 255,
};
pub const HIGHLIGHT_BACKGROUND_NOT_EDITABLE: ColorUiResource = Color {
    red: 184,
    green: 230,
    blue: 230,
};
pub const RUN_HIGHLIGHT_BACKGROUND: ColorUiResource = Color {
    red: 204,
    green: 255,
    blue: 204,
};
pub const RUN_HIGHLIGHT_BACKGROUND_NOT_EDITABLE: ColorUiResource = Color {
    red: 184,
    green: 230,
    blue: 184,
};
pub const VIOLET: ColorUiResource = Color {
    red: 199,
    green: 173,
    blue: 224,
};
pub const FOREGROUND: ColorUiResource = Color {
    red: 0,
    green: 0,
    blue: 0,
};
pub const BACKGROUND_GREYOUT: ColorUiResource = Color {
    red: 25,
    green: 25,
    blue: 25,
};
pub const CELL_DISABLED_FOREGROUND: ColorUiResource = Color {
    red: 120,
    green: 120,
    blue: 120,
};
pub const AVAILABLE_BACKGROUND: Color = Color {
    red: 224,
    green: 240,
    blue: 255,
};
pub const AVAILABLE_BORDER: Color = Color {
    red: 153,
    green: 204,
    blue: 255,
};
pub const FIELD_HIGHLIGHT: Color = Color {
    red: 0,
    green: 0,
    blue: 185,
};
const BACKGROUND_ADJUSTMENT: i32 = 20;
pub const HEADER_BACKGROUND: ColorUiResource = Color {
    red: 239,
    green: 239,
    blue: 239,
};

static BACKGROUND_A: LazyLock<Mutex<Option<Color>>> = LazyLock::new(|| Mutex::new(None));
static BACKGROUND_B: LazyLock<Mutex<Option<Color>>> = LazyLock::new(|| Mutex::new(None));
static BACKGROUND_JOIN: LazyLock<Mutex<Option<Color>>> = LazyLock::new(|| Mutex::new(None));
static BACKGROUND_PARALLEL: LazyLock<Mutex<Option<Color>>> = LazyLock::new(|| Mutex::new(None));
static BACKGROUND_BATCHRUNTOMO: LazyLock<Mutex<Option<Color>>> = LazyLock::new(|| Mutex::new(None));
static BACKGROUND_SERIAL_SECTIONS: LazyLock<Mutex<Option<Color>>> =
    LazyLock::new(|| Mutex::new(None));
static BACKGROUND_TOOLS: LazyLock<Mutex<Option<Color>>> = LazyLock::new(|| Mutex::new(None));
static CELL_NOT_EDITABLE_BACKGROUND: LazyLock<Mutex<Option<ColorUiResource>>> =
    LazyLock::new(|| Mutex::new(None));

/// Rust namespace for Java's final static `Colors` class.
pub struct Colors;

impl Colors {
    /// Java `getBackgroundA()`.
    pub fn get_background_a() -> Color {
        let mut value = BACKGROUND_A.lock().expect("backgroundA mutex poisoned");
        if value.is_none() {
            *value = Some(if !*utilities::APRIL_FOOLS {
                Color {
                    red: 173,
                    green: 199,
                    blue: 224,
                }
            } else {
                Color {
                    red: 163,
                    green: 214,
                    blue: 247,
                }
            });
        }
        value.expect("backgroundA initialized")
    }

    /// Java `getBackgroundB()`.
    pub fn get_background_b() -> Color {
        let mut value = BACKGROUND_B.lock().expect("backgroundB mutex poisoned");
        if value.is_none() {
            *value = Some(if !*utilities::APRIL_FOOLS {
                Color {
                    red: 173,
                    green: 224,
                    blue: 199,
                }
            } else {
                Color {
                    red: 255,
                    green: 216,
                    blue: 141,
                }
            });
        }
        value.expect("backgroundB initialized")
    }

    /// Java `getBackgroundJoin()`.
    pub fn get_background_join() -> Color {
        let mut value = BACKGROUND_JOIN
            .lock()
            .expect("backgroundJoin mutex poisoned");
        if value.is_none() {
            *value = Some(if !*utilities::APRIL_FOOLS {
                VIOLET
            } else {
                Color {
                    red: 162,
                    green: 167,
                    blue: 255,
                }
            });
        }
        value.expect("backgroundJoin initialized")
    }

    /// Java `getBackgroundParallel()`.
    pub fn get_background_parallel() -> Color {
        let mut value = BACKGROUND_PARALLEL
            .lock()
            .expect("backgroundParallel mutex poisoned");
        if value.is_none() {
            *value = Some(if !*utilities::APRIL_FOOLS {
                Color {
                    red: 186,
                    green: 224,
                    blue: 173,
                }
            } else {
                Color {
                    red: 255,
                    green: 253,
                    blue: 216,
                }
            });
        }
        value.expect("backgroundParallel initialized")
    }

    /// Java `getBackgroundBatchruntomo()`.
    pub fn get_background_batchruntomo() -> Color {
        let mut value = BACKGROUND_BATCHRUNTOMO
            .lock()
            .expect("backgroundBatchruntomo mutex poisoned");
        if value.is_none() {
            *value = Some(if !*utilities::APRIL_FOOLS {
                VIOLET
            } else {
                Color {
                    red: 255,
                    green: 239,
                    blue: 192,
                }
            });
        }
        value.expect("backgroundBatchruntomo initialized")
    }

    /// Java `getBackgroundSerialSections()`.
    pub fn get_background_serial_sections() -> Color {
        let mut value = BACKGROUND_SERIAL_SECTIONS
            .lock()
            .expect("backgroundSerialSections mutex poisoned");
        if value.is_none() {
            *value = Some(if !*utilities::APRIL_FOOLS {
                Color {
                    red: 218,
                    green: 232,
                    blue: 250,
                }
            } else {
                Color {
                    red: 194,
                    green: 247,
                    blue: 159,
                }
            });
        }
        value.expect("backgroundSerialSections initialized")
    }

    /// Java `getBackgroundTools()`.
    pub fn get_background_tools() -> Color {
        let mut value = BACKGROUND_TOOLS
            .lock()
            .expect("backgroundTools mutex poisoned");
        if value.is_none() {
            *value = Some(if !*utilities::APRIL_FOOLS {
                Color {
                    red: 173,
                    green: 212,
                    blue: 224,
                }
            } else {
                Color {
                    red: 52,
                    green: 130,
                    blue: 218,
                }
            });
        }
        value.expect("backgroundTools initialized")
    }

    /// Java `getCellNotEditableBackground()`.
    pub fn get_cell_not_editable_background() -> ColorUiResource {
        let mut value = CELL_NOT_EDITABLE_BACKGROUND
            .lock()
            .expect("cellNotEditableBackground mutex poisoned");
        if value.is_none() {
            *value = Some(Self::subtract_color(BACKGROUND, BACKGROUND_GREYOUT));
        }
        value.expect("cellNotEditableBackground initialized")
    }

    /// Java `subtractColor(Color, Color)`.
    pub fn subtract_color(color: Color, subtract_color: Color) -> ColorUiResource {
        Color {
            red: color.red - subtract_color.red,
            green: color.green - subtract_color.green,
            blue: color.blue - subtract_color.blue,
        }
    }

    /// Java private `addColor(Color, Color)`.
    fn add_color(color: Color, subtract_color: Color) -> ColorUiResource {
        Color {
            red: color.red + subtract_color.red,
            green: color.green + subtract_color.green,
            blue: color.blue + subtract_color.blue,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_retain_source_rgb_values() {
        assert_eq!(
            CELL_ERROR_BACKGROUND,
            Color {
                red: 255,
                green: 204,
                blue: 204
            }
        );
        assert_eq!(
            WARNING_BACKGROUND,
            Color {
                red: 255,
                green: 255,
                blue: 204
            }
        );
        assert_eq!(
            VIOLET,
            Color {
                red: 199,
                green: 173,
                blue: 224
            }
        );
        assert_eq!(
            FIELD_HIGHLIGHT,
            Color {
                red: 0,
                green: 0,
                blue: 185
            }
        );
    }

    #[test]
    fn cached_backgrounds_follow_source_date_branch() {
        let (a, b, join, parallel, batchruntomo, serial_sections, tools) =
            if *utilities::APRIL_FOOLS {
                (
                    Color {
                        red: 163,
                        green: 214,
                        blue: 247,
                    },
                    Color {
                        red: 255,
                        green: 216,
                        blue: 141,
                    },
                    Color {
                        red: 162,
                        green: 167,
                        blue: 255,
                    },
                    Color {
                        red: 255,
                        green: 253,
                        blue: 216,
                    },
                    Color {
                        red: 255,
                        green: 239,
                        blue: 192,
                    },
                    Color {
                        red: 194,
                        green: 247,
                        blue: 159,
                    },
                    Color {
                        red: 52,
                        green: 130,
                        blue: 218,
                    },
                )
            } else {
                (
                    Color {
                        red: 173,
                        green: 199,
                        blue: 224,
                    },
                    Color {
                        red: 173,
                        green: 224,
                        blue: 199,
                    },
                    VIOLET,
                    Color {
                        red: 186,
                        green: 224,
                        blue: 173,
                    },
                    VIOLET,
                    Color {
                        red: 218,
                        green: 232,
                        blue: 250,
                    },
                    Color {
                        red: 173,
                        green: 212,
                        blue: 224,
                    },
                )
            };
        assert_eq!(Colors::get_background_a(), a);
        assert_eq!(Colors::get_background_b(), b);
        assert_eq!(Colors::get_background_join(), join);
        assert_eq!(Colors::get_background_parallel(), parallel);
        assert_eq!(Colors::get_background_batchruntomo(), batchruntomo);
        assert_eq!(Colors::get_background_serial_sections(), serial_sections);
        assert_eq!(Colors::get_background_tools(), tools);
    }

    #[test]
    fn source_color_arithmetic_and_cached_not_editable_color_are_preserved() {
        assert_eq!(
            Colors::subtract_color(BACKGROUND, BACKGROUND_GREYOUT),
            Color {
                red: 230,
                green: 230,
                blue: 230
            }
        );
        assert_eq!(
            Colors::add_color(BACKGROUND_GREYOUT, BACKGROUND_GREYOUT),
            Color {
                red: 50,
                green: 50,
                blue: 50
            }
        );
        assert_eq!(
            Colors::get_cell_not_editable_background(),
            Color {
                red: 230,
                green: 230,
                blue: 230
            }
        );
    }
}
