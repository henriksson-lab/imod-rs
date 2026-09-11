//! `IMOD/Etomo/src/etomo/ui/swing/Panel.java`.
//!
//! `JPanel` is a native-widget boundary.  This unit retains the one source
//! override: maximum dimensions are scaled at the point Java scales them.
#![allow(dead_code)]

/// Java `java.awt.Dimension` values used by `Panel.setMaximumSize`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Dimension {
    pub width: i32,
    pub height: i32,
}

/// Java package-private final `Panel`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Panel {
    /// The Swing superclass's stored maximum size.
    pub maximum_size: Option<Dimension>,
    /// `UIUtilities.scaleByFontSize` is a direct user-preferences boundary.
    /// `None` is Java's normal pre-preferences-loaded path, which preserves
    /// both dimensions unchanged.
    pub user_font_size: Option<i32>,
}

impl Panel {
    /// Java `setMaximumSize(Dimension)`.
    pub fn set_maximum_size(&mut self, mut maximum_size: Option<Dimension>) {
        if let Some(size) = maximum_size.as_mut() {
            if let Some(font_size) = self.user_font_size {
                if size.width > 0 {
                    size.width =
                        ((font_size as f32 / 14.0 * size.width as f32).round() as i32).max(1);
                }
                if size.height > 0 {
                    size.height =
                        ((font_size as f32 / 14.0 * size.height as f32).round() as i32).max(1);
                }
            }
        }
        self.maximum_size = maximum_size;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_maximum_size_scales_each_positive_dimension() {
        let mut panel = Panel {
            user_font_size: Some(21),
            ..Default::default()
        };
        panel.set_maximum_size(Some(Dimension {
            width: 8,
            height: 10,
        }));
        assert_eq!(
            panel.maximum_size,
            Some(Dimension {
                width: 12,
                height: 15,
            })
        );
    }
}
