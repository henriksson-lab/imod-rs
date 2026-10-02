//! `IMOD/Etomo/src/etomo/ui/swing/Minibutton.java`.
//!
//! A small unfocusable `JButton`: square (optionally with an icon), or round with
//! its own fill, rollover, pressed and outline colours.
//!
//! Java `final class Minibutton extends JButton`: the Swing button is held as
//! `button`.  Painting and geometry are not modelled by the Swing stand-in, so
//! `paintComponent`, `paintBorder`, `contains(int, int)` and the `shape` field have
//! no Rust counterpart; the round-button colours they read are kept.  An `Icon` is
//! represented by its image name, as in `complete_icon.rs`.

use std::cell::Cell;
use std::rc::Rc;

use crate::imod::etomo::jdk::{Color, JComponent};

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java package-private `final class Minibutton extends JButton`.
pub struct Minibutton {
    /// The `JButton` this class extends.
    button: Rc<JComponent>,
    /// Java final `round` (read only by the painting methods).
    #[allow(dead_code)]
    round: bool,
    /// Java `rolloverColor` (painting only).
    #[allow(dead_code)]
    rollover_color: Cell<Option<Color>>,
    /// Java `pressedColor` (painting only).
    #[allow(dead_code)]
    pressed_color: Cell<Option<Color>>,
    /// Java `outlineColor` (painting only).
    #[allow(dead_code)]
    outline_color: Cell<Option<Color>>,
}

impl Minibutton {
    /// Java private `Minibutton(String, Icon)`.
    fn new_string_icon(text: Option<&str>, _icon: Option<&str>) -> Minibutton {
        // super(icon)
        let button = JComponent::new_button("");
        // Swing painting: the icon is the button's icon.
        let instance = Minibutton {
            button,
            round: false,
            rollover_color: Cell::new(None),
            pressed_color: Cell::new(None),
            outline_color: Cell::new(None),
        };
        instance.button.set_text(text.unwrap_or(""));
        // Swing layout: setFocusable(false).
        instance
    }

    /// Java private `Minibutton(String, boolean, Color, Color, Color, Color)`.
    fn new_string_boolean_color_color_color_color(
        text: Option<&str>,
        round: bool,
        color: Option<Color>,
        rollover_color: Option<Color>,
        pressed_color: Option<Color>,
        outline_color: Option<Color>,
    ) -> Minibutton {
        // super()
        let instance = Minibutton {
            button: JComponent::new_button(""),
            round,
            rollover_color: Cell::new(None),
            pressed_color: Cell::new(None),
            outline_color: Cell::new(None),
        };
        instance.button.set_text(text.unwrap_or(""));
        // Swing layout: setFocusable(false).
        if round {
            if outline_color.is_some() {
                instance.outline_color.set(outline_color);
            } else {
                instance.outline_color.set(instance.button.get_foreground());
            }
            // Swing painting: setBackground(color) when color != null; otherwise
            // color = getBackground() (backgrounds are not modelled, so it stays
            // null).
            if rollover_color.is_some() {
                instance.rollover_color.set(rollover_color);
            } else if color.is_some() {
                // Swing painting: this.rolloverColor = color.brighter().
                instance.rollover_color.set(None);
            } else {
                instance.rollover_color.set(None);
            }
            if pressed_color.is_some() {
                instance.pressed_color.set(pressed_color);
            } else if color.is_some() {
                // Swing painting: this.pressedColor = color.darker().
                instance.pressed_color.set(None);
            } else {
                instance.pressed_color.set(None);
            }
            // Swing painting: setContentAreaFilled(false).
        }
        instance
    }

    /// Java static `getSquareInstance(String, Border)`.  The border is painting.
    pub fn get_square_instance_string_border(label: Option<&str>) -> Rc<Minibutton> {
        let instance = Minibutton::new_string_icon(label, None);
        // Swing painting: instance.setBorder(border).
        instance.set_size_void();
        Rc::new(instance)
    }

    /// Java static `getSquareInstance(Icon, Border)`.  The border is painting.
    pub fn get_square_instance_icon_border(icon: Option<&str>) -> Rc<Minibutton> {
        let instance = Minibutton::new_string_icon(None, icon);
        // Swing painting: instance.setBorder(border).
        instance.set_size_void();
        Rc::new(instance)
    }

    /// Java static `getSquareInstance(Border)`.  The border is painting.
    pub fn get_square_instance_border() -> Rc<Minibutton> {
        let instance = Minibutton::new_string_icon(None, None);
        // Swing painting: instance.setBorder(border).
        instance.set_size_void();
        Rc::new(instance)
    }

    /// Java static `getRoundInstance(String, boolean, boolean, Color, Color, Color,
    /// Color)`.
    pub fn get_round_instance(
        label: Option<&str>,
        italics: bool,
        _small: bool,
        color: Option<Color>,
        rollover_color: Option<Color>,
        pressed_color: Option<Color>,
        outline_color: Option<Color>,
    ) -> Rc<Minibutton> {
        let mut label_size = 1;
        if let Some(label) = label {
            // String.length(): UTF-16 code units.
            label_size = label.encode_utf16().count();
        }
        // Java string concatenation writes a null label as "null".
        let text = format!(
            "{}{}{}{}",
            if italics { "<html><i>" } else { "" },
            if label_size > 1 { " " } else { "" },
            label.unwrap_or("null"),
            if label_size > 1 { " " } else { "" }
        );
        let instance = Minibutton::new_string_boolean_color_color_color_color(
            Some(&text),
            true,
            color,
            rollover_color,
            pressed_color,
            outline_color,
        );
        // Swing painting: instance.setBorder(small ? BorderFactory.createEmptyBorder()
        // : BorderFactory.createEtchedBorder()).
        instance.set_size_void();
        Rc::new(instance)
    }

    /// Java static `getBlueInstance(String, boolean, boolean)`.
    pub fn get_blue_instance(label: Option<&str>, italics: bool, small: bool) -> Rc<Minibutton> {
        Minibutton::get_round_instance(
            label,
            italics,
            small,
            Some((176, 248, 255)),
            Some((203, 232, 255)),
            Some((134, 189, 255)),
            Some((15, 4, 75)),
        )
    }

    /// Java static `getGreenInstance(String, boolean, boolean)`.
    pub fn get_green_instance(label: Option<&str>, italics: bool, small: bool) -> Rc<Minibutton> {
        Minibutton::get_round_instance(
            label,
            italics,
            small,
            Some((188, 254, 186)),
            Some((231, 254, 245)),
            Some((86, 226, 138)),
            Some((0, 40, 2)),
        )
    }

    /// Java `setIcon(Image)`.
    pub fn set_icon(&self, image: Option<&str>) {
        if image.is_some() {
            // Swing painting: setIcon(new ImageIcon(image)).
        }
    }

    /// Java `setSize()`.
    pub fn set_size_void(&self) {
        // Swing layout: size = getPreferredSize(); if (size.width < size.height)
        // size.width = size.height; setSize(size).
    }

    /// Java `@Override setSize(Dimension)`.
    pub fn set_size_dimension(&self) {
        // Swing layout: setPreferredSize(size); setMaximumSize(size).
    }

    /// The `JButton` itself (Java uses `this` as the Component).
    pub fn get_component(&self) -> Rc<JComponent> {
        self.button.clone()
    }
}
