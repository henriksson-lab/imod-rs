//! `IMOD/Etomo/src/etomo/type/UITestFieldType.java`.
//!
//! The Java class is an enumeration of static instances; here they are
//! associated constants (`UITestFieldType::BUTTON`), mirrored as module
//! constants (`ui_test_field_type::BUTTON`).  Java `componentClass` is kept as
//! the class's simple name.

/// Java `UITestFieldType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UITestFieldType {
    string: &'static str,
    descr: &'static str,
    /// Java `componentClass` (simple name).
    component_class: &'static str,
    named: bool,
    use_index: bool,
    unlimited_segments: bool,
    /// A field on the dialog that may receive a simulated mouse click or a doClick call.
    /// Does not include popups or file choosers.
    clickable_field: bool,
    /// Field is tested to see if it was successfully clicked.
    has_click_test: bool,
}

impl UITestFieldType {
    /// Java private constructor.
    #[allow(clippy::too_many_arguments)]
    const fn new(
        string: &'static str,
        component_class: &'static str,
        named: bool,
        use_index: bool,
        unlimited_segments: bool,
        descr: &'static str,
        clickable_field: bool,
        has_click_test: bool,
    ) -> UITestFieldType {
        UITestFieldType {
            string,
            descr,
            component_class,
            named,
            use_index,
            unlimited_segments,
            clickable_field,
            has_click_test,
        }
    }

    pub const BUTTON: UITestFieldType = UITestFieldType::new(
        "bn",
        "AbstractButton",
        true,
        true,
        true,
        "button",
        true,
        false,
    );
    pub const CHECK_BOX: UITestFieldType =
        UITestFieldType::new("cb", "JCheckBox", true, true, true, "check box", true, true);
    pub const CHECK_BOX_MENU_ITEM: UITestFieldType = UITestFieldType::new(
        "cbmn",
        "JCheckBoxMenuItem",
        true,
        false,
        true,
        "check box menu item",
        true,
        false,
    );
    pub const COMBO_BOX: UITestFieldType = UITestFieldType::new(
        "cbb",
        "JComboBox",
        true,
        true,
        true,
        "combo-box",
        true,
        false,
    );
    pub const FILE_CHOOSER: UITestFieldType = UITestFieldType::new(
        "file-chooser",
        "JFileChooser",
        false,
        false,
        true,
        "file chooser",
        false,
        false,
    );
    pub const HEADER_CELL: UITestFieldType = UITestFieldType::new(
        "hc",
        "AbstractButton",
        true,
        true,
        true,
        "header cell",
        false,
        false,
    );
    pub const IMAGE_FILE_CHOOSER: UITestFieldType = UITestFieldType::new(
        "image-file-chooser",
        "JFileChooser",
        false,
        false,
        true,
        "image file chooser",
        false,
        false,
    );
    pub const MENU_ITEM: UITestFieldType = UITestFieldType::new(
        "mn",
        "JMenuItem",
        true,
        false,
        true,
        "menu item",
        true,
        false,
    );
    pub const MINI_BUTTON: UITestFieldType = UITestFieldType::new(
        "mb",
        "AbstractButton",
        true,
        true,
        true,
        "minibutton",
        true,
        true,
    );
    pub const PANEL: UITestFieldType =
        UITestFieldType::new("pnl", "JPanel", true, false, true, "panel", false, false);
    pub const POPUP: UITestFieldType = UITestFieldType::new(
        "popup",
        "JOptionPane",
        false,
        false,
        true,
        "popup",
        false,
        false,
    );
    pub const RADIO_BUTTON: UITestFieldType = UITestFieldType::new(
        "rb",
        "JRadioButton",
        true,
        true,
        true,
        "radio button",
        true,
        true,
    );
    // Spinner is only clicked when the values "up" and "down" are used. The resulting value
    // is not simple to predict.
    pub const SPINNER: UITestFieldType =
        UITestFieldType::new("sp", "JSpinner", true, true, true, "spinner", true, false);
    pub const TAB: UITestFieldType =
        UITestFieldType::new("tb", "JTabbedPane", true, false, true, "tab", true, false);
    pub const TEXT_FIELD: UITestFieldType = UITestFieldType::new(
        "tf",
        "JTextField",
        true,
        true,
        true,
        "text field",
        false,
        false,
    );
    pub const CONTROL_BUTTON: UITestFieldType = UITestFieldType::new(
        "ctb",
        "AbstractButton",
        true,
        true,
        true,
        "control button",
        true,
        false,
    );
    pub const CONTROL_TOGGLE_BUTTON: UITestFieldType = UITestFieldType::new(
        "cttb",
        "AbstractButton",
        true,
        true,
        true,
        "control toggle button",
        true,
        false,
    );
    pub const LABEL: UITestFieldType =
        UITestFieldType::new("l", "JLabel", true, true, false, "label", false, false);
    pub const TEXT_AREA: UITestFieldType = UITestFieldType::new(
        "ta",
        "JTextArea",
        true,
        true,
        true,
        "text area",
        false,
        false,
    );

    /// Java `getInstance(String)`.
    pub fn get_instance(string: Option<&str>) -> Option<UITestFieldType> {
        let string = string?;
        for instance in [
            Self::BUTTON,
            Self::CHECK_BOX,
            Self::COMBO_BOX,
            Self::FILE_CHOOSER,
            Self::HEADER_CELL,
            Self::IMAGE_FILE_CHOOSER,
            Self::MENU_ITEM,
            Self::MINI_BUTTON,
            Self::PANEL,
            Self::POPUP,
            Self::RADIO_BUTTON,
            Self::SPINNER,
            Self::TAB,
            Self::TEXT_FIELD,
            Self::CONTROL_BUTTON,
            Self::CONTROL_TOGGLE_BUTTON,
            Self::LABEL,
            Self::TEXT_AREA,
        ] {
            if string == instance.string {
                return Some(instance);
            }
        }
        None
    }

    /// Java `isClickableField()`.
    pub fn is_clickable_field(&self) -> bool {
        self.clickable_field
    }
    /// Java `hasClickTest()`.
    pub fn has_click_test(&self) -> bool {
        self.has_click_test
    }
    /// Java `isUnlimitedSegments()`.
    pub fn is_unlimited_segments(&self) -> bool {
        self.unlimited_segments
    }
    /// Java `getDescr()`.
    pub fn get_descr(&self) -> &'static str {
        self.descr
    }
    /// Java `getComponentClass()` (simple class name).
    pub fn get_component_class(&self) -> &'static str {
        self.component_class
    }
    /// Java `isNamed()`.
    pub fn is_named(&self) -> bool {
        self.named
    }
    /// Java `isUseIndex()`.
    pub fn is_use_index(&self) -> bool {
        self.use_index
    }
}

impl std::fmt::Display for UITestFieldType {
    /// Java `toString()`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.string)
    }
}

pub const BUTTON: UITestFieldType = UITestFieldType::BUTTON;
pub const CHECK_BOX: UITestFieldType = UITestFieldType::CHECK_BOX;
pub const CHECK_BOX_MENU_ITEM: UITestFieldType = UITestFieldType::CHECK_BOX_MENU_ITEM;
pub const COMBO_BOX: UITestFieldType = UITestFieldType::COMBO_BOX;
pub const FILE_CHOOSER: UITestFieldType = UITestFieldType::FILE_CHOOSER;
pub const HEADER_CELL: UITestFieldType = UITestFieldType::HEADER_CELL;
pub const IMAGE_FILE_CHOOSER: UITestFieldType = UITestFieldType::IMAGE_FILE_CHOOSER;
pub const MENU_ITEM: UITestFieldType = UITestFieldType::MENU_ITEM;
pub const MINI_BUTTON: UITestFieldType = UITestFieldType::MINI_BUTTON;
pub const PANEL: UITestFieldType = UITestFieldType::PANEL;
pub const POPUP: UITestFieldType = UITestFieldType::POPUP;
pub const RADIO_BUTTON: UITestFieldType = UITestFieldType::RADIO_BUTTON;
pub const SPINNER: UITestFieldType = UITestFieldType::SPINNER;
pub const TAB: UITestFieldType = UITestFieldType::TAB;
pub const TEXT_FIELD: UITestFieldType = UITestFieldType::TEXT_FIELD;
pub const CONTROL_BUTTON: UITestFieldType = UITestFieldType::CONTROL_BUTTON;
pub const CONTROL_TOGGLE_BUTTON: UITestFieldType = UITestFieldType::CONTROL_TOGGLE_BUTTON;
pub const LABEL: UITestFieldType = UITestFieldType::LABEL;
pub const TEXT_AREA: UITestFieldType = UITestFieldType::TEXT_AREA;
