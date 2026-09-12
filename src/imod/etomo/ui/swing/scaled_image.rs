//! `IMOD/Etomo/src/etomo/ui/swing/ScaledImage.java`.
#![allow(dead_code)]

/// Java `MediaTracker` image-loading statuses used by `ScaledImage`.
pub const COMPLETE: i32 = 8;

/// The Java singleton image identities and their resource file names.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScaledImageName {
    OpenFile,
    OpenFilePeet,
    OpenFileFool,
    OpenFileRed,
    Clear,
    ClearBlue,
    ClearRed,
    SmallX,
    SmallXPressed,
    SmallXRollover,
    Imod,
    ImodPressed,
    ImodRollover,
    Etomo,
    EtomoPressed,
    EtomoRollover,
    EtomoLog,
    EtomoLogPressed,
    EtomoLogRollover,
    BrtLog,
    BrtLogPressed,
    BrtLogRollover,
}

impl ScaledImageName {
    /// Java private `ScaledImage(String)` file-name argument.
    pub fn file_name(self) -> &'static str {
        match self {
            Self::OpenFile => "openFile.gif",
            Self::OpenFilePeet => "openFilePeet.png",
            Self::OpenFileFool => "openFileFool.png",
            Self::OpenFileRed => "openFileRed.png",
            Self::Clear => "clear.png",
            Self::ClearBlue => "clearBlue.png",
            Self::ClearRed => "clearRed.png",
            Self::SmallX => "smallX.png",
            Self::SmallXPressed => "smallX-pressed.png",
            Self::SmallXRollover => "smallX-rollover.png",
            Self::Imod => "b3dicon.png",
            Self::ImodPressed => "b3dicon-pressed.png",
            Self::ImodRollover => "b3dicon-rollover.png",
            Self::Etomo => "etomoicon.png",
            Self::EtomoPressed => "etomoicon-pressed.png",
            Self::EtomoRollover => "etomoicon-rollover.png",
            Self::EtomoLog => "projlogicon.png",
            Self::EtomoLogPressed => "projlogicon-pressed.png",
            Self::EtomoLogRollover => "projlogicon-rollover.png",
            Self::BrtLog => "logicon.png",
            Self::BrtLogPressed => "logicon-pressed.png",
            Self::BrtLogRollover => "logicon-rollover.png",
        }
    }
}

/// Java final `ScaledImage`; toolkit loading and pixel scaling remain a native GUI boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScaledImage {
    pub file_name: &'static str,
    pub orig_image_present: bool,
    pub orig_image_load_status: Option<i32>,
    pub image_present: bool,
    pub scale: Option<bool>,
    pub ratio_milli: Option<i32>,
}

impl ScaledImage {
    /// Java private `ScaledImage(String)`.
    pub fn new(name: ScaledImageName, resource_present: bool, load_status: Option<i32>) -> Self {
        Self {
            file_name: name.file_name(),
            orig_image_present: resource_present,
            orig_image_load_status: resource_present.then_some(load_status.unwrap_or(COMPLETE)),
            image_present: false,
            scale: None,
            ratio_milli: None,
        }
    }
    /// Java private static synchronized `init()` state transition.
    pub fn init(
        &mut self,
        user_preference_loaded: bool,
        font_size: i32,
        above: i32,
        below: i32,
    ) -> bool {
        if self.scale.is_some() {
            return true;
        }
        if !user_preference_loaded {
            return false;
        }
        if font_size > above || font_size < below {
            self.scale = Some(true);
            self.ratio_milli = Some((font_size * 1000) / above);
        } else {
            self.scale = Some(false);
        }
        true
    }
    /// Java `getImage(ImageObserver)`, retaining loading/scaling decisions but not native pixels.
    pub fn get_image(
        &mut self,
        user_preference_loaded: bool,
        font_size: i32,
        above: i32,
        below: i32,
    ) -> bool {
        if self.image_present {
            return true;
        }
        if !self.orig_image_present {
            return false;
        }
        if self.orig_image_load_status != Some(COMPLETE) {
            return true;
        }
        if self.scale.is_none() && !self.init(user_preference_loaded, font_size, above, below) {
            return true;
        }
        self.image_present = true;
        self.orig_image_present = false;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn resources_and_delayed_preference_initialization_follow_source() {
        let mut image = ScaledImage::new(ScaledImageName::Imod, true, Some(COMPLETE));
        assert_eq!(image.file_name, "b3dicon.png");
        assert!(image.get_image(false, 12, 12, 10));
        assert!(!image.image_present);
        assert!(image.get_image(true, 18, 12, 10));
        assert!(image.image_present);
        assert_eq!(image.ratio_milli, Some(1500));
        assert!(!image.orig_image_present);
    }
}
