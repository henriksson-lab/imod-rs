//! `IMOD/Etomo/src/etomo/ui/swing/ScaledImage.java`.
//!
//! An icon image (from the `images/` resources) scaled with the user's font size.
//! Loading, scaling and drawing the image are painting and are not modelled: an image
//! is identified by its file name, which is what [`ScaledImage::get_image`] returns.
//! The decision whether to scale (`init`, `Scale`, `Ratio`) is kept.

use std::sync::Mutex;

use crate::imod::etomo::etomo_director::{self, EtomoDirector};

/// Java `OPEN_FILE`.
pub static OPEN_FILE: ScaledImage = ScaledImage::new("openFile.gif");
/// Java `OPEN_FILE_PEET`.
pub static OPEN_FILE_PEET: ScaledImage = ScaledImage::new("openFilePeet.png");
/// Java `OPEN_FILE_FOOL`.
pub static OPEN_FILE_FOOL: ScaledImage = ScaledImage::new("openFileFool.png");
/// Java `OPEN_FILE_RED`.
pub static OPEN_FILE_RED: ScaledImage = ScaledImage::new("openFileRed.png");
/// Java `CLEAR`.
pub static CLEAR: ScaledImage = ScaledImage::new("clear.png");
/// Java `CLEAR_BLUE`.
pub static CLEAR_BLUE: ScaledImage = ScaledImage::new("clearBlue.png");
/// Java `CLEAR_RED`.
pub static CLEAR_RED: ScaledImage = ScaledImage::new("clearRed.png");
/// Java `SMALL_X`.
pub static SMALL_X: ScaledImage = ScaledImage::new("smallX.png");
/// Java `SMALL_X_PRESSED`.
pub static SMALL_X_PRESSED: ScaledImage = ScaledImage::new("smallX-pressed.png");
/// Java `SMALL_X_ROLLOVER`.
pub static SMALL_X_ROLLOVER: ScaledImage = ScaledImage::new("smallX-rollover.png");
/// Java `IMOD`.
pub static IMOD: ScaledImage = ScaledImage::new("b3dicon.png");
/// Java `IMOD_PRESSED`.
pub static IMOD_PRESSED: ScaledImage = ScaledImage::new("b3dicon-pressed.png");
/// Java `IMOD_ROLLOVER`.
pub static IMOD_ROLLOVER: ScaledImage = ScaledImage::new("b3dicon-rollover.png");
/// Java `ETOMO`.
pub static ETOMO: ScaledImage = ScaledImage::new("etomoicon.png");
/// Java `ETOMO_PRESSED`.
pub static ETOMO_PRESSED: ScaledImage = ScaledImage::new("etomoicon-pressed.png");
/// Java `ETOMO_ROLLOVER`.
pub static ETOMO_ROLLOVER: ScaledImage = ScaledImage::new("etomoicon-rollover.png");
/// Java `ETOMO_LOG`.
pub static ETOMO_LOG: ScaledImage = ScaledImage::new("projlogicon.png");
/// Java `ETOMO_LOG_PRESSED`.
pub static ETOMO_LOG_PRESSED: ScaledImage = ScaledImage::new("projlogicon-pressed.png");
/// Java `ETOMO_LOG_ROLLOVER`.
pub static ETOMO_LOG_ROLLOVER: ScaledImage = ScaledImage::new("projlogicon-rollover.png");
/// Java `BRT_LOG`.
pub static BRT_LOG: ScaledImage = ScaledImage::new("logicon.png");
/// Java `BRT_LOG_PRESSED`.
pub static BRT_LOG_PRESSED: ScaledImage = ScaledImage::new("logicon-pressed.png");
/// Java `BRT_LOG_ROLLOVER`.
pub static BRT_LOG_ROLLOVER: ScaledImage = ScaledImage::new("logicon-rollover.png");

/// Java private static `Scale`.
static SCALE: Mutex<Option<bool>> = Mutex::new(None);
/// Java private static `Ratio`.
static RATIO: Mutex<Option<f32>> = Mutex::new(None);

/// Java `ScaledImage`.
pub struct ScaledImage {
    /// Java `fileName`.
    file_name: &'static str,
    // Java fields `origImage`, `origImageLoadStatus`, `image`: the loaded and scaled
    // images - painting, not modelled.
}

impl ScaledImage {
    /// Java private `ScaledImage(String)`.
    const fn new(file_name: &'static str) -> ScaledImage {
        // Swing painting: url = ClassLoader.getSystemResource("images/" + fileName); if
        // found, origImage = Toolkit.getDefaultToolkit().getImage(url) and
        // origImageLoadStatus = loadImage(origImage).
        ScaledImage { file_name }
    }

    // Java private `loadImage(Image)`: builds an `ImageIcon` to wait for the image and
    // prints "Warning: difficulty loading <fileName>." unless loading completed.
    // Painting, not modelled.

    /// Java private static synchronized `init()`.  Initialize the first time an image
    /// is requested.  Returns true if successful.
    fn init() -> bool {
        let mut scale = SCALE.lock().unwrap();
        if scale.is_some() {
            // Already initialized.
            return true;
        }
        if !EtomoDirector::is_user_preference_loaded() {
            return false;
        }
        let font_size = EtomoDirector::get_user_font_size();
        if font_size > etomo_director::SCALE_IMAGES_ABOVE_FONT_SIZE
            || font_size < etomo_director::SCALE_IMAGES_BELOW_FONT_SIZE
        {
            // Scaling is necessary.
            *scale = Some(true);
            *RATIO.lock().unwrap() =
                Some(font_size as f32 / etomo_director::SCALE_IMAGES_ABOVE_FONT_SIZE as f32);
        } else {
            *scale = Some(false);
        }
        true
    }

    /// Java public synchronized `getImage(ImageObserver)`.  Return the original image
    /// if scaling is unnecessary, or attempt to scale it.  The image is identified by
    /// its file name.
    pub fn get_image(&self) -> &'static str {
        // Swing painting: return the cached scaled image if one was made; return null if
        // the original image was not found; retry loadImage and return the original
        // with "Warning: unable to load <fileName>." if it still fails.
        let scale_is_null = SCALE.lock().unwrap().is_none();
        if scale_is_null && !ScaledImage::init() {
            // The user configuration hasn't been loaded yet.
            return self.file_name;
        }
        // Scale the image if required.
        let scale = *SCALE.lock().unwrap();
        if scale == Some(true) {
            // Swing painting: draw origImage into a BufferedImage of
            // max(round(width * Ratio), 1) x max(round(height * Ratio), 1) with bilinear
            // interpolation; if it does not load, print "Warning: unable to scale
            // <fileName>." and return the original.  (Ratio is RATIO; Math.round is
            // ui_utilities::java_math_round_f32.)
        }
        // Since the image has been set, delete origImage.
        self.file_name
    }

    /// Java `fileName` (the image's identity in this stand-in).
    pub fn get_file_name(&self) -> &'static str {
        self.file_name
    }
}
