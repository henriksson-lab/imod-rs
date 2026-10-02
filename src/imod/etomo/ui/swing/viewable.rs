//! `IMOD/Etomo/src/etomo/ui/swing/Viewable.java`.

use std::rc::Rc;

use crate::imod::etomo::jdk::JComponent;

/// Java `Viewable`: a paged table.
pub trait Viewable {
    /// Java `msgViewportPaged()`.  Only called when the viewport is paged.  Should
    /// remove and redisplay the rows in the table.
    fn msg_viewport_paged(&self);

    /// Java `size()`.  Should return the total number of rows in the table.
    fn size(&self) -> i32;

    /// Java `getFocusableParents()`.  Returns the components that can be focused on
    /// for the use of hotkeys.
    fn get_focusable_parents(&self) -> Vec<Rc<JComponent>>;
}
