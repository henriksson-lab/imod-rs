//! `IMOD/Etomo/src/etomo/ui/swing/Viewport.java`.
//!
//! The window of rows a paged table ([`Viewable`]) shows, and the paging panel that
//! moves it.

use std::cell::{Cell, RefCell};
use std::rc::{Rc, Weak};

use super::paging_panel::PagingPanel;
use super::viewable::Viewable;
use crate::imod::etomo::jdk::JComponent;
use crate::imod::etomo::r#type::const_etomo_number;

/// Java private `MINUMUM_SIZE`.
const MINUMUM_SIZE: i32 = 5;

/// Java `Viewport`.
pub struct Viewport {
    /// Java `viewable`: the table, which owns this viewport (so a `Weak`).
    viewable: Weak<dyn Viewable>,
    /// Java `size`.
    size: i32,
    /// Java `uniqueKey`.
    unique_key: Option<String>,
    /// Java `start`.
    start: Cell<i32>,
    /// Java `end`.
    end: Cell<i32>,
    /// Java `pagingPanel`.
    paging_panel: RefCell<Option<Rc<PagingPanel>>>,
}

impl Viewport {
    /// Java `Viewport(Viewable, int, String)`.
    pub fn new(viewable: Weak<dyn Viewable>, size: i32, unique_key: Option<&str>) -> Rc<Viewport> {
        let size = if size < MINUMUM_SIZE || size == const_etomo_number::INTEGER_NULL_VALUE {
            MINUMUM_SIZE
        } else {
            size
        };
        Rc::new(Viewport {
            viewable,
            size,
            unique_key: unique_key.map(str::to_owned),
            start: Cell::new(0),
            end: Cell::new(-1),
            paging_panel: RefCell::new(None),
        })
    }

    /// Java `initPaging()`.
    pub fn init_paging(self: &Rc<Self>) {
        let paging_panel = PagingPanel::get_instance(self, self.unique_key.as_deref());
        paging_panel.set_visible(false);
        *self.paging_panel.borrow_mut() = Some(paging_panel);
    }

    /// Java `getFocusableParents()`.
    pub fn get_focusable_parents(&self) -> Vec<Rc<JComponent>> {
        self.viewable.upgrade().expect("Viewable dropped").get_focusable_parents()
    }

    /// Java `homeButtonAction()`.
    pub fn home_button_action(&self) {
        self.page_viewport(0);
    }

    /// Java `pageUpButtonAction()`.
    pub fn page_up_button_action(&self) {
        self.page_viewport(self.start.get() - self.size);
    }

    /// Java `upButtonAction()`.
    pub fn up_button_action(&self) {
        self.page_viewport(self.start.get() - 1);
    }

    /// Java `downButtonAction()`.
    pub fn down_button_action(&self) {
        self.page_viewport(self.start.get() + 1);
    }

    /// Java `pageDownButtonAction()`.
    pub fn page_down_button_action(&self) {
        self.page_viewport(self.start.get() + self.size);
    }

    /// Java `endButtonAction()`.
    pub fn end_button_action(&self) {
        self.page_viewport(self.viewable.upgrade().expect("Viewable dropped").size() - self.size);
    }

    /// Java private `pageViewport(int)`.  Sets a new start value.  Notifies viewable if
    /// there has been a change in the viewport.
    fn page_viewport(&self, new_start: i32) {
        // save old start and end values
        let orig_start = self.start.get();
        let orig_end = self.end.get();
        // Reset viewport and decide whether viewable must update its display.
        // Java precedence: (reset && start != origStart) || end != origEnd.
        if (self.reset_viewport(new_start) && self.start.get() != orig_start)
            || self.end.get() != orig_end
        {
            self.viewable.upgrade().expect("Viewable dropped").msg_viewport_paged();
        }
    }

    /// Java private `resetViewport(int)`.  Takes newStart, fixes it based on the size
    /// and viewport size, sets start and end, hides or shows the paging panel and
    /// enables its buttons.  Returns false if the viewable size is too small to page.
    ///
    /// Fixed in translation: Java dereferences `pagingPanel` unconditionally and throws
    /// a NullPointerException if `initPaging` has not been called; the paging-panel
    /// statements are skipped instead.
    fn reset_viewport(&self, new_start: i32) -> bool {
        let viewable_size = self.viewable.upgrade().expect("Viewable dropped").size();
        let paging_panel = self.paging_panel.borrow().clone();
        // No paging if the viewport is set to the entire viewable area and the
        // viewable area is smaller or equals to the viewport.
        if viewable_size <= self.size {
            // Hide the paging panel when it is unnecesary.
            if let Some(paging_panel) = &paging_panel {
                paging_panel.set_visible(false);
            }
        } else if let Some(paging_panel) = &paging_panel {
            paging_panel.set_visible(true);
        }
        let changed;
        // No paging if the viewport is set to the entire viewable area and the
        // viewable area is smaller or equals to the viewport.
        if viewable_size <= self.size
            && self.start.get() == 0
            && self.end.get() == viewable_size - 1
        {
            changed = false;
        } else {
            changed = true;
            // set start
            self.start.set(new_start);
            // fix start
            // Prevent the viewport from going out of the viewable area or displaying
            // less then the viewport size.
            if self.start.get() > viewable_size - self.size {
                self.start.set(viewable_size - self.size);
            }
            // Handles the case where the viewable area is smaller then the viewport
            // size.
            if self.start.get() < 0 {
                self.start.set(0);
            }
            // set end
            self.end.set(self.start.get() + self.size - 1);
            // fix end
            // Handles the case where the viewable area is smaller then the viewport
            // size.
            if self.end.get() >= viewable_size {
                self.end.set(viewable_size - 1);
            }
        }
        if let Some(paging_panel) = &paging_panel {
            paging_panel.set_up_enabled(self.start.get() > 0);
            paging_panel.set_down_enabled(self.end.get() < self.viewable.upgrade().expect("Viewable dropped").size() - 1);
        }
        changed
    }

    /// Java `getPagingPanel()`.  Returns the component containing the paging buttons.
    pub fn get_paging_panel(&self) -> Option<Rc<JComponent>> {
        self.paging_panel
            .borrow()
            .as_ref()
            .map(|paging_panel| paging_panel.get_container())
    }

    /// Java `msgViewableChanged()`.  Runs resetViewport with the existing start value.
    pub fn msg_viewable_changed(&self) {
        self.reset_viewport(self.start.get());
    }

    /// Java `adjustViewport(int)`.  Force newIndex to appear in the viewport.  Returns
    /// true if the viewport may have changed.
    pub fn adjust_viewport(&self, new_index: i32) -> bool {
        // If newIndex is below the viewport, find a start value which puts the index
        // in the viewport.
        if new_index > self.end.get() {
            return self.reset_viewport(new_index - self.size + 1);
        }
        self.reset_viewport(new_index)
    }

    /// Java `inViewport(int)`.  Returns true if an index is currently in the viewport.
    pub fn in_viewport(&self, index: i32) -> bool {
        self.reset_viewport(self.start.get());
        index >= self.start.get() && index <= self.end.get()
    }
}
