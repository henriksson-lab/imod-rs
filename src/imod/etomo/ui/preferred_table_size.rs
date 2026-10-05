//! `IMOD/Etomo/src/etomo/ui/PreferredTableSize.java`.
//!
//! Used for storing and comparing the preferred size of table components.  The
//! components are event dispatch thread objects (`Rc`).

use std::cell::RefCell;
use std::rc::Rc;

use super::table_component::TableComponent;

/// Java `rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java `public final class PreferredTableSize`.
pub struct PreferredTableSize {
    /// Java private final `columnList`.
    column_list: Option<Vec<Column>>,
}

impl PreferredTableSize {
    /// Java `PreferredTableSize(int)`.
    pub fn new(num_columns: i32) -> PreferredTableSize {
        if num_columns > 0 {
            let mut column_list = Vec::new();
            for _ in 0..num_columns {
                column_list.push(Column::new());
            }
            PreferredTableSize {
                column_list: Some(column_list),
            }
        } else {
            PreferredTableSize { column_list: None }
        }
    }

    /// Java `addColumn(int, TableComponent)`.
    pub fn add_column(&self, index: i32, component: Option<Rc<dyn TableComponent>>) {
        if let (Some(component), Some(column_list)) = (component, &self.column_list)
            && index >= 0
            && (index as usize) < column_list.len()
        {
            column_list[index as usize].add(component);
        }
    }

    /// Java `addColumn(int, TableComponent, TableComponent)`.
    pub fn add_column_pair(
        &self,
        index: i32,
        component1: Option<Rc<dyn TableComponent>>,
        component2: Option<Rc<dyn TableComponent>>,
    ) {
        if let Some(column_list) = &self.column_list
            && index >= 0
            && (index as usize) < column_list.len()
        {
            match (component1, component2) {
                (Some(component1), Some(component2)) => {
                    column_list[index as usize].add_pair(component1, component2)
                }
                (Some(component1), None) => column_list[index as usize].add(component1),
                (None, Some(component2)) => column_list[index as usize].add(component2),
                (None, None) => {}
            }
        }
    }

    /// Java `getPreferredWidth()`.  Returns the sum of the widest component or group of
    /// components in each column.
    pub fn get_preferred_width(&self) -> i32 {
        if let Some(column_list) = &self.column_list {
            let mut width = 0;
            for column in column_list {
                width += column.get_preferred_width();
            }
            return width;
        }
        0
    }
}

/// Java private static final nested `Column`.
struct Column {
    /// Java private final `list`.
    list: RefCell<Vec<Rc<dyn TableComponent>>>,
}

impl Column {
    fn new() -> Column {
        Column {
            list: RefCell::new(Vec::new()),
        }
    }

    /// Java private `add(TableComponent)`.
    fn add(&self, component: Rc<dyn TableComponent>) {
        self.list.borrow_mut().push(component);
    }

    /// Java private `add(TableComponent, TableComponent)`.
    fn add_pair(&self, component1: Rc<dyn TableComponent>, component2: Rc<dyn TableComponent>) {
        let component_list = ComponentList::new();
        component_list.add(component1);
        component_list.add(component2);
        self.list.borrow_mut().push(Rc::new(component_list));
    }

    /// Java private `getPreferredWidth()`.  Returns the width of the widest component or
    /// group of components in this column.
    fn get_preferred_width(&self) -> i32 {
        let mut width = 0;
        for component in self.list.borrow().iter() {
            width = width.max(component.get_preferred_width());
        }
        width
    }
}

/// Java private static final nested `ComponentList implements TableComponent`.
struct ComponentList {
    /// Java private final `list`.
    list: RefCell<Vec<Rc<dyn TableComponent>>>,
}

impl ComponentList {
    fn new() -> ComponentList {
        ComponentList {
            list: RefCell::new(Vec::new()),
        }
    }

    /// Java private `add(TableComponent)`.
    fn add(&self, component: Rc<dyn TableComponent>) {
        self.list.borrow_mut().push(component);
    }
}

impl TableComponent for ComponentList {
    /// Java `getPreferredWidth()`.  Returns the sum of the preferred width of all the
    /// components.
    fn get_preferred_width(&self) -> i32 {
        let mut width = 0;
        for component in self.list.borrow().iter() {
            width += component.get_preferred_width();
        }
        width
    }
}
