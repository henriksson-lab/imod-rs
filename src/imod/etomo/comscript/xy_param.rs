//! `IMOD/Etomo/src/etomo/comscript/XYParam.java`.
#![allow(dead_code)]

use std::collections::BTreeMap;

use crate::imod::etomo::r#type::const_etomo_number::ConstEtomoNumber;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `XYParam`.
#[derive(Clone, Debug)]
pub struct XYParam {
    /// Java field `xMin`.
    x_min: EtomoNumber,
    /// Java field `xMax`.
    x_max: EtomoNumber,
    /// Java field `yMin`.
    y_min: EtomoNumber,
    /// Java field `yMax`.
    y_max: EtomoNumber,
}

/// Java `toString`.
impl std::fmt::Display for XYParam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[xMin:{},xMax:{},yMin:{},yMax:{}]",
            self.x_min, self.x_max, self.y_min, self.y_max
        )
    }
}

impl XYParam {
    /// Java `XYParam(String)`.
    pub fn new(name: &str) -> XYParam {
        XYParam {
            x_min: EtomoNumber::new_with_name(&format!("{}XMin", name)),
            x_max: EtomoNumber::new_with_name(&format!("{}XMax", name)),
            y_min: EtomoNumber::new_with_name(&format!("{}YMin", name)),
            y_max: EtomoNumber::new_with_name(&format!("{}YMax", name)),
        }
    }

    /// Java package-private `store(Properties, String)`.
    pub(crate) fn store(&self, props: &mut BTreeMap<String, String>, prepend: Option<&str>) {
        self.x_min.store_with_prepend(props, prepend);
        self.x_max.store_with_prepend(props, prepend);
        self.y_min.store_with_prepend(props, prepend);
        self.y_max.store_with_prepend(props, prepend);
    }

    /// Java package-private `load(Properties, String)`.
    pub(crate) fn load(&mut self, props: &BTreeMap<String, String>, prepend: Option<&str>) {
        self.x_min.load_with_prepend(props, prepend);
        self.x_max.load_with_prepend(props, prepend);
        self.y_min.load_with_prepend(props, prepend);
        self.y_max.load_with_prepend(props, prepend);
    }

    /// Java `getXMin`.
    pub fn get_x_min(&self) -> &ConstEtomoNumber {
        &self.x_min
    }

    /// Java `getXMax`.
    pub fn get_x_max(&self) -> &ConstEtomoNumber {
        &self.x_max
    }

    /// Java `getYMin`.
    pub fn get_y_min(&self) -> &ConstEtomoNumber {
        &self.y_min
    }

    /// Java `getYMax`.
    pub fn get_y_max(&self) -> &ConstEtomoNumber {
        &self.y_max
    }

    /// Java `setXMin(String)`.
    pub fn set_x_min_string(&mut self, x_min: Option<&str>) {
        self.x_min.set_string(x_min);
    }

    /// Java `setXMax(String)`.
    pub fn set_x_max_string(&mut self, x_max: Option<&str>) {
        self.x_max.set_string(x_max);
    }

    /// Java `setYMin(String)`.
    pub fn set_y_min_string(&mut self, y_min: Option<&str>) {
        self.y_min.set_string(y_min);
    }

    /// Java `setYMax(String)`.
    pub fn set_y_max_string(&mut self, y_max: Option<&str>) {
        self.y_max.set_string(y_max);
    }

    /// Java `setXMin(int)`.
    pub fn set_x_min_int(&mut self, x_min: i32) {
        self.x_min.set_int(x_min);
    }

    /// Java `setXMax(int)`.
    pub fn set_x_max_int(&mut self, x_max: i32) {
        self.x_max.set_int(x_max);
    }

    /// Java `setYMin(int)`.
    pub fn set_y_min_int(&mut self, y_min: i32) {
        self.y_min.set_int(y_min);
    }

    /// Java `setYMax(int)`.
    pub fn set_y_max_int(&mut self, y_max: i32) {
        self.y_max.set_int(y_max);
    }

    /// Java `equals(XYParam)`.
    pub fn equals(&self, xy_param: &XYParam) -> bool {
        if !self
            .x_min
            .equals_const_etomo_number(Some(&xy_param.x_min.base))
        {
            return false;
        }
        if !self
            .x_max
            .equals_const_etomo_number(Some(&xy_param.x_max.base))
        {
            return false;
        }
        if !self
            .y_min
            .equals_const_etomo_number(Some(&xy_param.y_min.base))
        {
            return false;
        }
        if !self
            .y_max
            .equals_const_etomo_number(Some(&xy_param.y_max.base))
        {
            return false;
        }
        true
    }
}
