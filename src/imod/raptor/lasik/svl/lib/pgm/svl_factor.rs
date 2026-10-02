//! Translation of `IMOD/raptor/lasik/svl/lib/pgm/svlFactor.h` and
//! `svlFactor.cpp`, the parts `MarkersCorrespond` reaches.
//!
//! A factor's table lives in an `svlFactorStorage` that several factors may
//! share (the message-passing engine gives all its intermediate factors one
//! shared scratch table).  The C++ `svlFactorStorage*` is an
//! `Rc<RefCell<SvlFactorStorage>>` here: a factor that owns its table holds
//! the only reference, a shared table is cloned between factors exactly
//! where the C++ copies the pointer, and `delete` of an owned table is
//! `Drop`.  `operator[]` is [`SvlFactor::get`]/[`SvlFactor::set`].

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use crate::imod::raptor::lasik::svl::lib::base::svl_logger::{SvlLogLevel, svl_log};

/// The `SVL_ASSERT(C)` macro (`svlLogger.h:51`): a fatal log (which aborts)
/// when `cond` is false.  `text` is the stringised condition.
fn svl_assert(cond: bool, line: u32, text: &str) {
    if !cond {
        svl_log(SvlLogLevel::Fatal, "svlFactor.cpp", line, text);
    }
}

/// `svlFactor::_tol` (`svlFactor.cpp:50`).
pub const TOL: f64 = 1.0e-9;

/// `class svlFactorStorage` (`svlFactor.h:169`).
#[derive(Clone, Debug)]
pub struct SvlFactorStorage {
    /// Shared factor.
    b_shared: bool,
    /// Amount of memory allocated.
    data_size: i32,
    /// Memory allocation (`new double[]`, uninitialised in C++; zero here
    /// until written).
    pub data: Vec<f64>,
}

impl SvlFactorStorage {
    /// `svlFactorStorage(int nSize, bool bShared)` (`svlFactor.cpp:1004`).
    pub fn new(n_size: i32, b_shared: bool) -> SvlFactorStorage {
        let mut s = SvlFactorStorage {
            b_shared,
            data_size: 0,
            data: Vec::new(),
        };
        s.reserve(n_size);
        s
    }

    /// `isShared()`.
    pub fn is_shared(&self) -> bool {
        self.b_shared
    }

    /// `reserve(nSize)` (`svlFactor.cpp:1020`): grow the allocation, keeping
    /// the first `_dataSize` entries.
    pub fn reserve(&mut self, n_size: i32) {
        if self.data_size < n_size {
            self.data.resize(n_size as usize, 0.0);
            self.data_size = n_size;
        }
    }

    /// `fill(v, nSize)` (`svlFactor.cpp:1054`).
    pub fn fill(&mut self, v: f64, n_size: i32) {
        let mut n_size = n_size;
        if n_size < 0 {
            n_size = self.data_size;
        }
        self.reserve(n_size);
        for x in &mut self.data[..n_size as usize] {
            *x = v;
        }
    }

    /// `copy(const svlFactorStorage* p, int nSize)` (`svlFactor.cpp:1082`).
    pub fn copy(&mut self, p: &SvlFactorStorage, n_size: i32) {
        svl_assert(
            p.data_size >= n_size,
            1084,
            "(p != NULL) && (p->_dataSize >= nSize)",
        );
        let mut n_size = n_size;
        if n_size < 0 {
            n_size = self.data_size;
        }
        self.reserve(n_size);
        let n = n_size as usize;
        self.data[..n].copy_from_slice(&p.data[..n]);
    }
}

/// A C++ `svlFactorStorage*`.
pub type StorageRef = Rc<RefCell<SvlFactorStorage>>;

/// `class svlFactor` (`svlFactor.h:62`).
#[derive(Debug)]
pub struct SvlFactor {
    /// List of variables in factor (by index).
    variables: Vec<i32>,
    /// Index of variable in factor (by var).
    var_index: BTreeMap<i32, i32>,
    /// Cardinality of each variable (by index).
    cards: Vec<i32>,
    /// Stride of variable in table (by index).
    stride: Vec<i32>,
    /// Total size of factor.
    n_size: i32,
    data: Option<StorageRef>,
}

impl Default for SvlFactor {
    fn default() -> SvlFactor {
        SvlFactor::new()
    }
}

/// `svlFactor(const svlFactor& phi)` (`svlFactor.cpp:101`): a shared table is
/// shared with the copy, an owned one is copied.
impl Clone for SvlFactor {
    fn clone(&self) -> SvlFactor {
        let mut data = None;
        if let Some(d) = &self.data {
            if d.borrow().is_shared() {
                data = Some(Rc::clone(d));
            } else {
                let mut storage = SvlFactorStorage::new(self.n_size, false);
                let n = storage.data_size;
                storage.copy(&d.borrow(), n);
                data = Some(Rc::new(RefCell::new(storage)));
            }
        }
        SvlFactor {
            variables: self.variables.clone(),
            var_index: self.var_index.clone(),
            cards: self.cards.clone(),
            stride: self.stride.clone(),
            n_size: self.n_size,
            data,
        }
    }
}

impl SvlFactor {
    /// `svlFactor()` (`svlFactor.cpp:77`).
    pub fn new() -> SvlFactor {
        SvlFactor {
            variables: Vec::new(),
            var_index: BTreeMap::new(),
            cards: Vec::new(),
            stride: Vec::new(),
            n_size: 0,
            data: None,
        }
    }

    /// `svlFactor(svlFactorStorage* sharedStorage)` (`svlFactor.cpp:168`).
    pub fn with_shared_storage(shared_storage: &StorageRef) -> SvlFactor {
        svl_assert(
            shared_storage.borrow().is_shared(),
            173,
            "(sharedStorage != NULL) && (sharedStorage->isShared())",
        );
        let mut f = SvlFactor::new();
        f.data = Some(Rc::clone(shared_storage));
        f
    }

    /// `empty()`.
    pub fn empty(&self) -> bool {
        self.variables.is_empty()
    }

    /// `size()`.
    pub fn size(&self) -> i32 {
        self.n_size
    }

    /// `numVars()`.
    pub fn num_vars(&self) -> i32 {
        self.variables.len() as i32
    }

    /// `cards()`.
    pub fn cards(&self) -> &[i32] {
        &self.cards
    }

    /// `hasVariable(v)`.
    pub fn has_variable(&self, v: i32) -> bool {
        self.var_index.contains_key(&v)
    }

    /// `isShared()` (`svlFactor.h:230`).
    pub fn is_shared(&self) -> bool {
        match &self.data {
            None => false,
            Some(d) => d.borrow().is_shared(),
        }
    }

    /// The table (`_data`); `None` for a factor with no storage.
    pub fn storage(&self) -> Option<&StorageRef> {
        self.data.as_ref()
    }

    /// `operator[](index) const`.
    pub fn get(&self, index: usize) -> f64 {
        self.data.as_ref().unwrap().borrow().data[index]
    }

    /// `operator[](index) = value`.
    pub fn set(&mut self, index: usize, value: f64) {
        self.data.as_ref().unwrap().borrow_mut().data[index] = value;
    }

    /// `addVariable(v, d)` (`svlFactor.cpp:208`).
    pub fn add_variable(&mut self, v: i32, d: i32) -> i32 {
        let vvec = vec![v];
        let dvec = vec![d];
        self.add_variables(&vvec, &dvec)
    }

    /// `addVariables(const vector<int>& v, const vector<int>& d)`
    /// (`svlFactor.cpp:215`).
    pub fn add_variables(&mut self, v: &[i32], d: &[i32]) -> i32 {
        svl_assert(v.len() == d.len(), 217, "v.size() == d.size()");

        let old_size = self.n_size;
        for i in 0..v.len() {
            svl_assert(d[i] >= 1, 221, "d[i] >= 1");
            svl_assert(!self.has_variable(v[i]), 222, "!hasVariable(v[i])");

            self.variables.push(v[i]);
            self.var_index.insert(v[i], self.variables.len() as i32 - 1);
            self.cards.push(d[i]);
            self.stride
                .push(if self.n_size == 0 { 1 } else { self.n_size });
            self.n_size = *self.stride.last().unwrap() * d[i];
        }

        if old_size == self.n_size {
            return self.n_size;
        }

        match &self.data {
            None => {
                self.data = Some(Rc::new(RefCell::new(SvlFactorStorage::new(
                    self.n_size,
                    false,
                ))));
                self.initialize();
            }
            Some(d) if d.borrow().is_shared() => {
                let n_size = self.n_size;
                d.borrow_mut().reserve(n_size);
                // replicate table
                if old_size != 0 {
                    let mut storage = d.borrow_mut();
                    let mut i = old_size;
                    while i < n_size {
                        let (head, rest) = storage.data.split_at_mut(i as usize);
                        rest[..old_size as usize].copy_from_slice(&head[..old_size as usize]);
                        i += old_size;
                    }
                } else {
                    self.initialize();
                }
            }
            Some(d) => {
                let mut new_data = SvlFactorStorage::new(self.n_size, false);
                // replicate table
                svl_assert(old_size != 0, 257, "oldSize != 0");
                {
                    let old = d.borrow();
                    let mut i = 0;
                    while i < self.n_size {
                        new_data.data[i as usize..(i + old_size) as usize]
                            .copy_from_slice(&old.data[..old_size as usize]);
                        i += old_size;
                    }
                }
                self.data = Some(Rc::new(RefCell::new(new_data)));
            }
        }

        self.n_size
    }

    /// `addVariables(const svlFactor& phi)` (`svlFactor.cpp:268`).
    pub fn add_variables_of(&mut self, phi: &SvlFactor) -> i32 {
        let mut vvec = Vec::new();
        let mut dvec = Vec::new();

        for i in 0..phi.variables.len() {
            let v = phi.variables[i];
            if self.has_variable(v) {
                svl_assert(
                    self.cards[self.var_index[&v] as usize] == phi.cards[i],
                    276,
                    "_cards[_varIndex.find(v)->second] == phi._cards[i]",
                );
                continue;
            }
            vvec.push(v);
            dvec.push(phi.cards[i]);
        }

        self.add_variables(&vvec, &dvec)
    }

    /// `indexOf(var, val, indx)` (`svlFactor.h:234`).
    fn index_of(&self, var: i32, val: i32, indx: i32) -> i32 {
        let vi = self.var_index[&var] as usize;
        let mut val = val;
        let mut indx = indx;
        val -= (indx / self.stride[vi]) % self.cards[vi];
        indx += val * self.stride[vi];
        indx
    }

    /// `valueOf(var, indx)` (`svlFactor.h:270`).
    fn value_of(&self, var: i32, indx: i32) -> i32 {
        let vi = self.var_index[&var] as usize;
        (indx / self.stride[vi]) % self.cards[vi]
    }

    /// `initialize()` (`svlFactor.cpp:311`).
    pub fn initialize(&mut self) -> &mut SvlFactor {
        self.fill(1.0)
    }

    /// `fill(alpha)` (`svlFactor.cpp:316`).
    pub fn fill(&mut self, alpha: f64) -> &mut SvlFactor {
        match &self.data {
            None => return self,
            Some(d) => d.borrow_mut().fill(alpha, self.n_size),
        }
        self
    }

    /// `normalize()` (`svlFactor.cpp:345`).
    pub fn normalize(&mut self) -> &mut SvlFactor {
        let Some(d) = &self.data else {
            return self;
        };
        let n = self.n_size as usize;
        let mut storage = d.borrow_mut();
        let mut total = 0.0f64;
        for i in 0..n {
            total += storage.data[i];
        }
        if total > 0.0 {
            if total != 1.0 {
                let inv_total = 1.0 / total;
                for i in 0..n {
                    storage.data[i] *= inv_total;
                }
            }
        } else {
            storage.fill(1.0 / n as f64, n as i32);
        }
        drop(storage);
        self
    }

    /// `product(const svlFactor& phi)` (`svlFactor.cpp:557`).
    pub fn product(&mut self, phi: &SvlFactor) -> &mut SvlFactor {
        let mut index: i32;

        // singleton factor
        if phi.n_size == 0 {
            return self;
        } else if self.n_size == 0 {
            self.assign(phi);
            return self;
        }

        // check variables are the correct size and add missing to this
        for i in 0..phi.variables.len() {
            if self.has_variable(phi.variables[i]) {
                index = self.var_index[&phi.variables[i]];
                svl_assert(
                    self.cards[index as usize] == phi.cards[i],
                    579,
                    "_cards[index] == phi._cards[i]",
                );
            } else {
                // replicates factor entries to all values of new variable
                self.add_variable(phi.variables[i], phi.cards[i]);
            }
        }

        let phi_data = phi.data.as_ref().unwrap().borrow();
        let mut data = self.data.as_ref().unwrap().borrow_mut();

        // special case for multiplying by a single variable factor
        if phi.variables.len() == 1 {
            index = self.var_index[&phi.variables[0]];
            let stride = self.stride[index as usize];
            let mut i = 0i32;
            for k in 0..self.n_size {
                data.data[k as usize] *= phi_data.data[i as usize];
                if k % stride == stride - 1 {
                    i = (i + 1) % phi.n_size;
                }
            }
            drop(data);
            drop(phi_data);
            return self;
        }

        // full factor multiplication
        for k in 0..self.n_size {
            let mut k_phi = 0;
            for &var in &phi.variables {
                let value = self.value_of(var, k);
                k_phi = phi.index_of(var, value, k_phi);
            }
            data.data[k as usize] *= phi_data.data[k_phi as usize];
        }
        drop(data);
        drop(phi_data);
        self
    }

    /// `dataCompare(const svlFactor& phi)` (`svlFactor.cpp:816`), the
    /// unrolled arm the source compiles.
    pub fn data_compare(&self, phi: &SvlFactor) -> bool {
        if phi.n_size != self.n_size {
            return false;
        }
        let p = self.data.as_ref().unwrap().borrow();
        let q = phi.data.as_ref().unwrap().borrow();
        let mut k = 0usize;
        let mut i = self.n_size / 2;
        while i != 0 {
            if (p.data[k] - q.data[k]).abs() > TOL || (p.data[k + 1] - q.data[k + 1]).abs() > TOL {
                return false;
            }
            i -= 1;
            k += 2;
        }
        (self.n_size % 2 == 0) || ((p.data[k] - q.data[k]).abs() <= TOL)
    }

    /// `operator=(const svlFactor& phi)` (`svlFactor.cpp:896`).
    pub fn assign(&mut self, phi: &SvlFactor) -> &mut SvlFactor {
        let same = match (&self.data, &phi.data) {
            (None, None) => true,
            (Some(a), Some(b)) => Rc::ptr_eq(a, b),
            _ => false,
        };
        if same {
            // Really want to check (*this == phi), but check on data is
            // much quicker. Also works for _data == NULL
            return self;
        }

        if let Some(d) = &self.data {
            if self.n_size != phi.n_size {
                if d.borrow().is_shared() {
                    d.borrow_mut().reserve(phi.n_size);
                } else {
                    self.data = None;
                }
            }
        }

        self.n_size = phi.n_size;
        self.variables = phi.variables.clone();
        self.var_index = phi.var_index.clone();
        self.cards = phi.cards.clone();
        self.stride = phi.stride.clone();

        if let Some(pd) = &phi.data {
            if self.data.is_none() {
                self.data = Some(Rc::new(RefCell::new(SvlFactorStorage::new(
                    self.n_size,
                    false,
                ))));
            }
            self.data
                .as_ref()
                .unwrap()
                .borrow_mut()
                .copy(&pd.borrow(), self.n_size);
        }

        self
    }

    /// `mapFrom(const svlFactor& phi)` (`svlFactor.cpp:929`): for each entry
    /// of this factor, the index of the entry of `phi` with the same
    /// assignment to `phi`'s variables.
    pub fn map_from(&self, phi: &SvlFactor) -> Vec<i32> {
        if phi.empty() || self.empty() {
            return vec![0; self.size() as usize];
        }

        let mut mapping = vec![0i32; self.size() as usize];

        // special case for speed
        if phi.num_vars() == 1 {
            svl_assert(
                self.has_variable(phi.variables[0]),
                939,
                "hasVariable(phi._variables[0])",
            );
            let vi = self.var_index[&phi.variables[0]] as usize;
            let stride = self.stride[vi] as usize;
            let mut it = 0usize;
            while it != mapping.len() {
                for k_phi in 0..self.cards[vi] {
                    mapping[it..it + stride].fill(k_phi);
                    it += stride;
                }
            }
            return mapping;
        }

        if self.num_vars() == 1 {
            let v = phi.var_index.get(&self.variables[0]);
            svl_assert(v.is_some(), 952, "v != phi._varIndex.end()");
            let phi_stride = phi.stride[*v.unwrap() as usize];
            let mut k_phi = 0;
            for m in mapping.iter_mut() {
                *m = k_phi;
                k_phi += phi_stride;
            }
            return mapping;
        }

        // slower case
        let mut assignment = vec![0i32; self.num_vars() as usize];
        let phi_stride = phi.stride_mapping(&self.variables);

        let mut k_phi = 0;
        for k in 0..self.size() as usize {
            svl_assert(
                (k_phi >= 0) && (k_phi < phi.size()),
                968,
                "(kPhi >= 0) && (kPhi < phi.size())",
            );
            mapping[k] = k_phi;
            assignment[0] += 1;
            k_phi += phi_stride[0];
            for i in 1..self.num_vars() as usize {
                if assignment[i - 1] < self.cards[i - 1] {
                    break;
                }
                assignment[i - 1] = 0;
                assignment[i] += 1;
                k_phi += phi_stride[i];
            }
        }

        mapping
    }

    /// `mapOnto(const svlFactor& phi)` (`svlFactor.h:161`).
    pub fn map_onto(&self, phi: &SvlFactor) -> Vec<i32> {
        phi.map_from(self)
    }

    /// `strideMapping(const vector<int>& vars)` (`svlFactor.cpp:984`).
    fn stride_mapping(&self, vars: &[i32]) -> Vec<i32> {
        let mut stride_map = vec![0i32; vars.len()];
        let mut last_v: Option<i32> = None;
        for i in 0..vars.len() {
            let v = self.var_index.get(&vars[i]).copied();
            if let Some(vi) = v {
                stride_map[i] = self.stride[vi as usize];
            }
            if let Some(li) = last_v {
                stride_map[i] -= self.cards[li as usize] * self.stride[li as usize];
            }
            last_v = v;
        }

        stride_map
    }
}
