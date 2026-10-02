//! Translation of `IMOD/raptor/lasik/svl/lib/pgm/svlFactorOperations.h` and
//! `svlFactorOperations.cpp`, the parts `MarkersCorrespond` reaches: the
//! product, marginalisation and normalisation operations and the atomic
//! operation that chains them, with `CACHE_INDEX_MAPPING` true and
//! `USE_SHARED_INDEX_CACHE` false (`svlFactorOperations.cpp:91-92`).
//!
//! The C++ operations hold `svlFactor*` into the inference engine's factor
//! lists; here they hold [`FactorPtr`]s to the same factors.  With the shared
//! index cache off, `svlFactorIndexCache::find` returns a fresh copy of each
//! mapping behind an `svlSmartPointer`, so a mapping is simply an owned
//! `Vec<i32>`.  The virtual `svlFactorOperation` hierarchy is the
//! [`SvlFactorOperation`] enum.

use std::cell::RefCell;
use std::rc::Rc;

use super::svl_factor::SvlFactor;
use crate::imod::raptor::lasik::svl::lib::base::svl_logger::{SvlLogLevel, svl_log};

/// A C++ `svlFactor*` held by an operation or the inference engine.
pub type FactorPtr = Rc<RefCell<SvlFactor>>;

/// `SVL_ASSERT(C)` for this unit.
fn svl_assert(cond: bool, line: u32, text: &str) {
    if !cond {
        svl_log(SvlLogLevel::Fatal, "svlFactorOperations.cpp", line, text);
    }
}

/// `class svlFactorProductOp` (an `svlFactorNAryOp`).
#[derive(Debug)]
pub struct SvlFactorProductOp {
    target: FactorPtr,
    factors: Vec<FactorPtr>,
    mappings: Vec<Vec<i32>>,
}

impl SvlFactorProductOp {
    /// `svlFactorProductOp(svlFactor* target, const vector<const svlFactor*>&
    /// A)` (`svlFactorOperations.cpp:371`), through
    /// `svlFactorNAryOp(target, A)` (`:209`) and its `initialize()` (`:236`).
    pub fn new(target: &FactorPtr, a: &[FactorPtr]) -> SvlFactorProductOp {
        svl_assert(!a.is_empty(), 213, "!A.empty()");
        let mut op = SvlFactorProductOp {
            target: Rc::clone(target),
            factors: a.to_vec(),
            mappings: Vec::new(),
        };

        // add variables and check domains match
        {
            let mut t = op.target.borrow_mut();
            if t.empty() {
                for f in &op.factors {
                    t.add_variables_of(&f.borrow());
                }
            } else {
                // `SVL_ASSERT(checkTarget())`: the engine builds every
                // product into an empty intermediate factor.
                unreachable!("svlFactorNAryOp::initialize on a non-empty target");
            }
        }

        // create mappings
        op.mappings = op
            .factors
            .iter()
            .map(|f| op.target.borrow().map_from(&f.borrow()))
            .collect();
        op
    }

    /// `svlFactorProductOp::execute()` (`svlFactorOperations.cpp:389`).
    pub fn execute(&self) {
        let target = self.target.borrow();
        let size = target.size() as usize;
        let mut t = target.storage().unwrap().borrow_mut();
        let f0 = self.factors[0].borrow();
        if f0.empty() {
            t.fill(1.0, size as i32);
        } else {
            let d0 = f0.storage().unwrap().borrow();
            let m0 = &self.mappings[0];
            for i in 0..size {
                t.data[i] = d0.data[m0[i] as usize];
            }
        }

        for k in 1..self.factors.len() {
            let fk = self.factors[k].borrow();
            if fk.empty() {
                continue;
            }
            let dk = fk.storage().unwrap().borrow();
            let mk = &self.mappings[k];
            for i in 0..size {
                t.data[i] *= dk.data[mk[i] as usize];
            }
        }
    }
}

/// `class svlFactorMarginalizeOp`: sums `A` onto the target's variables.
#[derive(Debug)]
pub struct SvlFactorMarginalizeOp {
    target: FactorPtr,
    a: FactorPtr,
    mapping_a: Vec<i32>,
}

impl SvlFactorMarginalizeOp {
    /// `svlFactorMarginalizeOp(svlFactor* target, const svlFactor* A)`
    /// (`svlFactorOperations.cpp:907`); `checkTarget()` is `true`.
    pub fn new(target: &FactorPtr, a: &FactorPtr) -> SvlFactorMarginalizeOp {
        let mapping_a = target.borrow().map_onto(&a.borrow());
        SvlFactorMarginalizeOp {
            target: Rc::clone(target),
            a: Rc::clone(a),
            mapping_a,
        }
    }

    /// `svlFactorMarginalizeOp::execute()` (`svlFactorOperations.cpp:987`).
    pub fn execute(&self) {
        self.target.borrow_mut().fill(0.0);

        let target = self.target.borrow();
        let mut t = target.storage().unwrap().borrow_mut();
        let a = self.a.borrow();
        let da = a.storage().unwrap().borrow();
        for i in 0..self.mapping_a.len() {
            t.data[self.mapping_a[i] as usize] += da.data[i];
        }
    }
}

/// `class svlFactorNormalizeOp`.
#[derive(Debug)]
pub struct SvlFactorNormalizeOp {
    target: FactorPtr,
}

impl SvlFactorNormalizeOp {
    /// `svlFactorNormalizeOp(svlFactor* target)` (`svlFactorOperations.cpp:1155`).
    pub fn new(target: &FactorPtr) -> SvlFactorNormalizeOp {
        SvlFactorNormalizeOp {
            target: Rc::clone(target),
        }
    }

    /// `svlFactorNormalizeOp::execute()` (`svlFactorOperations.cpp:1166`).
    pub fn execute(&self) {
        let mut target = self.target.borrow_mut();
        if target.empty() {
            return;
        }

        let size = target.size() as usize;
        let mut total = 0.0f64;
        {
            let t = target.storage().unwrap().borrow();
            for i in 0..size {
                total += t.data[i];
            }
        }
        if total > 0.0 {
            if total != 1.0 {
                let inv_total = 1.0 / total;
                let mut t = target.storage().unwrap().borrow_mut();
                for i in 0..size {
                    t.data[i] *= inv_total;
                }
            }
        } else {
            target.fill(1.0 / size as f64);
        }
    }
}

/// The reached members of the `svlFactorOperation` hierarchy.
#[derive(Debug)]
pub enum SvlFactorOperation {
    Product(SvlFactorProductOp),
    Marginalize(SvlFactorMarginalizeOp),
    Normalize(SvlFactorNormalizeOp),
}

impl SvlFactorOperation {
    /// `svlFactorOperation::target()` (`svlFactorOperations.h:93`).
    pub fn target(&self) -> &FactorPtr {
        match self {
            SvlFactorOperation::Product(op) => &op.target,
            SvlFactorOperation::Marginalize(op) => &op.target,
            SvlFactorOperation::Normalize(op) => &op.target,
        }
    }

    /// The virtual `execute()`.
    pub fn execute(&self) {
        match self {
            SvlFactorOperation::Product(op) => op.execute(),
            SvlFactorOperation::Marginalize(op) => op.execute(),
            SvlFactorOperation::Normalize(op) => op.execute(),
        }
    }
}

/// `class svlFactorAtomicOp`: a list of operations executed in order.
#[derive(Debug)]
pub struct SvlFactorAtomicOp {
    target: FactorPtr,
    computations: Vec<SvlFactorOperation>,
}

impl SvlFactorAtomicOp {
    /// `svlFactorAtomicOp(const vector<svlFactorOperation*>& ops)`
    /// (`svlFactorOperations.cpp:284`).
    pub fn new(ops: Vec<SvlFactorOperation>) -> SvlFactorAtomicOp {
        svl_assert(!ops.is_empty(), 287, "!_computations.empty()");
        let target = Rc::clone(ops.last().unwrap().target());
        SvlFactorAtomicOp {
            target,
            computations: ops,
        }
    }

    /// `target()`.
    pub fn target(&self) -> &FactorPtr {
        &self.target
    }

    /// `svlFactorAtomicOp::execute()` (`svlFactorOperations.cpp:303`).
    pub fn execute(&self) {
        for c in &self.computations {
            c.execute();
        }
    }
}
