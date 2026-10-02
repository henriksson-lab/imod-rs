//! Translation of `IMOD/raptor/lasik/svl/lib/pgm/svlMessagePassing.h` and
//! `svlMessagePassing.cpp`, the parts `MarkersCorrespond` reaches: residual
//! belief propagation in normal space (`SVL_MP_RBP_SUMPROD`, Elidan et al.,
//! UAI 2006), with `SINGLETON_MARGINALS_ONLY` true
//! (`svlMessagePassing.cpp:154`).  `svlMarkerCorrespondenceLBModel` never
//! selects another algorithm (`_maxProduct` is false); the other
//! algorithms, their computation-graph builders and loops are recorded in
//! `DEAD_CODE.md`.
//!
//! The engine's `svlFactor*` lists hold [`FactorPtr`]s; `delete` in
//! `reset()` and the destructor is `Drop`.

use std::cell::RefCell;
use std::rc::Rc;

use super::svl_cluster_graph::SvlClusterGraph;
use super::svl_factor::{StorageRef, SvlFactor, SvlFactorStorage};
use super::svl_factor_operations::{
    FactorPtr, SvlFactorAtomicOp, SvlFactorMarginalizeOp, SvlFactorNormalizeOp, SvlFactorOperation,
    SvlFactorProductOp,
};
use crate::imod::c_sort::std_sort;
use crate::imod::raptor::lasik::svl::lib::base::svl_code_profiler;
use crate::imod::raptor::lasik::svl::lib::base::svl_logger::{SvlLogLevel, svl_log};

/// `svlMessagePassingAlgorithms` (`svlMessagePassing.h:52`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SvlMessagePassingAlgorithms {
    None,
    SumProd,
    MaxProd,
    LogMaxProd,
    SumProdDiv,
    MaxProdDiv,
    LogMaxProdDiv,
    AsyncSumProd,
    AsyncMaxProd,
    AsyncLogMaxProd,
    AsyncLogMaxProdLazy,
    AsyncSumProdDiv,
    AsyncMaxProdDiv,
    AsyncLogMaxProdDiv,
    RbpSumProd,
    RbpMaxProd,
    RbpLogMaxProd,
    Gemplp,
    Sontag08,
}

/// `svlMessagePassingInference::SINGLETON_MARGINALS_ONLY`.
const SINGLETON_MARGINALS_ONLY: bool = true;

/// `SVL_ASSERT(C)` for this unit.
fn svl_assert(cond: bool, line: u32, text: &str) {
    if !cond {
        svl_log(SvlLogLevel::Fatal, "svlMessagePassing.cpp", line, text);
    }
}

/// `class svlMessagePassingInference` (`svlMessagePassing.h:81`).
pub struct SvlMessagePassingInference {
    /// Cluster graph for inference (includes initial clique potentials).
    graph: SvlClusterGraph,
    /// Final clique potentials.
    clique_potentials: Vec<FactorPtr>,
    /// Forward and backward messages during each iteration.
    forward_messages: Vec<FactorPtr>,
    backward_messages: Vec<FactorPtr>,
    old_forward_messages: Vec<FactorPtr>,
    old_backward_messages: Vec<FactorPtr>,
    /// Computation tree: intermediate factors, and (atomic) factor
    /// operations.
    intermediate_factors: Vec<FactorPtr>,
    computations: Vec<SvlFactorAtomicOp>,
    algorithm: SvlMessagePassingAlgorithms,
    /// Shared storage for intermediate factors.
    shared_storage: Vec<StorageRef>,
}

impl SvlMessagePassingInference {
    /// `svlMessagePassingInference(svlClusterGraph& graph)`
    /// (`svlMessagePassing.cpp:156`): the engine keeps its own copy of the
    /// graph.
    pub fn new(graph: &SvlClusterGraph) -> SvlMessagePassingInference {
        SvlMessagePassingInference {
            graph: graph.clone(),
            clique_potentials: Vec::new(),
            forward_messages: Vec::new(),
            backward_messages: Vec::new(),
            old_forward_messages: Vec::new(),
            old_backward_messages: Vec::new(),
            intermediate_factors: Vec::new(),
            computations: Vec::new(),
            algorithm: SvlMessagePassingAlgorithms::None,
            shared_storage: Vec::new(),
        }
    }

    /// `reset()` (`svlMessagePassing.cpp:171`): free memory used by
    /// intermediate factors and computations.
    pub fn reset(&mut self) {
        self.computations.clear();
        self.intermediate_factors.clear();
        self.forward_messages.clear();
        self.backward_messages.clear();
        self.old_forward_messages.clear();
        self.old_backward_messages.clear();
        self.clique_potentials.clear();
        self.algorithm = SvlMessagePassingAlgorithms::None;
        self.shared_storage.clear();
    }

    /// `operator[](index)` (`svlMessagePassing.h:149`): access to final
    /// clique potentials.
    pub fn clique_potential(&self, index: usize) -> std::cell::Ref<'_, SvlFactor> {
        self.clique_potentials[index].borrow()
    }

    /// `inference(mpAlgorithm, maxIterations)` (`svlMessagePassing.cpp:249`).
    pub fn inference(
        &mut self,
        mp_algorithm: SvlMessagePassingAlgorithms,
        max_iterations: i32,
    ) -> bool {
        // make sure we're running the same algorithm as the computation graph
        if mp_algorithm != self.algorithm {
            self.reset();
        }
        self.algorithm = mp_algorithm;
        // Only `SVL_MP_RBP_SUMPROD` is reached (see the module comment).
        svl_assert(
            self.algorithm == SvlMessagePassingAlgorithms::RbpSumProd,
            259,
            "_algorithm == SVL_MP_RBP_SUMPROD",
        );

        // assign initial clique potentials and messages
        self.initialize_message_passing();

        // update old message cache
        if self.old_forward_messages.is_empty() {
            self.old_forward_messages
                .reserve(self.forward_messages.len());
            self.old_backward_messages
                .reserve(self.backward_messages.len());
            for _ in 0..self.forward_messages.len() {
                self.old_forward_messages
                    .push(Rc::new(RefCell::new(SvlFactor::new())));
                self.old_backward_messages
                    .push(Rc::new(RefCell::new(SvlFactor::new())));
            }
        }

        for i in 0..self.forward_messages.len() {
            self.old_forward_messages[i]
                .borrow_mut()
                .assign(&self.forward_messages[i].borrow());
            self.old_backward_messages[i]
                .borrow_mut()
                .assign(&self.backward_messages[i].borrow());
        }

        // set up computation graph
        if self.computations.is_empty() {
            let handle =
                svl_code_profiler::get_handle("svlMessagePassingInference::buildComputationGraph");
            svl_code_profiler::tic(handle);
            self.build_residual_bp_computation_graph();
            svl_code_profiler::toc(handle);
        }

        // assert that all messages and intermediate factors are non-empty
        for i in 0..self.forward_messages.len() {
            svl_assert(
                !self.forward_messages[i].borrow().empty(),
                372,
                "!_forwardMessages[i]->empty()",
            );
            svl_assert(
                !self.backward_messages[i].borrow().empty(),
                373,
                "!_backwardMessages[i]->empty()",
            );
        }

        // run inference
        let b_converged = self.residual_bp_message_passing_loop(max_iterations);

        // compute final clique potentials
        self.finalize_message_passing();

        b_converged
    }

    /// `initializeMessagePassing()` (`svlMessagePassing.cpp:443`).
    fn initialize_message_passing(&mut self) {
        // initialize clique potentials
        if self.clique_potentials.is_empty() {
            for n in 0..self.graph.num_cliques() {
                self.clique_potentials.push(Rc::new(RefCell::new(
                    self.graph.get_clique_potential(n).clone(),
                )));
            }
        } else {
            svl_assert(
                self.clique_potentials.len() == self.graph.num_cliques() as usize,
                453,
                "_cliquePotentials.size() == (unsigned)_graph.numCliques()",
            );
            for n in 0..self.graph.num_cliques() {
                self.clique_potentials[n as usize]
                    .borrow_mut()
                    .assign(self.graph.get_clique_potential(n));
            }
        }

        // set up forward and backward messages
        if self.forward_messages.is_empty() {
            self.forward_messages
                .reserve(self.graph.num_edges() as usize);
            self.backward_messages
                .reserve(self.graph.num_edges() as usize);
            for i in 0..self.graph.num_edges() {
                let mut msg = SvlFactor::new();
                let s = self.graph.get_sep_set(i).clone();
                for &j in &s {
                    msg.add_variable(j, self.graph.get_cardinality(j));
                }
                msg.fill(1.0);
                let back = msg.clone();
                self.forward_messages.push(Rc::new(RefCell::new(msg)));
                self.backward_messages.push(Rc::new(RefCell::new(back)));
            }
        } else {
            // reset messages to all ones
            for i in 0..self.forward_messages.len() {
                self.forward_messages[i].borrow_mut().fill(1.0);
                self.backward_messages[i].borrow_mut().fill(1.0);
            }
        }
    }

    /// `finalizeMessagePassing()` (`svlMessagePassing.cpp:613`): compute
    /// final beliefs.
    fn finalize_message_passing(&mut self) {
        for m in 0..self.graph.num_edges() {
            let e = self.graph.get_edge(m);
            if SINGLETON_MARGINALS_ONLY {
                // don't compute beliefs on non-singleton cliques
                if self.graph.get_clique(e.0).len() == 1 {
                    self.clique_potentials[e.0 as usize]
                        .borrow_mut()
                        .product(&self.backward_messages[m as usize].borrow());
                }
                if self.graph.get_clique(e.1).len() == 1 {
                    self.clique_potentials[e.1 as usize]
                        .borrow_mut()
                        .product(&self.forward_messages[m as usize].borrow());
                }
            }
        }

        // normalize
        for n in 0..self.graph.num_cliques() as usize {
            let mut p = self.clique_potentials[n].borrow_mut();
            if p.empty() || p.is_shared() {
                continue;
            }
            if SINGLETON_MARGINALS_ONLY && (p.num_vars() != 1) {
                continue;
            }
            p.normalize();
        }
    }

    /// `buildResidualBPComputationGraph()` (`svlMessagePassing.cpp:1477`).
    fn build_residual_bp_computation_graph(&mut self) {
        self.intermediate_factors
            .reserve(2 * self.graph.num_edges() as usize);

        // shared storage for intermediate factors
        self.shared_storage
            .push(Rc::new(RefCell::new(SvlFactorStorage::new(0, true))));

        // incident edges
        let num_cliques = self.graph.num_cliques() as usize;
        let mut fwd_incident_edges: Vec<Vec<i32>> = vec![Vec::new(); num_cliques];
        let mut bck_incident_edges: Vec<Vec<i32>> = vec![Vec::new(); num_cliques];
        for m in 0..self.graph.num_edges() {
            fwd_incident_edges[self.graph.get_edge(m).0 as usize].push(m);
            bck_incident_edges[self.graph.get_edge(m).1 as usize].push(m);
        }

        // add computation for each edge
        for m in 0..self.graph.num_edges() {
            let fwd_indx = self.graph.get_edge(m).0 as usize;
            let bck_indx = self.graph.get_edge(m).1 as usize;
            let mut incoming_fwd_msgs: Vec<FactorPtr> = Vec::new();
            let mut incoming_bck_msgs: Vec<FactorPtr> = Vec::new();

            incoming_fwd_msgs.push(Rc::clone(&self.clique_potentials[fwd_indx]));
            incoming_bck_msgs.push(Rc::clone(&self.clique_potentials[bck_indx]));

            for &k in &fwd_incident_edges[fwd_indx] {
                if k != m {
                    incoming_fwd_msgs.push(Rc::clone(&self.backward_messages[k as usize]));
                }
            }
            for &k in &bck_incident_edges[fwd_indx] {
                if k != m {
                    incoming_fwd_msgs.push(Rc::clone(&self.forward_messages[k as usize]));
                }
            }
            for &k in &fwd_incident_edges[bck_indx] {
                if k != m {
                    incoming_bck_msgs.push(Rc::clone(&self.backward_messages[k as usize]));
                }
            }
            for &k in &bck_incident_edges[bck_indx] {
                if k != m {
                    incoming_bck_msgs.push(Rc::clone(&self.forward_messages[k as usize]));
                }
            }

            // forwards
            let intermediate = Rc::new(RefCell::new(SvlFactor::with_shared_storage(
                &self.shared_storage[0],
            )));
            self.intermediate_factors.push(Rc::clone(&intermediate));
            let mut atom: Vec<SvlFactorOperation> = Vec::new();
            atom.push(SvlFactorOperation::Product(SvlFactorProductOp::new(
                &intermediate,
                &incoming_fwd_msgs,
            )));
            atom.push(SvlFactorOperation::Marginalize(
                SvlFactorMarginalizeOp::new(&self.forward_messages[m as usize], &intermediate),
            ));
            atom.push(SvlFactorOperation::Normalize(SvlFactorNormalizeOp::new(
                &self.forward_messages[m as usize],
            )));
            self.computations.push(SvlFactorAtomicOp::new(atom));

            // backwards
            let intermediate = Rc::new(RefCell::new(SvlFactor::with_shared_storage(
                &self.shared_storage[0],
            )));
            self.intermediate_factors.push(Rc::clone(&intermediate));
            let mut atom: Vec<SvlFactorOperation> = Vec::new();
            atom.push(SvlFactorOperation::Product(SvlFactorProductOp::new(
                &intermediate,
                &incoming_bck_msgs,
            )));
            atom.push(SvlFactorOperation::Marginalize(
                SvlFactorMarginalizeOp::new(&self.backward_messages[m as usize], &intermediate),
            ));
            atom.push(SvlFactorOperation::Normalize(SvlFactorNormalizeOp::new(
                &self.backward_messages[m as usize],
            )));
            self.computations.push(SvlFactorAtomicOp::new(atom));
        }
    }

    /// `residualBPMessagePassingLoop(maxIterations)`
    /// (`svlMessagePassing.cpp:2031`).
    fn residual_bp_message_passing_loop(&mut self, max_iterations: i32) -> bool {
        // run initial iteration
        for c in &self.computations {
            c.execute();
        }

        // compute residuals
        let num_edges = self.graph.num_edges() as usize;
        let mut q: Vec<(f64, i32)> = vec![(0.0, 0); 2 * num_edges];
        for i in 0..num_edges {
            q[2 * i].0 = {
                let msg = self.forward_messages[i].borrow();
                let old = self.old_forward_messages[i].borrow();
                let d = msg.storage().unwrap().borrow();
                let o = old.storage().unwrap().borrow();
                let mut r = 0.0f64;
                for k in 0..msg.size() as usize {
                    r += (d.data[k] - o.data[k]).abs();
                }
                r
            };
            q[2 * i].1 = 2 * i as i32;

            q[2 * i + 1].0 = {
                let msg = self.backward_messages[i].borrow();
                let old = self.old_backward_messages[i].borrow();
                let d = msg.storage().unwrap().borrow();
                let o = old.storage().unwrap().borrow();
                let mut r = 0.0f64;
                for k in 0..msg.size() as usize {
                    r += (d.data[k] - o.data[k]).abs();
                }
                r
            };
            q[2 * i + 1].1 = 2 * i as i32 + 1;
        }
        std_sort(&mut q, &mut pair_less);

        // loop until convergence (the verbose log line is below the level)
        let mut n_iteration = 0;
        let mut b_converged;

        loop {
            b_converged = q.last().unwrap().0 < 1.0e-9;
            if b_converged {
                break;
            }
            n_iteration += 1;
            if n_iteration > max_iterations {
                break;
            }

            // send message at front of queue
            let mut edge_indx = q.last().unwrap().1;
            let node_indx;
            if edge_indx % 2 == 0 {
                edge_indx /= 2;
                self.old_forward_messages[edge_indx as usize]
                    .borrow_mut()
                    .assign(&self.forward_messages[edge_indx as usize].borrow());
                q.last_mut().unwrap().0 = 0.0;
                node_indx = self.graph.get_edge(edge_indx).1;
            } else {
                edge_indx = (edge_indx - 1) / 2;
                self.old_backward_messages[edge_indx as usize]
                    .borrow_mut()
                    .assign(&self.backward_messages[edge_indx as usize].borrow());
                q.last_mut().unwrap().0 = 0.0;
                node_indx = self.graph.get_edge(edge_indx).0;
            }

            svl_assert(node_indx >= 0, 2085, "nodeIndx >= 0");

            // update neighbours
            for i in 0..q.len() {
                edge_indx = q[i].1;
                if edge_indx % 2 == 0 {
                    edge_indx /= 2;
                    if self.graph.get_edge(edge_indx).0 == node_indx {
                        self.computations[q[i].1 as usize].execute();
                        q[i].0 = {
                            let msg = self.forward_messages[edge_indx as usize].borrow();
                            let old = self.old_forward_messages[edge_indx as usize].borrow();
                            let d = msg.storage().unwrap().borrow();
                            let o = old.storage().unwrap().borrow();
                            let mut r = 0.0f64;
                            for k in 0..msg.size() as usize {
                                r += (d.data[k] - o.data[k]).abs();
                            }
                            r
                        };
                    }
                } else {
                    edge_indx = (edge_indx - 1) / 2;
                    if self.graph.get_edge(edge_indx).1 == node_indx {
                        self.computations[q[i].1 as usize].execute();
                        q[i].0 = {
                            let msg = self.backward_messages[edge_indx as usize].borrow();
                            let old = self.old_backward_messages[edge_indx as usize].borrow();
                            let d = msg.storage().unwrap().borrow();
                            let o = old.storage().unwrap().borrow();
                            let mut r = 0.0f64;
                            for k in 0..msg.size() as usize {
                                r += (d.data[k] - o.data[k]).abs();
                            }
                            r
                        };
                    }
                }
            }

            // resort Q
            std_sort(&mut q, &mut pair_less);
        }

        if !b_converged {
            let mut n_converged = 0;
            for i in 0..num_edges {
                if self.old_forward_messages[i]
                    .borrow()
                    .data_compare(&self.forward_messages[i].borrow())
                {
                    n_converged += 1;
                }
                if self.old_backward_messages[i]
                    .borrow()
                    .data_compare(&self.backward_messages[i].borrow())
                {
                    n_converged += 1;
                }
            }

            svl_log(
                SvlLogLevel::Warning,
                "svlMessagePassing.cpp",
                2131,
                &format!(
                    "message passing failed to converge after {} iterations ({} of {} messages converged)",
                    n_iteration,
                    n_converged,
                    (2 * num_edges) as i32
                ),
            );
        }

        b_converged
    }
}

/// `operator<` of `std::pair<double, int>`, the comparison `sort(Q.begin(),
/// Q.end())` uses.
fn pair_less(a: &(f64, i32), b: &(f64, i32)) -> bool {
    a.0 < b.0 || (!(b.0 < a.0) && a.1 < b.1)
}
