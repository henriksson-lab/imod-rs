//! Translation of `IMOD/raptor/lasik/svl/lib/pgm/svlClusterGraph.h` and
//! `svlClusterGraph.cpp`, the parts `MarkersCorrespond` reaches.

use std::collections::BTreeSet;

use super::svl_factor::SvlFactor;
use crate::imod::raptor::lasik::svl::lib::base::svl_logger::{SvlLogLevel, svl_log};

/// `typedef set<int> svlClique` (`svlClusterGraph.h:47`): iteration is in
/// ascending order, as `std::set<int>`'s.
pub type SvlClique = BTreeSet<i32>;

/// `SVL_ASSERT(C)` for this unit.
fn svl_assert(cond: bool, line: u32, text: &str) {
    if !cond {
        svl_log(SvlLogLevel::Fatal, "svlClusterGraph.cpp", line, text);
    }
}

/// `class svlClusterGraph` (`svlClusterGraph.h:51`).
#[derive(Clone, Debug, Default)]
pub struct SvlClusterGraph {
    /// Number of variables.
    n_vars: i32,
    /// Size of domain of each variable.
    var_cards: Vec<i32>,
    cliques: Vec<SvlClique>,
    edges: Vec<(i32, i32)>,
    separators: Vec<SvlClique>,
    initial_potentials: Vec<SvlFactor>,
}

impl SvlClusterGraph {
    /// `svlClusterGraph()` (`svlClusterGraph.cpp:56`).
    pub fn new() -> SvlClusterGraph {
        SvlClusterGraph::default()
    }

    /// `svlClusterGraph(int nVars, const vector<int>& varCards)`
    /// (`svlClusterGraph.cpp:71`).
    pub fn with_cards(n_vars: i32, var_cards: &[i32]) -> SvlClusterGraph {
        svl_assert(
            (n_vars > 0) && (var_cards.len() == n_vars as usize),
            74,
            "(nVars > 0) && (varCards.size() == (unsigned)nVars)",
        );
        SvlClusterGraph {
            n_vars,
            var_cards: var_cards.to_vec(),
            cliques: Vec::with_capacity(n_vars as usize),
            edges: Vec::new(),
            separators: Vec::new(),
            initial_potentials: Vec::with_capacity(n_vars as usize),
        }
    }

    /// `numCliques()`.
    pub fn num_cliques(&self) -> i32 {
        self.cliques.len() as i32
    }

    /// `numEdges()`.
    pub fn num_edges(&self) -> i32 {
        self.edges.len() as i32
    }

    /// `getCardinality(v)`.
    pub fn get_cardinality(&self, v: i32) -> i32 {
        self.var_cards[v as usize]
    }

    /// `getEdge(e)`.
    pub fn get_edge(&self, e: i32) -> (i32, i32) {
        self.edges[e as usize]
    }

    /// `addClique(const svlClique& c, const svlFactor& phi)`
    /// (`svlClusterGraph.cpp:103`).
    pub fn add_clique(&mut self, c: &SvlClique, phi: &SvlFactor) {
        self.cliques.push(c.clone());
        self.initial_potentials.push(phi.clone());
    }

    /// `getClique(indx)` (`svlClusterGraph.cpp:115`).
    pub fn get_clique(&self, indx: i32) -> &SvlClique {
        svl_assert(
            (indx >= 0) && (indx < self.cliques.len() as i32),
            117,
            "(indx >= 0) && (indx < (int)_cliques.size())",
        );
        &self.cliques[indx as usize]
    }

    /// `getSepSet(indx)` (`svlClusterGraph.cpp:121`).
    pub fn get_sep_set(&self, indx: i32) -> &SvlClique {
        svl_assert(
            (indx >= 0) && (indx < self.separators.len() as i32),
            123,
            "(indx >= 0) && (indx < (int)_separators.size())",
        );
        &self.separators[indx as usize]
    }

    /// `getCliquePotential(indx)` (`svlClusterGraph.cpp:146`).
    pub fn get_clique_potential(&self, indx: i32) -> &SvlFactor {
        svl_assert(
            (indx >= 0) && (indx < self.cliques.len() as i32),
            148,
            "(indx >= 0) && (indx < (int)_cliques.size())",
        );
        &self.initial_potentials[indx as usize]
    }

    /// `betheApprox()` (`svlClusterGraph.cpp:548`): connect graph using the
    /// bethe-approximation to the energy functional. All messages pass
    /// through marginals.
    pub fn bethe_approx(&mut self) -> bool {
        self.edges.clear();
        self.separators.clear();

        // find singleton connecting nodes
        let mut singleton_nodes = vec![-1i32; self.n_vars as usize];
        for i in 0..self.cliques.len() {
            if self.cliques[i].len() != 1 {
                continue;
            }
            let first = *self.cliques[i].iter().next().unwrap() as usize;
            if singleton_nodes[first] == -1 {
                singleton_nodes[first] = i as i32;
            }
        }

        // require singleton nodes
        for i in 0..self.n_vars as usize {
            if singleton_nodes[i] == -1 {
                svl_log(
                    SvlLogLevel::Fatal,
                    "svlClusterGraph.cpp",
                    577,
                    &format!("missing node for variable {i}"),
                );
                return false;
            }
        }

        // connect clique to singleton nodes
        for i in 0..self.cliques.len() {
            if (self.cliques[i].len() == 1)
                && (singleton_nodes[*self.cliques[i].iter().next().unwrap() as usize] == i as i32)
            {
                continue;
            }

            for &it in &self.cliques[i] {
                svl_assert(
                    singleton_nodes[it as usize] != -1,
                    591,
                    "singletonNodes[*it] != -1",
                );
                self.edges.push((i as i32, singleton_nodes[it as usize]));
                self.separators
                    .push(self.cliques[singleton_nodes[it as usize] as usize].clone());
            }
        }

        true
    }
}
