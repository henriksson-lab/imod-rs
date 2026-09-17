//! Owned translation of `svlClusterGraph.{h,cpp}`.
use super::svl_factor::SvlFactor;
use std::collections::{BTreeSet, VecDeque};
use std::fs;
use std::path::Path;
pub type SvlClique = BTreeSet<i32>;
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SvlClusterGraph {
    pub variable_cards: Vec<usize>,
    pub cliques: Vec<SvlClique>,
    pub edges: Vec<(usize, usize)>,
    pub separators: Vec<SvlClique>,
    pub initial_potentials: Vec<SvlFactor>,
}

/// `svlClusterGraph::svlClusterGraph()` (`svlClusterGraph.cpp:56`).
pub fn svl_cluster_graph() -> SvlClusterGraph {
    SvlClusterGraph::new()
}

/// `svlClusterGraph(int nVars, int varCards)` (`svlClusterGraph.cpp:62`).
pub fn svl_cluster_graph_with_uniform_cards(
    nvars: usize,
    cards: usize,
) -> Result<SvlClusterGraph, String> {
    SvlClusterGraph::with_uniform_cards(nvars, cards)
}

/// `svlClusterGraph(int nVars, const vector<int>& varCards)`
/// (`svlClusterGraph.cpp:71`).
pub fn svl_cluster_graph_with_cards(cards: Vec<usize>) -> Result<SvlClusterGraph, String> {
    SvlClusterGraph::with_cards(cards)
}

/// `svlClusterGraph(int nVars, const vector<int>&, const vector<svlClique>&)`
/// (`svlClusterGraph.cpp:79`).
pub fn svl_cluster_graph_with_cliques(
    cards: Vec<usize>,
    cliques: Vec<SvlClique>,
) -> Result<SvlClusterGraph, String> {
    let mut graph = SvlClusterGraph::with_cards(cards)?;
    if cliques.iter().any(|clique| {
        clique
            .iter()
            .any(|&variable| variable < 0 || variable as usize >= graph.num_variables())
    }) {
        return Err("clique variable out of range".into());
    }
    for clique in cliques {
        graph.add_clique(clique);
    }
    Ok(graph)
}
impl SvlClusterGraph {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_uniform_cards(n: usize, c: usize) -> Result<Self, String> {
        if n == 0 || c < 2 {
            Err("invalid graph dimensions".into())
        } else {
            Ok(Self {
                variable_cards: vec![c; n],
                ..Self::new()
            })
        }
    }
    pub fn with_cards(cards: Vec<usize>) -> Result<Self, String> {
        if cards.is_empty() || cards.iter().any(|&x| x < 2) {
            Err("invalid graph cardinalities".into())
        } else {
            Ok(Self {
                variable_cards: cards,
                ..Self::new()
            })
        }
    }
    pub fn num_variables(&self) -> usize {
        self.variable_cards.len()
    }
    pub fn num_cliques(&self) -> usize {
        self.cliques.len()
    }
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }
    pub fn cardinality(&self, v: usize) -> Option<usize> {
        self.variable_cards.get(v).copied()
    }
    pub fn clique(&self, i: usize) -> Option<&SvlClique> {
        self.cliques.get(i)
    }
    pub fn separator(&self, i: usize) -> Option<&SvlClique> {
        self.separators.get(i)
    }
    pub fn add_clique(&mut self, c: SvlClique) {
        self.cliques.push(c);
        self.initial_potentials.push(SvlFactor::new())
    }
    pub fn add_clique_with_potential(&mut self, c: SvlClique, p: SvlFactor) {
        self.cliques.push(c);
        self.initial_potentials.push(p)
    }
    pub fn add_factor_clique(&mut self, p: SvlFactor) {
        self.add_clique_with_potential(p.variables.iter().copied().collect(), p)
    }
    pub fn set_clique_potential(&mut self, i: usize, p: SvlFactor) -> Result<(), String> {
        if self
            .cliques
            .get(i)
            .is_none_or(|c| c.len() != p.num_vars() || !c.iter().all(|v| p.has_variable(*v)))
        {
            Err("potential does not match clique".into())
        } else {
            self.initial_potentials[i] = p;
            Ok(())
        }
    }
    pub fn clique_potential(&self, i: usize) -> Option<&SvlFactor> {
        self.initial_potentials.get(i)
    }
    pub fn clique_potential_mut(&mut self, i: usize) -> Option<&mut SvlFactor> {
        self.initial_potentials.get_mut(i)
    }
    pub fn potential_for_clique(&self, _: &SvlClique) -> SvlFactor {
        SvlFactor::new()
    }
    pub fn potential_for_variable(&self, _: i32) -> SvlFactor {
        SvlFactor::new()
    }
    pub fn energy(&self, x: &[usize], log: bool) -> Option<f64> {
        if x.len() != self.num_variables() {
            return None;
        }
        let mut e = 0.;
        for p in &self.initial_potentials {
            if p.empty() {
                continue;
            }
            let a: Vec<_> = p
                .variables
                .iter()
                .map(|&v| x.get(v as usize).copied())
                .collect::<Option<_>>()?;
            let v = p.data[p.index_of(&a)?];
            e -= if log { v } else { v.ln() };
        }
        Some(e)
    }
    pub fn decode_map(&self, start: Option<usize>) -> Vec<Option<usize>> {
        let mut x = vec![None; self.num_variables()];
        let mut root = start.or_else(|| {
            self.cliques
                .iter()
                .enumerate()
                .max_by_key(|(_, c)| c.len())
                .map(|x| x.0)
        });
        let mut seen = vec![false; self.num_cliques()];
        while let Some(r) = root {
            let mut q = VecDeque::from([r]);
            while let Some(n) = q.pop_front() {
                if seen[n] {
                    continue;
                }
                let mut p = self.initial_potentials[n].clone();
                for (v, a) in x
                    .iter()
                    .enumerate()
                    .filter_map(|(v, a)| a.map(|a| (v as i32, a)))
                {
                    if p.has_variable(v) {
                        let _ = p.reduce(v, a);
                    }
                }
                if let Some(k) = p.index_of_max() {
                    if let Some(a) = p.assignment_of(k) {
                        for (i, &v) in p.variables.iter().enumerate() {
                            x[v as usize] = Some(a[i]);
                        }
                    }
                }
                for &(a, b) in &self.edges {
                    if a == n && !seen[b] {
                        q.push_back(b)
                    }
                    if b == n && !seen[a] {
                        q.push_back(a)
                    }
                }
                seen[n] = true;
            }
            root = seen.iter().position(|x| !x);
        }
        x
    }
    pub fn check_run_int_prop(&self) -> bool {
        for v in 0..self.num_variables() as i32 {
            let nodes: Vec<_> = self
                .cliques
                .iter()
                .enumerate()
                .filter_map(|(i, c)| c.contains(&v).then_some(i))
                .collect();
            if nodes.len() < 2 {
                continue;
            }
            let mut found = BTreeSet::from([nodes[0]]);
            let mut q = VecDeque::from([nodes[0]]);
            while let Some(n) = q.pop_front() {
                for ((a, b), s) in self.edges.iter().zip(&self.separators) {
                    if !s.contains(&v) {
                        continue;
                    }
                    let m = if *a == n {
                        Some(*b)
                    } else if *b == n {
                        Some(*a)
                    } else {
                        None
                    };
                    if let Some(m) = m {
                        if self.cliques[m].contains(&v) && found.insert(m) {
                            q.push_back(m)
                        }
                    }
                }
            }
            if nodes.iter().any(|n| !found.contains(n)) {
                return false;
            }
        }
        true
    }
    pub fn connect_graph(&mut self) -> Result<(), String> {
        self.edges.clear();
        self.separators.clear();
        for v in 0..self.num_variables() as i32 {
            let nodes: Vec<_> = self
                .cliques
                .iter()
                .enumerate()
                .filter_map(|(i, c)| c.contains(&v).then_some(i))
                .collect();
            if nodes.len() < 2 {
                continue;
            }
            for pair in nodes.windows(2) {
                let e = (pair[0].min(pair[1]), pair[0].max(pair[1]));
                if let Some(i) = self.edges.iter().position(|x| *x == e) {
                    self.separators[i].insert(v);
                } else {
                    self.edges.push(e);
                    self.separators.push(BTreeSet::from([v]));
                }
            }
        }
        Ok(())
    }
    pub fn connect_graph_with_edges(&mut self, e: Vec<(usize, usize)>) -> Result<(), String> {
        if e.iter()
            .any(|&(a, b)| a >= self.num_cliques() || b >= self.num_cliques())
        {
            return Err("invalid edge".into());
        }
        self.edges = e;
        self.compute_separator_sets();
        Ok(())
    }
    pub fn bethe_approx(&mut self) -> bool {
        self.edges.clear();
        self.separators.clear();
        let single: Vec<_> = (0..self.num_variables() as i32)
            .map(|v| {
                self.cliques
                    .iter()
                    .position(|c| c.len() == 1 && c.contains(&v))
            })
            .collect();
        if single.iter().any(Option::is_none) {
            return false;
        }
        for (i, c) in self.cliques.iter().enumerate() {
            if c.len() == 1 && single[*c.iter().next().unwrap() as usize] == Some(i) {
                continue;
            }
            for &v in c {
                self.edges.push((i, single[v as usize].unwrap()));
                self.separators.push(BTreeSet::from([v]));
            }
        }
        true
    }
    pub fn write(&self) -> String {
        format!(
            "<ClusterGraph vars=\"{}\" nodes=\"{}\" edges=\"{}\">\n<VarCards>{}</VarCards>\n<Cliques>{}</Cliques>\n<Edges>{}</Edges>\n</ClusterGraph>\n",
            self.num_variables(),
            self.num_cliques(),
            self.num_edges(),
            self.variable_cards
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(" "),
            self.cliques
                .iter()
                .map(|c| c
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(" "))
                .collect::<Vec<_>>()
                .join(";"),
            self.edges
                .iter()
                .map(|(a, b)| format!("{a} {b}"))
                .collect::<Vec<_>>()
                .join(";")
        )
    }
    pub fn write_file(&self, p: impl AsRef<Path>) -> bool {
        fs::write(p, self.write()).is_ok()
    }
    pub fn read_file(&mut self, p: impl AsRef<Path>) -> bool {
        let Ok(s) = fs::read_to_string(p) else {
            return false;
        };
        let Some(cards) = s
            .split("<VarCards>")
            .nth(1)
            .and_then(|x| x.split("</VarCards>").next())
            .and_then(|x| {
                x.split_whitespace()
                    .map(str::parse)
                    .collect::<Result<Vec<_>, _>>()
                    .ok()
            })
        else {
            return false;
        };
        *self = match Self::with_cards(cards) {
            Ok(x) => x,
            Err(_) => return false,
        };
        true
    }
    fn compute_separator_sets(&mut self) {
        self.separators = self
            .edges
            .iter()
            .map(|&(a, b)| {
                self.cliques[a]
                    .intersection(&self.cliques[b])
                    .copied()
                    .collect()
            })
            .collect()
    }
}

/// Source-named read accessors from `svlClusterGraph.cpp`.  Rust returns
/// `Option` for the C assertion/error boundary rather than exposing invalid
/// clique indices.
pub fn get_clique(graph: &SvlClusterGraph, index: usize) -> Option<&SvlClique> {
    graph.clique(index)
}
pub fn get_sep_set(graph: &SvlClusterGraph, index: usize) -> Option<&SvlClique> {
    graph.separator(index)
}
pub fn get_clique_potential(graph: &SvlClusterGraph, index: usize) -> Option<&SvlFactor> {
    graph.clique_potential(index)
}
pub fn get_potential(graph: &SvlClusterGraph, clique: &SvlClique) -> SvlFactor {
    graph.potential_for_clique(clique)
}
pub fn get_energy(
    graph: &SvlClusterGraph,
    assignment: &[usize],
    log_potentials: bool,
) -> Option<f64> {
    graph.energy(assignment, log_potentials)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn connects_and_decodes() {
        let mut g = SvlClusterGraph::with_uniform_cards(3, 2).unwrap();
        g.add_clique(BTreeSet::from([0, 1]));
        g.add_clique(BTreeSet::from([1, 2]));
        g.connect_graph().unwrap();
        assert!(g.check_run_int_prop());
        g.set_clique_potential(
            0,
            SvlFactor::from_parts(vec![0, 1], vec![2, 2], Some(vec![1., 2., 3., 4.])).unwrap(),
        )
        .unwrap();
        assert_eq!(g.decode_map(None)[0], Some(1));
    }
    #[test]
    fn source_graph_factories_preserve_cardinality_and_clique_construction() {
        assert_eq!(svl_cluster_graph().num_variables(), 0);
        assert_eq!(
            svl_cluster_graph_with_uniform_cards(2, 3)
                .unwrap()
                .variable_cards,
            [3, 3]
        );
        let graph =
            svl_cluster_graph_with_cliques(vec![2, 3], vec![BTreeSet::from([0, 1])]).unwrap();
        assert_eq!(graph.cliques, [BTreeSet::from([0, 1])]);
        assert!(svl_cluster_graph_with_cards(vec![1]).is_err());
    }
    #[test]
    fn bethe_needs_singletons() {
        let mut g = SvlClusterGraph::with_uniform_cards(2, 2).unwrap();
        g.add_clique(BTreeSet::from([0]));
        g.add_clique(BTreeSet::from([1]));
        g.add_clique(BTreeSet::from([0, 1]));
        assert!(g.bethe_approx());
    }
}
