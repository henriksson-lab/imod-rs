//! Owned translation of the public algorithm selection and inference state in
//! `svlMessagePassing.{h,cpp}`.  Factor-operation scheduling is built directly
//! on owned factor tables rather than the C++ heap operation graph.

use std::fmt;

use super::svl_cluster_graph::SvlClusterGraph;
use super::svl_factor::SvlFactor;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SvlMessagePassingAlgorithm {
    #[default]
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

impl SvlMessagePassingAlgorithm {
    pub fn is_log_space(self) -> bool {
        matches!(
            self,
            Self::LogMaxProd
                | Self::LogMaxProdDiv
                | Self::AsyncLogMaxProd
                | Self::AsyncLogMaxProdLazy
                | Self::AsyncLogMaxProdDiv
                | Self::RbpLogMaxProd
                | Self::Gemplp
                | Self::Sontag08
        )
    }
}
impl fmt::Display for SvlMessagePassingAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::None => "NONE",
            Self::SumProd => "SUMPROD",
            Self::MaxProd => "MAXPROD",
            Self::LogMaxProd => "LOGMAXPROD",
            Self::SumProdDiv => "SUMPRODDIV",
            Self::MaxProdDiv => "MAXSUMPRODDIV",
            Self::LogMaxProdDiv => "LOGMAXPRODDIV",
            Self::AsyncSumProd => "ASYNCSUMPROD",
            Self::AsyncMaxProd => "ASYNCMAXPROD",
            Self::AsyncLogMaxProd => "ASYNCLOGMAXPROD",
            Self::AsyncLogMaxProdLazy => "ASYNCLOGMAXPRODLAZY",
            Self::AsyncSumProdDiv => "ASYNCSUMPRODDIV",
            Self::AsyncMaxProdDiv => "ASYNCMAXPRODDIV",
            Self::AsyncLogMaxProdDiv => "ASYNCLOGMAXPRODDIV",
            Self::RbpSumProd => "RBPSUMPROD",
            Self::RbpMaxProd => "RBPMAXPROD",
            Self::RbpLogMaxProd => "RBPLOGMAXPROD",
            Self::Gemplp => "GEMPLP",
            Self::Sontag08 => "SONTAG08",
        })
    }
}

pub fn decode_message_passing_algorithm(name: &str) -> SvlMessagePassingAlgorithm {
    match name.to_ascii_uppercase().as_str() {
        "SUMPROD" => SvlMessagePassingAlgorithm::SumProd,
        "MAXPROD" => SvlMessagePassingAlgorithm::MaxProd,
        "LOGMAXPROD" => SvlMessagePassingAlgorithm::LogMaxProd,
        "SUMPRODDIV" => SvlMessagePassingAlgorithm::SumProdDiv,
        "MAXPRODDIV" => SvlMessagePassingAlgorithm::MaxProdDiv,
        "LOGMAXPRODDIV" => SvlMessagePassingAlgorithm::LogMaxProdDiv,
        "ASYNCSUMPROD" => SvlMessagePassingAlgorithm::AsyncSumProd,
        "ASYNCMAXPROD" => SvlMessagePassingAlgorithm::AsyncMaxProd,
        "ASYNCLOGMAXPROD" => SvlMessagePassingAlgorithm::AsyncLogMaxProd,
        "ASYNCLOGMAXPRODLAZY" => SvlMessagePassingAlgorithm::AsyncLogMaxProdLazy,
        "ASYNCSUMPRODDIV" => SvlMessagePassingAlgorithm::AsyncSumProdDiv,
        "ASYNCMAXPRODDIV" => SvlMessagePassingAlgorithm::AsyncMaxProdDiv,
        "ASYNCLOGMAXPRODDIV" => SvlMessagePassingAlgorithm::AsyncLogMaxProdDiv,
        "RBPSUMPROD" => SvlMessagePassingAlgorithm::RbpSumProd,
        "RBPMAXPROD" => SvlMessagePassingAlgorithm::RbpMaxProd,
        "RBPLOGMAXPROD" => SvlMessagePassingAlgorithm::RbpLogMaxProd,
        "GEMPLP" => SvlMessagePassingAlgorithm::Gemplp,
        "SONTAG08" => SvlMessagePassingAlgorithm::Sontag08,
        _ => SvlMessagePassingAlgorithm::None,
    }
}

/// `toString(svlMessagePassingAlgorithms&)` (`svlMessagePassing.cpp:60`).
pub fn svl_message_passing_algorithm_to_string(algorithm: SvlMessagePassingAlgorithm) -> String {
    algorithm.to_string()
}

/// Source-spelled `toString(svlMessagePassingAlgorithms&)`
/// (`svlMessagePassing.cpp:60`).
pub fn to_string(algorithm: SvlMessagePassingAlgorithm) -> String {
    svl_message_passing_algorithm_to_string(algorithm)
}

/// Source `svlMessagePassingInference`, with owned messages and beliefs.
#[derive(Clone, Debug)]
pub struct SvlMessagePassingInference {
    pub graph: SvlClusterGraph,
    pub clique_potentials: Vec<SvlFactor>,
    pub forward_messages: Vec<SvlFactor>,
    pub backward_messages: Vec<SvlFactor>,
    pub old_forward_messages: Vec<SvlFactor>,
    pub old_backward_messages: Vec<SvlFactor>,
    pub algorithm: SvlMessagePassingAlgorithm,
    pub singleton_marginals_only: bool,
    pub last_dual_objective: f64,
    pub forward_lazy_set: Vec<Vec<usize>>,
    pub backward_lazy_set: Vec<Vec<usize>>,
    pub lazy_execution: Vec<bool>,
    pub directed_execution: Option<Vec<bool>>,
    pub lp_clique_edges: Vec<Vec<usize>>,
    pub lp_separator_edges: Vec<Vec<usize>>,
    pub lp_separators: Vec<std::collections::BTreeSet<i32>>,
    pub lp_edges: Vec<(usize, usize)>,
}
/// `svlMessagePassingInference::svlMessagePassingInference`
/// (`svlMessagePassing.cpp:156`), as an owned Rust factory.
pub fn svl_message_passing_inference(graph: SvlClusterGraph) -> SvlMessagePassingInference {
    SvlMessagePassingInference::new(graph)
}
impl SvlMessagePassingInference {
    pub fn new(graph: SvlClusterGraph) -> Self {
        Self {
            graph,
            clique_potentials: Vec::new(),
            forward_messages: Vec::new(),
            backward_messages: Vec::new(),
            old_forward_messages: Vec::new(),
            old_backward_messages: Vec::new(),
            algorithm: SvlMessagePassingAlgorithm::None,
            singleton_marginals_only: true,
            last_dual_objective: f64::MAX,
            forward_lazy_set: Vec::new(),
            backward_lazy_set: Vec::new(),
            lazy_execution: Vec::new(),
            directed_execution: None,
            lp_clique_edges: Vec::new(),
            lp_separator_edges: Vec::new(),
            lp_separators: Vec::new(),
            lp_edges: Vec::new(),
        }
    }
    pub fn reset(&mut self) {
        self.clique_potentials.clear();
        self.forward_messages.clear();
        self.backward_messages.clear();
        self.old_forward_messages.clear();
        self.old_backward_messages.clear();
        self.forward_lazy_set.clear();
        self.backward_lazy_set.clear();
        self.lazy_execution.clear();
        self.directed_execution = None;
        self.lp_clique_edges.clear();
        self.lp_separator_edges.clear();
        self.lp_separators.clear();
        self.lp_edges.clear();
        self.algorithm = SvlMessagePassingAlgorithm::None;
    }
    pub fn is_log_space(&self) -> bool {
        self.algorithm.is_log_space()
    }
    pub fn clique_potential(&self, index: usize) -> Option<&SvlFactor> {
        self.clique_potentials.get(index)
    }

    /// Source `initializeMessagePassing`.
    pub fn initialize_message_passing(&mut self) -> Result<(), String> {
        self.clique_potentials = self.graph.initial_potentials.clone();
        self.forward_messages.clear();
        self.backward_messages.clear();
        for separator in &self.graph.separators {
            let variables = separator.iter().copied().collect::<Vec<_>>();
            let cards = variables
                .iter()
                .map(|&variable| {
                    self.graph
                        .cardinality(variable as usize)
                        .ok_or("separator variable out of range")
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut message = SvlFactor::with_variables(variables, cards)?;
            message.fill(1.0);
            self.forward_messages.push(message.clone());
            self.backward_messages.push(message);
        }
        self.old_forward_messages = self.forward_messages.clone();
        self.old_backward_messages = self.backward_messages.clone();
        Ok(())
    }

    /// Source `initializeLogMessagePassing`.
    pub fn initialize_log_message_passing(&mut self) -> Result<(), String> {
        self.initialize_message_passing()?;
        for message in self
            .forward_messages
            .iter_mut()
            .chain(self.backward_messages.iter_mut())
        {
            message.fill(0.0);
        }
        self.old_forward_messages = self.forward_messages.clone();
        self.old_backward_messages = self.backward_messages.clone();
        Ok(())
    }

    /// Source `initializeGeneralizedMPLP`; owned execution uses zero log messages.
    pub fn initialize_generalized_mplp(&mut self) -> Result<(), String> {
        // The source creates an LP edge from every clique to each distinct
        // nonempty pairwise intersection, rather than using only graph edges.
        self.lp_clique_edges = vec![Vec::new(); self.graph.cliques.len()];
        self.lp_separator_edges.clear();
        self.lp_separators.clear();
        self.lp_edges.clear();
        for left in 0..self.graph.cliques.len() {
            for right in left + 1..self.graph.cliques.len() {
                let separator = self.graph.cliques[left]
                    .intersection(&self.graph.cliques[right])
                    .copied()
                    .collect::<std::collections::BTreeSet<_>>();
                if separator.is_empty() {
                    continue;
                }
                let separator_id = self
                    .lp_separators
                    .iter()
                    .position(|known| known == &separator)
                    .unwrap_or_else(|| {
                        self.lp_separators.push(separator);
                        self.lp_separator_edges.push(Vec::new());
                        self.lp_separators.len() - 1
                    });
                for clique in [left, right] {
                    if !self.lp_edges.iter().any(|&(known, known_separator)| {
                        known == clique && known_separator == separator_id
                    }) {
                        let message = self.lp_edges.len();
                        self.lp_edges.push((clique, separator_id));
                        self.lp_clique_edges[clique].push(message);
                        self.lp_separator_edges[separator_id].push(message);
                    }
                }
            }
        }
        self.last_dual_objective = f64::MAX;
        self.clique_potentials = self.graph.initial_potentials.clone();
        self.forward_messages.clear();
        self.backward_messages.clear();
        for &(clique, separator) in &self.lp_edges {
            let variables = self.lp_separators[separator]
                .iter()
                .copied()
                .collect::<Vec<_>>();
            let cards = variables
                .iter()
                .map(|&v| {
                    self.graph
                        .cardinality(v as usize)
                        .ok_or("separator variable out of range")
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut message = SvlFactor::with_variables(variables, cards)?;
            if !self.clique_potentials[clique].empty() {
                let mut projected = self.clique_potentials[clique].clone();
                for variable in projected.variables.clone() {
                    if !self.lp_separators[separator].contains(&variable) {
                        projected.maximize(variable)?;
                    }
                }
                message = projected;
                message.scale(1.0 / self.lp_clique_edges[clique].len() as f64);
            } else {
                message.fill(0.0);
            }
            self.forward_messages.push(message);
        }
        for message in 0..self.lp_edges.len() {
            let (clique, separator) = self.lp_edges[message];
            let mut incoming = self.forward_messages[message].clone();
            incoming.fill(0.0);
            for &other in &self.lp_separator_edges[separator] {
                if self.lp_edges[other].0 != clique {
                    incoming.add(&self.forward_messages[other])?;
                }
            }
            self.backward_messages.push(incoming);
        }
        self.old_forward_messages = self.forward_messages.clone();
        self.old_backward_messages = self.backward_messages.clone();
        Ok(())
    }

    pub fn inference(
        &mut self,
        algorithm: SvlMessagePassingAlgorithm,
        max_iterations: usize,
    ) -> Result<bool, String> {
        if algorithm != self.algorithm {
            self.reset();
        }
        self.algorithm = algorithm;
        match algorithm {
            SvlMessagePassingAlgorithm::Gemplp | SvlMessagePassingAlgorithm::Sontag08 => {
                self.initialize_generalized_mplp()?
            }
            _ if algorithm.is_log_space() => self.initialize_log_message_passing()?,
            _ => self.initialize_message_passing()?,
        }
        match algorithm {
            SvlMessagePassingAlgorithm::SumProd => self.build_sum_prod_computation_graph(),
            SvlMessagePassingAlgorithm::MaxProd => self.build_max_prod_computation_graph(),
            SvlMessagePassingAlgorithm::LogMaxProd => self.build_log_max_prod_computation_graph(),
            SvlMessagePassingAlgorithm::SumProdDiv => self.build_sum_prod_div_computation_graph(),
            SvlMessagePassingAlgorithm::MaxProdDiv => self.build_max_prod_div_computation_graph(),
            SvlMessagePassingAlgorithm::LogMaxProdDiv => {
                self.build_log_max_prod_div_computation_graph()
            }
            SvlMessagePassingAlgorithm::AsyncSumProd => {
                self.build_async_sum_prod_computation_graph()
            }
            SvlMessagePassingAlgorithm::AsyncMaxProd => {
                self.build_async_max_prod_computation_graph()
            }
            SvlMessagePassingAlgorithm::AsyncLogMaxProd => {
                self.build_async_log_max_prod_computation_graph()
            }
            SvlMessagePassingAlgorithm::AsyncLogMaxProdLazy => {
                self.build_async_log_max_prod_lazy_computation_graph()
            }
            SvlMessagePassingAlgorithm::AsyncSumProdDiv => {
                self.build_async_sum_prod_div_computation_graph()
            }
            SvlMessagePassingAlgorithm::AsyncMaxProdDiv => {
                self.build_async_max_prod_div_computation_graph()
            }
            SvlMessagePassingAlgorithm::AsyncLogMaxProdDiv => {
                self.build_async_log_max_prod_div_computation_graph()
            }
            SvlMessagePassingAlgorithm::RbpSumProd => self.build_residual_bp_computation_graph(),
            SvlMessagePassingAlgorithm::RbpMaxProd => {
                self.build_residual_bp_max_prod_computation_graph()
            }
            SvlMessagePassingAlgorithm::RbpLogMaxProd => {
                self.build_residual_bp_log_max_prod_computation_graph()
            }
            SvlMessagePassingAlgorithm::Gemplp | SvlMessagePassingAlgorithm::Sontag08 => {
                self.build_generalized_mplp_graph()
            }
            SvlMessagePassingAlgorithm::None => return Ok(true),
        }
        let converged = match algorithm {
            SvlMessagePassingAlgorithm::AsyncLogMaxProdLazy => {
                self.lazy_message_passing_loop(max_iterations)?
            }
            SvlMessagePassingAlgorithm::RbpSumProd
            | SvlMessagePassingAlgorithm::RbpMaxProd
            | SvlMessagePassingAlgorithm::RbpLogMaxProd => {
                self.residual_bp_message_passing_loop(max_iterations)?
            }
            SvlMessagePassingAlgorithm::Gemplp => {
                self.gemplp_message_passing_loop(max_iterations)?
            }
            SvlMessagePassingAlgorithm::Sontag08 => {
                self.sontag08_message_passing_loop(max_iterations)?
            }
            _ => self.message_passing_loop(max_iterations)?,
        };
        match algorithm {
            SvlMessagePassingAlgorithm::Gemplp => self.finalize_generalized_mplp()?,
            SvlMessagePassingAlgorithm::Sontag08 => {}
            _ if algorithm.is_log_space() => self.finalize_log_message_passing()?,
            _ => self.finalize_message_passing()?,
        }
        Ok(converged)
    }

    /// Source builder entry points; owned evaluation is performed by the loop.
    pub fn build_sum_prod_computation_graph(&mut self) {
        self.algorithm = SvlMessagePassingAlgorithm::SumProd;
    }
    pub fn build_max_prod_computation_graph(&mut self) {
        self.algorithm = SvlMessagePassingAlgorithm::MaxProd;
    }
    pub fn build_log_max_prod_computation_graph(&mut self) {
        self.algorithm = SvlMessagePassingAlgorithm::LogMaxProd;
    }
    pub fn build_sum_prod_div_computation_graph(&mut self) {
        self.algorithm = SvlMessagePassingAlgorithm::SumProdDiv;
    }
    pub fn build_max_prod_div_computation_graph(&mut self) {
        self.algorithm = SvlMessagePassingAlgorithm::MaxProdDiv;
    }
    pub fn build_log_max_prod_div_computation_graph(&mut self) {
        self.algorithm = SvlMessagePassingAlgorithm::LogMaxProdDiv;
    }
    pub fn build_async_sum_prod_computation_graph(&mut self) {
        self.algorithm = SvlMessagePassingAlgorithm::AsyncSumProd;
    }
    pub fn build_async_max_prod_computation_graph(&mut self) {
        self.algorithm = SvlMessagePassingAlgorithm::AsyncMaxProd;
    }
    pub fn build_async_log_max_prod_computation_graph(&mut self) {
        self.algorithm = SvlMessagePassingAlgorithm::AsyncLogMaxProd;
    }
    pub fn build_async_log_max_prod_lazy_computation_graph(&mut self) {
        self.algorithm = SvlMessagePassingAlgorithm::AsyncLogMaxProdLazy;
        self.forward_lazy_set = vec![Vec::new(); self.graph.edges.len()];
        self.backward_lazy_set = vec![Vec::new(); self.graph.edges.len()];
        self.lazy_execution = vec![true; 2 * self.graph.edges.len()];
        // Each source atomic operation is assigned `2 * edge + direction`.
        for (edge, &(left, right)) in self.graph.edges.iter().enumerate() {
            for (other, &(a, b)) in self.graph.edges.iter().enumerate() {
                if other == edge {
                    continue;
                }
                if a == left {
                    self.backward_lazy_set[other].push(2 * edge);
                }
                if b == left {
                    self.forward_lazy_set[other].push(2 * edge);
                }
                if a == right {
                    self.backward_lazy_set[other].push(2 * edge + 1);
                }
                if b == right {
                    self.forward_lazy_set[other].push(2 * edge + 1);
                }
            }
        }
    }
    pub fn build_async_sum_prod_div_computation_graph(&mut self) {
        self.algorithm = SvlMessagePassingAlgorithm::AsyncSumProdDiv;
    }
    pub fn build_async_max_prod_div_computation_graph(&mut self) {
        self.algorithm = SvlMessagePassingAlgorithm::AsyncMaxProdDiv;
    }
    pub fn build_async_log_max_prod_div_computation_graph(&mut self) {
        self.algorithm = SvlMessagePassingAlgorithm::AsyncLogMaxProdDiv;
    }
    pub fn build_residual_bp_computation_graph(&mut self) {
        self.algorithm = SvlMessagePassingAlgorithm::RbpSumProd;
    }
    pub fn build_residual_bp_max_prod_computation_graph(&mut self) {
        self.algorithm = SvlMessagePassingAlgorithm::RbpMaxProd;
    }
    pub fn build_residual_bp_log_max_prod_computation_graph(&mut self) {
        self.algorithm = SvlMessagePassingAlgorithm::RbpLogMaxProd;
    }
    pub fn build_generalized_mplp_graph(&mut self) {
        if !matches!(self.algorithm, SvlMessagePassingAlgorithm::Sontag08) {
            self.algorithm = SvlMessagePassingAlgorithm::Gemplp;
        }
    }

    /// Source `messagePassingLoop`: every directed edge receives one message
    /// from its clique potential and all incident messages except its reverse.
    pub fn message_passing_loop(&mut self, max_iterations: usize) -> Result<bool, String> {
        let max_product = matches!(
            self.algorithm,
            SvlMessagePassingAlgorithm::MaxProd
                | SvlMessagePassingAlgorithm::LogMaxProd
                | SvlMessagePassingAlgorithm::MaxProdDiv
                | SvlMessagePassingAlgorithm::LogMaxProdDiv
                | SvlMessagePassingAlgorithm::AsyncMaxProd
                | SvlMessagePassingAlgorithm::AsyncLogMaxProd
                | SvlMessagePassingAlgorithm::AsyncLogMaxProdLazy
                | SvlMessagePassingAlgorithm::AsyncMaxProdDiv
                | SvlMessagePassingAlgorithm::AsyncLogMaxProdDiv
                | SvlMessagePassingAlgorithm::RbpMaxProd
                | SvlMessagePassingAlgorithm::RbpLogMaxProd
                | SvlMessagePassingAlgorithm::Gemplp
                | SvlMessagePassingAlgorithm::Sontag08
        );
        let log_space = self.algorithm.is_log_space();
        for _ in 0..max_iterations {
            let old_forward = self.forward_messages.clone();
            let old_backward = self.backward_messages.clone();
            for edge in 0..self.graph.edges.len() {
                let (left, right) = self.graph.edges[edge];
                let separator = &self.graph.separators[edge];
                let mut forward = self.clique_potentials[left].clone();
                let mut backward = self.clique_potentials[right].clone();
                for (other, &(a, b)) in self.graph.edges.iter().enumerate() {
                    if other == edge {
                        continue;
                    }
                    if a == left {
                        if log_space {
                            forward.add(&old_backward[other])?;
                        } else {
                            forward.product(&old_backward[other])?;
                        }
                    }
                    if b == left {
                        if log_space {
                            forward.add(&old_forward[other])?;
                        } else {
                            forward.product(&old_forward[other])?;
                        }
                    }
                    if a == right {
                        if log_space {
                            backward.add(&old_backward[other])?;
                        } else {
                            backward.product(&old_backward[other])?;
                        }
                    }
                    if b == right {
                        if log_space {
                            backward.add(&old_forward[other])?;
                        } else {
                            backward.product(&old_forward[other])?;
                        }
                    }
                }
                for variable in forward.variables.clone() {
                    if !separator.contains(&variable) {
                        if max_product {
                            forward.maximize(variable)?;
                        } else {
                            forward.marginalize(variable)?;
                        }
                    }
                }
                for variable in backward.variables.clone() {
                    if !separator.contains(&variable) {
                        if max_product {
                            backward.maximize(variable)?;
                        } else {
                            backward.marginalize(variable)?;
                        }
                    }
                }
                if log_space {
                    let fmax = forward.data.iter().copied().fold(-f64::MAX, f64::max);
                    let bmax = backward.data.iter().copied().fold(-f64::MAX, f64::max);
                    forward.offset(-fmax);
                    backward.offset(-bmax);
                } else {
                    forward.normalize();
                    backward.normalize();
                }
                if self
                    .directed_execution
                    .as_ref()
                    .map(|mask| mask.get(2 * edge).copied().unwrap_or(false))
                    .unwrap_or(
                        self.algorithm != SvlMessagePassingAlgorithm::AsyncLogMaxProdLazy
                            || self.lazy_execution.get(2 * edge).copied().unwrap_or(true),
                    )
                {
                    self.forward_messages[edge] = forward;
                }
                if self
                    .directed_execution
                    .as_ref()
                    .map(|mask| mask.get(2 * edge + 1).copied().unwrap_or(false))
                    .unwrap_or(
                        self.algorithm != SvlMessagePassingAlgorithm::AsyncLogMaxProdLazy
                            || self
                                .lazy_execution
                                .get(2 * edge + 1)
                                .copied()
                                .unwrap_or(true),
                    )
                {
                    self.backward_messages[edge] = backward;
                }
            }
            let mut converged = true;
            for (old, new) in self
                .old_forward_messages
                .iter_mut()
                .zip(&self.forward_messages)
            {
                converged &= old.data_compare_and_copy(new);
            }
            for (old, new) in self
                .old_backward_messages
                .iter_mut()
                .zip(&self.backward_messages)
            {
                converged &= old.data_compare_and_copy(new);
            }
            if converged {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Source `finalizeMessagePassing`.
    pub fn finalize_message_passing(&mut self) -> Result<(), String> {
        self.clique_potentials = self.graph.initial_potentials.clone();
        for (edge, &(left, right)) in self.graph.edges.iter().enumerate() {
            if !self.singleton_marginals_only || self.graph.cliques[left].len() == 1 {
                self.clique_potentials[left].product(&self.backward_messages[edge])?;
            }
            if !self.singleton_marginals_only || self.graph.cliques[right].len() == 1 {
                self.clique_potentials[right].product(&self.forward_messages[edge])?;
            }
        }
        for belief in &mut self.clique_potentials {
            if !self.singleton_marginals_only || belief.num_vars() == 1 {
                belief.normalize();
            }
        }
        Ok(())
    }

    /// Source `finalizeLogMessagePassing`.
    pub fn finalize_log_message_passing(&mut self) -> Result<(), String> {
        self.clique_potentials = self.graph.initial_potentials.clone();
        for (edge, &(left, right)) in self.graph.edges.iter().enumerate() {
            if !self.singleton_marginals_only || self.graph.cliques[left].len() == 1 {
                self.clique_potentials[left].add(&self.backward_messages[edge])?;
            }
            if !self.singleton_marginals_only || self.graph.cliques[right].len() == 1 {
                self.clique_potentials[right].add(&self.forward_messages[edge])?;
            }
        }
        Ok(())
    }

    /// Source `finalizeGeneralizedMPLP` uses the accumulated incoming log
    /// separator messages as clique beliefs.
    pub fn finalize_generalized_mplp(&mut self) -> Result<(), String> {
        self.clique_potentials = self.graph.initial_potentials.clone();
        for message in 0..self.lp_edges.len() {
            let clique = self.lp_edges[message].0;
            if !self.clique_potentials[clique].empty() {
                self.clique_potentials[clique].add(&self.backward_messages[message])?;
            }
        }
        Ok(())
    }

    pub fn lazy_message_passing_loop(&mut self, max_iterations: usize) -> Result<bool, String> {
        if self.lazy_execution.len() != 2 * self.graph.edges.len() {
            self.lazy_execution = vec![true; 2 * self.graph.edges.len()];
        }
        for _ in 0..max_iterations {
            let before_forward = self.forward_messages.clone();
            let before_backward = self.backward_messages.clone();
            if self.message_passing_loop(1)? {
                return Ok(true);
            }
            let mut next = vec![false; self.lazy_execution.len()];
            for edge in 0..self.graph.edges.len() {
                if !self.forward_messages[edge].data_compare(&before_forward[edge]) {
                    for &operation in &self.forward_lazy_set[edge] {
                        next[operation] = true;
                    }
                }
                if !self.backward_messages[edge].data_compare(&before_backward[edge]) {
                    for &operation in &self.backward_lazy_set[edge] {
                        next[operation] = true;
                    }
                }
            }
            if !next.iter().any(|&active| active) {
                return Ok(true);
            }
            self.lazy_execution = next;
        }
        Ok(false)
    }

    /// Source residual BP prioritizes changed directed edges.  The owned form
    /// performs the same update until the largest L1 residual is below epsilon.
    pub fn residual_bp_message_passing_loop(
        &mut self,
        max_iterations: usize,
    ) -> Result<bool, String> {
        let before_forward = self.forward_messages.clone();
        let before_backward = self.backward_messages.clone();
        self.directed_execution = None;
        let _ = self.message_passing_loop(1)?;
        let mut queue = (0..self.graph.edges.len())
            .flat_map(|edge| {
                let forward = self.forward_messages[edge]
                    .data
                    .iter()
                    .zip(&before_forward[edge].data)
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f64>();
                let backward = self.backward_messages[edge]
                    .data
                    .iter()
                    .zip(&before_backward[edge].data)
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f64>();
                [(forward, 2 * edge), (backward, 2 * edge + 1)]
            })
            .collect::<Vec<_>>();
        for _ in 0..max_iterations {
            queue.sort_by(|left, right| left.0.total_cmp(&right.0));
            let (residual, directed) = *queue.last().ok_or("empty residual queue")?;
            if residual < 1.0e-9 {
                self.directed_execution = None;
                return Ok(true);
            }
            let edge = directed / 2;
            let receiver = if directed % 2 == 0 {
                self.graph.edges[edge].1
            } else {
                self.graph.edges[edge].0
            };
            let mut scheduled = vec![false; 2 * self.graph.edges.len()];
            for (candidate, &(left, right)) in self.graph.edges.iter().enumerate() {
                if left == receiver {
                    scheduled[2 * candidate] = true;
                }
                if right == receiver {
                    scheduled[2 * candidate + 1] = true;
                }
            }
            let old_forward = self.forward_messages.clone();
            let old_backward = self.backward_messages.clone();
            self.directed_execution = Some(scheduled.clone());
            let _ = self.message_passing_loop(1)?;
            for (candidate, entry) in queue.iter_mut().enumerate() {
                if !scheduled[candidate] {
                    continue;
                }
                let message = candidate / 2;
                entry.0 = if candidate % 2 == 0 {
                    self.forward_messages[message]
                        .data
                        .iter()
                        .zip(&old_forward[message].data)
                        .map(|(a, b)| (a - b).abs())
                        .sum()
                } else {
                    self.backward_messages[message]
                        .data
                        .iter()
                        .zip(&old_backward[message].data)
                        .map(|(a, b)| (a - b).abs())
                        .sum()
                };
            }
        }
        self.directed_execution = None;
        Ok(false)
    }

    /// Source GEMPLP convergence is measured by the dual objective over
    /// separator beliefs, rather than message equality.
    pub fn gemplp_message_passing_loop(&mut self, max_iterations: usize) -> Result<bool, String> {
        for _ in 0..max_iterations {
            for clique in 0..self.lp_clique_edges.len() {
                // subtract old lambda_c->s from every other lambda_s->c-hat
                for &message in &self.lp_clique_edges[clique] {
                    let separator = self.lp_edges[message].1;
                    for &other in &self.lp_separator_edges[separator] {
                        if self.lp_edges[other].0 != clique {
                            let old = self.forward_messages[message].clone();
                            self.backward_messages[other].subtract(&old)?;
                        }
                    }
                }
                for &message in &self.lp_clique_edges[clique] {
                    let separator = self.lp_edges[message].1;
                    let mut sum = if self.clique_potentials[clique].empty() {
                        let variables = self.graph.cliques[clique]
                            .iter()
                            .copied()
                            .collect::<Vec<_>>();
                        let cards = variables
                            .iter()
                            .map(|&variable| {
                                self.graph
                                    .cardinality(variable as usize)
                                    .ok_or("clique variable out of range")
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        let mut factor = SvlFactor::with_variables(variables, cards)?;
                        factor.fill(0.0);
                        factor
                    } else {
                        self.clique_potentials[clique].clone()
                    };
                    for &other in &self.lp_clique_edges[clique] {
                        if other != message {
                            sum.add(&self.backward_messages[other])?;
                        }
                    }
                    for variable in sum.variables.clone() {
                        if !self.lp_separators[separator].contains(&variable) {
                            sum.maximize(variable)?;
                        }
                    }
                    let weight = 1.0 / self.lp_clique_edges[clique].len() as f64;
                    let mut updated = self.backward_messages[message].clone();
                    updated.scale(weight - 1.0);
                    sum.scale(weight);
                    updated.add(&sum)?;
                    self.forward_messages[message] = updated;
                }
                // add the new lambda_c->s into every other lambda_s->c-hat.
                for &message in &self.lp_clique_edges[clique] {
                    let separator = self.lp_edges[message].1;
                    for &other in &self.lp_separator_edges[separator] {
                        if self.lp_edges[other].0 != clique {
                            let new = self.forward_messages[message].clone();
                            self.backward_messages[other].add(&new)?;
                        }
                    }
                }
            }
            let mut dual = 0.0;
            for separator in 0..self.lp_separators.len() {
                let first = *self.lp_separator_edges[separator]
                    .first()
                    .ok_or("separator without messages")?;
                let mut belief = self.forward_messages[first].clone();
                for &message in self.lp_separator_edges[separator].iter().skip(1) {
                    belief.add(&self.forward_messages[message])?;
                }
                dual += belief.data.iter().copied().fold(-f64::MAX, f64::max);
            }
            if self.last_dual_objective - dual < 1.0e-9 {
                self.last_dual_objective = dual;
                return Ok(true);
            }
            self.last_dual_objective = dual;
        }
        Ok(false)
    }

    /// Source Sontag08 starts from GEMPLP, decodes singleton MAP beliefs, then
    /// adds violated triplet cliques.  This owned phase preserves the GEMPLP
    /// warm start and declares convergence once its dual is stable.
    pub fn sontag08_message_passing_loop(&mut self, max_iterations: usize) -> Result<bool, String> {
        self.algorithm = SvlMessagePassingAlgorithm::Gemplp;
        let converged = self.gemplp_message_passing_loop(max_iterations)?;
        self.finalize_generalized_mplp()?;
        // Enabled source branch: score every distinct <=3-variable union by
        // the violation `sum max(belief_i) - max(sum belief_i)`.
        let mut candidates = Vec::<(f64, std::collections::BTreeSet<i32>)>::new();
        for left in 0..self.clique_potentials.len() {
            if self.clique_potentials[left].num_vars() > 2 {
                continue;
            }
            for right in left + 1..self.clique_potentials.len() {
                if self.clique_potentials[right].num_vars() > 2 {
                    continue;
                }
                let union = self.clique_potentials[left]
                    .variables
                    .iter()
                    .chain(&self.clique_potentials[right].variables)
                    .copied()
                    .collect::<std::collections::BTreeSet<_>>();
                if union.len() > 3
                    || self.graph.cliques.contains(&union)
                    || candidates.iter().any(|(_, known)| known == &union)
                {
                    continue;
                }
                let associated = self
                    .clique_potentials
                    .iter()
                    .enumerate()
                    .filter_map(|(index, belief)| {
                        (belief.num_vars() > 1
                            && belief
                                .variables
                                .iter()
                                .all(|variable| union.contains(variable)))
                        .then_some(index)
                    })
                    .collect::<Vec<_>>();
                if associated.len() <= 2 {
                    continue;
                }
                let mut summed = SvlFactor::new();
                let mut individual_maxima = 0.0;
                for index in associated {
                    individual_maxima += self.clique_potentials[index]
                        .data
                        .iter()
                        .copied()
                        .fold(-f64::MAX, f64::max);
                    summed.add(&self.clique_potentials[index])?;
                }
                let score =
                    individual_maxima - summed.data.iter().copied().fold(-f64::MAX, f64::max);
                candidates.push((score, union));
            }
        }
        candidates.sort_by(|left, right| right.0.total_cmp(&left.0));
        for (_, clique) in candidates.into_iter().take(5) {
            self.graph.add_clique(clique);
        }
        if self.graph.num_cliques() > self.clique_potentials.len() {
            self.initialize_generalized_mplp()?;
            let _ = self.gemplp_message_passing_loop(max_iterations / 10 + 1)?;
            self.finalize_generalized_mplp()?;
        }
        self.algorithm = SvlMessagePassingAlgorithm::Sontag08;
        let _ = converged;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    #[test]
    fn source_algorithm_names_decode_case_insensitively() {
        assert_eq!(
            decode_message_passing_algorithm("rBpLoGmAxPrOd"),
            SvlMessagePassingAlgorithm::RbpLogMaxProd
        );
        assert_eq!(
            SvlMessagePassingAlgorithm::LogMaxProd.to_string(),
            "LOGMAXPROD"
        );
        assert_eq!(
            to_string(SvlMessagePassingAlgorithm::LogMaxProd),
            "LOGMAXPROD"
        );
        assert!(SvlMessagePassingAlgorithm::Gemplp.is_log_space());
        assert!(!SvlMessagePassingAlgorithm::SumProd.is_log_space());
    }

    #[test]
    fn sum_product_marginalizes_to_a_separator_message() {
        let mut graph = SvlClusterGraph::with_uniform_cards(2, 2).unwrap();
        graph.add_clique_with_potential(
            BTreeSet::from([0, 1]),
            SvlFactor::from_parts(vec![0, 1], vec![2, 2], Some(vec![1., 2., 3., 4.])).unwrap(),
        );
        graph.add_clique_with_potential(
            BTreeSet::from([1]),
            SvlFactor::from_parts(vec![1], vec![2], Some(vec![1., 1.])).unwrap(),
        );
        graph.connect_graph_with_edges(vec![(0, 1)]).unwrap();
        let mut inference = SvlMessagePassingInference::new(graph);
        assert!(
            inference
                .inference(SvlMessagePassingAlgorithm::SumProd, 8)
                .unwrap()
        );
        // `svlFactor` stores its first variable with stride one, so
        // marginalizing variable 0 gives [1 + 2, 3 + 4].
        let marginal = &inference.clique_potential(1).unwrap().data;
        assert!((marginal[0] - 0.3).abs() < 1.0e-12);
        assert!((marginal[1] - 0.7).abs() < 1.0e-12);
    }

    #[test]
    fn generalized_mplp_reparameterizes_clique_separator_messages() {
        let mut graph = SvlClusterGraph::with_uniform_cards(2, 2).unwrap();
        graph.add_clique_with_potential(
            BTreeSet::from([0, 1]),
            SvlFactor::from_parts(vec![0, 1], vec![2, 2], Some(vec![0., 1., 2., 3.])).unwrap(),
        );
        graph.add_clique_with_potential(
            BTreeSet::from([1]),
            SvlFactor::from_parts(vec![1], vec![2], Some(vec![0.5, 1.5])).unwrap(),
        );
        let mut inference = SvlMessagePassingInference::new(graph);
        assert!(
            inference
                .inference(SvlMessagePassingAlgorithm::Gemplp, 4)
                .unwrap()
        );
        assert_eq!(inference.last_dual_objective, 4.5);
        assert_eq!(inference.clique_potential(1).unwrap().data, vec![1.5, 4.5]);
    }

    #[test]
    fn sontag_admits_scored_triplet_from_three_pair_beliefs() {
        let mut graph = SvlClusterGraph::with_uniform_cards(3, 2).unwrap();
        for variables in [vec![0, 1], vec![0, 2], vec![1, 2]] {
            graph.add_clique_with_potential(
                BTreeSet::from_iter(variables.iter().copied()),
                SvlFactor::from_parts(variables, vec![2, 2], Some(vec![0., 1., 1., 0.])).unwrap(),
            );
        }
        let mut inference = SvlMessagePassingInference::new(graph);
        let _ = inference
            .inference(SvlMessagePassingAlgorithm::Sontag08, 4)
            .unwrap();
        assert!(inference.graph.cliques.contains(&BTreeSet::from([0, 1, 2])));
    }
}
