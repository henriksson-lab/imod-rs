//! Owned translation of `IMOD/raptor/lasik/svl/lib/pgm/svlFactorOperations.{h,cpp}`.

use std::collections::BTreeSet;

use super::svl_factor::SvlFactor;

/// C++ `svlFactorIndexCache`, whose shared-pointer values are naturally owned
/// vectors in Rust.
#[derive(Clone, Debug, Default)]
pub struct SvlFactorIndexCache {
    cache: Vec<Vec<usize>>,
}

impl SvlFactorIndexCache {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn clear(&mut self) {
        self.cache.clear();
    }
    pub fn find(&mut self, index: &[usize], use_shared_index_cache: bool) -> Vec<usize> {
        if !use_shared_index_cache {
            return index.to_vec();
        }
        if let Some(existing) = self
            .cache
            .iter()
            .find(|existing| existing.as_slice() == index)
        {
            return existing.clone();
        }
        self.cache.push(index.to_vec());
        index.to_vec()
    }
}

/// Configuration retained from `svlFactorOperation` and its config module.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SvlFactorOperationsConfig {
    pub cache_index_mapping: bool,
    pub use_shared_index_cache: bool,
}
impl Default for SvlFactorOperationsConfig {
    fn default() -> Self {
        Self {
            cache_index_mapping: true,
            use_shared_index_cache: false,
        }
    }
}
impl SvlFactorOperationsConfig {
    pub const NAME: &'static str = "svlPGM.svlFactorOperations";
    pub fn usage(self) -> String {
        format!(
            "      cacheIndexMapping :: default: {}\n      useSharedIndexCache :: default: {}\n",
            self.cache_index_mapping, self.use_shared_index_cache
        )
    }
    pub fn set_configuration(&mut self, name: &str, value: &str) -> Result<(), &'static str> {
        let value = value.eq_ignore_ascii_case("true") || value == "1";
        match name {
            "cacheIndexMapping" => self.cache_index_mapping = value,
            "useSharedIndexCache" => self.use_shared_index_cache = value,
            _ => return Err("unrecognized configuration option for svlPGM.svlFactorOperations"),
        }
        Ok(())
    }
}

fn target_value(
    source: &SvlFactor,
    target: &SvlFactor,
    target_index: usize,
) -> Result<f64, String> {
    if source.empty() {
        return Ok(1.0);
    }
    let assignment = target
        .assignment_of(target_index)
        .ok_or("invalid target factor index")?;
    let mut source_assignment = Vec::with_capacity(source.num_vars());
    for variable in &source.variables {
        let position = target
            .variables
            .iter()
            .position(|candidate| candidate == variable)
            .ok_or("target missing factor variable")?;
        source_assignment.push(assignment[position]);
    }
    Ok(source.data[source
        .index_of(&source_assignment)
        .ok_or("invalid factor mapping")?])
}

fn initialize_target(target: &mut SvlFactor, factors: &[&SvlFactor]) -> Result<(), String> {
    if target.empty() {
        for factor in factors {
            target.add_factor_variables(factor)?;
        }
    }
    for factor in factors {
        for (position, variable) in factor.variables.iter().enumerate() {
            if target.var_cardinality(*variable) != Some(factor.cards[position]) {
                return Err("factor target variables or cardinalities differ".into());
            }
        }
    }
    Ok(())
}

/// `svlFactorCopyOp::execute`.
pub fn factor_copy(target: &mut SvlFactor, source: &SvlFactor) -> Result<(), String> {
    if target.empty() {
        *target = source.clone();
    }
    if target.variables != source.variables || target.cards != source.cards {
        return Err("factor target variables or cardinalities differ".into());
    }
    target.data.copy_from_slice(&source.data);
    Ok(())
}

/// `svlFactorProductOp::execute` for the source's binary and n-ary constructors.
pub fn factor_product(target: &mut SvlFactor, factors: &[&SvlFactor]) -> Result<(), String> {
    if factors.is_empty() {
        return Err("product requires a factor".into());
    }
    initialize_target(target, factors)?;
    for index in 0..target.size() {
        let mut value = 1.0;
        for factor in factors {
            if !factor.empty() {
                value *= target_value(factor, target, index)?;
            }
        }
        target.data[index] = value;
    }
    Ok(())
}

/// `svlFactorDivideOp::execute`; source numerator zero produces zero even for
/// a zero denominator.
pub fn factor_divide(
    target: &mut SvlFactor,
    numerator: &SvlFactor,
    denominator: &SvlFactor,
) -> Result<(), String> {
    initialize_target(target, &[numerator, denominator])?;
    for index in 0..target.size() {
        let value = target_value(numerator, target, index)?;
        target.data[index] = if value == 0.0 {
            0.0
        } else {
            value / target_value(denominator, target, index)?
        };
    }
    Ok(())
}

/// `svlFactorAdditionOp::execute` for binary and n-ary factors.
pub fn factor_addition(target: &mut SvlFactor, factors: &[&SvlFactor]) -> Result<(), String> {
    if factors.is_empty() {
        return Err("addition requires a factor".into());
    }
    initialize_target(target, factors)?;
    for index in 0..target.size() {
        let mut value = 0.0;
        for factor in factors {
            if !factor.empty() {
                value += target_value(factor, target, index)?;
            }
        }
        target.data[index] = value;
    }
    Ok(())
}

/// `svlFactorSubtractOp::execute`.
pub fn factor_subtract(
    target: &mut SvlFactor,
    first: &SvlFactor,
    second: &SvlFactor,
) -> Result<(), String> {
    initialize_target(target, &[first, second])?;
    for index in 0..target.size() {
        target.data[index] = match (first.empty(), second.empty()) {
            (true, true) => 0.0,
            (true, false) => -target_value(second, target, index)?,
            (false, true) => target_value(first, target, index)?,
            (false, false) => {
                target_value(first, target, index)? - target_value(second, target, index)?
            }
        };
    }
    Ok(())
}

/// `svlFactorWeightedSumOp::execute`.
pub fn factor_weighted_sum(
    target: &mut SvlFactor,
    first: &SvlFactor,
    second: &SvlFactor,
    first_weight: f64,
    second_weight: f64,
) -> Result<(), String> {
    initialize_target(target, &[first, second])?;
    for index in 0..target.size() {
        target.data[index] = match (first.empty(), second.empty()) {
            (true, true) => 0.0,
            (true, false) => second_weight * target_value(second, target, index)?,
            (false, true) => first_weight * target_value(first, target, index)?,
            (false, false) => {
                first_weight * target_value(first, target, index)?
                    + second_weight * target_value(second, target, index)?
            }
        };
    }
    Ok(())
}

/// `svlFactorMarginalizeOp::execute`. `eliminated` is the source constructor's
/// variable/set argument; an empty set retains the already configured target.
pub fn factor_marginalize(
    target: &mut SvlFactor,
    source: &SvlFactor,
    eliminated: &BTreeSet<i32>,
) -> Result<(), String> {
    if target.empty() {
        for (position, &variable) in source.variables.iter().enumerate() {
            if !eliminated.contains(&variable) {
                target.add_variable(variable, source.cards[position])?;
            }
        }
    }
    target.fill(0.0);
    for (source_index, &value) in source.data.iter().enumerate() {
        let assignment = source
            .assignment_of(source_index)
            .ok_or("invalid source factor index")?;
        let target_assignment: Vec<_> = target
            .variables
            .iter()
            .map(|variable| {
                assignment[source
                    .variables
                    .iter()
                    .position(|candidate| candidate == variable)
                    .expect("target variable comes from source")]
            })
            .collect();
        let target_index = target
            .index_of(&target_assignment)
            .ok_or("invalid target factor index")?;
        target.data[target_index] += value;
    }
    Ok(())
}

/// `svlFactorMaximizeOp::execute`.
pub fn factor_maximize(
    target: &mut SvlFactor,
    source: &SvlFactor,
    eliminated: &BTreeSet<i32>,
) -> Result<(), String> {
    if target.empty() {
        for (position, &variable) in source.variables.iter().enumerate() {
            if !eliminated.contains(&variable) {
                target.add_variable(variable, source.cards[position])?;
            }
        }
    }
    if target.empty() {
        return Ok(());
    }
    target.fill(-f64::MAX);
    for (source_index, &value) in source.data.iter().enumerate() {
        let assignment = source
            .assignment_of(source_index)
            .ok_or("invalid source factor index")?;
        let target_assignment: Vec<_> = target
            .variables
            .iter()
            .map(|variable| {
                assignment[source
                    .variables
                    .iter()
                    .position(|candidate| candidate == variable)
                    .expect("target variable comes from source")]
            })
            .collect();
        let target_index = target
            .index_of(&target_assignment)
            .ok_or("invalid target factor index")?;
        if target.data[target_index] < value {
            target.data[target_index] = value;
        }
    }
    Ok(())
}

/// `svlFactorNormalizeOp::execute`.
pub fn factor_normalize(target: &mut SvlFactor) {
    if target.empty() {
        return;
    }
    let total: f64 = target.data.iter().sum();
    if total > 0.0 {
        if total != 1.0 {
            target.scale(1.0 / total);
        }
    } else {
        target.fill(1.0 / target.size() as f64);
    }
}

/// `svlFactorLogNormalizeOp::execute`.
pub fn factor_log_normalize(target: &mut SvlFactor) {
    if target.empty() {
        return;
    }
    let maximum = target.data.iter().copied().fold(-f64::MAX, f64::max);
    for value in &mut target.data {
        *value -= maximum;
    }
}

/// Rust's replacement for the C++ virtual `svlFactorOperation` hierarchy.
/// Operations own snapshots of their source factors, so they can safely be
/// scheduled and reused without the pointer lifetime contract of the C++ API.
pub trait FactorOperation {
    fn execute(&self, target: &mut SvlFactor) -> Result<(), String>;
}

#[derive(Clone, Debug)]
pub struct SvlFactorCopyOp {
    source: SvlFactor,
}
impl SvlFactorCopyOp {
    pub fn new(source: SvlFactor) -> Self {
        Self { source }
    }
}
impl FactorOperation for SvlFactorCopyOp {
    fn execute(&self, target: &mut SvlFactor) -> Result<(), String> {
        factor_copy(target, &self.source)
    }
}

#[derive(Clone, Debug)]
pub struct SvlFactorProductOp {
    sources: Vec<SvlFactor>,
}
impl SvlFactorProductOp {
    pub fn new(sources: Vec<SvlFactor>) -> Self {
        Self { sources }
    }
}
impl FactorOperation for SvlFactorProductOp {
    fn execute(&self, target: &mut SvlFactor) -> Result<(), String> {
        factor_product(target, &self.sources.iter().collect::<Vec<_>>())
    }
}

#[derive(Clone, Debug)]
pub struct SvlFactorDivideOp {
    numerator: SvlFactor,
    denominator: SvlFactor,
}
impl SvlFactorDivideOp {
    pub fn new(numerator: SvlFactor, denominator: SvlFactor) -> Self {
        Self {
            numerator,
            denominator,
        }
    }
}
impl FactorOperation for SvlFactorDivideOp {
    fn execute(&self, target: &mut SvlFactor) -> Result<(), String> {
        factor_divide(target, &self.numerator, &self.denominator)
    }
}

#[derive(Clone, Debug)]
pub struct SvlFactorAdditionOp {
    sources: Vec<SvlFactor>,
}
impl SvlFactorAdditionOp {
    pub fn new(sources: Vec<SvlFactor>) -> Self {
        Self { sources }
    }
}
impl FactorOperation for SvlFactorAdditionOp {
    fn execute(&self, target: &mut SvlFactor) -> Result<(), String> {
        factor_addition(target, &self.sources.iter().collect::<Vec<_>>())
    }
}

#[derive(Clone, Debug)]
pub struct SvlFactorSubtractOp {
    first: SvlFactor,
    second: SvlFactor,
}
impl SvlFactorSubtractOp {
    pub fn new(first: SvlFactor, second: SvlFactor) -> Self {
        Self { first, second }
    }
}
impl FactorOperation for SvlFactorSubtractOp {
    fn execute(&self, target: &mut SvlFactor) -> Result<(), String> {
        factor_subtract(target, &self.first, &self.second)
    }
}

#[derive(Clone, Debug)]
pub struct SvlFactorWeightedSumOp {
    first: SvlFactor,
    second: SvlFactor,
    first_weight: f64,
    second_weight: f64,
}
impl SvlFactorWeightedSumOp {
    pub fn new(first: SvlFactor, second: SvlFactor, first_weight: f64, second_weight: f64) -> Self {
        Self {
            first,
            second,
            first_weight,
            second_weight,
        }
    }
}
impl FactorOperation for SvlFactorWeightedSumOp {
    fn execute(&self, target: &mut SvlFactor) -> Result<(), String> {
        factor_weighted_sum(
            target,
            &self.first,
            &self.second,
            self.first_weight,
            self.second_weight,
        )
    }
}

#[derive(Clone, Debug)]
pub struct SvlFactorMarginalizeOp {
    source: SvlFactor,
    eliminated: BTreeSet<i32>,
}
impl SvlFactorMarginalizeOp {
    pub fn new(source: SvlFactor, eliminated: BTreeSet<i32>) -> Self {
        Self { source, eliminated }
    }
    pub fn one(source: SvlFactor, variable: i32) -> Self {
        Self::new(source, BTreeSet::from([variable]))
    }
}
impl FactorOperation for SvlFactorMarginalizeOp {
    fn execute(&self, target: &mut SvlFactor) -> Result<(), String> {
        factor_marginalize(target, &self.source, &self.eliminated)
    }
}

#[derive(Clone, Debug)]
pub struct SvlFactorMaximizeOp {
    source: SvlFactor,
    eliminated: BTreeSet<i32>,
}
impl SvlFactorMaximizeOp {
    pub fn new(source: SvlFactor, eliminated: BTreeSet<i32>) -> Self {
        Self { source, eliminated }
    }
    pub fn one(source: SvlFactor, variable: i32) -> Self {
        Self::new(source, BTreeSet::from([variable]))
    }
}
impl FactorOperation for SvlFactorMaximizeOp {
    fn execute(&self, target: &mut SvlFactor) -> Result<(), String> {
        factor_maximize(target, &self.source, &self.eliminated)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SvlFactorNormalizeOp;
impl FactorOperation for SvlFactorNormalizeOp {
    fn execute(&self, target: &mut SvlFactor) -> Result<(), String> {
        factor_normalize(target);
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SvlFactorLogNormalizeOp;
impl FactorOperation for SvlFactorLogNormalizeOp {
    fn execute(&self, target: &mut SvlFactor) -> Result<(), String> {
        factor_log_normalize(target);
        Ok(())
    }
}

/// Equivalent of C++ `svlFactorAtomicOp`: runs queued owned computations.
#[derive(Default)]
pub struct SvlFactorAtomicOp {
    operations: Vec<Box<dyn FactorOperation>>,
}
impl SvlFactorAtomicOp {
    pub fn new(operation: impl FactorOperation + 'static) -> Self {
        Self {
            operations: vec![Box::new(operation)],
        }
    }
    pub fn from_operations(operations: Vec<Box<dyn FactorOperation>>) -> Self {
        Self { operations }
    }
    pub fn add_operation(&mut self, operation: impl FactorOperation + 'static) {
        self.operations.push(Box::new(operation));
    }
}
impl FactorOperation for SvlFactorAtomicOp {
    fn execute(&self, target: &mut SvlFactor) -> Result<(), String> {
        for operation in &self.operations {
            operation.execute(target)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn product_divide_and_weighted_operations_match_source_rules() {
        let first = SvlFactor::from_parts(vec![1], vec![2], Some(vec![2., 0.])).unwrap();
        let second = SvlFactor::from_parts(vec![2], vec![2], Some(vec![3., 5.])).unwrap();
        let mut target = SvlFactor::new();
        factor_product(&mut target, &[&first, &second]).unwrap();
        assert_eq!(target.data, vec![6., 0., 10., 0.]);
        let mut quotient = SvlFactor::new();
        factor_divide(&mut quotient, &first, &second).unwrap();
        assert_eq!(quotient.data, vec![2. / 3., 0., 2. / 5., 0.]);
        factor_weighted_sum(&mut target, &first, &second, 2., 3.).unwrap();
        assert_eq!(target.data, vec![13., 9., 19., 15.]);
    }
    #[test]
    fn reductions_and_normalizers_cover_source_operations() {
        let source =
            SvlFactor::from_parts(vec![1, 2], vec![2, 2], Some(vec![1., 4., 3., 2.])).unwrap();
        let mut sum = SvlFactor::new();
        factor_marginalize(&mut sum, &source, &BTreeSet::from([2])).unwrap();
        assert_eq!(sum.data, vec![4., 6.]);
        let mut maximum = SvlFactor::new();
        factor_maximize(&mut maximum, &source, &BTreeSet::from([2])).unwrap();
        assert_eq!(maximum.data, vec![3., 4.]);
        factor_normalize(&mut sum);
        assert!((sum.data[0] - 0.4).abs() < f64::EPSILON);
        assert!((sum.data[1] - 0.6).abs() < f64::EPSILON);
        factor_log_normalize(&mut maximum);
        assert_eq!(maximum.data, vec![-1., 0.]);
    }
    #[test]
    fn owned_operations_and_atomic_schedule_have_no_pointer_lifetime_contract() {
        let source = SvlFactor::from_parts(vec![7], vec![2], Some(vec![2., 4.])).unwrap();
        let mut target = SvlFactor::new();
        let mut operations = SvlFactorAtomicOp::new(SvlFactorCopyOp::new(source));
        operations.add_operation(SvlFactorNormalizeOp);
        operations.execute(&mut target).unwrap();
        assert_eq!(target.data, vec![1. / 3., 2. / 3.]);
    }
}
