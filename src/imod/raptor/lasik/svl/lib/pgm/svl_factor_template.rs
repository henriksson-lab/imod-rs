//! Owned translation of `svlFactorTemplate.{h,cpp}`.

use std::collections::BTreeSet;

/// Dense factor table used by `SvlFactorTemplate`.
#[derive(Clone, Debug, PartialEq)]
pub struct SvlFactor {
    pub variables: Vec<i32>,
    pub cards: Vec<usize>,
    pub values: Vec<f64>,
}

impl SvlFactor {
    pub fn new(variables: Vec<i32>, cards: Vec<usize>) -> Result<Self, String> {
        if variables.len() != cards.len() || cards.contains(&0) {
            return Err("invalid factor dimensions".into());
        }
        let size = cards
            .iter()
            .try_fold(1usize, |n, &card| n.checked_mul(card))
            .ok_or("factor size overflow")?;
        Ok(Self {
            variables,
            cards,
            values: vec![0.0; size],
        })
    }
    pub fn size(&self) -> usize {
        self.values.len()
    }
    pub fn num_vars(&self) -> usize {
        self.variables.len()
    }
}

/// Template entries are `(weight_index, feature_index)` pairs. Negative indices
/// retain SVL's constant-one sentinel convention.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SvlFactorTemplate {
    cards: Vec<usize>,
    stride: Vec<usize>,
    n_size: usize,
    entry_mapping: Vec<Vec<(isize, isize)>>,
}

impl SvlFactorTemplate {
    pub fn new() -> Self {
        Self {
            n_size: 1,
            ..Self::default()
        }
    }

    pub fn with_repeated_dimension(dim: usize, repeats: usize) -> Result<Self, String> {
        let mut template = Self::new();
        for _ in 0..repeats {
            template.add_variable(dim)?;
        }
        Ok(template)
    }

    pub fn with_dimensions(dimensions: &[usize]) -> Result<Self, String> {
        let mut template = Self::new();
        template.add_variables(dimensions)?;
        Ok(template)
    }

    /// `svlFactorTemplate(XMLNode&)`, using its `<Cards>` and ordered `<Entry>` payloads.
    pub fn from_xml(xml: &str) -> Result<Self, String> {
        let cards_open =
            xml.find("<Cards>").ok_or("FactorTemplate has no Cards")? + "<Cards>".len();
        let cards_close = xml[cards_open..]
            .find("</Cards>")
            .map(|at| cards_open + at)
            .ok_or("unterminated Cards")?;
        let cards = xml[cards_open..cards_close]
            .split_whitespace()
            .map(|word| {
                word.parse::<usize>()
                    .map_err(|_| format!("invalid card {word}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut template = Self::with_dimensions(&cards)?;
        let mut cursor = cards_close + "</Cards>".len();
        let mut entries = Vec::new();
        while let Some(relative_open) = xml[cursor..].find("<Entry>") {
            let open = cursor + relative_open + "<Entry>".len();
            let close = xml[open..]
                .find("</Entry>")
                .map(|at| open + at)
                .ok_or("unterminated Entry")?;
            let items = xml[open..close]
                .split_whitespace()
                .map(|word| {
                    word.parse::<isize>()
                        .map_err(|_| format!("invalid Entry value {word}"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            if items.len() % 2 != 0 {
                return Err("Entry needs weight/feature pairs".into());
            }
            entries.push(
                items
                    .chunks_exact(2)
                    .map(|pair| (pair[0], pair[1]))
                    .collect(),
            );
            cursor = close + "</Entry>".len();
        }
        if entries.len() != template.n_size {
            return Err(format!(
                "expected {} Entry nodes, got {}",
                template.n_size,
                entries.len()
            ));
        }
        template.entry_mapping = entries;
        Ok(template)
    }

    pub fn empty(&self) -> bool {
        self.cards.is_empty()
    }
    pub fn size(&self) -> usize {
        self.n_size
    }
    pub fn num_vars(&self) -> usize {
        self.cards.len()
    }
    pub fn var_cardinality(&self, variable: usize) -> Result<usize, String> {
        self.cards
            .get(variable)
            .copied()
            .ok_or("variable out of range".into())
    }
    pub fn entries(&self, index: usize) -> Result<&[(isize, isize)], String> {
        self.entry_mapping
            .get(index)
            .map(Vec::as_slice)
            .ok_or("entry out of range".into())
    }
    pub fn entries_mut(&mut self, index: usize) -> Result<&mut Vec<(isize, isize)>, String> {
        self.entry_mapping
            .get_mut(index)
            .ok_or("entry out of range".into())
    }

    pub fn add_variable(&mut self, dimension: usize) -> Result<usize, String> {
        if dimension <= 1 {
            return Err("variable cardinality must exceed one".into());
        }
        self.cards.push(dimension);
        self.stride.push(self.n_size);
        self.n_size = self
            .n_size
            .checked_mul(dimension)
            .ok_or("factor size overflow")?;
        self.entry_mapping.resize(self.n_size, Vec::new());
        Ok(self.n_size)
    }

    pub fn add_variables(&mut self, dimensions: &[usize]) -> Result<usize, String> {
        for &dimension in dimensions {
            self.add_variable(dimension)?;
        }
        Ok(self.n_size)
    }

    pub fn max_weight_index(&self) -> isize {
        self.entry_mapping
            .iter()
            .flatten()
            .map(|pair| pair.0)
            .fold(0, isize::max)
    }
    pub fn min_weight_index(&self) -> isize {
        self.entry_mapping
            .iter()
            .flatten()
            .map(|pair| pair.0)
            .min()
            .unwrap_or(isize::MAX)
    }
    pub fn offset_weights(&mut self, offset: isize) -> Result<(), String> {
        for pair in self.entry_mapping.iter_mut().flatten() {
            pair.0 += offset;
            if pair.0 < 0 {
                return Err("weight index became negative".into());
            }
        }
        Ok(())
    }
    pub fn weight_indices(&self) -> BTreeSet<isize> {
        self.entry_mapping
            .iter()
            .flatten()
            .map(|pair| pair.0)
            .collect()
    }
    pub fn max_feature_index(&self) -> isize {
        self.entry_mapping
            .iter()
            .flatten()
            .map(|pair| pair.1)
            .fold(-1, isize::max)
    }
    pub fn min_feature_index(&self) -> isize {
        self.entry_mapping
            .iter()
            .flatten()
            .map(|pair| pair.1)
            .min()
            .unwrap_or(isize::MAX)
    }

    pub fn index_of_value(
        &self,
        variable: usize,
        value: usize,
        index: usize,
    ) -> Result<usize, String> {
        let card = self.var_cardinality(variable)?;
        if value >= card {
            return Err("value out of range".into());
        }
        Ok(index - (index / self.stride[variable]) % card + value * self.stride[variable])
    }
    pub fn index_of(&self, assignment: &[usize]) -> Result<usize, String> {
        if assignment.len() != self.cards.len() {
            return Err("wrong assignment arity".into());
        }
        assignment
            .iter()
            .enumerate()
            .try_fold(0, |index, (variable, &value)| {
                self.index_of_value(variable, value, index)
            })
    }
    pub fn value_of(&self, variable: usize, index: usize) -> Result<usize, String> {
        Ok(
            (index / self.stride.get(variable).ok_or("variable out of range")?)
                % self.cards[variable],
        )
    }
    pub fn assignment_of(&self, index: usize) -> Result<Vec<usize>, String> {
        if index >= self.n_size {
            return Err("index out of range".into());
        }
        (0..self.cards.len())
            .map(|variable| self.value_of(variable, index))
            .collect()
    }
    pub fn entry_log_value(
        &self,
        index: usize,
        weights: &[f64],
        features: &[f64],
    ) -> Result<f64, String> {
        self.entries(index)?
            .iter()
            .try_fold(0.0, |sum, &(weight, feature)| {
                let w = if weight < 0 {
                    1.0
                } else {
                    *weights
                        .get(weight as usize)
                        .ok_or("weight index out of range")?
                };
                let x = if feature < 0 {
                    1.0
                } else {
                    *features
                        .get(feature as usize)
                        .ok_or("feature index out of range")?
                };
                Ok(sum + w * x)
            })
    }

    pub fn create_factor(
        &self,
        variables: &[i32],
        weights: &[f64],
        features: &[f64],
    ) -> Result<SvlFactor, String> {
        let mut factor = SvlFactor::new(variables.to_vec(), self.cards.clone())?;
        self.update_factor(&mut factor, weights, features)?;
        Ok(factor)
    }
    pub fn update_factor(
        &self,
        factor: &mut SvlFactor,
        weights: &[f64],
        features: &[f64],
    ) -> Result<(), String> {
        self.update_log_factor(factor, weights, features)?;
        factor
            .values
            .iter_mut()
            .for_each(|value| *value = value.exp());
        Ok(())
    }
    pub fn create_log_factor(
        &self,
        variables: &[i32],
        weights: &[f64],
        features: &[f64],
    ) -> Result<SvlFactor, String> {
        let mut factor = SvlFactor::new(variables.to_vec(), self.cards.clone())?;
        self.update_log_factor(&mut factor, weights, features)?;
        Ok(factor)
    }
    pub fn update_log_factor(
        &self,
        factor: &mut SvlFactor,
        weights: &[f64],
        features: &[f64],
    ) -> Result<(), String> {
        if factor.size() != self.n_size || factor.num_vars() != self.cards.len() {
            return Err("factor dimensions differ from template".into());
        }
        if weights.len() <= self.max_weight_index() as usize
            || (self.max_feature_index() >= 0
                && features.len() <= self.max_feature_index() as usize)
        {
            return Err("weights or features too short".into());
        }
        let mut maximum = -f64::MAX;
        for index in 0..self.n_size {
            factor.values[index] = self.entry_log_value(index, weights, features)?;
            maximum = maximum.max(factor.values[index]);
        }
        factor.values.iter_mut().for_each(|value| *value -= maximum);
        Ok(())
    }

    pub fn create_reduced_factor(
        &self,
        variables: &[i32],
        weights: &[f64],
        features: &[f64],
        assignments: &[isize],
    ) -> Result<SvlFactor, String> {
        let mut factor =
            self.create_reduced_log_factor(variables, weights, features, assignments)?;
        factor
            .values
            .iter_mut()
            .for_each(|value| *value = value.exp());
        Ok(factor)
    }
    pub fn create_reduced_log_factor(
        &self,
        variables: &[i32],
        weights: &[f64],
        features: &[f64],
        assignments: &[isize],
    ) -> Result<SvlFactor, String> {
        if variables.len() != self.cards.len() || assignments.len() != self.cards.len() {
            return Err("wrong variable or assignment arity".into());
        }
        let kept = assignments
            .iter()
            .enumerate()
            .filter(|(_, value)| **value < 0)
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        let mut factor = SvlFactor::new(
            kept.iter().map(|&i| variables[i]).collect(),
            kept.iter().map(|&i| self.cards[i]).collect(),
        )?;
        self.update_reduced_log_factor(&mut factor, weights, features, assignments)?;
        Ok(factor)
    }
    pub fn update_reduced_factor(
        &self,
        factor: &mut SvlFactor,
        weights: &[f64],
        features: &[f64],
        assignments: &[isize],
    ) -> Result<(), String> {
        self.update_reduced_log_factor(factor, weights, features, assignments)?;
        factor
            .values
            .iter_mut()
            .for_each(|value| *value = value.exp());
        Ok(())
    }
    pub fn update_reduced_log_factor(
        &self,
        factor: &mut SvlFactor,
        weights: &[f64],
        features: &[f64],
        assignments: &[isize],
    ) -> Result<(), String> {
        if assignments.len() != self.cards.len() {
            return Err("wrong assignment arity".into());
        }
        let unknown = assignments
            .iter()
            .enumerate()
            .filter(|(_, value)| **value < 0)
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        if factor.num_vars() != unknown.len() {
            return Err("reduced factor arity differs".into());
        }
        let mut maximum = -f64::MAX;
        for reduced_index in 0..factor.size() {
            let mut complete = assignments.to_vec();
            let mut quotient = reduced_index;
            let reduced_assignment = factor
                .cards
                .iter()
                .map(|&card| {
                    let value = quotient % card;
                    quotient /= card;
                    value
                })
                .collect::<Vec<_>>();
            for (&variable, &value) in unknown.iter().zip(&reduced_assignment) {
                complete[variable] = value as isize;
            }
            let complete = complete
                .iter()
                .map(|&value| {
                    usize::try_from(value).map_err(|_| "negative observed assignment".to_string())
                })
                .collect::<Result<Vec<_>, _>>()?;
            factor.values[reduced_index] =
                self.entry_log_value(self.index_of(&complete)?, weights, features)?;
            maximum = maximum.max(factor.values[reduced_index]);
        }
        factor.values.iter_mut().for_each(|value| *value -= maximum);
        Ok(())
    }

    pub fn accumulate_statistics(
        &self,
        statistics: &mut [f64],
        assignment: &[usize],
        features: &[f64],
        weight: f64,
    ) -> Result<(), String> {
        for &(weight_index, feature_index) in self.entries(self.index_of(assignment)?)? {
            if weight_index >= 0 {
                let value = if feature_index < 0 {
                    1.0
                } else {
                    *features
                        .get(feature_index as usize)
                        .ok_or("feature index out of range")?
                };
                *statistics
                    .get_mut(weight_index as usize)
                    .ok_or("statistic index out of range")? += weight * value;
            }
        }
        Ok(())
    }
    /// The source overload immediately asserts false; retain that unsupported status.
    pub fn accumulate_statistics_with_factor(
        &self,
        _statistics: &mut [f64],
        _assignment: &[usize],
        _features: &[f64],
        _weights: &SvlFactor,
    ) -> Result<(), String> {
        Err("svlFactorTemplate weighted statistic accumulation is unimplemented in source".into())
    }
    pub fn write(&self, indent: usize) -> String {
        let pad = " ".repeat(indent);
        let mut output = format!("{pad}<FactorTemplate>\n");
        if self.empty() {
            return format!("{pad}</FactorTemplate>\n");
        }
        output.push_str(&format!(
            "{pad}  <Cards>\n   {pad}{}\n{pad}  </Cards>\n",
            self.cards
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(" ")
        ));
        for index in 0..self.n_size {
            output.push_str(&format!(
                "{pad}  <!-- variable assignment: ({:?} ) -->\n{pad}  <Entry>\n",
                self.assignment_of(index).expect("table index")
            ));
            for &(weight, feature) in &self.entry_mapping[index] {
                output.push_str(&format!("{pad}    {weight} {feature}\n"));
            }
            output.push_str(&format!("{pad}  </Entry>\n"));
        }
        output.push_str(&format!("{pad}</FactorTemplate>\n"));
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn factors_normalize_log_scores_and_reduce_assignments() {
        let mut t = SvlFactorTemplate::with_dimensions(&[2, 2]).unwrap();
        t.entries_mut(0).unwrap().push((0, 0));
        t.entries_mut(1).unwrap().push((0, -1));
        t.entries_mut(2).unwrap().push((0, -1));
        t.entries_mut(3).unwrap().push((0, 0));
        let log = t.create_log_factor(&[4, 8], &[2.0], &[3.0]).unwrap();
        assert_eq!(log.values, vec![0.0, -4.0, -4.0, 0.0]);
        let reduced = t
            .create_reduced_factor(&[4, 8], &[2.0], &[3.0], &[-1, 0])
            .unwrap();
        assert_eq!(reduced.values, vec![1.0, (-4.0f64).exp()]);
    }
}
