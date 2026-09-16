//! Owned translation of `svlFactor.{h,cpp}`.

pub const FACTOR_TOLERANCE: f64 = 1.0e-9;
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SvlFactorStorage {
    pub shared: bool,
    pub data: Vec<f64>,
}
impl SvlFactorStorage {
    pub fn new(size: usize, shared: bool) -> Self {
        Self {
            shared,
            data: vec![0.; size],
        }
    }
    pub fn reserve(&mut self, size: usize) {
        self.data.resize(size, 0.)
    }
    pub fn zero(&mut self, size: Option<usize>) {
        let size = size.unwrap_or(self.data.len());
        self.data[..size].fill(0.)
    }
    pub fn fill(&mut self, value: f64, size: Option<usize>) {
        let size = size.unwrap_or(self.data.len());
        self.data[..size].fill(value)
    }
    pub fn copy(&mut self, values: &[f64], size: Option<usize>) {
        let n = size.unwrap_or(values.len());
        self.reserve(n);
        self.data[..n].copy_from_slice(&values[..n])
    }
}
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SvlFactor {
    pub variables: Vec<i32>,
    pub cards: Vec<usize>,
    pub stride: Vec<usize>,
    pub data: Vec<f64>,
}
impl SvlFactor {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_variable(variable: i32, card: usize) -> Result<Self, String> {
        Self::with_variables(vec![variable], vec![card])
    }
    pub fn with_variables(variables: Vec<i32>, cards: Vec<usize>) -> Result<Self, String> {
        let mut result = Self::new();
        result.add_variables(&variables, &cards)?;
        Ok(result)
    }
    pub fn from_parts(
        variables: Vec<i32>,
        cards: Vec<usize>,
        data: Option<Vec<f64>>,
    ) -> Result<Self, String> {
        let mut result = Self::with_variables(variables, cards)?;
        if let Some(data) = data {
            if data.len() != result.data.len() {
                return Err("factor data size mismatch".into());
            }
            result.data = data;
        }
        Ok(result)
    }
    pub fn empty(&self) -> bool {
        self.variables.is_empty()
    }
    pub fn size(&self) -> usize {
        self.data.len()
    }
    pub fn num_vars(&self) -> usize {
        self.variables.len()
    }
    pub fn has_variable(&self, v: i32) -> bool {
        self.variables.contains(&v)
    }
    pub fn variable_id(&self, index: usize) -> Option<i32> {
        self.variables.get(index).copied()
    }
    pub fn var_cardinality(&self, v: i32) -> Option<usize> {
        self.variables
            .iter()
            .position(|&x| x == v)
            .map(|i| self.cards[i])
    }
    pub fn add_variable(&mut self, v: i32, d: usize) -> Result<usize, String> {
        self.add_variables(&[v], &[d])
    }
    pub fn add_variables(&mut self, vars: &[i32], cards: &[usize]) -> Result<usize, String> {
        if vars.len() != cards.len()
            || cards.iter().any(|&d| d == 0)
            || vars.iter().any(|v| self.has_variable(*v))
        {
            return Err("invalid factor variable".into());
        }
        for (&v, &d) in vars.iter().zip(cards) {
            let old_size = self.data.len();
            self.variables.push(v);
            self.cards.push(d);
            self.stride.push(if self.stride.is_empty() {
                1
            } else {
                self.data.len()
            });
            let next = if self.data.is_empty() {
                d
            } else {
                self.data.len() * d
            };
            self.data = if old_size == 0 {
                vec![1.; next]
            } else {
                (0..next).map(|i| self.data[i % old_size]).collect()
            };
        }
        Ok(self.data.len())
    }
    pub fn add_factor_variables(&mut self, other: &Self) -> Result<usize, String> {
        let vars: Vec<_> = other
            .variables
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| (!self.has_variable(v)).then_some((v, other.cards[i])))
            .collect();
        self.add_variables(
            &vars.iter().map(|x| x.0).collect::<Vec<_>>(),
            &vars.iter().map(|x| x.1).collect::<Vec<_>>(),
        )
    }
    pub fn index_of(&self, assignment: &[usize]) -> Option<usize> {
        if assignment.len() != self.cards.len()
            || assignment.iter().zip(&self.cards).any(|(&v, &c)| v >= c)
        {
            None
        } else {
            Some(
                assignment
                    .iter()
                    .zip(&self.stride)
                    .map(|(&v, &s)| v * s)
                    .sum(),
            )
        }
    }
    pub fn index_of_variable(&self, var: i32, val: usize, index: usize) -> Option<usize> {
        let i = self.variables.iter().position(|&v| v == var)?;
        if val >= self.cards[i] || index >= self.size() {
            return None;
        }
        let old = index / self.stride[i] % self.cards[i];
        usize::try_from(index as isize + (val as isize - old as isize) * self.stride[i] as isize)
            .ok()
    }
    pub fn value_of(&self, var: i32, index: usize) -> Option<usize> {
        let i = self.variables.iter().position(|&v| v == var)?;
        Some(index / self.stride[i] % self.cards[i])
    }
    pub fn assignment_of(&self, index: usize) -> Option<Vec<usize>> {
        (index < self.size()).then(|| {
            self.cards
                .iter()
                .enumerate()
                .map(|(i, &c)| index / self.stride[i] % c)
                .collect()
        })
    }
    pub fn index_of_max(&self) -> Option<usize> {
        self.data
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|x| x.0)
    }
    pub fn index_of_min(&self) -> Option<usize> {
        self.data
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .map(|x| x.0)
    }
    pub fn initialize(&mut self) -> &mut Self {
        self.fill(1.)
    }
    pub fn fill(&mut self, value: f64) -> &mut Self {
        self.data.fill(value);
        self
    }
    pub fn scale(&mut self, value: f64) -> &mut Self {
        for x in &mut self.data {
            *x *= value;
        }
        self
    }
    pub fn offset(&mut self, value: f64) -> &mut Self {
        for x in &mut self.data {
            *x += value;
        }
        self
    }
    pub fn normalize(&mut self) -> &mut Self {
        let total: f64 = self.data.iter().sum();
        if total > 0. {
            self.scale(1. / total);
        } else if !self.data.is_empty() {
            self.fill(1. / self.size() as f64);
        }
        self
    }
    fn remove_variable(&mut self, index: usize, mode: impl Fn(f64, f64) -> f64, initial: f64) {
        let stride = self.stride[index];
        let card = self.cards[index];
        let new_size = self.size() / card;
        let mut result = vec![initial; new_size];
        for out in 0..new_size {
            let base = (out / stride) * stride * card + out % stride;
            for value in 0..card {
                result[out] = mode(result[out], self.data[base + value * stride]);
            }
        }
        self.variables.remove(index);
        self.cards.remove(index);
        self.data = result;
        self.stride = (0..self.cards.len())
            .map(|i| self.cards[..i].iter().product::<usize>().max(1))
            .collect();
    }
    pub fn marginalize(&mut self, var: i32) -> Result<&mut Self, String> {
        let index = self
            .variables
            .iter()
            .position(|&v| v == var)
            .ok_or("unknown factor variable")?;
        if self.num_vars() == 1 {
            *self = Self::new()
        } else {
            self.remove_variable(index, |a, b| a + b, 0.)
        }
        Ok(self)
    }
    pub fn maximize(&mut self, var: i32) -> Result<&mut Self, String> {
        let index = self
            .variables
            .iter()
            .position(|&v| v == var)
            .ok_or("unknown factor variable")?;
        if self.num_vars() == 1 {
            *self = Self::new()
        } else {
            self.remove_variable(index, f64::max, -f64::MAX)
        }
        Ok(self)
    }
    pub fn reduce(&mut self, var: i32, val: usize) -> Result<&mut Self, String> {
        let index = self
            .variables
            .iter()
            .position(|&v| v == var)
            .ok_or("unknown factor variable")?;
        if val >= self.cards[index] {
            return Err("invalid factor value".into());
        }
        if self.num_vars() == 1 {
            *self = Self::new()
        } else {
            let stride = self.stride[index];
            let card = self.cards[index];
            let mut result = Vec::with_capacity(self.size() / card);
            for out in 0..self.size() / card {
                let base = (out / stride) * stride * card + out % stride;
                result.push(self.data[base + val * stride]);
            }
            self.variables.remove(index);
            self.cards.remove(index);
            self.data = result;
            self.stride = (0..self.cards.len())
                .map(|i| self.cards[..i].iter().product::<usize>().max(1))
                .collect();
        }
        Ok(self)
    }
    fn combine(&mut self, other: &Self, op: impl Fn(f64, f64) -> f64) -> Result<(), String> {
        if other.empty() {
            return Ok(());
        }
        if self.empty() {
            *self = other.clone();
            if op(2., 3.) != 6. {
                for x in &mut self.data {
                    *x = op(0., *x)
                }
            }
            return Ok(());
        }
        for (i, &v) in other.variables.iter().enumerate() {
            if let Some(c) = self.var_cardinality(v) {
                if c != other.cards[i] {
                    return Err("factor cardinality mismatch".into());
                }
            } else {
                self.add_variable(v, other.cards[i])?;
            }
        }
        for index in 0..self.size() {
            let mut other_index = 0;
            for (i, &v) in other.variables.iter().enumerate() {
                other_index += self.value_of(v, index).unwrap() * other.stride[i];
            }
            self.data[index] = op(self.data[index], other.data[other_index]);
        }
        Ok(())
    }
    pub fn product(&mut self, other: &Self) -> Result<&mut Self, String> {
        self.combine(other, |a, b| a * b)?;
        Ok(self)
    }
    pub fn divide(&mut self, other: &Self) -> Result<&mut Self, String> {
        self.combine(other, |a, b| a / b)?;
        Ok(self)
    }
    pub fn add(&mut self, other: &Self) -> Result<&mut Self, String> {
        self.combine(other, |a, b| a + b)?;
        Ok(self)
    }
    pub fn subtract(&mut self, other: &Self) -> Result<&mut Self, String> {
        self.combine(other, |a, b| a - b)?;
        Ok(self)
    }
    pub fn data_compare(&self, other: &Self) -> bool {
        self.data.len() == other.data.len()
            && self
                .data
                .iter()
                .zip(&other.data)
                .all(|(a, b)| (a - b).abs() <= FACTOR_TOLERANCE)
    }
    pub fn data_compare_and_copy(&mut self, other: &Self) -> bool {
        let same = self.data_compare(other);
        *self = other.clone();
        same
    }
    pub fn equivalent(&self, other: &Self) -> bool {
        if self.size() != other.size() || self.variables.len() != other.variables.len() {
            return false;
        }
        for (i, &v) in self.variables.iter().enumerate() {
            if other.var_cardinality(v) != Some(self.cards[i]) {
                return false;
            }
        }
        (0..self.size()).all(|i| {
            let a = self.assignment_of(i).unwrap();
            let b: Vec<_> = other
                .variables
                .iter()
                .map(|v| a[self.variables.iter().position(|x| x == v).unwrap()])
                .collect();
            (self.data[i] - other.data[other.index_of(&b).unwrap()]).abs() <= FACTOR_TOLERANCE
        })
    }
    pub fn map_from(&self, other: &Self) -> Option<Vec<usize>> {
        if self.empty() || other.empty() {
            return Some(vec![0; self.size()]);
        }
        (0..self.size())
            .map(|i| {
                let assignment = self.assignment_of(i)?;
                let mapped: Vec<_> = other
                    .variables
                    .iter()
                    .map(|v| assignment[self.variables.iter().position(|x| x == v).unwrap()])
                    .collect();
                other.index_of(&mapped)
            })
            .collect()
    }
    pub fn stride_mapping(&self, vars: &[i32]) -> Vec<isize> {
        vars.iter()
            .map(|v| {
                self.variables
                    .iter()
                    .position(|x| x == v)
                    .map(|i| self.stride[i] as isize)
                    .unwrap_or(0)
            })
            .collect()
    }
    pub fn write(&self, indent: usize) -> String {
        let p = " ".repeat(indent);
        if self.empty() {
            return format!("{p}<Factor>\n{p}</Factor>\n");
        }
        format!(
            "{p}<Factor>\n{p}  <Vars>\n   {p} {}\n{p}  </Vars>\n{p}  <Cards>\n   {p} {}\n{p}  </Cards>\n{p}  <Data>\n{}{}{p}  </Data>\n{p}</Factor>\n",
            self.variables
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(" "),
            self.cards
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(" "),
            self.data
                .iter()
                .map(|x| format!("{p}    {x}\n"))
                .collect::<String>(),
            ""
        )
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn factor_product_and_marginalize() {
        let mut a = SvlFactor::from_parts(vec![1], vec![2], Some(vec![2., 3.])).unwrap();
        let b = SvlFactor::from_parts(vec![2], vec![2], Some(vec![5., 7.])).unwrap();
        a.product(&b).unwrap();
        assert_eq!(a.data, vec![10., 15., 14., 21.]);
        a.marginalize(2).unwrap();
        assert_eq!(a.data, vec![24., 36.]);
    }
    #[test]
    fn factor_index_and_reduction() {
        let mut f = SvlFactor::from_parts(
            vec![1, 2],
            vec![2, 3],
            Some((0..6).map(|x| x as f64).collect()),
        )
        .unwrap();
        assert_eq!(f.index_of(&[1, 2]), Some(5));
        f.reduce(2, 1).unwrap();
        assert_eq!(f.data, vec![2., 3.]);
    }
}
