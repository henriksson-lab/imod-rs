//! Owned Rust translation of `svlOptimizer.{h,cpp}`.

/// C++ `svlLBFGSResult`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SvlLbfgsResult {
    Error,
    MaxIterations,
    ConvergedFunction,
    ConvergedGradient,
    ConvergedPosition,
}

/// Objective/gradient interface replacing C++ virtual methods.
pub trait OptimizerObjective {
    fn objective(&self, x: &[f64]) -> f64;
    fn gradient(&self, x: &[f64], gradient: &mut [f64]);
    fn objective_and_gradient(&self, x: &[f64], gradient: &mut [f64]) -> f64 {
        self.gradient(x, gradient);
        self.objective(x)
    }
    fn monitor(&self, _iteration: usize, _objective: f64) {}
}

/// External-LBFGS `fdf` callback from `svlOptimizer.cpp`.  The C callback
/// copies an external one-based array into optimizer storage, evaluates both
/// objective and gradient, then copies the gradient back; owned slices make
/// each transfer explicit.
pub fn fdf<O: OptimizerObjective>(
    optimizer: &mut SvlOptimizer,
    objective: &O,
    iterate: &[f64],
) -> Result<(f64, Vec<f64>), SvlLbfgsResult> {
    if iterate.len() != optimizer.x.len() {
        return Err(SvlLbfgsResult::Error);
    }
    optimizer.x.copy_from_slice(iterate);
    let value = objective.objective_and_gradient(&optimizer.x, &mut optimizer.df);
    Ok((value, optimizer.df.clone()))
}

/// External-LBFGS `newiter` callback: retain the source's observable state
/// update and monitoring hook without a raw backend parameter pointer.
pub fn newiter<O: OptimizerObjective>(
    optimizer: &mut SvlOptimizer,
    objective: &O,
    iteration: usize,
    iterate: &[f64],
    objective_value: f64,
) -> Result<(), SvlLbfgsResult> {
    if iterate.len() != optimizer.x.len() {
        return Err(SvlLbfgsResult::Error);
    }
    optimizer.x.copy_from_slice(iterate);
    objective.monitor(iteration, objective_value);
    Ok(())
}

/// C++ `svlOptimizer`, with owned vectors replacing `double *` allocation.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SvlOptimizer {
    pub x: Vec<f64>,
    pub df: Vec<f64>,
}

/// `svlOptimizer()`, the native empty optimizer constructor.
pub fn svl_optimizer() -> SvlOptimizer {
    SvlOptimizer::new()
}

/// `svlOptimizer(unsigned n)`, with zeroed parameter values and gradient
/// workspace just as the source constructor allocates.
pub fn svl_optimizer_with_size(size: usize) -> SvlOptimizer {
    SvlOptimizer::with_size(size)
}

/// `svlOptimizer(const svlOptimizer &)`, retaining both parameter and
/// gradient state rather than aliasing the source's heap arrays.
pub fn svl_optimizer_copy(source: &SvlOptimizer) -> SvlOptimizer {
    source.clone()
}

/// `~svlOptimizer()`: owned vectors are released together.
pub fn free_svl_optimizer(optimizer: SvlOptimizer) {
    drop(optimizer);
}

impl SvlOptimizer {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_size(size: usize) -> Self {
        Self {
            x: vec![0.0; size],
            df: vec![0.0; size],
        }
    }
    pub fn size(&self) -> usize {
        self.x.len()
    }
    pub fn initialize(&mut self, size: usize, values: Option<&[f64]>) -> Result<(), &'static str> {
        if size == 0 || values.is_some_and(|values| values.len() != size) {
            return Err("invalid optimizer initialization");
        }
        self.x = values.map_or_else(|| vec![0.0; size], ToOwned::to_owned);
        self.df = vec![0.0; size];
        Ok(())
    }
    pub fn initialize_current(&mut self, values: Option<&[f64]>) -> Result<(), &'static str> {
        if self.x.is_empty() || values.is_some_and(|values| values.len() != self.x.len()) {
            return Err("invalid optimizer initialization");
        }
        if let Some(values) = values {
            self.x.copy_from_slice(values);
        } else {
            self.x.fill(0.0);
        }
        Ok(())
    }
    pub fn parameter(&self, index: usize) -> Option<f64> {
        self.x.get(index).copied()
    }
    pub fn parameter_mut(&mut self, index: usize) -> Option<&mut f64> {
        self.x.get_mut(index)
    }
    /// C++ `solve`, using the source's internal L-BFGS path (the shipped C++
    /// enables an external backend, which is intentionally not retained).
    pub fn solve<O: OptimizerObjective>(
        &mut self,
        objective: &O,
        max_iterations: usize,
        tolerance: f64,
        monitor: bool,
    ) -> Result<f64, SvlLbfgsResult> {
        if self.x.is_empty() {
            return Err(SvlLbfgsResult::Error);
        }
        let memory = self.x.len().min(7);
        let result = self.lbfgs_minimize(
            objective,
            memory,
            max_iterations,
            1.0e-6,
            tolerance,
            1.0e-6,
            monitor,
        );
        match result {
            SvlLbfgsResult::Error => Err(result),
            _ => Ok(objective.objective(&self.x)),
        }
    }
    pub fn lbfgs_minimize<O: OptimizerObjective>(
        &mut self,
        objective: &O,
        memory: usize,
        max_iterations: usize,
        eps_g: f64,
        eps_f: f64,
        eps_x: f64,
        monitor: bool,
    ) -> SvlLbfgsResult {
        if memory == 0 || memory > self.x.len() || eps_g < 0.0 || eps_f < 0.0 || eps_x < 0.0 {
            return SvlLbfgsResult::Error;
        }
        let n = self.x.len();
        let mut value = objective.objective_and_gradient(&self.x, &mut self.df);
        let mut history: Vec<(Vec<f64>, Vec<f64>, f64)> = Vec::new();
        for iteration in 0..max_iterations {
            let previous_x = self.x.clone();
            let previous_g = self.df.clone();
            let previous_value = value;
            let mut direction = self.df.iter().map(|value| -*value).collect::<Vec<_>>();
            if !history.is_empty() {
                let mut alpha = Vec::with_capacity(history.len());
                for (s, y, rho) in history.iter().rev() {
                    let value = rho * dot(s, &direction);
                    alpha.push(value);
                    for index in 0..n {
                        direction[index] -= value * y[index];
                    }
                }
                let (s, y, _) = history.last().unwrap();
                let scale = dot(s, y) / dot(y, y);
                for value in &mut direction {
                    *value *= scale;
                }
                for ((s, y, rho), alpha) in history.iter().zip(alpha.into_iter().rev()) {
                    let beta = rho * dot(y, &direction);
                    for index in 0..n {
                        direction[index] += s[index] * (alpha - beta);
                    }
                }
            }
            let mut step = if iteration == 0 {
                1.0 / norm(&self.df).max(f64::MIN_POSITIVE)
            } else {
                1.0
            };
            if !self.lbfgs_search(
                objective,
                &previous_x,
                &previous_g,
                previous_value,
                &direction,
                &mut step,
                &mut value,
            ) {
                self.x = previous_x;
                self.df = previous_g;
                return SvlLbfgsResult::Error;
            }
            let s: Vec<f64> = self
                .x
                .iter()
                .zip(&previous_x)
                .map(|(new, old)| new - old)
                .collect();
            let y: Vec<f64> = self
                .df
                .iter()
                .zip(&previous_g)
                .map(|(new, old)| new - old)
                .collect();
            let ys = dot(&y, &s);
            if ys > 0.0 {
                if history.len() == memory {
                    history.remove(0);
                }
                history.push((s, y, 1.0 / ys));
            }
            if monitor {
                objective.monitor(iteration + 1, value);
            }
            if norm(&self.df) <= eps_g {
                return SvlLbfgsResult::ConvergedGradient;
            }
            let scale = previous_value.abs().max(value.abs()).max(1.0);
            if previous_value - value <= eps_f * scale {
                return SvlLbfgsResult::ConvergedFunction;
            }
            if norm(
                &self
                    .x
                    .iter()
                    .zip(&previous_x)
                    .map(|(new, old)| new - old)
                    .collect::<Vec<_>>(),
            ) <= eps_x
            {
                return SvlLbfgsResult::ConvergedPosition;
            }
        }
        SvlLbfgsResult::MaxIterations
    }
    /// Source `lbfgsSearch`, expressed as a bounded Armijo/Wolfe search with
    /// all vectors borrowed rather than Eigen map aliases.
    pub fn lbfgs_search<O: OptimizerObjective>(
        &mut self,
        objective: &O,
        origin: &[f64],
        origin_gradient: &[f64],
        origin_value: f64,
        direction: &[f64],
        step: &mut f64,
        value: &mut f64,
    ) -> bool {
        let derivative = dot(origin_gradient, direction);
        if *step <= 0.0 || derivative >= 0.0 {
            return false;
        }
        let mut trial = *step;
        for _ in 0..20 {
            for index in 0..self.x.len() {
                self.x[index] = origin[index] + trial * direction[index];
            }
            *value = objective.objective_and_gradient(&self.x, &mut self.df);
            let trial_derivative = dot(&self.df, direction);
            if *value <= origin_value + 0.0001 * trial * derivative
                && trial_derivative.abs() <= -0.9 * derivative
            {
                *step = trial;
                return true;
            }
            trial *= 0.5;
            if trial < 1.0e-20 {
                break;
            }
        }
        false
    }
}
fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}
fn norm(values: &[f64]) -> f64 {
    dot(values, values).sqrt()
}

/// Scalar state passed by C++ `lbfgsStep` reference arguments.
#[derive(Clone, Copy, Debug)]
pub struct LbfgsStepState {
    pub stx: f64,
    pub fx: f64,
    pub dx: f64,
    pub sty: f64,
    pub fy: f64,
    pub dy: f64,
    pub stp: f64,
    pub brackt: bool,
}
/// C++ `lbfgsStep`; this is retained independently for audit and line-search users.
pub fn lbfgs_step(state: &mut LbfgsStepState, fp: f64, dp: f64, stmin: f64, stmax: f64) -> bool {
    if (state.brackt
        && (state.stp <= state.stx.min(state.sty) || state.stp >= state.stx.max(state.sty)))
        || state.dx * (state.stp - state.stx) >= 0.0
        || stmax < stmin
    {
        return false;
    }
    let sgnd = dp * (state.dx / state.dx.abs());
    let (bound, stpf);
    if fp > state.fx {
        let theta = 3.0 * (state.fx - fp) / (state.stp - state.stx) + state.dx + dp;
        let scale = theta.abs().max(state.dx.abs()).max(dp.abs());
        let mut gamma =
            scale * ((theta / scale).powi(2) - (state.dx / scale) * (dp / scale)).sqrt();
        if state.stp < state.stx {
            gamma = -gamma;
        }
        let r = (gamma - state.dx + theta) / (gamma - state.dx + gamma + dp);
        let cubic = state.stx + r * (state.stp - state.stx);
        let quadratic = state.stx
            + state.dx / ((state.fx - fp) / (state.stp - state.stx) + state.dx) / 2.0
                * (state.stp - state.stx);
        stpf = if (cubic - state.stx).abs() < (quadratic - state.stx).abs() {
            cubic
        } else {
            cubic + (quadratic - cubic) / 2.0
        };
        bound = true;
        state.brackt = true;
    } else if sgnd < 0.0 {
        let theta = 3.0 * (state.fx - fp) / (state.stp - state.stx) + state.dx + dp;
        let scale = theta.abs().max(state.dx.abs()).max(dp.abs());
        let mut gamma =
            scale * ((theta / scale).powi(2) - (state.dx / scale) * (dp / scale)).sqrt();
        if state.stp > state.stx {
            gamma = -gamma;
        }
        let r = (gamma - dp + theta) / (gamma - dp + gamma + state.dx);
        let cubic = state.stp + r * (state.stx - state.stp);
        let quadratic = state.stp + dp / (dp - state.dx) * (state.stx - state.stp);
        stpf = if (cubic - state.stp).abs() > (quadratic - state.stp).abs() {
            cubic
        } else {
            quadratic
        };
        bound = false;
        state.brackt = true;
    } else {
        stpf = if state.brackt {
            0.5 * (state.stx + state.sty)
        } else if state.stp > state.stx {
            stmax
        } else {
            stmin
        };
        bound = false;
    }
    if fp > state.fx {
        state.sty = state.stp;
        state.fy = fp;
        state.dy = dp;
    } else {
        if sgnd < 0.0 {
            state.sty = state.stx;
            state.fy = state.fx;
            state.dy = state.dx;
        }
        state.stx = state.stp;
        state.fx = fp;
        state.dx = dp;
    }
    state.stp = stpf.clamp(stmin, stmax);
    if state.brackt && bound {
        state.stp = if state.sty > state.stx {
            state.stp.min(state.stx + 0.66 * (state.sty - state.stx))
        } else {
            state.stp.max(state.stx + 0.66 * (state.sty - state.stx))
        };
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructor_facades_preserve_owned_optimizer_state() {
        assert_eq!(svl_optimizer(), SvlOptimizer::default());
        let mut sized = svl_optimizer_with_size(2);
        sized.x[1] = 4.0;
        sized.df[0] = -3.0;
        assert_eq!(svl_optimizer_copy(&sized), sized);
        free_svl_optimizer(sized);
    }

    struct Quadratic;
    impl OptimizerObjective for Quadratic {
        fn objective(&self, x: &[f64]) -> f64 {
            (x[0] - 3.0).powi(2) + (x[1] + 2.0).powi(2)
        }
        fn gradient(&self, x: &[f64], g: &mut [f64]) {
            g[0] = 2.0 * (x[0] - 3.0);
            g[1] = 2.0 * (x[1] + 2.0);
        }
    }
    #[test]
    fn owned_lbfgs_minimizes_quadratic() {
        let mut optimizer = SvlOptimizer::with_size(2);
        let value = optimizer.solve(&Quadratic, 40, 1e-12, false).unwrap();
        assert!(value < 1e-10);
        assert!((optimizer.x[0] - 3.0).abs() < 1e-5);
    }
    #[test]
    fn initialization_and_step_state_are_checked() {
        let mut optimizer = SvlOptimizer::new();
        assert!(optimizer.initialize(2, Some(&[1.])).is_err());
        optimizer.initialize(2, None).unwrap();
        assert_eq!(optimizer.size(), 2);
        let mut state = LbfgsStepState {
            stx: 0.,
            fx: 1.,
            dx: -1.,
            sty: 0.,
            fy: 1.,
            dy: -1.,
            stp: 1.,
            brackt: false,
        };
        assert!(lbfgs_step(&mut state, 0.5, -0.5, 0., 2.));
    }
}
