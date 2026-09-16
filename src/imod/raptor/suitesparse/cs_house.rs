//! Translation of `IMOD/raptor/suitesparse/cs_house.c`.

/// C `cs_house`: overwrites a vector with a Householder reflector and returns `(s, beta)`.
pub fn cs_house(values: &mut [f64]) -> Option<(f64, f64)> {
    let first = *values.first()?;
    let sigma = values[1..].iter().map(|value| value * value).sum::<f64>();
    if sigma == 0.0 {
        values[0] = 1.0;
        Some((first.abs(), if first <= 0.0 { 2.0 } else { 0.0 }))
    } else {
        let s = (first * first + sigma).sqrt();
        values[0] = if first <= 0.0 {
            first - s
        } else {
            -sigma / (first + s)
        };
        Some((s, -1.0 / (s * values[0])))
    }
}
