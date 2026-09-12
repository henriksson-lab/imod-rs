//! Translation of `IMOD/3dmod/histwidget.cpp` and `histwidget.h`.
//!
//! `QWidget`, `QPainter`, `QPen`, and `QPaintEvent` are represented by the
//! small boundary trait below.  The histogram state and all of the source
//! arithmetic remain in this translation unit.
#![allow(dead_code)]

/// Native Qt painting surface used by `HistWidget::paintEvent`.
pub trait HistWidgetNativeBoundary {
    fn set_black_pen(&mut self, width: i32);
    fn draw_line(&mut self, x1: i32, y1: i32, x2: i32, y2: i32);
}

/// `HistWidget`.
#[derive(Clone, Debug, PartialEq)]
pub struct HistWidget {
    pub hist: [f32; 256],
    pub max_hist: f32,
    pub min_hist: f32,
    pub min: i32,
    pub max: i32,
}

impl Default for HistWidget {
    fn default() -> Self {
        Self {
            hist: [0.0; 256],
            max_hist: 0.0,
            min_hist: 0.0,
            min: 0,
            max: 0,
        }
    }
}

impl HistWidget {
    /// `HistWidget::HistWidget`.
    ///
    /// The source constructor delegates entirely to `QWidget(parent)`; the
    /// caller owns construction of the native parent/widget surface.
    pub fn new() -> Self {
        Self::default()
    }

    /// `HistWidget::setHistMinMax`.
    pub fn set_hist_min_max(&mut self) {
        let mut maxbin1 = 0;
        let mut maxbin2 = -1;
        let mut max2 = -1.0;
        let mut max3 = -1.0;
        self.min_hist = self.hist[0];
        self.max_hist = self.hist[0];
        for i in 1..256 {
            if self.hist[i] > self.max_hist {
                self.max_hist = self.hist[i];
                maxbin1 = i as i32;
            }
            if self.hist[i] < self.min_hist {
                self.min_hist = self.hist[i];
            }
        }
        for i in 0..256 {
            if i as i32 != maxbin1 && self.hist[i] > max2 {
                max2 = self.hist[i];
                maxbin2 = i as i32;
            }
        }
        for i in 0..256 {
            if i as i32 != maxbin1 && i as i32 != maxbin2 && self.hist[i] > max3 {
                max3 = self.hist[i];
            }
        }
        if max3 > 0.0 && self.max_hist > 1.1 * max3 {
            self.max_hist = 1.1 * max3;
        }
    }

    /// `HistWidget::setMinMax`.
    pub fn set_min_max(&mut self, min_in: i32, max_in: i32) {
        self.min = min_in;
        self.max = max_in;
    }

    /// `HistWidget::getHist`.
    pub fn get_hist(&mut self) -> &mut [f32; 256] {
        &mut self.hist
    }

    /// `HistWidget::thresholdForPercentile`.
    pub fn threshold_for_percentile(&self, percentile: f32) -> f32 {
        let mut i = 0usize;
        let mut curr_percent = 0.0;
        while i < 256 && curr_percent < percentile {
            curr_percent += self.hist[i];
            i += 1;
        }
        if i > 0 && curr_percent > percentile && self.hist[i - 1] > 0.0 {
            return i as f32 - (curr_percent - percentile) / self.hist[i - 1];
        }
        i as f32 + 0.5
    }

    /// `HistWidget::percentileAtPoint`.
    pub fn percentile_at_point(&self, threshold: f32) -> f32 {
        let int_thresh = threshold as i32;
        let mut curr_percent = 0.0;
        let mut i = 0;
        while i < int_thresh && i < 256 {
            curr_percent += self.hist[i as usize];
            i += 1;
        }
        if i < 256 {
            curr_percent += (threshold - int_thresh as f32) * self.hist[i as usize];
        }
        curr_percent
    }

    /// `HistWidget::paintEvent`.
    pub fn paint_event(
        &self,
        x_range: i32,
        y_range: i32,
        native: &mut dyn HistWidgetNativeBoundary,
    ) {
        let denominator = self.max - self.min + 1;
        native.set_black_pen(x_range / denominator);
        for i in self.min..=self.max {
            let x_coord = x_range as f32 * (i - self.min) as f32 / (self.max - self.min) as f32;
            native.draw_line(
                x_coord as i32,
                (y_range as f32 * (self.max_hist - self.hist[i as usize])
                    / (self.max_hist - self.min_hist)) as i32,
                x_coord as i32,
                y_range,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Native {
        width: i32,
        lines: Vec<(i32, i32, i32, i32)>,
    }
    impl HistWidgetNativeBoundary for Native {
        fn set_black_pen(&mut self, width: i32) {
            self.width = width;
        }
        fn draw_line(&mut self, x1: i32, y1: i32, x2: i32, y2: i32) {
            self.lines.push((x1, y1, x2, y2));
        }
    }

    #[test]
    fn limits_peak_to_110_percent_of_third_highest_bin() {
        let mut widget = HistWidget::new();
        widget.hist[10] = 100.0;
        widget.hist[11] = 50.0;
        widget.hist[12] = 40.0;
        widget.set_hist_min_max();
        assert_eq!(widget.min_hist, 0.0);
        assert_eq!(widget.max_hist, 44.0);
    }

    #[test]
    fn percentiles_use_the_source_interpolation() {
        let mut widget = HistWidget::new();
        widget.hist[0] = 0.2;
        widget.hist[1] = 0.3;
        widget.hist[2] = 0.5;
        assert!((widget.threshold_for_percentile(0.35) - 1.5).abs() < 1.0e-6);
        assert!((widget.percentile_at_point(1.5) - 0.35).abs() < 1.0e-6);
    }

    #[test]
    fn paint_uses_selected_bins_and_qt_boundary() {
        let mut widget = HistWidget::new();
        widget.hist[3] = 1.0;
        widget.hist[4] = 3.0;
        widget.min_hist = 1.0;
        widget.max_hist = 3.0;
        widget.set_min_max(3, 4);
        let mut native = Native::default();
        widget.paint_event(20, 10, &mut native);
        assert_eq!(native.width, 10);
        assert_eq!(native.lines, vec![(0, 10, 0, 10), (20, 0, 20, 10)]);
    }
}
