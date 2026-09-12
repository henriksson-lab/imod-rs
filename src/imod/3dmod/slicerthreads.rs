//! Translation of `IMOD/3dmod/slicerthreads.cpp`.
//!
//! The original uses file-static state because `SlicerThread::run` has no
//! arguments.  [`SlicerThreadState`] is that state made explicit: it avoids a
//! process-global mutable image while retaining the exact `fillArraySegment`
//! traversal and interpolation algorithm.  Qt worker scheduling, cache
//! access, FFT, and Fourier filtering remain direct integration boundaries.
#![allow(dead_code)]

use super::slicer::{SLICE_ZSCALE_AFTER, SlicerFuncs};

pub const MAX_THREADS: usize = 16;

/// Source `ivwFastGetValue` / pyramid-cache boundary.
pub trait SlicerVoxelSource: Sync {
    fn fast_get_value(&self, x: i32, y: i32, z: i32) -> i32;
}

/// A direct, in-memory form of an IMOD byte/ushort volume, useful both for
/// the source translation and conformance fixtures.
#[derive(Clone, Debug)]
pub struct SlicerVolume {
    pub xsize: i32,
    pub ysize: i32,
    pub zsize: i32,
    pub data: Vec<i32>,
}

impl SlicerVoxelSource for SlicerVolume {
    fn fast_get_value(&self, x: i32, y: i32, z: i32) -> i32 {
        self.data[(x + self.xsize * (y + self.ysize * z)) as usize]
    }
}

/// The data saved in file statics by `slicerthreads.cpp` for its workers.
pub struct SlicerThreadState<'a> {
    pub xsx: f32,
    pub ysx: f32,
    pub zsx: f32,
    pub xsy: f32,
    pub ysy: f32,
    pub zsy: f32,
    pub xsz: f32,
    pub ysz: f32,
    pub zsz: f32,
    pub xsize: i32,
    pub ysize: i32,
    pub zsize: i32,
    pub xzoff_start: f32,
    pub yzoff_start: f32,
    pub zzoff_start: f32,
    pub izoom: i32,
    pub shortcut: i32,
    pub ilim_short: i32,
    pub jlim_short: i32,
    pub isize: i32,
    pub jsize: i32,
    pub ksize: i32,
    pub no_data_val: i32,
    pub cidata: Vec<i32>,
    pub max_val: i32,
    pub min_val: i32,
    pub ss_hq: i32,
    pub ss_winx: i32,
    pub voxels: &'a dyn SlicerVoxelSource,
}

impl<'a> SlicerThreadState<'a> {
    /// `SlicerThread::SlicerThread`.
    pub fn new(voxels: &'a dyn SlicerVoxelSource, xsize: i32, ysize: i32, zsize: i32) -> Self {
        Self {
            xsx: 1.,
            ysx: 0.,
            zsx: 0.,
            xsy: 0.,
            ysy: 1.,
            zsy: 0.,
            xsz: 0.,
            ysz: 0.,
            zsz: 1.,
            xsize,
            ysize,
            zsize,
            xzoff_start: 0.,
            yzoff_start: 0.,
            zzoff_start: 0.,
            izoom: 1,
            shortcut: 0,
            ilim_short: 0,
            jlim_short: 0,
            isize: 0,
            jsize: 0,
            ksize: 1,
            no_data_val: 0,
            cidata: Vec::new(),
            max_val: 255,
            min_val: 0,
            ss_hq: 0,
            ss_winx: 0,
            voxels,
        }
    }

    /// `SlicerThread::run`.
    pub fn run(&mut self, jstart: i32, jlimit: i32) {
        self.fill_array_segment(jstart, jlimit);
    }

    /// `fillArraySegment`.  Calls are intentionally sequential here; a Qt
    /// worker adapter may give disjoint rows to this same source algorithm.
    pub fn fill_array_segment(&mut self, jstart: i32, jlimit: i32) {
        let mut xzo = self.xzoff_start;
        let mut yzo = self.yzoff_start;
        let mut zzo = self.zzoff_start;
        for k in 0..self.ksize {
            let mut xo = xzo;
            let mut yo = yzo;
            let mut zo = zzo;
            for _ in 0..jstart {
                xo += self.xsy;
                yo += self.ysy;
                zo += self.zsy;
            }
            for j in jstart..jlimit {
                let mut fstart = 0.;
                let mut fend = self.isize as f32;
                find_index_limits(
                    self.isize,
                    self.xsize,
                    xo,
                    self.xsx,
                    0.,
                    &mut fstart,
                    &mut fend,
                );
                find_index_limits(
                    self.isize,
                    self.ysize,
                    yo,
                    self.ysx,
                    0.,
                    &mut fstart,
                    &mut fend,
                );
                find_index_limits(
                    self.isize,
                    self.zsize,
                    zo,
                    self.zsx,
                    0.5,
                    &mut fstart,
                    &mut fend,
                );
                let (mut outer_start, mut inner_start, mut inner_end, mut outer_end) =
                    if fstart >= fend {
                        let middle = self.isize / 2;
                        (middle, middle, middle, middle)
                    } else {
                        let mut outer_start = (fstart - 2.) as i32;
                        if outer_start < 0 {
                            outer_start = 0;
                        }
                        let mut outer_end = (fend + 2.) as i32;
                        if outer_end > self.isize {
                            outer_end = self.isize;
                        }
                        let (inner_start, inner_end) = if self.ss_hq == 0 {
                            let mut inner_start = outer_start + 4;
                            let mut inner_end = outer_end - 4;
                            if inner_start >= inner_end {
                                inner_start = outer_start;
                                inner_end = outer_start;
                            }
                            (inner_start, inner_end)
                        } else {
                            (outer_start, outer_start)
                        };
                        (outer_start, inner_start, inner_end, outer_end)
                    };
                if self.ss_hq != 0 && self.shortcut != 0 {
                    if j >= self.izoom && j < self.jlim_short && j % self.izoom != 0 {
                        outer_start = 0;
                        outer_end = self.isize;
                    } else {
                        outer_start = self.izoom * (outer_start / self.izoom);
                    }
                }
                let cindex = (j * self.isize) as usize;
                for i in 0..outer_start {
                    if k != 0 {
                        self.cidata[cindex + i as usize] += self.no_data_val;
                    } else {
                        self.cidata[cindex + i as usize] = self.no_data_val;
                    }
                }
                for i in outer_end..self.isize {
                    if k != 0 {
                        self.cidata[cindex + i as usize] += self.no_data_val;
                    } else {
                        self.cidata[cindex + i as usize] = self.no_data_val;
                    }
                }
                if self.ss_hq != 0 {
                    let mut x = xo + outer_start as f32 * self.xsx;
                    let mut y = yo + outer_start as f32 * self.ysx;
                    let mut z = zo + outer_start as f32 * self.zsx;
                    let mut i = outer_start;
                    while i < outer_end {
                        let xi = x as i32;
                        let yi = y as i32;
                        let zi = (z + 0.5) as i32;
                        let val = if x >= 0.
                            && xi < self.xsize
                            && y >= 0.
                            && yi < self.ysize
                            && z > -0.5
                            && zi < self.zsize
                        {
                            let value = self.voxels.fast_get_value(xi, yi, zi);
                            let dx = x - xi as f32 - 0.5;
                            let dy = y - yi as f32 - 0.5;
                            let dz = z - zi as f32;
                            let pxi = (xi - 1).max(0);
                            let nxi = if xi + 1 >= self.xsize { xi } else { xi + 1 };
                            let pyi = (yi - 1).max(0);
                            let nyi = if yi + 1 >= self.ysize { yi } else { yi + 1 };
                            let pzi = (zi - 1).max(0);
                            let nzi = if zi + 1 >= self.zsize { zi } else { zi + 1 };
                            let x1 = self.voxels.fast_get_value(pxi, yi, zi) as f32;
                            let x2 = self.voxels.fast_get_value(nxi, yi, zi) as f32;
                            let y1 = self.voxels.fast_get_value(xi, pyi, zi) as f32;
                            let y2 = self.voxels.fast_get_value(xi, nyi, zi) as f32;
                            let z1 = self.voxels.fast_get_value(xi, yi, pzi) as f32;
                            let z2 = self.voxels.fast_get_value(xi, yi, nzi) as f32;
                            let ival = ((x1 + x2) * 0.5 - value as f32) * dx * dx
                                + ((y1 + y2) * 0.5 - value as f32) * dy * dy
                                + ((z1 + z2) * 0.5 - value as f32) * dz * dz
                                + (x2 - x1) * 0.5 * dx
                                + (y2 - y1) * 0.5 * dy
                                + (z2 - z1) * 0.5 * dz
                                + value as f32;
                            (ival.clamp(self.min_val as f32, self.max_val as f32) + 0.5) as i32
                        } else {
                            self.no_data_val
                        };
                        if k != 0 {
                            self.cidata[cindex + i as usize] += val;
                        } else {
                            self.cidata[cindex + i as usize] = val;
                        }
                        x += self.xsx;
                        y += self.ysx;
                        z += self.zsx;
                        if self.shortcut != 0
                            && i >= self.izoom
                            && i < self.ilim_short
                            && j >= self.izoom
                            && j < self.jlim_short
                        {
                            let ishort = if j % self.izoom != 0 {
                                self.ilim_short - self.izoom - 1
                            } else {
                                self.izoom - 1
                            };
                            x += self.xsx * ishort as f32;
                            y += self.ysx * ishort as f32;
                            z += self.zsx * ishort as f32;
                            i += ishort;
                        }
                        i += 1;
                    }
                } else {
                    for i in outer_start..outer_end {
                        let x = xo + i as f32 * self.xsx;
                        let y = yo + i as f32 * self.ysx;
                        let z = zo + i as f32 * self.zsx;
                        let xi = x as i32;
                        let yi = y as i32;
                        let zi = (z + 0.5) as i32;
                        let inside = i >= inner_start && i < inner_end
                            || (x >= 0.
                                && xi < self.xsize
                                && y >= 0.
                                && yi < self.ysize
                                && z > -0.5
                                && zi < self.zsize);
                        let val = if inside {
                            self.voxels.fast_get_value(xi, yi, zi)
                        } else {
                            self.no_data_val
                        };
                        if k != 0 {
                            self.cidata[cindex + i as usize] += val;
                        } else {
                            self.cidata[cindex + i as usize] = val;
                        }
                    }
                }
                xo += self.xsy;
                yo += self.ysy;
                zo += self.zsy;
            }
            xzo += self.xsz;
            yzo += self.ysz;
            zzo += self.zsz;
        }
    }
}

/// `findIndexLimits`.
pub fn find_index_limits(
    isize: i32,
    xsize: i32,
    xo: f32,
    xsx: f32,
    offset: f32,
    fstart: &mut f32,
    fend: &mut f32,
) {
    let end_coord = xo + (isize - 1) as f32 * xsx + offset;
    let start_coord = xo + offset;
    if (start_coord < 0.1 && end_coord < 0.1)
        || (start_coord >= xsize as f32 - 0.1 && end_coord >= xsize as f32 - 0.1)
    {
        *fstart = isize as f32 / 2.;
        *fend = *fstart;
    } else if xsx > 1.0e-6 || xsx < -1.0e-6 {
        let mut flower = (0.1 - start_coord) / xsx;
        let mut fupper = (xsize as f32 - 0.1 - start_coord) / xsx;
        if xsx < 0. {
            std::mem::swap(&mut flower, &mut fupper);
        }
        if flower > *fstart {
            *fstart = flower;
        }
        if fupper < *fend {
            *fend = fupper;
        }
    }
}

impl SlicerFuncs {
    /// `SlicerFuncs::fillImageArray`, through the array sampling phase.
    /// Image-cache setup, RGB selection, statistics/FFT/filter dispatch and
    /// Qt threading are paired subsystem boundaries; all coordinate setup and
    /// the worker algorithm are preserved here.
    pub fn fill_image_array(
        &mut self,
        state: &mut SlicerThreadState<'_>,
        panning: i32,
        mean_only: i32,
        _rgb_channel: i32,
    ) {
        if self.zoom <= 0. {
            return;
        }
        state.xsize = self.view.xsize;
        state.ysize = self.view.ysize;
        state.zsize = self.view.zsize;
        state.max_val = 255;
        state.min_val = 0;
        state.no_data_val = 0;
        state.ss_winx = self.winx;
        state.izoom = self.zoom as i32;
        state.ss_hq = if mean_only != 0 { 0 } else { self.hq };
        self.no_pixel_zoom = (state.ss_hq != 0 || state.izoom as f32 != self.zoom)
            && (self.fft_mode == 0 || self.zoom < 1.);
        state.shortcut = if state.ss_hq != 0
            && (self.zoom == state.izoom as f32 || self.zoom > 2.)
            && self.zoom > 1.
            && self.fft_mode == 0
        {
            1
        } else {
            0
        };
        state.ksize = if mean_only != 0 {
            1
        } else {
            self.nslice.max(1)
        };
        if panning != 0 {
            state.ss_hq = 0;
            state.shortcut = 0;
        }
        let mut xzoom = self.zoom;
        let mut yzoom = self.zoom;
        let mut zzoom = self.zoom;
        let mut xo = self.cx;
        let mut yo = self.cy;
        let mut zo = self.cz;
        self.lang = self.tang;
        state.xsx = self.xstep[0];
        state.ysx = self.xstep[1];
        state.zsx = self.xstep[2];
        state.xsy = self.ystep[0];
        state.ysy = self.ystep[1];
        state.zsy = self.ystep[2];
        state.xsz = self.zstep[0];
        state.ysz = self.zstep[1];
        state.zsz = self.zstep[2];
        let mut zs = 1. / self.get_z_scale_before();
        if self.scalez == SLICE_ZSCALE_AFTER && self.view.zscale > 0. {
            zs = self.view.zscale;
            xzoom = self.zoom
                * ((state.xsx * state.xsx
                    + state.ysx * state.ysx
                    + state.zsx * state.zsx * zs * zs)
                    / (state.xsx * state.xsx + state.ysx * state.ysx + state.zsx * state.zsx))
                    .sqrt();
            yzoom = self.zoom
                * ((state.xsy * state.xsy
                    + state.ysy * state.ysy
                    + state.zsy * state.zsy * zs * zs)
                    / (state.xsy * state.xsy + state.ysy * state.ysy + state.zsy * state.zsy))
                    .sqrt();
            zzoom = self.zoom
                * ((state.xsz * state.xsz
                    + state.ysz * state.ysz
                    + state.zsz * state.zsz * zs * zs)
                    / (state.xsz * state.xsz + state.ysz * state.ysz + state.zsz * state.zsz))
                    .sqrt();
            zs = 1.;
        }
        state.isize = (self.winx as f32 / self.zoom + 0.9) as i32;
        state.jsize = (self.winy as f32 / self.zoom + 0.9) as i32;
        self.remaining_zoom = 1.;
        if self.no_pixel_zoom {
            if self.hq != 0 && self.zoom < 1. {
                self.remaining_zoom = self.zoom;
            }
            state.xsx *= self.remaining_zoom / xzoom;
            state.ysx *= self.remaining_zoom / xzoom;
            state.zsx *= self.remaining_zoom * zs / xzoom;
            state.xsy *= self.remaining_zoom / yzoom;
            state.ysy *= self.remaining_zoom / yzoom;
            state.zsy *= self.remaining_zoom * zs / yzoom;
            xo -= state.isize as f32 / 2. * state.xsx + state.jsize as f32 / 2. * state.xsy;
            yo -= state.isize as f32 / 2. * state.ysx + state.jsize as f32 / 2. * state.ysy;
            zo -= state.isize as f32 / 2. * state.zsx + state.jsize as f32 / 2. * state.zsy;
        } else {
            state.xsx *= self.zoom / xzoom;
            state.ysx *= self.zoom / xzoom;
            state.zsx *= zs * self.zoom / xzoom;
            state.xsy *= self.zoom / yzoom;
            state.ysy *= self.zoom / yzoom;
            state.zsy *= zs * self.zoom / yzoom;
            xo -= (self.winx as f32 / 2. / self.zoom) * state.xsx
                + (self.winy as f32 / 2. / self.zoom) * state.xsy;
            yo -= (self.winx as f32 / 2. / self.zoom) * state.ysx
                + (self.winy as f32 / 2. / self.zoom) * state.ysy;
            zo -= (self.winx as f32 / 2. / self.zoom) * state.zsx
                + (self.winy as f32 / 2. / self.zoom) * state.zsy;
        }
        state.xsz *= self.zoom / zzoom;
        state.ysz *= self.zoom / zzoom;
        state.zsz *= zs * self.zoom / zzoom;
        self.xo = xo;
        self.yo = yo;
        self.zo = zo;
        let zoffset = (state.ksize - 1) as f32 * 0.5;
        state.xzoff_start = xo - zoffset * state.xsz;
        state.yzoff_start = yo - zoffset * state.ysz;
        state.zzoff_start = zo - zoffset * state.zsz;
        if state.shortcut != 0 {
            state.ilim_short = state.izoom * ((state.isize - 1) / state.izoom - 1);
            state.jlim_short = state.izoom * ((state.jsize - 1) / state.izoom - 1);
        }
        if state.izoom == 0 {
            state.izoom = 1;
        }
        state.cidata.resize((state.isize * state.jsize) as usize, 0);
        state.fill_array_segment(0, state.jsize);
        self.xzoom = xzoom;
        self.yzoom = yzoom;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::three_dmod::slicer::SlicerView;

    #[test]
    fn find_index_limits_clips_and_skips() {
        let (mut start, mut end) = (0., 10.);
        find_index_limits(10, 5, -2., 1., 0., &mut start, &mut end);
        assert_eq!((start, end), (2.1, 6.9));
        find_index_limits(10, 5, -10., 1., 0., &mut start, &mut end);
        assert_eq!(start, 5.);
        assert_eq!(end, 5.);
    }

    #[test]
    fn fill_array_segment_samples_volume_and_padding() {
        let volume = SlicerVolume {
            xsize: 3,
            ysize: 1,
            zsize: 1,
            data: vec![10, 20, 30],
        };
        let mut state = SlicerThreadState::new(&volume, 3, 1, 1);
        state.isize = 5;
        state.jsize = 1;
        state.xzoff_start = -1.;
        state.yzoff_start = 0.5;
        state.cidata.resize(5, -1);
        state.fill_array_segment(0, 1);
        assert_eq!(state.cidata, vec![0, 10, 20, 30, 0]);
    }

    #[test]
    fn fill_image_array_preserves_source_step_setup() {
        let volume = SlicerVolume {
            xsize: 3,
            ysize: 3,
            zsize: 1,
            data: vec![7; 9],
        };
        let mut slicer = SlicerFuncs::new(
            SlicerView {
                xsize: 3,
                ysize: 3,
                zsize: 1,
                ..Default::default()
            },
            0,
        );
        slicer.winx = 3;
        slicer.winy = 3;
        slicer.cx = 1.;
        slicer.cy = 1.;
        slicer.cz = 0.;
        let mut state = SlicerThreadState::new(&volume, 3, 3, 1);
        slicer.fill_image_array(&mut state, 0, 0, 0);
        assert_eq!(state.cidata.len(), 9);
        assert_eq!(state.cidata[4], 7);
    }
}
