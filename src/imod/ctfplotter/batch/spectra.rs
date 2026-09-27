//! Reading the tilt series, dividing each view into overlapping tiles, and
//! computing the hyper-resolution rotationally averaged power spectrum of
//! each strip of tiles.
//!
//! The tiling follows the native program (`slicecache.cpp`,
//! `setupStripCache`/`getHyperPS`): tiles of `raw` pixels overlapping by 50%
//! with a 16-pixel trim, narrowed perpendicular to the tilt axis at high tilt,
//! tapered 9 pixels at their edges, and grouped into strips parallel to the
//! tilt axis whose defocus spread is under the defocus tolerance.  Each strip
//! is split into its left and right halves; for each half the sum over tiles
//! of the FFT power is binned into `hyper` sub-bins per final spectrum bin,
//! and optionally also into angular sectors for astigmatism analysis.
//!
//! Differences from native, all intentional: FFTs are done here with
//! `rustfft` (plans shared across threads) instead of IMOD's `todfft`, the
//! cropped spectrum is taken directly from the raw-tile FFT instead of
//! through `fourierReduceImage`, and the whole view is processed at once with
//! `rayon`.

use std::collections::HashMap;
use std::sync::Arc;

use rayon::prelude::*;
use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};

use crate::imod::libcfshr::b3dutil::ImodFile;
use crate::imod::libcfshr::taperpad::{PadIn, slice_taper_in_pad};
use crate::imod::libiimod::iimage::ii_fopen;
use crate::imod::libiimod::mrcfiles::{MrcHeader, mrc_head_read, mrc_read_slice};

use super::ctf::{MY_PI, RADIANS_PER_DEGREE};

/// Tilt angles below this (radians) are treated as zero.
const MIN_ANGLE: f64 = 1.0e-6;

/// An open image stack whose views are read as floats, with an optional
/// offset added to make values proportional to electrons.
pub struct ImageStack {
    fp: ImodFile,
    pub header: MrcHeader,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub data_offset: f32,
    bytes: Vec<u8>,
}

impl ImageStack {
    pub fn open(path: &str) -> Result<Self, String> {
        let Some(mut fp) = ii_fopen(path.as_bytes(), "rb") else {
            return Err(format!("could not open input file {path}"));
        };
        let mut header = MrcHeader::default();
        if mrc_head_read(&mut fp, &mut header) != 0 {
            return Err(format!("could not read header of input file {path}"));
        }
        let bytes_per = match header.mode {
            0 => 1,
            1 | 6 => 2,
            2 => 4,
            mode => return Err(format!("unsupported data mode {mode} in {path}")),
        };
        let (nx, ny, nz) = (header.nx as usize, header.ny as usize, header.nz as usize);
        Ok(ImageStack {
            fp,
            header,
            nx,
            ny,
            nz,
            data_offset: 0.0,
            bytes: vec![0u8; nx * ny * bytes_per],
        })
    }

    /// Reads view `z` (from 0) as floats with the data offset added.
    pub fn read_view(&mut self, z: usize) -> Result<Vec<f32>, String> {
        if mrc_read_slice(
            &mut self.bytes,
            &mut self.fp,
            &mut self.header,
            z as i32,
            b'Z',
        ) != 0
        {
            return Err(format!("Reading slice {z}"));
        }
        let off = self.data_offset;
        let b = &self.bytes;
        let out = match self.header.mode {
            0 => b.iter().map(|&v| v as f32 + off).collect(),
            1 => b
                .chunks_exact(2)
                .map(|c| i16::from_ne_bytes([c[0], c[1]]) as f32 + off)
                .collect(),
            6 => b
                .chunks_exact(2)
                .map(|c| u16::from_ne_bytes([c[0], c[1]]) as f32 + off)
                .collect(),
            _ => b
                .chunks_exact(4)
                .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]) + off)
                .collect(),
        };
        Ok(out)
    }
}

/// Parameters that determine how views are cut into tiles and strips.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TileParams {
    /// Size of the spectrum tile (the `TileSize` entry).
    pub tile: usize,
    /// Size of the tile read from the image (larger when cropping).
    pub raw: usize,
    /// Defocus tolerance in nm that sets the strip width.
    pub def_tol: f64,
    pub left_tol: f64,
    pub right_tol: f64,
    /// Tilt axis angle from Y, degrees, CCW positive.
    pub axis_angle: f64,
    /// Offset added to tilt angles, degrees.
    pub tilt_offset: f64,
    /// True image pixel size in nm.
    pub pixel_size: f64,
}

/// One strip: the tiles in it and the mean distance from the tilt axis of
/// the tiles on each side.
#[derive(Clone, Debug, Default)]
pub struct StripDef {
    /// (tile x index, tile y index, is on left of axis)
    pub tiles: Vec<(usize, usize, bool)>,
    pub n_left: usize,
    pub n_right: usize,
    pub left_dist: f64,
    pub right_dist: f64,
}

/// Placement of tiles in one view.
#[derive(Clone, Debug, Default)]
pub struct ViewGeometry {
    pub nxt: usize,
    pub nyt: usize,
    pub x_spacing: i64,
    pub x_start: i64,
    pub y_spacing: i64,
    pub y_start: i64,
    pub x_pad: i64,
    pub y_pad: i64,
    pub strips: Vec<StripDef>,
}

fn nint(v: f64) -> i64 {
    (v + if v >= 0.0 { 0.5 } else { -0.5 }) as i64
}

/// Lays out tiles and strips for one view with tilt angle `angle` degrees
/// (`None` for a noise image, which is analyzed as untilted).
pub fn view_geometry(nx: usize, ny: usize, angle: Option<f64>, p: &TileParams) -> ViewGeometry {
    let num_trim = 16i64;
    let (nxi, nyi) = (nx as i64, ny as i64);
    let nx_trim = nxi - num_trim;
    let ny_trim = nyi - num_trim;
    let raw = p.raw as i64;
    let half = raw / 2;
    let max_aspect = 2.5f64;
    let diagonal = ((nxi * nxi + nyi * nyi) as f64).sqrt();
    let mut axis = p.axis_angle;
    while axis > 180.0 {
        axis -= 360.0;
    }
    while axis <= -180.0 {
        axis += 360.0;
    }
    let pad_x_axis = axis.abs() < 45.0 || axis.abs() > 135.0;

    // Strip width from the tilt angle and defocus tolerance
    let cur = angle.map_or(0.0, |a| a * RADIANS_PER_DEGREE);
    let mut strip_width = if cur.abs() > MIN_ANGLE {
        (p.def_tol / cur.tan()).abs() / p.pixel_size
    } else {
        diagonal
    };
    strip_width = strip_width.min(diagonal);
    let mut g = ViewGeometry::default();
    if strip_width < raw as f64 {
        let mut raw_x = 2 * nint(0.5 * raw as f64 / max_aspect);
        raw_x = raw_x.max(2 * ((strip_width / 2.0) as i64));
        if pad_x_axis {
            g.x_pad = raw - raw_x;
        } else {
            g.y_pad = raw - raw_x;
        }
    }
    let raw_x = raw - g.x_pad;
    let raw_y = raw - g.y_pad;
    let nxt = nint(nx_trim as f32 as f64 / (raw_x / 2) as f64 - 1.0).max(1);
    g.x_spacing = (nx_trim - raw_x) / (nxt - 1).max(1);
    g.x_start = num_trim / 2 + ((nx_trim - raw_x) - (nxt - 1) * g.x_spacing) / 2;
    let nyt = nint(ny_trim as f32 as f64 / (raw_y / 2) as f64 - 1.0).max(1);
    g.y_spacing = (ny_trim - raw_y) / (nyt - 1).max(1);
    g.y_start = num_trim / 2 + ((ny_trim - raw_y) - (nyt - 1) * g.y_spacing) / 2;
    g.nxt = nxt as usize;
    g.nyt = nyt as usize;

    // Assign tiles to strips
    let mut axis_x = (90.0 + p.axis_angle) * MY_PI / 180.0;
    if axis_x < 0.0 {
        axis_x += 2.0 * MY_PI;
    }
    let cur = angle.map_or(0.0, |a| (a + p.tilt_offset) * RADIANS_PER_DEGREE);
    let tile_max = g.nxt * g.nyt;
    let mut scan_counter = 0usize;
    let mut iter = 0usize;
    while scan_counter < tile_max {
        let mut strip = StripDef::default();
        let (mut lsum, mut rsum) = (0.0f64, 0.0f64);
        for xt in 0..g.nxt {
            for yt in 0..g.nyt {
                let cx =
                    (xt as i64 * g.x_spacing + g.x_start + (half - g.x_pad / 2) - nxi / 2) as f64;
                let cy =
                    (yt as i64 * g.y_spacing + g.y_start + (half - g.y_pad / 2) - nyi / 2) as f64;
                let rad = (cx * cx + cy * cy).sqrt();
                let mut alpha = cy.atan2(cx);
                if alpha < 0.0 {
                    alpha += 2.0 * MY_PI;
                }
                let dist = rad * (alpha - axis_x).sin().abs();
                if dist < iter as f64 * strip_width / 2.0
                    || dist > (iter + 1) as f64 * strip_width / 2.0
                {
                    continue;
                }
                scan_counter += 1;
                let on_left = if axis_x <= MY_PI {
                    alpha > axis_x && alpha < axis_x + MY_PI
                } else {
                    alpha > axis_x || alpha < axis_x - MY_PI
                };
                let def_diff = dist * p.pixel_size * cur.tan().abs();
                if (on_left && def_diff > p.left_tol) || (!on_left && def_diff > p.right_tol) {
                    continue;
                }
                if on_left {
                    strip.n_left += 1;
                    lsum += dist;
                } else {
                    strip.n_right += 1;
                    rsum += dist;
                }
                strip.tiles.push((xt, yt, on_left));
            }
        }
        strip.left_dist = lsum / strip.n_left.max(1) as f64;
        strip.right_dist = rsum / strip.n_right.max(1) as f64;
        g.strips.push(strip);
        iter += 1;
        if iter > 100_000 {
            break;
        }
    }
    g
}

/// Mapping from FFT components of a tile to hyper-resolution radial bins
/// and angular sectors, and the number of components in each bin.
#[derive(Clone, Debug)]
pub struct Binning {
    pub dim: usize,
    pub hyper: usize,
    pub num_sectors: usize,
    /// Number of hyper bins in one spectrum.
    pub one_ps: usize,
    pub tile: usize,
    pub raw: usize,
    /// (index in the column-major FFT columns, hyper bin), by bin
    to_ps: Vec<(u32, u32)>,
    /// (index in the FFT columns, sector * one_ps + hyper bin), by bin
    to_sector: Vec<(u32, u32)>,
    /// Components per hyper bin, whole spectrum.
    pub count: Vec<i32>,
    /// Components per hyper bin in each sector (sector-major).
    pub sector_count: Vec<i32>,
}

impl Binning {
    pub fn new(dim: usize, hyper: usize, num_sectors: usize, tile: usize, raw: usize) -> Self {
        let one_ps = (dim + 1) * hyper + 1;
        let fft_xdim = tile / 2 + 1;
        let freq_inc = 1.0 / ((dim - 1) * hyper) as f64;
        let angle_inc = 180.0 / num_sectors as f64;
        let mut count = vec![0i32; one_ps];
        let mut sector_count = vec![0i32; one_ps * num_sectors];
        let mut comps = Vec::new();
        let sector_of = |ang: f64| -> usize { (((ang + 90.0) / angle_inc) as usize) % num_sectors };
        for iy in 0..=tile / 2 {
            for ix in 0..fft_xdim {
                let r = (((iy * iy + ix * ix) as f64).sqrt() / (fft_xdim - 1) as f64 / freq_inc)
                    as usize;
                let r = r.min(one_ps - 1);
                let ang = (iy as f64).atan2(ix as f64) / RADIANS_PER_DEGREE;
                let s = sector_of(ang);
                count[r] += 1;
                sector_count[s * one_ps + r] += 1;
                comps.push((iy as u32, ix as u32, r as u32, s as u32));
                if iy > 0 && iy < tile / 2 {
                    let s2 = sector_of(-ang);
                    count[r] += 1;
                    sector_count[s2 * one_ps + r] += 1;
                    comps.push(((raw - iy) as u32, ix as u32, r as u32, s2 as u32));
                }
            }
        }
        // Index tables sorted by destination bin, for cache-friendly sums
        let mut to_ps: Vec<(u32, u32)> = comps
            .iter()
            .map(|&(row, col, r, _)| (col * raw as u32 + row, r))
            .collect();
        to_ps.sort_by_key(|&(src, dst)| (dst, src));
        let mut to_sector: Vec<(u32, u32)> = comps
            .iter()
            .map(|&(row, col, r, s)| (col * raw as u32 + row, s * one_ps as u32 + r))
            .collect();
        to_sector.sort_by_key(|&(src, dst)| (dst, src));
        Binning {
            dim,
            hyper,
            num_sectors,
            one_ps,
            tile,
            raw,
            to_ps,
            to_sector,
            count,
            sector_count,
        }
    }
}

/// Summed power spectrum of the tiles on one side of one strip.
#[derive(Clone, Debug, Default)]
pub struct SideSpectrum {
    /// Sum over tiles of power in each hyper bin.
    pub ps: Vec<f32>,
    /// Same, per sector (sector-major), when sectors were computed.
    pub sectors: Vec<f32>,
    /// Mean of the tapered tiles.
    pub mean: f64,
    /// Number of tiles.
    pub count: usize,
    /// Mean distance of tiles from the tilt axis, pixels.
    pub axis_dist: f64,
}

/// Spectra of all strips of one view: `[left, right]` per strip.
#[derive(Clone, Debug, Default)]
pub struct ViewSpectra {
    pub strips: Vec<[SideSpectrum; 2]>,
    pub has_sectors: bool,
}

/// Computes power spectra of tiles with shared FFT plans.
pub struct TileTransformer {
    raw: usize,
    fft: Arc<dyn Fft<f32>>,
}

impl TileTransformer {
    pub fn new(raw: usize) -> Self {
        let mut planner = FftPlanner::<f32>::new();
        TileTransformer {
            raw,
            fft: planner.plan_fft_forward(raw),
        }
    }

    /// Computes the 2-D FFT of the tapered tile (stride `raw + 2`) for the
    /// columns `0..ncols` and returns them column-major (`ncols * raw`).
    fn transform(
        &self,
        tile: &[f32],
        ncols: usize,
        rowbuf: &mut Vec<Complex32>,
        cols: &mut Vec<Complex32>,
        scratch: &mut Vec<Complex32>,
    ) {
        let r = self.raw;
        let stride = r + 2;
        rowbuf.resize(r, Complex32::new(0.0, 0.0));
        cols.resize(ncols * r, Complex32::new(0.0, 0.0));
        scratch.resize(self.fft.get_inplace_scratch_len(), Complex32::new(0.0, 0.0));
        for y in 0..r {
            for x in 0..r {
                rowbuf[x] = Complex32::new(tile[y * stride + x], 0.0);
            }
            self.fft.process_with_scratch(rowbuf, scratch);
            for x in 0..ncols {
                cols[x * r + y] = rowbuf[x];
            }
        }
        for x in 0..ncols {
            self.fft
                .process_with_scratch(&mut cols[x * r..(x + 1) * r], scratch);
        }
    }
}

/// Accumulator for one side of one strip while tiles are processed.
struct SideAccum {
    ps: Vec<f64>,
    sectors: Vec<f64>,
    mean: f64,
}

/// Computes the strip spectra of one view.
pub fn compute_view_spectra(
    data: &[f32],
    nx: usize,
    geom: &ViewGeometry,
    bin: &Binning,
    transformer: &TileTransformer,
    want_sectors: bool,
) -> ViewSpectra {
    let raw = bin.raw;
    let one_ps = bin.one_ps;
    let ncols = bin.tile / 2 + 1;
    let norm = 1.0 / (raw * raw) as f64;
    let nsec = if want_sectors {
        bin.num_sectors * one_ps
    } else {
        0
    };
    let nthreads = rayon::current_num_threads().max(1);
    let strips = geom
        .strips
        .iter()
        .map(|strip| {
            // A bounded number of chunks, each summed sequentially with its own
            // accumulators, and the chunks then added in order: few large
            // allocations and a summation order independent of scheduling
            let chunk = strip.tiles.len().div_ceil(2 * nthreads).max(1);
            let parts: Vec<(SideAccum, SideAccum)> = strip
                .tiles
                .par_chunks(chunk)
                .map(|tiles| {
                    let mk = || SideAccum {
                        ps: vec![0.0; one_ps],
                        sectors: vec![0.0; nsec],
                        mean: 0.0,
                    };
                    let (mut la, mut ra) = (mk(), mk());
                    let mut tile = vec![0f32; (raw + 2) * raw];
                    let (mut rowbuf, mut cols, mut scratch) = (Vec::new(), Vec::new(), Vec::new());
                    for &(xt, yt, on_left) in tiles {
                        let ix0 = xt as i64 * geom.x_spacing + geom.x_start;
                        let iy0 = yt as i64 * geom.y_spacing + geom.y_start;
                        slice_taper_in_pad(
                            PadIn::Float(data),
                            2,
                            nx as i32,
                            ix0 as i32,
                            (ix0 + raw as i64 - 1 - geom.x_pad) as i32,
                            iy0 as i32,
                            (iy0 + raw as i64 - 1 - geom.y_pad) as i32,
                            &mut tile,
                            (raw + 2) as i32,
                            raw as i32,
                            raw as i32,
                            9,
                            9,
                        );
                        let mut mean = 0.0f64;
                        for y in 0..raw {
                            for x in 0..raw {
                                mean += tile[y * (raw + 2) + x] as f64;
                            }
                        }
                        mean /= (raw * raw) as f64;
                        transformer.transform(&tile, ncols, &mut rowbuf, &mut cols, &mut scratch);
                        let acc = if on_left { &mut la } else { &mut ra };
                        acc.mean += mean;
                        let (table, dest) = if want_sectors {
                            (&bin.to_sector, &mut acc.sectors)
                        } else {
                            (&bin.to_ps, &mut acc.ps)
                        };
                        for &(src, dst) in table {
                            let v = cols[src as usize];
                            dest[dst as usize] +=
                                (v.re as f64 * v.re as f64 + v.im as f64 * v.im as f64) * norm;
                        }
                    }
                    (la, ra)
                })
                .collect();
            let mk = || SideAccum {
                ps: vec![0.0; one_ps],
                sectors: vec![0.0; nsec],
                mean: 0.0,
            };
            let (mut left, mut right) = (mk(), mk());
            for (a2, b2) in &parts {
                for (acc, add) in [(&mut left, a2), (&mut right, b2)] {
                    acc.mean += add.mean;
                    acc.ps.iter_mut().zip(&add.ps).for_each(|(x, y)| *x += y);
                    acc.sectors
                        .iter_mut()
                        .zip(&add.sectors)
                        .for_each(|(x, y)| *x += y);
                }
            }
            // With sectors, the whole spectrum is the sum over sectors
            if want_sectors {
                for acc in [&mut left, &mut right] {
                    for s in 0..bin.num_sectors {
                        for r in 0..one_ps {
                            acc.ps[r] += acc.sectors[s * one_ps + r];
                        }
                    }
                }
            }
            let finish = |acc: SideAccum, count: usize, dist: f64| SideSpectrum {
                ps: acc.ps.iter().map(|&v| v as f32).collect(),
                sectors: acc.sectors.iter().map(|&v| v as f32).collect(),
                mean: if count > 0 {
                    acc.mean / count as f64
                } else {
                    0.0
                },
                count,
                axis_dist: dist,
            };
            [
                finish(left, strip.n_left, strip.left_dist),
                finish(right, strip.n_right, strip.right_dist),
            ]
        })
        .collect();
    ViewSpectra {
        strips,
        has_sectors: want_sectors,
    }
}

/// Cache of view geometry and strip spectra for a stack, invalidated when
/// the tiling parameters change.
pub struct SpectrumCache {
    pub stack: ImageStack,
    pub params: TileParams,
    pub bin: Binning,
    transformer: TileTransformer,
    geoms: Vec<Option<ViewGeometry>>,
    spectra: HashMap<usize, ViewSpectra>,
    /// Views in the order they were computed, to evict the oldest.
    order: Vec<usize>,
    max_bytes: f64,
    used_bytes: f64,
    angles: Option<Vec<f32>>,
}

impl SpectrumCache {
    pub fn new(
        stack: ImageStack,
        params: TileParams,
        bin: Binning,
        angles: Option<Vec<f32>>,
        max_mb: f64,
    ) -> Self {
        let nz = stack.nz;
        SpectrumCache {
            transformer: TileTransformer::new(params.raw),
            stack,
            params,
            bin,
            geoms: vec![None; nz],
            spectra: HashMap::new(),
            order: Vec::new(),
            max_bytes: max_mb * 1024.0 * 1024.0,
            used_bytes: 0.0,
            angles,
        }
    }

    /// Changes tiling or binning parameters, clearing what depends on them.
    pub fn reset(&mut self, params: TileParams, bin: Binning) {
        if params.raw != self.params.raw {
            self.transformer = TileTransformer::new(params.raw);
        }
        self.params = params;
        self.bin = bin;
        self.geoms.iter_mut().for_each(|g| *g = None);
        self.spectra.clear();
        self.order.clear();
        self.used_bytes = 0.0;
    }

    pub fn geometry(&mut self, view: usize) -> &ViewGeometry {
        if self.geoms[view].is_none() {
            let angle = self.angles.as_ref().map(|a| a[view] as f64);
            self.geoms[view] = Some(view_geometry(
                self.stack.nx,
                self.stack.ny,
                angle,
                &self.params,
            ));
        }
        self.geoms[view].as_ref().unwrap()
    }

    /// The strip spectra of a view (after [`Self::ensure_view`]) with the
    /// binning they were computed with.
    pub fn get(&self, view: usize) -> (&ViewSpectra, &Binning) {
        (self.spectra.get(&view).unwrap(), &self.bin)
    }

    /// Computes the strip spectra of a view if they are not cached.
    pub fn ensure_view(&mut self, view: usize, want_sectors: bool) -> Result<(), String> {
        let have = self
            .spectra
            .get(&view)
            .is_some_and(|s| s.has_sectors || !want_sectors);
        if !have {
            self.geometry(view);
            let data = self.stack.read_view(view)?;
            let geom = self.geoms[view].as_ref().unwrap();
            let spec = compute_view_spectra(
                &data,
                self.stack.nx,
                geom,
                &self.bin,
                &self.transformer,
                want_sectors,
            );
            let bytes = spec
                .strips
                .iter()
                .map(|s| {
                    (s[0].ps.len() + s[0].sectors.len() + s[1].ps.len() + s[1].sectors.len()) * 4
                })
                .sum::<usize>() as f64;
            if let Some(old) = self.spectra.remove(&view) {
                self.used_bytes -= old
                    .strips
                    .iter()
                    .map(|s| {
                        (s[0].ps.len() + s[0].sectors.len() + s[1].ps.len() + s[1].sectors.len())
                            * 4
                    })
                    .sum::<usize>() as f64;
                self.order.retain(|&v| v != view);
            }
            while self.used_bytes + bytes > self.max_bytes && !self.order.is_empty() {
                let oldest = self.order.remove(0);
                if let Some(old) = self.spectra.remove(&oldest) {
                    self.used_bytes -= old
                        .strips
                        .iter()
                        .map(|s| {
                            (s[0].ps.len()
                                + s[0].sectors.len()
                                + s[1].ps.len()
                                + s[1].sectors.len())
                                * 4
                        })
                        .sum::<usize>() as f64;
                }
            }
            self.used_bytes += bytes;
            self.order.push(view);
            self.spectra.insert(view, spec);
        }
        Ok(())
    }
}
