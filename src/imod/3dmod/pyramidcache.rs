//! Translation of `IMOD/3dmod/pyramidcache.cpp` and `pyramidcache.h`.
//!
//! This is deliberately a tile cache, rather than a replacement image store:
//! opening an IFD, montage piece selection, and reading a binned section are
//! supplied by [`PyramidCacheBoundary`], exactly at the `ivwReadBinnedSection`
//! seam in the C++ source.
#![allow(dead_code, unused_variables)]

use std::collections::{BTreeMap, VecDeque};

use crate::imod::three_dmod::mv_image::{ImagePixel, MvImageSource};

pub const FAKE_STRIP_PIXELS: i32 = 1_000_000;
pub const DEFAULT_TILE_CACHE_LIMIT: i32 = 20_000;
pub const FULL_SEC_BUF: usize = 3;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct CacheSlice {
    pub cache_ind: i32,
    pub section: i32,
    pub x_tile_ind: i32,
    pub y_tile_ind: i32,
    pub used_at_count: f64,
    pub xsize: i32,
    pub ysize: i32,
    pub data: Vec<u8>,
}
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TileCache {
    pub x_tile_size: i32,
    pub y_tile_size: i32,
    pub x_overlap: i32,
    pub y_overlap: i32,
    pub num_x_tiles: i32,
    pub num_y_tiles: i32,
    pub xy_scale: i32,
    pub z_scale: i32,
    pub tile_index: Vec<i32>,
    pub full_x_size: i32,
    pub full_y_size: i32,
    pub full_z_size: i32,
    pub plist_size: i32,
    pub pcoords: Vec<i32>,
    pub x_offset: f32,
    pub y_offset: f32,
    pub z_offset: f32,
    pub min_x_load: i32,
    pub max_x_load: i32,
    pub min_y_load: i32,
    pub max_y_load: i32,
    pub min_z_load: i32,
    pub max_z_load: i32,
    pub load_x_size: i32,
    pub load_y_size: i32,
    pub load_z_size: i32,
    pub tile_x_delta: i32,
    pub tile_y_delta: i32,
    pub first_x_offset: i32,
    pub first_y_offset: i32,
    pub file_y_tile_offset: i32,
    pub start_x_tile: i32,
    pub start_y_tile: i32,
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LoadTileRequest {
    pub cache_ind: i32,
    pub section: i32,
    pub x_tile_ind: i32,
    pub y_tile_ind: i32,
    pub ind_in_cache: i32,
    pub buf_ind: i32,
    pub num_x_copy: i32,
    pub num_y_copy: i32,
    pub from_x_start: i32,
    pub from_y_start: i32,
    pub to_x_start: i32,
    pub to_y_start: i32,
}
/// `FastSegment` (`pyramidcache.h:70-76`).  The source's `unsigned char *line`
/// points into the tile that owns the data; here it is the borrow of that
/// tile's buffer starting at the same element.
#[derive(Clone, Copy, Debug)]
pub struct FastSegment<'a> {
    pub xor_y: i32,
    pub yor_z: i32,
    pub length: i32,
    pub stride: i32,
    pub line: &'a [u8],
}

/// Source-visible `ViewInfo` members used by this unit, separated from Qt and I/O.
#[derive(Clone, Debug, Default)]
pub struct PyramidCacheView {
    pub xsize: i32,
    pub ysize: i32,
    pub zsize: i32,
    pub xybin: i32,
    pub ushort_store: bool,
    pub white: i32,
    pub black: i32,
    pub zmouse: f32,
    pub xmin: i32,
    pub xmax: i32,
    pub ymin: i32,
    pub ymax: i32,
    pub zmin: i32,
    pub zmax: i32,
}
/// The direct `ivwReadBinnedSection`/event loop boundary in this source file.
pub trait PyramidCacheBoundary {
    fn read_binned_tile(
        &mut self,
        cache_ind: i32,
        file_z: i32,
        limits: (i32, i32, i32, i32),
        out: &mut [u8],
    ) -> Result<(), ()>;
    fn process_events(&mut self) {}
    fn start_tile_loading(&mut self) {}
}

pub struct PyramidCache {
    pub view: PyramidCacheView,
    pub base_index: i32,
    pub num_caches: i32,
    pub pyr_ind: i32,
    pub tiles: Vec<CacheSlice>,
    pub vm_pixels: f64,
    pub total_pixels: f64,
    pub use_count: f64,
    // f64 has no Ord; source counters are integral values, so their bit representation preserves order.
    pub use_map: BTreeMap<u64, i32>,
    pub free_tiles: Vec<i32>,
    pub requests: VecDeque<LoadTileRequest>,
    pub tile_caches: Vec<TileCache>,
    pub ordered_list: Vec<i32>,
    pub slice_bufs: [Vec<u8>; 3],
    pub buf_x_start: i32,
    pub buf_y_start: i32,
    pub buf_x_size: usize,
    pub buf_y_size: usize,
    pub buf_tot_size: usize,
    pub buf_section: i32,
    pub buf_load_z: [i32; 3],
    pub buf_cache_ind: i32,
    pub last_request_size: i32,
    pub zoom_up_limit: f32,
    pub zoom_down_limit: f32,
    pub full_sec_buf: Vec<u8>,
    pub full_sec_z: i32,
    pub location: (i32, i32, i32),
}

/// `PyramidCache()`: build the native cache state from descriptors already
/// opened and validated by the caller's image-loading boundary.
pub fn pyramid_cache(
    view: PyramidCacheView,
    tile_caches: Vec<TileCache>,
    vm_pixels: f64,
) -> Result<PyramidCache, i32> {
    PyramidCache::new(view, tile_caches, vm_pixels)
}

impl PyramidCache {
    /// `PyramidCache::PyramidCache`; image descriptors have already been checked/opened by `imodview`.
    pub fn new(
        view: PyramidCacheView,
        tile_caches: Vec<TileCache>,
        vm_pixels: f64,
    ) -> Result<Self, i32> {
        if tile_caches.is_empty() {
            return Err(2);
        }
        let mut ordered_list: Vec<i32> = (0..tile_caches.len() as i32).collect();
        ordered_list.sort_by_key(|&i| tile_caches[i as usize].xy_scale);
        let mut base_index = -1;
        let (mut mx, mut my, mut mz) = (0, 0, 0);
        for c in &tile_caches {
            mx = mx.max(c.full_x_size);
            my = my.max(c.full_y_size);
            mz = mz.max(c.full_z_size);
        }
        for (i, c) in tile_caches.iter().enumerate() {
            if c.full_x_size == mx && c.full_y_size == my && c.full_z_size == mz {
                base_index = i as i32;
            }
        }
        if base_index < 0 {
            return Err(10);
        }
        Ok(Self {
            view,
            base_index,
            num_caches: tile_caches.len() as i32,
            pyr_ind: -1,
            tiles: Vec::new(),
            vm_pixels,
            total_pixels: 0.,
            use_count: 0.,
            use_map: BTreeMap::new(),
            free_tiles: Vec::new(),
            requests: VecDeque::new(),
            tile_caches,
            ordered_list,
            slice_bufs: [Vec::new(), Vec::new(), Vec::new()],
            buf_x_start: 0,
            buf_y_start: 0,
            buf_x_size: 0,
            buf_y_size: 0,
            buf_tot_size: 0,
            buf_section: -1,
            buf_load_z: [-1; 3],
            buf_cache_ind: -1,
            last_request_size: 0,
            zoom_up_limit: 1.55,
            zoom_down_limit: 0.7,
            full_sec_buf: Vec::new(),
            full_sec_z: -1,
            location: (0, 0, 0),
        })
    }
    /// `pyramidError`: the caller owns source-compatible fatal UI/error policy.
    pub fn pyramid_error(&self, error: i32) -> Result<(), i32> {
        Err(error)
    }
    /// `setupTileCache`.  Image metadata is deliberately supplied here instead
    /// of reaching through the `ImodImageFile` C pointer.
    pub fn setup_tile_cache(
        &self,
        full_x: i32,
        full_y: i32,
        full_z: i32,
        tile_x: i32,
        tile_y: i32,
        overlap_x: i32,
        overlap_y: i32,
        pcoords: Vec<i32>,
    ) -> Result<TileCache, i32> {
        if full_x % self.view.xybin.max(1) != 0 || full_y % self.view.xybin.max(1) != 0 {
            return Err(16);
        }
        if tile_x != 0
            && (tile_x % self.view.xybin.max(1) != 0 || tile_y % self.view.xybin.max(1) != 0)
        {
            return Err(17);
        }
        if overlap_x % self.view.xybin.max(1) != 0 || overlap_y % self.view.xybin.max(1) != 0 {
            return Err(18);
        }
        let mut c = TileCache {
            full_x_size: full_x,
            full_y_size: full_y,
            full_z_size: full_z,
            x_tile_size: tile_x,
            y_tile_size: tile_y,
            x_overlap: overlap_x,
            y_overlap: overlap_y,
            xy_scale: 1,
            z_scale: 1,
            plist_size: (pcoords.len() / 3) as i32,
            pcoords,
            ..Default::default()
        };
        if tile_x > 0 {
            c.num_x_tiles = (full_x + tile_x - 1) / tile_x;
            c.num_y_tiles = (full_y + tile_y - 1) / tile_y;
        }
        Ok(c)
    }
    /// `adjustForBinning`.
    pub fn adjust_for_binning(&mut self) {
        let bin = self.view.xybin.max(1);
        for c in &mut self.tile_caches {
            c.full_x_size /= bin;
            c.full_y_size /= bin;
            c.x_tile_size /= bin;
            c.y_tile_size /= bin;
            c.x_overlap /= bin;
            c.y_overlap /= bin;
            c.x_offset /= bin as f32;
            c.y_offset /= bin as f32;
            for p in (0..c.pcoords.len()).step_by(3) {
                c.pcoords[p] /= bin;
                c.pcoords[p + 1] /= bin;
            }
        }
    }
    /// `setupStripOrTileCache`; constructor inputs already carry the parsed descriptor.
    pub fn setup_strip_or_tile_cache(&mut self) {
        self.base_index = 0;
        if let Some(c) = self.tile_caches.first_mut() {
            c.xy_scale = 1;
            c.z_scale = 1;
            c.x_offset = 0.;
            c.y_offset = 0.;
            c.z_offset = 0.;
        }
    }
    /// `initializeCaches`.
    pub fn initialize_caches(&mut self) {
        for c in &mut self.tile_caches {
            Self::set_load_limits(
                c.xy_scale,
                c.full_x_size,
                self.view.xmin,
                self.view.xmax,
                &mut c.x_offset,
                &mut c.min_x_load,
                &mut c.max_x_load,
            );
            Self::set_load_limits(
                c.xy_scale,
                c.full_y_size,
                self.view.ymin,
                self.view.ymax,
                &mut c.y_offset,
                &mut c.min_y_load,
                &mut c.max_y_load,
            );
            Self::set_load_limits(
                c.z_scale,
                c.full_z_size,
                self.view.zmin,
                self.view.zmax,
                &mut c.z_offset,
                &mut c.min_z_load,
                &mut c.max_z_load,
            );
            c.load_x_size = c.max_x_load + 1 - c.min_x_load;
            c.load_y_size = c.max_y_load + 1 - c.min_y_load;
            c.load_z_size = c.max_z_load + 1 - c.min_z_load;
            Self::set_tile_geometry(
                c.full_x_size,
                c.min_x_load,
                c.max_x_load,
                c.x_overlap,
                0,
                &mut c.x_tile_size,
                &mut c.num_x_tiles,
                &mut c.start_x_tile,
                &mut c.first_x_offset,
            );
            Self::set_tile_geometry(
                c.full_y_size,
                c.min_y_load,
                c.max_y_load,
                c.y_overlap,
                c.file_y_tile_offset,
                &mut c.y_tile_size,
                &mut c.num_y_tiles,
                &mut c.start_y_tile,
                &mut c.first_y_offset,
            );
            c.tile_x_delta = c.x_tile_size - c.x_overlap;
            c.tile_y_delta = c.y_tile_size - c.y_overlap;
            c.tile_index =
                vec![-1; (c.num_x_tiles * c.num_y_tiles * c.load_z_size).max(0) as usize];
        }
    }
    /// `freeForNeededPixels`.
    pub fn free_for_needed_pixels(
        &mut self,
        cache_ind: i32,
        zval: i32,
        other_z: i32,
        pixels: f64,
        loops: i32,
    ) -> i32 {
        for pass in 0..loops {
            while pixels + self.total_pixels > self.vm_pixels {
                let chosen = self.use_map.iter().find_map(|(&k, &i)| {
                    let t = &self.tiles[i as usize];
                    (pass != 0
                        || t.cache_ind != cache_ind
                        || (t.section != zval && t.section != other_z))
                        .then_some((k, i))
                });
                let Some((key, ind)) = chosen else { break };
                self.free_tile(ind, true);
                self.use_map.remove(&key);
            }
            if pixels + self.total_pixels <= self.vm_pixels {
                break;
            }
        }
        0
    }
    /// `makeNewCacheItem`.
    pub fn make_new_cache_item(
        &mut self,
        cache_ind: i32,
        xtile: i32,
        ytile: i32,
        zval: i32,
        other_z: i32,
        xs: i32,
        ys: i32,
    ) -> i32 {
        if self.free_for_needed_pixels(cache_ind, zval, other_z, (xs * ys) as f64, 2) != 0 {
            return -1;
        }
        let ind = self.free_tiles.pop().unwrap_or_else(|| {
            self.tiles.push(CacheSlice::default());
            self.tiles.len() as i32 - 1
        });
        let key = self.use_count.to_bits();
        self.tiles[ind as usize] = CacheSlice {
            cache_ind,
            section: zval,
            x_tile_ind: xtile,
            y_tile_ind: ytile,
            used_at_count: self.use_count,
            xsize: xs,
            ysize: ys,
            data: vec![0; (xs * ys * if self.view.ushort_store { 2 } else { 1 }).max(0) as usize],
        };
        let c = &mut self.tile_caches[cache_ind as usize];
        let index = (xtile + (ytile + zval * c.num_y_tiles) * c.num_x_tiles) as usize;
        if index < c.tile_index.len() {
            c.tile_index[index] = ind;
        }
        self.use_map.insert(key, ind);
        self.total_pixels += (xs * ys) as f64;
        self.use_count += 1.;
        ind
    }
    /// `freeTile`.
    pub fn free_tile(&mut self, ind: i32, have_user_iter: bool) {
        if ind < 0 || ind as usize >= self.tiles.len() {
            return;
        }
        let t = &self.tiles[ind as usize];
        self.total_pixels -= (t.xsize * t.ysize) as f64;
        let cache = t.cache_ind;
        if cache >= 0 {
            let c = &mut self.tile_caches[cache as usize];
            let at = (t.x_tile_ind + (t.y_tile_ind + t.section * c.num_y_tiles) * c.num_x_tiles)
                as usize;
            if at < c.tile_index.len() {
                c.tile_index[at] = -1;
            }
        }
        let key = t.used_at_count.to_bits();
        self.tiles[ind as usize].cache_ind = -1;
        self.tiles[ind as usize].data.clear();
        self.free_tiles.push(ind);
        if !have_user_iter {
            self.use_map.remove(&key);
        }
    }
    /// `loadRequestedTiles`.
    pub fn load_requested_tiles(
        &mut self,
        async_load: i32,
        boundary: &mut dyn PyramidCacheBoundary,
    ) -> i32 {
        let mut ret = 0;
        while let Some(mut r) = self.requests.pop_front() {
            let (mut llx, mut urx, mut lly, mut ury) =
                self.find_load_limits_for_cache(r.cache_ind, r.x_tile_ind, r.y_tile_ind, false);
            let z = r.section + self.tile_caches[r.cache_ind as usize].min_z_load;
            let ind = self.make_new_cache_item(
                r.cache_ind,
                r.x_tile_ind,
                r.y_tile_ind,
                r.section,
                -1,
                urx + 1 - llx,
                ury + 1 - lly,
            );
            if ind < 0 {
                return 1;
            }
            if boundary
                .read_binned_tile(
                    r.cache_ind,
                    z,
                    (llx, urx, lly, ury),
                    &mut self.tiles[ind as usize].data,
                )
                .is_err()
            {
                ret = 1;
            } else {
                r.ind_in_cache = ind;
                self.copy_tile_into_buffer(&r, false);
            }
            if async_load < 0 {
                return ret;
            }
            if async_load > 0 {
                boundary.process_events();
            }
        }
        ret
    }
    /// `adjustLoadLimitsForMont`.
    pub fn adjust_load_limits_for_mont(
        &self,
        cache_ind: i32,
        nx: i32,
        ny: i32,
        limits: &mut (i32, i32, i32, i32),
        z: &mut i32,
    ) -> i32 {
        let c = &self.tile_caches[cache_ind as usize];
        if c.plist_size == 0 {
            return 0;
        }
        for p in c.pcoords.chunks_exact(3) {
            if p[2] == *z
                && limits.0 >= p[0]
                && limits.1 < p[0] + nx / self.view.xybin.max(1)
                && limits.2 >= p[1]
                && limits.3 < p[1] + ny / self.view.xybin.max(1)
            {
                *z = (p.as_ptr() as usize - c.pcoords.as_ptr() as usize) as i32
                    / (3 * std::mem::size_of::<i32>() as i32);
                limits.0 -= p[0];
                limits.1 -= p[0];
                limits.2 -= p[1];
                limits.3 -= p[1];
                return 0;
            }
        }
        1
    }
    /// `copyOrQueueTile`.
    pub fn copy_or_queue_tile(
        &mut self,
        cache_ind: i32,
        xtile: i32,
        ytile: i32,
        zsec: i32,
        buf_ind: i32,
        bx: i32,
        bxs: i32,
        by: i32,
        bys: i32,
    ) -> bool {
        let c = &self.tile_caches[cache_ind as usize];
        let at = (xtile + (ytile + zsec * c.num_y_tiles) * c.num_x_tiles) as usize;
        let ind = c.tile_index.get(at).copied().unwrap_or(-1);
        let (xmin, xmax, ymin, ymax) =
            self.find_load_limits_for_cache(cache_ind, xtile, ytile, true);
        let ux0 = xmin.max(bx);
        let ux1 = xmax.min(bx + bxs - 1);
        let uy0 = ymin.max(by);
        let uy1 = ymax.min(by + bys - 1);
        let r = LoadTileRequest {
            cache_ind,
            section: zsec,
            x_tile_ind: xtile,
            y_tile_ind: ytile,
            ind_in_cache: ind,
            buf_ind,
            from_x_start: ux0 - xmin,
            to_x_start: ux0 - bx,
            num_x_copy: ux1 + 1 - ux0,
            from_y_start: uy0 - ymin,
            to_y_start: uy0 - by,
            num_y_copy: uy1 + 1 - uy0,
        };
        if ind < 0 {
            self.requests.push_back(r);
            true
        } else {
            self.copy_tile_into_buffer(&r, true);
            false
        }
    }
    /// `copyTileIntoBuffer`.
    pub fn copy_tile_into_buffer(&mut self, r: &LoadTileRequest, manage: bool) {
        if r.ind_in_cache < 0 || r.num_x_copy <= 0 || r.num_y_copy <= 0 {
            return;
        }
        let t = &self.tiles[r.ind_in_cache as usize];
        let pixel = if self.view.ushort_store { 2 } else { 1 };
        let (dst, stride) = if r.buf_ind as usize == FULL_SEC_BUF {
            (&mut self.full_sec_buf, self.view.xsize as usize)
        } else {
            (&mut self.slice_bufs[r.buf_ind as usize], self.buf_x_size)
        };
        for y in 0..r.num_y_copy as usize {
            let from = ((r.from_y_start as usize + y) * t.xsize as usize + r.from_x_start as usize)
                * pixel;
            let to = ((r.to_y_start as usize + y) * stride + r.to_x_start as usize) * pixel;
            let n = r.num_x_copy as usize * pixel;
            if from + n <= t.data.len() && to + n <= dst.len() {
                dst[to..to + n].copy_from_slice(&t.data[from..from + n]);
            }
        }
        if manage {
            let old = self.tiles[r.ind_in_cache as usize].used_at_count.to_bits();
            self.use_map.remove(&old);
            self.tiles[r.ind_in_cache as usize].used_at_count = self.use_count;
            self.use_map
                .insert(self.use_count.to_bits(), r.ind_in_cache);
            self.use_count += 1.;
        }
    }
    /// `loadBaseTileWithPoint`.
    pub fn load_base_tile_with_point(
        &mut self,
        x: i32,
        y: i32,
        z: i32,
        b: &mut dyn PyramidCacheBoundary,
    ) -> i32 {
        self.load_tiles_containing_area(self.base_index, x, y, 1, 1, z, b)
    }
    /// `loadTilesContainingArea`.
    pub fn load_tiles_containing_area(
        &mut self,
        cache_ind: i32,
        bx: i32,
        by: i32,
        bxs: i32,
        bys: i32,
        z: i32,
        boundary: &mut dyn PyramidCacheBoundary,
    ) -> i32 {
        let (xs, ys, nx, ny, _, _) = self.scaled_area_size(cache_ind, bx, by, bxs, bys, false);
        let (sx, sy, _, _) = self.get_tile_and_position_in_tile(cache_ind, xs, ys);
        let (ex, ey, _, _) =
            self.get_tile_and_position_in_tile(cache_ind, xs + nx - 1, ys + ny - 1);
        let mut begin = false;
        for y in sy..=ey {
            for x in sx..=ex {
                let c = &self.tile_caches[cache_ind as usize];
                let at = (x + (y + z * c.num_y_tiles) * c.num_x_tiles) as usize;
                if c.tile_index.get(at).copied().unwrap_or(-1) < 0 {
                    self.requests.push_back(LoadTileRequest {
                        cache_ind,
                        section: z,
                        x_tile_ind: x,
                        y_tile_ind: y,
                        ..Default::default()
                    });
                    begin = true;
                }
            }
        }
        if begin && self.load_requested_tiles(0, boundary) != 0 {
            -1
        } else {
            0
        }
    }
    /// `fillCacheForArea`; caller supplies the selected displayed limits after
    /// the Qt dialog-manager/XYZ/Slicer/Zap priority logic in the C++ source.
    pub fn fill_cache_for_area(
        &mut self,
        section: i32,
        base_x: i32,
        base_y: i32,
        nx: i32,
        ny: i32,
        zoom: f64,
        boundary: &mut dyn PyramidCacheBoundary,
    ) -> i32 {
        let (ci, _) =
            PyramidCache::pick_best_cache(self, zoom, self.zoom_up_limit, self.zoom_down_limit);
        let (_, _, _, _, sx, sy) = self.scaled_area_size(ci, base_x, base_y, nx, ny, false);
        let (_, _, scale, _) = self.scaled_range_in_z(ci, section, section);
        let c = &self.tile_caches[ci as usize];
        let z = ((section as f32 / scale as f32).round() as i32).clamp(0, c.load_z_size - 1);
        self.load_tiles_containing_area(ci, base_x, base_y, nx, ny, z, boundary)
    }
    /// `getSectionArea`; returns tightly packed display rows rather than C line-pointer allocation.
    pub fn get_section_area(
        &mut self,
        section: i32,
        fx: i32,
        fy: i32,
        nx: i32,
        ny: i32,
        zoom: f64,
        async_load: bool,
        boundary: &mut dyn PyramidCacheBoundary,
    ) -> Result<(Vec<u8>, i32, i32, f32, f32, i32, i32), i32> {
        let (which, scale) =
            PyramidCache::pick_best_cache(self, zoom, self.zoom_up_limit, self.zoom_down_limit);
        let (xs, ys, ox, oy, xo, yo) = self.scaled_area_size(which, fx, fy, nx, ny, true);
        self.buf_x_start = xs;
        self.buf_y_start = ys;
        self.buf_x_size = ox as usize;
        self.buf_y_size = oy as usize;
        self.buf_tot_size = self.buf_x_size * self.buf_y_size;
        self.buf_cache_ind = which;
        let pixel = if self.view.ushort_store { 2 } else { 1 };
        self.slice_bufs[0] =
            vec![((self.view.white + self.view.black) / 2) as u8; self.buf_tot_size * pixel];
        self.buf_load_z[0] = section;
        let (sx, sy, _, _) = self.get_tile_and_position_in_tile(which, xs, ys);
        let (ex, ey, _, _) = self.get_tile_and_position_in_tile(which, xs + ox - 1, ys + oy - 1);
        self.requests.clear();
        for y in sy..=ey {
            for x in sx..=ex {
                self.copy_or_queue_tile(which, x, y, section, 0, xs, ox, ys, oy);
            }
        }
        let status = self.requests.len() as i32;
        if !self.requests.is_empty() {
            self.load_requested_tiles(if async_load { -1 } else { 0 }, boundary);
            if async_load && !self.requests.is_empty() {
                boundary.start_tile_loading();
            }
        }
        self.buf_section = section;
        self.last_request_size = self.requests.len() as i32;
        Ok((self.slice_bufs[0].clone(), ox, oy, xo, yo, scale, status))
    }
    /// `getFullSection`.
    pub fn get_full_section(
        &mut self,
        z: i32,
        boundary: &mut dyn PyramidCacheBoundary,
    ) -> Option<&[u8]> {
        if self.full_sec_buf.is_empty() {
            self.full_sec_buf.resize(
                (self.view.xsize * self.view.ysize * if self.view.ushort_store { 2 } else { 1 })
                    as usize,
                0,
            );
            self.total_pixels += (self.view.xsize * self.view.ysize) as f64;
        }
        if z != self.full_sec_z {
            self.requests.clear();
            let (nxt, nyt) = {
                let c = &self.tile_caches[self.base_index as usize];
                (c.num_x_tiles, c.num_y_tiles)
            };
            for y in 0..nyt {
                for x in 0..nxt {
                    self.copy_or_queue_tile(
                        self.base_index,
                        x,
                        y,
                        z,
                        FULL_SEC_BUF as i32,
                        0,
                        self.view.xsize,
                        0,
                        self.view.ysize,
                    );
                }
            }
            self.load_requested_tiles(0, boundary);
            self.full_sec_z = z;
        }
        Some(&self.full_sec_buf)
    }
    /// `freeFullSection`.
    pub fn free_full_section(&mut self) {
        if !self.full_sec_buf.is_empty() {
            self.total_pixels -= (self.view.xsize * self.view.ysize) as f64;
        }
        self.full_sec_buf.clear();
        self.full_sec_z = -1;
    }
    /// `setupFastAccess`.
    pub fn setup_fast_access(
        &self,
        cache_ind: i32,
    ) -> (Vec<Option<&[u8]>>, Vec<i32>, i32, i32, i32, i32, i32) {
        let c = &self.tile_caches[cache_ind as usize];
        let mut data = Vec::with_capacity(c.tile_index.len());
        let mut widths = Vec::with_capacity(c.tile_index.len());
        let mut sum = 0;
        for &i in &c.tile_index {
            if i >= 0 {
                let t = &self.tiles[i as usize];
                data.push(Some(t.data.as_slice()));
                widths.push(t.xsize);
                sum += i;
            } else {
                data.push(None);
                widths.push(0);
            }
        }
        (
            data,
            widths,
            sum,
            c.tile_x_delta,
            c.tile_y_delta,
            c.first_x_offset - c.x_overlap / 2,
            c.first_y_offset - c.y_overlap / 2,
        )
    }
    /// `loadedCacheSum`.
    pub fn loaded_cache_sum(&self, cache_ind: i32) -> i32 {
        self.tile_caches[cache_ind as usize]
            .tile_index
            .iter()
            .filter(|&&x| x >= 0)
            .sum()
    }
    /// `fastPlaneAccess`.
    pub fn fast_plane_access<'a>(
        &'a self,
        cache_ind: i32,
        plane: i32,
        axis: i32,
        segments: &mut Vec<FastSegment<'a>>,
        starts: &mut Vec<i32>,
    ) -> i32 {
        if cache_ind < 0 || cache_ind >= self.num_caches {
            return 1;
        }
        let c = &self.tile_caches[cache_ind as usize];
        segments.clear();
        let nz = c.load_z_size;
        if axis == 0 {
            starts.resize((nz + 1) as usize, 0);
            let (xt, _, xi, _) = self.get_tile_and_position_in_tile(cache_ind, plane, 0);
            for z in 0..nz {
                starts[z as usize] = segments.len() as i32;
                for y in 0..c.num_y_tiles {
                    let ind = c.tile_index[(xt + (y + z * c.num_y_tiles) * c.num_x_tiles) as usize];
                    if ind >= 0 {
                        let (_, _, a, b) = self.find_load_limits_for_cache(cache_ind, xt, y, true);
                        let t = &self.tiles[ind as usize];
                        segments.push(FastSegment {
                            xor_y: a,
                            yor_z: z,
                            length: b - a + 1,
                            stride: t.xsize,
                            line: &t.data[xi as usize..],
                        });
                    }
                }
            }
            starts[nz as usize] = segments.len() as i32;
            0
        } else if axis == 1 {
            starts.resize((nz + 1) as usize, 0);
            let (_, yt, _, yi) = self.get_tile_and_position_in_tile(cache_ind, 0, plane);
            for z in 0..nz {
                starts[z as usize] = segments.len() as i32;
                for x in 0..c.num_x_tiles {
                    let ind = c.tile_index[(x + (yt + z * c.num_y_tiles) * c.num_x_tiles) as usize];
                    if ind >= 0 {
                        let (a, b, _, _) = self.find_load_limits_for_cache(cache_ind, x, yt, true);
                        let t = &self.tiles[ind as usize];
                        segments.push(FastSegment {
                            xor_y: a,
                            yor_z: z,
                            length: b - a + 1,
                            stride: 1,
                            line: &t.data[yi as usize * t.xsize as usize..],
                        });
                    }
                }
            }
            starts[nz as usize] = segments.len() as i32;
            0
        } else {
            1
        }
    }
    /// `loadedMeanSD` (the source's sampling path, with optional percentile seam omitted).
    pub fn loaded_mean_sd(
        &self,
        cache_ind: i32,
        section: i32,
        sample: f32,
        ix: i32,
        iy: i32,
        nx: i32,
        ny: i32,
        cache_sum: &mut i32,
        old_sum: i32,
    ) -> Result<(f32, f32), i32> {
        let ci = if cache_ind < 0 {
            self.base_index
        } else {
            cache_ind
        };
        let sum = self.loaded_cache_sum(ci);
        *cache_sum = sum;
        if sum == old_sum {
            return Ok((0., 0.));
        }
        let (xs, ys, xx, yy, _, _) = self.scaled_area_size(ci, ix, iy, nx, ny, false);
        let mut vals = Vec::new();
        for y in ys..ys + yy {
            for x in xs..xs + xx {
                let v = self.value_from_cache(ci, x, y, section);
                if let Some(v) = v {
                    vals.push(v as f64);
                }
            }
        }
        if vals.len() < 30 {
            return Err(1);
        }
        let stride = (1.0 / sample.max(0.0001)).round().max(1.) as usize;
        let vals: Vec<f64> = vals.into_iter().step_by(stride).collect();
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        let sd = (vals.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>()
            / (vals.len() - 1).max(1) as f64)
            .sqrt();
        Ok((mean as f32, sd as f32))
    }
    /// `getValueFromBaseCache`.
    pub fn get_value_from_base_cache(&self, x: i32, y: i32, z: i32) -> i32 {
        self.value_from_cache(self.base_index, x, y, z).unwrap_or(0)
    }
    /// `getBaseFileCoords`.
    pub fn get_base_file_coords(&self, x: i32, y: i32, z: i32) -> Result<(i32, i32, i32), i32> {
        let (xt, yt, xi, yi) = self.get_tile_and_position_in_tile(self.base_index, x, y);
        let (a, _, c, _) = self.find_load_limits_for_cache(self.base_index, xt, yt, false);
        Ok((a + xi, c + yi, z))
    }
    /// `pickBestCache`.
    pub fn pick_best_cache(&self, zoom: f64, up: f32, down: f32) -> (i32, i32) {
        let which = if zoom >= 1. {
            self.base_index
        } else {
            let mut i = 1usize;
            while i < self.ordered_list.len()
                && zoom < 1. / self.tile_caches[self.ordered_list[i] as usize].xy_scale as f64
            {
                i += 1;
            }
            if i >= self.ordered_list.len() {
                *self.ordered_list.last().unwrap()
            } else if zoom * self.tile_caches[self.ordered_list[i] as usize].xy_scale as f64
                > up as f64
                || zoom * self.tile_caches[self.ordered_list[i - 1] as usize].xy_scale as f64
                    > down as f64
            {
                self.ordered_list[i - 1]
            } else {
                self.ordered_list[i]
            }
        };
        (which, self.tile_caches[which as usize].xy_scale)
    }
    /// `getTileAndPositionInTile`.
    pub fn get_tile_and_position_in_tile(&self, ci: i32, x: i32, y: i32) -> (i32, i32, i32, i32) {
        let c = &self.tile_caches[ci as usize];
        let xt =
            ((x + c.first_x_offset - c.x_overlap / 2) / c.tile_x_delta).clamp(0, c.num_x_tiles - 1);
        let yt =
            ((y + c.first_y_offset - c.y_overlap / 2) / c.tile_y_delta).clamp(0, c.num_y_tiles - 1);
        let xi = if xt != 0 {
            x + c.first_x_offset - c.x_overlap / 2 - xt * c.tile_x_delta
        } else {
            x
        };
        let yi = if yt != 0 {
            y + c.first_y_offset - c.y_overlap / 2 - yt * c.tile_y_delta
        } else {
            y
        };
        (xt, yt, xi, yi)
    }
    /// `setLoadLimits`.
    pub fn set_load_limits(
        scale: i32,
        size: i32,
        full_min: i32,
        full_max: i32,
        offset: &mut f32,
        min: &mut i32,
        max: &mut i32,
    ) {
        *min = ((full_min as f32 / scale as f32 - *offset).round() as i32).max(0);
        *max = (((full_max as f32 + 1.) / scale as f32 - *offset).round() as i32 - 1).min(size - 1);
        *offset = scale as f32 * (*min as f32 + *offset) - full_min as f32;
    }
    /// `setTileGeometry`.
    pub fn set_tile_geometry(
        full: i32,
        min: i32,
        max: i32,
        overlap: i32,
        file: i32,
        size: &mut i32,
        num: &mut i32,
        start: &mut i32,
        first: &mut i32,
    ) {
        if *size == 0 {
            *first = 0;
            *num = 1;
            *size = full;
            *start = 0;
            return;
        }
        let delta = *size - overlap;
        let end;
        *start = ((min - overlap / 2 - file) / delta).clamp(0, *num - 1);
        end = ((max - overlap / 2 - file) / delta).clamp(0, *num - 1);
        *first = min + file - *start * delta;
        *num = end + 1 - *start;
    }
    /// `findLoadLimits` overload taking primitive geometry.
    pub fn find_load_limits(
        size: i32,
        overlap: i32,
        tile: i32,
        num: i32,
        start: i32,
        file: i32,
        full_min: i32,
        full_max: i32,
        loaded: bool,
    ) -> (i32, i32) {
        let tile = tile + start;
        let num = num + start;
        let (mut lo, mut hi) = if tile == 0 {
            (0, size - (overlap - overlap / 2))
        } else {
            let lo = tile * (size - overlap) + overlap / 2;
            (
                lo,
                if tile < num - 1 {
                    lo + size - overlap
                } else {
                    lo + size - overlap / 2
                },
            )
        };
        lo = (lo - file).max(full_min);
        hi = (hi - 1 - file).min(full_max);
        if loaded {
            (lo - full_min, hi - full_min)
        } else {
            (lo, hi)
        }
    }
    /// `findLoadLimits` cache overload.
    pub fn find_load_limits_for_cache(
        &self,
        ci: i32,
        xt: i32,
        yt: i32,
        loaded: bool,
    ) -> (i32, i32, i32, i32) {
        let c = &self.tile_caches[ci as usize];
        let (a, b) = Self::find_load_limits(
            c.x_tile_size,
            c.x_overlap,
            xt,
            c.num_x_tiles,
            c.start_x_tile,
            0,
            c.min_x_load,
            c.max_x_load,
            loaded,
        );
        let (d, e) = Self::find_load_limits(
            c.y_tile_size,
            c.y_overlap,
            yt,
            c.num_y_tiles,
            c.start_y_tile,
            c.file_y_tile_offset,
            c.min_y_load,
            c.max_y_load,
            loaded,
        );
        (a, b, d, e)
    }
    /// primitive `scaledAreaSize`.
    pub fn scaled_area_size_axis(
        scale: i32,
        full_start: i32,
        size_in: i32,
        full_size: i32,
        offset: f32,
        inside: bool,
    ) -> (i32, i32, f32) {
        if scale == 1 {
            return (full_start, size_in, 0.);
        }
        let (mut start, mut end) = if inside {
            (
                ((full_start as f32 - offset) / scale as f32).ceil() as i32,
                ((full_start + size_in) as f32 - offset)
                    .div_euclid(scale as f32)
                    .floor() as i32,
            )
        } else {
            (
                ((full_start as f32 - offset) / scale as f32).floor() as i32,
                ((full_start + size_in) as f32 - offset)
                    .div_euclid(scale as f32)
                    .ceil() as i32,
            )
        };
        start = start.max(0);
        end = end.clamp(start + 1, full_size);
        let mut out = end - start;
        while inside && out * scale > size_in && out > 1 {
            out -= 1;
        }
        (
            start,
            out,
            start as f32 * scale as f32 + offset - full_start as f32,
        )
    }
    /// cache `scaledAreaSize` overload.
    pub fn scaled_area_size(
        &self,
        ci: i32,
        bx: i32,
        by: i32,
        nx: i32,
        ny: i32,
        inside: bool,
    ) -> (i32, i32, i32, i32, f32, f32) {
        let c = &self.tile_caches[ci as usize];
        let (a, b, e) =
            Self::scaled_area_size_axis(c.xy_scale, bx, nx, c.full_x_size, c.x_offset, inside);
        let (d, f, g) =
            Self::scaled_area_size_axis(c.xy_scale, by, ny, c.full_y_size, c.y_offset, inside);
        (a, d, b, f, e, g)
    }
    /// `scaledRangeInZ`.
    pub fn scaled_range_in_z(&self, ci: i32, bz0: i32, bz1: i32) -> (i32, i32, i32, f32) {
        let c = &self.tile_caches[ci as usize];
        let get = |z: i32| {
            (((z as f32 + 0.5 + self.view.zmin as f32) / c.z_scale as f32
                - 0.5
                - c.z_offset
                - c.min_z_load as f32)
                .clamp(0., (c.load_z_size - 1) as f32))
            .round() as i32
        };
        let real = (bz0 as f32 + 0.5 + self.view.zmin as f32) / c.z_scale as f32
            - 0.5
            - c.z_offset
            - c.min_z_load as f32;
        (
            get(bz0),
            get(bz1),
            c.z_scale,
            bz0 as f32 - c.z_scale as f32 * real,
        )
    }
    /// `findInterpolatedZvals`.
    pub fn find_interpolated_zvals(
        &self,
        ci: i32,
        section: i32,
    ) -> (bool, [i32; 3], [i32; 3], f32) {
        let c = &self.tile_caches[ci as usize];
        if c.z_scale == 1 {
            return (false, [section, 0, 0], [0, 0, 0], 0.);
        }
        let mut real =
            (section as f32 + 0.5 + self.view.zmin as f32) / c.z_scale as f32 - 0.5 - c.z_offset;
        real = real.clamp(c.min_z_load as f32, c.max_z_load as f32) - c.min_z_load as f32;
        let z = real as i32;
        let f = real - z as f32;
        if f > 0.9 {
            return (false, [z + 1, z, z + 1], [0, z + 1, z], f);
        }
        if f < 0.1 {
            return (false, [z, z, z + 1], [0, z + 1, z], f);
        }
        (true, [section, z, z + 1], [0, z + 1, z], f)
    }
    /// `optimalBufferSize`.
    pub fn optimal_buffer_size(start: i32, size: i32, load: i32) -> (i32, usize) {
        let extra = (0.5 * size as f64 * (1000f64 / (size.max(1000) as f64)).sqrt()).round() as i32;
        let bs = (start - extra).max(0);
        let end = (start + size + extra - 1).min(load - 1);
        (bs, (end + 1 - bs) as usize)
    }
    /// `zoomRequiresBigLoad`.
    pub fn zoom_requires_big_load(&self, zoom: f64, wx: i32, wy: i32) -> bool {
        let (ci, scale) = self.pick_best_cache(zoom, self.zoom_up_limit, self.zoom_down_limit);
        let c = &self.tile_caches[ci as usize];
        let wz = zoom * scale as f64;
        let sx = ((wx as f64 / wz) as i32).min(c.load_x_size);
        let sy = ((wy as f64 / wz) as i32).min(c.load_y_size);
        let (bx, bsx) = Self::optimal_buffer_size((c.load_x_size - sx) / 2, sx, c.load_x_size);
        let (by, bsy) = Self::optimal_buffer_size((c.load_y_size - sy) / 2, sy, c.load_y_size);
        let (a, b, _, _) = self.get_tile_and_position_in_tile(ci, bx, by);
        let (d, e, _, _) =
            self.get_tile_and_position_in_tile(ci, bx + bsx as i32 - 1, by + bsy as i32 - 1);
        (c.x_tile_size * c.y_tile_size * (d - a + 1) * (e - b + 1)) as f64 > 50. * 1024. * 1024.
    }
    /// `getCacheTileNumbers`.
    pub fn get_cache_tile_numbers(&self, ci: i32) -> Result<(i32, i32, i32), i32> {
        self.tile_caches
            .get(ci as usize)
            .map(|c| (c.num_x_tiles, c.num_y_tiles, c.load_z_size))
            .ok_or(1)
    }
    fn value_from_cache(&self, ci: i32, x: i32, y: i32, z: i32) -> Option<i32> {
        let (xt, yt, xi, yi) = self.get_tile_and_position_in_tile(ci, x, y);
        let c = &self.tile_caches[ci as usize];
        let ind = *c
            .tile_index
            .get((xt + (yt + z * c.num_y_tiles) * c.num_x_tiles) as usize)?;
        if ind < 0 {
            return None;
        }
        let t = &self.tiles[ind as usize];
        let at = (yi * t.xsize + xi) as usize;
        if self.view.ushort_store {
            let a = at * 2;
            Some(u16::from_ne_bytes([*t.data.get(a)?, *t.data.get(a + 1)?]) as i32)
        } else {
            Some(*t.data.get(at)? as i32)
        }
    }
}

/// The `PyramidCache` side of the `mv_image.cpp` image-source seam.  Section
/// acquisition still belongs to `getSectionArea`, because it can be async and
/// requires the source I/O boundary; cached individual pixels are available
/// immediately to Model View's direct access path.
impl MvImageSource for PyramidCache {
    fn dimensions(&self) -> (i32, i32, i32) {
        (self.view.xsize, self.view.ysize, self.view.zsize)
    }
    fn location(&self) -> (i32, i32, i32) {
        self.location
    }
    fn set_location(&mut self, x: i32, y: i32, z: i32) {
        self.location = (x, y, z);
    }
    fn z_section(&mut self, _z: i32) -> Option<&[u8]> {
        None
    }
    fn ushort_store(&self) -> bool {
        self.view.ushort_store
    }
    fn pixel(&mut self, x: i32, y: i32, z: i32) -> Option<ImagePixel> {
        let value = self.get_value_from_base_cache(x, y, z);
        if self.view.ushort_store {
            Some(ImagePixel::UShort(value as u16))
        } else {
            Some(ImagePixel::Byte(value as u8))
        }
    }
    fn pick_best_cache(&mut self, zoom: f64, up: f32, down: f32) -> Option<(i32, i32)> {
        Some(PyramidCache::pick_best_cache(self, zoom, up, down))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_constructor_facade_owns_validated_tile_descriptors() {
        let cache = pyramid_cache(
            PyramidCacheView::default(),
            vec![TileCache {
                full_x_size: 2,
                full_y_size: 2,
                full_z_size: 1,
                ..Default::default()
            }],
            100.,
        )
        .unwrap();
        assert_eq!(cache.base_index, 0);
    }
    struct Reader;
    impl PyramidCacheBoundary for Reader {
        fn read_binned_tile(
            &mut self,
            _: i32,
            z: i32,
            l: (i32, i32, i32, i32),
            o: &mut [u8],
        ) -> Result<(), ()> {
            for (i, v) in o.iter_mut().enumerate() {
                *v = (z + l.0 + l.2 + i as i32) as u8;
            }
            Ok(())
        }
    }
    fn cache() -> PyramidCache {
        let c = TileCache {
            x_tile_size: 4,
            y_tile_size: 4,
            num_x_tiles: 2,
            num_y_tiles: 2,
            xy_scale: 1,
            z_scale: 1,
            full_x_size: 8,
            full_y_size: 8,
            full_z_size: 2,
            ..Default::default()
        };
        let v = PyramidCacheView {
            xsize: 8,
            ysize: 8,
            zsize: 2,
            xmax: 7,
            ymax: 7,
            zmax: 1,
            white: 255,
            ..Default::default()
        };
        let mut p = PyramidCache::new(v, vec![c], 1000.).unwrap();
        p.initialize_caches();
        p
    }
    #[test]
    fn geometry_and_loading() {
        let mut p = cache();
        let mut r = Reader;
        assert_eq!(p.load_base_tile_with_point(6, 6, 0, &mut r), 0);
        assert_ne!(p.get_value_from_base_cache(6, 6, 0), 0);
        assert_eq!(p.get_cache_tile_numbers(0), Ok((2, 2, 2)));
    }
    #[test]
    fn scaled_and_choice() {
        let p = cache();
        assert_eq!(p.pick_best_cache(1., 1.5, 0.7), (0, 1));
        assert_eq!(
            PyramidCache::find_load_limits(4, 0, 1, 2, 0, 0, 0, 7, true),
            (4, 7)
        );
    }
}
