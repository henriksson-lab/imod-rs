//! Translation of `IMOD/3dmod/cachefill.cpp` and `cachefill.h`.
//!
//! Cache slots and image-file operations are owned by the viewer/image units.
//! They remain explicit through [`CacheFillBoundary`]; this module keeps the
//! original fill ordering, time balancing, eviction, and flipped-volume paths.
#![allow(dead_code, unused_variables)]

pub const IMOD_DRAW_IMAGE: i32 = 1;
pub const IMOD_DRAW_MOD: i32 = 1 << 2;

/// `VmCache` fields read by `cachefill.cpp`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CacheSlot {
    pub cz: i32,
    pub ct: i32,
    pub used: i32,
    pub data: Vec<u8>,
}

/// The cache/image subset of `ViewInfo` used by this source file.
#[derive(Clone, Debug, Default)]
pub struct CacheFillView {
    pub zsize: i32,
    pub xsize: i32,
    pub ysize: i32,
    pub num_times: i32,
    pub cur_time: i32,
    pub zmouse: f32,
    pub vm_size: i32,
    pub vm_tdim: i32,
    pub vm_tbase: i32,
    pub vm_count: i32,
    pub loading_image: bool,
    pub full_cache_flipped: bool,
    pub pyr_cache: bool,
    pub volume_stack: bool,
    pub load_axis: i32,
    pub raw_image_store: i32,
    pub image_lly: i32,
    pub image_ury: i32,
    pub image_axis: i32,
    pub cache_index: Vec<i32>,
    pub vm_cache: Vec<CacheSlot>,
    pub blank_sections: Vec<bool>,
}

impl CacheFillView {
    /// Cache-index expression repeated in the source.
    pub fn cache_index_at(&self, z: i32, ct: i32) -> Option<usize> {
        let index = z * self.vm_tdim + ct - self.vm_tbase;
        (index >= 0)
            .then_some(index as usize)
            .filter(|&i| i < self.cache_index.len())
    }

    /// `ivwPlistBlank` at the source image/piece-list seam.
    pub fn plist_blank(&self, z: i32) -> bool {
        self.blank_sections
            .get(z as usize)
            .copied()
            .unwrap_or(false)
    }
}

/// Direct `iview`, `PyramidCache`, Qt-current-directory, progress, and draw
/// calls made by the C++ code.  This is deliberately a source-service seam,
/// not a substitute cache or image implementation.
pub trait CacheFillBoundary {
    fn pyramid_fill_cache_for_area(&mut self, cz: i32, source: i32);
    fn close_time_image(&mut self, time: i32);
    fn reopen_time_image(&mut self, time: i32);
    fn select_time_image(&mut self, time: i32);
    fn set_ifd_current_dir(&mut self);
    fn restore_current_dir(&mut self);
    fn image_count(&mut self, message: &str);
    fn read_z(&mut self, cache_data: &mut [u8], z: i32);
    fn read_binned_section(&mut self, buffer: &mut [u8], adjusted_z: i32);
    fn draw(&mut self, flags: i32);
    fn rounded_style(&self) -> bool {
        false
    }
    fn dialog_change_event(&mut self) {}
    fn check_and_set_mac_menu(&mut self) {}
    fn remove_dialog(&mut self) {}
    fn accept_close_event(&mut self) {}
    fn close_dialog(&mut self) {}
    fn control_key(&mut self, _: bool) {}
}

/// File-static `imodCacheFillData`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ImodCacheFillData {
    pub dialog_open: bool,
    pub autofill: i32,
    pub fracfill: i32,
    pub balance: i32,
    pub overlap: i32,
}

/// `ImodCacheFill` (`cachefill.h`), with Qt widget pointers represented by
/// their source-visible selected/enabled values.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ImodCacheFill {
    pub top_window_open: bool,
    pub fill_group: i32,
    pub balance_group: i32,
    pub overlap_group: i32,
    /// `mOverlapRadio[3]`; these pointers are created but otherwise unused by
    /// the source after the group is connected.
    pub overlap_radio: [bool; 3],
    pub auto_checked: bool,
    pub overlap_box_enabled: bool,
    pub rounded_style: bool,
}

/// `walk_in_z`.
pub fn walk_in_z(view: &CacheFillView, mut cz: i32, mut steps: i32) -> i32 {
    let idir = if steps > 0 { 1 } else { -1 };
    while steps != 0 {
        cz += idir;
        if cz < 0 || cz >= view.zsize {
            break;
        }
        if !view.plist_blank(cz) {
            steps -= idir;
        }
    }
    cz
}

/// `set_z_limits`.
pub fn set_z_limits(
    view: &CacheFillView,
    zstart: &mut i32,
    zend: &mut i32,
    nfill: i32,
    cz: i32,
    ovbefore: i32,
    ovafter: i32,
) {
    let (nbefore, nafter) = if ovbefore > ovafter {
        let before = ovbefore * nfill / (ovbefore + ovafter);
        (before, nfill - before - 1)
    } else {
        let after = ovafter * nfill / (ovbefore + ovafter);
        (nfill - after - 1, after)
    };
    *zstart = walk_in_z(view, cz, -nbefore);
    if *zstart < 0 {
        *zstart = 0;
        *zend = walk_in_z(view, 0, nfill - 1);
        if *zend >= view.zsize {
            *zend = view.zsize - 1;
        }
    } else {
        *zend = walk_in_z(view, cz, nafter);
        if *zend >= view.zsize {
            *zend = view.zsize - 1;
            *zstart = walk_in_z(view, view.zsize - 1, -(nfill - 1));
            if *zstart < 0 {
                *zstart = 0;
            }
        }
    }
}

/// `fill_cache`.
pub fn fill_cache(
    view: &mut CacheFillView,
    data: &ImodCacheFillData,
    cz: i32,
    ovbefore: i32,
    ovafter: i32,
    source: i32,
    native: &mut dyn CacheFillBoundary,
) -> i32 {
    const FILL_TABLE: [i32; 3] = [1, 2, 4];
    if view.full_cache_flipped {
        return 0;
    }
    if view.pyr_cache {
        if view.loading_image {
            return 1;
        }
        view.loading_image = true;
        native.pyramid_fill_cache_for_area(cz, source);
        view.loading_image = false;
        native.draw(IMOD_DRAW_IMAGE | IMOD_DRAW_MOD);
        return 0;
    }
    let ntimes = if view.num_times != 0 {
        view.num_times
    } else {
        1
    };
    let curtime = if view.num_times != 0 {
        view.cur_time
    } else {
        1
    };
    let fraction = data.fracfill.clamp(0, (FILL_TABLE.len() - 1) as i32) as usize;
    let mut nfill = view.vm_size / FILL_TABLE[fraction];
    if nfill == 0 {
        nfill = 1;
    }
    if view.vm_cache.len() < view.vm_size.max(0) as usize {
        return 1;
    }
    view.loading_image = true;
    let count = (ntimes + 1) as usize;
    let mut zstart = vec![0; count];
    let mut zend = vec![0; count];
    let mut zstall = vec![0; count];
    let mut zndall = vec![0; count];
    if data.balance >= 2 || view.num_times == 0 {
        set_z_limits(
            view,
            &mut zstart[curtime as usize],
            &mut zend[curtime as usize],
            nfill,
            cz,
            ovbefore,
            ovafter,
        );
        if ntimes > 1 {
            let nleft = nfill - (zend[curtime as usize] + 1 - zstart[curtime as usize]);
            let nbase = nleft / (ntimes - 1);
            let nextra = nleft % (ntimes - 1);
            for time in 1..=view.num_times {
                if time == curtime {
                    continue;
                }
                let mut nshare = nbase + i32::from(time <= nextra);
                if data.balance == 3 {
                    nshare = 0;
                }
                set_z_limits(
                    view,
                    &mut zstart[time as usize],
                    &mut zend[time as usize],
                    nshare,
                    cz,
                    ovbefore,
                    ovafter,
                );
            }
        }
    } else if data.balance == 1 {
        let tstart = (curtime - 1).max(1);
        let tend = (curtime + 1).min(view.num_times);
        let nadj = tend + 1 - tstart;
        let nshare = nfill / nadj;
        let mut nleft = nfill;
        for time in tstart..=tend {
            set_z_limits(
                view,
                &mut zstart[time as usize],
                &mut zend[time as usize],
                nshare,
                cz,
                ovbefore,
                ovafter,
            );
            nleft -= zend[time as usize] + 1 - zstart[time as usize];
        }
        if ntimes > nadj {
            let nbase = nleft / (ntimes - nadj);
            let nextra = nleft % (ntimes - nadj);
            for time in 1..=view.num_times {
                if time >= tstart && time <= tend {
                    continue;
                }
                let share = nbase + i32::from(time <= nextra);
                set_z_limits(
                    view,
                    &mut zstart[time as usize],
                    &mut zend[time as usize],
                    share,
                    cz,
                    ovbefore,
                    ovafter,
                );
            }
        }
    } else {
        let nbase = nfill / ntimes;
        let nextra = nfill % ntimes;
        for time in 1..=view.num_times {
            let share = nbase + i32::from(time <= nextra);
            set_z_limits(
                view,
                &mut zstart[time as usize],
                &mut zend[time as usize],
                share,
                cz,
                ovbefore,
                ovafter,
            );
        }
    }
    for time in 1..=ntimes {
        let ct = if view.num_times != 0 { time } else { 0 };
        zstall[time as usize] = zstart[time as usize];
        zndall[time as usize] = zend[time as usize];
        let mut z = zstart[time as usize];
        while z <= zend[time as usize] {
            if let Some(index) = view.cache_index_at(z, ct) {
                let sl = view.cache_index[index];
                if sl >= 0 {
                    view.vm_cache[sl as usize].used = view.vm_count + 1;
                } else if view.load_axis == 2 {
                    break;
                }
            }
            z += 1;
        }
        if view.load_axis == 2 {
            zstart[time as usize] = z;
            let mut z = zend[time as usize];
            while z >= zstart[time as usize] {
                if let Some(index) = view.cache_index_at(z, ct) {
                    let sl = view.cache_index[index];
                    if sl >= 0 {
                        view.vm_cache[sl as usize].used = view.vm_count + 1;
                    } else {
                        break;
                    }
                }
                z -= 1;
            }
            zend[time as usize] = z;
        }
    }
    if view.num_times != 0 {
        if !view.volume_stack {
            native.close_time_image(view.cur_time);
        }
        native.set_ifd_current_dir();
    }
    for time in 1..=ntimes {
        if zstart[time as usize] > zend[time as usize] {
            continue;
        }
        let ct = if view.num_times != 0 {
            native.select_time_image(time);
            if !view.volume_stack {
                native.reopen_time_image(time);
            } else {
                view.cur_time = time;
            }
            time
        } else {
            0
        };
        if view.load_axis != 2 {
            for z in zstart[time as usize]..=zend[time as usize] {
                let Some(index) = view.cache_index_at(z, ct) else {
                    continue;
                };
                if view.cache_index[index] >= 0 || view.plist_blank(z) {
                    continue;
                }
                native.image_count(&if view.num_times != 0 {
                    format!("Reading image file # {time:03}, Z = {}\r", z + 1)
                } else {
                    format!("Reading image file, Z = {}\r", z + 1)
                });
                let mut minused = view.vm_count + 1;
                let mut slmin = 0usize;
                for (sl, cache) in view.vm_cache.iter().enumerate() {
                    if cache.used < minused {
                        minused = cache.used;
                        slmin = sl;
                    }
                }
                let old = &view.vm_cache[slmin];
                if old.cz >= 0 && old.ct >= view.vm_tbase {
                    if let Some(old_index) = view.cache_index_at(old.cz, ct) {
                        view.cache_index[old_index] = -1;
                    }
                }
                native.read_z(&mut view.vm_cache[slmin].data, z);
                view.vm_cache[slmin].cz = z;
                view.vm_cache[slmin].ct = ct;
                view.vm_cache[slmin].used = view.vm_count + 1;
                view.cache_index[index] = slmin as i32;
            }
        } else {
            let llysave = view.image_lly;
            let urysave = view.image_ury;
            view.image_lly += zstart[time as usize];
            view.image_ury = llysave + zend[time as usize];
            view.image_axis = 3;
            let pix_size = match view.raw_image_store {
                1 => 2,
                2 | 3 => 4,
                4 => 8,
                16 => 3,
                _ => 1,
            };
            let nslice = zend[time as usize] + 1 - zstart[time as usize];
            let mut buffer = vec![0u8; (view.xsize * nslice * pix_size).max(0) as usize];
            let mut loadtbl = Vec::with_capacity(nslice.max(0) as usize);
            for _ in 0..nslice {
                let mut minused = view.vm_count + 1;
                let mut slmin = 0usize;
                for (sl, cache) in view.vm_cache.iter().enumerate() {
                    if cache.used < minused {
                        minused = cache.used;
                        slmin = sl;
                    }
                }
                loadtbl.push(slmin);
                view.vm_cache[slmin].used = view.vm_count + 1;
                let old = &view.vm_cache[slmin];
                if old.cz >= 0 && old.ct >= view.vm_tbase {
                    if let Some(old_index) = view.cache_index_at(old.cz, ct) {
                        view.cache_index[old_index] = -1;
                    }
                }
            }
            for sect in 0..view.ysize {
                let z = sect + view.image_lly;
                native.image_count(&if view.num_times != 0 {
                    format!("Reading image # {time:03}, file Z = {z}")
                } else {
                    format!("Reading image, file Z = {z}")
                });
                native.read_binned_section(
                    &mut buffer,
                    if view.volume_stack {
                        z + (view.cur_time - 1) * view.zsize
                    } else {
                        z
                    },
                );
                let offset = (sect * view.xsize * pix_size).max(0) as usize;
                let bytes = (view.xsize * pix_size).max(0) as usize;
                for (i, &sl) in loadtbl.iter().enumerate() {
                    let from = i * bytes;
                    if offset + bytes <= view.vm_cache[sl].data.len()
                        && from + bytes <= buffer.len()
                    {
                        view.vm_cache[sl].data[offset..offset + bytes]
                            .copy_from_slice(&buffer[from..from + bytes]);
                    }
                }
            }
            for (i, &sl) in loadtbl.iter().enumerate() {
                let z = i as i32 + zstart[time as usize];
                view.vm_cache[sl].cz = z;
                view.vm_cache[sl].ct = ct;
                if let Some(index) = view.cache_index_at(z, ct) {
                    view.cache_index[index] = sl as i32;
                }
            }
            view.image_lly = llysave;
            view.image_ury = urysave;
            view.image_axis = 2;
        }
        if view.num_times != 0 && !view.volume_stack {
            native.close_time_image(time);
        }
    }
    native.image_count("\n");
    if view.num_times != 0 {
        if view.volume_stack {
            view.cur_time = curtime;
        }
        native.select_time_image(view.cur_time);
        if !view.volume_stack {
            native.reopen_time_image(view.cur_time);
        }
        native.restore_current_dir();
    }
    let maxdtime = (ntimes - curtime).max(curtime - 1);
    for dtime in (0..=maxdtime).rev() {
        let tdirlim = if dtime != 0 { -1 } else { 1 };
        let mut tdir = 1;
        while tdir >= tdirlim {
            let time = curtime + dtime * tdir;
            if time >= 1 && time <= ntimes {
                let ct = if view.num_times != 0 { time } else { 0 };
                let maxdz = (zndall[time as usize] - cz).max(cz - zstall[time as usize]);
                for dz in (0..=maxdz).rev() {
                    let zdirlim = if dz != 0 { -1 } else { 1 };
                    let mut zdir = 1;
                    while zdir >= zdirlim {
                        let z = cz + dz * zdir;
                        if z >= zstall[time as usize] && z <= zndall[time as usize] {
                            if let Some(index) = view.cache_index_at(z, ct) {
                                let sl = view.cache_index[index];
                                if sl >= 0 {
                                    view.vm_count += 1;
                                    view.vm_cache[sl as usize].used = view.vm_count;
                                }
                            }
                        }
                        zdir -= 2;
                    }
                }
            }
            tdir -= 2;
        }
    }
    clean_fill();
    view.loading_image = false;
    native.draw(IMOD_DRAW_IMAGE);
    0
}

/// `clean_fill`; `Vec` owns the C allocations in this translation.
pub fn clean_fill() {}

/// `icfGetAutofill`.
pub fn icf_get_autofill(data: &ImodCacheFillData) -> i32 {
    data.autofill
}

/// `imodCacheFill`.
pub fn imod_cache_fill(
    view: &mut CacheFillView,
    data: &ImodCacheFillData,
    source: i32,
    native: &mut dyn CacheFillBoundary,
) -> i32 {
    if !view.loading_image {
        fill_cache(view, data, view.zmouse as i32, 1, 1, source, native)
    } else {
        -1
    }
}

/// `icfDoAutofill`.
pub fn icf_do_autofill(
    view: &mut CacheFillView,
    data: &ImodCacheFillData,
    cz: i32,
    native: &mut dyn CacheFillBoundary,
) -> Option<Vec<u8>> {
    if view.loading_image {
        return None;
    }
    let ct = view.cur_time;
    let ifbefore = cz > 0
        && view
            .cache_index_at(cz - 1, ct)
            .is_some_and(|i| view.cache_index[i] >= 0);
    let ifafter = cz < view.zsize - 1
        && view
            .cache_index_at(cz + 1, ct)
            .is_some_and(|i| view.cache_index[i] >= 0);
    let table = [1, 3, 7];
    let overlap = table[data.overlap.clamp(0, 2) as usize];
    let ovbefore = if ifafter && !ifbefore { overlap } else { 1 };
    let ovafter = if !ifafter && ifbefore { overlap } else { 1 };
    if fill_cache(view, data, cz, ovbefore, ovafter, 0, native) != 0 {
        return None;
    }
    view.cache_index_at(cz, ct)
        .and_then(|i| view.cache_index[i].try_into().ok())
        .and_then(|sl: usize| view.vm_cache.get(sl))
        .map(|cache| cache.data.clone())
}

/// `imodCacheFillDialog`.
pub fn imod_cache_fill_dialog(data: &mut ImodCacheFillData) -> ImodCacheFill {
    data.dialog_open = true;
    ImodCacheFill {
        top_window_open: true,
        fill_group: data.fracfill,
        balance_group: data.balance,
        overlap_group: data.overlap,
        overlap_radio: [false; 3],
        auto_checked: data.autofill != 0,
        overlap_box_enabled: data.autofill != 0,
        rounded_style: false,
    }
}

impl ImodCacheFill {
    /// `ImodCacheFill::ImodCacheFill`.
    pub fn new(data: &ImodCacheFillData) -> Self {
        imod_cache_fill_dialog(&mut data.clone())
    }
    /// `ImodCacheFill::~ImodCacheFill`.
    pub fn destroy(&mut self) {}
    /// `ImodCacheFill::buttonPressed`.
    pub fn button_pressed(
        &mut self,
        which: i32,
        view: &mut CacheFillView,
        data: &ImodCacheFillData,
        native: &mut dyn CacheFillBoundary,
    ) {
        if which == 0 && !view.loading_image {
            let _ = fill_cache(view, data, view.zmouse as i32, 1, 1, 0, native);
        }
    }
    /// `ImodCacheFill::fractionSelected`.
    pub fn fraction_selected(&mut self, which: i32, data: &mut ImodCacheFillData) {
        data.fracfill = which;
        self.fill_group = which;
    }
    /// `ImodCacheFill::balanceSelected`.
    pub fn balance_selected(&mut self, which: i32, data: &mut ImodCacheFillData) {
        data.balance = which;
        self.balance_group = which;
    }
    /// `ImodCacheFill::overlapSelected`.
    pub fn overlap_selected(&mut self, which: i32, data: &mut ImodCacheFillData) {
        data.overlap = which;
        self.overlap_group = which;
    }
    /// `ImodCacheFill::autoToggled`.
    pub fn auto_toggled(&mut self, state: bool, data: &mut ImodCacheFillData) {
        data.autofill = i32::from(state);
        self.auto_checked = state;
        self.overlap_box_enabled = state;
    }
    /// `ImodCacheFill::topChangeEvent`.
    pub fn top_change_event(&mut self, native: &mut dyn CacheFillBoundary) {
        self.rounded_style = native.rounded_style();
        native.dialog_change_event();
        native.check_and_set_mac_menu();
    }
    /// `ImodCacheFill::topCloseEvent`.
    pub fn top_close_event(
        &mut self,
        data: &mut ImodCacheFillData,
        native: &mut dyn CacheFillBoundary,
    ) {
        native.remove_dialog();
        data.dialog_open = false;
        self.top_window_open = false;
        native.accept_close_event();
    }
    /// `ImodCacheFill::keyPressEvent`.
    pub fn key_press_event(
        &mut self,
        close_key: bool,
        _: &mut ImodCacheFillData,
        native: &mut dyn CacheFillBoundary,
    ) {
        if close_key {
            native.close_dialog();
        } else {
            native.control_key(false);
        }
    }
    /// `ImodCacheFill::keyReleaseEvent`.
    pub fn key_release_event(&mut self, native: &mut dyn CacheFillBoundary) {
        native.control_key(true);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Native {
        reads: Vec<i32>,
        draws: Vec<i32>,
        events: Vec<&'static str>,
    }
    impl CacheFillBoundary for Native {
        fn pyramid_fill_cache_for_area(&mut self, _: i32, _: i32) {}
        fn close_time_image(&mut self, _: i32) {}
        fn reopen_time_image(&mut self, _: i32) {}
        fn select_time_image(&mut self, _: i32) {}
        fn set_ifd_current_dir(&mut self) {}
        fn restore_current_dir(&mut self) {}
        fn image_count(&mut self, _: &str) {}
        fn read_z(&mut self, out: &mut [u8], z: i32) {
            out.fill(z as u8);
            self.reads.push(z);
        }
        fn read_binned_section(&mut self, _: &mut [u8], _: i32) {}
        fn draw(&mut self, flags: i32) {
            self.draws.push(flags);
        }
        fn rounded_style(&self) -> bool {
            true
        }
        fn dialog_change_event(&mut self) {
            self.events.push("change");
        }
        fn check_and_set_mac_menu(&mut self) {
            self.events.push("menu");
        }
        fn close_dialog(&mut self) {
            self.events.push("close");
        }
        fn control_key(&mut self, release: bool) {
            self.events.push(if release { "release" } else { "press" });
        }
    }
    fn view() -> CacheFillView {
        CacheFillView {
            zsize: 5,
            xsize: 2,
            ysize: 2,
            cur_time: 0,
            vm_size: 3,
            vm_tdim: 1,
            vm_tbase: 0,
            cache_index: vec![-1; 5],
            vm_cache: (0..3)
                .map(|_| CacheSlot {
                    cz: -1,
                    ct: -1,
                    used: 0,
                    data: vec![0; 4],
                })
                .collect(),
            blank_sections: vec![false; 5],
            ..Default::default()
        }
    }
    #[test]
    fn z_limits_follow_overlap_and_bounds() {
        let v = view();
        let (mut a, mut b) = (0, 0);
        set_z_limits(&v, &mut a, &mut b, 3, 0, 1, 1);
        assert_eq!((a, b), (0, 2));
        set_z_limits(&v, &mut a, &mut b, 3, 4, 1, 1);
        assert_eq!((a, b), (2, 4));
    }
    #[test]
    fn fill_evicts_and_prioritizes_source_slots() {
        let mut v = view();
        v.zmouse = 2.;
        let mut n = Native::default();
        assert_eq!(
            imod_cache_fill(
                &mut v,
                &ImodCacheFillData {
                    fracfill: 0,
                    balance: 2,
                    ..Default::default()
                },
                0,
                &mut n
            ),
            0
        );
        assert_eq!(n.reads, vec![1, 2, 3]);
        assert_eq!(n.draws, vec![IMOD_DRAW_IMAGE]);
        assert!(v.vm_cache.iter().all(|s| s.used > 0));
    }
    #[test]
    fn dialog_change_and_key_routes_match_source() {
        let mut dialog = ImodCacheFill::default();
        let mut data = ImodCacheFillData::default();
        let mut native = Native::default();
        dialog.top_change_event(&mut native);
        dialog.key_press_event(false, &mut data, &mut native);
        dialog.key_release_event(&mut native);
        dialog.key_press_event(true, &mut data, &mut native);
        assert!(dialog.rounded_style);
        assert_eq!(
            native.events,
            ["change", "menu", "press", "release", "close"]
        );
    }
}
