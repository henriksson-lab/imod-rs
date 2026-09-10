//! Translation of `IMOD/libwarp/hull-ch.c`.
#![allow(dead_code)]

use crate::imod::libwarp::hull::{
    BASIS_LIST, BASIS_SIZE, Basis, CDIM, Coord, GET_SITE, Neighbor, RDIM, SIMPLEX_LIST,
    SIMPLEX_SIZE, SITE_NUM, Simplex, Site, buildhull, free_basis_storage, free_fg_storage,
    free_simplex_storage, free_tree_storage, new_block_basis, new_block_simplex, visit_hull,
    visit_triang_gen,
};
use crate::imod::libwarp::pointops::{PDIM, print_point};

/// C globals defined by `hull-ch.c`.
pub static mut CHECK_OVERSHOOT_F: i16 = 0;
pub static mut CH_ROOT: *mut Simplex = core::ptr::null_mut();
pub static mut HULL_INFINITY: [Coord; 10] = [57.2, 0., 0., 0., 0., 0., 0., 0., 0., 0.];
/// C `mo` (`hull-ch.c:624`), shared with the source face-graph traversal.
pub static mut MO: [i16; 10_000] = [0; 10_000];
pub static mut MI: [i16; 10_000] = [0; 10_000];
pub static mut MINS: [Coord; 8] = [0.; 8];
pub static mut MAXS: [Coord; 8] = [0.; 8];
pub static mut HUGE: f64 = 0.;
pub static mut HUGE_CRIT: f64 = 0.;
pub static mut EXACT_BITS: i32 = 0;
pub static mut B_ERR_MIN: f32 = 0.;
pub static mut B_ERR_MIN_SQ: f32 = 0.;
pub static mut BASIS_VEC_SIZE: usize = 0;
static mut VD: i16 = 0;
static mut INFINITY_BASIS: *mut Basis = core::ptr::null_mut();
static mut TT_BASIS: Basis = Basis {
    next: core::ptr::null_mut(),
    ref_count: 1,
    lscale: -1,
    sqa: 0.,
    sqb: 0.,
    vecs: [0.],
};
static mut TT_BASIS_POINTER: *mut Basis = core::ptr::addr_of_mut!(TT_BASIS);

/// Original C-private `sP_neigh` state (`hull-ch.c:65`).
static mut S_P_NEIGH: Neighbor = Neighbor {
    vert: core::ptr::null_mut(),
    simp: core::ptr::null_mut(),
    basis: core::ptr::null_mut(),
};
/// Original C-private `sB` state (`hull-ch.c:66`).
static mut S_B: *mut Basis = core::ptr::null_mut();
static mut CHECK_PERPS_B: *mut Basis = core::ptr::null_mut();

/// Original `hullchCleanup` (`IMOD/libwarp/hull-ch.c:70`).
pub unsafe fn hull_ch_cleanup() {
    unsafe {
        libc::free(S_P_NEIGH.basis.cast());
        S_P_NEIGH.basis = core::ptr::null_mut();
        libc::free(S_B.cast());
        S_B = core::ptr::null_mut();
    }
}

/// Original `print_site` (`IMOD/libwarp/hull-ch.c:158`).
pub unsafe fn print_site(site: Site, file: *mut libc::FILE) {
    unsafe {
        print_point(file, PDIM, site);
        libc::fprintf(file, c"\n".as_ptr());
    }
}

/// Original `set_ch_root` (`IMOD/libwarp/hull-ch.c:742`).
pub unsafe fn set_ch_root(simplex: *mut Simplex) {
    unsafe {
        CH_ROOT = simplex;
    }
}

/// Original static `Vec_dot` (`IMOD/libwarp/hull-ch.c:79`).
unsafe fn vec_dot(left: Site, right: Site) -> Coord {
    unsafe {
        let mut sum = 0.;
        for index in 0..RDIM {
            sum += *left.add(index as usize) * *right.add(index as usize);
        }
        sum
    }
}

/// Original static `Vec_dot_pdim` (`IMOD/libwarp/hull-ch.c:86`).
unsafe fn vec_dot_pdim(left: Site, right: Site) -> Coord {
    unsafe {
        let mut sum = 0.;
        for index in 0..PDIM {
            sum += *left.add(index as usize) * *right.add(index as usize);
        }
        sum
    }
}

/// Original static `Norm2` (`IMOD/libwarp/hull-ch.c:94`).
unsafe fn norm2(vector: Site) -> Coord {
    unsafe {
        let mut sum = 0.;
        for index in 0..RDIM {
            sum += *vector.add(index as usize) * *vector.add(index as usize);
        }
        sum
    }
}

/// Original static `Ax_plus_y` (`IMOD/libwarp/hull-ch.c:101`).
unsafe fn ax_plus_y(scale: Coord, mut source: Site, mut target: Site) {
    unsafe {
        for _ in 0..RDIM {
            *target += scale * *source;
            source = source.add(1);
            target = target.add(1);
        }
    }
}

/// Original static `Ax_plus_y_test` (`IMOD/libwarp/hull-ch.c:108`).
unsafe fn ax_plus_y_test(scale: Coord, source: Site, target: Site) {
    // `check_overshoot` only emits source debug diagnostics; its arithmetic is
    // exactly the `Ax_plus_y` operation.
    unsafe { ax_plus_y(scale, source, target) }
}

/// Original static `Vec_scale` (`IMOD/libwarp/hull-ch.c:116`).
unsafe fn vec_scale(count: i32, scale: Coord, vector: *mut Coord) {
    unsafe {
        for index in 0..count {
            *vector.add(index as usize) *= scale;
        }
    }
}

/// Original static `Vec_scale_test` (`IMOD/libwarp/hull-ch.c:123`).
unsafe fn vec_scale_test(count: i32, scale: Coord, vector: *mut Coord) {
    // As above, the source test variant differs solely by diagnostics.
    unsafe { vec_scale(count, scale, vector) }
}

/// Original static `lower_terms` (`IMOD/libwarp/hull-ch.c:227`).
unsafe fn lower_terms(basis: *mut Basis) -> f64 {
    unsafe {
        let vector = (*basis).vecs.as_mut_ptr();
        let factors = [2., 3., 5., 7., 11., 13.];
        let mut output = 1.;
        for factor in factors {
            loop {
                let mut index = 0;
                while index < 2 * RDIM
                    && factor * (*vector.add(index as usize) / factor).floor()
                        == *vector.add(index as usize)
                {
                    index += 1;
                }
                if index != 2 * RDIM {
                    break;
                }
                output *= factor;
                for divisor_index in 0..2 * RDIM {
                    *vector.add(divisor_index as usize) /= factor;
                }
            }
        }
        output
    }
}

/// Original static `lower_terms_point` (`IMOD/libwarp/hull-ch.c:250`).
unsafe fn lower_terms_point(vector: Site) -> f64 {
    unsafe {
        let factors = [2., 3., 5., 7., 11., 13.];
        let mut output = 1.;
        for factor in factors {
            loop {
                let mut index = 0;
                while index < 2 * RDIM
                    && factor * (*vector.add(index as usize) / factor).floor()
                        == *vector.add(index as usize)
                {
                    index += 1;
                }
                if index != 2 * RDIM {
                    break;
                }
                output *= factor;
                for divisor_index in 0..2 * RDIM {
                    *vector.add(divisor_index as usize) /= factor;
                }
            }
        }
        output
    }
}

/// Original static `sc` (`IMOD/libwarp/hull-ch.c:175`).
unsafe fn sc(basis: *mut Basis, simplex: *mut Simplex, k: i32, j: i32) -> f64 {
    unsafe {
        static mut LSCALE: i32 = 0;
        static mut MAX_SCALE: f64 = 0.;
        static mut LDETBOUND: f64 = 0.;
        static mut SB: f64 = 0.;
        if j < 10 {
            let labound = (*basis).sqa.log2() / 2.;
            MAX_SCALE = EXACT_BITS as f64 - labound - 0.66 * (k - 2) as f64 - 1.;
            if MAX_SCALE < 1. {
                MAX_SCALE = 1.;
            }
            if j == 0 {
                LDETBOUND = 0.;
                SB = 0.;
                for index in (1..k).rev() {
                    let neighbour_basis = (*simplex)
                        .neigh
                        .as_mut_ptr()
                        .add(index as usize)
                        .read()
                        .basis;
                    SB += (*neighbour_basis).sqb;
                    LDETBOUND += (*neighbour_basis).sqb.log2() / 2. + 1.;
                    LDETBOUND -= (*neighbour_basis).lscale as f64;
                }
            }
        }
        if (*basis).sqb <= 0.
            || LDETBOUND - (*basis).lscale as f64 + (*basis).sqb.log2() / 2. + 1. < 0.
        {
            return 0.;
        }
        LSCALE = (2. * SB / ((*basis).sqb + (*basis).sqa * B_ERR_MIN as f64))
            .log2()
            .floor() as i32
            / 2;
        if LSCALE as f64 > MAX_SCALE {
            LSCALE = MAX_SCALE.floor() as i32;
        } else if LSCALE < 0 {
            LSCALE = 0;
        }
        (*basis).lscale += LSCALE;
        if LSCALE < 20 {
            (1_i32 << LSCALE) as f64
        } else {
            2_f64.powi(LSCALE)
        }
    }
}

/// Original static `reduce_inner` (`IMOD/libwarp/hull-ch.c:269`).
unsafe fn reduce_inner(basis: *mut Basis, simplex: *mut Simplex, k: i32) -> i32 {
    unsafe {
        let vector_a = (*basis).vecs.as_mut_ptr().add(RDIM as usize);
        let vector_b = (*basis).vecs.as_mut_ptr();
        (*basis).sqa = norm2(vector_b);
        (*basis).sqb = (*basis).sqa;
        if k <= 1 {
            libc::memcpy(vector_b.cast(), vector_a.cast(), BASIS_VEC_SIZE);
            return 1;
        }
        for iteration in 0..250 {
            libc::memcpy(vector_b.cast(), vector_a.cast(), BASIS_VEC_SIZE);
            for index in (1..k).rev() {
                let neighbour_basis = (*simplex)
                    .neigh
                    .as_mut_ptr()
                    .add(index as usize)
                    .read()
                    .basis;
                let coefficient = -vec_dot((*neighbour_basis).vecs.as_mut_ptr(), vector_b)
                    / (*neighbour_basis).sqb;
                ax_plus_y(
                    coefficient,
                    (*neighbour_basis).vecs.as_mut_ptr().add(RDIM as usize),
                    vector_b,
                );
            }
            (*basis).sqb = norm2(vector_b);
            (*basis).sqa = norm2(vector_a);
            if 2. * (*basis).sqb >= (*basis).sqa {
                return 1;
            }
            vec_scale_test(RDIM, sc(basis, simplex, k, iteration), vector_a);
            for index in (1..k).rev() {
                let neighbour_basis = (*simplex)
                    .neigh
                    .as_mut_ptr()
                    .add(index as usize)
                    .read()
                    .basis;
                let coefficient = (-vec_dot((*neighbour_basis).vecs.as_mut_ptr(), vector_a)
                    / (*neighbour_basis).sqb
                    + 0.5)
                    .floor();
                ax_plus_y_test(
                    coefficient,
                    (*neighbour_basis).vecs.as_mut_ptr().add(RDIM as usize),
                    vector_a,
                );
            }
        }
        0
    }
}

/// Original static `reduce` (`IMOD/libwarp/hull-ch.c:335`).
unsafe fn reduce(basis: *mut *mut Basis, point: Site, simplex: *mut Simplex, k: i32) -> i32 {
    unsafe {
        if (*basis).is_null() {
            if BASIS_LIST.is_null() {
                new_block_basis(1);
            }
            *basis = BASIS_LIST;
            BASIS_LIST = (*BASIS_LIST).next;
            (**basis).ref_count = 1;
        } else {
            (**basis).lscale = 0;
        }
        let vector = (**basis).vecs.as_mut_ptr();
        let base = (*simplex).neigh.as_mut_ptr().read().vert;
        if VD != 0 && point == core::ptr::addr_of_mut!(HULL_INFINITY).cast::<Coord>() {
            libc::memcpy((*basis).cast(), INFINITY_BASIS.cast(), BASIS_SIZE);
        } else {
            for index in 0..PDIM {
                let difference = *point.add(index as usize) - *base.add(index as usize);
                *vector.add(index as usize + RDIM as usize) = difference;
                *vector.add(index as usize) = difference;
            }
            if VD != 0 {
                let lifted = vec_dot_pdim(vector, vector);
                *vector.add((2 * RDIM - 1) as usize) = lifted;
                *vector.add((RDIM - 1) as usize) = lifted;
            }
        }
        reduce_inner(*basis, simplex, k)
    }
}

/// Original `get_basis_sede` (`IMOD/libwarp/hull-ch.c:350`).
pub unsafe fn get_basis_sede(simplex: *mut Simplex) {
    unsafe {
        let mut k = 1;
        let neighbour_zero = (*simplex).neigh.as_mut_ptr();
        let mut neighbour = neighbour_zero.add(1);
        if VD != 0
            && (*neighbour_zero).vert == core::ptr::addr_of_mut!(HULL_INFINITY).cast::<Coord>()
            && CDIM > 1
        {
            core::ptr::swap(neighbour_zero, neighbour);
            if !(*neighbour_zero).basis.is_null() {
                (*(*neighbour_zero).basis).ref_count -= 1;
                if (*(*neighbour_zero).basis).ref_count == 0 {
                    libc::memset((*neighbour_zero).basis.cast(), 0, BASIS_SIZE);
                    (*(*neighbour_zero).basis).next = BASIS_LIST;
                    BASIS_LIST = (*neighbour_zero).basis;
                }
                (*neighbour_zero).basis = core::ptr::null_mut();
            }
            (*neighbour_zero).basis = TT_BASIS_POINTER;
            (*TT_BASIS_POINTER).ref_count += 1;
        } else if (*neighbour_zero).basis.is_null() {
            (*neighbour_zero).basis = TT_BASIS_POINTER;
            (*TT_BASIS_POINTER).ref_count += 1;
        } else {
            while k < CDIM && !(*neighbour).basis.is_null() {
                k += 1;
                neighbour = neighbour.add(1);
            }
        }
        while k < CDIM {
            if !(*neighbour).basis.is_null() {
                (*(*neighbour).basis).ref_count -= 1;
                if (*(*neighbour).basis).ref_count == 0 {
                    libc::memset((*neighbour).basis.cast(), 0, BASIS_SIZE);
                    (*(*neighbour).basis).next = BASIS_LIST;
                    BASIS_LIST = (*neighbour).basis;
                }
                (*neighbour).basis = core::ptr::null_mut();
            }
            reduce(
                core::ptr::addr_of_mut!((*neighbour).basis),
                (*neighbour).vert,
                simplex,
                k,
            );
            k += 1;
            neighbour = neighbour.add(1);
        }
    }
}

/// Original `out_of_flat` (`IMOD/libwarp/hull-ch.c:375`).
pub unsafe fn out_of_flat(root: *mut Simplex, point: Site) -> i32 {
    unsafe {
        if S_P_NEIGH.basis.is_null() {
            S_P_NEIGH.basis = libc::malloc(BASIS_SIZE).cast::<Basis>();
        }
        S_P_NEIGH.vert = point;
        CDIM += 1;
        let root_neighbour = (*root).neigh.as_mut_ptr().add((CDIM - 1) as usize);
        (*root_neighbour).vert = (*root).peak.vert;
        if !(*root_neighbour).basis.is_null() {
            (*(*root_neighbour).basis).ref_count -= 1;
            if (*(*root_neighbour).basis).ref_count == 0 {
                libc::memset((*root_neighbour).basis.cast(), 0, BASIS_SIZE);
                (*(*root_neighbour).basis).next = BASIS_LIST;
                BASIS_LIST = (*root_neighbour).basis;
            }
            (*root_neighbour).basis = core::ptr::null_mut();
        }
        get_basis_sede(root);
        if VD != 0
            && (*root).neigh.as_mut_ptr().read().vert
                == core::ptr::addr_of_mut!(HULL_INFINITY).cast::<Coord>()
        {
            return 1;
        }
        reduce(core::ptr::addr_of_mut!(S_P_NEIGH.basis), point, root, CDIM);
        if (*S_P_NEIGH.basis).sqa != 0. {
            return 1;
        }
        CDIM -= 1;
        0
    }
}

/// Original static `cosangle_sq` (`IMOD/libwarp/hull-ch.c:393`).
unsafe fn cosangle_sq(left: *mut Basis, right: *mut Basis) -> f64 {
    unsafe {
        let dot = vec_dot((*left).vecs.as_mut_ptr(), (*right).vecs.as_mut_ptr());
        dot * dot / norm2((*left).vecs.as_mut_ptr()) / norm2((*right).vecs.as_mut_ptr())
    }
}

/// Original `check_perps` (`IMOD/libwarp/hull-ch.c:402`).
pub unsafe fn check_perps(simplex: *mut Simplex) -> i32 {
    unsafe {
        for index in 1..CDIM {
            if (*(*simplex)
                .neigh
                .as_mut_ptr()
                .add(index as usize)
                .read()
                .basis)
                .sqb
                == 0.
            {
                return 0;
            }
        }
        if CHECK_PERPS_B.is_null() {
            CHECK_PERPS_B = libc::malloc(BASIS_SIZE).cast::<Basis>();
        } else {
            (*CHECK_PERPS_B).lscale = 0;
        }
        let vector = (*CHECK_PERPS_B).vecs.as_mut_ptr();
        let base = (*simplex).neigh.as_mut_ptr().read().vert;
        for index in 1..CDIM {
            let vertex = (*simplex)
                .neigh
                .as_mut_ptr()
                .add(index as usize)
                .read()
                .vert;
            if VD != 0 && vertex == core::ptr::addr_of_mut!(HULL_INFINITY).cast::<Coord>() {
                libc::memcpy(CHECK_PERPS_B.cast(), INFINITY_BASIS.cast(), BASIS_SIZE);
            } else {
                for coordinate in 0..PDIM {
                    let difference =
                        *vertex.add(coordinate as usize) - *base.add(coordinate as usize);
                    *vector.add(coordinate as usize + RDIM as usize) = difference;
                    *vector.add(coordinate as usize) = difference;
                }
                if VD != 0 {
                    let lifted = vec_dot_pdim(vector, vector);
                    *vector.add((2 * RDIM - 1) as usize) = lifted;
                    *vector.add((RDIM - 1) as usize) = lifted;
                }
            }
            if !(*simplex).normal.is_null()
                && cosangle_sq(CHECK_PERPS_B, (*simplex).normal) > B_ERR_MIN_SQ as f64
            {
                return 0;
            }
            for other in index + 1..CDIM {
                let other_basis = (*simplex)
                    .neigh
                    .as_mut_ptr()
                    .add(other as usize)
                    .read()
                    .basis;
                if cosangle_sq(CHECK_PERPS_B, other_basis) > B_ERR_MIN_SQ as f64 {
                    return 0;
                }
            }
        }
        1
    }
}

/// Original `get_normal_sede` (`IMOD/libwarp/hull-ch.c:440`).
pub unsafe fn get_normal_sede(simplex: *mut Simplex) {
    unsafe {
        get_basis_sede(simplex);
        if RDIM == 3 && CDIM == 3 {
            if BASIS_LIST.is_null() {
                new_block_basis(1);
            }
            (*simplex).normal = BASIS_LIST;
            BASIS_LIST = (*BASIS_LIST).next;
            (*(*simplex).normal).ref_count = 1;
            let left = (*(*simplex).neigh.as_mut_ptr().add(1).read().basis)
                .vecs
                .as_mut_ptr();
            let right = (*(*simplex).neigh.as_mut_ptr().add(2).read().basis)
                .vecs
                .as_mut_ptr();
            let normal = (*(*simplex).normal).vecs.as_mut_ptr();
            *normal = *left.add(1) * *right.add(2) - *left.add(2) * *right.add(1);
            *normal.add(1) = *left.add(2) * *right - *left * *right.add(2);
            *normal.add(2) = *left * *right.add(1) - *left.add(1) * *right;
            (*(*simplex).normal).sqb = norm2(normal);
            let mut remaining = CDIM + 1;
            let mut root_neighbour = (*CH_ROOT).neigh.as_mut_ptr().add((CDIM - 1) as usize);
            while remaining != 0 {
                let mut index = 0;
                while index < CDIM
                    && (*root_neighbour).vert
                        == (*simplex)
                            .neigh
                            .as_mut_ptr()
                            .add(index as usize)
                            .read()
                            .vert
                {
                    index += 1;
                }
                if index == CDIM {
                    if (*root_neighbour).vert
                        == core::ptr::addr_of_mut!(HULL_INFINITY).cast::<Coord>()
                    {
                        if *normal.add(2) > -(B_ERR_MIN as f64) {
                            remaining -= 1;
                            root_neighbour = root_neighbour.sub(1);
                            continue;
                        }
                    } else if sees((*root_neighbour).vert, simplex) == 0 {
                        remaining -= 1;
                        root_neighbour = root_neighbour.sub(1);
                        continue;
                    }
                    *normal = -*normal;
                    *normal.add(1) = -*normal.add(1);
                    *normal.add(2) = -*normal.add(2);
                    break;
                }
                remaining -= 1;
                root_neighbour = root_neighbour.sub(1);
            }
            return;
        }
        let mut remaining = CDIM + 1;
        let mut root_neighbour = (*CH_ROOT).neigh.as_mut_ptr().add((CDIM - 1) as usize);
        while remaining != 0 {
            let mut index = 0;
            while index < CDIM
                && (*root_neighbour).vert
                    == (*simplex)
                        .neigh
                        .as_mut_ptr()
                        .add(index as usize)
                        .read()
                        .vert
            {
                index += 1;
            }
            if index == CDIM {
                reduce(
                    core::ptr::addr_of_mut!((*simplex).normal),
                    (*root_neighbour).vert,
                    simplex,
                    CDIM,
                );
                if (*(*simplex).normal).sqb != 0. {
                    break;
                }
            }
            remaining -= 1;
            root_neighbour = root_neighbour.sub(1);
        }
    }
}

/// Original `get_normal` (`IMOD/libwarp/hull-ch.c:481`).
pub unsafe fn get_normal(simplex: *mut Simplex) {
    unsafe { get_normal_sede(simplex) }
}

/// Original `sees` (`IMOD/libwarp/hull-ch.c:483`).
pub unsafe fn sees(point: Site, simplex: *mut Simplex) -> i32 {
    unsafe {
        if S_B.is_null() {
            S_B = libc::malloc(BASIS_SIZE).cast::<Basis>();
        } else {
            (*S_B).lscale = 0;
        }
        let vector = (*S_B).vecs.as_mut_ptr();
        if CDIM == 0 {
            return 0;
        }
        if (*simplex).normal.is_null() {
            get_normal_sede(simplex);
            for index in 0..CDIM {
                let neighbour = (*simplex).neigh.as_mut_ptr().add(index as usize);
                if !(*neighbour).basis.is_null() {
                    (*(*neighbour).basis).ref_count -= 1;
                    if (*(*neighbour).basis).ref_count == 0 {
                        libc::memset((*neighbour).basis.cast(), 0, BASIS_SIZE);
                        (*(*neighbour).basis).next = BASIS_LIST;
                        BASIS_LIST = (*neighbour).basis;
                    }
                    (*neighbour).basis = core::ptr::null_mut();
                }
            }
        }
        let base = (*simplex).neigh.as_mut_ptr().read().vert;
        if VD != 0 && point == core::ptr::addr_of_mut!(HULL_INFINITY).cast::<Coord>() {
            libc::memcpy(S_B.cast(), INFINITY_BASIS.cast(), BASIS_SIZE);
        } else {
            for index in 0..PDIM {
                let difference = *point.add(index as usize) - *base.add(index as usize);
                *vector.add(index as usize + RDIM as usize) = difference;
                *vector.add(index as usize) = difference;
            }
            if VD != 0 {
                let lifted = vec_dot_pdim(vector, vector);
                *vector.add((2 * RDIM - 1) as usize) = lifted;
                *vector.add((RDIM - 1) as usize) = lifted;
            }
        }
        for _ in 0..3 {
            let dot = vec_dot(vector, (*(*simplex).normal).vecs.as_mut_ptr());
            if dot == 0. {
                return 0;
            }
            let normalized_dot = dot * dot / (*(*simplex).normal).sqb / norm2(vector);
            if normalized_dot > B_ERR_MIN_SQ as f64 {
                return (dot < 0.) as i32;
            }
            get_basis_sede(simplex);
            reduce_inner(S_B, simplex, CDIM);
        }
        0
    }
}

/// Original static `radsq` (`IMOD/libwarp/hull-ch.c:527`).
unsafe fn radsq(simplex: *mut Simplex) -> f64 {
    unsafe {
        for index in 0..CDIM {
            if (*simplex)
                .neigh
                .as_mut_ptr()
                .add(index as usize)
                .read()
                .vert
                == core::ptr::addr_of_mut!(HULL_INFINITY).cast::<Coord>()
            {
                return HUGE;
            }
        }
        if (*simplex).normal.is_null() {
            get_normal_sede(simplex);
        }
        let normal = (*(*simplex).normal).vecs.as_mut_ptr();
        if *normal.add((RDIM - 1) as usize) < f32::EPSILON as f64
            && *normal.add((RDIM - 1) as usize) > -(f32::EPSILON as f64)
        {
            return HUGE;
        }
        vec_dot_pdim(normal, normal)
            / 4.
            / *normal.add((RDIM - 1) as usize)
            / *normal.add((RDIM - 1) as usize)
    }
}

/// Original static `zero_marks` (`hull-ch.c:548`).
unsafe fn zero_marks(s: *mut Simplex, _: *mut core::ffi::c_void) -> *mut core::ffi::c_void {
    unsafe {
        (*s).mark = 0;
        core::ptr::null_mut()
    }
}
/// Original static `one_marks` (`hull-ch.c:550`).
unsafe fn one_marks(s: *mut Simplex, _: *mut core::ffi::c_void) -> *mut core::ffi::c_void {
    unsafe {
        (*s).mark = 1;
        core::ptr::null_mut()
    }
}
/// Original static `conv_facetv` (`hull-ch.c:608`).
unsafe fn conv_facetv(s: *mut Simplex, _: *mut core::ffi::c_void) -> *mut core::ffi::c_void {
    unsafe {
        for i in 0..CDIM {
            if (*s).neigh.as_mut_ptr().add(i as usize).read().vert
                == core::ptr::addr_of_mut!(HULL_INFINITY).cast()
            {
                return s.cast();
            }
        }
        core::ptr::null_mut()
    }
}
/// Original static `mark_points` (`hull-ch.c:616`).
unsafe fn mark_points(s: *mut Simplex, _: *mut core::ffi::c_void) -> *mut core::ffi::c_void {
    unsafe {
        for i in 0..CDIM {
            let v = (*s).neigh.as_mut_ptr().add(i as usize).read().vert;
            if v != core::ptr::addr_of_mut!(HULL_INFINITY).cast() {
                let n = SITE_NUM.unwrap()(v) as usize;
                if (*s).mark != 0 { MO[n] = 1 } else { MI[n] = 1 }
            }
        }
        core::ptr::null_mut()
    }
}

/// Original `alph_test` (`hull-ch.c:560`).
pub unsafe fn alph_test(
    simplex: *mut Simplex,
    index: i32,
    alpha_pointer: *mut core::ffi::c_void,
) -> i32 {
    unsafe {
        static mut ALPHA: f64 = 0.;
        if !alpha_pointer.is_null() {
            ALPHA = *(alpha_pointer.cast::<f64>());
            if simplex.is_null() {
                return 1;
            }
        }
        if index == -1 {
            return 0;
        }
        let adjacent = (*simplex)
            .neigh
            .as_mut_ptr()
            .add(index as usize)
            .read()
            .simp;
        for k in 0..CDIM {
            if (*simplex).neigh.as_mut_ptr().add(k as usize).read().vert
                == core::ptr::addr_of_mut!(HULL_INFINITY).cast()
                && k != index
            {
                return 1;
            }
        }
        let rs = radsq(simplex);
        let rsi = radsq(adjacent);
        if rs < ALPHA && rsi < ALPHA {
            return 1;
        }
        let last = (*simplex).neigh.as_mut_ptr().add((CDIM - 1) as usize);
        let chosen = (*simplex).neigh.as_mut_ptr().add(index as usize);
        core::ptr::swap(&mut (*last).vert, &mut (*chosen).vert);
        CDIM -= 1;
        get_basis_sede(simplex);
        reduce(
            core::ptr::addr_of_mut!((*simplex).normal),
            core::ptr::addr_of_mut!(HULL_INFINITY).cast(),
            simplex,
            CDIM,
        );
        let merged = radsq(simplex);
        let mut k = 0;
        while k < CDIM && (*adjacent).neigh.as_mut_ptr().add(k as usize).read().simp != simplex {
            k += 1;
        }
        let sees_self = sees((*last).vert, simplex);
        let sees_other = if sees_self == 0 {
            sees(
                (*adjacent).neigh.as_mut_ptr().add(k as usize).read().vert,
                simplex,
            )
        } else {
            0
        };
        core::ptr::swap(&mut (*last).vert, &mut (*chosen).vert);
        CDIM += 1;
        (*simplex).normal = core::ptr::null_mut();
        (*chosen).basis = core::ptr::null_mut();
        if sees_self != 0 {
            (ALPHA < rs) as i32
        } else if sees_other != 0 {
            (ALPHA < rsi) as i32
        } else {
            (ALPHA <= merged) as i32
        }
    }
}

/// Original `visit_outside_ashape` (`hull-ch.c:629`).
pub unsafe fn visit_outside_ashape(
    root: *mut Simplex,
    visitor: unsafe fn(*mut Simplex, *mut core::ffi::c_void) -> *mut core::ffi::c_void,
) -> *mut core::ffi::c_void {
    unsafe { visit_triang_gen(visit_hull(root, conv_facetv).cast(), visitor, alph_test) }
}
/// Original static `check_ashape` (`hull-ch.c:633`).
unsafe fn check_ashape(root: *mut Simplex, alpha: f64) -> i32 {
    unsafe {
        for i in 0..10_000 {
            MI[i] = 0;
            MO[i] = 0;
        }
        visit_hull(root, zero_marks);
        alph_test(
            core::ptr::null_mut(),
            0,
            core::ptr::addr_of!(alpha).cast_mut().cast(),
        );
        visit_outside_ashape(root, one_marks);
        visit_hull(root, mark_points);
        for i in 0..10_000 {
            if MO[i] != 0 && MI[i] == 0 {
                return 0;
            }
        }
        1
    }
}
/// Original `find_alpha` (`hull-ch.c:651`).
pub unsafe fn find_alpha(root: *mut Simplex) -> f64 {
    unsafe {
        let mut low = 0_f32;
        let mut high = 0_f32;
        for i in 0..PDIM {
            let d = MAXS[i as usize] - MINS[i as usize];
            high += (d * d) as f32;
        }
        check_ashape(root, high as f64);
        for _ in 0..17 {
            let mid = (low + high) / 2.;
            if check_ashape(root, mid as f64) != 0 {
                high = mid
            } else {
                low = mid
            }
            if (high - low) / high < 0.5 {
                break;
            }
        }
        1.1 * high as f64
    }
}

/// Original `build_convex_hull` (`hull-ch.c:745`).
pub unsafe fn build_convex_hull(
    get: crate::imod::libwarp::hull::GetSite,
    number: crate::imod::libwarp::hull::SiteNum,
    dimension: i16,
    delaunay: i16,
) -> *mut Simplex {
    unsafe {
        CDIM = 0;
        GET_SITE = Some(get);
        SITE_NUM = Some(number);
        PDIM = dimension as i32;
        VD = delaunay;
        let rdim = if VD != 0 { PDIM + 1 } else { PDIM };
        if rdim > 8 {
            return core::ptr::null_mut();
        }
        RDIM = rdim;
        EXACT_BITS = (f64::MANTISSA_DIGITS as f64 * (f64::RADIX as f64).log2()).floor() as i32;
        B_ERR_MIN = f64::EPSILON as f32 * 8. * 256. * 8. * 3.01;
        B_ERR_MIN_SQ = B_ERR_MIN * B_ERR_MIN;
        BASIS_VEC_SIZE = core::mem::size_of::<Coord>() * RDIM as usize;
        BASIS_SIZE =
            core::mem::size_of::<Basis>() + (2 * RDIM as usize - 1) * core::mem::size_of::<Coord>();
        SIMPLEX_SIZE = core::mem::size_of::<Simplex>()
            + (RDIM as usize - 1) * core::mem::size_of::<Neighbor>();
        let point = if VD != 0 {
            if BASIS_LIST.is_null() {
                new_block_basis(1);
            }
            INFINITY_BASIS = BASIS_LIST;
            BASIS_LIST = (*BASIS_LIST).next;
            (*INFINITY_BASIS).ref_count = 1;
            *(*INFINITY_BASIS)
                .vecs
                .as_mut_ptr()
                .add((2 * RDIM - 1) as usize) = 1.;
            *(*INFINITY_BASIS).vecs.as_mut_ptr().add((RDIM - 1) as usize) = 1.;
            (*INFINITY_BASIS).sqa = 1.;
            (*INFINITY_BASIS).sqb = 1.;
            core::ptr::addr_of_mut!(HULL_INFINITY).cast()
        } else {
            get()
        };
        if point.is_null() {
            return core::ptr::null_mut();
        }
        if SIMPLEX_LIST.is_null() {
            new_block_simplex(1);
        }
        let root = SIMPLEX_LIST;
        SIMPLEX_LIST = (*SIMPLEX_LIST).next;
        if SIMPLEX_LIST.is_null() {
            new_block_simplex(1);
        }
        let peak = SIMPLEX_LIST;
        SIMPLEX_LIST = (*SIMPLEX_LIST).next;
        libc::memcpy(peak.cast(), root.cast(), SIMPLEX_SIZE);
        (*root).peak.vert = point;
        (*root).peak.simp = peak;
        (*peak).peak.simp = root;
        CH_ROOT = root;
        buildhull(root);
        root
    }
}

/// Original `free_hull_storage` (`hull-ch.c:817`).
pub unsafe fn free_hull_storage() {
    unsafe {
        free_basis_storage();
        free_simplex_storage();
        free_tree_storage();
        free_fg_storage();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_factor_lowering_keeps_factor_order_and_mutates_both_halves() {
        unsafe {
            let previous_rdim = RDIM;
            RDIM = 1;
            let basis = libc::calloc(
                1,
                core::mem::size_of::<Basis>() + core::mem::size_of::<Coord>(),
            )
            .cast::<Basis>();
            *(*basis).vecs.as_mut_ptr() = 12.;
            *(*basis).vecs.as_mut_ptr().add(1) = 18.;
            assert_eq!(lower_terms(basis), 6.);
            assert_eq!(*(*basis).vecs.as_mut_ptr(), 2.);
            assert_eq!(*(*basis).vecs.as_mut_ptr().add(1), 3.);
            let mut vector = [20., 50.];
            assert_eq!(lower_terms_point(vector.as_mut_ptr()), 10.);
            assert_eq!(vector, [2., 5.]);
            libc::free(basis.cast());
            RDIM = previous_rdim;
        }
    }
}
