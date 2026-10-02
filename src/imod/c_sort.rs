//! Rust-only stand-ins for the two library sorts the translated sources call:
//! C `qsort` as glibc 2.35 performs it, and C++ `std::sort` as libstdc++ 11
//! instantiates it.  Runtime boundaries like `printf` or `gfortran_rt`, not
//! translated units; the single place either algorithm lives in this tree.
//!
//! Neither may be replaced by `slice::sort_by`/`sort_unstable_by`.  Whenever a
//! comparison calls two distinguishable elements equal (a struct compared on
//! one field, `-0.0` against `0.0`, an index array compared through its
//! values), or is not a total order at all (any float comparison once a NaN is
//! present), the permutation left behind is a property of the *algorithm*, and
//! the callers read it.  Rust's sorts land elsewhere and, since 1.81, may
//! **panic** on a comparator they detect as inconsistent -- which `ccderaser`
//! on an image with NaN pixels did through `robuststat`.

/// C library `qsort` as glibc 2.35 (the libc the reference build runs on)
/// performs it: `qsort_r` -> `msort_with_tmp`, a top-down merge sort that
/// splits `n` into `n / 2` and `n - n / 2` and, while merging, takes the left
/// element whenever `compar(left, right) <= 0`.  For elements over 32 bytes
/// glibc sorts an array of pointers with the same merge and then permutes, so
/// the resulting order is the same for every element size.  The quicksort
/// fallback is taken only when the scratch buffer cannot be allocated (more
/// than a quarter of physical memory), which no caller here approaches.
pub fn qsort<T: Copy>(b: &mut [T], compar: &mut dyn FnMut(&T, &T) -> i32) {
    if b.len() <= 1 {
        return;
    }
    // glibc's `p.t`: one scratch buffer for the whole sort.
    let mut t: Vec<T> = b.to_vec();
    msort_with_tmp(b, &mut t, compar);
}

/// glibc 2.35 `msort_with_tmp` (`stdlib/msort.c`), the recursion of [`qsort`].
fn msort_with_tmp<T: Copy>(b: &mut [T], t: &mut [T], compar: &mut dyn FnMut(&T, &T) -> i32) {
    let n = b.len();
    if n <= 1 {
        return;
    }
    let n1 = n / 2;
    {
        let (b1, b2) = b.split_at_mut(n1);
        msort_with_tmp(b1, t, compar);
        msort_with_tmp(b2, t, compar);
    }
    let (mut i1, mut i2, mut k) = (0usize, n1, 0usize);
    while i1 < n1 && i2 < n {
        if compar(&b[i1], &b[i2]) <= 0 {
            t[k] = b[i1];
            i1 += 1;
        } else {
            t[k] = b[i2];
            i2 += 1;
        }
        k += 1;
    }
    // `if (n1 > 0) memcpy (tmp, b1, n1 * s); memcpy (b, p->t, (n - n2) * s);`
    let rest = n1 - i1;
    t[k..k + rest].copy_from_slice(&b[i1..n1]);
    k += rest;
    b[..k].copy_from_slice(&t[..k]);
}

/// libstdc++ 11 `std::sort(first, last, comp)` (`bits/stl_algo.h`
/// `std::__sort`): introsort with a depth limit of `2 * __lg(n)`, a
/// median-of-three pivot moved to the front, a Hoare partition, a heap sort
/// once the depth limit is spent, and a final insertion sort over runs of
/// `_S_threshold` (16).  `less` is the strict-weak-order predicate the C++
/// passes (`operator<` for the default overload).  Not stable.
///
/// Two loops in the source are *unguarded* -- they rely on the comparator
/// being a strict weak order to stop before the ends of the range, and with a
/// NaN they need not.  Where libstdc++ would then read outside the array
/// (undefined behaviour, nothing to reproduce), these stop at the range end.
pub fn std_sort<T: Copy, F: FnMut(&T, &T) -> bool + ?Sized>(v: &mut [T], less: &mut F) {
    let n = v.len();
    if n == 0 {
        return;
    }
    // `std::__lg(__last - __first) * 2`
    let lg = (usize::BITS - 1 - n.leading_zeros()) as usize;
    introsort_loop(v, 0, n, lg * 2, less);
    final_insertion_sort(v, 0, n, less);
}

/// `std::__introsort_loop`.
fn introsort_loop<T: Copy, F: FnMut(&T, &T) -> bool + ?Sized>(
    v: &mut [T],
    first: usize,
    mut last: usize,
    mut depth_limit: usize,
    less: &mut F,
) {
    while last - first > 16 {
        if depth_limit == 0 {
            // `std::__partial_sort(__first, __last, __last, __comp)`: the
            // heap select's loop over `[middle, last)` is empty.
            make_heap(v, first, last, less);
            sort_heap(v, first, last, less);
            return;
        }
        depth_limit -= 1;
        let cut = unguarded_partition_pivot(v, first, last, less);
        introsort_loop(v, cut, last, depth_limit, less);
        last = cut;
    }
}

/// `std::__unguarded_partition_pivot`.
fn unguarded_partition_pivot<T: Copy, F: FnMut(&T, &T) -> bool + ?Sized>(
    v: &mut [T],
    first: usize,
    last: usize,
    less: &mut F,
) -> usize {
    let mid = first + (last - first) / 2;
    move_median_to_first(v, first, first + 1, mid, last - 1, less);
    unguarded_partition(v, first + 1, last, first, less)
}

/// `std::__move_median_to_first`.
fn move_median_to_first<T: Copy, F: FnMut(&T, &T) -> bool + ?Sized>(
    v: &mut [T],
    result: usize,
    a: usize,
    b: usize,
    c: usize,
    less: &mut F,
) {
    if less(&v[a], &v[b]) {
        if less(&v[b], &v[c]) {
            v.swap(result, b);
        } else if less(&v[a], &v[c]) {
            v.swap(result, c);
        } else {
            v.swap(result, a);
        }
    } else if less(&v[a], &v[c]) {
        v.swap(result, a);
    } else if less(&v[b], &v[c]) {
        v.swap(result, c);
    } else {
        v.swap(result, b);
    }
}

/// `std::__unguarded_partition`; the pivot is the element at `pivot`, compared
/// in place (the partition never moves it).
fn unguarded_partition<T: Copy, F: FnMut(&T, &T) -> bool + ?Sized>(
    v: &mut [T],
    mut first: usize,
    mut last: usize,
    pivot: usize,
    less: &mut F,
) -> usize {
    let end = v.len();
    loop {
        while first < end && less(&v[first], &v[pivot]) {
            first += 1;
        }
        last -= 1;
        while last > pivot && less(&v[pivot], &v[last]) {
            last -= 1;
        }
        if !(first < last) {
            return first;
        }
        v.swap(first, last);
        first += 1;
    }
}

/// `std::__final_insertion_sort`.
fn final_insertion_sort<T: Copy, F: FnMut(&T, &T) -> bool + ?Sized>(
    v: &mut [T],
    first: usize,
    last: usize,
    less: &mut F,
) {
    if last - first > 16 {
        insertion_sort(v, first, first + 16, less);
        // `std::__unguarded_insertion_sort(__first + 16, __last)`
        for i in first + 16..last {
            unguarded_linear_insert(v, i, less);
        }
    } else {
        insertion_sort(v, first, last, less);
    }
}

/// `std::__insertion_sort`.
fn insertion_sort<T: Copy, F: FnMut(&T, &T) -> bool + ?Sized>(
    v: &mut [T],
    first: usize,
    last: usize,
    less: &mut F,
) {
    if first == last {
        return;
    }
    for i in first + 1..last {
        if less(&v[i], &v[first]) {
            let val = v[i];
            v.copy_within(first..i, first + 1);
            v[first] = val;
        } else {
            unguarded_linear_insert(v, i, less);
        }
    }
}

/// `std::__unguarded_linear_insert`.
fn unguarded_linear_insert<T: Copy, F: FnMut(&T, &T) -> bool + ?Sized>(
    v: &mut [T],
    mut last: usize,
    less: &mut F,
) {
    let val = v[last];
    while last > 0 && less(&val, &v[last - 1]) {
        v[last] = v[last - 1];
        last -= 1;
    }
    v[last] = val;
}

/// `std::__make_heap` over `[first, last)`.
fn make_heap<T: Copy, F: FnMut(&T, &T) -> bool + ?Sized>(
    v: &mut [T],
    first: usize,
    last: usize,
    less: &mut F,
) {
    if last - first < 2 {
        return;
    }
    let len = last - first;
    let mut parent = (len - 2) / 2;
    loop {
        let value = v[first + parent];
        adjust_heap(v, first, parent, len, value, less);
        if parent == 0 {
            return;
        }
        parent -= 1;
    }
}

/// `std::__sort_heap` over `[first, last)`, with `std::__pop_heap` inlined as
/// the source's `__pop_heap(__first, __last, __last)`.
fn sort_heap<T: Copy, F: FnMut(&T, &T) -> bool + ?Sized>(
    v: &mut [T],
    first: usize,
    mut last: usize,
    less: &mut F,
) {
    while last - first > 1 {
        last -= 1;
        let value = v[last];
        v[last] = v[first];
        adjust_heap(v, first, 0, last - first, value, less);
    }
}

/// `std::__adjust_heap` followed by its `std::__push_heap`.
fn adjust_heap<T: Copy, F: FnMut(&T, &T) -> bool + ?Sized>(
    v: &mut [T],
    first: usize,
    mut hole_index: usize,
    len: usize,
    value: T,
    less: &mut F,
) {
    let top_index = hole_index;
    let mut second_child = hole_index;
    while second_child < (len - 1) / 2 {
        second_child = 2 * (second_child + 1);
        if less(&v[first + second_child], &v[first + (second_child - 1)]) {
            second_child -= 1;
        }
        v[first + hole_index] = v[first + second_child];
        hole_index = second_child;
    }
    if (len & 1) == 0 && second_child == (len - 2) / 2 {
        second_child = 2 * (second_child + 1);
        v[first + hole_index] = v[first + (second_child - 1)];
        hole_index = second_child - 1;
    }
    // `std::__push_heap(__first, __holeIndex, __topIndex, __value)`
    let mut parent = (hole_index.wrapping_sub(1)) / 2;
    while hole_index > top_index && less(&v[first + parent], &value) {
        v[first + hole_index] = v[first + parent];
        hole_index = parent;
        parent = (hole_index.wrapping_sub(1)) / 2;
    }
    v[first + hole_index] = value;
}

/// libstdc++ 11 `std::list<T>::sort(comp)` (`bits/list.tcc`), over the
/// list's elements in order: the bottom-up merge sort with a `__carry` list
/// and 64 `__tmp` bucket lists, where `x.merge(y)` keeps `x`'s element when
/// `comp(y, x)` is false (so the sort is stable).  `less` is the predicate
/// (`operator<` for the default overload).  With a strict weak order every
/// stable sort agrees; with one that is not (a NaN among doubles) the
/// result is this algorithm's, which is why it is reproduced rather than
/// substituted.
pub fn list_sort<T: Copy>(v: &mut Vec<T>, less: &mut dyn FnMut(&T, &T) -> bool) {
    // Do nothing if the list has length 0 or 1.
    if v.len() < 2 {
        return;
    }
    // `list::merge(__x)`: merge `x` into `this`.
    fn merge<T: Copy>(this: Vec<T>, x: Vec<T>, less: &mut dyn FnMut(&T, &T) -> bool) -> Vec<T> {
        let mut out = Vec::with_capacity(this.len() + x.len());
        let (mut i, mut j) = (0, 0);
        while i < this.len() && j < x.len() {
            if less(&x[j], &this[i]) {
                out.push(x[j]);
                j += 1;
            } else {
                out.push(this[i]);
                i += 1;
            }
        }
        out.extend_from_slice(&this[i..]);
        out.extend_from_slice(&x[j..]);
        out
    }
    let mut input: std::collections::VecDeque<T> = v.drain(..).collect();
    let mut tmp: Vec<Vec<T>> = Vec::new();
    // `__fill`: the number of bucket lists in use.
    let mut fill = 0usize;
    loop {
        // `__carry.splice(__carry.begin(), *this, begin())`
        let mut carry = vec![input.pop_front().unwrap()];
        let mut counter = 0usize;
        while counter != fill && !tmp[counter].is_empty() {
            // `__counter->merge(__carry); __carry.swap(*__counter);`
            let bucket = std::mem::take(&mut tmp[counter]);
            carry = merge(bucket, carry, less);
            counter += 1;
        }
        // `__carry.swap(*__counter)`
        if counter == tmp.len() {
            tmp.push(Vec::new());
        }
        std::mem::swap(&mut carry, &mut tmp[counter]);
        if counter == fill {
            fill += 1;
        }
        if input.is_empty() {
            break;
        }
    }
    // `for (__counter = __tmp + 1; __counter != __fill; ++__counter)
    //    __counter->merge(*(__counter - 1));`
    for counter in 1..fill {
        let prev = std::mem::take(&mut tmp[counter - 1]);
        let cur = std::mem::take(&mut tmp[counter]);
        tmp[counter] = merge(cur, prev, less);
    }
    // `swap(*(__fill - 1))`
    *v = std::mem::take(&mut tmp[fill - 1]);
}
