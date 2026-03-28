//! Avganova - Average with ANOVA weighting
//!
//! Performs statistical comparisons using nested analysis of variance on
//! the output of IMAVGSTAT. Compares groups of data sets across collections
//! of summing regions, with optional rescaling of individual data sets.
//!
//! Translated from IMOD avganova.f

use clap::Parser;
use std::fs;
use std::io::{self, BufRead};
use std::path::PathBuf;
use std::process;

// ---------------------------------------------------------------------------
// Statistics helpers
// ---------------------------------------------------------------------------

fn betai(a: f32, b: f32, x: f32) -> f32 {
    if x <= 0.0 { return 0.0; }
    if x >= 1.0 { return 1.0; }
    let ln_beta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    let front = (x.ln() * a + (1.0 - x).ln() * b - ln_beta).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        front * betacf(a, b, x) / a
    } else {
        1.0 - front * betacf(b, a, 1.0 - x) / b
    }
}

fn betacf(a: f32, b: f32, x: f32) -> f32 {
    let fpmin = 1.0e-30_f32;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0_f32;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < fpmin { d = fpmin; }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=200 {
        let mf = m as f32;
        let aa = mf * (b - mf) * x / ((qam + 2.0 * mf) * (a + 2.0 * mf));
        d = 1.0 + aa * d; if d.abs() < fpmin { d = fpmin; }
        c = 1.0 + aa / c; if c.abs() < fpmin { c = fpmin; }
        d = 1.0 / d; h *= d * c;
        let aa = -(a + mf) * (qab + mf) * x / ((a + 2.0 * mf) * (qap + 2.0 * mf));
        d = 1.0 + aa * d; if d.abs() < fpmin { d = fpmin; }
        c = 1.0 + aa / c; if c.abs() < fpmin { c = fpmin; }
        d = 1.0 / d;
        let del = d * c; h *= del;
        if (del - 1.0).abs() < 1.0e-7 { return h; }
    }
    h
}

fn ln_gamma(x: f32) -> f32 {
    let coeffs: [f64; 6] = [
        76.18009172947146, -86.50532032941677, 24.01409824083091,
        -1.231739572450155, 0.001208650973866179, -5.395239384953e-6,
    ];
    let xx = x as f64;
    let mut y = xx;
    let tmp = xx + 5.5;
    let tmp = tmp - (xx + 0.5) * tmp.ln();
    let mut ser = 1.000000000190015_f64;
    for c in &coeffs { y += 1.0; ser += c / y; }
    (-tmp + (2.5066282746310005_f64 * ser / xx).ln()) as f32
}

/// t-distribution probability.
fn t_prob(ndf: i32, t: f32) -> f32 {
    let x = ndf as f32 / (ndf as f32 + t * t);
    1.0 - betai(0.5 * ndf as f32, 0.5, x) / 2.0
}

/// Compute mean and standard deviation.
fn avg_sd(vals: &[f32]) -> (f32, f32, f32) {
    let n = vals.len() as f32;
    if n < 1.0 { return (0.0, 0.0, 0.0); }
    let mean: f32 = vals.iter().sum::<f32>() / n;
    let var: f32 = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / (n - 1.0).max(1.0);
    let sd = var.sqrt();
    let sem = sd / n.sqrt();
    (mean, sd, sem)
}

/// t-statistic for two-sample comparison of means.
fn t_stat(x1: f32, s1: f32, n1: i32, x2: f32, s2: f32, n2: i32) -> (f32, i32) {
    let nf1 = (n1 - 1) as f32;
    let nf2 = (n2 - 1) as f32;
    let nt = n1 + n2 - 2;
    let sp = ((nf1 * s1 * s1 + nf2 * s2 * s2) / nt as f32).sqrt();
    let t = (x1 - x2) / (sp * (1.0 / n1 as f32 + 1.0 / n2 as f32).sqrt());
    (t, nt)
}

/// Simple least-squares fit: y = a*x + b.
fn ls_fit(x: &[f32], y: &[f32]) -> (f32, f32, f32) {
    let n = x.len() as f32;
    let sx: f32 = x.iter().sum();
    let sy: f32 = y.iter().sum();
    let sxy: f32 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
    let sxx: f32 = x.iter().map(|a| a * a).sum();
    let syy: f32 = y.iter().map(|a| a * a).sum();
    let denom = n * sxx - sx * sx;
    if denom.abs() < 1.0e-20 {
        return (1.0, 0.0, 0.0);
    }
    let slope = (n * sxy - sx * sy) / denom;
    let intercept = (sy - slope * sx) / n;
    let r = if (n * sxx - sx * sx).abs() > 0.0 && (n * syy - sy * sy).abs() > 0.0 {
        (n * sxy - sx * sy) / ((n * sxx - sx * sx) * (n * syy - sy * sy)).sqrt()
    } else {
        0.0
    };
    (slope, intercept, r)
}

// ---------------------------------------------------------------------------
// Nested ANOVA
// ---------------------------------------------------------------------------

fn nested_anova(
    ngroups: usize,
    nb: &[usize],
    nn: &[Vec<i32>],
    xb: &[Vec<f32>],
    sds: &[Vec<f32>],
) {
    // Degrees of freedom
    let df_group = (ngroups - 1) as i32;
    let mut df_subgr = 0i32;
    let mut df_within = 0i32;
    for g in 0..ngroups {
        df_subgr += nb[g] as i32 - 1;
        for j in 0..nb[g] {
            df_within += nn[g][j] - 1;
        }
    }

    // Compute sums of squares
    let mut grand_sum = 0.0f32;
    let mut grand_n = 0i32;
    let mut group_means = vec![0.0f32; ngroups];
    let mut group_n = vec![0i32; ngroups];
    for g in 0..ngroups {
        for j in 0..nb[g] {
            group_means[g] += nn[g][j] as f32 * xb[g][j];
            group_n[g] += nn[g][j];
        }
        if group_n[g] > 0 { group_means[g] /= group_n[g] as f32; }
        grand_sum += group_means[g] * group_n[g] as f32;
        grand_n += group_n[g];
    }
    let grand_mean = if grand_n > 0 { grand_sum / grand_n as f32 } else { 0.0 };

    let mut ss_group = 0.0f32;
    for g in 0..ngroups {
        ss_group += group_n[g] as f32 * (group_means[g] - grand_mean).powi(2);
    }
    let mut ss_within = 0.0f32;
    for g in 0..ngroups {
        for j in 0..nb[g] {
            ss_within += (nn[g][j] - 1) as f32 * sds[g][j].powi(2);
        }
    }
    let mut ss_subgr = 0.0f32;
    for g in 0..ngroups {
        for j in 0..nb[g] {
            ss_subgr += nn[g][j] as f32 * (xb[g][j] - group_means[g]).powi(2);
        }
    }

    let ms_group = if df_group > 0 { ss_group / df_group as f32 } else { 0.0 };
    let ms_subgr = if df_subgr > 0 { ss_subgr / df_subgr as f32 } else { 1.0e-10 };
    let ms_within = if df_within > 0 { ss_within / df_within as f32 } else { 1.0e-10 };
    let f_group = ms_group / ms_subgr;
    let f_subgr = ms_subgr / ms_within;

    let p_group = betai(0.5 * df_subgr as f32, 0.5 * df_group as f32,
        df_subgr as f32 / (df_subgr as f32 + df_group as f32 * f_group));
    let p_subgr = betai(0.5 * df_within as f32, 0.5 * df_subgr as f32,
        df_within as f32 / (df_within as f32 + df_subgr as f32 * f_subgr));

    println!("  ANOVA Table:");
    println!("  Source        df      SS          MS          F        P");
    println!("  Groups     {:5}  {:10.3}  {:10.3}  {:8.3}  {:7.4}",
        df_group, ss_group, ms_group, f_group, p_group);
    println!("  Subgroups  {:5}  {:10.3}  {:10.3}  {:8.3}  {:7.4}",
        df_subgr, ss_subgr, ms_subgr, f_subgr, p_subgr);
    println!("  Within     {:5}  {:10.3}  {:10.3}",
        df_within, ss_within, ms_within);

    // Satterthwaite approximation
    if df_subgr < 100 && df_subgr < 2 * df_within {
        // Compute n0 for Satterthwaite
        let mut n0_sum = 0.0f32;
        let mut total_n = 0.0f32;
        let mut sum_nsq = 0.0f32;
        for g in 0..ngroups {
            for j in 0..nb[g] {
                let nij = nn[g][j] as f32;
                total_n += nij;
                sum_nsq += nij * nij;
            }
        }
        let a_total: usize = nb.iter().sum();
        if a_total > 0 && total_n > 0.0 {
            let n0 = (total_n - sum_nsq / total_n) / (a_total as f32 - ngroups as f32);
            let denom = ms_subgr + (n0 - 1.0).max(0.0) * ms_within;
            if denom > 0.0 {
                let fp = ms_group / denom;
                let dfp = (denom * denom)
                    / (ms_subgr * ms_subgr / df_subgr as f32
                     + ((n0 - 1.0).max(0.0) * ms_within).powi(2) / df_within as f32);
                let pp = betai(0.5 * dfp, 0.5 * df_group as f32,
                    dfp / (dfp + df_group as f32 * fp));
                println!("  Satterthwaite: F'({},{:.1}) = {:.3}, P = {:.4}",
                    df_group, dfp, fp, pp);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Statistics file reader
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct StatsData {
    nregion: usize,
    nsumarea: Vec<usize>,
    npixarea: Vec<usize>,
    nsets: usize,
    nsampl: Vec<i32>,
    avg: Vec<f32>,
    sd: Vec<f32>,
    sem: Vec<f32>,
    ntotarea: usize,
}

fn read_stats_file(path: &str) -> io::Result<StatsData> {
    let content = fs::read_to_string(path)?;
    let mut lines = content.lines();

    let nregion: usize = lines.next().unwrap_or("0").trim().parse().unwrap_or(0);
    let nsumarea: Vec<usize> = lines.next().unwrap_or("")
        .split_whitespace()
        .filter_map(|s| s.parse().ok())
        .collect();
    let ntotarea: usize = nsumarea.iter().sum();

    let ntotin: usize = lines.next().unwrap_or("0").trim().parse().unwrap_or(0);
    let npixarea: Vec<usize> = lines.next().unwrap_or("")
        .split_whitespace()
        .filter_map(|s| s.parse().ok())
        .collect();

    let nsets: usize = lines.next().unwrap_or("0").trim().parse().unwrap_or(0);

    let mut nsampl = Vec::with_capacity(nsets);
    let mut avg = Vec::with_capacity(nsets * ntotarea);
    let mut sd = Vec::with_capacity(nsets * ntotarea);
    let mut sem = Vec::with_capacity(nsets * ntotarea);

    for _iset in 0..nsets {
        let ns: i32 = lines.next().unwrap_or("1").trim().parse().unwrap_or(1);
        nsampl.push(ns);
        for _iarea in 0..ntotarea {
            let line = lines.next().unwrap_or("");
            let nums: Vec<f32> = line.split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();
            avg.push(*nums.get(1).unwrap_or(&0.0));
            sd.push(*nums.get(2).unwrap_or(&0.0));
            sem.push(*nums.get(3).unwrap_or(&0.0));
        }
    }

    Ok(StatsData {
        nregion, nsumarea, npixarea, nsets, nsampl, avg, sd, sem, ntotarea,
    })
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(name = "avganova")]
#[command(about = "Statistical comparisons using nested ANOVA on IMAVGSTAT output")]
struct Cli {
    /// Statistics file from IMAVGSTAT
    #[arg(short = 'f', long = "file")]
    stats_file: PathBuf,

    /// Group definitions: comma-separated set numbers per group, semicolon between groups
    /// e.g. "1,2,3;4,5,6" for two groups
    #[arg(short = 'g', long = "groups")]
    groups: String,

    /// Regions to test (comma-separated, ranges ok)
    #[arg(short = 'r', long = "regions")]
    regions: String,

    /// Use integrals (mean * pixels) instead of means
    #[arg(long = "integral")]
    integral: bool,
}

fn parse_range_list(s: &str) -> Vec<usize> {
    let mut result = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if let Some(dash) = part.find('-') {
            if dash > 0 {
                if let (Ok(start), Ok(end)) = (
                    part[..dash].parse::<usize>(),
                    part[dash + 1..].parse::<usize>(),
                ) {
                    for v in start..=end { result.push(v); }
                    continue;
                }
            }
        }
        if let Ok(v) = part.parse::<usize>() {
            result.push(v);
        }
    }
    result
}

fn main() {
    let cli = Cli::parse();

    let stats_path = cli.stats_file.to_str().unwrap_or("");
    let data = read_stats_file(stats_path).unwrap_or_else(|e| {
        eprintln!("ERROR: AVGANOVA - Cannot read stats file: {e}");
        process::exit(1);
    });

    println!("AVGANOVA: Nested ANOVA on IMAVGSTAT output");
    println!("  {} data sets, {} summing regions with {} total areas",
        data.nsets, data.nregion, data.ntotarea);
    println!("  Areas per region: {:?}", data.nsumarea);

    // Parse groups
    let group_strs: Vec<&str> = cli.groups.split(';').collect();
    let ngroups = group_strs.len();
    if ngroups < 2 {
        eprintln!("ERROR: AVGANOVA - Need at least 2 groups");
        process::exit(1);
    }

    let mut group_sets: Vec<Vec<usize>> = Vec::new();
    for gs in &group_strs {
        let sets = parse_range_list(gs);
        group_sets.push(sets);
    }

    let regions = parse_range_list(&cli.regions);

    // Build region index
    let mut ind_region = vec![0usize; data.nregion];
    let mut acc = 0;
    for i in 0..data.nregion {
        ind_region[i] = acc;
        acc += data.nsumarea[i];
    }

    // For each region and each area within it, run the ANOVA
    for &jreg in &regions {
        if jreg == 0 || jreg > data.nregion {
            eprintln!("WARNING: Region {jreg} out of range");
            continue;
        }
        let reg_idx = jreg - 1;
        let start = ind_region[reg_idx];
        let n_areas = data.nsumarea[reg_idx];

        for ia in 0..n_areas {
            let indar = start + ia;
            println!("\n{:>20}Comparison for area {} in region {}",
                "", ia + 1, jreg);

            let mut nb = Vec::new();
            let mut nn: Vec<Vec<i32>> = Vec::new();
            let mut xb: Vec<Vec<f32>> = Vec::new();
            let mut sds_vec: Vec<Vec<f32>> = Vec::new();
            let mut avav = Vec::new();
            let mut sdav = Vec::new();

            for (gi, sets) in group_sets.iter().enumerate() {
                let mut group_nn = Vec::new();
                let mut group_xb = Vec::new();
                let mut group_sd = Vec::new();
                let mut group_vals = Vec::new();

                for &jset in sets {
                    if jset == 0 || jset > data.nsets { continue; }
                    let idx = indar + data.ntotarea * (jset - 1);
                    let mut val = data.avg[idx];
                    let sd_val = data.sd[idx];
                    if cli.integral {
                        let npix = data.npixarea.get(indar).copied().unwrap_or(1) as f32;
                        val *= npix;
                    }
                    group_nn.push(data.nsampl[jset - 1]);
                    group_xb.push(val);
                    group_sd.push(sd_val);
                    group_vals.push(val);
                }

                let (mean, sd, sem) = avg_sd(&group_vals);
                println!(" Group {:2}, n={:3}, mean of means={:12.3}, SD={:11.3}, SEM={:11.3}",
                    gi + 1, group_vals.len(), mean, sd, sem);
                avav.push(mean);
                sdav.push(sd);
                nb.push(group_xb.len());
                nn.push(group_nn);
                xb.push(group_xb);
                sds_vec.push(group_sd);
            }

            // Simple t-test between first two groups
            if ngroups >= 2 && nb[0] > 0 && nb[1] > 0 {
                let (t, nt) = t_stat(
                    avav[0], sdav[0], nb[0] as i32,
                    avav[1], sdav[1], nb[1] as i32,
                );
                let pval = 1.0 - t_prob(nt, t.abs());
                println!("\n t({nt}) = {t:10.2},  P = {pval:7.4}");
            }

            nested_anova(ngroups, &nb, &nn, &xb, &sds_vec);
        }
    }

    println!("\nAVGANOVA complete.");
}
