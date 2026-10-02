//! Translation of `IMOD/raptor/correspondence/svlMarkerCorrespondenceLBModel.h`
//! and `svlMarkerCorrespondenceLBModel.cpp`: the Markov random field that
//! corresponds the markers of one projection with the candidates of the
//! next, solved by loopy (residual) belief propagation.
//!
//! Members the class declares but no reached function reads (the GGL-era
//! appearance, feature, angle and random-edge parameters, the clique index
//! lists) are not kept.  The four `new double[]`/`new int[]` arrays that
//! `ReadConfig` allocates are `Vec`s; see [`SvlMarkerCorrespondenceLbModel::read_config`]
//! for their size.
//!
//! `pow` is the C library's: gcc folds the source's `pow(x, 2)` into `x*x`
//! (which is what Rust's `powf(2.0)` becomes too), but keeps every other
//! `pow` a libm call, and glibc's `pow(x, 0.5)` is not `sqrt(x)` (it differs
//! in the last bit for about 1 in 1200 arguments).  So those go through
//! [`pow`], which LLVM cannot rewrite.

use std::collections::VecDeque;

use crate::imod::c_sort::list_sort;
use crate::imod::cxx_stream::{IStream, cerr, cout, ostream_double};
use crate::imod::libcfshr::b3dutil::cputime;
use crate::imod::raptor::external::tnt_array2d::Array2D;
use crate::imod::raptor::lasik::svl::lib::pgm::svl_cluster_graph::{SvlClique, SvlClusterGraph};
use crate::imod::raptor::lasik::svl::lib::pgm::svl_factor::SvlFactor;
use crate::imod::raptor::lasik::svl::lib::pgm::svl_message_passing::{
    SvlMessagePassingAlgorithms, SvlMessagePassingInference,
};

/// `typedef TNT::Array2D<double> gglMatrix`.
pub type GglMatrix = Array2D;

/// C library `pow(x, y)`, called through a pointer the optimiser cannot see
/// through, so that it is glibc's `pow` exactly as the native program calls
/// it (see the module comment).
fn pow(x: f64, y: f64) -> f64 {
    unsafe extern "C" {
        fn pow(x: f64, y: f64) -> f64;
    }
    let f: unsafe extern "C" fn(f64, f64) -> f64 = std::hint::black_box(pow);
    // SAFETY: glibc `pow` is a pure function of two doubles.
    unsafe { f(x, y) }
}

/// `clock()` in ticks of `CLOCKS_PER_SEC` (1 000 000 on this platform),
/// recovered exactly from `cputime()`, which is `clock() / CLOCKS_PER_SEC`.
fn clock_ticks() -> i64 {
    (cputime() * 1_000_000.0).round() as i64
}

/// `class svlMarkerCorrespondenceLBModel`.
pub struct SvlMarkerCorrespondenceLbModel {
    pub pair_to_clique: Vec<Vec<i32>>,
    pub pair_distances: Vec<Vec<f64>>,

    lb_graph: SvlClusterGraph,
    lb_cards: Vec<i32>,

    // loopy
    inference_built: bool,
    max_product: bool,
    max_messages: i32,

    /// For S_INITIAL_PROXIMITY: only use points within this distance.
    proximity_threshold: f64,
    /// Minimum number of candidates per marker to enforce.
    min_cands: u32,
    allow_vals: Vec<Vec<i32>>,

    marker_locations: GglMatrix,
    marker_locations2: GglMatrix,
    marker_candidates: GglMatrix,
    sp_matrix: GglMatrix,
    test_name: String,
    sing_po_choice: i32,
    pair_po_choice: i32,
    intcpt_b: i32,
    min_pw_cliques: i32,
    max_mrkr_pair_dist_cfg: f64,
    dist_thr_min: Vec<f64>,
    dist_thr_max: Vec<f64>,
    lock_pot: Vec<i32>,
    locked_pot_val: Vec<i32>,
    lb_infer: Option<SvlMessagePassingInference>,

    /// Only for debugging initial beliefs.
    debug_initial_beliefs: i32,
    /// Print out messages.
    verbose: i32,
    /// Measure time.
    mtime: i32,
    debug_pair_pots: i32,
    pair_pots_m1: i32,
    pair_pots_m2: i32,
    /// For distance decay function.
    dist_diff_intcpt: f64,
    max_mrkr_pair_dist: f64,
    /// Initial potential for garbage can.
    gcan_potential: f64,
    ppot_scale_factor: f64,
    num_markers: i32,
    num_candidates: i32,
    num_markers_for_dist: i32,

    graph_built: bool,
    model_built: bool,
}

impl SvlMarkerCorrespondenceLbModel {
    /// The constructor (`svlMarkerCorrespondenceLBModel.cpp:36`).
    ///
    /// `verbose` is read by `ReadConfig`'s `if(verbose)` lines before this
    /// constructor assigns it (`:47` versus `:55`), an uninitialised read in
    /// the source; it is 0 until then here (`BUGS.md`).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        marker_locations1: &GglMatrix,
        marker_locations2: &GglMatrix,
        marker_candidates2: &GglMatrix,
        sp_matrix: &GglMatrix,
        test_name: &str,
        lock_pots_file_exists: i32,
        infilename: &mut IStream,
        infilename_lock_pots: Option<&mut IStream>,
        debug_init_beliefs: i32,
        debug_pair_pots: i32,
        pair_pots_m1: i32,
        pair_pots_m2: i32,
        _dist_diff_intcpt: f64,
        verbose: i32,
        mtime: i32,
    ) -> SvlMarkerCorrespondenceLbModel {
        let mut model = SvlMarkerCorrespondenceLbModel {
            pair_to_clique: Vec::new(),
            pair_distances: Vec::new(),
            lb_graph: SvlClusterGraph::new(),
            lb_cards: Vec::new(),
            inference_built: false,
            max_product: false,
            max_messages: 0,
            proximity_threshold: 0.0,
            min_cands: 0,
            allow_vals: Vec::new(),
            marker_locations: marker_locations1.clone(),
            marker_locations2: marker_locations2.clone(),
            marker_candidates: marker_candidates2.clone(),
            sp_matrix: sp_matrix.clone(),
            test_name: String::new(),
            sing_po_choice: 0,
            pair_po_choice: 0,
            intcpt_b: 0,
            min_pw_cliques: 0,
            max_mrkr_pair_dist_cfg: 0.0,
            dist_thr_min: Vec::new(),
            dist_thr_max: Vec::new(),
            lock_pot: Vec::new(),
            locked_pot_val: Vec::new(),
            lb_infer: None,
            debug_initial_beliefs: 0,
            verbose: 0,
            mtime: 0,
            debug_pair_pots: 0,
            pair_pots_m1: 0,
            pair_pots_m2: 0,
            dist_diff_intcpt: 0.0,
            max_mrkr_pair_dist: 0.0,
            gcan_potential: 0.0,
            ppot_scale_factor: 0.0,
            num_markers: 0,
            num_candidates: 0,
            num_markers_for_dist: 0,
            graph_built: false,
            model_built: false,
        };
        // Member Variables
        model.num_markers = model.marker_locations.dim2();
        model.num_candidates = model.marker_candidates.dim2();
        model.test_name = test_name.to_owned();
        model.read_config(infilename);
        if lock_pots_file_exists != 0 {
            if let Some(lock) = infilename_lock_pots {
                model.read_lock_pots(lock);
            }
        }
        model.debug_initial_beliefs = debug_init_beliefs;
        model.verbose = verbose;
        model.mtime = mtime;
        model.debug_pair_pots = debug_pair_pots;
        model.pair_pots_m1 = pair_pots_m1;
        model.pair_pots_m2 = pair_pots_m2;
        model.dist_diff_intcpt = model.intcpt_b as f64;
        model.max_mrkr_pair_dist = model.max_mrkr_pair_dist_cfg;
        model.init_values();
        model
    }

    /// `InitValues()` (`svlMarkerCorrespondenceLBModel.cpp:97`).
    fn init_values(&mut self) {
        // loopy
        self.lb_infer = None;
        // true->we use max-product;false->we use sum product
        self.max_product = false;
        // for ggl it was 25000
        self.max_messages = 10000;

        self.lb_cards.clear();
        self.inference_built = false;

        // best contour
        self.model_built = false;
        self.graph_built = false;
    }

    /// `ReadConfig(istream& in)` (`svlMarkerCorrespondenceLBModel.h:285`).
    ///
    /// The four per-marker arrays are `new double[_NumMarkers_for_dist]` in
    /// the source and are indexed by marker, so a configuration whose first
    /// line is smaller than the number of markers in the `.mrkr` file makes
    /// the source read past them (`BUGS.md`).  Here they are sized for
    /// `max(_NumMarkers_for_dist, number of markers)`; the entries the
    /// configuration does not supply are thresholds of 0 and unlocked.
    fn read_config(&mut self, input: &mut IStream) {
        input.read_i32(&mut self.num_markers_for_dist);
        if self.verbose != 0 {
            cerr(&format!(
                "read in num markers = {}\n",
                self.num_markers_for_dist
            ));
        }

        input.read_i32(&mut self.sing_po_choice);
        if self.verbose != 0 {
            cerr(&format!(
                "read in single potential method {}\n",
                self.sing_po_choice
            ));
        }
        input.read_i32(&mut self.pair_po_choice);
        if self.verbose != 0 {
            cerr(&format!(
                "read in pairwise potential method {}\n",
                self.pair_po_choice
            ));
        }
        input.read_i32(&mut self.intcpt_b);
        if self.verbose != 0 {
            cerr(&format!("read in rolloff factor k2 of {}\n", self.intcpt_b));
        }
        input.read_f64(&mut self.max_mrkr_pair_dist_cfg);
        if self.verbose != 0 {
            cerr(&format!(
                "read in maximum marker pair distance for pairwise cliques of {}\n",
                ostream_double(self.max_mrkr_pair_dist_cfg)
            ));
        }
        input.read_f64(&mut self.gcan_potential);
        if self.verbose != 0 {
            cerr(&format!(
                "read in garbage can potential: {}\n ",
                ostream_double(self.gcan_potential)
            ));
        }
        input.read_f64(&mut self.ppot_scale_factor);
        if self.verbose != 0 {
            cerr(&format!(
                "read in pairwise potential scale factor: {}\n ",
                ostream_double(self.ppot_scale_factor)
            ));
        }
        input.read_i32(&mut self.min_pw_cliques);
        if self.verbose != 0 {
            cerr(&format!(
                "read in minimum number of pairwise cliques: {}\n ",
                self.min_pw_cliques
            ));
        }

        input.read_f64(&mut self.proximity_threshold);
        if self.verbose != 0 {
            cerr(&format!(
                "read in proximity threshold:  {}\n ",
                ostream_double(self.proximity_threshold)
            ));
        }
        let mut min_cands = self.min_cands as i32;
        input.read_i32(&mut min_cands);
        self.min_cands = min_cands as u32;
        if self.verbose != 0 {
            cerr(&format!(
                "read in min number of candidates :  {}\n ",
                self.min_cands
            ));
        }

        let size = self.num_markers_for_dist.max(self.num_markers).max(0) as usize;
        self.dist_thr_min = vec![0.0; size];
        self.dist_thr_max = vec![0.0; size];
        self.lock_pot = vec![0; size];
        self.locked_pot_val = vec![-1; size];
        for i in 0..self.num_markers_for_dist.max(0) as usize {
            // initialize lockdown potentials
            self.lock_pot[i] = 0;
            self.locked_pot_val[i] = -1;

            input.read_f64(&mut self.dist_thr_min[i]);
            input.read_f64(&mut self.dist_thr_max[i]);
            if self.verbose != 0 {
                cerr(&format!(
                    "read in min distance threshold : {}: {}\n",
                    i,
                    ostream_double(self.dist_thr_min[i])
                ));
            }
            if self.verbose != 0 {
                cerr(&format!(
                    "read in max distance threshold : {}\n",
                    ostream_double(self.dist_thr_max[i])
                ));
            }
        }
    }

    /// `ReadLockPots(istream& in)` (`svlMarkerCorrespondenceLBModel.h:328`).
    ///
    /// The source loops `while (!in.eof())`, which never ends once a read
    /// fails before the end of the file (a non-numeric token), and stores
    /// through `_lock_pot[marker]` for any `marker` it reads.  Here the loop
    /// also stops when the stream fails, and a marker outside the arrays is
    /// ignored (`BUGS.md`).
    fn read_lock_pots(&mut self, input: &mut IStream) {
        let mut marker = 0i32;
        let mut candidate = 0i32;
        if input.ok() {
            while !input.eof() && input.ok() {
                input.read_i32(&mut marker);
                input.read_i32(&mut candidate);
                if marker >= 0 && (marker as usize) < self.lock_pot.len() {
                    self.lock_pot[marker as usize] = 1;
                    self.locked_pot_val[marker as usize] = candidate;
                }
                if self.verbose != 0 {
                    cerr(&format!(
                        "locked marker  {marker} to candidate {candidate}\n"
                    ));
                }
            }
        }
        if self.verbose != 0 {
            cerr("finished with locked markers\n ");
        }
    }

    /// `GetFinalMarginalBeliefs()` (`svlMarkerCorrespondenceLBModel.cpp:134`):
    /// runs inference and returns the beliefs, one row per marker and one
    /// column per candidate plus the garbage can.
    pub fn get_final_marginal_beliefs(&mut self) -> GglMatrix {
        let init_total_time = clock_ticks();
        // Garbage can
        let mut c = GglMatrix::new(self.num_markers, self.num_candidates + 1);
        self.build_model();

        if !self.inference_built {
            if self.verbose != 0 {
                cerr("Constructing loopy inference and propagating messages\n");
            }
            let init_time = clock_ticks();
            let succ;
            let mut infer = SvlMessagePassingInference::new(&self.lb_graph);
            if self.max_product {
                succ = infer.inference(SvlMessagePassingAlgorithms::RbpMaxProd, self.max_messages);
            } else {
                succ = infer.inference(SvlMessagePassingAlgorithms::RbpSumProd, self.max_messages);
            }
            self.lb_infer = Some(infer);

            if self.verbose != 0 {
                cerr("Finished Calculating Probabilities...\n");
            }
            let final_time = clock_ticks();
            if self.mtime != 0 {
                cerr(&format!(
                    "Inference took {} seconds\n",
                    (final_time - init_time) / 1_000_000
                ));
            }
            if self.mtime != 0 {
                cerr(&format!(
                    "TOTAL OPERATION took {} seconds\n",
                    (final_time - init_total_time) / 1_000_000
                ));
            }
            if !succ {
                cout("NOT CONVERGED!\n");
            } else {
                cout("CONVERGED!\n");
            }
        }

        if self.verbose != 0 {
            cout("\nThe WINNERS ARE:\n");
        }
        let infer = self.lb_infer.as_ref().unwrap();
        for i in 0..self.num_markers as usize {
            let mut winner = 0;
            let cvec = [self.lb_cards[i]];

            // Get the final belief over variable i
            let meas_ptr = infer.clique_potential(i).clone();

            let mut max = -1000000000.0f64;
            // Initialize C to all very small numbers
            for jj in 0..=self.num_candidates as usize {
                c[i][jj] = -1e12;
            }
            // Garbage Can
            for jj in 1..=cvec[0] as usize {
                let j = self.allow_vals[i][jj];

                // THIS IS THE PROBABILITY THAT MARKER i
                // IS ASSIGNED TO CANDIDATE j
                let val = meas_ptr.get(jj - 1);
                if val > max {
                    max = val;
                    winner = j;
                }
                c[i][j as usize] = val.ln();
            }
            if self.verbose != 0 {
                cout(&format!("{i}:{winner}\n"));
            }
        }
        if self.verbose != 0 {
            cout("\n");
        }
        c
    }

    /// `ComputeAllowedValues()` (`svlMarkerCorrespondenceLBModel.cpp:254`):
    /// based only on proximity.
    fn compute_allowed_values(&mut self) {
        self.allow_vals.clear();
        self.allow_values_by_initial_proximity();
    }

    /// `AllowValuesByInitialProximity()` (`svlMarkerCorrespondenceLBModel.cpp:263`).
    fn allow_values_by_initial_proximity(&mut self) {
        let ncandidates = self.marker_candidates.dim2() as u32;
        let nmarkers = self.marker_locations.dim2() as u32;
        let mut cand_assigned_flag = vec![0i32; ncandidates as usize];

        // figure out minimum number of candidates
        let min_cands = if self.min_cands < ncandidates {
            self.min_cands
        } else {
            ncandidates
        };

        // For Each Point, Allow
        for i in 0..nmarkers as usize {
            // reset candidates assigned flag
            for j in 0..ncandidates as usize {
                cand_assigned_flag[j] = 0;
            }
            let mut tmp: Vec<i32> = vec![-1];
            let mut thresh = self.proximity_threshold;
            while (tmp.len() as u32) < min_cands {
                for j in 0..ncandidates as usize {
                    if (cand_assigned_flag[j] == 0)
                        && (self.dist_l2(
                            self.marker_locations[0][i],
                            self.marker_locations[1][i],
                            self.marker_candidates[0][j],
                            self.marker_candidates[1][j],
                        ) < thresh)
                    {
                        cand_assigned_flag[j] = 1;
                        tmp.push(j as i32);
                    }
                }
                thresh *= 1.5;
            }
            // automatically add garbage can as allowed value
            tmp.push(ncandidates as i32);

            if self.verbose != 0 {
                cerr(&format!(
                    "THRESH = {}: Allowed {} values for marker {}\n",
                    ostream_double(thresh),
                    tmp.len() - 1,
                    i
                ));
            }
            self.allow_vals.push(tmp);
        }
    }

    /// `BuildModel()` (`svlMarkerCorrespondenceLBModel.cpp:334`).
    fn build_model(&mut self) {
        if self.model_built {
            return;
        }

        if self.verbose != 0 {
            cerr("Constructing lbModel object\n");
        }

        // STEP -1: Compute allowed values
        self.compute_allowed_values();

        // STEP 0:
        self.build_cards();

        // build cluster graph
        if self.graph_built {
            return;
        }
        self.lb_graph = SvlClusterGraph::with_cards(self.num_markers, &self.lb_cards);
        self.graph_built = true;

        // STEP 1:
        let init_time = clock_ticks();
        self.build_singleton_potentials();
        let final_time = clock_ticks();
        if self.mtime != 0 {
            cerr(&format!(
                "Singleton Potentials took {} seconds\n",
                (final_time - init_time) / 1_000_000
            ));
        }

        // STEP 2: build pairwise potentials
        let init_time = clock_ticks();
        self.build_pairwise_cliques();
        self.build_pairwise_potentials();
        let final_time = clock_ticks();
        if self.mtime != 0 {
            cerr(&format!(
                "Pairwise Potentials took {} seconds\n",
                (final_time - init_time) / 1_000_000
            ));
        }

        // STEP 3: connect cliques
        self.lb_graph.bethe_approx();

        self.model_built = true;
    }

    /// `BuildCards()` (`svlMarkerCorrespondenceLBModel.cpp:376`): the
    /// cardinality of each marker variable is its number of allowed
    /// candidates, garbage can included.
    fn build_cards(&mut self) {
        let nmarkers = self.marker_locations.dim2();

        self.lb_cards.clear();
        self.lb_cards.reserve(nmarkers.max(0) as usize);
        for i in 0..nmarkers as usize {
            // Garbage Can
            self.lb_cards.push(self.allow_vals[i].len() as i32 - 1);
        }
    }

    /// `BuildSingletonPotentials()` (`svlMarkerCorrespondenceLBModel.cpp:465`):
    /// occlusion prior potentials; also writes `<test>_init_beliefs_scr.m`.
    fn build_singleton_potentials(&mut self) {
        let nmarkers = self.marker_locations.dim2();
        let ncandidates = self.marker_candidates.dim2();
        let mut max_dot_i_j_k_l: f64;
        let mut norm_i_j_k_l: f64;
        let mut dist_mrkr_cand: f64;
        let mut factor: f64;
        let mut potential_value: f64;
        let mut ind = 0;

        let fname = format!("{}_init_beliefs_scr.m", self.test_name);
        let init_beliefs_var_name = format!("{}_init_beliefs", self.test_name);
        let mut out = String::new();

        // For each marker, candidate pair..
        if self.verbose != 0 {
            cout("\n\n\nCreating Singleton Potential Matrix Initial Belief SP:\n ");
        }
        out.push_str(&format!("{init_beliefs_var_name}=[ \n "));
        for i in 0..nmarkers as usize {
            let cvec = [self.lb_cards[i]];
            // factor to store table for i-th marker
            let mut single_meas = SvlFactor::new();
            single_meas.add_variable(i as i32, self.lb_cards[i]);

            // ... And For each candidate,
            for jj in 1..=cvec[0] as usize {
                let j = self.allow_vals[i][jj];

                // FIRST: Check if Potential is LOCKED DOWN:
                if self.lock_pot[i] == 1 {
                    if self.locked_pot_val[i] == j {
                        potential_value = 1.0;
                    } else {
                        potential_value = 1e-10;
                    }
                }
                // OTHERWISE calculate potential as usual:
                else {
                    // Initialize potential... VALUE OF ASSIGNING MARKER i TO CANDIDATE j
                    potential_value = 1e-10;

                    // evaluate singleton potentials other than garbage can
                    if j < ncandidates {
                        let j = j as usize;
                        // singleton potentials options:
                        // 1:uniform
                        // 2:dot product based
                        // 3:read in from file directly with no qualification
                        // 4:uniform subject to distance constraint
                        // 5:read in from file directly multiplied by Gaussian distance rolloff distribution
                        if self.sing_po_choice == 1 {
                            // Farshid 4-17-06- using uniform potentials
                            potential_value = 1.0;
                        }

                        if self.sing_po_choice == 2 {
                            // For each remaining marker...
                            for k in 0..nmarkers as usize {
                                if k != i {
                                    max_dot_i_j_k_l = 0.0;
                                    // .... And For each remaining candidate...
                                    for l in 0..ncandidates as usize {
                                        if l != j {
                                            // calculate dot products, take max
                                            norm_i_j_k_l = self.norm_dot(
                                                self.marker_locations[0][i],
                                                self.marker_locations[1][i],
                                                self.marker_candidates[0][j],
                                                self.marker_candidates[1][j],
                                                self.marker_locations[0][k],
                                                self.marker_locations[1][k],
                                                self.marker_candidates[0][l],
                                                self.marker_candidates[1][l],
                                                self.dist_thr_max[i],
                                                self.dist_thr_min[i],
                                                self.dist_thr_max[k],
                                                self.dist_thr_min[k],
                                            );
                                            if norm_i_j_k_l > max_dot_i_j_k_l {
                                                max_dot_i_j_k_l = norm_i_j_k_l;
                                            }
                                        }
                                    }

                                    // update potential accumulation for current marker-candidate pair
                                    potential_value += max_dot_i_j_k_l;
                                }
                            }
                        }

                        // Farshid 5-7-06 read singleton potentials directly from file
                        if (self.sing_po_choice == 3) || (self.sing_po_choice == 5) {
                            potential_value = self.sp_matrix[i][j];
                            // multiply by e^-(dist/dist_thr)^2
                            if self.sing_po_choice == 5 {
                                dist_mrkr_cand = self.dist_l2(
                                    self.marker_locations2[0][i],
                                    self.marker_locations2[1][i],
                                    self.marker_candidates[0][j],
                                    self.marker_candidates[1][j],
                                );

                                let r = dist_mrkr_cand / self.dist_thr_max[i];
                                factor = (-(r * r)).exp();
                                potential_value *= factor;
                            }
                        }

                        if self.sing_po_choice == 4 {
                            if self.dist_l2(
                                self.marker_locations[0][i],
                                self.marker_locations[1][i],
                                self.marker_candidates[0][j],
                                self.marker_candidates[1][j],
                            ) < self.dist_thr_max[i]
                            {
                                potential_value = 1.0;
                            } else {
                                potential_value = 1e-10;
                            }
                        }
                    }
                    // this is the garbage can
                    else {
                        potential_value = self.gcan_potential;
                    }
                }
                // put lower limit on potentials
                if potential_value < 1e-10 {
                    potential_value = 1e-10;
                }
                out.push_str(&ostream_double(potential_value));
                out.push_str("  ");
                if j == ncandidates {
                    out.push_str(";\n");
                } else {
                    out.push_str(", ");
                }

                single_meas.set(jj - 1, potential_value);
            }

            ind += 1;
            let mut cl = SvlClique::new();
            cl.insert(i as i32);
            self.lb_graph.add_clique(&cl, &single_meas);
        }
        if self.verbose != 0 {
            cerr(&format!(
                "\n\n\nCompleted Calculation of  {ind}  Singleton potentials\n"
            ));
        }

        out.push_str("];\n ");
        // `out.open(fname)`: nothing is written when it cannot be opened.
        let _ = std::fs::write(&fname, out.as_bytes());
    }

    /// `norm_dot(...)` (`svlMarkerCorrespondenceLBModel.cpp:679`): absolute
    /// value of the normalised dot product of two vectors, raised to the
    /// tenth power, when both lengths are within their thresholds.
    #[allow(clippy::too_many_arguments)]
    pub fn norm_dot(
        &self,
        x1: f64,
        y1: f64,
        x2: f64,
        y2: f64,
        x3: f64,
        y3: f64,
        x4: f64,
        y4: f64,
        dist_thr_max_a: f64,
        dist_thr_min_a: f64,
        dist_thr_max_b: f64,
        dist_thr_min_b: f64,
    ) -> f64 {
        // calculate unit vectors
        let norm_a = pow((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1), 0.5);
        let norm_b = pow((x4 - x3) * (x4 - x3) + (y4 - y3) * (y4 - y3), 0.5);

        let veca_x = x2 - x1;
        let veca_y = y2 - y1;
        let vecb_x = x4 - x3;
        let vecb_y = y4 - y3;

        let uveca_x = veca_x / norm_a;
        let uveca_y = veca_y / norm_a;
        let uvecb_x = vecb_x / norm_b;
        let uvecb_y = vecb_y / norm_b;

        // dotproduct of a/|a| and a
        let dist_a = uveca_x * veca_x + uveca_y * veca_y;

        // dotproduct of b/|b| and b
        let dist_b = uvecb_x * vecb_x + uvecb_y * vecb_y;

        // calculate dot product of unit vectors
        if dist_a < dist_thr_max_a
            && dist_a > dist_thr_min_a
            && dist_b > dist_thr_min_b
            && dist_b < dist_thr_max_b
        {
            // Farshid 5/21/2006 -added power for dot product
            pow((uveca_x * uvecb_x + uveca_y * uvecb_y).abs(), 10.0)
        } else {
            0.0
        }
    }

    /// `dist_l2(x1, y1, x2, y2)` (`svlMarkerCorrespondenceLBModel.cpp:723`):
    /// Euclidean distance between 2 points.
    pub fn dist_l2(&self, x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
        pow((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1), 0.5)
    }

    /// `BuildPairwiseCliques()` (`svlMarkerCorrespondenceLBModel.cpp:748`).
    ///
    /// For each marker the source sorts the distances to the other markers,
    /// pops `_min_pw_cliques` of them and takes the front of the rest.  With
    /// no more than `_min_pw_cliques` other markers (RAPTOR writes
    /// `_min_pw_cliques = M - 1` when it has at most 8 markers) the list is
    /// then empty and the source dereferences `end()` -- reading the list's
    /// size field, 0, as a double -- and would pop an empty list with fewer
    /// markers still.  Here popping stops at an empty list and the range is
    /// the last distance popped, the `_min_pw_cliques`-th smallest the
    /// comment asks for (the initial 1000000 when nothing was popped)
    /// (`BUGS.md`).
    fn build_pairwise_cliques(&mut self) {
        let cind = 1;
        let nmarkers = self.marker_locations.dim2();
        let mut p_clique_cnt = 0;
        let mut dist_list: VecDeque<f64> = VecDeque::new();

        let mut min_range_per_mrkr = vec![0.0f64; nmarkers.max(0) as usize];

        // maintain stats for marker distances
        self.pair_distances = vec![vec![0.0; nmarkers.max(0) as usize]; nmarkers.max(0) as usize];
        for i in 0..nmarkers as usize {
            // initialize min range for marker i
            min_range_per_mrkr[i] = 1000000.0;
            if !dist_list.is_empty() {
                cerr("BIG FAT ERROR! DIST LIST NOT EMPTY!\n");
            }
            for j in i..nmarkers as usize {
                // calculate distance between marker i and marker j, push onto list
                self.pair_distances[i][j] = self.dist_l2(
                    self.marker_locations[0][i],
                    self.marker_locations[1][i],
                    self.marker_locations[0][j],
                    self.marker_locations[1][j],
                );
                if i != j {
                    dist_list.push_back(self.pair_distances[i][j]);
                }
            }
            for j in 0..i {
                self.pair_distances[i][j] = self.pair_distances[j][i];
                if i != j {
                    dist_list.push_back(self.pair_distances[i][j]);
                }
            }
            // sort list of distances
            let mut sorted: Vec<f64> = dist_list.drain(..).collect();
            list_sort(&mut sorted, &mut |a: &f64, b: &f64| a < b);
            dist_list.extend(sorted);
            // pop first min_pw_cliques entries
            let mut last_popped = None;
            for _k in 0..self.min_pw_cliques {
                if let Some(d) = dist_list.pop_front() {
                    last_popped = Some(d);
                }
            }
            // get min_pw_cliques'th lowest distance
            min_range_per_mrkr[i] = match dist_list.front() {
                Some(&p) => p,
                None => last_popped.unwrap_or(min_range_per_mrkr[i]),
            };
            // clear list for next marker
            dist_list.clear();
        }

        self.pair_to_clique = vec![vec![0; nmarkers.max(0) as usize]; nmarkers.max(0) as usize];

        for i in 0..nmarkers as usize {
            for j in i..nmarkers as usize {
                let mrkr_pair_dist = self.pair_distances[i][j];
                let alice = i;
                // just for fun
                let bob = j;
                // check for markers being closer than the max, or close enough to guarantee
                // a minimum number of cliques
                if (j != i)
                    & ((mrkr_pair_dist < self.max_mrkr_pair_dist)
                        | (mrkr_pair_dist <= min_range_per_mrkr[i]))
                {
                    self.pair_to_clique[alice][bob] = cind;
                    p_clique_cnt += 1;
                } else {
                    self.pair_to_clique[alice][bob] = -1;
                }
            }
        }
        cerr(&format!("Created {p_clique_cnt} PAIRWISE cliques\n"));
    }

    /// `BuildPairwisePotentials()` (`svlMarkerCorrespondenceLBModel.cpp:884`):
    /// pairwise potentials of distance.
    ///
    /// The debug marginals (`debug_initial_beliefs`, which `MarkersCorrespond`
    /// sets only from its `-debug_pair_pots` option, through the swapped
    /// constructor arguments) are indexed by candidate in the source,
    /// including the garbage can at `ncandidates` one past the
    /// `new double[ncandidates]` rows, and printed over `nmarkers` columns;
    /// here the rows have `ncandidates + 1` entries and the print stops at
    /// that length (`BUGS.md`).
    fn build_pairwise_potentials(&mut self) {
        let mut ind = 0;

        let mut norm_i1j1_i2j2 = 0.0f64;
        let mut norm_i_j_k_l_1: f64;
        let mut norm_i_j_k_l: f64;
        let mut norm_prods_i1: Vec<Vec<Vec<f64>>> = Vec::new();

        // for precalculating distances between markers and candidates (pair_po_choice=8)
        let mut exp_mrkr_cand_dists: Vec<Vec<f64>> = Vec::new();

        // DEBUG potentials
        let mut marg_sing_pots: Vec<Vec<f64>> = Vec::new();

        let mut cand_pair_dist: f64;
        let mut mrkr_pair_dist: f64;
        let mut mrkr_cand_dist_1: f64;
        let mut x1: f64;
        let mut x2: f64;
        let mut y1: f64;
        let mut y2: f64;
        let mut dist_vec_norm: f64;
        let mut rolloff_fac: f64;
        let mut dist_diff: f64;
        let mut potential_value: f64;
        let mut num_cand_pairs = 0;

        // for each pair of markers:
        let nmarkers = self.marker_locations.dim2();
        let ncandidates = self.marker_candidates.dim2();
        let nm = nmarkers.max(0) as usize;
        let nc = ncandidates.max(0) as usize;
        let mut num_entries = 0;

        // _pair_po_choice
        // 1-use dot product of pairs of matches and that of all possible remaining matches
        // 2-use dot product of pairs only
        // 3-use dot product of pairs only and distance based constraint
        // 4-use only distance based constraint
        // 5-uniform
        // 6-use dot product of pairs only and new distance based constraint
        // 7-use only new distance based constraint
        // 8- 7+exponential rolloff

        // when using all dot products, precalculate them
        if self.pair_po_choice == 1 {
            norm_prods_i1 = vec![vec![vec![0.0; nc]; nm]; nc];
        }

        // when using rolloffs, precalculate them
        if self.pair_po_choice == 8 {
            exp_mrkr_cand_dists = vec![vec![0.0; nc]; nm];
        }

        // DEBUG potentials
        if self.debug_initial_beliefs != 0 {
            marg_sing_pots = vec![vec![0.0; nc + 1]; nm];
        }

        // precalculate exp(-dist^2) if using pair_po_choice=8
        if self.pair_po_choice == 8 {
            for i in 0..nm {
                for j in 0..nc {
                    // evaluate distance of marker i and candidate j
                    mrkr_cand_dist_1 = self.dist_l2(
                        self.marker_locations2[0][i],
                        self.marker_locations2[1][i],
                        self.marker_candidates[0][j],
                        self.marker_candidates[1][j],
                    );
                    let r = mrkr_cand_dist_1 / self.dist_thr_max[i];
                    exp_mrkr_cand_dists[i][j] = (-(r * r)).exp();
                }
            }
        }

        // DEBUG potentials-initialize marginals
        if self.debug_initial_beliefs != 0 {
            for i1 in 0..nm {
                for j1 in 0..nc {
                    marg_sing_pots[i1][j1] = 0.0;
                }
            }
        }

        // For every marker...
        for i1 in 0..nm {
            // precalculate normalized dot products
            if self.pair_po_choice == 1 {
                for i2 in 0..nm {
                    for k2 in 0..nc {
                        for l2 in 0..nc {
                            norm_prods_i1[k2][i2][l2] = self.norm_dot(
                                self.marker_locations[0][i1],
                                self.marker_locations[1][i1],
                                self.marker_candidates[0][k2],
                                self.marker_candidates[1][k2],
                                self.marker_locations[0][i2],
                                self.marker_locations[1][i2],
                                self.marker_candidates[0][l2],
                                self.marker_candidates[1][l2],
                                self.dist_thr_max[i1],
                                self.dist_thr_min[i1],
                                self.dist_thr_max[i2],
                                self.dist_thr_min[i2],
                            );
                        }
                    }
                }
            }

            // ....and for each other marker...
            for i2 in i1 + 1..nm {
                // If clique exists
                if self.pair_to_clique[i1][i2] != -1 {
                    // Create Potential
                    let cvec = [self.lb_cards[i1], self.lb_cards[i2]];

                    let mut pairwise_meas = SvlFactor::new();
                    pairwise_meas.add_variable(i1 as i32, self.lb_cards[i1]);
                    pairwise_meas.add_variable(i2 as i32, self.lb_cards[i2]);
                    let mut cl = SvlClique::new();
                    cl.insert(i1 as i32);
                    cl.insert(i2 as i32);

                    // calculate number of entries
                    num_entries = cvec[0] * cvec[1];

                    let debug_pair = self.debug_pair_pots != 0
                        && i1 as i32 == self.pair_pots_m1
                        && i2 as i32 == self.pair_pots_m2;
                    if debug_pair {
                        cout(&format!("PAIRWISE POT for MARKERS {i1} and  {i2}\n"));
                    }
                    // ... And For each pair of candidates,
                    for jj1 in 1..=cvec[0] as usize {
                        let j1 = self.allow_vals[i1][jj1];
                        for jj2 in 1..=cvec[1] as usize {
                            let j2 = self.allow_vals[i2][jj2];

                            // FIRST: Check if Potential(s) is (are) NOT LOCKED DOWN:
                            if ((self.lock_pot[i1] == 1)
                                && (self.lock_pot[i2] == 0)
                                && (self.locked_pot_val[i1] == j1))
                                || ((self.lock_pot[i1] == 0)
                                    && (self.lock_pot[i2] == 1)
                                    && (self.locked_pot_val[i2] == j2))
                                || ((self.lock_pot[i1] == 0) && (self.lock_pot[i2] == 0))
                            {
                                // calculate pairwise potential:
                                // VALUE OF ASSIGNING MARKER i1 TO CANDIDATE j1
                                potential_value = 1.0;
                                // real entries
                                if j1 < ncandidates && j2 < ncandidates {
                                    let (j1u, j2u) = (j1 as usize, j2 as usize);
                                    // one dot product only
                                    if (self.pair_po_choice == 2)
                                        || (self.pair_po_choice == 3)
                                        || (self.pair_po_choice == 6)
                                    {
                                        norm_i1j1_i2j2 = self.norm_dot(
                                            self.marker_locations[0][i1],
                                            self.marker_locations[1][i1],
                                            self.marker_candidates[0][j1u],
                                            self.marker_candidates[1][j1u],
                                            self.marker_locations[0][i2],
                                            self.marker_locations[1][i2],
                                            self.marker_candidates[0][j2u],
                                            self.marker_candidates[1][j2u],
                                            self.dist_thr_max[i1],
                                            self.dist_thr_min[i1],
                                            self.dist_thr_max[i2],
                                            self.dist_thr_min[i2],
                                        );
                                    }
                                    // Initialize potential...
                                    if self.pair_po_choice == 1 {
                                        // using sum of products, initialize sum to 0
                                        potential_value = 0.0;
                                    } else if self.pair_po_choice == 2 {
                                        // one dot product only
                                        potential_value = norm_i1j1_i2j2;
                                    }

                                    // By definition, i1!=i2 because pair ordering
                                    // doesn't matter for the markers but it does for
                                    // the candidates
                                    if j2 != j1 {
                                        // using original distance constraint
                                        if (self.pair_po_choice == 3) | (self.pair_po_choice == 4) {
                                            // evaluate pairwise distance of markers i1 and i2
                                            mrkr_pair_dist = self.dist_l2(
                                                self.marker_locations[0][i1],
                                                self.marker_locations[1][i1],
                                                self.marker_locations[0][i2],
                                                self.marker_locations[1][i2],
                                            );
                                            // evaluate pairwise distance of candidates j1 and j2
                                            cand_pair_dist = self.dist_l2(
                                                self.marker_candidates[0][j1u],
                                                self.marker_candidates[1][j1u],
                                                self.marker_candidates[0][j2u],
                                                self.marker_candidates[1][j2u],
                                            );
                                            dist_diff =
                                                (mrkr_pair_dist - cand_pair_dist).abs() + 1e-9;
                                            if dist_diff <= self.dist_diff_intcpt {
                                                potential_value =
                                                    1.0 - dist_diff / self.dist_diff_intcpt;
                                            } else {
                                                potential_value = 1e-10;
                                            }
                                        }

                                        // using new distance constraint
                                        if (self.pair_po_choice == 6)
                                            | (self.pair_po_choice == 7)
                                            | (self.pair_po_choice == 8)
                                        {
                                            // evaluate norm of of vector (A->B) - (1->2) of markers i1 and i2
                                            x1 = self.marker_locations2[0][i1]
                                                - self.marker_candidates[0][j1u];
                                            x2 = self.marker_locations2[0][i2]
                                                - self.marker_candidates[0][j2u];
                                            y1 = self.marker_locations2[1][i1]
                                                - self.marker_candidates[1][j1u];
                                            y2 = self.marker_locations2[1][i2]
                                                - self.marker_candidates[1][j2u];

                                            dist_vec_norm = self.dist_l2(x1, y1, x2, y2);
                                            // `mrkr_dist_i1_i2` is computed and never used
                                            let _mrkr_dist_i1_i2 = self.dist_l2(
                                                self.marker_locations2[0][i1],
                                                self.marker_locations2[1][i1],
                                                self.marker_locations2[0][i2],
                                                self.marker_locations2[1][i2],
                                            );
                                            rolloff_fac = self.dist_diff_intcpt;
                                            let r = dist_vec_norm / rolloff_fac;
                                            potential_value = (-(r * r)).exp();
                                            // multiply by exponential rolloffs...
                                            if self.pair_po_choice == 8 {
                                                if j1 >= ncandidates {
                                                    cerr(&format!("ERROR:J1= {j1}\n"));
                                                }
                                                if j2 >= ncandidates {
                                                    cerr(&format!("ERROR:J2= {j2}\n"));
                                                }
                                                potential_value = potential_value
                                                    * exp_mrkr_cand_dists[i1][j1u]
                                                    * exp_mrkr_cand_dists[i2][j2u];
                                            }
                                        }

                                        // FARSHID 0428- simplifying pairwise potentials
                                        if self.pair_po_choice == 1 {
                                            // For each remaining marker...
                                            for k in 0..nm {
                                                if k != i1 && k != i2 {
                                                    let mut max_dot_i_j_k_l = 0.0f64;
                                                    // .... And For each remaining candidate...
                                                    for l in 0..nc {
                                                        if l != j1u && l != j2u {
                                                            // calculate dot products, take max
                                                            norm_i_j_k_l_1 =
                                                                norm_prods_i1[j1u][k][l];
                                                            norm_i_j_k_l =
                                                                norm_i1j1_i2j2 * norm_i_j_k_l_1;
                                                            if norm_i_j_k_l > max_dot_i_j_k_l {
                                                                max_dot_i_j_k_l = norm_i_j_k_l;
                                                            }
                                                        }
                                                    }

                                                    // update potential accumulation for current marker-candidate pair
                                                    potential_value += max_dot_i_j_k_l;
                                                } else {
                                                    // k==i1 || k == i2
                                                    potential_value = 1e-10;
                                                }
                                            }
                                        }
                                    } else {
                                        // j1 == j2
                                        potential_value = 1e-10;
                                    }

                                    // Multiply final result by dot product of <i1,j1> and <i2,j2>
                                    if (self.pair_po_choice == 3) || (self.pair_po_choice == 6) {
                                        potential_value *= norm_i1j1_i2j2;
                                    }

                                    // uniform
                                    if self.pair_po_choice == 5 {
                                        potential_value = 1.0;
                                    }
                                }
                                // Garbage Can
                                else {
                                    potential_value = self.gcan_potential;
                                }

                                if debug_pair {
                                    cout(&format!(" {}, ", ostream_double(potential_value)));
                                }

                                // put lower limit on potentials
                                if potential_value < 1e-10 {
                                    potential_value = 1e-10;
                                }
                                // Added scale factor for pairwise potentials (nominally 1)
                                potential_value *= self.ppot_scale_factor;
                                pairwise_meas
                                    .set(jj1 - 1 + cvec[0] as usize * (jj2 - 1), potential_value);

                                // DEBUG potentials
                                if self.debug_initial_beliefs != 0 {
                                    marg_sing_pots[i1][j1 as usize] += potential_value;
                                    marg_sing_pots[i2][j2 as usize] += potential_value;
                                }
                            }
                            // OTHERWISE LOCK values down
                            else {
                                if (self.lock_pot[i1] == 1)
                                    && (self.locked_pot_val[i1] == j1)
                                    && (self.lock_pot[i2] == 1)
                                    && (self.locked_pot_val[i2] == j2)
                                {
                                    potential_value = 1.0;
                                } else {
                                    potential_value = 1e-10;
                                }
                                potential_value *= self.ppot_scale_factor;
                                pairwise_meas
                                    .set(jj1 - 1 + cvec[0] as usize * (jj2 - 1), potential_value);
                            }

                            // update number of candidate pairs recorded (the
                            // source's lower limit here is on a value it no
                            // longer uses)
                            num_cand_pairs += 1;
                        }
                        if debug_pair {
                            cout(";\n");
                        }
                    }

                    if num_cand_pairs == 0 {
                        cerr(&format!("WARNING: mrkr {i1} - mrkr {i2}: NO CANDIDATES\n"));
                    }

                    self.lb_graph.add_clique(&cl, &pairwise_meas);
                    ind += 1;
                }
            }
            if self.verbose != 0 {
                cerr(&format!(
                    "{num_entries} Pairwise Potentials for marker  {i1} ...\n"
                ));
            }
        }

        // DEBUG potentials -output the marginalized pairwise potentials
        if self.debug_initial_beliefs != 0 {
            cout("MARGINALIZED PAIRWISE POTENTIALS ");
            for i1 in 0..nm {
                for j1 in 0..nm.min(nc + 1) {
                    cout(&format!("{} ", ostream_double(marg_sing_pots[i1][j1])));
                }
                cout(";\n ");
            }
        }

        if self.verbose != 0 {
            cerr(&format!(
                "Completed Calculation of  {ind}  Pairwise potentials\n"
            ));
        }
    }
}
