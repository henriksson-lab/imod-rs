//! Translation of `IMOD/flib/image/rotmatwarp.f90`.
//!
//! `rotmatwarp.f` - contains a common module for Rotatevol, Matchvol, and
//! Warpvol.
//!
//! The Fortran `module rotmatwarp` is a set of process-global variables that
//! the programs and the subroutines of `rotmatwarpsubs.f90` share by `use`.
//! Here it is one struct, [`RotMatWarp`], that the program owns and passes by
//! `&mut` to each of those subroutines.  The `equivalence`d scalar names
//! (`nxIn`, `cxOut`, `nxChunk`, ...) are the array elements they alias and
//! are written as those elements.  Fields keep the module's declaration
//! order.  The allocatable arrays keep their Fortran shapes in column-major
//! order; each carries its dimensions beside it where the shape is not
//! already a field.

/// `parameter (LMCUBE = 2500)` (`rotmatwarp.f90:8`).
pub const LMCUBE: i32 = 2500;

/// `module rotmatwarp` (`rotmatwarp.f90:6`).
pub struct RotMatWarp {
    /// `inputDim(3)`.
    pub input_dim: [i32; 3],
    /// `idimOut(3)`.
    pub idim_out: [i32; 3],
    /// `needScratch(4)`.
    pub need_scratch: [bool; 4],
    /// `chunkedHDF/.false./`.
    pub chunked_hdf: bool,
    /// `memoryLim/4096/`.
    pub memory_lim: i32,
    /// `arraySize8`.
    pub array_size8: f64,
    /// `iVerbose/0/`.
    pub i_verbose: i32,
    /// `maxZout/64/`.
    pub max_zout: i32,
    /// `nxyzIn(3)`, equivalenced to `nxIn, nyIn, nzin`.
    pub nxyz_in: [i32; 3],
    /// `nxyzOut(3)`, equivalenced to `nxOut, nyOut, nzOut`.
    pub nxyz_out: [i32; 3],
    /// `nxyzChunk(3)`, equivalenced to `nxChunk/-1/, nyChunk/-1/,
    /// nzChunk/-1/`.
    pub nxyz_chunk: [i32; 3],
    /// `cxyzIn(3)`, equivalenced to `cxIn, cyIn, czIn`.
    pub cxyz_in: [f32; 3],
    /// `cxyzOut(3)`, equivalenced to `cxOut, cyOut, czOut`.
    pub cxyz_out: [f32; 3],
    /// `aInv(3,3)`, column major: `aInv(i,j)` is `a_inv[(i-1) + 3*(j-1)]`.
    pub a_inv: [f32; 9],
    /// `title(20)`: the 80-byte label, zero until a program writes it.
    pub title: [u8; 80],
    /// `dmeanIn`.
    pub dmean_in: f32,
    /// `nCubes(3)`.
    pub n_cubes: [i32; 3],
    /// `mode`.
    pub mode: i32,
    /// `ioutXaxis`.
    pub iout_xaxis: i32,
    /// `ioutYaxis`.
    pub iout_yaxis: i32,
    /// `ioutZaxis`.
    pub iout_zaxis: i32,
    /// `limInner`.
    pub lim_inner: i32,
    /// `limOuter`.
    pub lim_outer: i32,
    /// `idirXaxis`.
    pub idir_xaxis: i32,
    /// `nxyzCube(3,LMCUBE)`: `nxyzCube(i,j)` is `nxyz_cube[j-1][i-1]`.
    pub nxyz_cube: Vec<[i32; 3]>,
    /// `ixyzCube(3,LMCUBE)`: `ixyzCube(i,j)` is `ixyz_cube[j-1][i-1]`.
    pub ixyz_cube: Vec<[i32; 3]>,
    /// `izInFile(:,:,:)`, allocated `(nCubes(1), nCubes(2), nxyzScr(3))`.
    pub iz_in_file: Vec<i32>,
    /// The allocated shape of `izInFile`.
    pub iz_in_file_dim: [i32; 3],
    /// `ifile(:,:)` (`integer*2`), allocated `(nCubes(1), nCubes(2))`.
    pub ifile: Vec<i16>,
    /// `array(:,:,:)`, allocated `(inputDim(1), inputDim(2), inputDim(3) +
    /// nzExtra)`.
    pub array: Vec<f32>,
    /// `brray(:)`, allocated `idimOut(1) * idimOut(2) * maxZout`.
    pub brray: Vec<f32>,
}

impl Default for RotMatWarp {
    /// The module's initial state: the `/value/` initialisers, zero for the
    /// rest (module variables live in zero-initialised static storage), and
    /// the allocatables unallocated.
    fn default() -> Self {
        Self {
            input_dim: [0; 3],
            idim_out: [0; 3],
            need_scratch: [false; 4],
            chunked_hdf: false,
            memory_lim: 4096,
            array_size8: 0.,
            i_verbose: 0,
            max_zout: 64,
            nxyz_in: [0; 3],
            nxyz_out: [0; 3],
            nxyz_chunk: [-1; 3],
            cxyz_in: [0.; 3],
            cxyz_out: [0.; 3],
            a_inv: [0.; 9],
            title: [0; 80],
            dmean_in: 0.,
            n_cubes: [0; 3],
            mode: 0,
            iout_xaxis: 0,
            iout_yaxis: 0,
            iout_zaxis: 0,
            lim_inner: 0,
            lim_outer: 0,
            idir_xaxis: 0,
            nxyz_cube: vec![[0; 3]; LMCUBE as usize],
            ixyz_cube: vec![[0; 3]; LMCUBE as usize],
            iz_in_file: Vec::new(),
            iz_in_file_dim: [0; 3],
            ifile: Vec::new(),
            array: Vec::new(),
            brray: Vec::new(),
        }
    }
}
