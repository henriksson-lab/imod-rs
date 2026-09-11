//! `IMOD/Etomo/src/etomo/type/ProcessName.java`.
//!
//! Java's typesafe-enum pattern is mirrored as a struct plus one `pub const` per
//! `public static final` singleton.  The class is designed to be inherited (see
//! `constructInstance`), so it cannot be a Rust enum; instances are compared by
//! `name`, which the class guarantees is unique (a repeated name is refused by
//! `constructInstance`), where the source compares by identity.
#![allow(dead_code)]
#![allow(non_upper_case_globals)]

use super::axis_id::AxisID;
use super::status::Status;

/// Java `DatasetFiles.COMSCRIPT_EXT` (DatasetFiles.java:33).
use crate::imod::etomo::util::dataset_files::COMSCRIPT_EXT;

// known process names
const xcorr: &str = "xcorr";
const prenewst: &str = "prenewst";
const track: &str = "track";
const align: &str = "align";
const newst: &str = "newst";
const tilt: &str = "tilt";
const mtffilter: &str = "mtffilter";
const solvematchshift: &str = "solvematchshift";
const solvematchmod: &str = "solvematchmod";
const patchcorr: &str = "patchcorr";
const matchorwarp: &str = "matchorwarp";
const tomopitch: &str = "tomopitch";
const sample: &str = "sample";
const combine: &str = "combine";
const matchvol1: &str = "matchvol1";
const volcombine: &str = "volcombine";
const preblend: &str = "preblend";
const blend: &str = "blend";
const undistort: &str = "undistort";
const solvematch: &str = "solvematch";
const startjoin: &str = "startjoin";
const processchunks: &str = "processchunks";
const tomosnapshot: &str = "tomosnapshot";
const transferfid: &str = "transferfid";
const clipflipyz: &str = "clipflipyz";
const finishjoin: &str = "finishjoin";
const makejoincom: &str = "makejoincom";
const trimvol: &str = "trimvol";
const squeezevol: &str = "squeezevol";
const archiveorig: &str = "archiveorig";
const splittilt: &str = "splittilt";
const splitcombine: &str = "splitcombine";
const extractmagrad: &str = "extractmagrad";
const extracttilts: &str = "extracttilts";
const extractpieces: &str = "extractpieces";
const xfalign: &str = "xfalign";
const xfjointomo: &str = "xfjointomo";
const xftoxg: &str = "xftoxg";
const xfmodel: &str = "xfmodel";
const remapmodel: &str = "remapmodel";
const peetParser: &str = "prmParser";
const anisotropicDiffusion: &str = "nad_eed_3d";
const chunksetup: &str = "chunksetup";
const ctfPlotter: &str = "ctfplotter";
const ctfCorrection: &str = "ctfcorrection";
const splitCorrection: &str = "splitcorrection";
const golderaser: &str = "golderaser";
const clip: &str = "clip";
const runraptor: &str = "runraptor";
const flattenwarp: &str = "flattenwarp";
const flatten: &str = "flatten";
const newst_3dfind: &str = "newst_3dfind";
const blend_3dfind: &str = "blend_3dfind";
const tilt_3dfind: &str = "tilt_3dfind";
const findbeads3d: &str = "findbeads3d";
const tilt_3dfind_reproject: &str = "tilt_3dfind_reproject";
const midas: &str = "midas";
const xcorr_pt: &str = "xcorr_pt";
const prochunks_csh: &str = "prochunks.csh";

/// Java `ProcessName`.  Represents processes of various types.  Only one process with
/// a given name can exist - and case does not count.  Processes with different cases
/// and the same characters will collide.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProcessName {
    /// Java field `name`.
    name: Option<&'static str>,
    /// Java public field `resumable`.
    pub resumable: bool,
}

/// Java `INSTANCE_MAP`, the map every constructed instance registers itself in.  Key
/// is lower case version of name.  The built-in singletons below register themselves
/// in declaration order at class-initialization time; that fixed part of the map is
/// `BUILT_IN_INSTANCES` and the part `constructInstance` adds at run time is
/// `EXTENDED_INSTANCES`.
static EXTENDED_INSTANCES: std::sync::Mutex<Vec<ProcessName>> = std::sync::Mutex::new(Vec::new());

impl ProcessName {
    /// Java `ERASER`, constructed from `"eraser"`.
    pub const ERASER: ProcessName = ProcessName {
        name: Some("eraser"),
        resumable: false,
    };
    /// Java `XCORR`, constructed from `xcorr`.
    pub const XCORR: ProcessName = ProcessName {
        name: Some("xcorr"),
        resumable: false,
    };
    /// Java `PRENEWST`, constructed from `prenewst`.
    pub const PRENEWST: ProcessName = ProcessName {
        name: Some("prenewst"),
        resumable: false,
    };
    /// Java `TRACK`, constructed from `track`.
    pub const TRACK: ProcessName = ProcessName {
        name: Some("track"),
        resumable: false,
    };
    /// Java `ALIGN`, constructed from `align`.
    pub const ALIGN: ProcessName = ProcessName {
        name: Some("align"),
        resumable: false,
    };
    /// Java `NEWST`, constructed from `newst`.
    pub const NEWST: ProcessName = ProcessName {
        name: Some("newst"),
        resumable: false,
    };
    /// Java `TILT`, constructed from `tilt`.
    pub const TILT: ProcessName = ProcessName {
        name: Some("tilt"),
        resumable: true,
    };
    /// Java `MTFFILTER`, constructed from `mtffilter`.
    pub const MTFFILTER: ProcessName = ProcessName {
        name: Some("mtffilter"),
        resumable: false,
    };
    /// Java `SOLVEMATCHSHIFT`, constructed from `solvematchshift`.
    pub const SOLVEMATCHSHIFT: ProcessName = ProcessName {
        name: Some("solvematchshift"),
        resumable: false,
    };
    /// Java `SOLVEMATCHMOD`, constructed from `solvematchmod`.
    pub const SOLVEMATCHMOD: ProcessName = ProcessName {
        name: Some("solvematchmod"),
        resumable: false,
    };
    /// Java `PATCHCORR`, constructed from `patchcorr`.
    pub const PATCHCORR: ProcessName = ProcessName {
        name: Some("patchcorr"),
        resumable: false,
    };
    /// Java `MATCHORWARP`, constructed from `matchorwarp`.
    pub const MATCHORWARP: ProcessName = ProcessName {
        name: Some("matchorwarp"),
        resumable: false,
    };
    /// Java `TOMOPITCH`, constructed from `tomopitch`.
    pub const TOMOPITCH: ProcessName = ProcessName {
        name: Some("tomopitch"),
        resumable: false,
    };
    /// Java `SAMPLE`, constructed from `sample`.
    pub const SAMPLE: ProcessName = ProcessName {
        name: Some("sample"),
        resumable: false,
    };
    /// Java `COMBINE`, constructed from `combine`.
    pub const COMBINE: ProcessName = ProcessName {
        name: Some("combine"),
        resumable: false,
    };
    /// Java `MATCHVOL1`, constructed from `matchvol1`.
    pub const MATCHVOL1: ProcessName = ProcessName {
        name: Some("matchvol1"),
        resumable: false,
    };
    /// Java `VOLCOMBINE`, constructed from `volcombine`.
    pub const VOLCOMBINE: ProcessName = ProcessName {
        name: Some("volcombine"),
        resumable: false,
    };
    /// Java `PREBLEND`, constructed from `preblend`.
    pub const PREBLEND: ProcessName = ProcessName {
        name: Some("preblend"),
        resumable: false,
    };
    /// Java `BLEND`, constructed from `blend`.
    pub const BLEND: ProcessName = ProcessName {
        name: Some("blend"),
        resumable: false,
    };
    /// Java `UNDISTORT`, constructed from `undistort`.
    pub const UNDISTORT: ProcessName = ProcessName {
        name: Some("undistort"),
        resumable: false,
    };
    /// Java `SOLVEMATCH`, constructed from `solvematch`.
    pub const SOLVEMATCH: ProcessName = ProcessName {
        name: Some("solvematch"),
        resumable: false,
    };
    /// Java `STARTJOIN`, constructed from `startjoin`.
    pub const STARTJOIN: ProcessName = ProcessName {
        name: Some("startjoin"),
        resumable: false,
    };
    /// Java `PROCESSCHUNKS`, constructed from `processchunks`.
    pub const PROCESSCHUNKS: ProcessName = ProcessName {
        name: Some("processchunks"),
        resumable: true,
    };
    /// Java `TOMOSNAPSHOT`, constructed from `tomosnapshot`.
    pub const TOMOSNAPSHOT: ProcessName = ProcessName {
        name: Some("tomosnapshot"),
        resumable: false,
    };
    /// Java `TRANSFERFID`, constructed from `transferfid`.
    pub const TRANSFERFID: ProcessName = ProcessName {
        name: Some("transferfid"),
        resumable: false,
    };
    /// Java `CLIPFLIPYZ`, constructed from `clipflipyz`.
    pub const CLIPFLIPYZ: ProcessName = ProcessName {
        name: Some("clipflipyz"),
        resumable: false,
    };
    /// Java `FINISHJOIN`, constructed from `finishjoin`.
    pub const FINISHJOIN: ProcessName = ProcessName {
        name: Some("finishjoin"),
        resumable: false,
    };
    /// Java `MAKEJOINCOM`, constructed from `makejoincom`.
    pub const MAKEJOINCOM: ProcessName = ProcessName {
        name: Some("makejoincom"),
        resumable: false,
    };
    /// Java `TRIMVOL`, constructed from `trimvol`.
    pub const TRIMVOL: ProcessName = ProcessName {
        name: Some("trimvol"),
        resumable: false,
    };
    /// Java `SQUEEZEVOL`, constructed from `squeezevol`.
    pub const SQUEEZEVOL: ProcessName = ProcessName {
        name: Some("squeezevol"),
        resumable: false,
    };
    /// Java `ARCHIVEORIG`, constructed from `archiveorig`.
    pub const ARCHIVEORIG: ProcessName = ProcessName {
        name: Some("archiveorig"),
        resumable: false,
    };
    /// Java `SPLITTILT`, constructed from `splittilt`.
    pub const SPLITTILT: ProcessName = ProcessName {
        name: Some("splittilt"),
        resumable: false,
    };
    /// Java `SPLITCOMBINE`, constructed from `splitcombine`.
    pub const SPLITCOMBINE: ProcessName = ProcessName {
        name: Some("splitcombine"),
        resumable: false,
    };
    /// Java `EXTRACTMAGRAD`, constructed from `extractmagrad`.
    pub const EXTRACTMAGRAD: ProcessName = ProcessName {
        name: Some("extractmagrad"),
        resumable: false,
    };
    /// Java `EXTRACTTILTS`, constructed from `extracttilts`.
    pub const EXTRACTTILTS: ProcessName = ProcessName {
        name: Some("extracttilts"),
        resumable: false,
    };
    /// Java `EXTRACTPIECES`, constructed from `extractpieces`.
    pub const EXTRACTPIECES: ProcessName = ProcessName {
        name: Some("extractpieces"),
        resumable: false,
    };
    /// Java `XFALIGN`, constructed from `xfalign`.
    pub const XFALIGN: ProcessName = ProcessName {
        name: Some("xfalign"),
        resumable: false,
    };
    /// Java `XFJOINTOMO`, constructed from `xfjointomo`.
    pub const XFJOINTOMO: ProcessName = ProcessName {
        name: Some("xfjointomo"),
        resumable: false,
    };
    /// Java `XFTOXG`, constructed from `xftoxg`.
    pub const XFTOXG: ProcessName = ProcessName {
        name: Some("xftoxg"),
        resumable: false,
    };
    /// Java `XFMODEL`, constructed from `xfmodel`.
    pub const XFMODEL: ProcessName = ProcessName {
        name: Some("xfmodel"),
        resumable: false,
    };
    /// Java `REMAPMODEL`, constructed from `remapmodel`.
    pub const REMAPMODEL: ProcessName = ProcessName {
        name: Some("remapmodel"),
        resumable: false,
    };
    /// Java `PEET_PARSER`, constructed from `peetParser`.
    pub const PEET_PARSER: ProcessName = ProcessName {
        name: Some("prmParser"),
        resumable: false,
    };
    /// Java `AVERAGE_ALL`, constructed from `"averageAll"`.
    pub const AVERAGE_ALL: ProcessName = ProcessName {
        name: Some("averageAll"),
        resumable: false,
    };
    /// Java `ANISOTROPIC_DIFFUSION`, constructed from `anisotropicDiffusion`.
    pub const ANISOTROPIC_DIFFUSION: ProcessName = ProcessName {
        name: Some("nad_eed_3d"),
        resumable: false,
    };
    /// Java `CHUNKSETUP`, constructed from `chunksetup`.
    pub const CHUNKSETUP: ProcessName = ProcessName {
        name: Some("chunksetup"),
        resumable: false,
    };
    /// Java `CTF_PLOTTER`, constructed from `ctfPlotter`.
    pub const CTF_PLOTTER: ProcessName = ProcessName {
        name: Some("ctfplotter"),
        resumable: false,
    };
    /// Java `CTF_CORRECTION`, constructed from `ctfCorrection`.
    pub const CTF_CORRECTION: ProcessName = ProcessName {
        name: Some("ctfcorrection"),
        resumable: false,
    };
    /// Java `SPLIT_CORRECTION`, constructed from `splitCorrection`.
    pub const SPLIT_CORRECTION: ProcessName = ProcessName {
        name: Some("splitcorrection"),
        resumable: false,
    };
    /// Java `GOLD_ERASER`, constructed from `golderaser`.
    pub const GOLD_ERASER: ProcessName = ProcessName {
        name: Some("golderaser"),
        resumable: false,
    };
    /// Java `CLIP`, constructed from `clip`.
    pub const CLIP: ProcessName = ProcessName {
        name: Some("clip"),
        resumable: false,
    };
    /// Java `RUNRAPTOR`, constructed from `runraptor`.
    pub const RUNRAPTOR: ProcessName = ProcessName {
        name: Some("runraptor"),
        resumable: false,
    };
    /// Java `FLATTEN_WARP`, constructed from `flattenwarp`.
    pub const FLATTEN_WARP: ProcessName = ProcessName {
        name: Some("flattenwarp"),
        resumable: false,
    };
    /// Java `FLATTEN`, constructed from `flatten`.
    pub const FLATTEN: ProcessName = ProcessName {
        name: Some("flatten"),
        resumable: false,
    };
    /// Java `GPU_TILT_TEST`, constructed from `"gputilttest"`.
    pub const GPU_TILT_TEST: ProcessName = ProcessName {
        name: Some("gputilttest"),
        resumable: false,
    };
    /// Java `NEWST_3D_FIND`, constructed from `newst_3dfind`.
    pub const NEWST_3D_FIND: ProcessName = ProcessName {
        name: Some("newst_3dfind"),
        resumable: false,
    };
    /// Java `BLEND_3D_FIND`, constructed from `blend_3dfind`.
    pub const BLEND_3D_FIND: ProcessName = ProcessName {
        name: Some("blend_3dfind"),
        resumable: false,
    };
    /// Java `TILT_3D_FIND`, constructed from `tilt_3dfind`.
    pub const TILT_3D_FIND: ProcessName = ProcessName {
        name: Some("tilt_3dfind"),
        resumable: false,
    };
    /// Java `FIND_BEADS_3D`, constructed from `findbeads3d`.
    pub const FIND_BEADS_3D: ProcessName = ProcessName {
        name: Some("findbeads3d"),
        resumable: false,
    };
    /// Java `TILT_3D_FIND_REPROJECT`, constructed from `tilt_3dfind_reproject`.
    pub const TILT_3D_FIND_REPROJECT: ProcessName = ProcessName {
        name: Some("tilt_3dfind_reproject"),
        resumable: false,
    };
    /// Java `MIDAS`, constructed from `midas`.
    pub const MIDAS: ProcessName = ProcessName {
        name: Some("midas"),
        resumable: false,
    };
    /// Java `XCORR_PT`, constructed from `xcorr_pt`.
    pub const XCORR_PT: ProcessName = ProcessName {
        name: Some("xcorr_pt"),
        resumable: false,
    };
    /// Java `PROCHUNKS_CSH`, constructed from `prochunks_csh`.
    pub const PROCHUNKS_CSH: ProcessName = ProcessName {
        name: Some("prochunks.csh"),
        resumable: true,
    };
    /// Java `SIRTSETUP`, constructed from `"sirtsetup"`.
    pub const SIRTSETUP: ProcessName = ProcessName {
        name: Some("sirtsetup"),
        resumable: false,
    };
    /// Java `TILT_SIRT`, constructed from `"tilt_sirt"`.
    pub const TILT_SIRT: ProcessName = ProcessName {
        name: Some("tilt_sirt"),
        resumable: false,
    };
    /// Java `AUTOFIDSEED`, constructed from `"autofidseed"`.
    pub const AUTOFIDSEED: ProcessName = ProcessName {
        name: Some("autofidseed"),
        resumable: false,
    };
    /// Java `BATCHRUNTOMO`, constructed from `"batchruntomo"`.
    pub const BATCHRUNTOMO: ProcessName = ProcessName {
        name: Some("batchruntomo"),
        resumable: false,
    };
    /// Java `MAKECOMFILE`, constructed from `"makecomfile"`.
    pub const MAKECOMFILE: ProcessName = ProcessName {
        name: Some("makecomfile"),
        resumable: false,
    };
    /// Java `COPYTOMOCOMS`, constructed from `"copytomocoms"`.
    pub const COPYTOMOCOMS: ProcessName = ProcessName {
        name: Some("copytomocoms"),
        resumable: false,
    };
    /// Java `TOMODATAPLOTS`, constructed from `"tomodataplots"`.
    pub const TOMODATAPLOTS: ProcessName = ProcessName {
        name: Some("tomodataplots"),
        resumable: false,
    };
    /// Java `DUALVOLMATCH`, constructed from `"dualvolmatch"`.
    pub const DUALVOLMATCH: ProcessName = ProcessName {
        name: Some("dualvolmatch"),
        resumable: false,
    };
    /// Java `SETUPCOMBINE`, constructed from `"setupcombine"`.
    pub const SETUPCOMBINE: ProcessName = ProcessName {
        name: Some("setupcombine"),
        resumable: false,
    };
    /// Java `RESTRICTALIGN`, constructed from `"restrictalign"`.
    pub const RESTRICTALIGN: ProcessName = ProcessName {
        name: Some("restrictalign"),
        resumable: false,
    };
    /// Java `CRYO_POSITION`, constructed from `"cryoposition"`.
    pub const CRYO_POSITION: ProcessName = ProcessName {
        name: Some("cryoposition"),
        resumable: false,
    };
    /// Java `FIND_SECTION`, constructed from `"findsection"`.
    pub const FIND_SECTION: ProcessName = ProcessName {
        name: Some("findsection"),
        resumable: false,
    };
    /// Java `EXCLUDE_VIEWS`, constructed from `"excludeviews"`.
    pub const EXCLUDE_VIEWS: ProcessName = ProcessName {
        name: Some("excludeviews"),
        resumable: false,
    };
    /// Java `FIND_SECTION_POS`, constructed from `"findsection_pos"`.
    pub const FIND_SECTION_POS: ProcessName = ProcessName {
        name: Some("findsection_pos"),
        resumable: false,
    };
    /// Java `FIND_SECTION_LIM`, constructed from `"findsection_lim"`.
    pub const FIND_SECTION_LIM: ProcessName = ProcessName {
        name: Some("findsection_lim"),
        resumable: false,
    };
    /// Java `MULTIFILT_SETUP`, constructed from `"multifiltsetup"`.
    pub const MULTIFILT_SETUP: ProcessName = ProcessName {
        name: Some("multifiltsetup"),
        resumable: false,
    };
    /// Java `TILT_MULTIFILT`, constructed from `"tilt_mulfil"`.
    pub const TILT_MULTIFILT: ProcessName = ProcessName {
        name: Some("tilt_mulfil"),
        resumable: false,
    };
    /// Java `CCDERASER`, constructed from `"ccderaser"`.
    pub const CCDERASER: ProcessName = ProcessName {
        name: Some("ccderaser"),
        resumable: false,
    };
    /// Java `JOIN_WARP_2_MODEL`, constructed from `"joinwarp2model"`.
    pub const JOIN_WARP_2_MODEL: ProcessName = ProcessName {
        name: Some("joinwarp2model"),
        resumable: false,
    };
    /// Java `CTF_3D_SETUP`, constructed from `"ctf3dsetup"`.
    pub const CTF_3D_SETUP: ProcessName = ProcessName {
        name: Some("ctf3dsetup"),
        resumable: false,
    };
    /// Java `ALIGN_FRAMES`, constructed from `"alignframes"`.
    pub const ALIGN_FRAMES: ProcessName = ProcessName {
        name: Some("alignframes"),
        resumable: false,
    };
    /// Java `CTF_3D`, constructed from `"ctf3d"`.
    pub const CTF_3D: ProcessName = ProcessName {
        name: Some("ctf3d"),
        resumable: false,
    };
    /// Java `SPLIT_BATCH`, constructed from `"splitbatch"`.
    pub const SPLIT_BATCH: ProcessName = ProcessName {
        name: Some("splitbatch"),
        resumable: false,
    };
    /// Java `SORT_TILT_FRAMES`, constructed from `"sorttiltframes"`.
    pub const SORT_TILT_FRAMES: ProcessName = ProcessName {
        name: Some("sorttiltframes"),
        resumable: false,
    };
    /// Java `SUBTOMO_SETUP`, constructed from `"subtomosetup"`.
    pub const SUBTOMO_SETUP: ProcessName = ProcessName {
        name: Some("subtomosetup"),
        resumable: false,
    };
    /// Java `TILT_ALIGN`, constructed from `"tiltalign"`.
    pub const TILT_ALIGN: ProcessName = ProcessName {
        name: Some("tiltalign"),
        resumable: false,
    };
    /// Java `ALT_TOMO_SETUP`, constructed from `"alttomosetup"`.
    pub const ALT_TOMO_SETUP: ProcessName = ProcessName {
        name: Some("alttomosetup"),
        resumable: false,
    };
    /// Java `ALT_TOMO_PROCESS_CHUNKS`, constructed from `"alttomo"`.
    pub const ALT_TOMO_PROCESS_CHUNKS: ProcessName = ProcessName {
        name: Some("alttomo"),
        resumable: false,
    };
    /// Java `REDUCE_FILT_VOL`, constructed from `"reducefiltvol"`.
    pub const REDUCE_FILT_VOL: ProcessName = ProcessName {
        name: Some("reducefiltvol"),
        resumable: false,
    };
    /// Java `MAN`, constructed from `"man"`.
    pub const MAN: ProcessName = ProcessName {
        name: Some("man"),
        resumable: false,
    };
    /// Java `TILT_XCORR`, constructed from `"tiltxcorr"`.
    pub const TILT_XCORR: ProcessName = ProcessName {
        name: Some("tiltxcorr"),
        resumable: false,
    };
    /// Java `SERIES_WATCHER`, constructed from `"serieswatcher"`.
    pub const SERIES_WATCHER: ProcessName = ProcessName {
        name: Some("serieswatcher"),
        resumable: false,
    };

    /// Java `ProcessName()`.  Creates an empty instance which is not added to the
    /// collection.  This constructor should not be used, and exists only to allow the
    /// class to be inherited.
    pub fn new_empty() -> ProcessName {
        ProcessName::new(None, false)
    }

    /// Java `ProcessName(String, boolean)`.
    fn new(name: Option<&'static str>, resumable: bool) -> ProcessName {
        let instance = ProcessName { name, resumable };
        if let Some(name) = name {
            let mut extended = EXTENDED_INSTANCES.lock().unwrap();
            // Java checks containsKey(name) - the un-lowercased name - but puts under
            // name.toLowerCase().
            let contains = BUILT_IN_INSTANCES
                .iter()
                .chain(extended.iter())
                .any(|instance| instance.name.unwrap() == name);
            if !contains {
                extended.push(instance);
            }
        }
        instance
    }

    /// Java `constructInstance` (String overload).  If an instance with this name
    /// doesn't already exist, constructs a new process name instance, otherwise
    /// returns the existing one.  The same name with a different case is not allowed.
    pub fn construct_instance(name: &'static str) -> ProcessName {
        let process_name = ProcessName::get_instance(Some(name));
        if let Some(process_name) = process_name {
            // Cannot replace an existing process name instance.
            eprintln!(
                "Warning: Failed attempt to replace a process name:{}.  Names that only have a different case will collide.",
                name
            );
            return process_name;
        }
        ProcessName::new(Some(name), false)
    }

    /// Java `constructInstance` (String, boolean overload).  If an instance with this
    /// name doesn't already exist, constructs a new process name instance, otherwise
    /// returns the existing one.  The same name with a different case is not allowed.
    fn construct_instance_resumable(name: &'static str, resumable: bool) -> ProcessName {
        let process_name = ProcessName::get_instance(Some(name));
        if let Some(process_name) = process_name {
            // Cannot replace an existing process name instance.
            eprintln!(
                "Warning: Failed attempt to replace an etomo process name:{}.  Names that only have a different case will collide.",
                name
            );
            return process_name;
        }
        ProcessName::new(Some(name), resumable)
    }

    /// Java `equals` (String overload).
    pub fn equals(&self, name: &str) -> bool {
        if ProcessName::get_instance(Some(name)) == Some(*self) {
            return true;
        }
        false
    }

    /// Java `equals` (String, AxisID overload).  Equals if command equals process
    /// name, or if command equals process name + axis extension.
    pub fn equals_with_axis(&self, command: &str, axis_id: AxisID) -> bool {
        if Some(command) == self.name {
            return true;
        }
        if command == format!("{}{}", self.name.unwrap_or(""), axis_id.get_extension()) {
            return true;
        }
        false
    }

    /// Java `getInstance` (String, AxisID overload).  Turn name into an instance of
    /// ProcessName.  Possibilities are checked in this order:
    /// 1. Name is a process name.
    /// 2. Name is a process name plus the axis extension.
    /// 3. Name is a file name where everything but the extension (starts with last
    ///    '.') is a process name.
    /// 4. Name is a file name where everything but the extension (starts with last
    ///    '.') is a process name plus the axis extension.
    pub fn get_instance_with_axis(name: &str, axis_id: AxisID) -> Option<ProcessName> {
        let mut name = name;
        // check if name equals process name
        let mut process_name = ProcessName::get_instance(Some(name));
        if process_name.is_some() {
            return process_name;
        }
        // check if name is process name plus axis extension
        if axis_id.get_extension() != "" && name.ends_with(&axis_id.get_extension()) {
            process_name = ProcessName::get_instance(Some(
                &name[0..name.len() - axis_id.get_extension().len()],
            ));
            if process_name.is_some() {
                return process_name;
            }
        }
        let ext_index = name.rfind('.');
        let ext_index = match ext_index {
            None => return None,
            Some(ext_index) => ext_index,
        };
        name = &name[0..ext_index];
        // check if file name equals process name plus file extension
        process_name = ProcessName::get_instance(Some(name));
        if process_name.is_some() {
            return process_name;
        }
        // check if file name is process name plus axis extension plus file extension
        if axis_id.get_extension() != "" && name.ends_with(&axis_id.get_extension()) {
            process_name = ProcessName::get_instance(Some(
                &name[0..name.len() - axis_id.get_extension().len()],
            ));
            if process_name.is_some() {
                return process_name;
            }
        }
        None
    }

    /// Java `getInstance` (String, AxisID, String overload).  If extension matches the
    /// end of name, strip off extension at the end of name before returning a call to
    /// `getInstance(name, axisID)`.
    pub fn get_instance_with_axis_and_extension(
        name: &str,
        axis_id: AxisID,
        extension: Option<&str>,
    ) -> Option<ProcessName> {
        let extension = match extension {
            None => return ProcessName::get_instance_with_axis(name, axis_id),
            Some(extension) if extension.is_empty() => {
                return ProcessName::get_instance_with_axis(name, axis_id);
            }
            Some(extension) => extension,
        };
        let ext_index = name.rfind(extension);
        let ext_index = match ext_index {
            None => return ProcessName::get_instance_with_axis(name, axis_id),
            Some(ext_index) => ext_index,
        };
        ProcessName::get_instance_with_axis(&name[0..ext_index], axis_id)
    }

    /// Java `getInstance` (String overload).  Takes a string representation of an
    /// ProcessName type and returns the correct static object.  The string is case
    /// insensitive.  Null is returned if the string is not one of the known process
    /// names.  All ProcessName instances, including those created in child classes,
    /// can be returned.
    pub fn get_instance(name: Option<&str>) -> Option<ProcessName> {
        let mut name = match name {
            None => return None,
            Some(name) => name,
        };
        if name.ends_with(COMSCRIPT_EXT) {
            let ext_index = name.rfind(COMSCRIPT_EXT).unwrap();
            name = &name[0..ext_index];
        }
        let key = name.to_lowercase();
        let extended = EXTENDED_INSTANCES.lock().unwrap();
        BUILT_IN_INSTANCES
            .iter()
            .chain(extended.iter())
            .rev()
            .find(|instance| instance.name.unwrap().to_lowercase() == key)
            .copied()
    }

    /// Java `getInstance` (File, String overload).
    pub fn get_instance_from_file(
        file: Option<&std::path::Path>,
        exclude_string: &str,
    ) -> Option<ProcessName> {
        let file = match file {
            None => return None,
            Some(file) => file,
        };
        let file_name = file
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        // Java throws StringIndexOutOfBoundsException when excludeString is absent.
        let process_string_buffer =
            file_name[0..file_name.rfind(exclude_string).unwrap()].to_string();
        let process_name;
        if {
            process_name = ProcessName::get_instance(Some(&process_string_buffer));
            process_name.is_some()
        } {
            return process_name;
        }
        if process_string_buffer.ends_with(&AxisID::First.get_extension()) {
            return ProcessName::get_instance(Some(
                &process_string_buffer[process_string_buffer
                    .rfind(&AxisID::First.get_extension())
                    .unwrap()..],
            ));
        }
        if process_string_buffer.ends_with(&AxisID::Second.get_extension()) {
            return ProcessName::get_instance(Some(
                &process_string_buffer[process_string_buffer
                    .rfind(&AxisID::Second.get_extension())
                    .unwrap()..],
            ));
        }
        None
    }

    /// Java `getComscript`.
    pub fn get_comscript(&self, axis_id: AxisID) -> String {
        format!(
            "{}{}{}",
            self.name.unwrap_or(""),
            axis_id.get_extension(),
            COMSCRIPT_EXT
        )
    }

    /// Java `getComscriptArray`.
    pub fn get_comscript_array(&self, axis_id: AxisID) -> Vec<String> {
        vec![format!(
            "{}{}{}",
            self.name.unwrap_or(""),
            axis_id.get_extension(),
            COMSCRIPT_EXT
        )]
    }
}

/// Java `getText`, declared by `Status`.
impl Status for ProcessName {
    fn get_text(&self) -> Option<&'static str> {
        self.name
    }
}

/// Java `toString`.  Returns a string representation of the object.
impl std::fmt::Display for ProcessName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name.unwrap_or(""))
    }
}

/// The built-in half of Java `INSTANCE_MAP`, in class-initialization order.
static BUILT_IN_INSTANCES: &[ProcessName] = &[
    ProcessName::ERASER,
    ProcessName::XCORR,
    ProcessName::PRENEWST,
    ProcessName::TRACK,
    ProcessName::ALIGN,
    ProcessName::NEWST,
    ProcessName::TILT,
    ProcessName::MTFFILTER,
    ProcessName::SOLVEMATCHSHIFT,
    ProcessName::SOLVEMATCHMOD,
    ProcessName::PATCHCORR,
    ProcessName::MATCHORWARP,
    ProcessName::TOMOPITCH,
    ProcessName::SAMPLE,
    ProcessName::COMBINE,
    ProcessName::MATCHVOL1,
    ProcessName::VOLCOMBINE,
    ProcessName::PREBLEND,
    ProcessName::BLEND,
    ProcessName::UNDISTORT,
    ProcessName::SOLVEMATCH,
    ProcessName::STARTJOIN,
    ProcessName::PROCESSCHUNKS,
    ProcessName::TOMOSNAPSHOT,
    ProcessName::TRANSFERFID,
    ProcessName::CLIPFLIPYZ,
    ProcessName::FINISHJOIN,
    ProcessName::MAKEJOINCOM,
    ProcessName::TRIMVOL,
    ProcessName::SQUEEZEVOL,
    ProcessName::ARCHIVEORIG,
    ProcessName::SPLITTILT,
    ProcessName::SPLITCOMBINE,
    ProcessName::EXTRACTMAGRAD,
    ProcessName::EXTRACTTILTS,
    ProcessName::EXTRACTPIECES,
    ProcessName::XFALIGN,
    ProcessName::XFJOINTOMO,
    ProcessName::XFTOXG,
    ProcessName::XFMODEL,
    ProcessName::REMAPMODEL,
    ProcessName::PEET_PARSER,
    ProcessName::AVERAGE_ALL,
    ProcessName::ANISOTROPIC_DIFFUSION,
    ProcessName::CHUNKSETUP,
    ProcessName::CTF_PLOTTER,
    ProcessName::CTF_CORRECTION,
    ProcessName::SPLIT_CORRECTION,
    ProcessName::GOLD_ERASER,
    ProcessName::CLIP,
    ProcessName::RUNRAPTOR,
    ProcessName::FLATTEN_WARP,
    ProcessName::FLATTEN,
    ProcessName::GPU_TILT_TEST,
    ProcessName::NEWST_3D_FIND,
    ProcessName::BLEND_3D_FIND,
    ProcessName::TILT_3D_FIND,
    ProcessName::FIND_BEADS_3D,
    ProcessName::TILT_3D_FIND_REPROJECT,
    ProcessName::MIDAS,
    ProcessName::XCORR_PT,
    ProcessName::PROCHUNKS_CSH,
    ProcessName::SIRTSETUP,
    ProcessName::TILT_SIRT,
    ProcessName::AUTOFIDSEED,
    ProcessName::BATCHRUNTOMO,
    ProcessName::MAKECOMFILE,
    ProcessName::COPYTOMOCOMS,
    ProcessName::TOMODATAPLOTS,
    ProcessName::DUALVOLMATCH,
    ProcessName::SETUPCOMBINE,
    ProcessName::RESTRICTALIGN,
    ProcessName::CRYO_POSITION,
    ProcessName::FIND_SECTION,
    ProcessName::EXCLUDE_VIEWS,
    ProcessName::FIND_SECTION_POS,
    ProcessName::FIND_SECTION_LIM,
    ProcessName::MULTIFILT_SETUP,
    ProcessName::TILT_MULTIFILT,
    ProcessName::CCDERASER,
    ProcessName::JOIN_WARP_2_MODEL,
    ProcessName::CTF_3D_SETUP,
    ProcessName::ALIGN_FRAMES,
    ProcessName::CTF_3D,
    ProcessName::SPLIT_BATCH,
    ProcessName::SORT_TILT_FRAMES,
    ProcessName::SUBTOMO_SETUP,
    ProcessName::TILT_ALIGN,
    ProcessName::ALT_TOMO_SETUP,
    ProcessName::ALT_TOMO_PROCESS_CHUNKS,
    ProcessName::REDUCE_FILT_VOL,
    ProcessName::MAN,
    ProcessName::TILT_XCORR,
    ProcessName::SERIES_WATCHER,
];
