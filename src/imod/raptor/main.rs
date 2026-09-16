//! Owned command orchestration translated from `IMOD/raptor/main.cpp`.
//!
//! The numerical phases live in the translated RAPTOR modules. This unit owns
//! command-line normalization, output layout, and the IMOD process-boundary
//! command plan that the C++ entry point built with `system()` strings.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RaptorMainError {
    Missing(&'static str),
    Invalid(&'static str),
    UnknownOption(String),
}

#[derive(Clone, Debug, PartialEq)]
pub struct RaptorOptions {
    pub executable_path: PathBuf,
    pub input_path: PathBuf,
    pub input_file: String,
    pub output_path: PathBuf,
    pub diameters: Vec<usize>,
    pub markers_per_image: Option<usize>,
    pub angles_in_header: bool,
    pub binning: usize,
    pub reconstruction: Option<i32>,
    pub thickness: Option<usize>,
    pub max_distance_candidate: Option<usize>,
    pub min_neighbors_mrf: Option<usize>,
    pub roll_off_mrf: Option<usize>,
    pub verbose: u8,
    pub white_markers: bool,
    pub tracking_only: bool,
    pub xray: bool,
    pub seed: Option<u64>,
}

impl RaptorOptions {
    /// Source defaults after `PipGet*` calls.
    pub fn from_options(values: &BTreeMap<String, String>) -> Result<Self, RaptorMainError> {
        let required = |name: &'static str| {
            values
                .get(name)
                .cloned()
                .ok_or(RaptorMainError::Missing(name))
        };
        let parse = |name: &'static str| {
            values
                .get(name)
                .map(|value| value.parse().map_err(|_| RaptorMainError::Invalid(name)))
                .transpose()
        };
        let flag = |name: &str| {
            values.get(name).is_some_and(|value| {
                value.is_empty() || value == "1" || value.eq_ignore_ascii_case("true")
            })
        };
        let diameters = required("diameter")?
            .split(',')
            .map(|value| {
                value
                    .parse()
                    .map_err(|_| RaptorMainError::Invalid("diameter"))
            })
            .collect::<Result<Vec<usize>, _>>()?;
        if diameters.is_empty() || diameters.contains(&0) {
            return Err(RaptorMainError::Invalid("diameter"));
        }
        let binning = parse("bin")?.unwrap_or(1usize);
        if binning == 0 {
            return Err(RaptorMainError::Invalid("bin"));
        }
        let verbose = values
            .get("verb")
            .map(|value| {
                value
                    .parse::<u8>()
                    .map_err(|_| RaptorMainError::Invalid("verb"))
            })
            .transpose()?
            .unwrap_or(1);
        Ok(Self {
            executable_path: PathBuf::from(required("exec-path")?),
            input_path: PathBuf::from(required("path")?),
            input_file: required("input")?,
            output_path: PathBuf::from(required("output")?),
            diameters,
            markers_per_image: parse("markers")?,
            angles_in_header: flag("angles"),
            binning,
            reconstruction: values
                .get("rec")
                .map(|value| {
                    value
                        .parse::<i32>()
                        .map_err(|_| RaptorMainError::Invalid("rec"))
                })
                .transpose()?,
            thickness: parse("thickness")?,
            max_distance_candidate: parse("max-dist")?,
            min_neighbors_mrf: parse("min-neigh")?,
            roll_off_mrf: parse("roll-off")?,
            verbose,
            white_markers: flag("white"),
            tracking_only: flag("tracking"),
            xray: flag("xray"),
            seed: values
                .get("seed")
                .map(|value| {
                    value
                        .parse::<u64>()
                        .map_err(|_| RaptorMainError::Invalid("seed"))
                })
                .transpose()?,
        })
    }

    /// Source's basename stripping, including its historical `_preali` rule.
    pub fn basename(&self) -> String {
        let mut base = self
            .input_file
            .rsplit_once('.')
            .map_or(self.input_file.as_str(), |(base, _)| base)
            .to_owned();
        if base.ends_with("_preali") {
            base.truncate(base.len() - 7);
        }
        base
    }

    /// Source output directory set (`align`, `IMOD`, `temp`, and debug only in
    /// verbose-debug mode), as owned paths.
    pub fn output_directories(&self) -> Vec<PathBuf> {
        let mut directories = vec![
            self.output_path.clone(),
            self.output_path.join("align"),
            self.output_path.join("IMOD"),
            self.output_path.join("temp"),
        ];
        if self.verbose >= 2 {
            directories.push(self.output_path.join("debug"));
        }
        directories
    }

    /// Creates exactly the source output layout without shell interpolation.
    pub fn create_output_directories(&self) -> std::io::Result<()> {
        for directory in self.output_directories() {
            std::fs::create_dir_all(directory)?;
        }
        Ok(())
    }

    /// Source defaults derived from stack side and extension.
    pub fn correspondence_defaults(&self, max_side: usize) -> (usize, usize) {
        let aligned = self.input_file.ends_with(".preali") || self.input_file.ends_with(".ali");
        (
            self.roll_off_mrf
                .unwrap_or(if aligned { max_side / 8 } else { max_side / 4 }),
            self.max_distance_candidate.unwrap_or(if aligned {
                (max_side / 10).max(50)
            } else {
                (max_side / 6).max(50)
            }),
        )
    }

    /// External IMOD commands built by the latter half of source `main`, held
    /// as argv vectors instead of unsafe `system()` strings.
    pub fn imod_commands(&self, discarded_frames: &[bool], alpha_final: f64) -> Vec<Vec<String>> {
        let base = self.basename();
        let input = self.input_path.join(&self.input_file);
        let imod = self.output_path.join("IMOD");
        let align = self.output_path.join("align");
        let mut commands = Vec::new();
        if self.angles_in_header {
            commands.push(vec![
                "extracttilts".into(),
                "-tilts".into(),
                "-input".into(),
                input.display().to_string(),
                "-output".into(),
                imod.join(format!("{base}.rawtlt")).display().to_string(),
            ]);
        }
        if !self.tracking_only {
            let alpha = if alpha_final - 90.0 <= -90.0 {
                alpha_final + 90.0
            } else {
                alpha_final - 90.0
            };
            commands.push(vec![
                "tiltalign".into(),
                "-param".into(),
                imod.join(format!("{base}_tiltalignScript.txt"))
                    .display()
                    .to_string(),
                format!("# alpha={alpha}"),
            ]);
            let sections = discarded_frames
                .iter()
                .enumerate()
                .filter_map(|(index, &discarded)| (!discarded).then_some(index.to_string()))
                .collect::<Vec<_>>()
                .join(",");
            commands.push(vec![
                "newstack".into(),
                "-input".into(),
                input.display().to_string(),
                "-output".into(),
                align.join(format!("{base}.ali")).display().to_string(),
                "-offset".into(),
                "0,0".into(),
                "-xform".into(),
                imod.join(format!("{base}.xf")).display().to_string(),
                "-secs".into(),
                sections,
            ]);
            if self.binning != 1 {
                commands.push(vec![
                    "newstack".into(),
                    "-input".into(),
                    align.join(format!("{base}.ali")).display().to_string(),
                    "-output".into(),
                    align.join(format!("{base}Bin.ali")).display().to_string(),
                    "-bin".into(),
                    self.binning.to_string(),
                ]);
            }
        }
        if let Some(reconstruction) = self.reconstruction.filter(|mode| (0..=2).contains(mode)) {
            commands.push(vec![
                "submfg".into(),
                align.join("tilt.com").display().to_string(),
                format!("# reconstruction={reconstruction}"),
            ]);
        }
        commands
    }
}

/// Parses the source's long option names and aliases from an argv sequence.
pub fn parse_raptor_arguments(arguments: &[String]) -> Result<RaptorOptions, RaptorMainError> {
    let mut values = BTreeMap::new();
    let aliases = BTreeMap::from([
        ("--execPath", "exec-path"),
        ("--path", "path"),
        ("--input", "input"),
        ("--output", "output"),
        ("--diameter", "diameter"),
        ("--markers", "markers"),
        ("--bin", "bin"),
        ("--rec", "rec"),
        ("--thickness", "thickness"),
        ("--maxDist", "max-dist"),
        ("--minNeigh", "min-neigh"),
        ("--rollOff", "roll-off"),
        ("--verb", "verb"),
        ("--seed", "seed"),
    ]);
    let flags = BTreeMap::from([
        ("--angles", "angles"),
        ("--white", "white"),
        ("--tracking", "tracking"),
        ("--xray", "xray"),
    ]);
    let mut index = 0;
    while index < arguments.len() {
        if let Some(&name) = flags.get(arguments[index].as_str()) {
            values.insert(name.into(), "true".into());
            index += 1;
        } else if let Some(&name) = aliases.get(arguments[index].as_str()) {
            let value = arguments
                .get(index + 1)
                .ok_or(RaptorMainError::Invalid(name))?;
            values.insert(name.into(), value.clone());
            index += 2;
        } else {
            return Err(RaptorMainError::UnknownOption(arguments[index].clone()));
        }
    }
    RaptorOptions::from_options(&values)
}

pub fn raptor_log_path(options: &RaptorOptions) -> PathBuf {
    options
        .output_path
        .join("align")
        .join(format!("{}_RAPTOR.log", options.basename()))
}
pub fn raptor_input_path(options: &RaptorOptions) -> PathBuf {
    options.input_path.join(&options.input_file)
}
pub fn raptor_path_exists(options: &RaptorOptions) -> bool {
    Path::new(&raptor_input_path(options)).exists()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_source_options_and_builds_safe_command_plan() {
        let options = parse_raptor_arguments(
            &[
                "--execPath",
                "/bin",
                "--path",
                "/input",
                "--input",
                "series.preali",
                "--output",
                "/out",
                "--diameter",
                "5,8",
                "--angles",
                "--bin",
                "2",
            ]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>(),
        )
        .unwrap();
        assert_eq!(options.basename(), "series");
        assert_eq!(options.correspondence_defaults(800), (100, 80));
        assert_eq!(options.output_directories().len(), 4);
        let commands = options.imod_commands(&[false, true, false], 10.0);
        assert_eq!(commands[0][0], "extracttilts");
        assert!(
            commands.iter().any(
                |command| command[0] == "newstack" && command.last() == Some(&"0,2".to_owned())
            )
        );
    }
    #[test]
    fn rejects_missing_required_and_invalid_binning() {
        assert!(
            parse_raptor_arguments(
                &["--path", "x"]
                    .into_iter()
                    .map(String::from)
                    .collect::<Vec<_>>()
            )
            .is_err()
        );
        assert!(
            parse_raptor_arguments(
                &[
                    "--execPath",
                    "x",
                    "--path",
                    "x",
                    "--input",
                    "x",
                    "--output",
                    "x",
                    "--diameter",
                    "2",
                    "--bin",
                    "0"
                ]
                .into_iter()
                .map(String::from)
                .collect::<Vec<_>>()
            )
            .is_err()
        );
    }
}
