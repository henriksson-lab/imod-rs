//! Translation units from `IMOD/raptor`: the `RAPTOR` program (`main.cpp`
//! and the units its `Makefile` links: `mainClasses`, `template`,
//! `correspondence`, `trajectory`, `optimization`, `fillContours`, and the
//! `opencv` and `suitesparse` libraries) and the `MarkersCorrespond` program
//! RAPTOR runs (`correspondence/markersCorrespondMainTest.cpp`,
//! `svlMarkerCorrespondenceLBModel.cpp` and the `lasik` StairVision
//! library).  Library code neither program reaches is recorded in
//! `DEAD_CODE.md` rather than translated.

pub mod correspondence;
pub mod external;
#[path = "fillContours/mod.rs"]
pub mod fill_contours;
pub mod lasik;
pub mod main;
#[path = "mainClasses/mod.rs"]
pub mod main_classes;
pub mod opencv;
pub mod optimization;
pub mod suitesparse;
pub mod template;
pub mod trajectory;
