//! Rust translation of `IMOD/qttools/processchunks/comfilejobs.{h,cpp}`.

use super::{CHUNK_NOT_DONE, CHUNK_SYNC};

/// C++ `ComFileJobs::Job`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Job {
    pub root: String,
    pub num_chunk_err: i32,
    pub flag: i32,
}

/// C++ `ComFileJobs`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComFileJobs {
    num_jobs: i32,
    job_array: Vec<Job>,
    com_extension: String,
}

impl ComFileJobs {
    /// C++ `ComFileJobs::ComFileJobs`.
    pub fn new(com_file_array: Vec<String>, single_file: bool, com_extension: String) -> Self {
        let num_jobs = com_file_array.len() as i32;
        let mut job_array = Vec::with_capacity(com_file_array.len());
        for com_file_name in com_file_array {
            let root = match com_file_name.rfind('.') {
                Some(ext_index) => com_file_name[..ext_index].to_owned(),
                None => com_file_name,
            };
            let flag = if single_file
                || root.ends_with("-start")
                || root.ends_with("-finish")
                || root.ends_with("-sync")
            {
                CHUNK_SYNC
            } else {
                CHUNK_NOT_DONE
            };
            job_array.push(Job {
                root,
                num_chunk_err: 0,
                flag,
            });
        }
        Self {
            num_jobs,
            job_array,
            com_extension,
        }
    }

    /// C++ declaration `ComFileJobs::setup`.
    ///
    /// The vendored C++ translation unit declares this method but has no
    /// definition.  Keeping it as an explicit unavailable operation preserves
    /// that source boundary rather than inventing behavior.
    pub fn setup(&mut self, _com_file: &str, _single_file: bool) {
        unimplemented!("ComFileJobs::setup has no definition in IMOD source")
    }

    /// C++ inline `ComFileJobs::getLogFileName`.
    pub fn get_log_file_name(&self, index: usize) -> String {
        format!("{}.log", self.job_array[index].root)
    }

    /// C++ inline `ComFileJobs::getPyFileName`.
    pub fn get_py_file_name(&self, index: usize) -> String {
        format!("{}.py", self.job_array[index].root)
    }

    /// C++ inline `ComFileJobs::getJobFileName`.
    pub fn get_job_file_name(&self, index: usize) -> String {
        format!("{}.job", self.job_array[index].root)
    }

    /// C++ inline `ComFileJobs::getQidFileName`.
    pub fn get_qid_file_name(&self, index: usize) -> String {
        format!("{}.qid", self.job_array[index].root)
    }

    /// C++ inline `ComFileJobs::getComFileName`.
    pub fn get_com_file_name(&self, index: usize) -> String {
        format!("{}{}", self.job_array[index].root, self.com_extension)
    }

    /// C++ inline `ComFileJobs::getNumChunkErr`.
    pub fn get_num_chunk_err(&self, index: usize) -> i32 {
        self.job_array[index].num_chunk_err
    }

    /// C++ inline `ComFileJobs::incrementNumChunkErr`.
    pub fn increment_num_chunk_err(&mut self, index: usize) {
        self.job_array[index].num_chunk_err += 1;
    }

    /// C++ `ComFileJobs::setFlagNotDone`.
    pub fn set_flag_not_done(&mut self, index: usize, single_file: bool) {
        let root = &self.job_array[index].root;
        self.job_array[index].flag = if single_file
            || root.ends_with("-start")
            || root.ends_with("-finish")
            || root.ends_with("-sync")
        {
            CHUNK_SYNC
        } else {
            CHUNK_NOT_DONE
        };
    }

    /// C++ inline `ComFileJobs::setFlag`.
    pub fn set_flag(&mut self, index: usize, flag: i32) {
        self.job_array[index].flag = flag;
    }

    /// C++ inline `ComFileJobs::getFlag`.
    pub fn get_flag(&self, index: usize) -> i32 {
        self.job_array[index].flag
    }

    /// C++ inline `ComFileJobs::getRoot`.
    pub fn get_root(&self, index: usize) -> &str {
        &self.job_array[index].root
    }

    /// C++ field `mNumJobs`, exposed to make the source's count available.
    pub fn num_jobs(&self) -> i32 {
        self.num_jobs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::qttools::processchunks::{CHUNK_DONE, CHUNK_SYNC};

    #[test]
    fn source_filename_and_status_rules_are_preserved() {
        let mut jobs = ComFileJobs::new(
            vec![
                "abc.001.com".to_owned(),
                "noextension".to_owned(),
                "batch-start.com".to_owned(),
            ],
            false,
            ".com".to_owned(),
        );
        assert_eq!(jobs.num_jobs(), 3);
        assert_eq!(jobs.get_root(0), "abc.001");
        assert_eq!(jobs.get_com_file_name(0), "abc.001.com");
        assert_eq!(jobs.get_py_file_name(1), "noextension.py");
        assert_eq!(jobs.get_flag(2), CHUNK_SYNC);
        jobs.set_flag(0, CHUNK_DONE);
        jobs.increment_num_chunk_err(0);
        assert_eq!(jobs.get_num_chunk_err(0), 1);
        jobs.set_flag_not_done(0, false);
        assert_eq!(jobs.get_flag(0), CHUNK_NOT_DONE);
    }
}
