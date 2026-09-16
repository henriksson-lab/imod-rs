//! Translation of `IMOD/mrc/manageshrmem.cpp`.

use crate::imod::libcfshr::parse_params::{
    pip_done, pip_get_boolean, pip_get_integer, pip_get_integer_array, pip_get_string,
    pip_number_of_entries, pip_read_or_parse_options,
};
use crate::imod::libiimod::iimage::{ii_delete, ii_new};
use crate::imod::libiimod::iishrmem::{
    ii_shr_mem_check_size, ii_shr_mem_create, ii_shr_mem_remove,
};

/// Source `earliestUse`/`latestUse` and maximum simultaneous allocation.
pub fn shared_memory_schedule(
    sizes_kib: &[usize],
    needed: &[Vec<usize>],
    keep: &[usize],
) -> Result<(Vec<usize>, Vec<usize>, usize), String> {
    let files = sizes_kib.len();
    let procs = needed.len();
    let mut early = vec![procs + 1; files];
    let mut late = vec![usize::MAX; files];
    for (proc, list) in needed.iter().enumerate() {
        for &one in list {
            if one == 0 || one > files {
                return Err("needed file index is out of range".into());
            }
            let f = one - 1;
            early[f] = early[f].min(proc);
            late[f] = if late[f] == usize::MAX {
                proc
            } else {
                late[f].max(proc)
            }
        }
    }
    for &one in keep {
        if one == 0 || one > files {
            return Err("kept file index is out of range".into());
        }
        late[one - 1] = procs + 1
    }
    if late.contains(&usize::MAX) {
        return Err("a file is not in any input/output map".into());
    }
    let mut now = 0;
    let mut maximum = 0;
    for proc in 0..procs {
        for file in 0..files {
            if early[file] == proc {
                now += sizes_kib[file]
            }
            if late[file] + 1 == proc {
                now -= sizes_kib[file]
            }
        }
        maximum = maximum.max(now)
    }
    Ok((early, late, maximum))
}
/// Run a minimal long-option form of the source command: repeated `--file`,
/// repeated `--command`, repeated comma-separated `--need`, and `--keep`.
pub fn manageshrmem(arguments: &[String]) -> i32 {
    let options: [&[u8]; 8] = [
        b"file:FileToCreate:FNM:",
        b"command:CommandToRun:CHM:",
        b"need:NeedFilesForCommand:IAM:",
        b"try:TrySizesFirst:I:",
        b"test:TestSizesInKilobytes:IA:",
        b"keep:ListOfFilesToKeep:LI:",
        b"remove:JustRemoveFiles:B:",
        b"help:Usage:B:",
    ];
    let argv = arguments
        .iter()
        .map(|value| value.as_bytes().to_vec())
        .collect::<Vec<_>>();
    let mut opt = 0;
    let mut non = 0;
    pip_read_or_parse_options(
        argv.len() as i32,
        &argv,
        &options,
        8,
        argv.first().map_or(b"manageshrmem", Vec::as_slice),
        1,
        0,
        0,
        &mut opt,
        &mut non,
        None,
    );
    let mut count = 0;
    pip_number_of_entries(b"FileToCreate", &mut count);
    let mut names = Vec::new();
    for _ in 0..count {
        let mut text = Vec::new();
        if pip_get_string(b"FileToCreate", &mut text) != 0 {
            pip_done();
            return 1;
        }
        names.push(String::from_utf8_lossy(&text).into_owned())
    }
    let mut commands = Vec::new();
    let mut needed = Vec::new();
    let mut keep = Vec::new();
    let mut remove = false;
    let mut remove_i = 0;
    let _ = pip_get_boolean(b"JustRemoveFiles", &mut remove_i);
    remove = remove_i != 0;
    let mut command_count = 0;
    pip_number_of_entries(b"CommandToRun", &mut command_count);
    for _ in 0..command_count {
        let mut text = Vec::new();
        if pip_get_string(b"CommandToRun", &mut text) != 0 {
            pip_done();
            return 1;
        }
        commands.push(String::from_utf8_lossy(&text).into_owned())
    }
    let mut needed_count = 0;
    pip_number_of_entries(b"NeedFilesForCommand", &mut needed_count);
    for _ in 0..needed_count {
        let mut values = [0_i32; 1000];
        let mut length = 0;
        if pip_get_integer_array(b"NeedFilesForCommand", &mut values, &mut length, 1000) != 0 {
            pip_done();
            return 1;
        }
        needed.push(
            values[..length as usize]
                .iter()
                .map(|v| *v as usize)
                .collect(),
        )
    }
    let mut keep_text = Vec::new();
    if pip_get_string(b"ListOfFilesToKeep", &mut keep_text) == 0 {
        keep = String::from_utf8_lossy(&keep_text)
            .split(',')
            .filter_map(|v| v.trim().parse().ok())
            .collect()
    };
    let mut try_sizes = 0;
    let _ = pip_get_integer(b"TrySizesFirst", &mut try_sizes);
    let mut test_sizes = [0_i32; 1000];
    let mut test_count = 0;
    let have_test_sizes = pip_get_integer_array(
        b"TestSizesInKilobytes",
        &mut test_sizes,
        &mut test_count,
        1000,
    ) == 0;
    pip_done();
    if have_test_sizes {
        let mut handles = Vec::with_capacity(test_count as usize);
        for file in 0..test_count as usize {
            if test_sizes[file] <= 0 {
                return 1;
            }
            let name = format!("/IMODShrMem_{}_dummy{file}", test_sizes[file]);
            let raw = ii_new();
            if raw.is_null() {
                return 1;
            }
            if unsafe { ii_shr_mem_create(&name, &mut *raw) } != 0 {
                unsafe {
                    ii_delete(raw);
                }
                for raw in handles {
                    unsafe {
                        ii_delete(raw);
                    }
                }
                return 1;
            }
            handles.push(raw);
        }
        for raw in handles {
            unsafe {
                ii_delete(raw);
            }
        }
        println!("All {test_count} shared memory files could be created");
        return 0;
    }
    if count <= 0 {
        return 1;
    }
    let mut sizes = Vec::new();
    for name in &names {
        let size = ii_shr_mem_check_size(name);
        if size == 0 {
            return 1;
        }
        sizes.push(size / 1024)
    }
    if remove {
        for name in &names {
            let _ = ii_shr_mem_remove(name);
        }
        return 0;
    }
    if needed.is_empty() && names.len() == commands.len().saturating_sub(1) {
        needed = (0..commands.len())
            .map(|proc| {
                let mut list = Vec::new();
                if proc > 0 {
                    list.push(proc);
                }
                if proc < names.len() {
                    list.push(proc + 1);
                }
                list
            })
            .collect();
    }
    if commands.len() < 2 || needed.len() != commands.len() {
        return 1;
    }
    let Ok((early, late, maximum)) = shared_memory_schedule(&sizes, &needed, &keep) else {
        return 1;
    };
    if try_sizes != 0 {
        let mut current = 0_usize;
        let mut peak_process = 0_usize;
        for process in 0..commands.len() {
            for file in 0..names.len() {
                if early[file] == process {
                    current += sizes[file];
                }
                if late[file] + 1 == process {
                    current -= sizes[file];
                }
            }
            if current == maximum {
                peak_process = process;
                break;
            }
        }
        let mut probes = Vec::new();
        for file in 0..names.len() {
            if early[file] <= peak_process && late[file] >= peak_process {
                let raw = ii_new();
                if raw.is_null() {
                    return 1;
                }
                if unsafe { ii_shr_mem_create(&names[file], &mut *raw) } != 0 {
                    unsafe {
                        ii_delete(raw);
                    }
                    for raw in probes {
                        unsafe {
                            ii_delete(raw);
                        }
                    }
                    return 1;
                }
                probes.push(raw);
            }
        }
        for raw in probes {
            unsafe {
                ii_delete(raw);
            }
        }
        if try_sizes < 0 {
            println!(
                "Shared memory files totalling {:.1} MB (the maximum) can be created",
                maximum as f32 / 1024.
            );
            return 0;
        }
    }
    let mut handles = (0..names.len()).map(|_| None).collect::<Vec<_>>();
    for proc in 0..=commands.len() {
        for file in 0..names.len() {
            if early[file] == proc {
                let raw = ii_new();
                if raw.is_null() {
                    return 1;
                }
                let status = unsafe { ii_shr_mem_create(&names[file], &mut *raw) };
                if status != 0 {
                    unsafe { ii_delete(raw) };
                    return 1;
                }
                handles[file] = Some(raw)
            }
            if late[file] + 1 == proc && !keep.contains(&(file + 1)) {
                if let Some(raw) = handles[file].take() {
                    unsafe { ii_delete(raw) }
                }
            }
        }
        if proc < commands.len() {
            let Ok(status) = std::process::Command::new("sh")
                .arg("-c")
                .arg(&commands[proc])
                .status()
            else {
                return 1;
            };
            if !status.success() {
                return 1;
            }
        }
    }
    0
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn schedules_overlapping_files() {
        let (e, l, max) =
            shared_memory_schedule(&[10, 20], &[vec![1], vec![1, 2], vec![2]], &[]).unwrap();
        assert_eq!(e, [0, 1]);
        assert_eq!(l, [1, 2]);
        assert_eq!(max, 30)
    }
    #[test]
    fn rejects_unreferenced_file() {
        assert!(shared_memory_schedule(&[1], &[vec![]], &[]).is_err())
    }

    #[cfg(not(windows))]
    #[test]
    fn test_sizes_uses_real_posix_shared_memory() {
        assert_eq!(
            manageshrmem(&["manageshrmem".into(), "-test".into(), "1,2".into()]),
            0
        );
    }
}
