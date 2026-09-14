//! Translation of `IMOD/libcfshr/winversion.c`.
#![allow(dead_code)]

#[cfg(windows)]
const VER_MINORVERSION: u32 = 0x0000_0001;
#[cfg(windows)]
const VER_MAJORVERSION: u32 = 0x0000_0002;
#[cfg(windows)]
const VER_PLATFORMID: u32 = 0x0000_0008;
#[cfg(windows)]
const VER_EQUAL: u8 = 1;
#[cfg(windows)]
const VER_GREATER_EQUAL: u8 = 3;
#[cfg(windows)]
const VER_PLATFORM_WIN32_NT: u32 = 2;

#[cfg(windows)]
#[repr(C)]
struct OsVersionInfoExW {
    dw_os_version_info_size: u32,
    dw_major_version: u32,
    dw_minor_version: u32,
    dw_build_number: u32,
    dw_platform_id: u32,
    sz_csd_version: [u16; 128],
    w_service_pack_major: u16,
    w_service_pack_minor: u16,
    w_suite_mask: u16,
    w_product_type: u8,
    w_reserved: u8,
}

/// The one foreign boundary of this unit: `kernel32`'s version-compare API,
/// which `winversion.c` calls directly.  Compiled only on Windows.
#[cfg(windows)]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn VerifyVersionInfoW(
        version_information: &OsVersionInfoExW,
        type_mask: u32,
        condition_mask: u64,
    ) -> i32;
    fn VerSetConditionMask(condition_mask: u64, type_mask: u32, condition: u8) -> u64;
}

/// Original `isWindows2000` (`winversion.c:10`).
pub fn is_windows_2000() -> i32 {
    #[cfg(windows)]
    {
        return is_windows_version(5, VER_EQUAL as i32, 0, VER_EQUAL as i32);
    }
    #[cfg(not(windows))]
    -1
}

/// Original `isWindowsXP` (`winversion.c:19`).
pub fn is_windows_xp() -> i32 {
    #[cfg(windows)]
    {
        return is_windows_version(5, VER_EQUAL as i32, 1, VER_GREATER_EQUAL as i32);
    }
    #[cfg(not(windows))]
    -1
}

/// Original `isWindowsVista` (`winversion.c:28`).
pub fn is_windows_vista() -> i32 {
    #[cfg(windows)]
    {
        return is_windows_version(6, VER_EQUAL as i32, 0, VER_EQUAL as i32);
    }
    #[cfg(not(windows))]
    -1
}

/// Original `isWindows7` (`winversion.c:37`).
pub fn is_windows_7() -> i32 {
    #[cfg(windows)]
    {
        return is_windows_version(6, VER_EQUAL as i32, 1, VER_EQUAL as i32);
    }
    #[cfg(not(windows))]
    -1
}

/// Original `isWindows8` (`winversion.c:47`).
pub fn is_windows_8() -> i32 {
    #[cfg(windows)]
    {
        return is_windows_version(6, VER_EQUAL as i32, 2, VER_GREATER_EQUAL as i32);
    }
    #[cfg(not(windows))]
    -1
}

/// Original `isWindows10` (`winversion.c:56`).
pub fn is_windows_10() -> i32 {
    #[cfg(windows)]
    {
        return is_windows_version(10, VER_EQUAL as i32, 0, VER_GREATER_EQUAL as i32);
    }
    #[cfg(not(windows))]
    -1
}

/// Original `isWindowsVersion` (`winversion.c:65`).
pub fn is_windows_version(major: i32, op_major: i32, minor: i32, op_minor: i32) -> i32 {
    #[cfg(windows)]
    unsafe {
        let osvi = OsVersionInfoExW {
            dw_os_version_info_size: core::mem::size_of::<OsVersionInfoExW>() as u32,
            dw_major_version: major as u32,
            dw_minor_version: minor as u32,
            dw_build_number: 0,
            dw_platform_id: VER_PLATFORM_WIN32_NT,
            sz_csd_version: [0; 128],
            w_service_pack_major: 0,
            w_service_pack_minor: 0,
            w_suite_mask: 0,
            w_product_type: 0,
            w_reserved: 0,
        };
        let mut condition_mask = 0_u64;
        condition_mask = VerSetConditionMask(condition_mask, VER_MAJORVERSION, op_major as u8);
        condition_mask = VerSetConditionMask(condition_mask, VER_MINORVERSION, op_minor as u8);
        condition_mask = VerSetConditionMask(condition_mask, VER_PLATFORMID, VER_EQUAL);
        return if VerifyVersionInfoW(
            &osvi,
            VER_MAJORVERSION | VER_MINORVERSION | VER_PLATFORMID,
            condition_mask,
        ) != 0
        {
            1
        } else {
            0
        };
    }
    #[cfg(not(windows))]
    -1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(windows))]
    #[test]
    fn source_non_windows_results_are_minus_one() {
        assert_eq!(is_windows_2000(), -1);
        assert_eq!(is_windows_xp(), -1);
        assert_eq!(is_windows_vista(), -1);
        assert_eq!(is_windows_7(), -1);
        assert_eq!(is_windows_8(), -1);
        assert_eq!(is_windows_10(), -1);
        assert_eq!(is_windows_version(10, 1, 0, 3), -1);
    }
}
