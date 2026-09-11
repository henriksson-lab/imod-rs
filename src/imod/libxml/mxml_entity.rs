//! Translation of `IMOD/libxml/mxml-entity.c`.
#![allow(dead_code, unsafe_op_in_unsafe_fn)]

use super::*;
use core::ffi::{c_char, c_int};
use std::ffi::CStr;

/// Matches C `mxmlEntityAddCallback` (`mxml-entity.c:26`).
pub unsafe fn mxml_entity_add_callback(cb: MxmlEntityCb) -> c_int {
    let global: *mut MxmlGlobal = mxml_global();

    if ((*global).num_entity_cbs as usize) < (*global).entity_cbs.len() {
        (*global).entity_cbs[(*global).num_entity_cbs as usize] = cb;
        (*global).num_entity_cbs += 1;

        0
    } else {
        mxml_error(c"Unable to add entity callback!".as_ptr());

        -1
    }
}

/// Matches C `mxmlEntityGetName` (`mxml-entity.c:52`).
pub unsafe fn mxml_entity_get_name(val: c_int) -> *const c_char {
    match val {
        38 => c"amp".as_ptr(),
        60 => c"lt".as_ptr(),
        62 => c"gt".as_ptr(),
        34 => c"quot".as_ptr(),
        _ => core::ptr::null(),
    }
}

/// Matches C `mxmlEntityGetValue` (`mxml-entity.c:80`).
pub unsafe fn mxml_entity_get_value(name: *const c_char) -> c_int {
    let mut i: c_int;
    let mut ch: c_int;
    let global: *mut MxmlGlobal = mxml_global();

    i = 0;
    while i < (*global).num_entity_cbs {
        if let Some(cb) = (*global).entity_cbs[i as usize] {
            ch = cb(name);
            if ch >= 0 {
                return ch;
            }
        }
        i += 1;
    }

    -1
}

/// Matches C `mxmlEntityRemoveCallback` (`mxml-entity.c:102`).
pub unsafe fn mxml_entity_remove_callback(cb: MxmlEntityCb) {
    let mut i: c_int;
    let global: *mut MxmlGlobal = mxml_global();

    i = 0;
    while i < (*global).num_entity_cbs {
        if cb.map(|f| f as usize) == (*global).entity_cbs[i as usize].map(|f| f as usize) {
            (*global).num_entity_cbs -= 1;

            if i < (*global).num_entity_cbs {
                let base = (&raw mut (*global).entity_cbs) as *mut MxmlEntityCb;
                libc::memmove(
                    base.add(i as usize) as *mut core::ffi::c_void,
                    base.add(i as usize + 1) as *const core::ffi::c_void,
                    ((*global).num_entity_cbs - i) as usize * core::mem::size_of::<MxmlEntityCb>(),
                );
            }

            return;
        }
        i += 1;
    }
}

/// The static `entities[]` table inside C `_mxml_entity_cb`
/// (`mxml-entity.c:134`), in source order.
static ENTITIES: [(&CStr, c_int); 257] = [
    (c"AElig", 198),
    (c"Aacute", 193),
    (c"Acirc", 194),
    (c"Agrave", 192),
    (c"Alpha", 913),
    (c"Aring", 197),
    (c"Atilde", 195),
    (c"Auml", 196),
    (c"Beta", 914),
    (c"Ccedil", 199),
    (c"Chi", 935),
    (c"Dagger", 8225),
    (c"Delta", 916),
    (c"Dstrok", 208),
    (c"ETH", 208),
    (c"Eacute", 201),
    (c"Ecirc", 202),
    (c"Egrave", 200),
    (c"Epsilon", 917),
    (c"Eta", 919),
    (c"Euml", 203),
    (c"Gamma", 915),
    (c"Iacute", 205),
    (c"Icirc", 206),
    (c"Igrave", 204),
    (c"Iota", 921),
    (c"Iuml", 207),
    (c"Kappa", 922),
    (c"Lambda", 923),
    (c"Mu", 924),
    (c"Ntilde", 209),
    (c"Nu", 925),
    (c"OElig", 338),
    (c"Oacute", 211),
    (c"Ocirc", 212),
    (c"Ograve", 210),
    (c"Omega", 937),
    (c"Omicron", 927),
    (c"Oslash", 216),
    (c"Otilde", 213),
    (c"Ouml", 214),
    (c"Phi", 934),
    (c"Pi", 928),
    (c"Prime", 8243),
    (c"Psi", 936),
    (c"Rho", 929),
    (c"Scaron", 352),
    (c"Sigma", 931),
    (c"THORN", 222),
    (c"Tau", 932),
    (c"Theta", 920),
    (c"Uacute", 218),
    (c"Ucirc", 219),
    (c"Ugrave", 217),
    (c"Upsilon", 933),
    (c"Uuml", 220),
    (c"Xi", 926),
    (c"Yacute", 221),
    (c"Yuml", 376),
    (c"Zeta", 918),
    (c"aacute", 225),
    (c"acirc", 226),
    (c"acute", 180),
    (c"aelig", 230),
    (c"agrave", 224),
    (c"alefsym", 8501),
    (c"alpha", 945),
    (c"amp", b'&' as c_int),
    (c"and", 8743),
    (c"ang", 8736),
    (c"apos", b'\'' as c_int),
    (c"aring", 229),
    (c"asymp", 8776),
    (c"atilde", 227),
    (c"auml", 228),
    (c"bdquo", 8222),
    (c"beta", 946),
    (c"brkbar", 166),
    (c"brvbar", 166),
    (c"bull", 8226),
    (c"cap", 8745),
    (c"ccedil", 231),
    (c"cedil", 184),
    (c"cent", 162),
    (c"chi", 967),
    (c"circ", 710),
    (c"clubs", 9827),
    (c"cong", 8773),
    (c"copy", 169),
    (c"crarr", 8629),
    (c"cup", 8746),
    (c"curren", 164),
    (c"dArr", 8659),
    (c"dagger", 8224),
    (c"darr", 8595),
    (c"deg", 176),
    (c"delta", 948),
    (c"diams", 9830),
    (c"die", 168),
    (c"divide", 247),
    (c"eacute", 233),
    (c"ecirc", 234),
    (c"egrave", 232),
    (c"empty", 8709),
    (c"emsp", 8195),
    (c"ensp", 8194),
    (c"epsilon", 949),
    (c"equiv", 8801),
    (c"eta", 951),
    (c"eth", 240),
    (c"euml", 235),
    (c"euro", 8364),
    (c"exist", 8707),
    (c"fnof", 402),
    (c"forall", 8704),
    (c"frac12", 189),
    (c"frac14", 188),
    (c"frac34", 190),
    (c"frasl", 8260),
    (c"gamma", 947),
    (c"ge", 8805),
    (c"gt", b'>' as c_int),
    (c"hArr", 8660),
    (c"harr", 8596),
    (c"hearts", 9829),
    (c"hellip", 8230),
    (c"hibar", 175),
    (c"iacute", 237),
    (c"icirc", 238),
    (c"iexcl", 161),
    (c"igrave", 236),
    (c"image", 8465),
    (c"infin", 8734),
    (c"int", 8747),
    (c"iota", 953),
    (c"iquest", 191),
    (c"isin", 8712),
    (c"iuml", 239),
    (c"kappa", 954),
    (c"lArr", 8656),
    (c"lambda", 955),
    (c"lang", 9001),
    (c"laquo", 171),
    (c"larr", 8592),
    (c"lceil", 8968),
    (c"ldquo", 8220),
    (c"le", 8804),
    (c"lfloor", 8970),
    (c"lowast", 8727),
    (c"loz", 9674),
    (c"lrm", 8206),
    (c"lsaquo", 8249),
    (c"lsquo", 8216),
    (c"lt", b'<' as c_int),
    (c"macr", 175),
    (c"mdash", 8212),
    (c"micro", 181),
    (c"middot", 183),
    (c"minus", 8722),
    (c"mu", 956),
    (c"nabla", 8711),
    (c"nbsp", 160),
    (c"ndash", 8211),
    (c"ne", 8800),
    (c"ni", 8715),
    (c"not", 172),
    (c"notin", 8713),
    (c"nsub", 8836),
    (c"ntilde", 241),
    (c"nu", 957),
    (c"oacute", 243),
    (c"ocirc", 244),
    (c"oelig", 339),
    (c"ograve", 242),
    (c"oline", 8254),
    (c"omega", 969),
    (c"omicron", 959),
    (c"oplus", 8853),
    (c"or", 8744),
    (c"ordf", 170),
    (c"ordm", 186),
    (c"oslash", 248),
    (c"otilde", 245),
    (c"otimes", 8855),
    (c"ouml", 246),
    (c"para", 182),
    (c"part", 8706),
    (c"permil", 8240),
    (c"perp", 8869),
    (c"phi", 966),
    (c"pi", 960),
    (c"piv", 982),
    (c"plusmn", 177),
    (c"pound", 163),
    (c"prime", 8242),
    (c"prod", 8719),
    (c"prop", 8733),
    (c"psi", 968),
    (c"quot", b'\"' as c_int),
    (c"rArr", 8658),
    (c"radic", 8730),
    (c"rang", 9002),
    (c"raquo", 187),
    (c"rarr", 8594),
    (c"rceil", 8969),
    (c"rdquo", 8221),
    (c"real", 8476),
    (c"reg", 174),
    (c"rfloor", 8971),
    (c"rho", 961),
    (c"rlm", 8207),
    (c"rsaquo", 8250),
    (c"rsquo", 8217),
    (c"sbquo", 8218),
    (c"scaron", 353),
    (c"sdot", 8901),
    (c"sect", 167),
    (c"shy", 173),
    (c"sigma", 963),
    (c"sigmaf", 962),
    (c"sim", 8764),
    (c"spades", 9824),
    (c"sub", 8834),
    (c"sube", 8838),
    (c"sum", 8721),
    (c"sup", 8835),
    (c"sup1", 185),
    (c"sup2", 178),
    (c"sup3", 179),
    (c"supe", 8839),
    (c"szlig", 223),
    (c"tau", 964),
    (c"there4", 8756),
    (c"theta", 952),
    (c"thetasym", 977),
    (c"thinsp", 8201),
    (c"thorn", 254),
    (c"tilde", 732),
    (c"times", 215),
    (c"trade", 8482),
    (c"uArr", 8657),
    (c"uacute", 250),
    (c"uarr", 8593),
    (c"ucirc", 251),
    (c"ugrave", 249),
    (c"uml", 168),
    (c"upsih", 978),
    (c"upsilon", 965),
    (c"uuml", 252),
    (c"weierp", 8472),
    (c"xi", 958),
    (c"yacute", 253),
    (c"yen", 165),
    (c"yuml", 255),
    (c"zeta", 950),
    (c"zwj", 8205),
    (c"zwnj", 8204),
];

/// Matches C `_mxml_entity_cb` (`mxml-entity.c:128`).
pub unsafe extern "C" fn mxml_entity_cb(name: *const c_char) -> c_int {
    let mut diff: c_int;
    let mut current: c_int;
    let mut first: c_int;
    let mut last: c_int;

    first = 0;
    last = (ENTITIES.len() - 1) as c_int;

    while (last - first) > 1 {
        current = (first + last) / 2;

        diff = libc::strcmp(name, ENTITIES[current as usize].0.as_ptr());
        if diff == 0 {
            return ENTITIES[current as usize].1;
        } else if diff < 0 {
            last = current;
        } else {
            first = current;
        }
    }

    if libc::strcmp(name, ENTITIES[first as usize].0.as_ptr()) == 0 {
        ENTITIES[first as usize].1
    } else if libc::strcmp(name, ENTITIES[last as usize].0.as_ptr()) == 0 {
        ENTITIES[last as usize].1
    } else {
        -1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `mxmlEntityGetName` only knows the four XML markup characters
    /// (`mxml-entity.c:52`) -- notably not `apos` -- while
    /// `mxmlEntityGetValue` goes through the 257-entry `_mxml_entity_cb`
    /// table.  All values below come from the reference `libimxml.so`.
    #[test]
    fn entity_name_and_value_tables_match_upstream() {
        unsafe {
            assert_eq!(
                std::ffi::CStr::from_ptr(mxml_entity_get_name(38)).to_bytes(),
                b"amp"
            );
            assert_eq!(
                std::ffi::CStr::from_ptr(mxml_entity_get_name(60)).to_bytes(),
                b"lt"
            );
            assert_eq!(
                std::ffi::CStr::from_ptr(mxml_entity_get_name(62)).to_bytes(),
                b"gt"
            );
            assert_eq!(
                std::ffi::CStr::from_ptr(mxml_entity_get_name(34)).to_bytes(),
                b"quot"
            );
            /* 39 is apostrophe: the C switch has no case for it. */
            assert!(mxml_entity_get_name(39).is_null());
            /* Nothing else is a markup character. */
            for v in 0..300 {
                if v != 34 && v != 38 && v != 60 && v != 62 {
                    assert!(mxml_entity_get_name(v).is_null(), "getName({v})");
                }
            }

            assert_eq!(mxml_entity_get_value(c"amp".as_ptr()), 38);
            assert_eq!(mxml_entity_get_value(c"apos".as_ptr()), 39);
            assert_eq!(mxml_entity_get_value(c"nbsp".as_ptr()), 160);
            assert_eq!(mxml_entity_get_value(c"AElig".as_ptr()), 198);
            assert_eq!(mxml_entity_get_value(c"Alpha".as_ptr()), 913);
            assert_eq!(mxml_entity_get_value(c"euro".as_ptr()), 8364);
            assert_eq!(mxml_entity_get_value(c"zwnj".as_ptr()), 8204);
            assert_eq!(mxml_entity_get_value(c"bogus".as_ptr()), -1);
            assert_eq!(mxml_entity_get_value(c"".as_ptr()), -1);
            assert_eq!(mxml_entity_get_value(c"zzzz".as_ptr()), -1);
            /* The binary search in _mxml_entity_cb needs a sorted table. */
            for w in ENTITIES.windows(2) {
                assert!(
                    libc::strcmp(w[0].0.as_ptr(), w[1].0.as_ptr()) < 0,
                    "entities[] must stay sorted"
                );
            }
        }
    }
}
