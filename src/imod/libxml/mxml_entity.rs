//! Translation of `IMOD/libxml/mxml-entity.c`.
#![allow(dead_code)]

use super::*;

/// Matches C `mxmlEntityAddCallback` (`mxml-entity.c:26`).
pub fn mxml_entity_add_callback(cb: MxmlEntityCb) -> i32 {
    let added = mxml_global().with_borrow_mut(|global| {
        if global.entity_cbs.len() < 100 {
            global.entity_cbs.push(cb);
            true
        } else {
            false
        }
    });

    /*
     * The C reports the failure while still holding the global; the borrow is
     * released first here because `mxml_error` reads the same block.
     */

    if added {
        0
    } else {
        mxml_error(b"Unable to add entity callback!");

        -1
    }
}

/// Matches C `mxmlEntityGetName` (`mxml-entity.c:52`).
pub fn mxml_entity_get_name(val: i32) -> Option<&'static [u8]> {
    match val {
        38 => Some(b"amp"),
        60 => Some(b"lt"),
        62 => Some(b"gt"),
        34 => Some(b"quot"),
        _ => None,
    }
}

/// Matches C `mxmlEntityGetValue` (`mxml-entity.c:80`).
pub fn mxml_entity_get_value(name: &[u8]) -> i32 {
    /*
     * The C reads `global->num_entity_cbs` and `global->entity_cbs[i]` on each
     * pass and then calls the callback; each read takes and drops the borrow so
     * that a callback may reach the global itself.
     */
    for cb in mxml_global().with_borrow(|global| global.entity_cbs.clone()) {
        if let Some(cb) = cb {
            let ch = cb(name);
            if ch >= 0 {
                return ch;
            }
        }
    }

    -1
}

/// Matches C `mxmlEntityRemoveCallback` (`mxml-entity.c:102`).
pub fn mxml_entity_remove_callback(cb: MxmlEntityCb) {
    mxml_global().with_borrow_mut(|global| {
        if let Some(index) = global
            .entity_cbs
            .iter()
            .position(|registered| cb.map(|f| f as usize) == registered.map(|f| f as usize))
        {
            global.entity_cbs.remove(index);
        }
    });
}

/// The static `entities[]` table inside C `_mxml_entity_cb`
/// (`mxml-entity.c:134`), in source order.
static ENTITIES: [(&[u8], i32); 257] = [
    (b"AElig", 198),
    (b"Aacute", 193),
    (b"Acirc", 194),
    (b"Agrave", 192),
    (b"Alpha", 913),
    (b"Aring", 197),
    (b"Atilde", 195),
    (b"Auml", 196),
    (b"Beta", 914),
    (b"Ccedil", 199),
    (b"Chi", 935),
    (b"Dagger", 8225),
    (b"Delta", 916),
    (b"Dstrok", 208),
    (b"ETH", 208),
    (b"Eacute", 201),
    (b"Ecirc", 202),
    (b"Egrave", 200),
    (b"Epsilon", 917),
    (b"Eta", 919),
    (b"Euml", 203),
    (b"Gamma", 915),
    (b"Iacute", 205),
    (b"Icirc", 206),
    (b"Igrave", 204),
    (b"Iota", 921),
    (b"Iuml", 207),
    (b"Kappa", 922),
    (b"Lambda", 923),
    (b"Mu", 924),
    (b"Ntilde", 209),
    (b"Nu", 925),
    (b"OElig", 338),
    (b"Oacute", 211),
    (b"Ocirc", 212),
    (b"Ograve", 210),
    (b"Omega", 937),
    (b"Omicron", 927),
    (b"Oslash", 216),
    (b"Otilde", 213),
    (b"Ouml", 214),
    (b"Phi", 934),
    (b"Pi", 928),
    (b"Prime", 8243),
    (b"Psi", 936),
    (b"Rho", 929),
    (b"Scaron", 352),
    (b"Sigma", 931),
    (b"THORN", 222),
    (b"Tau", 932),
    (b"Theta", 920),
    (b"Uacute", 218),
    (b"Ucirc", 219),
    (b"Ugrave", 217),
    (b"Upsilon", 933),
    (b"Uuml", 220),
    (b"Xi", 926),
    (b"Yacute", 221),
    (b"Yuml", 376),
    (b"Zeta", 918),
    (b"aacute", 225),
    (b"acirc", 226),
    (b"acute", 180),
    (b"aelig", 230),
    (b"agrave", 224),
    (b"alefsym", 8501),
    (b"alpha", 945),
    (b"amp", b'&' as i32),
    (b"and", 8743),
    (b"ang", 8736),
    (b"apos", b'\'' as i32),
    (b"aring", 229),
    (b"asymp", 8776),
    (b"atilde", 227),
    (b"auml", 228),
    (b"bdquo", 8222),
    (b"beta", 946),
    (b"brkbar", 166),
    (b"brvbar", 166),
    (b"bull", 8226),
    (b"cap", 8745),
    (b"ccedil", 231),
    (b"cedil", 184),
    (b"cent", 162),
    (b"chi", 967),
    (b"circ", 710),
    (b"clubs", 9827),
    (b"cong", 8773),
    (b"copy", 169),
    (b"crarr", 8629),
    (b"cup", 8746),
    (b"curren", 164),
    (b"dArr", 8659),
    (b"dagger", 8224),
    (b"darr", 8595),
    (b"deg", 176),
    (b"delta", 948),
    (b"diams", 9830),
    (b"die", 168),
    (b"divide", 247),
    (b"eacute", 233),
    (b"ecirc", 234),
    (b"egrave", 232),
    (b"empty", 8709),
    (b"emsp", 8195),
    (b"ensp", 8194),
    (b"epsilon", 949),
    (b"equiv", 8801),
    (b"eta", 951),
    (b"eth", 240),
    (b"euml", 235),
    (b"euro", 8364),
    (b"exist", 8707),
    (b"fnof", 402),
    (b"forall", 8704),
    (b"frac12", 189),
    (b"frac14", 188),
    (b"frac34", 190),
    (b"frasl", 8260),
    (b"gamma", 947),
    (b"ge", 8805),
    (b"gt", b'>' as i32),
    (b"hArr", 8660),
    (b"harr", 8596),
    (b"hearts", 9829),
    (b"hellip", 8230),
    (b"hibar", 175),
    (b"iacute", 237),
    (b"icirc", 238),
    (b"iexcl", 161),
    (b"igrave", 236),
    (b"image", 8465),
    (b"infin", 8734),
    (b"int", 8747),
    (b"iota", 953),
    (b"iquest", 191),
    (b"isin", 8712),
    (b"iuml", 239),
    (b"kappa", 954),
    (b"lArr", 8656),
    (b"lambda", 955),
    (b"lang", 9001),
    (b"laquo", 171),
    (b"larr", 8592),
    (b"lceil", 8968),
    (b"ldquo", 8220),
    (b"le", 8804),
    (b"lfloor", 8970),
    (b"lowast", 8727),
    (b"loz", 9674),
    (b"lrm", 8206),
    (b"lsaquo", 8249),
    (b"lsquo", 8216),
    (b"lt", b'<' as i32),
    (b"macr", 175),
    (b"mdash", 8212),
    (b"micro", 181),
    (b"middot", 183),
    (b"minus", 8722),
    (b"mu", 956),
    (b"nabla", 8711),
    (b"nbsp", 160),
    (b"ndash", 8211),
    (b"ne", 8800),
    (b"ni", 8715),
    (b"not", 172),
    (b"notin", 8713),
    (b"nsub", 8836),
    (b"ntilde", 241),
    (b"nu", 957),
    (b"oacute", 243),
    (b"ocirc", 244),
    (b"oelig", 339),
    (b"ograve", 242),
    (b"oline", 8254),
    (b"omega", 969),
    (b"omicron", 959),
    (b"oplus", 8853),
    (b"or", 8744),
    (b"ordf", 170),
    (b"ordm", 186),
    (b"oslash", 248),
    (b"otilde", 245),
    (b"otimes", 8855),
    (b"ouml", 246),
    (b"para", 182),
    (b"part", 8706),
    (b"permil", 8240),
    (b"perp", 8869),
    (b"phi", 966),
    (b"pi", 960),
    (b"piv", 982),
    (b"plusmn", 177),
    (b"pound", 163),
    (b"prime", 8242),
    (b"prod", 8719),
    (b"prop", 8733),
    (b"psi", 968),
    (b"quot", b'\"' as i32),
    (b"rArr", 8658),
    (b"radic", 8730),
    (b"rang", 9002),
    (b"raquo", 187),
    (b"rarr", 8594),
    (b"rceil", 8969),
    (b"rdquo", 8221),
    (b"real", 8476),
    (b"reg", 174),
    (b"rfloor", 8971),
    (b"rho", 961),
    (b"rlm", 8207),
    (b"rsaquo", 8250),
    (b"rsquo", 8217),
    (b"sbquo", 8218),
    (b"scaron", 353),
    (b"sdot", 8901),
    (b"sect", 167),
    (b"shy", 173),
    (b"sigma", 963),
    (b"sigmaf", 962),
    (b"sim", 8764),
    (b"spades", 9824),
    (b"sub", 8834),
    (b"sube", 8838),
    (b"sum", 8721),
    (b"sup", 8835),
    (b"sup1", 185),
    (b"sup2", 178),
    (b"sup3", 179),
    (b"supe", 8839),
    (b"szlig", 223),
    (b"tau", 964),
    (b"there4", 8756),
    (b"theta", 952),
    (b"thetasym", 977),
    (b"thinsp", 8201),
    (b"thorn", 254),
    (b"tilde", 732),
    (b"times", 215),
    (b"trade", 8482),
    (b"uArr", 8657),
    (b"uacute", 250),
    (b"uarr", 8593),
    (b"ucirc", 251),
    (b"ugrave", 249),
    (b"uml", 168),
    (b"upsih", 978),
    (b"upsilon", 965),
    (b"uuml", 252),
    (b"weierp", 8472),
    (b"xi", 958),
    (b"yacute", 253),
    (b"yen", 165),
    (b"yuml", 255),
    (b"zeta", 950),
    (b"zwj", 8205),
    (b"zwnj", 8204),
];

/// Matches C `_mxml_entity_cb` (`mxml-entity.c:128`).
pub fn mxml_entity_cb(name: &[u8]) -> i32 {
    let mut diff: i32;
    let mut current: i32;
    let mut first: i32;
    let mut last: i32;

    first = 0;
    last = (ENTITIES.len() - 1) as i32;

    while (last - first) > 1 {
        current = (first + last) / 2;

        /*
         * strcmp(name, entities[current].name): the table names hold no NUL,
         * so the byte-slice ordering has the sign strcmp returns, which is all
         * the three comparisons below use.
         */
        diff = match name.cmp(ENTITIES[current as usize].0) {
            core::cmp::Ordering::Less => -1,
            core::cmp::Ordering::Equal => 0,
            core::cmp::Ordering::Greater => 1,
        };
        if diff == 0 {
            return ENTITIES[current as usize].1;
        } else if diff < 0 {
            last = current;
        } else {
            first = current;
        }
    }

    if name == ENTITIES[first as usize].0 {
        ENTITIES[first as usize].1
    } else if name == ENTITIES[last as usize].0 {
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
        assert_eq!(mxml_entity_get_name(38), Some(b"amp".as_slice()));
        assert_eq!(mxml_entity_get_name(60), Some(b"lt".as_slice()));
        assert_eq!(mxml_entity_get_name(62), Some(b"gt".as_slice()));
        assert_eq!(mxml_entity_get_name(34), Some(b"quot".as_slice()));
        /* 39 is apostrophe: the C switch has no case for it. */
        assert!(mxml_entity_get_name(39).is_none());
        /* Nothing else is a markup character. */
        for v in 0..300 {
            if v != 34 && v != 38 && v != 60 && v != 62 {
                assert!(mxml_entity_get_name(v).is_none(), "getName({v})");
            }
        }

        assert_eq!(mxml_entity_get_value(b"amp"), 38);
        assert_eq!(mxml_entity_get_value(b"apos"), 39);
        assert_eq!(mxml_entity_get_value(b"nbsp"), 160);
        assert_eq!(mxml_entity_get_value(b"AElig"), 198);
        assert_eq!(mxml_entity_get_value(b"Alpha"), 913);
        assert_eq!(mxml_entity_get_value(b"euro"), 8364);
        assert_eq!(mxml_entity_get_value(b"zwnj"), 8204);
        assert_eq!(mxml_entity_get_value(b"bogus"), -1);
        assert_eq!(mxml_entity_get_value(b""), -1);
        assert_eq!(mxml_entity_get_value(b"zzzz"), -1);
        /* The binary search in _mxml_entity_cb needs a sorted table. */
        for w in ENTITIES.windows(2) {
            assert!(w[0].0 < w[1].0, "entities[] must stay sorted");
        }
    }
}
