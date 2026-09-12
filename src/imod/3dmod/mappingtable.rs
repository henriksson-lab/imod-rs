//! Translation of `IMOD/3dmod/mappingtable.cpp` and `mappingtable.h`.
//!
//! These are the upstream marching-cubes lookup tables.  The triangle table
//! is nibble-packed only to keep this source legible: zero is the upstream
//! `-1` terminator and one through twelve represent edge numbers zero through
//! eleven.  The initialized Rust value is therefore byte-for-byte equivalent
//! in element values to the original `int triangle_table[256][16]`.
#![allow(dead_code)]

/// `cube_edge_info`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CubeEdgeInfo {
    pub vertex_1: i32,
    pub vertex_2: i32,
    pub base_0: i32,
    pub base_1: i32,
    pub base_2: i32,
    pub axis: i32,
}

/// `cube_edges`.
pub const CUBE_EDGES: [CubeEdgeInfo; 12] = [
    CubeEdgeInfo {
        vertex_1: 0,
        vertex_2: 1,
        base_0: 0,
        base_1: 0,
        base_2: 0,
        axis: 0,
    },
    CubeEdgeInfo {
        vertex_1: 1,
        vertex_2: 2,
        base_0: 1,
        base_1: 0,
        base_2: 0,
        axis: 1,
    },
    CubeEdgeInfo {
        vertex_1: 2,
        vertex_2: 3,
        base_0: 0,
        base_1: 1,
        base_2: 0,
        axis: 0,
    },
    CubeEdgeInfo {
        vertex_1: 3,
        vertex_2: 0,
        base_0: 0,
        base_1: 0,
        base_2: 0,
        axis: 1,
    },
    CubeEdgeInfo {
        vertex_1: 4,
        vertex_2: 5,
        base_0: 0,
        base_1: 0,
        base_2: 1,
        axis: 0,
    },
    CubeEdgeInfo {
        vertex_1: 5,
        vertex_2: 6,
        base_0: 1,
        base_1: 0,
        base_2: 1,
        axis: 1,
    },
    CubeEdgeInfo {
        vertex_1: 6,
        vertex_2: 7,
        base_0: 0,
        base_1: 1,
        base_2: 1,
        axis: 0,
    },
    CubeEdgeInfo {
        vertex_1: 7,
        vertex_2: 4,
        base_0: 0,
        base_1: 0,
        base_2: 1,
        axis: 1,
    },
    CubeEdgeInfo {
        vertex_1: 0,
        vertex_2: 4,
        base_0: 0,
        base_1: 0,
        base_2: 0,
        axis: 2,
    },
    CubeEdgeInfo {
        vertex_1: 1,
        vertex_2: 5,
        base_0: 1,
        base_1: 0,
        base_2: 0,
        axis: 2,
    },
    CubeEdgeInfo {
        vertex_1: 2,
        vertex_2: 6,
        base_0: 1,
        base_1: 1,
        base_2: 0,
        axis: 2,
    },
    CubeEdgeInfo {
        vertex_1: 3,
        vertex_2: 7,
        base_0: 0,
        base_1: 1,
        base_2: 0,
        axis: 2,
    },
];

/// `face_corner_numbers` from the implementation-only source state.
pub const FACE_CORNER_NUMBERS: [[usize; 4]; 6] = [
    [0, 1, 2, 3],
    [0, 4, 5, 1],
    [1, 5, 6, 2],
    [2, 6, 7, 3],
    [3, 7, 4, 0],
    [4, 7, 6, 5],
];
/// `face_edge_numbers` from the implementation-only source state.
pub const FACE_EDGE_NUMBERS: [[i32; 4]; 6] = [
    [0, 1, 2, 3],
    [8, 4, 9, 0],
    [9, 5, 10, 1],
    [10, 6, 11, 2],
    [11, 7, 8, 3],
    [7, 6, 5, 4],
];

/// Right-hand triangle normal points opposite the gradient: `triangle_table`.
pub const TRIANGLE_TABLE: [[i32; 16]; 256] = {
    let packed = concat!(
        "0000000000000000914000000000000021a00000000000009249a2000000000032b000000000000091432b00000000003ab31a0000000000934b39ab90000000",
        "c430000000000000c13c910000000000a2143c0000000000c23a2c9ac0000000b42bc40000000000b1291bc9b0000000a41c4abca00000009ab9bc0000000000",
        "8590000000000000451485000000000021a598000000000025a852482000000032b598000000000054814532b00000003ab1a35980000000b3aa38834a850000",
        "598c4300000000005c83c513500000001a259843c000000085c5acca33a20000b42c4b9850000000c2b52c125c8500008591accab1c4000085cc5acab0000000",
        "6a500000000000006a591400000000006156210000000000695496246000000032b6a5000000000014932ba56000000036b5631530000000b363466455490000",
        "6a543c0000000000c1391ca56000000061521643c000000023663993c95600004bc2b46a50000000a56912b92c9b000056116cc6b1c4000056996b9bc0000000",
        "8a986a00000000004a16a48640000000819218628000000062464800000000008a96a82b300000002b36a146186400001933966986b30000b366346480000000",
        "a8698ac4300000006a88a33a183c000043c21982962800003c22c828600000006a96982b44bc000086116ac8112bbc10bc11c46b11988610bc6c860000000000",
        "7b60000000000000914b6700000000001a2b670000000000924a29b670000000726732000000000072632714900000007a61a73170000000a699633673490000",
        "43c7b600000000001c93c17b6000000021a43cb670000000b67a23ca39ac000047c674264000000091cc16612c670000c4741771661a000067aa7cac90000000",
        "b67859000000000045185467b0000000a21b6759800000007b6a28824a850000273672859000000032636714554800005981a6716317000048aa8534aa6773a0",
        "c439857b60000000b6785335183c000021a85943cb6700003a2ca35acc85b670598c46642c67000026cc6712cc8551c061a7164177c4598067aa7c85ac8a0000",
        "5ba57b0000000000b57a5b91400000001b27b1571000000049229779527b000052a325735000000014932a53a735000031535700000000004933953570000000",
        "5ba7b53c4000000091393ca5bb570000c4321771527b000057227b95223cc9207a54a72a47c40000c922917c22a55720c4774171500000005797c90000000000",
        "b8798ba9b0000000814b18a1b87b00007b8b2882992100007b88b2824000000032772992a798000073aa3287aa1448a098118717300000004838730000000000",
        "43c7b99ba798000013883ca1887bba80921829b2887b43c03c22c87b28720000a977982a77c44270a127c80000000000981187c417c10000c870000000000000",
        "78c00000000000001498c7000000000021a8c7000000000029a4928c700000002b3c78000000000032b149c780000000a31b3ac780000000c78b349b4ab90000",
        "3843780000000000189781371000000083743821a0000000723927a2989700008b72b842800000008b782b928129000041881bb1ab78000078bb89b9a0000000",
        "9759c7000000000074c147517000000079c5971a200000005a77a44a24c70000975c79b32000000032b14c71c5170000c5975c31ab3a0000ab44b35a44c77540",
        "39459375300000005137530000000000a214355374590000a255235370000000294792597b7200002b11b71750000000754459b7441aab40ab5b750000000000",
        "a5678c0000000000914a568c7000000016256178c00000008c749564524600006a52b378c0000000c7832b914a56000078c56b35b15300005496453466b38c70",
        "38478356a00000006a59177139780000743847621561000037997823995662906a52b7827428000072b8271288916a5015bb5641bb7884b078bb8956b95b0000",
        "a76c7a9ca000000074c714617a160000c1961c21676c0000c74476462000000032b6acca96c70000c1471ca1776a32b09c66c71966b33160c74476b346b40000",
        "96a36976394300006a77a171300000006299217699433790627237000000000042772b94776aa9702b11b76a1761000041976b00000000006b70000000000000",
        "6cb68c00000000006cb8c64910000000c68b6ca2100000008b6cb89a249200002c38c2682000000091432882638c00008a63a81a3c38000068338ca633499a30",
        "63b43684600000003916938963b600001a2b64468b4300009a33a28933b66830426846000000000091881282600000001a44a646800000009a8a680000000000",
        "965b69cb90000000165c61b6c4c1000021a59bb9c5b60000cb55b64c55a22450632936c39659000051cc1465cc3226c031661ac366599c605a6c340000000000",
        "63b6435469450000b633653510000000b4364b94665921a0b63365a235a30000596694642000000051612600000000005966941a641600005a60000000000000",
        "c58a5cbac0000000914a58ca8bac0000b2cc2552158c0000245549b2558ccb50c58ca53ac2a300008a5ca82acc3291408c55c353100000008c55c34953450000",
        "a3b83a43858a0000ba88a53b8891138084bb4358bb2115b0b238950000000000a522582840000000a52258912892000015458400000000009580000000000000",
        "ba9cb9000000000014aa4cacb000000021bb19b9c000000024b4cb000000000032cc2aca9000000014aa4c32ac3a000031c19c000000000034c0000000000000",
        "43993b9ba0000000ba3a13000000000043993b219b290000b2300000000000004292a90000000000a12000000000000041900000000000000000000000000000"
    );
    let bytes = packed.as_bytes();
    let mut table = [[0; 16]; 256];
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        let nibble = if byte <= b'9' {
            byte - b'0'
        } else {
            byte - b'a' + 10
        };
        table[index / 16][index % 16] = nibble as i32 - 1;
        index += 1;
    }
    table
};

/// `face_corner_bits`.
pub const FACE_CORNER_BITS: [[i32; 256]; 6] = {
    let mut table = [[0; 256]; 6];
    let mut face = 0;
    while face < 6 {
        let mut bits = 0;
        while bits < 256 {
            let mut corner = 0;
            let mut face_bits = 0;
            while corner < 4 {
                face_bits |= (((bits >> FACE_CORNER_NUMBERS[face][corner]) & 1) as i32) << corner;
                corner += 1;
            }
            table[face][bits] = face_bits;
            bits += 1;
        }
        face += 1;
    }
    table
};

/// Face-zero template in `cap_triangle_table`; the complete table applies
/// the source `face_edge_numbers` and `face_corner_numbers` permutations.
const CAP_TRIANGLE_BASE: [[i32; 10]; 16] = [
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 12, 3, -1, -1, -1, -1, -1, -1, -1],
    [1, 13, 0, -1, -1, -1, -1, -1, -1, -1],
    [13, 12, 1, 1, 12, 3, -1, -1, -1, -1],
    [2, 14, 1, -1, -1, -1, -1, -1, -1, -1],
    [0, 12, 3, 2, 14, 1, -1, -1, -1, -1],
    [14, 13, 2, 2, 13, 0, -1, -1, -1, -1],
    [3, 13, 12, 2, 13, 3, 14, 13, 2, -1],
    [3, 15, 2, -1, -1, -1, -1, -1, -1, -1],
    [12, 15, 0, 0, 15, 2, -1, -1, -1, -1],
    [1, 13, 0, 3, 15, 2, -1, -1, -1, -1],
    [13, 12, 1, 1, 12, 2, 2, 12, 15, -1],
    [15, 14, 3, 3, 14, 1, -1, -1, -1, -1],
    [12, 15, 0, 0, 15, 1, 1, 15, 14, -1],
    [0, 14, 13, 3, 14, 0, 15, 14, 3, -1],
    [13, 12, 14, 14, 12, 15, -1, -1, -1, -1],
];

/// `cap_triangle_table`.
pub const CAP_TRIANGLE_TABLE: [[[i32; 10]; 16]; 6] = {
    let mut table = [[[0; 10]; 16]; 6];
    let mut face = 0;
    while face < 6 {
        let mut row = 0;
        while row < 16 {
            let mut column = 0;
            while column < 10 {
                let value = CAP_TRIANGLE_BASE[row][column];
                table[face][row][column] = if value < 0 {
                    -1
                } else if value < 12 {
                    FACE_EDGE_NUMBERS[face][value as usize]
                } else {
                    12 + FACE_CORNER_NUMBERS[face][(value - 12) as usize] as i32
                };
                column += 1;
            }
            row += 1;
        }
        face += 1;
    }
    table
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_table_checksums_and_sentinels_match() {
        let triangle_sum: i64 = TRIANGLE_TABLE
            .iter()
            .flatten()
            .enumerate()
            .map(|(i, &v)| (i as i64 + 1) * (v as i64 + 31))
            .sum();
        let corner_sum: i64 = FACE_CORNER_BITS
            .iter()
            .flatten()
            .enumerate()
            .map(|(i, &v)| (i as i64 + 1) * (v as i64 + 31))
            .sum();
        let cap_sum: i64 = CAP_TRIANGLE_TABLE
            .iter()
            .flatten()
            .flatten()
            .enumerate()
            .map(|(i, &v)| (i as i64 + 1) * (v as i64 + 31))
            .sum();
        assert_eq!(triangle_sum, 285_470_687);
        assert_eq!(corner_sum, 45_602_432);
        assert_eq!(cap_sum, 17_137_932);
        assert_eq!(TRIANGLE_TABLE[1][..3], [8, 0, 3]);
        assert_eq!(CAP_TRIANGLE_TABLE[5][1][..3], [7, 16, 4]);
    }
}
