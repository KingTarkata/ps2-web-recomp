"""Writes a formatted, empty 8 MB PS2 memory card in the layout PCSX2 and the
native runner use: 16384 pages of 512 bytes, each followed by 16 bytes holding
the card's error correcting code. The game then skips its format prompt, which
the headless exploring runs cannot answer reliably.

usage: blank_card.py out.ps2
"""
import struct
import sys

PAGE, SPARE = 512, 16
PAGES = 0x4000
CLUSTER_PAGES, BLOCK_PAGES = 2, 16
CLUSTERS = PAGES // CLUSTER_PAGES
IFAT_CLUSTER = 8
FAT_CLUSTERS = list(range(9, 41))
ALLOC_OFFSET = 41
ALLOC_END = CLUSTERS - ALLOC_OFFSET - BLOCK_PAGES  # last two blocks are backups
FREE, LAST = 0x7FFFFFFF, 0xFFFFFFFF

# Column parity of a 128 byte chunk is a linear function of the XOR of its
# bytes; these are its values for each input bit.
COLUMN = [0x07, 0x16, 0x25, 0x34, 0x43, 0x52, 0x61, 0x70]


def parity(b):
    return bin(b).count('1') & 1


def ecc(chunk):
    x = 0
    for b in chunk:
        x ^= b
    column = 0x77
    for k in range(8):
        if x >> k & 1:
            column ^= COLUMN[k]
    line0 = line1 = 0x7F
    for i, b in enumerate(chunk):
        if parity(b):
            line0 ^= ~i & 0x7F
            line1 ^= i
    return bytes([column & 0x77, line0 & 0x7F, line1 & 0x7F])


def page_with_ecc(data):
    assert len(data) == PAGE
    spare = b''.join(ecc(data[i:i + 128]) for i in range(0, PAGE, 128)) + b'\0' * 4
    return data + spare


def superblock():
    sb = bytearray(PAGE)
    sb[0:28] = b'Sony PS2 Memory Card Format '
    sb[28:40] = b'1.2.0.0'.ljust(12, b'\0')
    struct.pack_into('<HHHHIIIIII', sb, 40, PAGE, CLUSTER_PAGES, BLOCK_PAGES, 0xFF00, CLUSTERS,
                     ALLOC_OFFSET, ALLOC_END, 0, PAGES // BLOCK_PAGES - 1, PAGES // BLOCK_PAGES - 2)
    ifat = [IFAT_CLUSTER] + [0] * 31
    struct.pack_into('<32I', sb, 0x50, *ifat)
    struct.pack_into('<32I', sb, 0xD0, *([LAST] * 32))
    sb[0x150] = 2  # card type: PS2
    sb[0x151] = 0x2B  # flags: ECC, bad blocks, erase sets to 0xFF
    # The driver's own bookkeeping after the format fields, as a freshly
    # formatted card of the same version carries it.
    sb[0x154:0x200] = bytes.fromhex('000400000001000008000000ffffffff000000000000000000000000411f00000000000000000000ffffffff00000000ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff')
    return bytes(sb)


def dir_entry(mode, length, name, stamp):
    e = bytearray(PAGE)
    struct.pack_into('<HHI', e, 0, mode, 0, length)
    e[8:16] = stamp
    struct.pack_into('<II', e, 16, 0, 0)
    e[24:32] = stamp
    e[64:64 + len(name)] = name
    return bytes(e)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    erased = bytes([0xFF]) * (PAGE + SPARE)
    pages = [erased] * PAGES

    def put_cluster(cluster, data):
        for i in range(CLUSTER_PAGES):
            pages[cluster * CLUSTER_PAGES + i] = page_with_ecc(data[i * PAGE:(i + 1) * PAGE])

    pages[0] = page_with_ecc(superblock())
    ifat = struct.pack('<%dI' % len(FAT_CLUSTERS), *FAT_CLUSTERS)
    put_cluster(IFAT_CLUSTER, ifat.ljust(PAGE * CLUSTER_PAGES, b'\xFF'))
    for n, cluster in enumerate(FAT_CLUSTERS):
        entries = [FREE] * 256
        if n == 0:
            entries[0] = LAST  # the root directory's one cluster
        put_cluster(cluster, struct.pack('<256I', *entries))
    stamp = bytes([0, 0, 0, 0, 1, 1, 0xD0, 0x07])  # 2000-01-01 00:00:00
    root = dir_entry(0x8427, 2, b'.', stamp) + dir_entry(0xA426, 0, b'..', stamp)
    put_cluster(ALLOC_OFFSET, root)
    with open(sys.argv[1], 'wb') as out:
        for p in pages:
            out.write(p)


if __name__ == '__main__':
    main()
