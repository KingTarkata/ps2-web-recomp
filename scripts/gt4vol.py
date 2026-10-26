"""Lists and extracts files from Gran Turismo 4's GT4.VOL (the "RoFS" 3.1
volume that holds the game's data, menus and scripts), for looking at what a
build loads. Written from the format as GT4FS (Razer2015) documents it.

usage: gt4vol.py list GT4.VOL
       gt4vol.py extract GT4.VOL OUT_DIR [PATTERN ...]   (fnmatch on the path)
"""
import fnmatch
import os
import struct
import sys
import zlib

PAGE = 0x800
MAGIC_PLAIN = 0x526F4653        # "RoFS"
MAGIC_CRYPT = (-0x526F4654) & 0xFFFFFFFF
VERSION = 0x00030001
OFFSET_KEY = 0x14AC327A
DATA_KEY = 0x55


def xor_key(data, key):
    return bytes(b ^ key[i % len(key)] for i, b in enumerate(data))


class Volume:
    def __init__(self, path):
        self.f = open(path, 'rb')
        self.f.seek(0, 2)
        size = self.f.tell()
        self.base = None
        for i in range(min(size // PAGE, 10000)):
            self.f.seek(i * PAGE)
            head = self.f.read(8)
            magic = struct.unpack('>I', head[:4])[0]
            if magic in (MAGIC_PLAIN, MAGIC_CRYPT) and struct.unpack('<I', head[4:])[0] == VERSION:
                self.base = i * PAGE
                self.encrypted = magic == MAGIC_CRYPT
                break
        if self.base is None:
            sys.exit('no RoFS 3.1 table of contents found')
        self.f.seek(self.base + 4)
        self.version, toc_len, page_count, self.page_len, entry_count = struct.unpack('<IiiHH', self.f.read(16))
        self.data_offset = self.base + page_count * self.page_len
        self.f.seek(self.base + 0x40)
        offsets = []
        for i in range(entry_count + 1):
            raw = self.f.read(4)
            if self.encrypted:
                raw = xor_key(raw, struct.pack('<i', (OFFSET_KEY * (i + 1)) & 0xFFFFFFFF if (OFFSET_KEY * (i + 1)) & 0x80000000 == 0 else ((OFFSET_KEY * (i + 1)) & 0xFFFFFFFF) - (1 << 32)))
            offsets.append(struct.unpack('<i', raw)[0])
        self.pages = [(self.base + a, b - a) for a, b in zip(offsets, offsets[1:])]
        self.entries = []
        for off, length in self.pages:
            self.entries.extend(self.read_page(off, length))

    def read_page(self, off, length):
        self.f.seek(off)
        data = zlib.decompress(self.f.read(length), -15)
        if self.encrypted:
            data = xor_key(data, bytes([DATA_KEY]))
        flag, count, _next, _prev = struct.unpack_from('<hhii', data, 0)
        if flag != 0:
            return []
        out = []
        for i in range(count // 2):
            eoff, elen, moff, mlen = struct.unpack_from('<hhhh', data, len(data) - (i * 8 + 8))
            parent = struct.unpack_from('>i', data, eoff)[0]
            name = data[eoff + 4:eoff + elen].decode('utf-8', 'replace')
            kind = data[moff]
            e = {'parent': parent, 'name': name, 'kind': kind}
            if kind == 0:
                e['node'] = struct.unpack_from('<i', data, moff + 1)[0]
            elif kind == 1:
                e['page'], e['date'], e['size'] = struct.unpack_from('<iii', data, moff + 1)
                e['packed'] = e['size']
            elif kind == 2:
                e['page'], e['date'], e['packed'], e['size'] = struct.unpack_from('<iiii', data, moff + 1)
            out.append(e)
        return out

    def paths(self):
        dirs = {e['node']: e for e in self.entries if e['kind'] == 0}

        def path_of(e):
            parts = []
            while e['parent'] != 0:
                parts.append(e['name'])
                e = dirs[e['parent']]
            return '/'.join(reversed(parts))
        return [(path_of(e), e) for e in self.entries if e['kind'] != 0]

    def read(self, e):
        self.f.seek(self.data_offset + e['page'] * self.page_len)
        data = self.f.read(e['packed'])
        if len(data) > 4 and data[:4] == b'\xC5\xEE\xF7\xFF':
            data = zlib.decompress(data[8:], -15)
        return data


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ('list', 'extract'):
        sys.exit(__doc__)
    vol = Volume(sys.argv[2])
    files = vol.paths()
    if sys.argv[1] == 'list':
        for path, e in files:
            print('%10d %s%s' % (e['size'], path, ' [z]' if e['kind'] == 2 else ''))
        return
    out = sys.argv[3]
    patterns = sys.argv[4:] or ['*']
    n = 0
    for path, e in files:
        if not any(fnmatch.fnmatch(path, p) for p in patterns):
            continue
        dest = os.path.join(out, path)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        data = vol.read(e)
        if len(data) != e['size']:
            print('size mismatch for %s: %d vs %d' % (path, len(data), e['size']), file=sys.stderr)
        with open(dest, 'wb') as f:
            f.write(data)
        n += 1
    print('%d files extracted' % n)


if __name__ == '__main__':
    main()
