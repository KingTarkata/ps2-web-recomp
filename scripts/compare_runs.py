"""Compares two headless runs of the same game and inputs, made on different
builds or machines, to check that the emulation is deterministic: every frame
the runs dumped (--frames-dir) must match byte for byte, and the RAM dumps
(PS2_DUMP) are diffed into ranges so a divergence can be placed. A few words
always differ, where the game stores the real-time clock it read from the
host; the report lists them so they can be told apart from real divergence.

usage: compare_runs.py DIR_A DIR_B   (each holding ram.bin and frame_*.ppm)
"""
import glob
import hashlib
import os
import sys


def digest(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    a, b = sys.argv[1], sys.argv[2]
    frames = sorted(os.path.basename(p) for p in glob.glob(os.path.join(a, 'frame_*.ppm')))
    same = different = missing = 0
    for name in frames:
        other = os.path.join(b, name)
        if not os.path.exists(other):
            missing += 1
            continue
        if digest(os.path.join(a, name)) == digest(other):
            same += 1
        else:
            different += 1
            print('frame differs:', name)
    print('frames: %d identical, %d different, %d missing in B' % (same, different, missing))
    ra, rb = os.path.join(a, 'ram.bin'), os.path.join(b, 'ram.bin')
    if not (os.path.exists(ra) and os.path.exists(rb)):
        print('no RAM dumps to compare')
        return
    da, db = open(ra, 'rb').read(), open(rb, 'rb').read()
    if len(da) != len(db):
        print('RAM dumps differ in size: %d vs %d' % (len(da), len(db)))
        return
    ranges = []
    for i in range(0, len(da), 4):
        if da[i:i + 4] != db[i:i + 4]:
            if ranges and i - ranges[-1][1] <= 64:
                ranges[-1][1] = i + 4
            else:
                ranges.append([i, i + 4])
    total = sum(hi - lo for lo, hi in ranges)
    print('RAM: %d differing ranges, %d bytes' % (len(ranges), total))
    for lo, hi in ranges[:60]:
        where = 'EE' if lo < 32 << 20 else 'IOP/other'
        print('  %08x-%08x %6d bytes %s  A=%s B=%s' % (lo, hi, hi - lo, where, da[lo:lo + 8].hex(), db[lo:lo + 8].hex()))
    if len(ranges) > 60:
        print('  ... %d more' % (len(ranges) - 60))


if __name__ == '__main__':
    main()
