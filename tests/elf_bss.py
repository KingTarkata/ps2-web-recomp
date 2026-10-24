#!/usr/bin/env python3
"""Do not translate the zero-filled tail of an executable ELF load segment."""
import argparse
from pathlib import Path
import struct
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--recompiler', type=Path, default=root / 'build/tools/eerecomp')
args = parser.parse_args()

with tempfile.TemporaryDirectory(prefix='ps2-bss-test-') as temp:
    work = Path(temp)
    data = bytearray(0x108)
    data[:7] = b'\x7fELF\x01\x01\x01'
    struct.pack_into('<HHIIIIIHHHHHH', data, 16, 2, 8, 1, 0x100000, 52, 0, 0, 52, 32, 1, 40, 0, 0)
    # One executable segment containing jr ra; nop, followed by zero-filled BSS.
    struct.pack_into('<8I', data, 52, 1, 0x100, 0x100000, 0, 8, 0x1000, 5, 4)
    struct.pack_into('<2I', data, 0x100, 0x03e00008, 0)
    (work / 'input.elf').write_bytes(data)
    # A runtime-discovered overlay entry inside BSS must not register the
    # executable's initial zeroes as the implementation of that address.
    (work / 'entries.txt').write_text('100100\n')
    result = subprocess.run([str(args.recompiler.resolve()), str(work / 'input.elf'),
                             '--entries', str(work / 'entries.txt'), '--out', str(work / 'generated')],
                            check=True, capture_output=True, text=True)
    assert 'code 2 instructions' in result.stdout, result.stdout
print('Executable BSS regression passed')
