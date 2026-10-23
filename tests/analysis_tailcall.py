#!/usr/bin/env python3
"""Discover an address-taken function that tail-calls outside its image."""
from pathlib import Path
import argparse
import struct
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--recompiler", type=Path, default=root / "build/tools/eerecomp")
args = parser.parse_args()

with tempfile.TemporaryDirectory(prefix='ps2-tailcall-test-') as temp:
    work = Path(temp)
    data = bytearray(0x150)
    data[:7] = b'\x7fELF\x01\x01\x01'
    struct.pack_into('<HHIIIIIHHHHHH', data, 16, 2, 8, 1, 0x100000, 52, 0, 0, 52, 32, 1, 40, 0, 0)
    struct.pack_into('<8I', data, 52, 1, 0x100, 0x100000, 0, 0x50, 0x50, 5, 4)
    # jal 0x100040; nop; take &0x100020; jr ra; nop
    struct.pack_into('<6I', data, 0x100, 0x0c040010, 0, 0x3c080010, 0x35080020, 0x03e00008, 0)
    # The address-taken function ends with a tail call outside this image.
    struct.pack_into('<3I', data, 0x120, 0x24020001, 0x08080000, 0)
    struct.pack_into('<2I', data, 0x140, 0x03e00008, 0)
    (work / 'input.elf').write_bytes(data)
    def check(words, count):
        struct.pack_into('<3I', data, 0x120, *words)
        (work / 'input.elf').write_bytes(data)
        result = subprocess.run([str(args.recompiler.resolve()), str(work / 'input.elf'),
                                 '--out', str(work / 'generated')], check=True, capture_output=True, text=True)
        assert f'code {count} instructions' in result.stdout, result.stdout

    check([0x24020001, 0x08080000, 0], 11)
    # An invalid delay slot still rejects the candidate.
    check([0x24020001, 0x08080000, 1], 8)
    # beq zero,zero,0x100100: external conditional branches remain rejected.
    check([0x24020001, 0x10000036, 0], 8)
    # j 0x100030: a local destination containing an invalid opcode is rejected.
    struct.pack_into('<I', data, 0x130, 1)
    check([0x24020001, 0x0804000c, 0], 8)
print('External tail-call discovery regression passed')
