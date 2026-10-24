# Recompiler regressions

These tests construct small ELF files from synthetic instructions and need no
game data. Build the recompiler and run them from the repository root:

```sh
cmake -S . -B build/tools -DCMAKE_BUILD_TYPE=Release
cmake --build build/tools --target eerecomp
python3 tests/analysis_tailcall.py
python3 tests/elf_bss.py
python3 tests/codegen_fallthrough.py
```

Each script accepts `--recompiler /path/to/eerecomp` to check another build.
The fallthrough test also uses `clang++` or `g++` from `PATH` to check that the
generated C++ compiles.
