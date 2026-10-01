#!/bin/bash
# build.sh: compile hw/display/reims-vgpu-dirty.c with QEMU's own flags (taken from the build tree's
# compile_commands.json, so run a QEMU build first) and link it with stress.c against libqemuutil.a.
# MSYS2 UCRT64. Result: ./stress.exe SECONDS (see stress.c for the model and matrix.py for the gate matrix).
set -e
export MSYSTEM=UCRT64 PATH="/c/msys64/ucrt64/bin:/c/msys64/usr/bin:$PATH"
HERE="$(cd "$(dirname "$0")" && pwd)"
Q="$(cd "$HERE/../../vendor/qemu" && pwd)"
cd "$Q/build"
FLAGS=$(python - <<'PY'
import json, shlex
for e in json.load(open('compile_commands.json')):
    if e['file'].endswith('reims-vgpu-dirty.c'):
        a = shlex.split(e['command'])
        out, skip = [], 0
        for x in a[1:]:
            if skip:
                skip -= 1
            elif x in ('-MD', '-Werror', '-c') or x == '-Dmain=SDL_main':
                continue
            elif x in ('-MQ', '-MF', '-o'):
                skip = 1
            elif x.endswith('reims-vgpu-dirty.c'):
                continue
            else:
                out.append(x)
        print(' '.join(shlex.quote(x) for x in out))
        break
PY
)
eval cc $FLAGS -I../hw/display -o "$HERE/dirty.o" -c ../hw/display/reims-vgpu-dirty.c
eval cc $FLAGS -I../hw/display -o "$HERE/stress.o" -c "$HERE/stress.c"
cc -m64 -o "$HERE/stress.exe" "$HERE/stress.o" "$HERE/dirty.o" libqemuutil.a \
    $(pkg-config --libs glib-2.0 gthread-2.0) -lws2_32 -lwinmm -lole32 -luuid -lbcrypt -lsynchronization -lpathcch
