#!/usr/bin/env bash
# Run in a fresh user-owned source snapshot. No sudo, system installs, or GPU job.
set -euo pipefail
cd "$(dirname "$0")/.."
PG_ENV="${PG_ENV:-$HOME/.venvs/mlds490-pg}"
case "$(realpath -m "$PG_ENV")" in
    "$HOME"/*) ;;
    *) printf '%s\n' 'PG_ENV must be inside your home directory.' >&2; exit 1 ;;
esac
if [ -e "$PG_ENV" ]; then
    printf '%s\n' 'Existing environment: inspect it rather than reinstalling automatically.' >&2
    exit 1
fi
if python3 -c 'import ensurepip' 2>/dev/null; then
    python3 -m venv "$PG_ENV"
else
    python3 -m venv --without-pip "$PG_ENV"
    # Ubuntu sometimes omits ensurepip. Bootstrap only the isolated user venv.
    PG_BOOTSTRAP="$(mktemp)"
    python3 -c 'import sys,urllib.request; urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", sys.argv[1])' "$PG_BOOTSTRAP"
    "$PG_ENV/bin/python" "$PG_BOOTSTRAP"
fi
"$PG_ENV/bin/python" -m pip install --upgrade pip
# GTX 1080 is Pascal. Use a known older build, then test a real kernel ONLY after allocation.
# CUDA wheel runtime is independent of the /usr/bin/nvcc toolkit (11.5 on this host).
"$PG_ENV/bin/python" -m pip install "torch==${PG_TORCH_VERSION:-2.5.1}" --index-url "${PG_TORCH_INDEX:-https://download.pytorch.org/whl/cu118}"
"$PG_ENV/bin/python" -m pip install -r requirements-pg.txt
mkdir -p results/policy_gradient_part1/remote
"$PG_ENV/bin/python" -m pip freeze > results/policy_gradient_part1/remote/environment.txt
"$PG_ENV/bin/python" scripts/run_pg_validation.py --pong
printf '%s\n' 'Setup/CPU validation complete. Resource policy must be confirmed before GPU use.'
