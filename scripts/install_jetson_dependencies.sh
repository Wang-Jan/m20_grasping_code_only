#!/usr/bin/env bash
# Install Jetson-only Python dependencies without accessing robot, camera, or M20 hardware.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
UV_BIN="${UV_BIN:-uv}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHEELHOUSE="${WHEELHOUSE:-${PROJECT_ROOT}/wheelhouse/jetson-jp62}"
PRIMARY_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
FALLBACK_MIRROR="https://pypi.mirrors.ustc.edu.cn/simple"
ULTRALYTICS_WEIGHTS_DIR="${ULTRALYTICS_WEIGHTS_DIR:-${PROJECT_ROOT}/weights}"
MOBILECLIP_FILE="mobileclip2_b.ts"
MOBILECLIP_RELEASE="${MOBILECLIP_RELEASE:-v8.4.0}"
MOBILECLIP_MIRROR_URL="${MOBILECLIP_MIRROR_URL:-https://ghfast.top/https://github.com/ultralytics/assets/releases/download/${MOBILECLIP_RELEASE}/${MOBILECLIP_FILE}}"
MOBILECLIP_OFFICIAL_URL="https://github.com/ultralytics/assets/releases/download/${MOBILECLIP_RELEASE}/${MOBILECLIP_FILE}"
TORCH_VERSION="2.8.0"
TORCHVISION_VERSION="0.23.0"
TORCH_WHEEL="${TORCH_WHEEL:-${WHEELHOUSE}/torch-2.8.0-cp310-cp310-linux_aarch64.whl}"
TORCHVISION_WHEEL="${TORCHVISION_WHEEL:-${WHEELHOUSE}/torchvision-0.23.0-cp310-cp310-linux_aarch64.whl}"
TORCH_SHA256="62a1beee9f2f147076a974d2942c90060c12771c94740830327cae705b2595fc"
TORCHVISION_SHA256="907c4c1933789645ebb20dd9181d40f8647978e6bd30086ae7b01febb937d2d1"

if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "This installer only supports Jetson aarch64 hosts." >&2
    exit 1
fi

if ! "${PYTHON_BIN}" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 10))'; then
    echo "This installer requires Python 3.10." >&2
    exit 1
fi

verify_jetson_wheel() {
    local wheel="$1"
    local checksum="$2"
    local wheel_name
    wheel_name="$(basename "${wheel}")"
    if [[ ! -f "${wheel}" ]]; then
        echo "Missing pre-downloaded Jetson wheel: ${wheel}" >&2
        echo "Place the verified ARM64 wheel in ${WHEELHOUSE}, or set its *_WHEEL variable." >&2
        exit 1
    fi
    if [[ "${wheel_name}" != *cp310* || "${wheel_name}" != *aarch64* ]]; then
        echo "Unexpected Jetson wheel filename: ${wheel_name}" >&2
        exit 1
    fi
    echo "${checksum}  ${wheel}" | sha256sum --check --status
    unzip -p "${wheel}" '*.dist-info/WHEEL' | grep -Eq '^Tag: cp310-cp310-(linux|manylinux).*aarch64$'
}

verify_jetson_wheel "${TORCH_WHEEL}" "${TORCH_SHA256}"
verify_jetson_wheel "${TORCHVISION_WHEEL}" "${TORCHVISION_SHA256}"

pip_install() {
    if command -v "${UV_BIN}" >/dev/null 2>&1; then
        "${UV_BIN}" pip install --python "${PYTHON_BIN}" "$@"
    else
        "${PYTHON_BIN}" -m pip install "$@"
    fi
}

install_common() {
    local mirror="$1"
    pip_install --upgrade --no-cache-dir --no-deps \
        --index-url "${mirror}" \
        -r "${PROJECT_ROOT}/requirements/jetson-jp62-runtime.txt"
    pip_install --upgrade --no-cache-dir --index-url "${mirror}" \
        pin==4.1.0 filelock 'typing-extensions>=4.10.0' 'setuptools<82' \
        'sympy>=1.13.3' networkx jinja2 fsspec matplotlib pillow requests scipy psutil polars \
        'nvidia-ml-py>=12.0.0' 'openai-clip==1.0.1'
    pip_install --no-deps --index-url "${mirror}" \
        'ultralytics==8.4.120' 'ultralytics-thop==2.1.6'
}

pip_install --no-deps "${TORCH_WHEEL}" "${TORCHVISION_WHEEL}"

if ! install_common "${PRIMARY_MIRROR}"; then
    echo "Primary mirror failed; retrying with USTC mirror." >&2
    install_common "${FALLBACK_MIRROR}"
fi

# Install the project package without importing or initializing hardware.
pip_install --no-deps --no-build-isolation --editable "${PROJECT_ROOT}"

download_mobileclip() {
    local target="${ULTRALYTICS_WEIGHTS_DIR}/${MOBILECLIP_FILE}"
    if [[ -s "${target}" && "$(stat -c '%s' "${target}")" -gt 100000 ]]; then
        echo "Using existing MobileCLIP weight: ${target}"
        return
    fi
    mkdir -p "${ULTRALYTICS_WEIGHTS_DIR}"
    local url
    local temporary="${target}.partial"
    for url in "${MOBILECLIP_MIRROR_URL}" "${MOBILECLIP_OFFICIAL_URL}"; do
        [[ -n "${url}" ]] || continue
        echo "Downloading ${MOBILECLIP_FILE} from ${url}"
        if command -v curl >/dev/null 2>&1; then
            if curl --fail --location --retry 2 --connect-timeout 10 --output "${temporary}" "${url}"; then
                if [[ "$(stat -c '%s' "${temporary}")" -gt 100000 ]]; then
                    mv -f "${temporary}" "${target}"
                    return
                fi
            fi
        else
            if "${PYTHON_BIN}" - "${temporary}" "${url}" <<'PY'
import sys
import urllib.request

urllib.request.urlretrieve(sys.argv[2], sys.argv[1])
PY
            then
                if [[ "$(stat -c '%s' "${temporary}")" -gt 100000 ]]; then
                    mv -f "${temporary}" "${target}"
                    return
                fi
            fi
        fi
        rm -f "${temporary}"
    done
    echo "Unable to download ${MOBILECLIP_FILE}; set MOBILECLIP_MIRROR_URL or place it at ${target}." >&2
    exit 1
}

download_mobileclip

"${PYTHON_BIN}" - <<'PY'
import importlib
from pathlib import Path

import numpy as np
import torch
import ultralytics
import pinocchio as pin
import motorbridge
import pyrealsense2

from m20_rebot_moving_grasp.common.model_assets import require_mobileclip_weight
from m20_rebot_moving_grasp.common.torch_numpy import install_torch_numpy_compat

install_torch_numpy_compat(torch)
torch.from_numpy(np.zeros((2, 2), dtype=np.float32))
mobileclip = require_mobileclip_weight()
if not Path(mobileclip).is_file():
    raise SystemExit(f"MobileCLIP weight missing: {mobileclip}")

print(f"torch={torch.__version__}")
print(f"torch_cuda_available={torch.cuda.is_available()}")
print(f"torch_cuda_version={torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch CUDA is unavailable")
device = torch.ones((32, 32), device="cuda")
torch.cuda.synchronize()
print(f"torch_cuda_smoke_sum={(device @ device).sum().item():.1f}")
print(f"ultralytics={ultralytics.__version__}; YOLO={ultralytics.YOLO.__name__}")
print(f"pinocchio={pin.__version__}")
print(f"motorbridge={getattr(motorbridge, '__version__', 'unknown')}")
print(f"pyrealsense2={pyrealsense2.__version__}")
PY
