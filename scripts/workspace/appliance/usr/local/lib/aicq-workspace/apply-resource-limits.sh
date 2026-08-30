#!/bin/bash
set -euo pipefail

if [[ "$(id -u)" != 0 ]]; then
    echo "apply-resource-limits must run as root" >&2
    exit 1
fi

manifest=/opt/aicq-workspace/protocol-manifest.json
mapfile -t limits < <(/usr/bin/python3 - "$manifest" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
values = manifest["limits"]
print(int(values["cpus"]))
print(int(values["memory_bytes"]))
print(int(values["pids"]))
PY
)

cpus=${limits[0]}
memory_bytes=${limits[1]}
pids=${limits[2]}
if (( cpus < 1 || memory_bytes < 1 || pids < 1 )); then
    echo "Invalid Agent computer resource limits" >&2
    exit 1
fi

# The rootless container is started by the broker and remains in its delegated
# system cgroup. Applying limits to that cgroup works without a user D-Bus and
# covers the container, conmon, network helper, and broker as one bounded unit.
/bin/systemctl daemon-reload
/bin/systemctl set-property aicq-workspace-broker.service \
    "CPUQuota=${cpus}00%" \
    "MemoryMax=$memory_bytes" \
    "MemorySwapMax=$memory_bytes" \
    "TasksMax=$pids"
