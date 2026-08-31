#!/bin/bash
set -euo pipefail

if [[ "$(id -u)" != 0 ]]; then
    echo "apply-resource-limits must run as root" >&2
    exit 1
fi

mode=${1:---apply}
if [[ "$mode" != --apply && "$mode" != --ensure ]]; then
    echo "Usage: apply-resource-limits.sh [--apply|--ensure]" >&2
    exit 2
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

limits_match() {
    local control_group cgroup_root
    local cpu_quota cpu_period observed_memory observed_memory_swap observed_pids

    control_group="$(/bin/systemctl show --property=ControlGroup --value aicq-workspace-broker.service)" || return 1
    if [[ "$control_group" != /* || "$control_group" == *"/../"* || "$control_group" == */.. ]]; then
        return 1
    fi
    cgroup_root="/sys/fs/cgroup${control_group}"
    [[ -r "$cgroup_root/cpu.max" \
        && -r "$cgroup_root/memory.max" \
        && -r "$cgroup_root/memory.swap.max" \
        && -r "$cgroup_root/pids.max" ]] || return 1

    read -r cpu_quota cpu_period <"$cgroup_root/cpu.max" || return 1
    observed_memory="$(<"$cgroup_root/memory.max")" || return 1
    observed_memory_swap="$(<"$cgroup_root/memory.swap.max")" || return 1
    observed_pids="$(<"$cgroup_root/pids.max")" || return 1

    [[ "$cpu_quota" == "$((cpus * 100000))" \
        && "$cpu_period" == 100000 \
        && "$observed_memory" == "$memory_bytes" \
        && "$observed_memory_swap" == "$memory_bytes" \
        && "$observed_pids" == "$pids" ]]
}

if [[ "$mode" == --ensure ]] && limits_match; then
    exit 0
fi

# The rootless container is started by the broker and remains in its delegated
# system cgroup. Applying limits to that cgroup works without a user D-Bus and
# covers the container, conmon, network helper, and broker as one bounded unit.
# Reapplying an unchanged property is intentional: after a systemd manager
# re-exec, a delegated controller can disappear from the live parent cgroup
# while its persisted property still looks correct. set-property re-realizes
# the controller and writes the configured limits back to the kernel.
/bin/systemctl daemon-reload
/bin/systemctl set-property aicq-workspace-broker.service \
    "CPUQuota=${cpus}00%" \
    "MemoryMax=$memory_bytes" \
    "MemorySwapMax=$memory_bytes" \
    "TasksMax=$pids"

if ! limits_match; then
    echo "Resource limits do not match after applying them" >&2
    exit 1
fi
