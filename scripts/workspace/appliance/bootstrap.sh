#!/bin/bash
set -euo pipefail

if [[ "$(id -u)" != 0 ]]; then
    echo "bootstrap must run as root" >&2
    exit 1
fi

stage="$(cd "$(dirname "$0")" && pwd)"
export DEBIAN_FRONTEND=noninteractive

packages=(ca-certificates curl python3 podman uidmap fuse-overlayfs passt slirp4netns nftables dbus-user-session)
missing_packages=0
for package in "${packages[@]}"; do
    if ! dpkg-query -W -f='${Status}\n' "$package" 2>/dev/null | grep -qx 'install ok installed'; then
        missing_packages=1
        break
    fi
done

if (( missing_packages )); then
    attempt=1
    while :; do
        rm -f /var/cache/apt/archives/*.deb /var/cache/apt/archives/partial/*
        if timeout --signal=TERM 300 apt-get -o Acquire::Retries=5 -o Acquire::ForceIPv4=true -o Acquire::http::Timeout=30 -o Acquire::http::No-Cache=true update \
            && timeout --signal=TERM 1500 apt-get -o Acquire::Retries=5 -o Acquire::ForceIPv4=true -o Acquire::http::Timeout=30 -o Acquire::http::No-Cache=true install -y --no-install-recommends "${packages[@]}"; then
            break
        fi
        if (( attempt >= 3 )); then
            echo '[computer] Appliance package bootstrap failed after 3 clean downloads' >&2
            exit 1
        fi
        delay=$((attempt * 3))
        echo "[computer][retry] Appliance package download or integrity check failed; discarding cached packages and retrying in ${delay}s (attempt $((attempt + 1))/3)"
        dpkg --configure -a || true
        sleep "$delay"
        attempt=$((attempt + 1))
    done
    apt-get clean
    rm -rf /var/lib/apt/lists/*
else
    echo '[computer] Appliance system packages are already installed; skipping APT refresh'
fi
# Ubuntu enables a rootful Podman API socket as a package side effect.  The
# workspace uses only daemonless rootless Podman, so keep that socket absent.
systemctl disable --now \
    podman.socket podman.service podman-auto-update.timer podman-auto-update.service \
    podman-clean-transient.service podman-restart.service >/dev/null 2>&1 || true
systemctl mask podman.socket podman.service >/dev/null 2>&1 || true

if ! id aicqws >/dev/null 2>&1; then
    useradd --create-home --shell /usr/sbin/nologin --user-group aicqws
fi
passwd -l aicqws >/dev/null 2>&1 || true
gpasswd -d aicqws sudo >/dev/null 2>&1 || true
install -d -m 0755 /var/lib/systemd/linger
touch /var/lib/systemd/linger/aicqws
chmod 0644 /var/lib/systemd/linger/aicqws

sed -i '/^aicqws:/d' /etc/subuid /etc/subgid
echo 'aicqws:100000:65536' >>/etc/subuid
echo 'aicqws:100000:65536' >>/etc/subgid

install -d -m 0755 /opt/aicq-workspace /usr/local/lib/aicq-workspace
cp -a "$stage/opt/aicq-workspace/." /opt/aicq-workspace/
install -m 0755 "$stage/opt/aicq-workspace/broker.py" /opt/aicq-workspace/broker.py
install -m 0755 "$stage/opt/aicq-workspace/bridge.py" /opt/aicq-workspace/bridge.py
install -m 0755 "$stage/opt/aicq-workspace/browser-connect.py" /opt/aicq-workspace/browser-connect.py
install -m 0755 "$stage/opt/aicq-workspace/provision-container.sh" /opt/aicq-workspace/provision-container.sh
install -m 0755 "$stage/opt/aicq-workspace/apply-container-settings.sh" /opt/aicq-workspace/apply-container-settings.sh
install -m 0755 "$stage/usr/local/lib/aicq-workspace/apply-firewall.sh" /usr/local/lib/aicq-workspace/apply-firewall.sh
install -m 0644 "$stage/etc/systemd/system/aicq-workspace-firewall.service" /etc/systemd/system/aicq-workspace-firewall.service
install -d -m 0755 /etc/systemd/user
install -m 0644 "$stage/etc/systemd/user/aicq-workspace-broker.service" /etc/systemd/user/aicq-workspace-broker.service
install -m 0644 "$stage/etc/wsl.conf" /etc/wsl.conf
ln -sf /opt/aicq-workspace/bridge.py /usr/local/bin/aicq-workspace-bridge
ln -sf /opt/aicq-workspace/browser-connect.py /usr/local/bin/aicq-workspace-browser-connect

install -d -m 0700 -o aicqws -g aicqws \
    /var/lib/aicq-workspace/home \
    /var/lib/aicq-workspace/commands \
    /home/aicqws/.config/containers \
    /home/aicqws/.local/share/containers/storage \
    /home/aicqws/.cache

cat >/home/aicqws/.config/containers/storage.conf <<'EOF'
[storage]
driver = "overlay"
graphroot = "/home/aicqws/.local/share/containers/storage"
runroot = "/run/aicq-workspace/runtime/containers"

[storage.options.overlay]
mount_program = "/usr/bin/fuse-overlayfs"
EOF

cat >/home/aicqws/.config/containers/containers.conf <<'EOF'
[engine]
events_logger = "file"
cgroup_manager = "systemd"

[network]
default_rootless_network_cmd = "pasta"
EOF

chown -R aicqws:aicqws /home/aicqws/.config /home/aicqws/.local /home/aicqws/.cache /var/lib/aicq-workspace
chmod 0700 /home/aicqws /home/aicqws/.config /home/aicqws/.local /home/aicqws/.cache /var/lib/aicq-workspace

rm -f /etc/systemd/system/aicq-workspace-broker.service \
    /etc/systemd/system/multi-user.target.wants/aicq-workspace-broker.service
install -d -m 0700 -o aicqws -g aicqws /home/aicqws/.config/systemd/user/default.target.wants
ln -sf /etc/systemd/user/aicq-workspace-broker.service \
    /home/aicqws/.config/systemd/user/default.target.wants/aicq-workspace-broker.service
chown -h aicqws:aicqws \
    /home/aicqws/.config/systemd/user/default.target.wants/aicq-workspace-broker.service

printf '%s\n' 'AICQ-Workspace appliance 5' >/etc/aicq-workspace-release
systemctl enable aicq-workspace-firewall.service
