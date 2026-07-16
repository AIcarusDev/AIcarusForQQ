#!/bin/bash
set -euo pipefail

uid="$(id -u aicqws)"
runtime_dir=/run/aicq-workspace
preview_port_file=/home/aicqws/.config/aicq-workspace/preview-port
rules_file="$(mktemp)"
trap 'rm -f "$rules_file"' EXIT

preview_port=
if [[ -f "$preview_port_file" ]]; then
    read -r preview_port <"$preview_port_file"
    if [[ ! "$preview_port" =~ ^[0-9]{1,5}$ ]] || (( preview_port < 1 || preview_port > 65535 )); then
        echo "Invalid workspace preview port file" >&2
        exit 1
    fi
fi

mapfile -t resolvers < <(awk '/^nameserver[[:space:]]+/ { print $2 }' /etc/resolv.conf | sort -u)
if ((${#resolvers[@]} == 0)); then
    echo "No DNS resolver found; refusing to install workspace firewall" >&2
    exit 1
fi

{
    echo 'table inet aicq_workspace {'
    echo '  set blocked_ipv4 {'
    echo '    type ipv4_addr'
    echo '    flags interval'
    echo '    elements = { 0.0.0.0/8, 10.0.0.0/8, 100.64.0.0/10, 127.0.0.0/8, 169.254.0.0/16, 172.16.0.0/12, 192.168.0.0/16, 224.0.0.0/4, 240.0.0.0/4 }'
    echo '  }'
    echo '  set blocked_ipv6 {'
    echo '    type ipv6_addr'
    echo '    flags interval'
    echo '    elements = { ::/128, ::1/128, 64:ff9b:1::/48, 100::/64, 2001:db8::/32, fc00::/7, fe80::/10, ff00::/8 }'
    echo '  }'
    echo '  chain output {'
    echo '    type filter hook output priority filter; policy accept;'
    if [[ -n "$preview_port" ]]; then
        printf '    meta skuid %s ip saddr 127.0.0.1 tcp sport %s ct state established counter accept comment "aicq-preview-loopback-return"\n' "$uid" "$preview_port"
        printf '    meta skuid %s ip daddr 127.0.0.1 tcp dport %s counter accept comment "aicq-preview-loopback"\n' "$uid" "$preview_port"
    fi
    for resolver in "${resolvers[@]}"; do
        if [[ "$resolver" == *:* ]]; then
            printf '    meta skuid %s ip6 daddr %s udp dport 53 counter accept comment "aicq-dns-v6"\n' "$uid" "$resolver"
            printf '    meta skuid %s ip6 daddr %s tcp dport 53 counter accept comment "aicq-dns-v6"\n' "$uid" "$resolver"
        else
            printf '    meta skuid %s ip daddr %s udp dport 53 counter accept comment "aicq-dns-v4"\n' "$uid" "$resolver"
            printf '    meta skuid %s ip daddr %s tcp dport 53 counter accept comment "aicq-dns-v4"\n' "$uid" "$resolver"
        fi
    done
    printf '    meta skuid %s ip daddr @blocked_ipv4 counter reject comment "aicq-block-private-v4"\n' "$uid"
    printf '    meta skuid %s ip6 daddr @blocked_ipv6 counter reject comment "aicq-block-private-v6"\n' "$uid"
    echo '  }'
    echo '}'
} >"$rules_file"

nft list table inet aicq_workspace >/dev/null 2>&1 && nft delete table inet aicq_workspace
nft -f "$rules_file"
nft list table inet aicq_workspace >/dev/null
install -d -m 0700 -o aicqws -g aicqws "$runtime_dir" "$runtime_dir/runtime"
touch "$runtime_dir/firewall.ready"
chown aicqws:aicqws "$runtime_dir/firewall.ready"
chmod 0600 "$runtime_dir/firewall.ready"
