#!/usr/bin/env bash
set -euo pipefail

# /home/agent is a persistent bind mount, so the image's home directory is
# hidden at runtime. Seed only missing standard Ubuntu user files and never
# replace anything the agent already owns.
install -d -m 0700 "$HOME"
shopt -s dotglob nullglob
for source in /etc/skel/*; do
  target="$HOME/${source##*/}"
  if [[ ! -e "$target" && ! -L "$target" ]]; then
    cp -a "$source" "$target"
  fi
done

exec /usr/bin/tini -- "$@"
