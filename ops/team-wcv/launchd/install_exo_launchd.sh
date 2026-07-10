#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
label="com.teamwcv.exo.bigbrain"
script_src="$script_dir/teamwcv-exo-bigbrain"
plist_template="$script_dir/${label}.plist.template"
script_dst="$HOME/bin/teamwcv-exo-bigbrain"
plist_dst="$HOME/Library/LaunchAgents/${label}.plist"
uid="$(id -u)"

mkdir -p "$HOME/bin" "$HOME/Library/LaunchAgents" "$HOME/.cache/exo"
install -m 0755 "$script_src" "$script_dst"
sed "s#__HOME__#$HOME#g" "$plist_template" > "$plist_dst"

launchctl bootout "gui/${uid}" "$plist_dst" >/dev/null 2>&1 || true
launchctl bootstrap "gui/${uid}" "$plist_dst"
launchctl enable "gui/${uid}/${label}"
launchctl kickstart -k "gui/${uid}/${label}"

echo "Installed ${label}"
echo "Logs: $HOME/.cache/exo/bigbrain-launchd.err.log"
