#!/usr/bin/env bash
# Encrypted backup and restore of the credentials deploy.sh cannot regenerate.
# A minted key is not the one the customer holds, so identity has to survive a
# reclaim rather than be recreated.
#
#   ./secrets-backup.sh backup [outfile]   encrypt the live secrets
#   ./secrets-backup.sh verify <file>      list contents without writing
#   ./secrets-backup.sh restore <file>     decrypt into place, correct modes
#
# Passphrase is prompted, or taken from $SECRETS_PASSPHRASE. Losing it is
# equivalent to losing the backup.
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"
REPO="$PWD"

# vllm-k3.env belongs here because its VLLM_API_KEY must equal api-key.txt;
# restoring one without the other 401s every request past nginx.
FILES=(api-key.txt dash-password.txt customers.tsv vllm-k3.env)

# High iteration count: this bundle is expected to sit in a git remote.
ENC_ARGS=(-aes-256-cbc -pbkdf2 -iter 600000 -salt)

die()  { echo "secrets-backup: $*" >&2; exit 1; }
log()  { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

get_pass() {
  if [ -n "${SECRETS_PASSPHRASE:-}" ]; then printf '%s' "$SECRETS_PASSPHRASE"; return; fi
  local p1 p2
  read -rsp "passphrase: " p1 </dev/tty; echo >/dev/tty
  if [ "${1:-}" = confirm ]; then
    read -rsp "confirm:    " p2 </dev/tty; echo >/dev/tty
    [ "$p1" = "$p2" ] || die "passphrases do not match"
    [ "${#p1}" -ge 12 ] || die "passphrase must be at least 12 characters"
  fi
  printf '%s' "$p1"
}

cmd="${1:-}"
case "$cmd" in

backup)
  out="${2:-/root/k3-secrets-$(date -u +%Y%m%d).enc}"
  missing=()
  for f in "${FILES[@]}"; do [ -s "$REPO/$f" ] || missing+=("$f"); done
  [ "${#missing[@]}" -eq 0 ] || die "missing on this box: ${missing[*]}"

  # Refuse to write the bundle inside the working tree unless asked twice.
  # An encrypted blob is still a credential, and `git add -A` does not ask.
  case "$(readlink -f "$out")" in
    "$REPO"/*) [ "${ALLOW_IN_REPO:-0}" = 1 ] ||
      die "refusing to write inside the repo ($out). Re-run with ALLOW_IN_REPO=1 if you
     intend to commit the encrypted bundle, and make sure .gitignore does not
     exclude it." ;;
  esac

  pass="$(get_pass confirm)"
  umask 077
  tar -cz -C "$REPO" "${FILES[@]}" \
    | openssl enc "${ENC_ARGS[@]}" -pass fd:3 -out "$out" 3<<<"$pass"
  chmod 600 "$out"
  unset pass

  n=$(awk -F'\t' '!/^[[:space:]]*#/&&NF>=2{n++} END{print n+0}' "$REPO/customers.tsv")
  log "wrote $out ($(du -h "$out" | cut -f1)), ${#FILES[@]} files, $n customer key(s)"
  cat <<EOF

  This file is now the only copy of those keys that survives the droplet.
  Put it somewhere that is NOT this box:
    - a password manager attachment, or
    - the private GitHub repo (it is encrypted; see docs/SECRETS.md), or
    - object storage you control.
  Losing both this file and the box means every customer must be re-keyed.
EOF
  ;;

verify)
  in="${2:?usage: secrets-backup.sh verify <file>}"
  [ -s "$in" ] || die "no such file: $in"
  pass="$(get_pass)"
  openssl enc -d "${ENC_ARGS[@]}" -pass fd:3 -in "$in" 3<<<"$pass" \
    | tar -tz || die "decryption failed - wrong passphrase or corrupt file"
  unset pass
  log "bundle decrypts cleanly"
  ;;

restore)
  in="${2:?usage: secrets-backup.sh restore <file>}"
  [ -s "$in" ] || die "no such file: $in"
  pass="$(get_pass)"

  # Decrypt to a staging dir first: a wrong passphrase must not leave the box
  # with half its credentials replaced.
  tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
  openssl enc -d "${ENC_ARGS[@]}" -pass fd:3 -in "$in" 3<<<"$pass" \
    | tar -xz -C "$tmp" || die "decryption failed - wrong passphrase or corrupt file"
  unset pass

  for f in "${FILES[@]}"; do
    [ -s "$tmp/$f" ] || die "bundle is missing $f; refusing a partial restore"
  done

  # The invariant that causes the most confusing outage if broken.
  up="$(tr -d '\r\n' < "$tmp/api-key.txt")"
  env_key="$(sed -n 's/^VLLM_API_KEY=//p' "$tmp/vllm-k3.env" | tr -d '\r\n')"
  [ "$up" = "$env_key" ] ||
    die "bundle is inconsistent: VLLM_API_KEY != api-key.txt. Restoring it would
     make nginx forward a credential vLLM rejects, 401ing every request."

  umask 077
  for f in "${FILES[@]}"; do
    [ -e "$REPO/$f" ] && cp -p "$REPO/$f" "$REPO/$f.pre-restore.bak"
    install -m 600 "$tmp/$f" "$REPO/$f"
  done

  n=$(awk -F'\t' '!/^[[:space:]]*#/&&NF>=2{n++} END{print n+0}' "$REPO/customers.tsv")
  log "restored ${#FILES[@]} files, $n customer key(s)"
  cat <<'EOF'

  Next, rebuild the nginx auth layer from the restored table and restart the
  engine so it picks up the restored upstream key:
      ./gen-keys.sh                 # regenerates 00-k3-keys.conf + k3-limits.inc
      systemctl restart k3.service  # only if api-key.txt changed (~6 min)
      ./verify-auth.sh
EOF
  ;;

*)
  sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
  exit 1 ;;
esac
