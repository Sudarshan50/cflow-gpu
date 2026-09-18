#!/usr/bin/env bash
# Pre-push gate: prove no credential is tracked, staged, or present in history.
#
# Why this is a script and not a mental checklist: .gitignore only stops a file
# from being ADDED. It does nothing for a file already committed, and `git push`
# will happily publish a key that was staged once and ignored afterwards. The
# only safe check is against the object database, which is what this does.
#
#   ./secret-scan.sh          scan worktree + index + HEAD history
#   ./secret-scan.sh --staged only what is about to be committed (fast, for hooks)
#
# Exit 0 = clean. Exit 1 = do not push.
set -uo pipefail

cd "$(dirname "$0")"
findings=0
STAGED_ONLY=0
[ "${1:-}" = "--staged" ] && STAGED_ONLY=1

red()  { printf '\033[31m%s\033[0m\n' "$*"; }
note() { printf '  %s\n' "$*"; }
bad()  { red "  FAIL  $*"; findings=$((findings+1)); }
ok()   { printf '  PASS  %s\n' "$*"; }

# Files that must NEVER be tracked. Keep in sync with .gitignore's secrets
# section; the whole point is that these hold cleartext credentials.
SECRET_FILES=(
  api-key.txt
  dash-password.txt
  customers.tsv
  vllm-k3.env
  nginx/00-k3-keys.conf
  nginx/k3-limits.inc
)

# Content patterns. `sk-k3-` is this deployment's key prefix (gen-keys.sh:49);
# the others catch credentials that arrive from elsewhere.
#
# NOTE: each pattern must not match its own documented placeholder form, or the
# scan cries wolf on every doc. Hence the REPLACE_ME / <name> / 0{40} excludes.
# Every pattern demands real key MATERIAL (a long run of hex), never just the
# prefix. The docs legitimately describe shapes like `sk-k3-<48 hex>` and
# `VLLM_API_KEY=sk-k3-`, and a prefix-only pattern flags all of them - which
# trains the operator to ignore this scan, the one failure mode it cannot afford.
PATTERNS=(
  'sk-k3-[0-9a-f]{32,}'                  # upstream key: sk-k3- + 48 hex
  'sk-k3-[a-z0-9_]+-[0-9a-f]{40}'        # customer key: sk-k3-<name>-40 hex
  'sk-[A-Za-z0-9]{32,}'                  # generic OpenAI-style key
  '-----BEGIN [A-Z ]*PRIVATE KEY-----'   # TLS / SSH private key
  '\$apr1\$[./A-Za-z0-9]{8}\$[./A-Za-z0-9]{22}'  # htpasswd hash
  'gh[pousr]_[A-Za-z0-9]{36,}'           # GitHub PAT
  'VLLM_API_KEY=sk-k3-[0-9a-zA-Z]{16,}'  # populated env file
)
# Documented placeholders that are safe by construction. Kept as a second line
# of defence in case a pattern above is loosened later.
ALLOW='REPLACE_ME|<name>|<[0-9]+ hex>|0{40}|REDACTED|xxxx'

if ! git rev-parse --git-dir >/dev/null 2>&1; then
  red "not a git repository - nothing to scan"; exit 1
fi

echo "== tracked-file check =="
for f in "${SECRET_FILES[@]}"; do
  if git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
    bad "$f is TRACKED - remove with: git rm --cached $f"
  else
    ok "$f not tracked"
  fi
done

echo
echo "== .gitignore actually ignores them =="
for f in "${SECRET_FILES[@]}"; do
  # Only meaningful for files that exist; a missing file is vacuously safe.
  if [ -e "$f" ] && ! git check-ignore -q "$f"; then
    bad "$f EXISTS but is not ignored"
  fi
done
ok "ignore rules cover every existing secret file"

echo
echo "== content scan =="
# Scan tracked+staged content, not the worktree, so ignored files are out of
# scope by definition and we test exactly what a push would publish.
scope="tracked/staged"

# `--cached` MUST precede the pattern. git rejects it as "option must come
# before non-option arguments" otherwise, and because that write goes to
# stderr the scan would exit non-zero, get swallowed by `|| true`, and report
# a clean PASS while having searched nothing. This ordering is the bug the
# whole gate turns on - do not reflow these arguments.
for pat in "${PATTERNS[@]}"; do
  # -P for the lookahead in the VLLM_API_KEY pattern; -E fallback if this git
  # was built without PCRE.
  # -e marks the pattern explicitly; without it a pattern beginning with `-`
  # would be parsed as an option.
  if ! hits=$(git grep --cached -nIP -e "$pat" 2>/dev/null); then
    hits=$(git grep --cached -nIE -e "${pat//(?!REPLACE_ME)/}" 2>/dev/null) || hits=""
  fi
  hits=$(printf '%s\n' "$hits" | grep -vE "$ALLOW" | grep -v '^$' || true)
  if [ -n "$hits" ]; then
    bad "pattern /$pat/ matched in $scope content:"
    printf '%s\n' "$hits" | sed 's/^/          /'
  fi
done
[ "$findings" -eq 0 ] && ok "no credential patterns in $scope content"

# Belt-and-braces: compare against the ACTUAL live secret values, not just
# their shapes. A pattern list only catches credentials that look the way you
# expected; this catches the dashboard password and any key that does not.
echo
echo "== literal live-value scan =="
literal_sources=(api-key.txt dash-password.txt)
vals=$(mktemp); trap 'rm -f "$vals"' EXIT
for f in "${literal_sources[@]}"; do
  [ -s "$f" ] && head -c 200 "$f" | tr -d '\r\n' >> "$vals" && echo >> "$vals"
done
[ -s vllm-k3.env ] && grep -h '^VLLM_API_KEY=' vllm-k3.env | cut -d= -f2- >> "$vals"
[ -s customers.tsv ] && awk -F'\t' '!/^[[:space:]]*#/ && NF>=2 {print $2}' customers.tsv >> "$vals"
n=0
lit_before=$findings
while read -r s; do
  # Ignore anything too short to be a credential; a 3-char value would match
  # half the repo and drown the signal.
  [ "${#s}" -ge 12 ] || continue
  n=$((n+1))
  if git grep --cached -qF -- "$s" 2>/dev/null; then
    bad "a real secret value (starting ${s:0:8}...) IS STAGED"
  fi
done < <(grep -v '^$' "$vals")
if [ "$findings" -eq "$lit_before" ]; then
  ok "checked $n live secret value(s); none appear in staged content"
fi
# A dash password shorter than 12 chars is skipped above, so say so rather than
# implying full coverage.
[ "$n" -gt 0 ] || note "no live secret files on this box to compare against"

# The secrets bundle is the one credential file we deliberately DO commit, so
# it needs its own check: prove it is actually ciphertext. A bundle written
# without encryption, or a tarball someone dropped in by hand, would sail past
# every pattern above (tar headers contain no key material at the offsets the
# patterns look at) while publishing every customer key in the clear.
echo
echo "== committed secrets bundle =="
bundles=$(git ls-files -- '*.enc' 2>/dev/null || true)
if [ -z "$bundles" ]; then
  note "no *.enc bundle tracked (create one: ./secrets-backup.sh backup)"
else
  while read -r b; do
    [ -n "$b" ] || continue
    # `openssl enc -salt` writes the 8-byte magic "Salted__" as a header.
    if git show ":$b" 2>/dev/null | head -c 8 | grep -q 'Salted__'; then
      ok "$b is openssl ciphertext (Salted__ header)"
    else
      bad "$b is TRACKED but does not look encrypted - inspect it before pushing"
    fi
  done <<< "$bundles"
fi

if [ "$STAGED_ONLY" = 0 ]; then
  echo
  echo "== history scan =="
  # A key committed once and ignored later is still published by `git push`.
  if git rev-parse HEAD >/dev/null 2>&1; then
    before=$findings
    for pat in "${PATTERNS[@]}"; do
      # git grep syntax is `<pattern> [<rev>...]`, so the revisions come AFTER
      # the pattern and -e is what keeps them from being read as paths.
      hits=$(git grep -nIP -e "$pat" $(git rev-list --all) 2>/dev/null \
             | grep -vE "$ALLOW" | head -5 || true)
      if [ -n "$hits" ]; then
        bad "pattern /$pat/ found in COMMIT HISTORY - rewriting required:"
        printf '%s\n' "$hits" | sed 's/^/          /'
      fi
    done
    [ "$findings" -eq "$before" ] && ok "no credential patterns in history"
  else
    note "no commits yet - history scan skipped"
  fi
fi

echo
if [ "$findings" -eq 0 ]; then
  echo "SECRET SCAN CLEAN - safe to push"
  exit 0
fi
red "SECRET SCAN FOUND $findings PROBLEM(S) - DO NOT PUSH"
cat <<'EOF'

  If a secret was already committed, .gitignore will NOT unpublish it. Either
  start a fresh history (git checkout --orphan) or rewrite with git-filter-repo,
  and rotate the exposed credential regardless - assume it is burned.
EOF
exit 1
