#!/usr/bin/env bash
# Reload nginx after certbot replaces the certificate.
#
# Installed by deploy.sh to /etc/letsencrypt/renewal-hooks/deploy/reload-nginx.sh.
# Everything in that directory runs after a SUCCESSFUL renewal, and only then.
#
# Why this is required rather than nice to have: issue-cert.sh requests the cert
# with `certonly --webroot`, which means the renewal config has an
# `authenticator` but NO `installer` (/etc/letsencrypt/renewal/<domain>.conf).
# certbot therefore rewrites the files under /etc/letsencrypt/live/ and stops.
# nginx holds the certificate it parsed at its last reload and keeps serving it,
# so renewal succeeds while clients continue to receive the OLD leaf - right up
# until it expires, at which point every customer gets a TLS error even though a
# valid certificate has been sitting on disk for weeks.
#
# `reload` is deliberate, not `restart`: reload re-reads the cert in new workers
# and drains the old ones, so in-flight completions are not cut off. A 256k
# prompt can stream for minutes and a restart would kill it.
set -euo pipefail

# certbot exports RENEWED_LINEAGE for deploy hooks. Guard on it so a manual
# invocation cannot reload nginx against a half-written cert.
[ -n "${RENEWED_LINEAGE:-}" ] || {
  echo "certbot-deploy-hook: RENEWED_LINEAGE unset; refusing to reload" >&2
  exit 0
}

# Fail closed. A reload against a config nginx rejects would leave the old
# workers serving the expiring cert with no signal that anything is wrong, so
# make the renewal loud instead.
if ! nginx -t 2>/dev/null; then
  echo "certbot-deploy-hook: nginx -t FAILED after renewing ${RENEWED_LINEAGE}; not reloading" >&2
  nginx -t 2>&1 | tail -5 >&2
  exit 1
fi

systemctl reload nginx
echo "certbot-deploy-hook: nginx reloaded for ${RENEWED_LINEAGE}"
