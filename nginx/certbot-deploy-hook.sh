#!/usr/bin/env bash
# Reload nginx after certbot replaces the certificate.
# Installed by deploy.sh to /etc/letsencrypt/renewal-hooks/deploy/reload-nginx.sh.
#
# issue-cert.sh uses `certonly --webroot`, so the renewal has no installer and
# certbot never reloads nginx. Without this hook nginx keeps serving the old
# leaf until it expires, despite a valid cert sitting on disk.
#
# reload, not restart: a 256k prompt can stream for minutes.
set -euo pipefail

# certbot sets this only for deploy hooks; guards against a manual run
# reloading against a half-written cert.
[ -n "${RENEWED_LINEAGE:-}" ] || {
  echo "certbot-deploy-hook: RENEWED_LINEAGE unset; refusing to reload" >&2
  exit 0
}

# Fail loudly: a rejected config would otherwise leave the old workers serving
# the expiring cert with no signal.
if ! nginx -t 2>/dev/null; then
  echo "certbot-deploy-hook: nginx -t FAILED after renewing ${RENEWED_LINEAGE}; not reloading" >&2
  nginx -t 2>&1 | tail -5 >&2
  exit 1
fi

systemctl reload nginx
echo "certbot-deploy-hook: nginx reloaded for ${RENEWED_LINEAGE}"
