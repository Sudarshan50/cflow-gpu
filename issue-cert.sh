#!/usr/bin/env bash
# Issue a Let's Encrypt cert for api.cflowx.in. Does not write the public
# /v1/ edge — that lives in /etc/nginx/conf.d/k3.conf (deploy.sh).
#
# Safe to re-run. Does nothing unless DNS actually points here, because a
# failed HTTP-01 burns Let's Encrypt rate limit (5 failures/hostname/hour).
#
#   ./issue-cert.sh          issue once, if DNS is ready
#   ./issue-cert.sh --watch  poll DNS every 60s, issue as soon as it resolves
set -uo pipefail

DOMAIN="${DOMAIN:-api.cflowx.in}"
WWW_OK=0
# A reclaimed spot droplet comes back with a DIFFERENT public IP. Hardcoding
# this made dns_ready() permanently false on any rebuilt box, so the cert would
# never issue and the failure looked like a DNS problem rather than a stale
# constant. Detect our own address; override with EXPECT_IP=... to pin it.
EXPECT_IP="${EXPECT_IP:-$(curl -s --max-time 10 https://api.ipify.org 2>/dev/null)}"
case "$EXPECT_IP" in
  *[!0-9.]*|'') echo "FATAL: could not determine this host's public IP; set EXPECT_IP=<addr>" >&2; exit 1;;
esac
EMAIL="admin@${DOMAIN}"
TLSCONF=/etc/nginx/conf.d/k3-tls.conf

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

dns_ready() {
    # Ask a public resolver, not the local cache, and require OUR ip.
    local got
    got=$(python3 - <<PY
import json,urllib.request
try:
    d=json.load(urllib.request.urlopen(
        "https://dns.google/resolve?name=${DOMAIN}&type=A", timeout=15))
    print(",".join(a["data"] for a in d.get("Answer",[]) if a.get("type")==1))
except Exception:
    print("")
PY
)
    [ -n "$got" ] && log "  ${DOMAIN} -> ${got}" || log "  ${DOMAIN} -> unresolved"
    [[ ",$got," == *",$EXPECT_IP,"* ]]
}

write_tls_server() {
    echo "FATAL: write_tls_server is retired. It proxied /v1/ to the engine" >&2
    echo "and included k3-limits.inc (Authorization swap). Live edge is" >&2
    echo "/etc/nginx/conf.d/k3.conf from redesign/deploy/deploy.sh." >&2
    return 1
}

issue() {
    # Try with www, but fall back to the bare domain: certbot fails the WHOLE
    # request if any single -d name fails HTTP-01, and www is often not set.
    log "requesting certificate for ${DOMAIN} (+www)"
    if ! certbot certonly --webroot -w /var/www/certbot \
            -d "$DOMAIN" -d "www.${DOMAIN}" \
            --non-interactive --agree-tos -m "$EMAIL" --keep-until-expiring; then
        log "www variant failed - retrying with ${DOMAIN} only"
        certbot certonly --webroot -w /var/www/certbot \
            -d "$DOMAIN" \
            --non-interactive --agree-tos -m "$EMAIL" --keep-until-expiring || {
                log "certbot FAILED - not touching nginx"; return 1; }
        # Drop the www server_name so nginx does not claim a name we cannot serve.
        WWW_OK=0
    fi

    [ -s "/etc/letsencrypt/live/${DOMAIN}/fullchain.pem" ] || {
        log "cert missing after certbot claimed success"; return 1; }

    rm -f "$TLSCONF"
    nginx -t && systemctl reload nginx
    log "certificate in place; /etc/nginx/conf.d/k3.conf remains the edge"
    log "verifying (no key must 401; do not print keys)"
    printf '  https /v1/models no key -> %s (want 401)\n' \
      "$(curl -sk -o /dev/null -w '%{http_code}' --max-time 20 --resolve "${DOMAIN}:443:127.0.0.1" "https://${DOMAIN}/v1/models")"
    printf '  cert expires            : %s\n' \
      "$(openssl x509 -enddate -noout -in "/etc/letsencrypt/live/${DOMAIN}/fullchain.pem" | cut -d= -f2)"
    log "DONE - https://${DOMAIN}/v1"
}

if [ "${1:-}" = "--watch" ]; then
    log "watching DNS for ${DOMAIN} -> ${EXPECT_IP} (checks every 60s)"
    for i in $(seq 1 240); do          # up to 4 hours
        if dns_ready; then log "DNS is correct"; issue && exit 0
             log "issuance failed; will not retry automatically"; exit 1
        fi
        sleep 60
    done
    log "gave up after 4h - DNS never pointed here"; exit 1
else
    dns_ready && issue || { log "DNS not pointing at ${EXPECT_IP} yet; not calling certbot"; exit 1; }
fi
