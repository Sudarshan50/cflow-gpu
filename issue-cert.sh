#!/usr/bin/env bash
# Issue a Let's Encrypt cert for cflox.store and put the API on TLS/443.
#
# Safe to re-run. Does nothing unless DNS actually points here, because a
# failed HTTP-01 burns Let's Encrypt rate limit (5 failures/hostname/hour).
#
#   ./issue-cert.sh          issue once, if DNS is ready
#   ./issue-cert.sh --watch  poll DNS every 60s, issue as soon as it resolves
set -uo pipefail

DOMAIN="${DOMAIN:-cflox.store}"
WWW_OK=1
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
KEYFILE=/scratch/deploy/api-key.txt

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
    local WWW_SUFFIX=""
    [ "${WWW_OK:-1}" = "1" ] && WWW_SUFFIX=" www.${DOMAIN}"
    # Only called once fullchain.pem exists; nginx -t fails on a missing cert.
    # Keep this block auth-free: the $k3_customer map lives in 00-k3-keys.conf,
    # and redeclaring map_hash_bucket_size here fails nginx -t as a duplicate.
    cat > "$TLSCONF" <<EOF
server {
    listen 80;
    listen [::]:80;
    server_name ${DOMAIN}${WWW_SUFFIX};
    location /.well-known/acme-challenge/ { root /var/www/certbot; access_log off; }
    location / { return 301 https://\$host\$request_uri; }
}

server {
    # nginx 1.24 rejects the standalone "http2 on;" added in 1.25.1.
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name ${DOMAIN}${WWW_SUFFIX};

    ssl_certificate     /etc/letsencrypt/live/${DOMAIN}/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/${DOMAIN}/privkey.pem;
    ssl_protocols       TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers off;
    ssl_session_cache   shared:SSL:10m;
    ssl_session_timeout 1d;
    add_header Strict-Transport-Security "max-age=31536000" always;

    client_max_body_size 256m;
    client_body_buffer_size 1m;

    # Same allowlist as the :8000 edge. vLLM's --api-key only guards
    # ("/v1","/v2","/inference"), leaving POST /invocations (verified 200 with
    # no credentials) and /scale_elastic_ep open. This is what closes them.
    location = /metrics { allow 127.0.0.1; deny all; access_log off; proxy_pass http://k3_backend; }
    location = /health  { allow 127.0.0.1; deny all; access_log off; proxy_pass http://k3_backend; }

    location /v1/ {
        if (\$k3_customer = "") { return 401 '{"error":{"message":"invalid or missing api key","type":"authentication_error"}}'; }
        default_type application/json;
        include /etc/nginx/conf.d/k3-limits.inc;
        # NOT under /var/log/nginx/: the nginx package's logrotate glob would
        # take this file at rotate 14 and delete the billing record after two
        # weeks. See nginx/k3-usage.logrotate.
        access_log /var/log/k3/usage.log k3usage;

        proxy_pass http://k3_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        # 12k-in/512-out runs ~112s; the 60s default would cut streams off.
        proxy_buffering off;
        proxy_cache off;
        proxy_request_buffering off;
        proxy_read_timeout 900s;
        proxy_send_timeout 900s;
    }

    error_page 429 = @throttled;
    location @throttled {
        add_header Retry-After 2 always;
        default_type application/json;
        return 429 '{"error":{"message":"rate limit exceeded for this api key","type":"rate_limit_error","code":429}}';
    }

    error_page 413 = @too_large;
    location @too_large {
        default_type application/json;
        return 413 '{"error":{"message":"request body exceeds 256 MB; send a smaller image or pass an image_url instead of base64","type":"invalid_request_error","code":413}}';
    }

    include /etc/nginx/conf.d/k3-dash.inc;

    location / { return 403 '{"error":{"message":"forbidden","type":"invalid_request_error"}}'; default_type application/json; }
}
EOF
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

    cp "$TLSCONF" "${TLSCONF}.pre-tls.bak" 2>/dev/null
    write_tls_server
    if ! nginx -t 2>&1 | tail -1; then
        log "nginx config invalid - rolling back"
        mv "${TLSCONF}.pre-tls.bak" "$TLSCONF"; nginx -t >/dev/null 2>&1; return 1
    fi
    systemctl reload nginx && log "nginx reloaded with TLS on 443"

    log "verifying"
    local k; k=$(cat "$KEYFILE")
    printf '  https /v1/models no key    -> %s (want 401)\n' \
      "$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 "https://${DOMAIN}/v1/models")"
    printf '  https /v1/models with key  -> %s (want 200)\n' \
      "$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 "https://${DOMAIN}/v1/models" -H "Authorization: Bearer $k")"
    printf '  https /invocations no key   -> %s (want 403)\n' \
      "$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 -X POST "https://${DOMAIN}/invocations" -d '{}')"
    printf '  http  -> https redirect    -> %s\n' \
      "$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 "http://${DOMAIN}/v1/models")"
    printf '  cert expires               : %s\n' \
      "$(openssl x509 -enddate -noout -in "/etc/letsencrypt/live/${DOMAIN}/fullchain.pem" | cut -d= -f2)"
    printf '  auto-renew timer           : %s\n' "$(systemctl is-enabled certbot.timer 2>/dev/null)"
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
