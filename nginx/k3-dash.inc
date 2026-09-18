    # Mounted at "/" not "/dash/": index.html fetches the absolute path
    # /api/state, which a prefix would break.
    location = / {
        auth_basic "k3 dashboard";
        auth_basic_user_file /etc/nginx/k3-dash.htpasswd;
        proxy_pass http://k3_dash;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /api/ {
        auth_basic "k3 dashboard";
        auth_basic_user_file /etc/nginx/k3-dash.htpasswd;
        proxy_pass http://k3_dash;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_buffering off;
        proxy_read_timeout 60s;
    }
