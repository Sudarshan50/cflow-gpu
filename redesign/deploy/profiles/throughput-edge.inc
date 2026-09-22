# Admission and tenant quotas are enforced after authentication. Accommodate
# 64 engine requests + 128 admission waiters + bounded preparation/transport.
# Per-IP request-rate buckets are unsuitable for shared provider microbursts.
limit_conn k3_ip_conn 256;
limit_conn k3_conn 256;
limit_conn_status 503;
