# Snapshots — the fast path to a running box

Since 2026-09-18 everything the service needs lives on the **boot disk**, so a
DigitalOcean droplet snapshot is self-contained. Restoring one gives you a box
that serves in about five minutes without downloading anything.

There are two ways to stand this service up. Use the snapshot path routinely;
the full pipeline in [`DEPLOY.md`](DEPLOY.md) is how you build the box that
becomes the first snapshot, and the fallback when no usable snapshot exists.

| | Snapshot restore | Full pipeline |
|---|---|---|
| Time to serving | ~5 min + DNS | under an hour + DNS |
| Downloads | none | 1.5 TB weights, 57 GB image |
| Needs the GitHub PAT | no | yes |
| Needs the bundle passphrase | no | yes |
| Customer keys preserved | yes, automatically | yes, via `k3-secrets.enc` |

---

## 1. What a snapshot contains

Everything below is on `/dev/vda1`, so all of it is captured:

| | Path |
|---|---|
| Model weights, 1.5 TB | `/scratch/hf` |
| Container image, 57 GB | `/var/lib/docker` |
| Repo, secrets, customer keys | `/scratch/deploy` |
| TLS certificate and account | `/etc/letsencrypt` |
| nginx config and auth maps | `/etc/nginx` |
| systemd units, all enabled | `/etc/systemd/system` |
| Dashboard | `/opt/k3dash` |

Because the units are enabled, a restored droplet starts the whole stack on
boot with no intervention.

**Not captured:** the 40 TB `DOSCRATCH` volume, which is a separate device. It
is mounted at `/mnt/bulk` and nothing depends on it. Its fstab entry is
`noauto,nofail` precisely so a restored droplet that has no such volume boots
normally instead of stalling.

---

## 2. Taking a snapshot

```bash
cd /scratch/deploy
sudo ./snapshot-prep.sh
```

That stops the engine and dashboard, flushes to disk, and trims free space so
the snapshot is no larger than it needs to be. Then, in the Control Panel:
power the droplet off, create the snapshot, power it back on. If you prefer to
snapshot live, the writers are already stopped, so bring them back afterwards
with `sudo ./snapshot-prep.sh --resume`.

Expect roughly 1.6 TB. DigitalOcean bills snapshots per GB stored, so decide
deliberately how many you keep.

**Refresh the snapshot** after anything you would not want to redo by hand:
adding or revoking a customer, rotating a key, changing `config.yaml`, or
upgrading the image. A stale snapshot restores a stale customer table.

---

## 3. Restoring

1. **Create a droplet from the snapshot** in the Control Panel. It must be the
   same MI355X spot type — the snapshot carries software, not GPUs, and
   `config.yaml` pins `tensor-parallel-size 8`.

2. **Point DNS at the new IP.** A reclaimed droplet always comes back on a
   different address. Update the `cflox.store` A record to the new one; nothing
   in any config pins an IP, so this is the only address-dependent step.

3. **Wait about five minutes.** The stack starts itself: weights load in ~137 s
   and the engine answers at ~260 s from boot.

4. **Verify.**

   ```bash
   cd /scratch/deploy && ./verify-auth.sh
   ```

You do not run `deploy.sh`, and you need neither the PAT nor the bundle
passphrase, because nothing is fetched or restored. Existing customer keys keep
working — `customers.tsv` came along in the snapshot.

### If something looks wrong

`sudo ./deploy.sh --watch` is safe to run on a restored droplet and re-asserts
every stage. It detects the weights and verifies the image by digest instead of
re-fetching them, so it skips both slow stages and only reconciles
configuration.

### The one thing that expires

The TLS certificate in the snapshot has a fixed expiry, so a snapshot older
than ~90 days restores with an expired certificate. Re-issue once DNS resolves
to the new box:

```bash
./issue-cert.sh --watch
```

`certbot` runs with `--keep-until-expiring`, so this is a no-op when the
certificate is still valid. Refreshing snapshots regularly means you never hit
this.

---

## 4. Rebuilding without a snapshot

If no usable snapshot exists, use the full pipeline — see
[`DEPLOY.md`](DEPLOY.md) and the quick start in the [README](../README.md):

```bash
mkdir -p /scratch
git clone https://Sudarshan50:<PAT>@github.com/Sudarshan50/cflow-gpu.git /scratch/deploy
cd /scratch/deploy
./secrets-backup.sh verify k3-secrets.enc
sudo ./deploy.sh --watch
```

This needs both the GitHub PAT and the `k3-secrets.enc` passphrase, which is
why both belong in a password manager rather than on the box. Once it is up and
verified, take a snapshot so the next rebuild is the five-minute path.
