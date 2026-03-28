# TLS / HTTPS Setup

Ragger's web UI transmits passwords and session tokens. **Use TLS for any
access beyond localhost.**

Three options below — pick whichever fits your setup.

---

## 1. Self-Signed Certificate

Generate your own certificate. No domain name or external service needed.

### Generate the certificate

```bash
# Create a cert valid for 10 years
sudo mkdir -p /etc/ragger/tls
sudo openssl req -x509 -newkey rsa:2048 -nodes \
    -keyout /etc/ragger/tls/key.pem \
    -out /etc/ragger/tls/cert.pem \
    -days 3650 \
    -subj "/CN=ragger" \
    -addext "subjectAltName=DNS:localhost,IP:192.168.0.166"
```

Replace `192.168.0.166` with your machine's LAN IP (add multiple IPs with
commas: `IP:192.168.0.166,IP:192.168.0.220`).

### Configure ragger

Add to `/etc/ragger.ini`:

```ini
[server]
tls_cert = /etc/ragger/tls/cert.pem
tls_key  = /etc/ragger/tls/key.pem
```

Restart the daemon. Access via `https://192.168.0.166:8432` (HTTPS works on
any port — 443 is just the convention).

### Trust the certificate

Browsers will warn about self-signed certs. To suppress:

**macOS:** Double-click `cert.pem` → opens Keychain Access → set to "Always Trust"

**iOS/iPadOS:** AirDrop or email `cert.pem` to the device → Settings → Profile
Downloaded → Install → Settings → General → About → Certificate Trust Settings
→ toggle on

**Linux:** Copy to `/usr/local/share/ca-certificates/ragger.crt` and run
`sudo update-ca-certificates`

**Windows:** Double-click → Install Certificate → Local Machine → Trusted Root
Certification Authorities

---

## 2. Let's Encrypt

Free, automatically renewed certificates from a trusted CA.

### Prerequisites

- A domain name pointing to your server's public IP
- Port 80 open temporarily (certbot uses it for ACME challenge verification)
- Ragger can serve HTTPS on any port (8432, 443, whatever you configure)

### Install certbot

```bash
# macOS (MacPorts)
sudo port install certbot

# macOS (Homebrew)
brew install certbot

# Debian/Ubuntu
sudo apt install certbot

# RHEL/Fedora
sudo dnf install certbot
```

### Get the certificate

```bash
# Standalone mode (temporarily binds port 80)
sudo certbot certonly --standalone -d chat.yourdomain.com
```

Certificates are saved to:
- `/etc/letsencrypt/live/chat.yourdomain.com/fullchain.pem`
- `/etc/letsencrypt/live/chat.yourdomain.com/privkey.pem`

### Configure ragger

```ini
[server]
tls_cert = /etc/letsencrypt/live/chat.yourdomain.com/fullchain.pem
tls_key  = /etc/letsencrypt/live/chat.yourdomain.com/privkey.pem
```

### Automatic renewal

Certbot installs a cron job or systemd timer that checks for renewal twice
daily. When a certificate is renewed, the new files are written to disk — but
ragger (and most servers) hold certs in memory. **You must configure a deploy
hook to restart ragger after renewal**, or it will continue serving the old
(eventually expired) certificate.

#### Step 1: Verify certbot's renewal timer is active

```bash
# Linux (systemd)
systemctl list-timers | grep certbot

# Or check cron
cat /etc/cron.d/certbot 2>/dev/null
crontab -l -u root 2>/dev/null | grep certbot
```

If nothing shows up, add a cron entry:

```bash
# Check twice daily (certbot only renews when needed)
echo "0 0,12 * * * root certbot renew -q" | sudo tee /etc/cron.d/certbot-renew
```

#### Step 2: Add the deploy hook

Edit the renewal config at
`/etc/letsencrypt/renewal/chat.yourdomain.com.conf` and add a `deploy_hook`
line under `[renewalparams]`:

**macOS (launchd):**

```ini
[renewalparams]
deploy_hook = launchctl kickstart -k system/com.ragger.daemon
```

**Linux (systemd):**

```ini
[renewalparams]
deploy_hook = systemctl restart ragger
```

The deploy hook runs only after a successful renewal, not on every check.

You can also set the hook globally (for all domains) by adding it to
`/etc/letsencrypt/cli.ini`:

```ini
deploy-hook = systemctl restart ragger
```

#### Step 3: Test it

```bash
# Dry run — checks renewal logic without actually renewing
sudo certbot renew --dry-run
```

If this succeeds, renewal will work automatically when the cert approaches
expiration (~30 days before).

### File permissions

Let's Encrypt stores private keys readable only by root. Since ragger's daemon
runs as root, this works out of the box. The certificate files are symlinks
into `/etc/letsencrypt/archive/` — don't move or copy them; let certbot manage
the symlinks so renewal stays automatic.

---

## 3. Reverse Proxy (Alternative)

If you already run a web server (nginx, Apache, Caddy), proxy to ragger
instead of adding TLS to ragger directly.

### Caddy (simplest)

```
chat.yourdomain.com {
    reverse_proxy localhost:8432
}
```

Caddy handles Let's Encrypt automatically — no certbot needed.

### nginx

```nginx
server {
    listen 443 ssl;
    server_name chat.yourdomain.com;

    ssl_certificate     /etc/letsencrypt/live/chat.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/chat.yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8432;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # SSE streaming support
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
    }
}
```

Don't forget to redirect HTTP → HTTPS:

```nginx
server {
    listen 80;
    server_name chat.yourdomain.com;
    return 301 https://$host$request_uri;
}
```

---

## Notes

- Plain HTTP on localhost is fine for local-only use — browsers won't leak
  tokens over loopback.
- Self-signed certs work anywhere but require manual trust setup on each
  client device.
- Let's Encrypt requires a domain name and port 80 access for verification.
- A reverse proxy keeps cert management separate from ragger — useful if
  you already run a web server.
