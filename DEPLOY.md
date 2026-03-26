# Production Deployment (Hetzner + OpenWebUI)

This runbook deploys OpenWebUI + API from this repository with HTTPS and private container ports.

## 1) Server setup (one-time)

Use Ubuntu 24.04 LTS (recommended minimum: 2 vCPU / 2 GB RAM).
On Hetzner Cloud, a practical starting point is `CPX21` (or larger if traffic grows).

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y ca-certificates curl gnupg git nginx certbot python3-certbot-nginx ufw

# Docker Engine + Compose plugin (Docker official repo)
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

sudo systemctl enable docker
sudo systemctl start docker

# Basic firewall
sudo ufw allow OpenSSH
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw --force enable
```

## 2) Clone and configure app

```bash
cd /opt
sudo git clone https://github.com/sn1887/kankor-rag-space.git
cd kankor-rag-space

sudo cp docker/.env.example docker/.env.prod
sudo chmod 600 docker/.env.prod
sudo nano docker/.env.prod
```

Set your real production values in `docker/.env.prod` (model keys, index/docstore paths, API key).

## 3) Transfer index artifacts

Run from your local machine (or the machine that already has the index):

```bash
rsync -avz ./data/index/ root@<HETZNER_IP>:/opt/kankor-rag-space/data/index/
```

## 4) Start services

```bash
cd /opt/kankor-rag-space
sudo docker compose -f docker/docker-compose.openwebui.yml --env-file docker/.env.prod up -d --build api openwebui
sudo docker compose -f docker/docker-compose.openwebui.yml --env-file docker/.env.prod ps
curl http://127.0.0.1:8000/v1/health
```

## 5) Nginx reverse proxy + TLS

Create site config:

```bash
sudo nano /etc/nginx/sites-available/kankor
```

```nginx
server {
    listen 80;
    server_name <YOUR_DOMAIN>;

    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

Enable and reload:

```bash
sudo ln -sf /etc/nginx/sites-available/kankor /etc/nginx/sites-enabled/kankor
sudo nginx -t
sudo systemctl reload nginx
```

Issue certificate:

```bash
sudo certbot --nginx -d <YOUR_DOMAIN>
sudo certbot renew --dry-run
systemctl status certbot.timer --no-pager
```

## 6) Monitoring and operations

- Uptime check: add `https://<YOUR_DOMAIN>/` in UptimeRobot (or equivalent).
- API health quick check:

```bash
curl -f http://127.0.0.1:8000/v1/health
```

- Logs:

```bash
sudo docker compose -f docker/docker-compose.openwebui.yml --env-file docker/.env.prod logs -f api openwebui
```

## 7) Reboot validation

```bash
sudo reboot
# reconnect, then:
cd /opt/kankor-rag-space
sudo docker compose -f docker/docker-compose.openwebui.yml --env-file docker/.env.prod ps
curl -f http://127.0.0.1:8000/v1/health
```

## Deployment Checklist

- [ ] `docker/.env.prod` secrets are updated
- [ ] `docker/.env.example` still reflects required variables
- [ ] Index data copied to server (`data/index/...`)
- [ ] Compose deploy command completed successfully
- [ ] `/v1/health` returns OK
- [ ] HTTPS works in browser
- [ ] `certbot.timer` is active
- [ ] Uptime monitor is active



