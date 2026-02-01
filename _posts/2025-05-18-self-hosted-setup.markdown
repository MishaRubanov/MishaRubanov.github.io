---
layout: post
title:  "My Self Hosted Setup"
date:   2025-05-18 12:00:45 -0600
page_excerpts: True
---

## Hetzner

I use Hetzner for hosting. Their CAX21 ARM server (4 vCPUs, 8GB RAM, 80GB NVMe) runs €6.49/month—far cheaper than AWS or DigitalOcean. [Server setup](https://docs.hetzner.com/cloud/servers/getting-started/creating-a-server/) and [Coolify installation](https://community.hetzner.com/tutorials/install-and-configure-coolify-on-linux) were straightforward.

The web interface handles provisioning, daily backups, and monitoring. Scaling is fast—I doubled my storage and RAM in minutes for an extra $3/month. DDoS protection and firewall management come included.

## Coolify

[Coolify](https://coolify.io/) is a self-hostable Heroku/Vercel alternative. I use it to deploy from GitHub, manage Docker containers, and spin up databases. It's a thin wrapper around Docker and Traefik with a UI—sometimes opaque, and wiring apps through Cloudflare has been painful—but it works.

### Networking

**[Traefik](https://coolify.io/docs/knowledge-base/proxy/traefik/overview)**: Coolify's default proxy. Handles SSL via Let's Encrypt and routes traffic.

**Cloudflare**: DNS, DDoS protection, caching. Cloudflare tunnels let me expose services without a public IP ([setup guide](https://rasmusgodske.com/posts/securely-expose-your-coolify-apps-with-the-magic-of-cloudflare-tunnels/)).

### Monitoring

**Uptime Kuma**: Lightweight uptime [monitoring](https://uptime.misharubanov.com/) with alerts.

**Glance**: Mobile-friendly [dashboard](dashboard.misharubanov.com) for RSS, weather, and container status.

**Duplicati**: Backups to cloud services, WEBDAV, and local storage.

**Dozzle**: Container logs and resource usage.

**ntfy**: Push notifications when services go down.

**Beszel**: Server monitoring with alerts for CPU/memory spikes. Combined with Dozzle, debugging is quick.

### The stack

Code in GitHub → Coolify pulls and deploys containers → Traefik routes traffic with auto-SSL → Uptime Kuma and Glance monitor everything.

## Apps

**Audiobookshelf**: My ebook/audiobook [library](http://bookshelf.misharubanov.com/). I set this up after learning Amazon was [removing Kindle download functionality](https://www.theverge.com/news/612898/amazon-removing-kindle-book-download-transfer-usb). Good web reader and Android app.

**Immich**: Self-hosted [Google Photos alternative](https://immich.app/) with facial recognition. Still in progress—TB+ of photos means I need a NAS before this makes financial sense.

**Gramps Web**: A [family tree app](https://family.misharubanov.com/) I set up after visiting ancestry in Uzbekistan. Multi-user with editor/guest roles, hot and cold backups.

**Vikunja**: Simple [todo app](https://todo.misharubanov.com/). Not as polished as commercial options, but I own my data.

