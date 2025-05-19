---
layout: post
title:  "My Self Hosted Setup"
date:   2025-05-18 12:00:45 -0600
page_excerpts: True
---

## Server infrastructure: Hetzner
I chose Hetzner as my hosting provider for several compelling reasons:

### 💰 Cost-Effective
Hetzner offers some of the most competitive prices in the industry. Their dedicated servers and cloud instances are significantly cheaper than comparable offerings from AWS, DigitalOcean, or Linode. For example, their CAX21 ARM cloud server (Ampere Altra, 4 vCPUs, 8GB RAM, 80GB NVMe) costs just €4.59/month—an incredible value for running lightweight workloads or self-hosted services.

### 🌍 European Infrastructure
With data centers in Germany and Finland, Hetzner provides excellent connectivity within Europe. This is particularly important for GDPR compliance and data sovereignty. Their network is robust and well-maintained, with multiple redundant connections.

### 🛠️ Hardware Quality
Hetzner uses enterprise-grade hardware in their dedicated servers. You get:
- Enterprise SSDs and NVMe drives
- ECC RAM for better reliability
- High-quality network equipment
- Regular hardware upgrades

### 🔧 Management Tools
Their web interface and API make server management straightforward:
- Easy server provisioning
- IP management
- Backup solutions
- Monitoring tools
- Rescue system for troubleshooting

### 📈 Scalability
While I started with a single dedicated server, Hetzner's cloud offerings make it easy to scale up or down as needed. Their cloud instances can be spun up in minutes, perfect for testing or handling traffic spikes. I quickly reached the 40GB limit, but within a few minutes I had doubled the available RAM.

### 🔒 Security
Hetzner takes security seriously:
- DDoS protection included
- Regular security updates
- Firewall management
- IP-based access controls

For my self-hosting needs, Hetzner provides the perfect balance of performance, reliability, and cost-effectiveness. While they might not have the extensive feature set of AWS or Google Cloud, their straightforward approach and competitive pricing make them an excellent choice for personal projects and small to medium-sized applications.


## Coolify & Essential Tools

One of the standout tools in my stack is **Coolify**—an open-source, self-hostable alternative to platforms like Heroku and Vercel. 

### 🚀 Coolify: The Heart of My Deployment

[Coolify](https://coolify.io/) is a modern platform that lets you deploy applications, databases, and static sites with ease. It supports Docker, integrates with Git, and has a  web UI for managing deployments. I use Coolify to:

- Deploy Node.js, Python, and static sites directly from GitHub.
- Manage Docker containers and images.
- Spin up databases like PostgreSQL and MySQL with a few clicks.
- Monitor resource usage and logs in real time.

### 🛠️ Other Tools in My Stack

#### 1. **[Traefik](https://coolify.io/docs/knowledge-base/proxy/traefik/overview)**
Coolify's default proxy. Traefik automatically handles SSL certificates (via Let's Encrypt) and routes traffic to my services.

#### 3. **Cloudflare**
For DNS management, DDoS protection, and caching. Cloudflare sits in front of my self-hosted services for extra security and performance. Cloudflare tunnels also enable secure access to all apps without needing to expose a public IP to the internet - I followed this [article](https://rasmusgodske.com/posts/securely-expose-your-coolify-apps-with-the-magic-of-cloudflare-tunnels/) which was very helpful.

#### 4. **Uptime Kuma**
A beautiful, self-hosted monitoring tool. I use it to keep tabs on all my services and get alerts if anything goes down.

#### 5. **Gitea**
A lightweight, self-hosted Git service. Perfect for private repositories and CI/CD pipelines.

#### 6. **Vaultwarden**
My go-to for password management. It's a self-hosted Bitwarden-compatible server.

#### 7. **Plausible Analytics**
A privacy-friendly, self-hosted alternative to Google Analytics. I use it to track visitors on my sites without compromising privacy.

### 🏗️ How It All Fits Together

- **Code** lives in Gitea or GitHub.
- **Deployments** are handled by Coolify, which pulls from my repos and spins up containers.
- **Traffic** is routed by Traefik, with SSL managed automatically.
- **Monitoring** is done via Uptime Kuma, with notifications sent to my phone.
- **Analytics** and **passwords** are managed by Plausible and Vaultwarden, respectively.

### 💡 Why Self-Host?

Self-hosting gives me full control, privacy, and the freedom to experiment. It's a learning journey, and with tools like Coolify, it's never been easier to get started.

