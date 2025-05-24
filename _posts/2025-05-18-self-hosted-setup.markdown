---
layout: post
title:  "My Self Hosted Setup"
date:   2025-05-18 12:00:45 -0600
page_excerpts: True
---

## Server infrastructure: Hetzner
I chose Hetzner as my hosting provider for several compelling reasons:

### 💰 Cost-Effective
Hetzner offers some of the most competitive prices in the industry. Their dedicated servers and cloud instances are significantly cheaper than comparable offerings from AWS, DigitalOcean, or Linode. For example, their CAX21 ARM cloud server (Ampere Altra, 4 vCPUs, 8GB RAM, 80GB NVMe) costs just €6.49/month (at the time of writing this). Simple tutorials for setting up servers based on standard images (i.e., ubuntu) or snapshots can be found [here](https://docs.hetzner.com/cloud/servers/getting-started/creating-a-server/). They also made it really easy to set up [coolify](https://community.hetzner.com/tutorials/install-and-configure-coolify-on-linux).


### 🔧 Management Tools
Their web interface and API make server management straightforward:
- Easy server provisioning
- IP management
- Backup solutions
  - I have automatic backups daily that can easily be restored
- Monitoring tools

### 📈 Scalability
While I started with a single dedicated server, Hetzner makes it easy to scale up or down as needed (for as little as an hour at a time). Their cloud instances can be spun up in minutes, perfect for testing or handling traffic spikes. I quickly reached the 40GB limit, but within a few minutes I had doubled the available memory + ram for 3$/month more.

### 🔒 Security
Hetzner takes security seriously:
- DDoS protection included
- Regular security updates
- Firewall management
- IP-based access controls

For my self-hosting needs, Hetzner provides the perfect balance of performance, reliability, and cost-effectiveness, and ease of use. Now that I have a VPS, I was able to begin building the variety of applications that lived on my *.misharubanov.com subdomains.


## Coolify & Essential Tools

One of the standout tools in my stack is **Coolify**—an open-source, self-hostable alternative to platforms like Heroku and Vercel. 

### 🚀 Coolify: The Heart of My Deployment

[Coolify](https://coolify.io/) is a modern platform that lets you deploy applications, databases, and static sites with ease. It supports Docker, integrates with Git, and has a  web UI for managing deployments. I use Coolify to:

- Deploy Python, and static sites directly from GitHub (public and private repos work)
- Manage Docker containers and images.
- Spin up databases like PostgreSQL and MySQL with a few clicks.
- Monitor resource usage and logs in real time.
- Find inspiration for cool open-source docker-based projects that people have made into one-click deployments

It's far from a perfect tool - but honestly it's just a great way to learn how to manage docker containers all living on the same server, each pointing to a different DNS. There are still mysteries to me (and to the self-hosting [reddit community](https://www.reddit.com/r/coolify/comments/1ivslne/nothing_works_on_coolify/)) on exactly what Docker is doing, since it's just a thin docker + traefik wrapper with a UI. But for now it seems to work enough - although dealing with how to wire certain apps through my firewall has been quite difficult.

### 🛠️ Other Tools in My Stack

#### 1. **[Traefik](https://coolify.io/docs/knowledge-base/proxy/traefik/overview)**
Coolify's default proxy. Traefik automatically handles SSL certificates (via Let's Encrypt) and routes traffic to my services.

#### 2. **Cloudflare**
For DNS management, DDoS protection, and caching. Cloudflare sits in front of my self-hosted services for extra security and performance. Cloudflare tunnels also enable secure access to all apps without needing to expose a public IP to the internet - I followed this [article](https://rasmusgodske.com/posts/securely-expose-your-coolify-apps-with-the-magic-of-cloudflare-tunnels/) which was very helpful. 

#### 3. **Uptime Kuma**
A beautiful, self-hosted monitoring tool. I use it to keep tabs on all my services and get alerts if anything goes down.

#### 4. **Glance**
A nice easily customizable [dashboard](dashboard.misharubanov.com) that let's me browse RSS feeds, weather, docker containers, websites, and a bunch of other stuff. It's focused on being very mobile-compatible, which is great for checking on all my self-hosted apps and server when travelling.

### 🏗️ How It All Fits Together

- **Code** lives in GitHub.
- **Deployments** are handled by Coolify, which pulls from my repos and spins up containers.
- **Traffic** is routed by Traefik, with SSL managed automatically.
- **Monitoring** is done via Uptime Kuma for in-depth site monitoring and Glance for previews and server stats, with notifications sent to my phone.

## Content and other apps

### 1. **Audiobookshelf**
My audiobook and ebook content management [platform](http://bookshelf.misharubanov.com/). This was actually the onus for going down this self-hosted rabbit hole - when I discovered that the library of kindle books I purchased was [no longer able to be downloaded](https://www.theverge.com/news/612898/amazon-removing-kindle-book-download-transfer-usb), I decided that it was time to set up my own library before that updated policy was implemented. Audiobookshelf is a great dockerized app that made it easy to self-host a browse-able library, manage library users, and listen/read in-browser.

### 2. **Immich**
A privacy-friendly self-hosted [alternative](https://immich.app/) to Google photos, with a lot of the same features (CV-based facial recognition, etc.). This is something I'm still developing - having TB+ levels of photos means this only makes sense if I connect a NAS to my remote server (buying TBs of storage doesn't really make financial sense). 

### 3. **WebTrees**
After a trip to my see my ancestry in Uzbekistan, I wanted to self-host my family tree and add as much information as we know about our ancestors 4+ generations ago. While the UI is not the most intuitive, this  allows both for storing and backing up all data in WEBCOM format

### 4. **Vikunja**
A nice, lightweight todo [app](https://todo.misharubanov.com/) that let's me keep track of all necessary tasks on my own. This is just nice to give me total control over - it's similar to the variety of budgeting/finance apps - they aren't as great as the latest and greatest company apps, but the tradeoff seems well worth it to me.

### 5. **CodeServer**
To be honest, this is a service I won't be using really often. This allows vscode to run on my server natively - but it's really only useful if I want to give my dev environment to someone else. A lot of the self-hosted community feels the same way - [it's just easier to use an SSH key instead of an in-browser server.](https://www.reddit.com/r/selfhosted/comments/15lqvyz/anyone_else_hosting_codeserver/) Personally, I prefer to use cursor and ssh directly from my local IDE.

### 💡 Why Self-Host?

Self-hosting gives me full control, privacy, and the freedom to experiment. It's a learning journey, and with tools like Coolify, it's never been easier to get started.

