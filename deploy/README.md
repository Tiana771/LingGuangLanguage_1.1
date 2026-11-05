# Server deployment

The application runs as `linguang.service` on `127.0.0.1:5001`. The stable
public HTTPS entry is `lgsy.huaxuefeng.com`, served by Caddy as a reverse proxy.
`Caddyfile` obtains and renews its HTTPS certificate automatically.
`linguang-tunnel.service` is an optional temporary fallback and is disabled
after the stable domain is confirmed.

Check services on the server:

```sh
sudo systemctl status linguang.service linguang-tunnel.service
```

Get the current public URL:

```sh
sudo journalctl -u linguang-tunnel.service --no-pager | grep -Eo 'https://[a-z0-9-]+\.trycloudflare\.com' | tail -n 1
```

If the domain is not available, the optional Quick Tunnel can be started with
`sudo systemctl enable --now linguang-tunnel.service`. Its URL is temporary and
can change after a tunnel restart or server reboot. The application's `.env`
belongs on the server with mode `600`; never commit it.

For the configured domain, create an Alibaba Cloud DNS `A` record for host
`lgsy` pointing to the server public IPv4 address, and allow inbound TCP ports
80 and 443 in the server security group. Caddy can then provision HTTPS.
