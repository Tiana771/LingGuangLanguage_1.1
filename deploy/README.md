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
can change after a tunnel restart or server reboot.

Production credentials belong in `/etc/linguang/linguang.env`, owned by root
with mode `600`, outside the application directory. Systemd loads this file
for the application; an `export` in an SSH terminal or a Windows environment
variable does not update the running service. Never commit credentials or put
a production `.env` in the application directory. Restart `linguang.service`
after changing the private environment file.

Only explicitly listed browser files are served from the project root.
When adding a new browser module or asset, update `PUBLIC_ROOT_FILES` or
`PUBLIC_ASSET_FILES` in `serve_sign_api.py`. MediaPipe uses the separate
`/vendor/mediapipe/` route. Caddy also blocks hidden paths and server files.

Access logs: `/var/log/caddy/linguang-access.json` (JSON, automatic rotation).
Gunicorn access/error logs: `sudo journalctl -u linguang.service`.
Model calls: `/var/log/linguang/model-usage.jsonl`, owned by ubuntu with mode
`600`, rotated by `/etc/logrotate.d/linguang-model`. Each request has one
`started` record and one terminal record, correlated with the `X-Request-ID`
response header. Terminal records include the account, client IP, requested
and returned model, provider identifiers when available, actual usage and
finish reason. Timestamps use ISO 8601 UTC; add eight hours for Beijing time.
Interrupted streams without provider usage are marked `unknown`, not zero.
Logs do not contain API keys, passwords or transcript contents.

`LINGUANG_TRUST_PROXY=1` trusts one local proxy hop. Keep Gunicorn bound to
`127.0.0.1`, with Caddy as its only public entry.

For the configured domain, create an Alibaba Cloud DNS `A` record for host
`lgsy` pointing to the server public IPv4 address, and allow inbound TCP ports
80 and 443 in the server security group. Caddy can then provision HTTPS.
