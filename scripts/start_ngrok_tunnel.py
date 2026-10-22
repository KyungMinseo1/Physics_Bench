"""Expose one or more local ports (e.g. peda_survey on 8080 and peda_survey2 on 8004) via
ngrok tunnels, so students can reach them with public links - e.g. from a Colab notebook
where the ports aren't otherwise reachable from outside the VM, or from any machine
without a public IP.

Pass multiple ports to open multiple tunnels from a SINGLE ngrok agent session - this
matters because ngrok's free plan only allows one simultaneous agent session per account
(ERR_NGROK_108 if you try to start a second `ngrok`/pyngrok process concurrently), but a
single agent can hold multiple tunnels at once, so running two apps through one call to
this script (instead of two separate script invocations) avoids that limit.

Usage:
    pip install pyngrok
    NGROK_AUTHTOKEN=xxxx python scripts/start_ngrok_tunnel.py 8080 8004

Get an authtoken (free tier is enough) from https://dashboard.ngrok.com/get-started/your-authtoken
"""

import os
import sys
import time

from pyngrok import ngrok

ports = [int(p) for p in sys.argv[1:]] or [8000]

authtoken = os.environ.get("NGROK_AUTHTOKEN")
if not authtoken:
    raise SystemExit("Set NGROK_AUTHTOKEN env var first (see docstring for where to get one).")

ngrok.set_auth_token(authtoken)
tunnels = [ngrok.connect(port, "http") for port in ports]

print("Public URLs:")
for port, tunnel in zip(ports, tunnels):
    print(f"  localhost:{port} -> {tunnel.public_url}")
print("Keep this process running - closing it tears down all of the tunnels above.")

try:
    while True:
        time.sleep(3600)
except KeyboardInterrupt:
    for tunnel in tunnels:
        ngrok.disconnect(tunnel.public_url)
