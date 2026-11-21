# Neurogenesis — iPad/Safari deployment

This project needs a Python server. Safari is the interface; it does not run `server.py` itself.

## Run locally on a computer
1. Install Python 3.10+.
2. In this folder, run: `python -m pip install -r requirements.txt`
3. Run: `python server.py`
4. Open `http://localhost:8000` on that same computer.

## Open on iPad
Deploy this folder to a Python-capable host (for example, a Replit Python project), start it with `python server.py`, and open the host's HTTPS web URL in Safari.

For a host that requires a port setting, configure port 8000. The server listens on all interfaces.

## Important
This is an early experimental prototype, not a human brain emulation. It uses a 320-neuron Izhikevich network and toy approach/avoid learning task. The server has no login/authentication, so do not publish it as a public service or expose it to untrusted users without adding authentication and other security protections.
