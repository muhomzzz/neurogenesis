# Neurogenesis — deploy to Render from an iPad

This version reads Render's `PORT` environment variable and binds to `0.0.0.0`.

## Deploy
1. Create a GitHub repository and upload the contents of this folder (not the ZIP file itself).
2. In Render, choose **New → Web Service** and connect that GitHub repository.
3. Use:
   - Runtime: Python
   - Build command: `pip install -r requirements.txt`
   - Start command: `python server.py`
4. Deploy. Open the generated `https://...onrender.com` address in Safari.

The included `render.yaml` records these settings for Render.

## Notes
- Render free web services can sleep after 15 minutes without incoming requests; the first request after sleeping can take time to wake the service.
- This prototype stores its save state in the app's local filesystem. On hosts with ephemeral filesystems, saved state may not survive a restart or redeploy. Add persistent storage before relying on long-term memory.
- The current prototype has no login/authentication. Anyone who can reach its public URL may be able to use its controls. Do not share the public URL broadly or store sensitive information.
- This is an experimental brain-inspired simulation, not a human brain emulation.
