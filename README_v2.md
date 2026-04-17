# AlpenMesh Worker - Docker Run Guide (v2)

This guide contains the exact commands to build, run, verify GPU support, and push the `alpenmesh-worker` image.

## 1) Verify image exists

```powershell
docker images
```

Expected repo/tag:

- `alpenmesh-worker:latest`

## 2) Verify Docker can access GPU

Run this first. If this fails, fix Docker/NVIDIA setup before running the worker.

```powershell
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## 3) Final run command (GPU-enabled)

Use this exact command from project root (`C:\Users\wiki8\Desktop\AlpenMesh\alpenmesh-edge-worker`):

```powershell
docker run --rm --gpus all --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -p 8081:8081 -v "${PWD}/worker_data:/app/data" --add-host host.docker.internal:host-gateway --entrypoint sh alpenmesh-worker:latest -c "ln -sf /usr/lib/wsl/drivers/*/libnvidia-ml.so.1 /usr/lib/libnvidia-ml.so && exec /app/alpenmesh-worker"
```

Why this works:

- Your container has `libnvidia-ml.so.1` but app looks for `libnvidia-ml.so`.
- The startup command creates the missing symlink, then starts `/app/alpenmesh-worker`.

## 4) Scheduler URL (optional override)

The worker reads **`SCHEDULER_URL`** from the environment (see `src/state.rs`). Use the **base URL only**—no trailing slash (e.g. `https://example.ngrok-free.app`, not `https://.../`).

**Local default:** if unset, the worker uses `http://127.0.0.1:3000`, which does not work inside Docker unless you point it at a reachable scheduler (host port mapping, `host.docker.internal`, or a public URL like ngrok).

**Example:** ngrok tunnel to your scheduler:

```powershell
docker run --rm --gpus all --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e SCHEDULER_URL="https://fac0-2400-adcc-2115-d900-cd7b-3871-342a-1a60.ngrok-free.app" -p 8081:8081 -v "${PWD}/worker_data:/app/data" --add-host host.docker.internal:host-gateway --entrypoint sh alpenmesh-worker:latest -c "ln -sf /usr/lib/wsl/drivers/*/libnvidia-ml.so.1 /usr/lib/libnvidia-ml.so && exec /app/alpenmesh-worker"
```

Replace the ngrok URL with your own tunnel URL when it changes. You can also set `SCHEDULER_URL` in a `.env` file if you run the binary outside Docker (see `.env.worker-a` for the variable name).

### Remote scheduler must call this worker back

Heartbeats only need the scheduler URL. **Stream assignment** needs the scheduler to HTTP(S) POST to **this** machine. `host.docker.internal` and `127.0.0.1` are wrong for a scheduler on another PC—they resolve on the scheduler’s host, not yours.

1. **On this worker machine**, start a **second** ngrok tunnel to the worker port (after Docker is publishing `8081`):

   ```powershell
   ngrok http 8081
   ```

   Note the HTTPS forwarding URL, e.g. `https://abcd-....ngrok-free.app` (ngrok terminates TLS on port **443**).

2. Set **`WORKER_EXTERNAL_ADDR`** to `hostname:port` the scheduler will use in its callback URL. If your scheduler builds `http://{WORKER_EXTERNAL_ADDR}/...`, you may need an HTTP tunnel or scheduler support for HTTPS—match what your scheduler expects.

   Example (HTTPS ngrok; port 443):

   ```powershell
   docker run --rm --gpus all --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e SCHEDULER_URL="https://fac0-2400-adcc-2115-d900-cd7b-3871-342a-1a60.ngrok-free.app" -e WORKER_EXTERNAL_ADDR="YOUR_WORKER_NGROK_HOST.ngrok-free.app:443" -p 8081:8081 -v "${PWD}/worker_data:/app/data" --add-host host.docker.internal:host-gateway --entrypoint sh alpenmesh-worker:latest -c "ln -sf /usr/lib/wsl/drivers/*/libnvidia-ml.so.1 /usr/lib/libnvidia-ml.so && exec /app/alpenmesh-worker"
   ```

   Replace `YOUR_WORKER_NGROK_HOST.ngrok-free.app` with the host from **your** worker tunnel (not the scheduler URL).

3. **Apply `app.yaml` changes** either by rebuilding (`docker build -t alpenmesh-worker:latest .`) **or** by mounting the file on `docker run` (no rebuild), e.g. add `-v "${PWD}/app.yaml:/app/app.yaml:ro"` to the command.

`app.yaml` uses `host: "0.0.0.0"` so the server accepts traffic from Docker and from tunnels, not only loopback.

### Tailscale (scheduler on another node in the same tailnet)

Use the scheduler’s Tailscale address (no trailing slash on `SCHEDULER_URL`):

- `http://100.78.61.103:3000` or  
- `http://desktop-qbaujvo.tarpan-herring.ts.net:3000`

Set **`WORKER_EXTERNAL_ADDR`** to **this machine’s** Tailscale IPv4 and port `8081` (what the scheduler will call back). On Windows:

```powershell
Get-NetIPAddress -InterfaceAlias "*Tailscale*" -AddressFamily IPv4 | Select-Object -ExpandProperty IPAddress
```

Example `docker run` (replace `100.104.x.x` with your worker IP from the command above):

```powershell
docker run --rm --name alpenmesh-worker --gpus all --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e SCHEDULER_URL="http://100.78.61.103:3000" -e WORKER_EXTERNAL_ADDR="100.104.x.x:8081" -p 8081:8081 -v "${PWD}/worker_data:/app/data" -v "${PWD}/app.yaml:/app/app.yaml:ro" --add-host host.docker.internal:host-gateway --entrypoint sh alpenmesh-worker:latest -c "ln -sf /usr/lib/wsl/drivers/*/libnvidia-ml.so.1 /usr/lib/libnvidia-ml.so && exec /app/alpenmesh-worker"
```

**Docker Desktop caveat:** Linux containers often **cannot** open `100.x` Tailscale addresses (the VM does not use your Windows Tailscale routes). If heartbeats time out inside the container but `Invoke-WebRequest http://100.78.61.103:3000/` works in PowerShell on the host, either:

- Expose the scheduler on **`http://host.docker.internal:3000`** using a **host-side TCP forward** to `100.78.61.103:3000` (e.g. Windows `netsh interface portproxy` + firewall rule, run as Administrator), then set `SCHEDULER_URL=http://host.docker.internal:3000`, or  
- Use a **public** scheduler URL (ngrok, etc.), or  
- Run the worker **without** Docker for this test.

Ensure the **scheduler machine** allows inbound TCP `3000` from your tailnet and this **worker** allows inbound TCP `8081` from the scheduler (Windows Firewall).

## 5) What success looks like

You should see logs similar to:

- `Found: NVIDIA GeForce ...`
- `POOL DEPLOYED: ... Parallel GPU Lanes Active`
- `Oxide server started ... address=0.0.0.0:8081`

## 6) Check container status

```powershell
docker ps --filter ancestor=alpenmesh-worker:latest
```

If running correctly, port mapping should include:

- `0.0.0.0:8081->8081/tcp`

## 7) Push image to Docker Hub

Replace `YOUR_DOCKERHUB_USERNAME` with your actual username.

```powershell
docker login
docker tag alpenmesh-worker:latest YOUR_DOCKERHUB_USERNAME/alpenmesh-worker:latest
docker push YOUR_DOCKERHUB_USERNAME/alpenmesh-worker:latest
```

## 8) Common issues

### `the input device is not a TTY`

Remove `-it` when running from non-interactive environments.

### `Failed to initialize NVML: libnvidia-ml.so not found`

Use the run command in section 3 (it creates the required symlink at container startup).

### Heartbeat errors (scheduler unreachable)

GPU is still working; heartbeats fail when the worker cannot reach the scheduler. Set **`SCHEDULER_URL`** (see section 4) to your real scheduler URL, or run the scheduler locally and expose it (`host.docker.internal` on Windows Docker Desktop, port forwarding, or ngrok).

### Scheduler logs: `Error connecting to worker ... host.docker.internal:8081`

The scheduler is calling an address that only exists on your Docker host. Use a **worker-side ngrok tunnel** and set **`WORKER_EXTERNAL_ADDR`** to that public host and port (see section 4).

