#!/usr/bin/env python3
"""Renting a GPU on Vast.ai or RunPod for lab.py.

The recipe's own digest-pinned image is the container image and its arguments
are the container command, so the launch contract under test is the one the
registry publishes. Rented containers have no docker daemon of their own, so
nothing here is docker-in-docker.

Standard library only. Credentials: RunPod from ~/.runpod/config.toml or
RUNPOD_API_KEY; Vast from the `vastai` CLI (~/.config/vastai/vast_api_key).

Provider notes:
  runpod  cannot pull from ghcr.io on community hosts (containers never start);
          fine for Docker Hub images (vllm, sglang). --cloud picks the cloud.
  vast    pulls any public registry through the host's docker; ports are
          mapped to a public ip:port; offers are chosen cheapest-first among
          verified hosts with fast downlink.
"""

import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

# registry hardware id -> provider GPU names. vast names use underscores for spaces in queries.
GPUS = {
    "gtx-1080-ti-11gb": {"vast": "GTX 1080 Ti"},
    "rtx-3060-12gb": {"runpod": ["NVIDIA GeForce RTX 3060"], "vast": "RTX 3060"},
    "rtx-3060-ti-8gb": {"runpod": ["NVIDIA GeForce RTX 3060 Ti"], "vast": "RTX 3060 Ti"},
    "rtx-4060-8gb": {"runpod": ["NVIDIA GeForce RTX 4060"], "vast": "RTX 4060"},
    "rtx-4060-ti-8gb": {"runpod": ["NVIDIA GeForce RTX 4060 Ti"], "vast": "RTX 4060 Ti"},
    "rtx-4060-ti-16gb": {"runpod": ["NVIDIA GeForce RTX 4060 Ti"], "vast": "RTX 4060 Ti"},
    "rtx-5060-8gb": {"runpod": ["NVIDIA GeForce RTX 5060"], "vast": "RTX 5060"},
    "rtx-5060-ti-8gb": {"runpod": ["NVIDIA GeForce RTX 5060 Ti"], "vast": "RTX 5060 Ti"},
    "rtx-5060-ti-16gb": {"runpod": ["NVIDIA GeForce RTX 5060 Ti"], "vast": "RTX 5060 Ti"},
    "rtx-3070-8gb": {"runpod": ["NVIDIA GeForce RTX 3070"], "vast": "RTX 3070"},
    "rtx-3070-ti-8gb": {"runpod": ["NVIDIA GeForce RTX 3070 Ti"], "vast": "RTX 3070 Ti"},
    "rtx-3080-10gb": {"runpod": ["NVIDIA GeForce RTX 3080"], "vast": "RTX 3080"},
    "rtx-3080-12gb": {"runpod": ["NVIDIA GeForce RTX 3080"], "vast": "RTX 3080"},
    "rtx-4070-super-12gb": {"runpod": ["NVIDIA GeForce RTX 4070 SUPER"], "vast": "RTX 4070S"},
    "rtx-3080-ti-12gb": {"runpod": ["NVIDIA GeForce RTX 3080 Ti"], "vast": "RTX 3080 Ti"},
    "rtx-3090-24gb": {"runpod": ["NVIDIA GeForce RTX 3090"], "vast": "RTX 3090"},
    "rtx-3090-ti-24gb": {"runpod": ["NVIDIA GeForce RTX 3090 Ti"], "vast": "RTX 3090 Ti"},
    "rtx-4070-12gb": {"runpod": ["NVIDIA GeForce RTX 4070"], "vast": "RTX 4070"},
    "rtx-4070-ti-12gb": {"runpod": ["NVIDIA GeForce RTX 4070 Ti"], "vast": "RTX 4070 Ti"},
    "rtx-4070-ti-super-16gb": {"runpod": ["NVIDIA GeForce RTX 4070 Ti SUPER"], "vast": "RTX 4070S Ti"},
    "rtx-4080-16gb": {"runpod": ["NVIDIA GeForce RTX 4080"], "vast": "RTX 4080"},
    "rtx-4080-super-16gb": {"runpod": ["NVIDIA GeForce RTX 4080 SUPER"], "vast": "RTX 4080S"},
    "rtx-4090-24gb": {"runpod": ["NVIDIA GeForce RTX 4090"], "vast": "RTX 4090"},
    "rtx-5070-12gb": {"runpod": ["NVIDIA GeForce RTX 5070"], "vast": "RTX 5070"},
    "rtx-5070-ti-16gb": {"runpod": ["NVIDIA GeForce RTX 5070 Ti"], "vast": "RTX 5070 Ti"},
    "rtx-5080-16gb": {"runpod": ["NVIDIA GeForce RTX 5080"], "vast": "RTX 5080"},
    "rtx-5090-32gb": {"runpod": ["NVIDIA GeForce RTX 5090"], "vast": "RTX 5090"},
    "rtx-2000-ada-16gb": {"runpod": ["NVIDIA RTX 2000 Ada Generation"], "vast": "RTX 2000Ada"},
    "rtx-4000-ada-20gb": {"runpod": ["NVIDIA RTX 4000 Ada Generation"], "vast": "RTX 4000Ada"},
    "rtx-6000-ada-48gb": {"runpod": ["NVIDIA RTX 6000 Ada Generation"], "vast": "RTX 6000Ada"},
    "rtx-a6000-48gb": {"runpod": ["NVIDIA RTX A6000"], "vast": "RTX A6000"},
    "rtx-pro-4000-blackwell-24gb": {"runpod": ["NVIDIA RTX PRO 4000 Blackwell"], "vast": "RTX PRO 4000"},
    "rtx-pro-4500-blackwell-32gb": {"runpod": ["NVIDIA RTX PRO 4500 Blackwell"], "vast": "RTX PRO 4500"},
    "rtx-pro-6000-blackwell-96gb": {"runpod": ["NVIDIA RTX PRO 6000 Blackwell Workstation Edition", "NVIDIA RTX PRO 6000 Blackwell Server Edition"], "vast": "RTX PRO 6000 WS"},
}


def log(msg):
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", file=sys.stderr, flush=True)


def http_probe(url, timeout=10):
    """-> 'ok' | 'refused' (host reachable, server not listening yet) | 'timeout' (port unreachable) | 'http-<code>'."""
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "local-ai-lab/1"}), timeout=timeout) as response:
            return "ok" if response.status == 200 else f"http-{response.status}"
    except urllib.error.HTTPError as error:
        return f"http-{error.code}"
    except urllib.error.URLError as error:
        reason = str(error.reason).lower()
        return "timeout" if "timed out" in reason else "refused"
    except (TimeoutError, OSError):
        return "timeout"


class StartStalled(Exception):
    """The host accepted the job but the container never started (image pull stalled or failed)."""


class Spec:
    """Base container launch specification; lab.py subclasses this with the rendered launch contract."""

    def onstart_script(self):
        """One /bin/sh line: provision, then exec the contract's entrypoint and arguments."""
        import base64
        import shlex
        python = self.entrypoint if self.entrypoint and re.search(r"python3?$", self.entrypoint) else "python3"
        parts = []
        for kind, target, payload in self.provision:
            if kind == "asset":
                encoded = base64.b64encode(payload.encode()).decode()
                parts.append(f"mkdir -p {shlex.quote(os.path.dirname(target))} && printf %s {encoded} | base64 -d > {shlex.quote(target)}")
            else:
                repo, revision, files = (list(payload) + [None])[:3]  # a 2-tuple from an older caller means the whole revision
                patterns = [p.strip() for p in files.split(",") if p.strip()] if isinstance(files, str) else list(files or [])
                allow = f", allow_patterns={patterns!r}" if patterns else ""
                code = f"from huggingface_hub import snapshot_download as d; d({repo!r}, revision={revision!r}, local_dir={target!r}{allow})"
                # some engine images (the official llama.cpp CUDA builds) ship python3 with no pip, so
                # `pip install huggingface_hub` cannot run; the download then falls back to curl (present in the
                # image), which retries a dropped transfer and resumes it instead of starting over
                fallback = ("import fnmatch,json,os,subprocess,urllib.request\n"
                            f"repo,rev,target={repo!r},{revision!r},{target!r}\n"
                            f"patterns={patterns!r}\n"
                            "os.makedirs(target,exist_ok=True)\n"
                            "tree=json.load(urllib.request.urlopen('https://huggingface.co/api/models/%s/tree/%s'%(repo,rev)))\n"
                            "files=[s for s in tree if s.get('type')=='file' and (not patterns or any(fnmatch.fnmatch(s['path'],p) for p in patterns))]\n"
                            "if patterns and not files: raise SystemExit('nothing in the revision matches %r' % (patterns,))\n"
                            "for s in files:\n"
                            " path=s['path'];dest=os.path.join(target,path)\n"
                            " if os.path.exists(dest) and os.path.getsize(dest)==s.get('size'): continue\n"
                            " print('downloading',path,s.get('size'),'bytes',flush=True)\n"
                            " subprocess.check_call(['curl','-L','--fail','--retry','8','--retry-all-errors','--retry-delay','5','-C','-','--create-dirs','-o',dest,'https://huggingface.co/%s/resolve/%s/%s'%(repo,rev,path)])\n"
                            " if s.get('size') is not None and os.path.getsize(dest)!=s['size']:\n"
                            "  raise SystemExit('short download: '+path)\n")
                parts.append(f"mkdir -p {shlex.quote(target)}")
                hub = (f"({shlex.join([python, '-c', 'import huggingface_hub'])} || "
                       f"{shlex.join([python, '-m', 'pip', 'install', '-q', 'huggingface_hub'])}) && "
                       f"{shlex.join([python, '-c', code])}")
                parts.append(f"({hub} || {shlex.join([python, '-c', fallback])})")
        command = [self.entrypoint] if self.entrypoint else []
        parts.append("exec " + shlex.join(command + self.arguments))
        return " && ".join(parts)


# ----------------------------------------------------------------------------- RunPod
class RunPod:
    API = "https://rest.runpod.io/v1"
    name = "runpod"
    start_grace = 1800  # a pod has no uptime until its image is pulled, and a big image can take 15 min

    def __init__(self, args):
        self.cloud = args.cloud

    @staticmethod
    def key():
        key = os.environ.get("RUNPOD_API_KEY")
        if key:
            return key
        cfg = Path.home() / ".runpod" / "config.toml"
        if cfg.exists():
            match = re.search(r'apikey\s*=\s*"([^"]+)"', cfg.read_text(encoding="utf-8"), re.IGNORECASE)
            if match:
                return match.group(1)
        raise SystemExit("no RunPod API key: set RUNPOD_API_KEY or run `runpodctl config`")

    def call(self, method, path, body=None, timeout=60):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(f"{self.API}{path}", data=data, method=method, headers={
            "Authorization": f"Bearer {self.key()}", "Content-Type": "application/json",
            "User-Agent": "local-ai-lab/1"})  # Cloudflare in front of RunPod blocks the urllib default (error 1010)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            raise SystemExit(f"RunPod {method} {path} failed: {error.code} {error.read().decode(errors='replace')[:500]}")
        return json.loads(raw) if raw else None

    def gpu_ids(self, spec):
        if spec.gpu_override:
            return [spec.gpu_override]
        ids = (GPUS.get(spec.hardware_id) or {}).get("runpod")
        if not ids:
            raise SystemExit(f"no RunPod GPU mapping for {spec.hardware_id}; pass --gpu")
        return ids

    def describe(self, spec):
        return f"{self.gpu_ids(spec)[0]} ({self.cloud})"

    def create(self, spec, exclude):
        body = {
            "name": spec.name, "imageName": spec.image,
            "dockerEntrypoint": [spec.entrypoint] if spec.entrypoint else [],
            "dockerStartCmd": spec.arguments, "env": spec.env, "ports": [f"{spec.port}/http"],
            "containerDiskInGb": spec.disk, "volumeInGb": 0, "gpuTypeIds": self.gpu_ids(spec),
            "gpuTypePriority": "custom", "gpuCount": 1, "interruptible": False,
        }
        if spec.provision:
            if not spec.entrypoint:
                raise SystemExit("in-container provisioning needs an explicit entrypoint in the contract")
            body["dockerEntrypoint"], body["dockerStartCmd"] = ["/bin/sh", "-c"], [spec.onstart_script()]
        for cloud in dict.fromkeys([self.cloud, "SECURE", "COMMUNITY"]):
            if cloud in exclude:  # a pod there never started: try the other cloud
                continue
            body["cloudType"] = cloud
            try:
                pod = self.call("POST", "/pods", body)
                self.cloud = cloud
                return {"id": pod["id"], "cost": pod.get("costPerHr"), "gpu": self.gpu_ids(spec)[0], "cloud": cloud,
                        "endpoint": f"https://{pod['id']}-{spec.port}.proxy.runpod.net"}
            except SystemExit as error:
                if not re.search(r"instances (currently )?available", str(error)):
                    raise
                log(f"no {cloud} instances for {self.gpu_ids(spec)[0]} right now")
        raise SystemExit(f"no RunPod instances available for {self.gpu_ids(spec)[0]}; try later or lower --disk")

    def poll(self, handle):
        state = self.call("GET", f"/pods/{handle['id']}")
        status = state.get("desiredStatus")
        if status in ("EXITED", "TERMINATED"):
            raise SystemExit(f"pod {status} before the server became healthy")
        started = bool(self.uptime(handle["id"]))
        handle["gpu"] = (state.get("machine") or {}).get("gpuTypeId") or handle["gpu"]
        return started, f"pod {status}"

    def uptime(self, pod_id):  # a REST pod has no runtime field; GraphQL has it
        query = '{ pod(input: {podId: "%s"}) { runtime { uptimeInSeconds } } }' % pod_id
        request = urllib.request.Request("https://api.runpod.io/graphql", data=json.dumps({"query": query}).encode(), headers={
            "Authorization": f"Bearer {self.key()}", "Content-Type": "application/json", "User-Agent": "local-ai-lab/1"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                pod = (json.loads(response.read()).get("data") or {}).get("pod") or {}
        except (OSError, ValueError):
            return None
        return (pod.get("runtime") or {}).get("uptimeInSeconds")

    def destroy(self, handle):
        self.call("DELETE", f"/pods/{handle['id']}")


# ----------------------------------------------------------------------------- Vast.ai
class Vast:
    name = "vast"

    def __init__(self, args):
        if not shutil.which("vastai"):
            raise SystemExit("vastai CLI not found: `uv tool install vastai` and put the key in ~/.config/vastai/vast_api_key")
        self.min_inet = args.vast_min_inet
        self.verified = "any" if getattr(args, "vast_any_host", False) else "true"  # unverified hosts: for cards verified hosts never offer
        self.min_cuda = args.vast_min_cuda
        self.max_price = args.vast_max_price
        # an offload recipe keeps experts in host RAM and reads tables from NVMe: rent only a host that has them
        self.min_ram = getattr(args, "vast_min_ram", 0) or 0
        self.min_disk_bw = getattr(args, "vast_min_disk_bw", 0) or 0
        self.min_pcie = getattr(args, "vast_min_pcie", 0) or 0
        # hosts bill the download per GB; at $0.04/GB a 70 GB image and weights pull costs more than hours of compute
        self.max_inet_cost = getattr(args, "vast_max_inet_cost", 0.02)

    @staticmethod
    def cli(*argv, timeout=120):
        # `destroy` asks for confirmation on stdin; `create` prints "Started. {python dict}" even with --raw
        # --raw must precede --args: everything after --args is passed to the container
        result = subprocess.run(["vastai", *argv[:2], "--raw", *argv[2:]], capture_output=True, text=True, timeout=timeout, input="y\n")
        text = (result.stdout or "").strip()
        err = (result.stderr or "").strip()
        if result.returncode != 0 or (not text and '"error": true' in err):
            raise SystemExit(f"vastai {' '.join(argv[:2])} failed: {(err or text)[:500]}")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        start = text.find("{")
        if start >= 0:
            import ast
            try:
                return ast.literal_eval(text[start:])
            except (ValueError, SyntaxError):
                pass
        if argv[0] == "destroy":
            return {"success": "destroying" in text}
        raise SystemExit(f"vastai {' '.join(argv[:2])}: unexpected output: {text[:300]}")

    def gpu_name(self, spec):
        if spec.gpu_override:
            return spec.gpu_override
        name = (GPUS.get(spec.hardware_id) or {}).get("vast")
        if not name:
            raise SystemExit(f"no Vast GPU mapping for {spec.hardware_id}; pass --gpu")
        return name

    def describe(self, spec):
        return f"{self.gpu_name(spec)} (vast)"

    def offers(self, spec, exclude):
        name = self.gpu_name(spec).replace(" ", "_")
        # query filter is in GB (results report MB); bound both sides so an 8 GB recipe never runs on the 16 GB variant of the same name
        lo, hi = int(max(spec.vram_gb - 1, 1)), int(spec.vram_gb + 1)
        # cuda_max_good is the host driver's CUDA ceiling; the image's NVIDIA_REQUIRE_CUDA floor must be at or below it
        # (12.9 for the pinned sglang cu12 image; 13.0 for vllm-openai and sglang dev-cu13; 13.2 for tabbyapi cu13 and
        # the attested tabbyapi-exl3). Older drivers cannot init the image (12.4 hosts fail llama.cpp too).
        query = (f"num_gpus=1 rentable=true verified={self.verified} gpu_name={name} gpu_ram>={lo} "  # the <= side is applied below: the server reads it in MB
                 f"inet_down>={self.min_inet} disk_space>={spec.disk + 5} reliability>0.9 cuda_max_good>={self.min_cuda} "
                 f"dph_total<={self.max_price} "  # the campaign ceiling; a card with no offer under it is reported, never rented above it
                 + (f"cpu_ram>={self.min_ram} " if self.min_ram else "")  # GB, like gpu_ram
                 + (f"disk_bw>={self.min_disk_bw} " if self.min_disk_bw else "")  # MB/s
                 + (f"pcie_bw>={self.min_pcie} " if self.min_pcie else "")  # GB/s host-to-device: offloaded experts cross it every token
                 + (f"inet_down_cost<={self.max_inet_cost} " if self.max_inet_cost else "")  # $/GB downloaded
                 + "geolocation notin [CN]")  # hosts that cannot reach Hugging Face never finish the weights download
        offers = self.cli("search", "offers", query, "-o", "dph_total")
        offers = [o for o in offers if o.get("id") not in exclude and o.get("machine_id") not in exclude
                  and (o.get("gpu_ram") or 0) <= hi * 1024]  # results report MB; keep the 8 GB variant off a 16 GB card and vice versa
        if not offers:
            raise SystemExit(f"no Vast offers for {query}")
        # a run's cost: about two hours of compute plus the pull (image and weights, at most the disk we ask for)
        est = lambda o: 2 * (o.get("dph_total") or 0) + (o.get("inet_down_cost") or 0) * spec.disk
        offers.sort(key=est)
        # cheapest hosts are often the slowest pullers: within 1.5x of the best estimate, take the fastest downlink first
        cap = est(offers[0]) * 1.5
        cheap = [o for o in offers if est(o) <= cap]
        cheap.sort(key=lambda o: -(o.get("inet_down") or 0))
        return cheap + [o for o in offers if o not in cheap]

    def create(self, spec, exclude):
        env = f"-p {spec.port}:{spec.port} " + " ".join(f"-e {k}={v}" for k, v in spec.env.items())
        for offer in self.offers(spec, exclude)[:5]:  # search results go stale; try the next cheapest
            argv = ["create", "instance", str(offer["id"]), "--image", spec.image, "--disk", str(spec.disk),
                    "--env", env, "--label", spec.name, "--cancel-unavail"]
            if spec.provision:
                # the plugin bind-mounts weights and config; here the container provisions them, then execs the contract
                if not spec.entrypoint:
                    raise SystemExit("in-container provisioning needs an explicit entrypoint in the contract")
                # the onstart override is exec'd as one argv token, so the shell must be the entrypoint
                argv += ["--onstart-cmd", "/bin/sh", "--args", "-c", spec.onstart_script()]
            else:
                if spec.entrypoint:
                    argv += ["--entrypoint", spec.entrypoint]
                if spec.arguments:
                    argv += ["--args", *spec.arguments]
            try:
                result = self.cli(*argv)
            except SystemExit as error:
                if "no_such_ask" in str(error) or "not available" in str(error):
                    log(f"offer {offer['id']} gone; trying the next one")
                    exclude.add(offer["id"])
                    continue
                raise
            if not result.get("success"):
                raise SystemExit(f"vast create failed: {result}")
            return {"id": result["new_contract"], "offer": offer["id"], "machine": offer.get("machine_id"),
                    "cost": offer.get("dph_total"), "gpu": offer.get("gpu_name"), "host": offer.get("geolocation"), "endpoint": None}
        raise SystemExit("every Vast offer tried was already gone; retry in a minute")

    def poll(self, handle):
        info = self.cli("show", "instance", str(handle["id"]))
        status = info.get("actual_status") or info.get("cur_state") or "?"
        full = (info.get("status_msg") or "").strip()
        if full and full != handle.get("last_msg"):
            handle["last_msg"] = full
            log(f"vast status: {full[:600]}")
        msg = full.splitlines()
        msg = msg[-1][:80] if msg else ""
        if status in ("exited", "stopped", "error") or (info.get("intended_status") == "stopped"):
            raise SystemExit(f"vast instance {status}: {msg}")
        ports = (info.get("ports") or {}).get(f"{handle['port']}/tcp") or []
        ip = info.get("public_ipaddr")
        if ports and ip:
            handle["endpoint"] = f"http://{ip}:{ports[0]['HostPort']}"
        started = status == "running"
        return started, f"vast {status} {msg}".strip()

    LISTENING = ("listening on http", "uvicorn running on", "application startup complete", "the server is fired up")

    def listening(self, handle):
        """True once the container log says the server is bound; only then does an unreachable port mean a firewalled host."""
        try:
            result = subprocess.run(["vastai", "logs", str(handle["id"]), "--tail", "400"], capture_output=True, text=True, timeout=90)
        except subprocess.TimeoutExpired:
            return False
        text = (result.stdout or "").lower()
        return any(marker in text for marker in self.LISTENING)

    def destroy(self, handle):
        self.cli("destroy", "instance", str(handle["id"]))


PROVIDERS = {"runpod": RunPod, "vast": Vast}
