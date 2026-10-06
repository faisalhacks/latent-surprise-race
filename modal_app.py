"""Phase 2 - frozen V-JEPA 2 inference on Modal.

Layout
------
* `extract_chunk`  - GPU. Generates its own clips (deterministic from the frozen
  seed, so nothing is uploaded), runs the strictly-online extraction, writes one
  npz per chunk to a Volume. Fanned out with `.map()`, so wall-clock drops while
  total GPU-seconds stay the same.
* `precision_check` - GPU. Compares fp32 against bf16 probe values. Probe A is a
  second difference of nearby vectors, which is the classic setting for
  catastrophic cancellation, so the precision choice is measured rather than
  assumed.
* `analyse_remote` - CPU. Merges chunks, runs the frozen Phase 1.3 analysis and
  the Phase 3 gate, writes the report to the Volume and returns it.

Usage
-----
    modal run modal_app.py --smoke                 # a few cents, validates GPU path
    modal run modal_app.py --precision-check       # fp32 vs bf16 numerics
    modal run modal_app.py                         # full Phase 2 run
    modal volume get lgw-results <tag>/ ./results  # fetch latents and report
"""
from __future__ import annotations

import json
import os
import time

import modal

APP_NAME = "lgw-vjepa2"
VOLUME_NAME = "lgw-results"
MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
#️ Modal validates every function in the app at hydration time, so a GPU-
# decorated function the run never calls still blocks the whole app on an
# account without a payment method. Setting LGW_DRYRUN=1 drops every GPU
# request so the CPU-only orchestration dry run can hydrate and execute.
GPU_TYPE = None if os.environ.get("LGW_DRYRUN") else os.environ.get("LGW_GPU", "A10G")
REMOTE_SRC = "/root/src"

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _download_weights() -> None:
    from transformers import VJEPA2Model
    VJEPA2Model.from_pretrained(MODEL_ID)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers==4.57.3", "numpy<3", "scipy",
                 "safetensors", "accelerate", "huggingface_hub", "matplotlib")
    .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})
    .run_function(_download_weights)          # bake weights into the image layer
    .add_local_dir("src", remote_path=REMOTE_SRC)
)


# --------------------------------------------------------------------------
# job planning (runs locally, pure bookkeeping)
# --------------------------------------------------------------------------
def plan_jobs(n_per_cond: int, n_calib: int, n_eval: int, chunk: int):
    """-> list of {split, specs:[(clip_index, condition_or_None)], chunk_id}"""
    import sys
    sys.path.insert(0, "src")
    from lgw import config as C

    specs = []
    idx = 0
    for cond in C.CONDITIONS:
        for _ in range(n_per_cond):
            specs.append(("anomaly", idx, cond))
            idx += 1
        idx = (C.CONDITIONS.index(cond) + 1) * C.N_PER_CONDITION
    for j in range(n_calib):
        specs.append(("control_calib", 1_000_000 + j, None))
    for j in range(n_eval):
        specs.append(("control_eval", 2_000_000 + j, None))

    jobs, by_split = [], {}
    for split, i, cond in specs:
        by_split.setdefault(split, []).append((i, cond))
    for split, items in by_split.items():
        for k in range(0, len(items), chunk):
            jobs.append({"split": split, "specs": items[k:k + chunk],
                         "chunk_id": len(jobs)})
    return jobs


# --------------------------------------------------------------------------
# GPU: extraction
# --------------------------------------------------------------------------
@app.function(image=image, gpu=GPU_TYPE, timeout=60 * 60,
              volumes={"/out": vol}, retries=1)
def extract_chunk(job: dict, tag: str, precision: str = "fp32",
                  batch_size: int = 4) -> dict:
    import sys
    sys.path.insert(0, REMOTE_SRC)

    import numpy as np
    import torch
    from transformers import VJEPA2Model

    from lgw.datagen import make_clip
    from lgw.extract import run_extraction
    from lgw.vjepa2_adapter import VJEPA2Adapter

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16,
             "fp16": torch.float16}[precision]

    t0 = time.time()
    clips = np.stack([make_clip(i, c)[0] for i, c in job["specs"]])
    conds = np.array([c if c is not None else "control" for _, c in job["specs"]])
    ids = np.array([i for i, _ in job["specs"]], dtype=np.int64)

    model = VJEPA2Model.from_pretrained(MODEL_ID, dtype=dtype,
                                        attn_implementation="sdpa")
    adapter = VJEPA2Adapter(model, device="cuda", dtype=dtype, name=MODEL_ID)
    out = run_extraction(clips, adapter, batch_size=batch_size, progress=False)

    d = os.path.join("/out", tag, "chunks")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "{}_{:04d}.npz".format(job["split"], job["chunk_id"]))
    np.savez_compressed(path, condition=conds, clip_id=ids,
                        **{k: out[k] for k in
                           ("Z", "loss_L0", "probe_A", "probe_B", "probe_C")})
    vol.commit()
    dt = time.time() - t0
    print("chunk {} [{}] {} clips in {:.1f}s".format(
        job["chunk_id"], job["split"], clips.shape[0], dt), flush=True)
    return {"path": path, "split": job["split"], "n": int(clips.shape[0]),
            "seconds": dt, "gpu": GPU_TYPE or "cpu"}


# --------------------------------------------------------------------------
# CPU: orchestration dry run with a tiny randomly-initialised V-JEPA 2
#
# Exercises the entire pipeline - fan-out, volume writes, chunk merge, the
# frozen analysis, the gate and the report - at the full 1200-clip scale,
# without a GPU. The only thing it does not test is the GPU forward itself, so
# when real weights run, the orchestration is already known good.
# Its numbers are meaningless: the weights are random.
# --------------------------------------------------------------------------
@app.function(image=image, timeout=60 * 60, volumes={"/out": vol},
              cpu=2.0, memory=8192, retries=1)
def extract_chunk_tiny(job: dict, tag: str, batch_size: int = 8) -> dict:
    import sys
    sys.path.insert(0, REMOTE_SRC)

    import numpy as np
    import torch
    from transformers import VJEPA2Config, VJEPA2Model

    from lgw.datagen import make_clip
    from lgw.extract import run_extraction
    from lgw.vjepa2_adapter import VJEPA2Adapter

    T_CTX, RES = 8, 32
    t0 = time.time()
    clips = np.stack([make_clip(i, c)[0] for i, c in job["specs"]])
    conds = np.array([c if c is not None else "control" for _, c in job["specs"]])
    ids = np.array([i for i, _ in job["specs"]], dtype=np.int64)

    cfg = VJEPA2Config(
        patch_size=16, crop_size=RES, frames_per_clip=T_CTX, tubelet_size=2,
        hidden_size=96, num_attention_heads=4, num_hidden_layers=2, mlp_ratio=2.0,
        pred_hidden_size=96, pred_num_attention_heads=4, pred_num_hidden_layers=2,
        pred_num_mask_tokens=4)
    torch.manual_seed(0)
    adapter = VJEPA2Adapter(VJEPA2Model(cfg), device="cpu", dtype=torch.float32,
                            t_ctx=T_CTX, res=RES, name="tiny-random")
    out = run_extraction(clips, adapter, batch_size=batch_size, progress=False)

    d = os.path.join("/out", tag, "chunks")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "{}_{:04d}.npz".format(job["split"], job["chunk_id"]))
    np.savez_compressed(path, condition=conds, clip_id=ids,
                        **{k: out[k] for k in
                           ("Z", "loss_L0", "probe_A", "probe_B", "probe_C")})
    vol.commit()
    dt = time.time() - t0
    print("chunk {} [{}] {} clips in {:.1f}s".format(
        job["chunk_id"], job["split"], clips.shape[0], dt), flush=True)
    return {"path": path, "split": job["split"], "n": int(clips.shape[0]),
            "seconds": dt, "gpu": "none (cpu dry run)"}


# --------------------------------------------------------------------------
# GPU: numerical precision diagnostic
# --------------------------------------------------------------------------
@app.function(image=image, gpu=GPU_TYPE, timeout=30 * 60, volumes={"/out": vol})
def precision_check(n_clips: int = 4) -> dict:
    import sys
    sys.path.insert(0, REMOTE_SRC)

    import numpy as np
    import torch
    from scipy import stats as sps
    from transformers import VJEPA2Model

    from lgw import config as C
    from lgw.datagen import make_clip
    from lgw.extract import run_extraction
    from lgw.vjepa2_adapter import VJEPA2Adapter

    torch.backends.cuda.matmul.allow_tf32 = True
    clips = np.stack([make_clip(i, C.CONDITIONS[i % len(C.CONDITIONS)])[0]
                      for i in range(n_clips)])

    runs = {}
    for prec, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16),
                        ("fp16", torch.float16)):
        model = VJEPA2Model.from_pretrained(MODEL_ID, dtype=dtype,
                                            attn_implementation="sdpa")
        ad = VJEPA2Adapter(model, device="cuda", dtype=dtype, name=MODEL_ID)
        t0 = time.time()
        runs[prec] = run_extraction(clips, ad, batch_size=2, progress=False)
        runs[prec]["_seconds"] = time.time() - t0
        del model, ad
        torch.cuda.empty_cache()

    ev = slice(C.EVAL_START, C.EVAL_END)
    report = {"seconds": {p: runs[p]["_seconds"] for p in runs}}
    for prec in ("bf16", "fp16"):
        rec = {}
        for k in ("loss_L0", "probe_A", "probe_B", "probe_C"):
            a = runs["fp32"][k][:, ev].ravel()
            b = runs[prec][k][:, ev].ravel()
            m = np.isfinite(a) & np.isfinite(b)
            a, b = a[m], b[m]
            denom = np.maximum(np.abs(a), 1e-12)
            rec[k] = {
                "median_rel_err": float(np.median(np.abs(a - b) / denom)),
                "max_rel_err": float(np.max(np.abs(a - b) / denom)),
                "spearman": float(sps.spearmanr(a, b).statistic),
                "pearson": float(np.corrcoef(a, b)[0, 1]),
            }
        report[prec] = rec
    print(json.dumps(report, indent=2), flush=True)
    return report


# --------------------------------------------------------------------------
# CPU: merge + frozen analysis + gate
# --------------------------------------------------------------------------
@app.function(image=image, timeout=60 * 60, volumes={"/out": vol},
              cpu=4.0, memory=16384)
def analyse_remote(tag: str, note: str = "") -> dict:
    import sys
    sys.path.insert(0, REMOTE_SRC)

    import glob
    import numpy as np

    from lgw.analysis import analyse

    vol.reload()
    scores = {}
    for split in ("anomaly", "control_calib", "control_eval"):
        files = sorted(glob.glob(os.path.join("/out", tag, "chunks",
                                              "{}_*.npz".format(split))))
        if not files:
            raise FileNotFoundError("no chunks for split " + split)
        parts = [np.load(f, allow_pickle=False) for f in files]
        merged = {k: np.concatenate([p[k] for p in parts], axis=0)
                  for k in ("Z", "loss_L0", "probe_A", "probe_B", "probe_C",
                            "condition", "clip_id")}
        order = np.argsort(merged["clip_id"], kind="stable")
        scores[split] = {k: v[order] for k, v in merged.items()}
        print("{:15s} {} clips".format(split, scores[split]["Z"].shape[0]), flush=True)

    out_dir = os.path.join("/out", tag, "analysis")
    r = analyse(scores, out_dir, model_name=MODEL_ID, note=note)
    vol.commit()
    return r


# --------------------------------------------------------------------------
# local entrypoint
# --------------------------------------------------------------------------
@app.local_entrypoint()
def main(smoke: bool = False, precision_check_only: bool = False,
         arch_dryrun: bool = False, precision: str = "fp32", chunk: int = 25,
         batch_size: int = 4, tag: str = "", n_per_cond: int = 0):
    import sys
    sys.path.insert(0, "src")
    from lgw import config as C

    if precision_check_only:
        print("running fp32/bf16/fp16 comparison on {} ...".format(GPU_TYPE))
        rep = precision_check.remote()
        os.makedirs("results", exist_ok=True)
        with open("results/precision_check.json", "w") as f:
            json.dump(rep, f, indent=2)
        print("wrote results/precision_check.json")
        return

    if arch_dryrun:
        n_per_cond = n_per_cond or C.N_PER_CONDITION
        n_calib, n_eval = C.N_CONTROL_CALIB, C.N_CONTROL_EVAL
        tag = tag or "archdry-{}".format(time.strftime("%Y%m%d-%H%M%S"))
        note = ("ORCHESTRATION DRY RUN - randomly initialised tiny V-JEPA 2 on CPU. "
                "Validates the pipeline at full scale. The numbers are meaningless.")
    elif smoke:
        n_per_cond, n_calib, n_eval = 3, 8, 8
        tag = tag or "smoke-{}".format(time.strftime("%Y%m%d-%H%M%S"))
        note = "SMOKE RUN - tiny subset, not a result."
    else:
        n_per_cond = n_per_cond or C.N_PER_CONDITION
        n_calib, n_eval = C.N_CONTROL_CALIB, C.N_CONTROL_EVAL
        tag = tag or "phase2-{}".format(time.strftime("%Y%m%d-%H%M%S"))
        note = "Phase 2: frozen V-JEPA 2, pre-registered protocol."

    jobs = plan_jobs(n_per_cond, n_calib, n_eval, chunk)
    n_clips = sum(len(j["specs"]) for j in jobs)
    print("tag={}  gpu={}  precision={}  clips={}  chunks={}".format(
        tag, GPU_TYPE, precision, n_clips, len(jobs)))

    t0 = time.time()
    if arch_dryrun:
        results = list(extract_chunk_tiny.map(
            jobs, kwargs={"tag": tag, "batch_size": 8}))
    else:
        results = list(extract_chunk.map(
            jobs, kwargs={"tag": tag, "precision": precision,
                          "batch_size": batch_size}))
    gpu_seconds = sum(r["seconds"] for r in results)
    wall = time.time() - t0
    print("\nextraction done: {} clips, {:.0f} GPU-s, {:.0f}s wall".format(
        sum(r["n"] for r in results), gpu_seconds, wall))
    print("GPU-seconds per clip: {:.2f}".format(gpu_seconds / max(n_clips, 1)))

    r = analyse_remote.remote(tag, note)
    g = r["gate"]
    print("\n================ Phase 3 gate ================")
    print("verdict:                     {}".format(g["verdict"]))
    print("early warners (primary):     {}".format(g["early_warners"] or "none"))
    print("primary median degenerate:   {}".format(g["primary_median_degenerate"]))
    print("RMTA early warners:          {}".format(
        g["secondary_rmta_early_warners"] or "none"))
    print("clip-matched median warners: {}".format(
        g["clip_matched_median_early_warners"] or "none"))
    print("clip-matched RMTA warners:   {}".format(
        g["clip_matched_rmta_early_warners"] or "none"))
    print("==============================================")

    local = os.path.join("results", tag)
    os.makedirs(local, exist_ok=True)
    with open(os.path.join(local, "results.json"), "w") as f:
        json.dump(r, f, indent=2, default=float)
    with open(os.path.join(local, "run_meta.json"), "w") as f:
        json.dump({"tag": tag, "gpu": GPU_TYPE, "precision": precision,
                   "gpu_seconds": gpu_seconds, "wall_seconds": wall,
                   "n_clips": n_clips, "chunks": len(jobs),
                   "model": MODEL_ID}, f, indent=2)
    print("\nwrote {}".format(os.path.abspath(local)))
    print("fetch full outputs:  modal volume get {} {}/ ./results".format(
        VOLUME_NAME, tag))
