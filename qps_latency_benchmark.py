#!/usr/bin/env python3
"""
Non-streaming QPS vs Median Latency benchmark for two backends:
  - Fireworks hosted deployment (OpenAI-compatible HTTP API)
  - SageMaker endpoint running the Fireworks container (boto3 invoke)

This script aligns request shape and prompt generation with benchmark/llm_bench/load_test.py:
  - Prompt uses a single-token prefix repeated plus a fixed suffix
  - Fireworks payload includes min_tokens to discourage early stop
  - Non-streaming only; use server 'usage' if present

It sweeps specified QPS values and, for each backend, issues requests at fixed QPS for a configured duration,
collects per-request total latency, and computes median latency. Results are saved to CSV and plotted.

Usage example:
uv run qps_latency_benchmark.py \
  --fireworks-model accounts/jmiano/deployedModels/qwen3-8b-123 \
  --fireworks-sagemaker-endpoint endpoint-123 \
  --hf-sagemaker-endpoint my-hf-endpoint \
  --region us-west-2 \
  --qps-list 0.5,1,2,5,10,20,50,100 \
  --duration-per-qps 20 \
  --results-dir results


Environment:
  - FIREWORKS_API_KEY can be set instead of --fireworks-api-key
  - .env is auto-loaded (python-dotenv) if available

Dependencies:
  pip install requests boto3 pandas matplotlib
"""

import argparse
import concurrent.futures
import dataclasses
import json
import math
import os
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import boto3
from botocore.config import Config as BotoConfig
import matplotlib.pyplot as plt
import pandas as pd
import requests
from requests.adapters import HTTPAdapter

try:
    # Auto-load environment variables from a .env file if present
    import dotenv  # type: ignore

    dotenv.load_dotenv()
except Exception:
    pass


# ----------------------------
# Prompt construction (matches load_test.py)
# ----------------------------

PROMPT_PREFIX_TOKEN = "Pad "  # exactly one token
PROMPT_SUFFIX = (
    "Generate a Django application with Authentication, JWT, Tests, DB support. "
    "Show docker-compose for python and postgres. Show the complete code for every file!"
)
PROMPT_SUFFIX_TOKENS = 35


def build_prompt_by_token_count(prompt_tokens: int) -> str:
    if prompt_tokens < PROMPT_SUFFIX_TOKENS:
        return PROMPT_PREFIX_TOKEN * prompt_tokens
    pad_tokens = prompt_tokens - PROMPT_SUFFIX_TOKENS
    return (PROMPT_PREFIX_TOKEN * pad_tokens) + PROMPT_SUFFIX


def maybe_randomize_prompt(prompt: str, enable: bool) -> str:
    if not enable:
        return prompt
    # Replace pad area tokens with random single-letter tokens to avoid server-side caching
    # Preserve the suffix for realism
    pad_len = max(0, len(prompt) - len(PROMPT_SUFFIX))
    if pad_len <= 0:
        # Just randomize full prompt length by characters (approx tokens)
        num_tokens = len(prompt) // len(PROMPT_PREFIX_TOKEN)
        import random

        return " ".join(chr(ord("a") + random.randint(0, 25)) for _ in range(num_tokens))
    else:
        import random

        num_random_tokens = pad_len // len(PROMPT_PREFIX_TOKEN)
        rand = " ".join(chr(ord("a") + random.randint(0, 25)) for _ in range(num_random_tokens))
        # Maintain one space before suffix when present
        return (rand + " " + PROMPT_SUFFIX) if PROMPT_SUFFIX else rand


# ----------------------------
# Fireworks client (non-streaming)
# ----------------------------


@dataclasses.dataclass
class GenResult:
    latency_s: float
    ok: bool
    status: Optional[int]
    error: Optional[str]
    response_input_tokens: Optional[int] = None
    response_output_tokens: Optional[int] = None


def _fireworks_headers(api_key: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def _fireworks_payload(
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: Optional[int],
    prompt_cache_max_len: int,
) -> Dict[str, Any]:
    data = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,  # discourage early stop (as in FireworksProvider)
        "temperature": temperature,
        "top_p": top_p,
        "n": 1,
        "stream": False,
        "prompt_cache_max_len": prompt_cache_max_len,
    }
    if top_k is not None:
        data["top_k"] = top_k
    return data


def fireworks_call(
    *,
    model: str,
    api_key: str,
    host: str,
    session: requests.Session,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: Optional[int],
    prompt_cache_max_len: int,
    timeout_s: int,
) -> GenResult:
    url = host.rstrip("/") + "/v1/completions"
    payload = _fireworks_payload(model, prompt, max_tokens, temperature, top_p, top_k, prompt_cache_max_len)
    headers = _fireworks_headers(api_key)
    t0 = time.perf_counter()
    try:
        r = session.post(url, headers=headers, data=json.dumps(payload), timeout=timeout_s)
        latency = time.perf_counter() - t0
        if r.status_code >= 200 and r.status_code < 300:
            resp_json = None
            try:
                resp_json = r.json()
            except Exception:
                resp_json = None
            in_tok = None
            out_tok = None
            if isinstance(resp_json, dict):
                usage = resp_json.get("usage") or {}
                in_tok = usage.get("prompt_tokens")
                out_tok = usage.get("completion_tokens")
            return GenResult(latency_s=latency, ok=True, status=r.status_code, error=None,
                             response_input_tokens=in_tok, response_output_tokens=out_tok)
        return GenResult(latency_s=latency, ok=False, status=r.status_code, error=r.text[:2000])
    except Exception as e:
        latency = time.perf_counter() - t0
        return GenResult(latency_s=latency, ok=False, status=None, error=str(e))


# ----------------------------
# SageMaker client (non-streaming)
# ----------------------------


class SageMakerInvoker:
    def __init__(
        self,
        region_name: str,
        endpoint_name: str,
        max_pool_connections: int,
        inference_component_name: Optional[str] = None,
        api_mode: str = "tgi",  # "tgi" (HF DLC) or "fireworks" (FW container)
        debug: bool = False,
    ):
        self.endpoint_name = endpoint_name
        self.inference_component_name = inference_component_name
        self.api_mode = api_mode
        self.debug = debug
        # boto3 clients are thread-safe
        cfg = BotoConfig(max_pool_connections=max(10, max_pool_connections))
        self.runtime = boto3.client("runtime.sagemaker", region_name=region_name, config=cfg)
        # Try to auto-detect inference component if not provided (for IC-based endpoints)
        if self.inference_component_name is None:
            try:
                sm = boto3.client("sagemaker", region_name=region_name)
                next_token = None
                ics = []
                while True:
                    kwargs = {"EndpointNameEquals": self.endpoint_name}
                    if next_token:
                        kwargs["NextToken"] = next_token
                    resp = sm.list_inference_components(**kwargs)
                    ics.extend(resp.get("InferenceComponents") or [])
                    next_token = resp.get("NextToken")
                    if not next_token:
                        break
                if ics:
                    self.inference_component_name = ics[0].get("InferenceComponentName")
                    if self.debug:
                        print(f"[debug] Using inference component: {self.inference_component_name}")
            except Exception as e:
                if self.debug:
                    print(f"[debug] list_inference_components failed: {e}")
                # Best-effort only; fall back to classic endpoint invocation
                self.inference_component_name = None

    def call(
        self,
        *,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: Optional[int],
        timeout_s: int,
    ) -> GenResult:
        if self.api_mode == "tgi":
            # Hugging Face TGI DLC schema
            payload: Dict[str, Any] = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_tokens,
                    "min_new_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                },
                "stream": False,
            }
            if top_k is not None:
                payload["parameters"]["top_k"] = top_k
        else:
            # Fireworks container schema (OpenAI-like)
            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "min_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": False,
            }
            if top_k is not None:
                payload["top_k"] = top_k
        t0 = time.perf_counter()
        try:
            # boto3 doesn't expose a direct per-request timeout; rely on client config/env for network timeouts
            invoke_kwargs = dict(
                EndpointName=self.endpoint_name,
                ContentType="application/json",
                Body=json.dumps(payload),
            )
            if self.inference_component_name:
                # SageMaker expects header X-Amzn-SageMaker-Inference-Component-Name via SDK param
                invoke_kwargs["InferenceComponentName"] = self.inference_component_name
                if self.debug:
                    print(f"[debug] invoking IC endpoint with component={self.inference_component_name}")
            resp = self.runtime.invoke_endpoint(**invoke_kwargs)
            # read body to ensure full roundtrip
            body_text = resp["Body"].read().decode()
            in_tok = None
            out_tok = None
            try:
                parsed = json.loads(body_text)
                if self.api_mode == "tgi":
                    # TGI non-streaming often returns a list with one item containing 'details'
                    item = parsed[0] if isinstance(parsed, list) and parsed else parsed
                    details = item.get("details") if isinstance(item, dict) else None
                    if isinstance(details, dict):
                        prefill = details.get("prefill")
                        if isinstance(prefill, list):
                            in_tok = len(prefill)
                        out_tok = details.get("generated_tokens")
                else:
                    # Fireworks container typically returns OpenAI-like JSON with 'usage'
                    if isinstance(parsed, dict):
                        usage = parsed.get("usage") or {}
                        in_tok = usage.get("prompt_tokens")
                        out_tok = usage.get("completion_tokens")
            except Exception:
                pass
            latency = time.perf_counter() - t0
            return GenResult(latency_s=latency, ok=True, status=200, error=None,
                             response_input_tokens=in_tok, response_output_tokens=out_tok)
        except Exception as e:
            latency = time.perf_counter() - t0
            return GenResult(latency_s=latency, ok=False, status=None, error=str(e))


# ----------------------------
# Fixed-QPS runner
# ----------------------------


def run_fixed_qps(
    qps: float,
    duration_s: float,
    submit_fn,
    max_workers: int,
    prompt_base: str,
    prompt_randomize: bool,
    debug: bool = False,
) -> Tuple[List[GenResult], int]:
    """
    Issues requests at approximately fixed QPS for duration_s seconds using a thread pool.
    submit_fn must be a callable with no args that performs one request and returns GenResult.
    Returns (list of results, scheduled_requests_count).
    """
    if qps <= 0:
        return [], 0

    # Precompute schedule to avoid drift; ensure we do not exceed max number due to rounding
    start_t = time.perf_counter()
    end_t = start_t + duration_s
    interval = 1.0 / qps
    scheduled_times: List[float] = []
    t = start_t
    while t < end_t:
        scheduled_times.append(t)
        t += interval

    results: List[GenResult] = []
    results_lock = threading.Lock()

    total_to_run = len(scheduled_times)
    if debug:
        print(
            f"[debug] Scheduling {total_to_run} requests at QPS={qps} over {duration_s:.1f}s; interval={interval:.3f}s",
            flush=True,
        )

    def worker_at(when: float):
        # Sleep until target time
        while True:
            now = time.perf_counter()
            delay = when - now
            if delay <= 0:
                break
            # cap sleep granularity
            time.sleep(min(delay, 0.005))
        # Build possibly randomized prompt
        prompt = maybe_randomize_prompt(prompt_base, prompt_randomize)
        if debug:
            t_start = time.perf_counter()
        r = submit_fn(prompt)
        if debug:
            t_end = time.perf_counter()
            print(
                f"[debug] request done ok={r.ok} status={r.status} measured_latency_ms={r.latency_s*1000:.1f} wall_ms={(t_end - t_start)*1000:.1f}",
                flush=True,
            )
        with results_lock:
            results.append(r)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker_at, st) for st in scheduled_times]
        completed = 0
        for f in concurrent.futures.as_completed(futures):
            _ = f.result()
            completed += 1
            if debug and (completed % max(1, total_to_run // 10) == 0 or completed == total_to_run):
                print(f"[debug] Progress: {completed}/{total_to_run} completed", flush=True)

    return results, len(scheduled_times)


# ----------------------------
# Orchestration
# ----------------------------


def summarize_latencies(results: Sequence[GenResult]) -> Dict[str, float]:
    latencies_ms = [r.latency_s * 1000.0 for r in results if r.ok]
    if not latencies_ms:
        return {
            "count": 0,
            "failures": len(results),
            "median_ms": float("nan"),
            "avg_ms": float("nan"),
            "p95_ms": float("nan"),
        }
    s = sorted(latencies_ms)
    n = len(s)
    median = s[n // 2] if n % 2 == 1 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    p95 = s[min(n - 1, int(math.ceil(0.95 * n)) - 1)]
    return {
        "count": n,
        "failures": len(results) - n,
        "median_ms": median,
        "avg_ms": sum(s) / n,
        "p95_ms": p95,
    }


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_qps_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="QPS vs Median Latency benchmark for Fireworks and SageMaker variants")
    p.add_argument("--fireworks-model", help="Fireworks model id (for fireworks-on-fireworks)")
    p.add_argument("--fireworks-api-key", help="Fireworks API key; otherwise read FIREWORKS_API_KEY env var")
    p.add_argument("--fireworks-host", default="https://api.fireworks.ai/inference", help="Fireworks inference base URL")
    p.add_argument("--fireworks-sagemaker-endpoint", help="Endpoint name for fireworks-on-sagemaker")
    p.add_argument("--hf-sagemaker-endpoint", help="Endpoint name for sagemaker-on-sagemaker (HF model default serving)")
    p.add_argument("--region", default=os.getenv("AWS_REGION", "us-west-2"), help="AWS region for SageMaker endpoint")

    p.add_argument("--prompt-tokens", type=int, default=512)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--prompt-cache-max-len", type=int, default=0)
    p.add_argument("--prompt-randomize", action="store_true", help="Randomize prompt pad region to avoid caching")

    p.add_argument("--qps-list", required=True, help="Comma-separated QPS values, e.g. 0.5,1,2,5,10")
    p.add_argument("--duration-per-qps", type=float, default=20.0, help="Seconds to run at each QPS")
    p.add_argument("--max-workers", type=int, default=100, help="Thread pool size")
    p.add_argument("--requests-timeout", type=int, default=60, help="Per-request timeout in seconds (HTTP only)")

    p.add_argument("--results-dir", default="results", help="Directory to write results and plots")
    p.add_argument("--run-name", default=None, help="Optional run name; defaults to timestamp")
    p.add_argument("--debug", action="store_true", help="Verbose scheduling and per-request progress logs")
    p.add_argument("--warmup-requests", type=int, default=5, help="Warmup requests per backend before measurements")
    p.add_argument(
        "--stabilization-seconds",
        type=float,
        default=5.0,
        help="Per-QPS stabilization window to discard (not recorded) before measuring",
    )

    args = p.parse_args(argv)

    fireworks_api_key = args.fireworks_api_key or os.getenv("FIREWORKS_API_KEY")

    qps_values = parse_qps_list(args.qps_list)
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.results_dir, f"qps_latency_{run_name}")
    ensure_dir(out_dir)

    # Static prompt base
    prompt_base = build_prompt_by_token_count(args.prompt_tokens)

    # Prepare possible backends based on provided flags
    backends: List[Tuple[str, Any]] = []
    # fireworks-on-fireworks
    if args.fireworks_model and fireworks_api_key:
        backends.append(("fireworks-on-fireworks", {
            "type": "fireworks",
        }))
    # fireworks-on-sagemaker
    fw_sm_invoker = None
    if args.fireworks_sagemaker_endpoint:
        fw_sm_invoker = SageMakerInvoker(
            region_name=args.region,
            endpoint_name=args.fireworks_sagemaker_endpoint,
            max_pool_connections=args.max_workers,
            api_mode="fireworks",
            debug=args.debug,
        )
        backends.append(("fireworks-on-sagemaker", {"type": "sagemaker", "invoker": fw_sm_invoker}))
    # sagemaker-on-sagemaker (HF default serving)
    hf_sm_invoker = None
    if args.hf_sagemaker_endpoint:
        hf_sm_invoker = SageMakerInvoker(
            region_name=args.region,
            endpoint_name=args.hf_sagemaker_endpoint,
            max_pool_connections=args.max_workers,
            api_mode="tgi",
            debug=args.debug,
        )
        backends.append(("sagemaker-on-sagemaker", {"type": "sagemaker", "invoker": hf_sm_invoker}))

    if not backends:
        print("ERROR: No backends selected. Provide at least one of --fireworks-model (with FIREWORKS_API_KEY), --fireworks-sagemaker-endpoint, or --hf-sagemaker-endpoint.", file=sys.stderr)
        return 2

    # Configure a requests session with a larger connection pool for high-QPS HTTP
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=max(10, args.max_workers), pool_maxsize=max(10, args.max_workers))
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    # Containers for results
    rows: List[Dict[str, Any]] = []
    # Per-request log entries for post-run token distribution checks
    req_logs: List[Dict[str, Any]] = []

    print("Starting benchmark...\n")
    print(f"QPS values: {qps_values}")
    print(f"Duration per QPS: {args.duration_per_qps}s")
    print(f"Max workers: {args.max_workers}")
    print()

    for backend_name, backend_info in backends:
        print(f"--- Backend: {backend_name} ---")
        # Warmup to mitigate cold start
        if args.warmup_requests > 0:
            print(f"Warming up {backend_name} with {args.warmup_requests} request(s)...", flush=True)
            for _ in range(args.warmup_requests):
                if backend_info["type"] == "fireworks":
                    _ = fireworks_call(
                        model=args.fireworks_model,
                        api_key=fireworks_api_key,
                        host=args.fireworks_host,
                        session=session,
                        prompt=prompt_base,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        prompt_cache_max_len=args.prompt_cache_max_len,
                        timeout_s=args.requests_timeout,
                    )
                else:
                    _ = backend_info["invoker"].call(
                        prompt=prompt_base,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        timeout_s=args.requests_timeout,
                    )

        for qps in qps_values:
            print(f"Running QPS={qps} ...", end=" ", flush=True)

            if backend_info["type"] == "fireworks":
                def submit_fn(prompt: str) -> GenResult:
                    return fireworks_call(
                        model=args.fireworks_model,
                        api_key=fireworks_api_key,
                        host=args.fireworks_host,
                        session=session,
                        prompt=prompt,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        prompt_cache_max_len=args.prompt_cache_max_len,
                        timeout_s=args.requests_timeout,
                    )
            else:
                def submit_fn(prompt: str) -> GenResult:
                    r = backend_info["invoker"].call(
                        prompt=prompt,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        timeout_s=args.requests_timeout,
                    )
                    if args.debug and not r.ok:
                        print(f"[debug] invoke failed: status={r.status} err={r.error[:200]}...")
                    return r

            # Optional stabilization phase at this QPS (discard results)
            if args.stabilization_seconds and args.stabilization_seconds > 0:
                _ = run_fixed_qps(
                    qps=qps,
                    duration_s=args.stabilization_seconds,
                    submit_fn=submit_fn,
                    max_workers=args.max_workers,
                    prompt_base=prompt_base,
                    prompt_randomize=args.prompt_randomize,
                    debug=False,
                )

            results, scheduled = run_fixed_qps(
                qps=qps,
                duration_s=args.duration_per_qps,
                submit_fn=submit_fn,
                max_workers=args.max_workers,
                prompt_base=prompt_base,
                prompt_randomize=args.prompt_randomize,
                debug=args.debug,
            )

            summary = summarize_latencies(results)
            print(
                f"done: scheduled={scheduled} ok={summary['count']} fail={summary['failures']} "
                f"median={summary['median_ms']:.1f}ms avg={summary['avg_ms']:.1f}ms p95={summary['p95_ms']:.1f}ms"
            )

            rows.append(
                {
                    "backend": backend_name,
                    "qps": qps,
                    "scheduled": scheduled,
                    "ok": summary["count"],
                    "fail": summary["failures"],
                    "median_ms": summary["median_ms"],
                    "avg_ms": summary["avg_ms"],
                    "p95_ms": summary["p95_ms"],
                }
            )

            # Append per-request logs for this QPS bucket
            for r in results:
                req_logs.append(
                    {
                        "backend": backend_name,
                        "qps": qps,
                        "ok": r.ok,
                        "status": r.status,
                        "latency_ms": r.latency_s * 1000.0,
                        "input_tokens": r.response_input_tokens,
                        "output_tokens": r.response_output_tokens,
                    }
                )

    # Save CSVs and plot
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "qps_latency_results.csv")
    df.sort_values(["backend", "qps"]).to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")

    # Save per-request log CSV
    if req_logs:
        df_logs = pd.DataFrame(req_logs)
        logs_path = os.path.join(out_dir, "qps_latency_request_logs.csv")
        df_logs.to_csv(logs_path, index=False)
        print(f"Saved per-request logs to {logs_path}")

    # Pivot for plotting
    # Combined plot: median latency vs QPS for all selected backends
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {
        "fireworks-on-fireworks": "#1f77b4",
        "fireworks-on-sagemaker": "#ff7f0e",
        "sagemaker-on-sagemaker": "#2ca02c",
    }
    for backend_name in df["backend"].unique():
        sub = df[df["backend"] == backend_name].sort_values("qps")
        ax.plot(sub["qps"], sub["median_ms"], marker="o", label=backend_name, color=colors.get(backend_name))
    ax.set_xlabel("QPS (requests/sec)")
    ax.set_ylabel("Median total latency (ms)")
    ax.set_title("QPS vs Median Latency (non-streaming)")
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend()
    plot_path = os.path.join(out_dir, "qps_vs_median_latency.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}")

    # Separate plots per backend with the same y-axis scale
    ymax = float(df["median_ms"].max()) if not df.empty else 0.0
    for backend_name in df["backend"].unique():
        sub = df[df["backend"] == backend_name].sort_values("qps")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.plot(sub["qps"], sub["median_ms"], marker="o", label=backend_name, color=colors.get(backend_name))
        ax2.set_xlabel("QPS (requests/sec)")
        ax2.set_ylabel("Median total latency (ms)")
        ax2.set_title(f"QPS vs Median Latency (non-streaming) - {backend_name}")
        ax2.grid(True, linestyle=":", linewidth=0.5)
        if ymax > 0:
            ax2.set_ylim(0, ymax * 1.05)
        ax2.legend()
        fname = f"qps_vs_median_latency_{backend_name.replace('/', '_').replace(' ', '_')}.png"
        out_path = os.path.join(out_dir, fname)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig2)
        print(f"Saved plot to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


