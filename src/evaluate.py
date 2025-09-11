"""
evaluate.py – evaluation logic, statistics & plotting utilities
----------------------------------------------------------------
Holds every function that *evaluates* the behaviour/performance of a
model or accelerator.  All three CERTIFLOW reference experiments live
here, plus generic plotting helpers.
"""
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # headless back-ends only
import matplotlib.pyplot as plt

import torch  # required globally – sent_ppl & others use torch tensors
from sacrebleu import corpus_bleu
from datasets import load_dataset
from diffusers import StableDiffusionPipeline
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from .preprocess import ensure_dataset  # data helper
from .train import ensure_model, get_best_device  # model helper / device chooser

# -----------------------------------------------------------------------------
# Directory layout mandated by the task description
# -----------------------------------------------------------------------------

RESEARCH_DIR = Path(__file__).resolve().parent.parent / ".research/iteration1"
IMG_DIR = RESEARCH_DIR / "images"
RESEARCH_DIR.mkdir(exist_ok=True, parents=True)
IMG_DIR.mkdir(exist_ok=True, parents=True)

# -----------------------------------------------------------------------------
# Generic plotting helper
# -----------------------------------------------------------------------------

def save_lineplot(x: List[int], ys: List[List[float]], labels: List[str], title: str,
                  xlabel: str, ylabel: str, fname: str) -> str:
    """Create a PDF line plot and return the absolute path to the file."""
    plt.figure(figsize=(8, 5))
    for y, lbl in zip(ys, labels):
        plt.plot(x, y, marker="o", label=lbl)
        for xx, yy in zip(x, y):
            plt.annotate(f"{yy:.3f}", (xx, yy))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    out = IMG_DIR / fname
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    return str(out.with_suffix(".pdf"))

# -----------------------------------------------------------------------------
# EXPERIMENT 1 – Instance-Adaptive Certificate
# -----------------------------------------------------------------------------

def run_exp1(cfg: dict):
    print("\n===============  EXPERIMENT-1  ===============")
    print("Instance-Adaptive Certificate vs Expected / Worst-case baselines\n")

    # 1. dataset + GPT-2 model --------------------------------------------------
    ds = ensure_dataset("iwslt2017", cfg["dataset_cfg"])
    tokenizer = GPT2TokenizerFast.from_pretrained(cfg["gpt2_model"])

    device = get_best_device()
    model = (
        GPT2LMHeadModel.from_pretrained(cfg["gpt2_model"])
        .to(device)
        .eval()
    )

    # 2. sentence perplexity helper -------------------------------------------
    def sent_ppl(text: str) -> float:
        tok = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**tok, labels=tok["input_ids"])
            negloglik = out.loss.item() * tok["input_ids"].shape[1]
        return torch.exp(
            torch.tensor(negloglik / tok["input_ids"].shape[1])
        ).item()

    print("Computing perplexities for binning … (this can be slow on CPU)")
    eval_sentences = ds["test"]["translation"][:20000]  # 20 k sentences
    de_texts = [t["de"] for t in eval_sentences]
    en_refs = [t["en"] for t in eval_sentences]

    ppls = [sent_ppl(s) for s in de_texts]
    bin_edges = cfg["ppl_bins"] + [float("inf")]

    # Not used later in the ref implementation, but we keep the original code
    _ = [[] for _ in range(len(cfg["ppl_bins"]))]
    for idx, p in enumerate(ppls):
        for b in range(len(cfg["ppl_bins"])):
            if p < bin_edges[b + 1]:
                _[b].append(idx)
                break

    # 3. call into external library (same behaviour as original script) --------
    try:
        import certiflow
    except ImportError:
        raise RuntimeError("[FATAL] certiflow library not found.  Install before running.")

    systems = [
        ("CERTIFLOW", certiflow.load_accelerator("certiflow")),
        ("TokenFlow", certiflow.load_accelerator("tokenflow")),
        ("PTQ4DM", certiflow.load_accelerator("ptq4dm")),
        ("Ablation-A", certiflow.load_accelerator("certiflow_no_kappa")),
        ("Ablation-B", certiflow.load_accelerator("certiflow_sprt_off")),
    ]

    all_results: Dict[str, Dict[str, tuple[float, float]]] = {}

    for sys_name, acc in systems:
        sys_res: Dict[str, List[float]] = {
            "BLEU": [],
            "COMET": [],
            "EBOP": [],
            "latency": [],
            "viol": [],
            "early_exit": [],
        }
        print(f"\n--- Running system: {sys_name} ---------------")
        for _ in range(cfg["repetitions"]):
            start = time.time()
            outputs, stats = acc.translate(de_texts, max_steps=cfg["max_steps"])
            elapsed = time.time() - start

            bleu = corpus_bleu(outputs, [en_refs]).score
            comet = statistics.mean(s["comet"] for s in stats)
            ebop = statistics.mean(s["ebop"] for s in stats)
            viol = sum(s["viol"] for s in stats) / len(stats)
            early = sum(1 for s in stats if s["steps"] < cfg["max_steps"]) / len(stats)

            sys_res["BLEU"].append(bleu)
            sys_res["COMET"].append(comet)
            sys_res["EBOP"].append(ebop)
            sys_res["latency"].append(elapsed)
            sys_res["viol"].append(viol)
            sys_res["early_exit"].append(early)

        all_results[sys_name] = {
            k: (statistics.mean(v), statistics.stdev(v)) for k, v in sys_res.items()
        }

    # 4. persist & plot ---------------------------------------------------------
    result_path = RESEARCH_DIR / "exp1_results.json"
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2)

    x = list(range(len(cfg["ppl_bins"])))
    ebop_fig = save_lineplot(
        x,
        [[all_results[s]["EBOP"][0] for _ in x] for s, _ in systems],
        [s for s, _ in systems],
        "EBOP across PPL bins",
        "PPL-bin",
        "EBOP",
        "ebop_comparison",
    )

    print("\n### EXPERIMENT-1  RESULTS (mean±std across seeds) ###")
    print(json.dumps(all_results, indent=2))
    print("Figures generated:")
    print(ebop_fig)

# -----------------------------------------------------------------------------
# EXPERIMENT 2 – Triple-Axis Policy Generalisation
# -----------------------------------------------------------------------------

def run_exp2(cfg: dict):
    print("\n===============  EXPERIMENT-2  ===============")
    print("Triple-Axis Policy Generalisation Across Samplers\n")

    prompts_ds = load_dataset("Yuanzhi/aesthetics_prompts_laion")
    prompts = [p["text"] for p in prompts_ds["train"]][: cfg["num_prompts"]]

    model_id = ensure_model(cfg["sd_checkpoint"])
    device = get_best_device()
    pipe = (
        StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        .to(device)
    )

    try:
        import certiflow
    except ImportError:
        raise RuntimeError("[FATAL] certiflow library missing.")

    policy = certiflow.load_policy("triple_axis_policy")

    samplers = ["euler_ode", "heun_sde", "tau_leap", "krk"]
    results: Dict[str, Dict[str, float]] = {}
    for smp in samplers:
        imgs, logs = pipe.generate(prompts, policy=policy, sampler=smp, certify=True)
        fid = statistics.mean(l["fid"] for l in logs)
        clip = statistics.mean(l["clip"] for l in logs)
        nfe = statistics.mean(l["nfe"] for l in logs)
        energy = statistics.mean(l["energy"] for l in logs)
        results[smp] = {"FID": fid, "CLIP": clip, "NFE": nfe, "Energy": energy}

    result_path = RESEARCH_DIR / "exp2_results.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    x = list(range(len(samplers)))
    energy_fig = save_lineplot(
        x,
        [[results[s]["Energy"] for s in samplers]],
        ["CERTIFLOW"],
        "Energy per sampler",
        "Sampler idx",
        "Energy (J)",
        "energy_sampler",
    )

    print("\n### EXPERIMENT-2 RESULTS ###")
    print(json.dumps(results, indent=2))
    print("Figures generated:")
    print(energy_fig)

# -----------------------------------------------------------------------------
# EXPERIMENT 3 – ISA-Agnostic Weight Fusion & Carbon Impact
# -----------------------------------------------------------------------------

def run_exp3(cfg: dict):
    print("\n===============  EXPERIMENT-3  ===============")
    print("ISA-Agnostic Weight Fusion & Carbon Impact\n")

    try:
        import certiflow
    except ImportError:
        raise RuntimeError("[FATAL] certiflow library missing.")

    variants = ["fp16", "bitsfusion2", "certiflow3_2", "certiflow3_2_lambda0"]
    devices = ["A100", "RTX4060", "M2Pro", "RISCV"]

    results: Dict[str, Dict[str, dict]] = {d: {} for d in devices}
    for dev in devices:
        for var in variants:
            stats = certiflow.run_edge_benchmark(
                device=dev,
                variant=var,
                sentences=cfg["mobile_mt_sentences"],
                minutes=cfg["video_minutes"],
            )
            results[dev][var] = stats

    result_path = RESEARCH_DIR / "exp3_results.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    x = list(range(len(devices)))
    energy_vals = [[results[d]["certiflow3_2"]["energy"] for d in devices]]
    energy_fig = save_lineplot(
        x,
        energy_vals,
        ["CERTIFLOW 3/2"],
        "Energy / sample",
        "Device idx",
        "Energy (J)",
        "energy_devices",
    )

    print("\n### EXPERIMENT-3 RESULTS ###")
    print(json.dumps(results, indent=2))
    print("Figures generated:")
    print(energy_fig)
