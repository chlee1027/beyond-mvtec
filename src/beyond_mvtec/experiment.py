"""설정 목록, 한 부류 실행, 끊겨도 이어 가는 격자 실행, 요약과 재현 판정."""
import csv
import gc
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import yaml

from . import data as data_lib
from . import metrics
from .padim import PaDiM
from .patchcore_adapter import PatchCoreAdapter

SHORT = {"resnet18": "r18", "wide_resnet50_2": "wrn50"}
COLUMNS = [
    "dataset", "category", "setting", "method", "backbone", "layers", "coreset_ratio",
    "reduced_dim", "cov_eps", "preprocess", "n_train_requested", "n_train", "seed", "n_test",
    "n_regions", "image_auroc", "pixel_auroc", "pro", "bank_size", "fit_sec", "predict_sec",
    "metric_sec", "peak_gpu_gb", "commit",
]
KEY = ["dataset", "category", "setting", "preprocess", "n_train_requested", "seed"]
METRICS = ["image_auroc", "pixel_auroc", "pro"]


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _layer_tag(layers):
    return "L" + "".join(str(i) for i in layers)


def patchcore_setting(backbone, layers, coreset_ratio):
    return {
        "setting": f"patchcore_{SHORT[backbone]}_{_layer_tag(layers)}_c{round(coreset_ratio * 100)}",
        "method": "patchcore", "backbone": backbone, "layers": list(layers),
        "coreset_ratio": coreset_ratio,
    }


def padim_setting(backbone, layers, reduced_dim, cov_eps):
    return {
        "setting": f"padim_{SHORT[backbone]}_{_layer_tag(layers)}_d{reduced_dim}",
        "method": "padim", "backbone": backbone, "layers": list(layers),
        "reduced_dim": reduced_dim, "cov_eps": cov_eps,
    }


def comparison_settings(config):
    """Q1 의 설정 12개."""
    pc, pa = config["methods"]["patchcore"], config["methods"]["padim"]
    out = [patchcore_setting(b, l, c) for b in pc["backbones"] for l in pc["layers"]
           for c in pc["coreset_ratio"]]
    out += [padim_setting(b, l, pa["reduced_dims"][b], pa["cov_eps"])
            for b in pa["backbones"] for l in pa["layers"]]
    return out


def reproduce_settings(config):
    pa = config["methods"]["padim"]
    out = []
    for item in config["reproduce"]["settings"]:
        if item["method"] == "patchcore":
            setting = patchcore_setting(item["backbone"], item["layers"], item["coreset_ratio"])
        else:
            setting = padim_setting(item["backbone"], item["layers"],
                                    pa["reduced_dims"][item["backbone"]], pa["cov_eps"])
        out.append({**setting, "target": item["target"]})
    return out


def fix_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_split(dataset, root, category, view="C1", json_root=None):
    if dataset == "mvtec_ad":
        return data_lib.list_mvtec(root, category)
    if dataset == "visa":
        return data_lib.list_visa(root, category)
    if dataset == "real_iad":
        return data_lib.list_real_iad(root, category, view=view, json_root=json_root)
    raise ValueError(dataset)


def build_model(setting, output_size, device, seed, faiss_on_gpu=False):
    if setting["method"] == "padim":
        fix_seeds(seed)
        return PaDiM(setting["backbone"], setting["layers"], setting["reduced_dim"], output_size,
                     device=device, seed=seed, eps=setting["cov_eps"])
    return PatchCoreAdapter(setting["backbone"], setting["layers"], setting["coreset_ratio"],
                            output_size, device, seed=seed, faiss_on_gpu=faiss_on_gpu)


def predict_all(model, loader, output_size):
    n = len(loader.dataset)
    maps = np.zeros((n, *output_size), dtype=np.float32)
    masks = np.zeros((n, *output_size), dtype=bool)
    scores, labels = np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.int64)
    i = 0
    for batch in loader:
        batch_scores, batch_maps = model.predict_batch(batch["image"])
        j = i + len(batch_scores)
        scores[i:j], maps[i:j] = batch_scores, batch_maps
        masks[i:j], labels[i:j] = batch["mask"].numpy(), batch["is_anomaly"].numpy()
        i = j
    return scores, maps, labels, masks


def run_one(setting, dataset, root, category, seed, preprocess_name, preprocess, n_train="all",
            device="cuda", batch_size=8, num_workers=4, faiss_on_gpu=False, save_dir=None,
            save_maps=False, commit="", view="C1", json_root=None):
    """부류 하나를 학습하고 채점해 CSV 한 줄을 돌려준다."""
    split = list_split(dataset, root, category, view, json_root)
    train = data_lib.subsample_train(split.train, n_train, seed)
    train_loader, test_loader, output_size = data_lib.make_loaders(
        train, split.test, preprocess, batch_size, num_workers)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model = build_model(setting, output_size, device, seed, faiss_on_gpu)
    start = time.time()
    model.fit(train_loader)
    fit_sec = time.time() - start

    start = time.time()
    scores, maps, labels, masks = predict_all(model, test_loader, output_size)
    predict_sec = time.time() - start

    start = time.time()
    result = metrics.evaluate(labels, scores, maps, masks)
    metric_sec = time.time() - start

    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
    row = {
        "dataset": dataset, "category": category, "setting": setting["setting"],
        "method": setting["method"], "backbone": setting["backbone"],
        "layers": "-".join(str(i) for i in setting["layers"]),
        "coreset_ratio": setting.get("coreset_ratio", ""),
        "reduced_dim": setting.get("reduced_dim", ""), "cov_eps": setting.get("cov_eps", ""),
        "preprocess": preprocess_name, "n_train_requested": str(n_train), "n_train": len(train),
        "seed": seed, "n_test": len(split.test), **result,
        "bank_size": getattr(model, "bank_size", ""), "fit_sec": round(fit_sec, 2),
        "predict_sec": round(predict_sec, 2), "metric_sec": round(metric_sec, 2),
        "peak_gpu_gb": "" if peak is None else round(peak, 2), "commit": commit,
    }
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        stem = f"{dataset}__{category}__{setting['setting']}__{preprocess_name}__n{n_train}__s{seed}"
        arrays = {"scores": scores, "labels": labels}
        if save_maps:  # PRO 정의를 바꿔 다시 잴 수 있게 지도를 남긴다
            arrays.update(maps=maps.astype(np.float16), masks=masks)
        np.savez_compressed(os.path.join(save_dir, stem + ".npz"), **arrays)
    del model
    # 원본 PatchCore 의 NearestNeighbourScorer 는 순환 참조라 del 만으로 FAISS 인덱스가 풀리지 않는다.
    # GPU 에서는 인덱스의 StandardGpuResources 하나가 임시 메모리 1.5 GiB 를 잡는다(faiss 소스).
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def _done_keys(csv_path):
    if not os.path.isfile(csv_path):
        return set()
    done = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    return set(map(tuple, done[KEY].values.tolist()))


def run_grid(settings, dataset, root, categories, seeds, preprocess_name, preprocess, csv_path,
             n_train="all", log=print, save_maps_seeds=(), **options):
    """끝난 줄은 CSV 를 보고 건너뛴다. 런타임이 끊겨도 같은 셀을 다시 돌리면 이어 간다."""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    done = _done_keys(csv_path)
    jobs = [(s, c, seed) for s in settings for seed in seeds for c in categories]
    for count, (setting, category, seed) in enumerate(jobs, 1):
        key = (dataset, category, setting["setting"], preprocess_name, str(n_train), str(seed))
        if key in done:
            continue
        row = run_one(setting, dataset, root, category, seed, preprocess_name, preprocess,
                      n_train=n_train, save_maps=seed in save_maps_seeds, **options)
        is_new = not os.path.isfile(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, COLUMNS)
            if is_new:
                writer.writeheader()
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())
        log(f"[{count}/{len(jobs)}] {setting['setting']} {category} s{seed} "
            f"I={row['image_auroc']:.4f} P={row['pixel_auroc']:.4f} PRO={row['pro']:.4f} "
            f"fit={row['fit_sec']}s pred={row['predict_sec']}s gpu={row['peak_gpu_gb']}GB")


def summarize(csv_path):
    """시드마다 부류 평균을 내고, 그 값의 시드 간 평균과 표준편차를 낸다. 단위는 %."""
    df = pd.read_csv(csv_path)
    group = ["dataset", "setting", "preprocess", "n_train_requested"]
    per_seed = df.groupby(group + ["seed"]).agg(
        **{m: (m, "mean") for m in METRICS}, n_categories=("category", "nunique")).reset_index()
    out = per_seed.groupby(group).agg(
        **{f"{m}_mean": (m, "mean") for m in METRICS},
        **{f"{m}_std": (m, lambda v: v.std(ddof=1) if len(v) > 1 else float("nan")) for m in METRICS},
        n_seeds=("seed", "nunique"), n_categories=("n_categories", "min")).reset_index()
    for m in METRICS:
        out[f"{m}_mean"] *= 100
        out[f"{m}_std"] *= 100
    return out


def judge_reproduction(summary, settings, tolerance):
    """README 의 판정 폭으로 재현 여부를 가른다. 목표가 없는 칸은 판정하지 않는다."""
    rows = []
    for setting in settings:
        hit = summary[summary["setting"] == setting["setting"]]
        for metric in METRICS:
            target = setting["target"].get(metric)
            value = float(hit[f"{metric}_mean"].iloc[0]) if len(hit) else float("nan")
            std = float(hit[f"{metric}_std"].iloc[0]) if len(hit) else float("nan")
            passed = None
            if target is not None and not np.isnan(value):
                passed = bool(abs(value - target) <= tolerance[metric])
            rows.append({
                "setting": setting["setting"], "metric": metric, "target": target,
                "value": round(value, 2), "seed_std": round(std, 3),
                "diff": None if target is None else round(value - target, 2),
                "tolerance": tolerance[metric], "passed": passed,
                "n_seeds": int(hit["n_seeds"].iloc[0]) if len(hit) else 0,
                "n_categories": int(hit["n_categories"].iloc[0]) if len(hit) else 0,
            })
    return pd.DataFrame(rows)
