import gc
import os

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from beyond_mvtec import experiment as ex

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = ex.load_config(os.path.join(ROOT, "configs", "experiments.yaml"))


REAL_IAD_ALL = [
    "audiojack", "bottle_cap", "button_battery", "end_cap", "eraser", "fire_hood", "mint", "mounts",
    "pcb", "phone_battery", "plastic_nut", "plastic_plug", "porcelain_doll", "regulator",
    "rolled_strip_base", "sim_card_set", "switch", "tape", "terminalblock", "toothbrush", "toy",
    "toy_brick", "transistor1", "u_block", "usb", "usb_adaptor", "vcpill", "wooden_beads",
    "woodstick", "zipper",
]


def test_ten_comparison_settings_without_layer1_at_ten_percent():
    settings = ex.comparison_settings(CONFIG)
    names = {s["setting"] for s in settings}
    assert len(settings) == 10 and len(names) == 10
    assert sum(s["method"] == "patchcore" for s in settings) == 6
    assert not names & {"patchcore_r18_L123_c10", "patchcore_wrn50_L123_c10"}
    assert {"patchcore_r18_L123_c1", "patchcore_wrn50_L123_c1", "patchcore_wrn50_L23_c10"} <= names


def test_without_exclude_the_grid_has_twelve():
    config = {**CONFIG, "methods": {**CONFIG["methods"],
                                    "patchcore": {**CONFIG["methods"]["patchcore"], "exclude": []}}}
    assert len(ex.comparison_settings(config)) == 12


def test_real_iad_categories_are_the_seeded_draw():
    draw = CONFIG["datasets"]["real_iad"]["draw"]
    assert ex.draw_categories(REAL_IAD_ALL, draw["k"], draw["seed"]) == CONFIG["datasets"]["real_iad"]["categories"]
    assert ex.draw_categories(list(reversed(REAL_IAD_ALL)), 10, 0) == CONFIG["datasets"]["real_iad"]["categories"]


def test_reproduce_targets_match_the_readme():
    targets = {s["setting"]: s["target"] for s in ex.reproduce_settings(CONFIG)}
    assert targets["patchcore_wrn50_L23_c10"] == {"image_auroc": 99.0, "pixel_auroc": 98.1, "pro": 93.5}
    assert targets["patchcore_wrn50_L23_c1"] == {"image_auroc": 99.0, "pixel_auroc": 98.0, "pro": 93.1}
    assert targets["padim_wrn50_L123_d550"] == {"image_auroc": 95.3, "pixel_auroc": 97.5, "pro": 92.1}
    assert targets["padim_r18_L123_d100"] == {"pixel_auroc": 96.7, "pro": 90.1}


def _row(setting, category, seed, value):
    return {"dataset": "mvtec_ad", "category": category, "setting": setting, "preprocess": "reproduce",
            "n_train_requested": "all", "seed": seed, "image_auroc": value, "pixel_auroc": value,
            "pro": value}


def test_summary_averages_categories_then_seeds(tmp_path):
    rows = [_row("a", "x", 0, 0.90), _row("a", "y", 0, 1.00), _row("a", "x", 1, 0.92), _row("a", "y", 1, 1.00)]
    path = tmp_path / "r.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    out = ex.summarize(str(path)).iloc[0]
    assert out["image_auroc_mean"] == pytest.approx(95.5)
    assert out["image_auroc_std"] == pytest.approx(np.std([95.0, 96.0], ddof=1))
    assert out["n_seeds"] == 2 and out["n_categories"] == 2


def test_judge_uses_the_tolerance_and_skips_missing_targets():
    summary = pd.DataFrame([{
        "setting": "a", "image_auroc_mean": 99.4, "pixel_auroc_mean": 96.9, "pro_mean": 90.0,
        "image_auroc_std": 0.1, "pixel_auroc_std": 0.1, "pro_std": 0.1, "n_seeds": 3, "n_categories": 15,
    }])
    settings = [{"setting": "a", "target": {"image_auroc": 99.0, "pixel_auroc": 98.1}}]
    table = ex.judge_reproduction(summary, settings, {"image_auroc": 0.5, "pixel_auroc": 1.0, "pro": 1.0})
    passed = dict(zip(table["metric"], table["passed"]))
    assert passed["image_auroc"] == True  # noqa: E712
    assert passed["pixel_auroc"] == False  # noqa: E712
    assert pd.isna(passed["pro"])


def _fake_mvtec(root):
    rng = np.random.RandomState(0)
    base = os.path.join(root, "toy")
    for folder in ("train/good", "test/good", "test/hole", "ground_truth/hole"):
        os.makedirs(os.path.join(base, folder))
    def noise():
        return rng.randint(100, 140, (64, 64, 3), dtype=np.uint8)
    for i in range(6):
        Image.fromarray(noise()).save(os.path.join(base, "train/good", f"{i:03d}.png"))
    for i in range(3):
        Image.fromarray(noise()).save(os.path.join(base, "test/good", f"{i:03d}.png"))
        broken = noise()
        broken[20:40, 20:40] = 255
        Image.fromarray(broken).save(os.path.join(base, "test/hole", f"{i:03d}.png"))
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:40, 20:40] = 255
        Image.fromarray(mask).save(os.path.join(base, "ground_truth/hole", f"{i:03d}_mask.png"))


def test_grid_resumes_from_the_csv(tmp_path, monkeypatch):
    _fake_mvtec(str(tmp_path))
    setting = ex.padim_setting("resnet18", [1, 2, 3], 10, 0.01)

    def untrained(setting, output_size, device, seed, faiss_on_gpu=False):
        return ex.PaDiM(setting["backbone"], setting["layers"], setting["reduced_dim"], output_size,
                        device=device, seed=seed, eps=setting["cov_eps"], pretrained=False)

    monkeypatch.setattr(ex, "build_model", untrained)
    csv_path = str(tmp_path / "out" / "r.csv")
    options = dict(preprocess_name="p", preprocess={"resize": [64, 64], "center_crop": None},
                   csv_path=csv_path, device="cpu", batch_size=2, num_workers=0,
                   save_dir=str(tmp_path / "npz"), save_maps_seeds=(0,), log=lambda _: None)
    ex.run_grid([setting], "mvtec_ad", str(tmp_path), ["toy"], [0], **options)
    first = pd.read_csv(csv_path)
    assert len(first) == 1 and first.loc[0, "n_regions"] == 3 and first.loc[0, "n_train"] == 6

    def must_not_run(*args, **kwargs):
        raise AssertionError("끝난 줄을 다시 돌렸다")

    monkeypatch.setattr(ex, "run_one", must_not_run)
    ex.run_grid([setting], "mvtec_ad", str(tmp_path), ["toy"], [0], **options)
    assert len(pd.read_csv(csv_path)) == 1
    saved = np.load(tmp_path / "npz" / "mvtec_ad__toy__padim_r18_L123_d10__p__nall__s0.npz")
    assert saved["maps"].shape == (6, 64, 64) and saved["masks"].sum() == 3 * 20 * 20


def test_patchcore_adapter_runs_on_upstream(tmp_path):
    pytest.importorskip("faiss")
    pytest.importorskip("patchcore.patchcore")
    _fake_mvtec(str(tmp_path))
    setting = ex.patchcore_setting("resnet18", [2, 3], 0.1)
    row = ex.run_one(setting, "mvtec_ad", str(tmp_path), "toy", 0, "p",
                     {"resize": [64, 64], "center_crop": None}, device="cpu", batch_size=2, num_workers=0)
    # 8 x 8 격자 x 6장 = 조각 384개, 10% 면 38개(원본은 int 내림)
    assert row["bank_size"] == 38
    assert row["image_auroc"] == 1.0


def test_run_one_releases_faiss_index(tmp_path):
    """원본 PatchCore 의 순환 참조로 FaissNN 이 남으면 GPU 에서 자원마다 임시 메모리 1.5 GiB 가 묶인다."""
    pytest.importorskip("faiss")
    pytest.importorskip("patchcore.patchcore")
    from patchcore.common import FaissNN

    def alive():
        return sum(type(o) is FaissNN for o in gc.get_objects())

    _fake_mvtec(str(tmp_path))
    setting = ex.patchcore_setting("resnet18", [2, 3], 0.1)
    preprocess = {"resize": [64, 64], "center_crop": None}
    options = dict(device="cpu", batch_size=2, num_workers=0)
    ex.run_one(setting, "mvtec_ad", str(tmp_path), "toy", 0, "p", preprocess, **options)
    # 자동 GC 가 우연히 돌면 차이가 안 보이므로 끄고 잰다
    gc.collect()
    before = alive()
    gc.disable()
    try:
        for seed in (1, 2):
            ex.run_one(setting, "mvtec_ad", str(tmp_path), "toy", seed, "p", preprocess, **options)
        assert alive() == before
    finally:
        gc.enable()


def test_kendall_pairs_counts_flipped_pairs():
    values = pd.DataFrame({"a": [0.9, 0.8, 0.7, 0.6], "b": [0.9, 0.8, 0.7, 0.6],
                           "c": [0.8, 0.9, 0.7, 0.6], "d": [0.6, 0.7, 0.8, 0.9]},
                          index=["s1", "s2", "s3", "s4"])
    out = ex.kendall_pairs(values).set_index("pair")
    assert out.loc["a - b", "tau_b"] == pytest.approx(1.0) and out.loc["a - b", "discordant"] == 0
    # 쌍 6개 중 (s1, s2) 하나만 뒤집힘 -> (5 - 1) / 6
    assert out.loc["a - c", "discordant"] == 1 and out.loc["a - c", "tau_b"] == pytest.approx(4 / 6)
    assert out.loc["a - d", "tau_b"] == pytest.approx(-1.0) and out.loc["a - d", "n_pairs"] == 6


def _q1_csv(path, dataset, values_by_seed):
    rows = []
    for seed, values in values_by_seed.items():
        for setting, (v1, v2) in values.items():
            for category, v in (("x", v1), ("y", v2)):
                rows.append({"dataset": dataset, "category": category, "setting": setting,
                             "preprocess": "compare", "n_train_requested": "all", "seed": seed,
                             "image_auroc": v, "pixel_auroc": v, "pro": v})
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def test_rank_and_kendall_across_csv_files(tmp_path):
    a = _q1_csv(tmp_path / "a.csv", "mvtec_ad", {0: {"s1": (1.0, 0.8), "s2": (0.8, 0.8), "s3": (0.6, 0.6)},
                                                1: {"s1": (1.0, 1.0), "s2": (0.8, 0.6), "s3": (0.6, 0.6)}})
    b = _q1_csv(tmp_path / "b.csv", "visa", {0: {"s1": (0.5, 0.5), "s2": (0.7, 0.7), "s3": (0.4, 0.4)},
                                             1: {"s1": (0.5, 0.5), "s2": (0.7, 0.7), "s3": (0.4, 0.4)}})
    values, ranks = ex.rank_by_dataset([a, b])
    assert values.loc["s1", "mvtec_ad"] == pytest.approx(95.0)  # 시드 평균 (90 + 100) / 2
    assert ranks["mvtec_ad"].tolist() == [1.0, 2.0, 3.0] and ranks["visa"].tolist() == [2.0, 1.0, 3.0]
    per_seed = ex.kendall_per_seed([a, b])
    assert sorted(per_seed["seed"]) == [0, 1]
    assert (per_seed["discordant"] == 1).all()


def test_releasing_sampler_picks_the_same_coreset():
    """감싼 샘플러가 원본과 같은 조각을 고르고, 사영 결과에 기울기 그래프가 붙지 않는다."""
    pytest.importorskip("patchcore.sampler")
    import torch
    import patchcore.sampler
    from beyond_mvtec.patchcore_adapter import _releasing_sampler

    features = np.random.RandomState(0).randn(500, 64).astype(np.float32)
    picked = []
    for cls in (patchcore.sampler.ApproximateGreedyCoresetSampler,
                _releasing_sampler(patchcore.sampler.ApproximateGreedyCoresetSampler)):
        torch.manual_seed(0)
        np.random.seed(0)
        picked.append(cls(0.1, torch.device("cpu"), dimension_to_project_features_to=16).run(features))
    assert picked[0].shape == (50, 64)
    np.testing.assert_array_equal(picked[0], picked[1])

    torch.manual_seed(0)
    reduced = _releasing_sampler(patchcore.sampler.ApproximateGreedyCoresetSampler)(
        0.1, torch.device("cpu"), dimension_to_project_features_to=16)._reduce_features(torch.from_numpy(features))
    assert reduced.grad_fn is None and not reduced.requires_grad
