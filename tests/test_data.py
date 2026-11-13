import json
import os
import zipfile

import numpy as np
import pytest
import torch
from PIL import Image

from beyond_mvtec import data


def _image(path, size=(32, 32), mode="RGB", value=128):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    color = value if mode in ("L", "P") else (value, value, value)
    Image.new(mode, size, color).save(path)


def _mask(path, size=(32, 32), box=(8, 8, 16, 16), value=255, mode="L"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    m = Image.new(mode, size, 0)
    m.paste(value, box)
    m.save(path)


@pytest.fixture
def mvtec(tmp_path):
    base = tmp_path / "bottle"
    for i in range(5):
        _image(str(base / "train" / "good" / f"{i:03d}.png"))
    _image(str(base / "test" / "good" / "000.png"))
    _image(str(base / "test" / "crack" / "000.png"))
    _mask(str(base / "ground_truth" / "crack" / "000_mask.png"))
    return str(tmp_path)


def test_mvtec_pairs_image_and_mask(mvtec):
    split = data.list_mvtec(mvtec, "bottle")
    assert len(split.train) == 5
    by_label = {s.label: s for s in split.test}
    assert by_label[0].mask is None
    assert by_label[1].mask.endswith("000_mask.png")


def test_mvtec_missing_mask_raises(mvtec):
    os.remove(os.path.join(mvtec, "bottle", "ground_truth", "crack", "000_mask.png"))
    with pytest.raises(FileNotFoundError):
        data.list_mvtec(mvtec, "bottle")


@pytest.fixture
def visa(tmp_path):
    rows = ["object,split,label,image,mask"]
    for i in range(3):
        rel = f"candle/Data/Images/Normal/{i:04d}.JPG"
        _image(str(tmp_path / rel), size=(64, 32))
        rows.append(f"candle,train,normal,{rel},")
    rel = "candle/Data/Images/Normal/0003.JPG"
    _image(str(tmp_path / rel), size=(64, 32))
    rows.append(f"candle,test,normal,{rel},")
    rel, mask = "candle/Data/Images/Anomaly/000.JPG", "candle/Data/Masks/Anomaly/000.png"
    _image(str(tmp_path / rel), size=(64, 32))
    _mask(str(tmp_path / mask), size=(64, 32), value=1)  # VisA 마스크는 작은 색인값이다
    rows.append(f"candle,test,anomaly,{rel},{mask}")
    os.makedirs(tmp_path / "split_csv")
    (tmp_path / "split_csv" / "1cls.csv").write_text("\n".join(rows) + "\n")
    return str(tmp_path)


def test_visa_follows_the_csv(visa):
    split = data.list_visa(visa, "candle")
    assert len(split.train) == 3
    assert sorted(s.label for s in split.test) == [0, 1]
    assert data.visa_categories(visa) == ["candle"]


def test_small_index_mask_is_still_a_defect(visa):
    split = data.list_visa(visa, "candle")
    image_tf, mask_tf, size = data.build_transforms({"resize": [32, 32], "center_crop": None})
    item = data.TestDataset(split.test, image_tf, mask_tf, size)[[s.label for s in split.test].index(1)]
    assert item["mask"].dtype == torch.bool and item["mask"].sum() > 0


@pytest.fixture
def real_iad(tmp_path):
    cat = tmp_path / "pcb"
    train, test = [], []
    for view in ("C1", "C2"):
        rel = f"OK/S0001/pcb_0001_OK_{view}_1.jpg"
        _image(str(cat / rel))
        train.append({"category": "OK", "anomaly_class": "OK", "image_path": rel})
        # 이상 표본인데 이 시점에는 결함이 안 보인다 -> 이미지 라벨은 정상
        rel = f"NG/AK/S0002/pcb_0002_AK_{view}_2.jpg"
        _image(str(cat / rel))
        test.append({"category": "NG", "anomaly_class": "OK", "image_path": rel})
        rel = f"NG/AK/S0003/pcb_0003_AK_{view}_3.jpg"
        _image(str(cat / rel))
        mask = rel.replace(".jpg", ".png")
        _mask(str(cat / mask))
        test.append({"category": "NG", "anomaly_class": "AK", "image_path": rel, "mask_path": mask})
    meta = {"meta": {"normal_class": "OK"}, "train": train, "test": test}
    # 단일 시점 분할: C1 만 있고 학습·시험을 다중 시점 분할과 다르게 나눈다(정상 둘을 학습에)
    c1 = lambda entries: [e for e in entries if "_C1_" in e["image_path"]]
    sv = {"meta": {"normal_class": "OK"},
          "train": c1(train) + [e for e in c1(test) if e["anomaly_class"] == "OK"],
          "test": [e for e in c1(test) if e["anomaly_class"] != "OK"]}
    for folder, payload in (("realiad_jsons", meta), ("realiad_jsons_sv", sv),
                            ("realiad_jsons_fuiad_0.0", {"meta": {}, "train": [], "test": []})):
        os.makedirs(tmp_path / "realiad_jsons" / folder)
        (tmp_path / "realiad_jsons" / folder / "pcb.json").write_text(json.dumps(payload))
    return str(tmp_path)


def test_real_iad_uses_single_view_split_by_default(real_iad):
    split = data.list_real_iad(real_iad, "pcb", view="C1")
    assert len(split.train) == 2 and [s.label for s in split.test] == [1]


def test_real_iad_single_view_split_rejects_other_views(real_iad):
    path = os.path.join(real_iad, "realiad_jsons", "realiad_jsons_sv", "pcb.json")
    meta = json.load(open(path))
    meta["train"].append({"anomaly_class": "OK", "image_path": "OK/S0001/pcb_0001_OK_C2_1.jpg"})
    json.dump(meta, open(path, "w"))
    with pytest.raises(ValueError, match="C1 아닌 항목 1개"):
        data.list_real_iad(real_iad, "pcb", view="C1")


def test_real_iad_filters_one_view_and_uses_image_labels(real_iad):
    split = data.list_real_iad(real_iad, "pcb", view="C1", split="multi")
    assert [os.path.basename(p) for p in split.train] == ["pcb_0001_OK_C1_1.jpg"]
    assert sorted(s.label for s in split.test) == [0, 1]
    normal = next(s for s in split.test if s.label == 0)
    assert "NG" in normal.image and normal.mask is None
    assert next(s for s in split.test if s.label == 1).mask.endswith("pcb_0003_AK_C1_3.png")


def test_real_iad_falls_back_to_sibling_mask(real_iad):
    path = os.path.join(real_iad, "realiad_jsons", "realiad_jsons", "pcb.json")
    meta = json.load(open(path))
    for entry in meta["test"]:
        entry.pop("mask_path", None)
    json.dump(meta, open(path, "w"))
    split = data.list_real_iad(real_iad, "pcb", view="C1", split="multi")
    assert next(s for s in split.test if s.label == 1).mask.endswith(".png")


def test_real_iad_missing_file_raises(real_iad):
    os.remove(os.path.join(real_iad, "pcb", "OK", "S0001", "pcb_0001_OK_C1_1.jpg"))
    with pytest.raises(FileNotFoundError):
        data.list_real_iad(real_iad, "pcb", view="C1")


def test_prepare_extracts_one_view_only(tmp_path):
    zip_path = tmp_path / "pcb.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("pcb/OK/S1/pcb_1_OK_C1_1.jpg", b"a")
        z.writestr("pcb/OK/S1/pcb_1_OK_C2_1.jpg", b"b")
    dest = data.prepare_real_iad(str(zip_path), str(tmp_path / "out"), "pcb", view="C1")
    found = [f for _, _, files in os.walk(dest) for f in files if f.endswith(".jpg")]
    assert found == ["pcb_1_OK_C1_1.jpg"]


def test_subsample_is_seeded_and_not_a_prefix():
    paths = [f"{i:03d}.png" for i in range(100)]
    a, b = data.subsample_train(paths, 10, 0), data.subsample_train(paths, 10, 1)
    assert a == data.subsample_train(paths, 10, 0)
    assert a != b and a != paths[:10] and len(set(a)) == 10
    assert data.subsample_train(paths, "all", 0) == paths
    assert data.subsample_train(paths, 500, 0) == paths


def test_reproduce_transform_crops_to_224(tmp_path):
    image_tf, _, size = data.build_transforms({"resize": 256, "center_crop": 224})
    path = str(tmp_path / "a.png")
    _image(path, size=(700, 700))
    assert size == (224, 224)
    assert image_tf(Image.open(path).convert("RGB")).shape == (3, 224, 224)


def test_compare_transform_keeps_the_long_edge(tmp_path):
    # 정사각이 아닌 이미지 오른쪽 끝의 표시가 줄인 뒤에도 남아야 한다
    image = np.zeros((300, 400), dtype=np.uint8)
    image[:, -20:] = 255
    image_tf, mask_tf, size = data.build_transforms({"resize": [256, 256], "center_crop": None})
    assert size == (256, 256)
    out = np.asarray(mask_tf(Image.fromarray(image)))
    assert out.shape == (256, 256) and out[:, -1].min() == 255


def test_int_resize_without_crop_is_rejected():
    with pytest.raises(ValueError):
        data.build_transforms({"resize": 256, "center_crop": None})


def test_grayscale_becomes_three_channels(tmp_path):
    path = str(tmp_path / "g.png")
    _image(path, mode="L")
    image_tf, _, _ = data.build_transforms({"resize": [32, 32], "center_crop": None})
    item = data.TrainDataset([path], image_tf)[0]
    assert item["image"].shape == (3, 32, 32)


def test_mask_stays_binary_after_shrinking(tmp_path):
    path = str(tmp_path / "m.png")
    _mask(path, size=(700, 700), box=(100, 100, 400, 400))
    _, mask_tf, _ = data.build_transforms({"resize": [256, 256], "center_crop": None})
    assert set(np.unique(np.asarray(mask_tf(Image.open(path))))) == {0, 255}


@pytest.fixture
def mvtec2(tmp_path):
    rng = np.random.RandomState(0)

    def noise(path, size, mode):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        shape = (size[1], size[0]) if mode == "L" else (size[1], size[0], 3)
        Image.fromarray(rng.randint(0, 256, shape).astype(np.uint8), mode).save(path)

    src = tmp_path / "mvtec2"
    for cat, mode in [("can", "RGB"), ("vial", "L")]:
        base = src / cat
        for i in range(3):
            noise(str(base / "train" / "good" / f"{i:03d}_regular.png"), (300, 140), mode)
        for cond in ["regular", "overexposed"]:
            noise(str(base / "test_public" / "good" / f"000_{cond}.png"), (300, 140), mode)
            noise(str(base / "test_public" / "bad" / f"000_{cond}.png"), (300, 140), mode)
            _mask(str(base / "test_public" / "ground_truth" / "bad" / f"000_{cond}_mask.png"),
                  size=(300, 140), box=(100, 40, 103, 43))
        noise(str(base / "validation" / "good" / "000_regular.png"), (300, 140), mode)
        noise(str(base / "test_private" / "000_regular.png"), (300, 140), mode)
    return str(src)


def test_mvtec2_lists_test_public_with_conditions(mvtec2):
    split = data.list_mvtec2(mvtec2, "can")
    assert len(split.train) == 3
    assert sorted((s.label, s.anomaly) for s in split.test) == [
        (0, "overexposed"), (0, "regular"), (1, "overexposed"), (1, "regular")]
    assert all(s.mask.endswith(f"000_{s.anomaly}_mask.png") for s in split.test if s.label)
    assert all("validation" not in p and "private" not in p for p in split.train)


def test_mvtec2_condition_keeps_underscores():
    assert data.mvtec2_condition("/x/012_shift_3.png") == "shift_3"
    with pytest.raises(ValueError):
        data.mvtec2_condition("/x/regular.png")


def test_mvtec2_missing_mask_raises(mvtec2):
    os.remove(os.path.join(mvtec2, "can", "test_public", "ground_truth", "bad", "000_regular_mask.png"))
    with pytest.raises(FileNotFoundError):
        data.list_mvtec2(mvtec2, "can")


@pytest.mark.parametrize("category", ["can", "vial"])
def test_mvtec2_cache_gives_the_same_tensors(mvtec2, tmp_path, category):
    """줄여 둔 파일을 읽어도 원본을 읽은 것과 텐서가 같다(RGB 와 흑백 부류 둘 다)."""
    size = (64, 64)
    cache = str(tmp_path / "cache")
    assert data.cache_mvtec2(mvtec2, cache, category, size, workers=2, log=lambda *_: None) == 9
    orig, cached = data.list_mvtec2(mvtec2, category), data.list_mvtec2(cache, category)
    assert [os.path.basename(p) for p in orig.train] == [os.path.basename(p) for p in cached.train]
    assert [(s.label, s.anomaly) for s in orig.test] == [(s.label, s.anomaly) for s in cached.test]
    image_tf, mask_tf, out = data.build_transforms({"resize": list(size), "center_crop": None})
    for a, b in zip(orig.train, cached.train):
        assert torch.equal(data.TrainDataset([a], image_tf)[0]["image"], data.TrainDataset([b], image_tf)[0]["image"])
    one, two = data.TestDataset(orig.test, image_tf, mask_tf, out), data.TestDataset(cached.test, image_tf, mask_tf, out)
    for i in range(len(one)):
        assert torch.equal(one[i]["image"], two[i]["image"])
        assert torch.equal(one[i]["mask"], two[i]["mask"])


def test_mvtec2_cache_resumes_and_marks_done(mvtec2, tmp_path):
    cache = str(tmp_path / "cache")
    first = os.path.join(cache, "can", "train", "good", "000_regular.png")
    data.cache_mvtec2(mvtec2, cache, "can", (64, 64), workers=2, log=lambda *_: None)
    os.remove(os.path.join(cache, "can", ".cached_64x64"))
    os.remove(first)
    # 끊긴 자리: 표시 파일이 없고 한 장이 빠졌으면 그 한 장만 다시 쓴다
    assert data.cache_mvtec2(mvtec2, cache, "can", (64, 64), workers=2, log=lambda *_: None) == 1
    assert os.path.isfile(first)
    assert data.cache_mvtec2(mvtec2, cache, "can", (64, 64), workers=2, log=lambda *_: None) == 0
    assert not any(f.endswith(".tmp") for _, _, fs in os.walk(cache) for f in fs)
