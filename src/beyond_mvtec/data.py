"""세 데이터셋의 파일 목록과 전처리. 원본 PatchCore 의 데이터 읽기를 대신한다.

원본은 MVTec 만 읽고, `Resize(정수)` 라 정사각이 아니면 긴 변이 잘리고, 마스크를 양선형으로 줄이고,
적은 장수를 정렬된 파일 앞에서부터 자른다. 이 파일이 그 네 곳을 대신한다.
"""
import csv
import json
import os
import re
import zipfile
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp")

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
    "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]
_VIEW = re.compile(r"_(C\d)_")


@dataclass(frozen=True)
class Sample:
    image: str
    label: int
    mask: Optional[str]
    anomaly: str


@dataclass
class Split:
    train: List[str]
    test: List[Sample]


def _images_in(folder):
    return sorted(
        os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXT)
    )


def list_mvtec(root, category):
    base = os.path.join(root, category)
    train = _images_in(os.path.join(base, "train", "good"))
    test = []
    test_dir = os.path.join(base, "test")
    for anomaly in sorted(os.listdir(test_dir)):
        if not os.path.isdir(os.path.join(test_dir, anomaly)):
            continue
        for path in _images_in(os.path.join(test_dir, anomaly)):
            if anomaly == "good":
                test.append(Sample(path, 0, None, anomaly))
                continue
            stem = os.path.splitext(os.path.basename(path))[0]
            mask = os.path.join(base, "ground_truth", anomaly, f"{stem}_mask.png")
            if not os.path.isfile(mask):
                raise FileNotFoundError(f"마스크가 없다: {mask}")
            test.append(Sample(path, 1, mask, anomaly))
    return Split(train, test)


MVTEC2_CATEGORIES = ["can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts"]
_MVTEC2_NAME = re.compile(r"^\d+_(.+)$")
# 줄여 둘 갈래. validation 은 PatchCore · PaDiM 이 쓰지 않고, test_private(_mixed) 는 정답이 없다
MVTEC2_CACHE_DIRS = [("train/good", "image"), ("test_public/good", "image"),
                     ("test_public/bad", "image"), ("test_public/ground_truth/bad", "mask")]


def mvtec2_condition(path):
    """파일 이름 `번호_조건.png` 의 조건: regular, overexposed, underexposed, shift_1 ...

    학습은 regular 만 있고, test_public 은 조건이 여럿이다(mvtec2_check2 노트북에서 셈).
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    match = _MVTEC2_NAME.match(stem)
    if not match:
        raise ValueError(f"이름이 `번호_조건` 이 아니다: {path}")
    return match.group(1)


def list_mvtec2(root, category):
    """MVTec AD 2. 정답이 공개된 test_public 으로만 채점한다.

    결함 폴더가 종류별로 나뉘지 않고 `bad` 하나라서, `anomaly` 칸에는 결함 종류 대신 조명 조건을 넣는다.
    """
    base = os.path.join(root, category)
    train = _images_in(os.path.join(base, "train", "good"))
    public = os.path.join(base, "test_public")
    test = [Sample(p, 0, None, mvtec2_condition(p)) for p in _images_in(os.path.join(public, "good"))]
    for path in _images_in(os.path.join(public, "bad")):
        stem = os.path.splitext(os.path.basename(path))[0]
        mask = os.path.join(public, "ground_truth", "bad", f"{stem}_mask.png")
        if not os.path.isfile(mask):
            raise FileNotFoundError(f"마스크가 없다: {mask}")
        test.append(Sample(path, 1, mask, mvtec2_condition(path)))
    return Split(train, test)


def cache_mvtec2(src_root, dst_root, category, size, workers=8, log=print):
    """원본(1400~4224 픽셀)을 비교 단계 크기로 한 번 줄여 같은 폴더 구조로 저장한다.

    이미지는 RGB 로 바꾼 뒤 `T.Resize(size)`, 마스크는 최근접으로 줄인다. TrainDataset · TestDataset 과
    같은 단계다. 실행 때 같은 크기로 다시 줄이는 단계는 Pillow 가 그대로 복사하므로, 원본을 읽은 것과
    값이 같다(test_mvtec2_cache_gives_the_same_tensors).
    파일마다 임시 이름으로 쓴 뒤 바꿔서 끊겨도 반쯤 쓴 파일이 남지 않는다. 다시 돌리면 있는 파일은
    건너뛰고, 부류가 다 끝나면 표시 파일을 남긴다. 돌려주는 값은 이번에 새로 쓴 장수다.
    """
    from concurrent.futures import ThreadPoolExecutor

    size = tuple(int(s) for s in size)
    marker = os.path.join(dst_root, category, f".cached_{size[0]}x{size[1]}")
    if os.path.isfile(marker):
        return 0
    image_resize = T.Resize(size)
    mask_resize = T.Resize(size, interpolation=T.InterpolationMode.NEAREST)
    jobs = []
    for sub, kind in MVTEC2_CACHE_DIRS:
        src_dir = os.path.join(src_root, category, *sub.split("/"))
        dst_dir = os.path.join(dst_root, category, *sub.split("/"))
        os.makedirs(dst_dir, exist_ok=True)
        for src in _images_in(src_dir):
            dst = os.path.join(dst_dir, os.path.basename(src))
            if not os.path.isfile(dst):
                jobs.append((src, dst, kind))

    def one(job):
        src, dst, kind = job
        with Image.open(src) as im:
            out = image_resize(im.convert("RGB")) if kind == "image" else mask_resize(im)
        tmp = dst + ".tmp"
        out.save(tmp, format="PNG")
        os.replace(tmp, dst)

    with ThreadPoolExecutor(workers) as pool:
        for i, _ in enumerate(pool.map(one, jobs), 1):
            if i % 200 == 0 or i == len(jobs):
                log(f"{category}: {i}/{len(jobs)}")
    open(marker, "w").close()
    return len(jobs)


def _visa_rows(root):
    path = os.path.join(root, "split_csv", "1cls.csv")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def visa_categories(root):
    return sorted({row["object"] for row in _visa_rows(root)})


def list_visa(root, category):
    """공식 1cls 분할 CSV 를 그대로 따른다. 열: object, split, label, image, mask."""
    train, test = [], []
    for row in _visa_rows(root):
        if row["object"] != category:
            continue
        image = os.path.join(root, row["image"])
        if row["split"] == "train":
            if row["label"] != "normal":
                raise ValueError(f"학습 분할에 이상 이미지가 있다: {row['image']}")
            train.append(image)
        elif row["label"] == "normal":
            test.append(Sample(image, 0, None, "normal"))
        else:
            test.append(Sample(image, 1, os.path.join(root, row["mask"]), "anomaly"))
    if not train:
        raise ValueError(f"CSV 에 없는 부류: {category}")
    return Split(sorted(train), test)


REAL_IAD_SPLITS = {"sv": "realiad_jsons_sv", "multi": "realiad_jsons"}


def _find_real_iad_json(json_root, category, split="sv"):
    """분할 JSON 을 폴더 이름으로 고른다.

    `realiad_jsons_sv` 는 단일 시점 분할이다(anomalib 문서: 「Single-view metadata」).
    `realiad_jsons` 는 다중 시점 분할이고, 같은 C1 사진을 학습·시험에 다르게 나눈다.
    """
    if split not in REAL_IAD_SPLITS:
        raise ValueError(f"split 은 {sorted(REAL_IAD_SPLITS)} 중 하나: {split}")
    want = REAL_IAD_SPLITS[split]
    for folder, _, files in os.walk(json_root):
        if os.path.basename(folder).lower() == want and f"{category}.json" in files:
            return os.path.join(folder, f"{category}.json")
    raise FileNotFoundError(f"{json_root} 아래 {want}/ 에 {category}.json 이 없다")


def _real_iad_is_normal(entry, normal_class):
    for key in ("anomaly_class", "specie_name"):
        if key in entry:
            return str(entry[key]).strip().upper() == normal_class
    raise ValueError(f"이미지 라벨 항목을 찾지 못했다. 항목의 키: {sorted(entry)}")


def view_of(path):
    match = _VIEW.search(os.path.basename(path))
    return match.group(1) if match else None


def list_real_iad(root, category, view="C1", json_root=None, split="sv"):
    """JSON 의 이미지 라벨로 채점한다. 이상 표본 안에서 결함이 안 보이는 시점은 정상이다.

    단일 시점 분할(sv)에 다른 시점의 항목이 있으면 오류를 낸다. 걸러 버리면 분할이 조용히 줄어든다.
    """
    json_root = json_root or os.path.join(root, "realiad_jsons")
    with open(_find_real_iad_json(json_root, category, split), encoding="utf-8") as f:
        meta = json.load(f)
    if split == "sv":
        other = [e["image_path"] for e in meta["train"] + meta["test"] if view_of(e["image_path"]) != view]
        if other:
            raise ValueError(f"{category}: 단일 시점 분할에 {view} 아닌 항목 {len(other)}개. 예: {other[:3]}")
    normal_class = str(meta.get("meta", {}).get("normal_class", "OK")).upper()

    by_name = {}
    for folder, _, files in os.walk(os.path.join(root, category)):
        for name in files:
            by_name.setdefault(name, os.path.join(folder, name))

    def resolve(raw):
        return by_name.get(os.path.basename(str(raw).replace("\\", "/")))

    train, test, missing = [], [], []
    for entry in meta["train"]:
        if view_of(entry["image_path"]) != view:
            continue
        if not _real_iad_is_normal(entry, normal_class):
            raise ValueError(f"학습 분할에 이상 이미지가 있다: {entry['image_path']}")
        path = resolve(entry["image_path"])
        if path:
            train.append(path)
        else:
            missing.append(entry["image_path"])
    for entry in meta["test"]:
        if view_of(entry["image_path"]) != view:
            continue
        path = resolve(entry["image_path"])
        if not path:
            missing.append(entry["image_path"])
            continue
        if _real_iad_is_normal(entry, normal_class):
            test.append(Sample(path, 0, None, "OK"))
            continue
        raw_mask = entry.get("mask_path") or os.path.splitext(entry["image_path"])[0] + ".png"
        mask = resolve(raw_mask)
        if not mask:
            missing.append(raw_mask)
            continue
        test.append(Sample(path, 1, mask, str(entry.get("anomaly_class", "NG"))))
    if missing:
        raise FileNotFoundError(f"{category}: JSON 에 있는 파일 {len(missing)}개가 없다. 예: {missing[:3]}")
    return Split(sorted(train), test)


def prepare_real_iad(zip_path, dest_root, category, view="C1"):
    """부류 zip 에서 한 시점의 이미지와 마스크만 푼다. 이미 풀었으면 건너뛴다."""
    dest = os.path.join(dest_root, category)
    marker = os.path.join(dest, f".extracted_{view}")
    if os.path.isfile(marker):
        return dest
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest, [n for n in z.namelist() if view_of(n) == view])
    open(marker, "w").close()
    return dest


def subsample_train(paths, n_train, seed):
    """Q2 용 표본 고르기. 시드마다 중복 없이 뽑는다. 원본의 앞에서부터 자르기를 대신한다."""
    if n_train in (None, "all") or int(n_train) >= len(paths):
        return list(paths)
    rng = np.random.RandomState(seed)
    return sorted(paths[i] for i in rng.choice(len(paths), int(n_train), replace=False))


def build_transforms(preprocess):
    """preprocess: {resize: 256 또는 [256, 256], center_crop: 224 또는 None}.

    resize 가 정수면 짧은 변만 맞추고, 목록이면 두 변을 다 맞춘다.
    """
    size = preprocess["resize"]
    size = tuple(size) if isinstance(size, (list, tuple)) else int(size)
    crop = preprocess.get("center_crop")
    if crop:
        output_size = (int(crop), int(crop))
    elif isinstance(size, tuple):
        output_size = size
    else:
        raise ValueError("자르지 않으려면 resize 를 [높이, 너비] 로 적는다")

    crop_step = [T.CenterCrop(int(crop))] if crop else []
    image_tf = T.Compose(
        [T.Resize(size), *crop_step, T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )
    mask_tf = T.Compose([T.Resize(size, interpolation=T.InterpolationMode.NEAREST), *crop_step])
    return image_tf, mask_tf, output_size


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, paths, image_tf):
        self.paths, self.image_tf = list(paths), image_tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        # convert("RGB") 가 흑백 이미지의 채널을 세 번 복사한다
        return {"image": self.image_tf(Image.open(self.paths[i]).convert("RGB"))}


class TestDataset(torch.utils.data.Dataset):
    __test__ = False  # pytest 가 시험 클래스로 모으지 않게 한다

    def __init__(self, samples, image_tf, mask_tf, output_size):
        self.samples, self.image_tf = list(samples), image_tf
        self.mask_tf, self.output_size = mask_tf, output_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        sample = self.samples[i]
        image = self.image_tf(Image.open(sample.image).convert("RGB"))
        if sample.mask is None:
            mask = np.zeros(self.output_size, dtype=bool)
        else:
            # 팔레트 이미지도 색인값 그대로 읽어 0 이 아닌 곳을 결함으로 본다(VisA)
            mask = np.asarray(self.mask_tf(Image.open(sample.mask)))
            mask = mask.any(axis=-1) if mask.ndim == 3 else mask > 0
        return {
            "image": image,
            "mask": torch.from_numpy(np.array(mask, dtype=bool)),
            "is_anomaly": sample.label,
        }


def make_loaders(train_paths, test_samples, preprocess, batch_size, num_workers):
    image_tf, mask_tf, output_size = build_transforms(preprocess)
    options = dict(batch_size=batch_size, shuffle=False, num_workers=num_workers,
                   pin_memory=torch.cuda.is_available())
    train = torch.utils.data.DataLoader(TrainDataset(train_paths, image_tf), **options)
    test = torch.utils.data.DataLoader(
        TestDataset(test_samples, image_tf, mask_tf, output_size), **options
    )
    return train, test, output_size
