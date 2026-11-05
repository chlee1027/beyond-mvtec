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


def _find_real_iad_json(json_root, category):
    """같은 이름의 JSON 이 여러 폴더에 있으면 기본 분할(`realiad_jsons`)을 고른다."""
    found = []
    for folder, _, files in os.walk(json_root):
        if f"{category}.json" in files:
            found.append(os.path.join(folder, f"{category}.json"))

    def rank(path):
        parent = os.path.basename(os.path.dirname(path)).lower()
        if parent == "realiad_jsons":
            return 0
        if "fuiad" in parent:
            return 2
        return 3 if parent.endswith("_sv") else 1

    if not found:
        raise FileNotFoundError(f"{json_root} 아래에 {category}.json 이 없다")
    return sorted(found, key=rank)[0]


def _real_iad_is_normal(entry, normal_class):
    for key in ("anomaly_class", "specie_name"):
        if key in entry:
            return str(entry[key]).strip().upper() == normal_class
    raise ValueError(f"이미지 라벨 항목을 찾지 못했다. 항목의 키: {sorted(entry)}")


def view_of(path):
    match = _VIEW.search(os.path.basename(path))
    return match.group(1) if match else None


def list_real_iad(root, category, view="C1", json_root=None):
    """JSON 의 이미지 라벨로 채점한다. 이상 표본 안에서 결함이 안 보이는 시점은 정상이다."""
    json_root = json_root or os.path.join(root, "realiad_jsons")
    with open(_find_real_iad_json(json_root, category), encoding="utf-8") as f:
        meta = json.load(f)
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
