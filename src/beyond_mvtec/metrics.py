"""이미지 AUROC, 픽셀 AUROC, PRO. 두 방법이 같은 코드로 재도록 여기 하나만 둔다.

PRO 의 세부 선택은 MVTec AD 평가 코드(`pro_curve_util.py`, `generic_util.py`)와 같다.
- 연결 성분: 정답 마스크에서 8-연결
- 문턱: 점수의 서로 다른 값 전부(문턱 개수를 따로 정하지 않는다)
- FPR: 결함 없는 시험 이미지까지 부류의 정상 픽셀을 모아 한 번에 나눈다
- PRO: 부류의 모든 성분에 같은 무게
- 적분: 사다리꼴, FPR 0.3 은 양옆 점을 선형 보간, 0.3 으로 나눠 최댓값 1
"""
import numpy as np
from scipy import ndimage
from sklearn import metrics as skm

_CONNECTIVITY = np.ones((3, 3), dtype=int)


def image_auroc(labels, scores):
    return float(skm.roc_auc_score(np.asarray(labels).astype(int), np.asarray(scores)))


def pixel_auroc(maps, masks):
    """정상 이미지까지 포함한 전체 픽셀로 잰다. 결함 픽셀이 없으면 nan."""
    masks = np.asarray(masks).astype(bool)
    if not masks.any():
        return float("nan")
    return float(skm.roc_auc_score(masks.ravel().astype(np.uint8), np.asarray(maps).ravel()))


def pro_curve(maps, masks):
    """(FPR, PRO) 곡선을 돌려준다. maps, masks 는 N x H x W."""
    maps = np.asarray(maps)
    masks = np.asarray(masks).astype(bool)
    if maps.shape != masks.shape or maps.ndim != 3:
        raise ValueError(f"maps {maps.shape} 와 masks {masks.shape} 는 같은 N x H x W 여야 한다")

    weights = np.zeros(masks.shape, dtype=np.float64)
    n_regions = 0
    for i, mask in enumerate(masks):
        if not mask.any():
            continue
        labeled, n = ndimage.label(mask, structure=_CONNECTIVITY)
        n_regions += n
        sizes = np.bincount(labeled.ravel())
        per_pixel = np.zeros_like(sizes, dtype=np.float64)
        per_pixel[1:] = 1.0 / sizes[1:]
        weights[i] = per_pixel[labeled]
    if n_regions == 0:
        raise ValueError("정답 마스크에 결함 영역이 하나도 없다")

    order = np.argsort(maps, axis=None)[::-1]
    scores = maps.ravel()[order]
    ok_sorted = ~masks.ravel()[order]
    weights_sorted = weights.ravel()[order]
    del order, weights

    fprs = np.cumsum(ok_sorted, dtype=np.float64) / ok_sorted.sum()
    pros = np.cumsum(weights_sorted) / n_regions
    # 같은 점수의 픽셀은 한 문턱에서 같이 넘어가므로 마지막 점만 남긴다
    keep = np.append(scores[1:] != scores[:-1], True)
    fprs, pros = np.clip(fprs[keep], None, 1.0), np.clip(pros[keep], None, 1.0)
    return np.concatenate(([0.0], fprs, [1.0])), np.concatenate(([0.0], pros, [1.0]))


def area_under_curve(x, y, x_max):
    """x_max 까지의 사다리꼴 넓이. x 는 오름차순이고 같은 값이 겹쳐도 된다."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    end = np.searchsorted(x, x_max, side="right")
    if not 0 < end <= len(x):
        raise ValueError(f"x_max={x_max} 가 곡선 범위 밖이다")
    area = np.sum(0.5 * (y[1:end] + y[:end - 1]) * (x[1:end] - x[:end - 1]))
    if x[end - 1] < x_max:  # x_max 가 점과 점 사이면 그 조각을 보간해 더한다
        slope = (y[end] - y[end - 1]) / (x[end] - x[end - 1])
        y_at = y[end - 1] + slope * (x_max - x[end - 1])
        area += 0.5 * (y_at + y[end - 1]) * (x_max - x[end - 1])
    return float(area)


def pro_auc(maps, masks, fpr_limit=0.3):
    fprs, pros = pro_curve(maps, masks)
    return area_under_curve(fprs, pros, fpr_limit) / fpr_limit


def count_regions(masks):
    masks = np.asarray(masks).astype(bool)
    return int(sum(ndimage.label(m, structure=_CONNECTIVITY)[1] for m in masks if m.any()))


def evaluate(labels, scores, maps, masks):
    """부류 하나의 세 지표. 결함 픽셀이 없으면 픽셀 지표는 nan."""
    has_regions = np.asarray(masks).any()
    return {
        "image_auroc": image_auroc(labels, scores),
        "pixel_auroc": pixel_auroc(maps, masks),
        "pro": pro_auc(maps, masks) if has_regions else float("nan"),
        "n_regions": count_regions(masks),
    }
