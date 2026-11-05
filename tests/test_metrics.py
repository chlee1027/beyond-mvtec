import numpy as np
import pytest

from beyond_mvtec import metrics


def test_perfect_detector_gets_one():
    masks = np.zeros((2, 8, 8), dtype=bool)
    masks[0, 2:4, 2:4] = True
    maps = masks.astype(np.float32)
    assert metrics.pro_auc(maps, masks) == pytest.approx(1.0)
    assert metrics.pixel_auroc(maps, masks) == pytest.approx(1.0)


def test_inverted_detector_gets_zero():
    masks = np.zeros((1, 8, 8), dtype=bool)
    masks[0, 2:4, 2:4] = True
    # 결함 픽셀에 가장 낮은 점수를 주면 FPR 0.3 까지 결함을 하나도 못 맞힌다
    assert metrics.pro_auc((~masks).astype(np.float32), masks) == pytest.approx(0.0)


def test_diagonal_pixels_are_one_region():
    masks = np.zeros((1, 6, 6), dtype=bool)
    masks[0, 1, 1] = masks[0, 2, 2] = True
    assert metrics.count_regions(masks) == 1


def test_regions_weigh_the_same_regardless_of_size():
    # 큰 영역(16픽셀)과 작은 영역(1픽셀). 작은 영역만 맞히면 PRO 는 0.5 다.
    masks = np.zeros((1, 20, 20), dtype=bool)
    masks[0, 0:4, 0:4] = True
    masks[0, 10, 10] = True
    maps = np.zeros((1, 20, 20), dtype=np.float32)
    maps[0, 10, 10] = 1.0
    fprs, pros = metrics.pro_curve(maps, masks)
    # 문턱 둘: 점수 1(작은 영역만), 점수 0(나머지 전부)
    assert fprs.tolist() == pytest.approx([0.0, 0.0, 1.0, 1.0])
    assert pros.tolist() == pytest.approx([0.0, 0.5, 1.0, 1.0])
    # FPR 0.3 의 PRO 는 0.5 + 0.5 x 0.3 = 0.65, 넓이는 (0.5 + 0.65) / 2 x 0.3, 정규화하면 0.575
    assert metrics.pro_auc(maps, masks) == pytest.approx(0.575)


def test_fpr_counts_normal_images_too():
    masks = np.zeros((2, 4, 4), dtype=bool)
    masks[0, 0, 0] = True
    maps = np.zeros((2, 4, 4), dtype=np.float32)
    maps[0, 0, 0] = 3.0  # 결함
    maps[1, 0:2, 0:2] = 2.0  # 정상 이미지의 오탐 4픽셀
    fprs, pros = metrics.pro_curve(maps, masks)
    # 정상 픽셀은 15 + 16 = 31개
    assert fprs[2] == pytest.approx(4 / 31)
    assert pros[1] == pytest.approx(1.0)


def test_interpolates_at_the_limit():
    x, y = [0.0, 0.2, 0.4], [0.0, 0.2, 0.4]
    assert metrics.area_under_curve(x, y, 0.3) == pytest.approx(0.5 * 0.3 * 0.3)


def test_limit_on_a_point_uses_no_interpolation():
    assert metrics.area_under_curve([0.0, 0.3, 1.0], [0.0, 0.6, 1.0], 0.3) == pytest.approx(0.09)


def test_repeated_fpr_keeps_the_vertical_step():
    # FPR 이 그대로인 채 PRO 만 오르는 구간은 넓이가 없다
    assert metrics.area_under_curve([0.0, 0.0, 0.3], [0.0, 1.0, 1.0], 0.3) == pytest.approx(0.3)


def test_evaluate_without_regions_returns_nan():
    result = metrics.evaluate([0, 1], [0.1, 0.9], np.zeros((2, 4, 4)), np.zeros((2, 4, 4), dtype=bool))
    assert result["image_auroc"] == 1.0
    assert np.isnan(result["pro"]) and np.isnan(result["pixel_auroc"])
    assert result["n_regions"] == 0


def _pro_by_definition(maps, masks, limit=0.3):
    """정의대로 문턱마다 직접 센다. 느리지만 구현과 독립이다."""
    from scipy import ndimage

    components = []
    for i, mask in enumerate(masks):
        labeled, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
        components += [(i, labeled == k + 1) for k in range(n)]
    normal = ~masks
    xs, ys = [0.0], [0.0]
    for t in np.unique(maps)[::-1]:
        predicted = maps >= t
        xs.append((predicted & normal).sum() / normal.sum())
        ys.append(np.mean([(predicted[i] & c).sum() / c.sum() for i, c in components]))
    xs, ys = np.array(xs), np.array(ys)
    y_at = np.interp(limit, xs, ys) if limit not in xs else ys[xs == limit][-1]
    keep = xs <= limit
    x, y = np.append(xs[keep], limit), np.append(ys[keep], y_at)
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])) / limit)


@pytest.mark.parametrize("seed", range(20))
def test_fast_curve_matches_counting_by_definition(seed):
    rng = np.random.RandomState(seed)
    n, h, w = rng.randint(2, 6), rng.randint(10, 24), rng.randint(10, 24)
    masks = np.zeros((n, h, w), dtype=bool)
    for i in range(n):
        for _ in range(rng.randint(0, 4)):  # 결함 없는 이미지와 붙은 영역도 나온다
            y, x = rng.randint(0, h - 2), rng.randint(0, w - 2)
            masks[i, y:y + rng.randint(1, 7), x:x + rng.randint(1, 7)] = True
    masks[0, 0, 0] = True
    maps = rng.rand(n, h, w).astype(np.float32) + masks * rng.uniform(0, 1)
    if seed % 2:  # 동점을 많이 만든다
        maps = np.round(maps * 8) / 8
    assert metrics.pro_auc(maps, masks) == pytest.approx(_pro_by_definition(maps, masks), abs=1e-6)
