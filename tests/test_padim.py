import numpy as np
import pytest
import torch

from beyond_mvtec.padim import PaDiM


def _loader(images, batch_size):
    return [{"image": images[i:i + batch_size]} for i in range(0, len(images), batch_size)]


def _model(reduced_dim=20, seed=0, eps=0.01):
    torch.manual_seed(0)  # 무작위 초기화 백본을 쓰므로 가중치도 고정한다
    return PaDiM("resnet18", [1, 2, 3], reduced_dim, (64, 64), seed=seed, eps=eps, chunk=7,
                 pretrained=False)


def _images(n, seed=1):
    return torch.randn(n, 3, 64, 64, generator=torch.Generator().manual_seed(seed))


def test_grid_and_raw_dimension_follow_the_layers():
    model = _model()
    assert model._concat_features(_images(1)).shape == (1, 64 + 128 + 256, 16, 16)
    two = PaDiM("resnet18", [2, 3], 20, (64, 64), pretrained=False)
    assert two._concat_features(_images(1)).shape == (1, 128 + 256, 8, 8)


def test_dims_are_distinct_and_follow_the_seed():
    a, b = _model(seed=0).dims, _model(seed=1).dims
    assert len(set(a.tolist())) == 20
    assert a.equal(_model(seed=0).dims) and not a.equal(b)


def test_mean_and_cov_match_numpy():
    images = _images(30)
    model = _model()
    model.fit(_loader(images, 4))
    with torch.no_grad():
        x = model.embed(images).double().numpy()  # N d H W
    for h, w in [(0, 0), (5, 11), (15, 15)]:
        sample = x[:, :, h, w]
        index = h * 16 + w
        expected = np.linalg.inv(np.cov(sample, rowvar=False, ddof=1) + 0.01 * np.eye(20))
        assert model.mean[index].numpy() == pytest.approx(sample.mean(0), abs=1e-4)
        assert model.inv_cov[index].numpy() == pytest.approx(expected, rel=1e-3, abs=1e-3)


def test_batch_size_does_not_change_the_fit():
    images = _images(12)
    a, b = _model(), _model()
    a.fit(_loader(images, 1))
    b.fit(_loader(images, 5))
    assert torch.allclose(a.inv_cov, b.inv_cov, rtol=1e-3, atol=1e-3)


def test_distance_matches_the_formula():
    images, query = _images(30), _images(2, seed=9)
    model = _model()
    model.fit(_loader(images, 8))
    grid = model.distance_grid(query)
    with torch.no_grad():
        q = model.embed(query).double().numpy()
        x = model.embed(images).double().numpy()
    b, h, w = 1, 3, 4
    sample = x[:, :, h, w]
    delta = q[b, :, h, w] - sample.mean(0)
    inv = np.linalg.inv(np.cov(sample, rowvar=False, ddof=1) + 0.01 * np.eye(20))
    assert grid[b, h, w].item() == pytest.approx(np.sqrt(delta @ inv @ delta), rel=1e-3)


def test_fewer_images_than_dimensions_stays_finite():
    # 공분산 계수가 차원보다 작아도 0.01 I 덕분에 역행렬이 나와야 한다
    model = _model(reduced_dim=100)
    model.fit(_loader(_images(5), 2))
    scores, maps = model.predict_batch(_images(2, seed=3))
    assert np.isfinite(scores).all() and np.isfinite(maps).all()
    assert maps.shape == (2, 64, 64)


def test_image_score_is_the_map_maximum():
    model = _model()
    model.fit(_loader(_images(10), 4))
    scores, maps = model.predict_batch(_images(3, seed=4))
    assert scores == pytest.approx(maps.reshape(3, -1).max(axis=1))


def test_one_image_cannot_fit():
    with pytest.raises(ValueError):
        _model().fit(_loader(_images(1), 1))


def test_too_many_dims_is_rejected():
    with pytest.raises(ValueError):
        PaDiM("resnet18", [2, 3], 1000, (64, 64), pretrained=False)
