"""PaDiM 직접 구현. 저자 공식 코드가 없다.

원 논문이 정한 값: 평균과 공분산(분모 N-1) + 0.01 I, 마할라노비스 거리의 제곱근, 쌍삼차 업샘플,
가우시안 sigma=4, 이미지 점수는 이상 지도의 최댓값.
원 논문이 안 정해서 이 레포가 고른 값: 작은 특징 지도는 최근접으로 격자에 맞춘다. 고를 차원은
시드당 한 벌을 전 위치에 쓰고 중복 없이 뽑는다. 평활 뒤 지도에서 최댓값을 잡는다.
"""
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from scipy import ndimage

_BACKBONES = {
    "resnet18": torchvision.models.resnet18,
    "wide_resnet50_2": torchvision.models.wide_resnet50_2,
}


class PaDiM:
    def __init__(self, backbone, layers, reduced_dim, output_size, device="cpu", seed=0,
                 eps=0.01, sigma=4.0, chunk=256, pretrained=True):
        weights = "IMAGENET1K_V1" if pretrained else None
        self.net = _BACKBONES[backbone](weights=weights).to(device).eval()
        self.layers = sorted(layers)
        self.output_size = tuple(output_size)
        self.device = device
        self.eps, self.sigma, self.chunk = eps, sigma, chunk
        self.mean = self.inv_cov = None
        self.n_train = 0

        with torch.no_grad():
            dummy = torch.zeros(1, 3, *self.output_size, device=device)
            total = self._concat_features(dummy).shape[1]
        if reduced_dim > total:
            raise ValueError(f"줄인 차원 {reduced_dim} 이 원래 차원 {total} 보다 크다")
        generator = torch.Generator().manual_seed(seed)
        self.dims = torch.randperm(total, generator=generator)[:reduced_dim].sort().values.to(device)

    def _concat_features(self, images):
        net = self.net
        x = net.maxpool(net.relu(net.bn1(net.conv1(images))))
        kept = []
        for i in range(1, max(self.layers) + 1):
            x = getattr(net, f"layer{i}")(x)
            if i in self.layers:
                kept.append(x)
        grid = kept[0].shape[-2:]
        return torch.cat([F.interpolate(f, size=grid, mode="nearest") for f in kept], dim=1)

    def embed(self, images):
        """B x d x H x W."""
        return self._concat_features(images).index_select(1, self.dims)

    @torch.no_grad()
    def fit(self, loader):
        """평균과 공분산을 위치마다 누적해 구한다. 합은 float64 로 둔다."""
        sum_x = sum_xxt = None
        n = 0
        for batch in loader:
            features = self.embed(batch["image"].to(self.device)).double()
            b, d, h, w = features.shape
            x = features.permute(2, 3, 0, 1).reshape(h * w, b, d)
            if sum_x is None:
                sum_x = torch.zeros(h * w, d, dtype=torch.float64, device=self.device)
                sum_xxt = torch.zeros(h * w, d, d, dtype=torch.float64, device=self.device)
            sum_x += x.sum(1)
            for s in range(0, h * w, self.chunk):
                part = x[s:s + self.chunk]
                sum_xxt[s:s + self.chunk] += part.transpose(1, 2).bmm(part)
            n += b
        if n < 2:
            raise ValueError("공분산에는 학습 이미지가 두 장 이상 필요하다")
        self.n_train = n

        mean = sum_x / n
        identity = torch.eye(mean.shape[1], dtype=torch.float64, device=self.device)
        self.mean = mean.float()
        self.inv_cov = torch.zeros(sum_xxt.shape, dtype=torch.float32, device=self.device)
        for s in range(0, mean.shape[0], self.chunk):
            m = mean[s:s + self.chunk]
            cov = (sum_xxt[s:s + self.chunk] - n * m.unsqueeze(2).bmm(m.unsqueeze(1))) / (n - 1)
            chol = torch.linalg.cholesky(cov + self.eps * identity)
            self.inv_cov[s:s + self.chunk] = torch.cholesky_inverse(chol).float()

    @torch.no_grad()
    def distance_grid(self, images):
        """B x H x W 마할라노비스 거리."""
        features = self.embed(images.to(self.device))
        b, d, h, w = features.shape
        delta = features.permute(2, 3, 0, 1).reshape(h * w, b, d) - self.mean.unsqueeze(1)
        squared = torch.zeros(h * w, b, device=self.device)
        for s in range(0, h * w, self.chunk):
            part = delta[s:s + self.chunk]
            squared[s:s + self.chunk] = (part.bmm(self.inv_cov[s:s + self.chunk]) * part).sum(-1)
        # float32 역행렬의 반올림으로 작은 음수가 나올 수 있어 0 에서 자른다
        return squared.clamp(min=0).sqrt().permute(1, 0).reshape(b, h, w)

    def predict_batch(self, images):
        """(이미지 점수, 이상 지도). 이미지 점수는 평활한 지도의 최댓값."""
        grid = self.distance_grid(images)
        maps = F.interpolate(grid.unsqueeze(1), size=self.output_size, mode="bicubic",
                             align_corners=False).squeeze(1).cpu().numpy()
        maps = np.stack([ndimage.gaussian_filter(m, sigma=self.sigma) for m in maps])
        return maps.reshape(len(maps), -1).max(axis=1), maps
