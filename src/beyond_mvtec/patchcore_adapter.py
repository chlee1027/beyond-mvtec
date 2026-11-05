"""원본 PatchCore(`amazon-science/patchcore-inspection`)를 파일 수정 없이 끼워 쓴다.

README 의 다섯 변경은 이렇게 들어간다.
(1) resnet18: 원본의 `_BACKBONES` 사전에 실행 중에 한 줄을 더한다(이 파일).
(2) 점수와 지도: 원본 `predict()` 는 마스크를 파이썬 목록으로 바꾼다. `_predict()` 를 직접
    불러 점수와 지도만 받고, 저장은 experiment.py 가 맡는다.
(3) 표본 고르기, (4) 데이터 읽기, (5) 정사각 줄이기: 원본 데이터셋 대신 data.py 의 로더를 넘긴다.

학습 순서는 원본 `bin/run_patchcore.py` 와 같다. 부류마다 시드 고정 -> 샘플러 -> 모델 -> fit.
"""
import numpy as np

UPSTREAM_COMMIT = "fcaa92f124fb1ad74a7acf56726decd4b27cbcad"
_UPSTREAM_NAME = {"resnet18": "resnet18", "wide_resnet50_2": "wideresnet50"}


class PatchCoreAdapter:
    def __init__(self, backbone, layers, coreset_ratio, output_size, device, seed=0,
                 patchsize=3, embed_dim=1024, faiss_on_gpu=False, faiss_workers=8):
        import patchcore.backbones
        import patchcore.common
        import patchcore.patchcore
        import patchcore.sampler
        import patchcore.utils

        patchcore.backbones._BACKBONES.setdefault("resnet18", "models.resnet18(pretrained=True)")
        patchcore.utils.fix_seeds(seed, device)
        sampler = patchcore.sampler.ApproximateGreedyCoresetSampler(coreset_ratio, device)
        name = _UPSTREAM_NAME[backbone]
        net = patchcore.backbones.load(name)
        net.name, net.seed = name, None
        self.model = patchcore.patchcore.PatchCore(device)
        self.model.load(
            backbone=net,
            layers_to_extract_from=[f"layer{i}" for i in sorted(layers)],
            device=device,
            input_shape=(3, *output_size),
            pretrain_embed_dimension=embed_dim,
            target_embed_dimension=embed_dim,
            patchsize=patchsize,
            featuresampler=sampler,
            anomaly_score_num_nn=1,
            nn_method=patchcore.common.FaissNN(faiss_on_gpu, faiss_workers),
        )

    def fit(self, loader):
        self.model.fit(loader)

    @property
    def bank_size(self):
        return int(self.model.anomaly_scorer.nn_method.search_index.ntotal)

    def predict_batch(self, images):
        scores, maps = self.model._predict(images)
        return np.asarray(scores), np.stack(maps)
