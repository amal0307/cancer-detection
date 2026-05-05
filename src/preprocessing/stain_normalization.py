import numpy as np, cv2
from typing import Optional

class MacenkoNormalizer:
    def __init__(self):
        self.stain_matrix_target = None
        self.maxC_target = None
        self.alpha = 1.0

    def _get_stain_matrix(self, I):
        I = I.astype(np.float64) / 255.0
        mask = (I > 0.05).all(axis=2)
        OD = -np.log(np.maximum(I, 1e-6))
        OD_hat = OD[mask]
        if OD_hat.shape[0] < 10:
            return np.eye(3)[:2], np.array([1.0, 1.0])
        _, V = np.linalg.eigh(np.cov(OD_hat.T))
        V = V[:, [2, 1]]
        if V[0, 0] < 0: V[:, 0] = -V[:, 0]
        if V[0, 1] < 0: V[:, 1] = -V[:, 1]
        That = OD_hat @ V
        phi = np.arctan2(That[:, 1], That[:, 0])
        minPhi = np.percentile(phi, self.alpha)
        maxPhi = np.percentile(phi, 100 - self.alpha)
        v1 = V @ np.array([np.cos(minPhi), np.sin(minPhi)])
        v2 = V @ np.array([np.cos(maxPhi), np.sin(maxPhi)])
        HE = np.column_stack([v1, v2] if v1[0] > v2[0] else [v2, v1])
        HE = HE / np.linalg.norm(HE, axis=0)
        C = np.linalg.lstsq(HE, OD_hat.T, rcond=None)[0]
        maxC = np.percentile(C, 99, axis=1)
        return HE, maxC

    def fit(self, target_image: np.ndarray):
        self.stain_matrix_target, self.maxC_target = self._get_stain_matrix(target_image)

    def transform(self, image: np.ndarray) -> np.ndarray:
        if self.stain_matrix_target is None:
            raise RuntimeError("Call fit() first")
        h, w = image.shape[:2]
        I = image.astype(np.float64) / 255.0
        OD = -np.log(np.maximum(I.reshape(-1, 3), 1e-6))
        HE_src, maxC_src = self._get_stain_matrix(image)
        C = np.linalg.lstsq(HE_src, OD.T, rcond=None)[0]
        C_norm = C / maxC_src[:, None] * self.maxC_target[:, None]
        I_norm = np.exp(-self.stain_matrix_target @ C_norm)
        I_norm = np.clip(I_norm, 0, 1).T.reshape(h, w, 3)
        return (I_norm * 255).astype(np.uint8)


class StainNormalizationPipeline:
    def __init__(self, method: str = "macenko"):
        self.normalizer = MacenkoNormalizer()
        self._fit_canonical_reference()

    def _fit_canonical_reference(self):
        ref = np.zeros((100, 100, 3), dtype=np.uint8)
        ref[:50, :] = [200, 150, 200]
        ref[50:, :] = [230, 180, 200]
        try: self.normalizer.fit(ref)
        except: pass

    def normalize(self, image: np.ndarray) -> np.ndarray:
        try: return self.normalizer.transform(image)
        except: return image

    def __call__(self, image): return self.normalize(image)