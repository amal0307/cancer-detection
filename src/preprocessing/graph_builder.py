import torch, numpy as np
from torch_geometric.data import Data, Batch
from skimage.segmentation import slic
from skimage.measure import regionprops
from scipy.spatial.distance import cdist
import cv2

def build_graph_from_image(image, feature_map=None, n_segments=64, n_neighbors=8):
    h, w = image.shape[:2]
    segments = slic(image, n_segments=n_segments, compactness=10, sigma=1, start_label=0)
    props = regionprops(segments + 1)
    centroids = np.array([p.centroid for p in props], dtype=np.float32)
    centroids[:, 0] /= h; centroids[:, 1] /= w
    num_nodes = len(props)

    if feature_map is not None:
        feat_np = feature_map.cpu().numpy()
        C, fh, fw = feat_np.shape
        node_feats = []
        for seg_id in range(num_nodes):
            mask = (segments == seg_id)
            mask_r = cv2.resize(mask.astype(np.uint8), (fw, fh), interpolation=cv2.INTER_NEAREST).astype(bool)
            nf = feat_np[:, mask_r].mean(axis=-1) if mask_r.sum() > 0 else np.zeros(C)
            node_feats.append(nf)
        node_feats = np.array(node_feats, dtype=np.float32)
    else:
        node_feats = np.array([[c[0], c[1], p.area/(h*w),
                                getattr(p,'eccentricity',0),
                                p.perimeter/(2*(h+w)) if hasattr(p,'perimeter') else 0,
                                getattr(p,'mean_intensity',0),
                                getattr(p,'solidity',0)]
                               for c, p in zip(centroids, props)], dtype=np.float32)

    if num_nodes > 1:
        dists = cdist(centroids, centroids)
        np.fill_diagonal(dists, np.inf)
        k = min(n_neighbors, num_nodes - 1)
        knn = np.argsort(dists, axis=1)[:, :k]
        src = np.repeat(np.arange(num_nodes), k)
        dst = knn.flatten()
        edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return Data(x=torch.tensor(node_feats, dtype=torch.float32),
                edge_index=edge_index, num_nodes=num_nodes)

def build_batch_graphs(images, feature_maps=None, n_segments=64, n_neighbors=8):
    graphs = [build_graph_from_image(img, feature_maps[i] if feature_maps else None,
                                     n_segments, n_neighbors)
              for i, img in enumerate(images)]
    return Batch.from_data_list(graphs)