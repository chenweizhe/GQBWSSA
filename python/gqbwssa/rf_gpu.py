

import numpy as np
import cupy as cp

_node_minmax_kernel = cp.RawKernel(r"""
extern "C" __global__
void node_minmax(const unsigned char* Xb, const int* samp, const int* node, const int* feats,
                 long long P, int k, int F, int* mn, int* mx) {
    long long tid = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= P * k) return;
    long long p = tid / k;
    int j = (int)(tid % k);
    int b = Xb[(long long)samp[p] * F + feats[j]];
    int idx = node[p] * k + j;
    if (b < mn[idx]) atomicMin(&mn[idx], b);
    if (b > mx[idx]) atomicMax(&mx[idx], b);
}
""", "node_minmax")


class FeatureBinner:
    def __init__(self, n_bins=64):
        if not 2 <= n_bins <= 256:
            raise ValueError("n_bins 必须在 [2, 256] 之间")
        self.n_bins = n_bins
        self.edges = []

    def fit(self, X):
        
        self.edges = []
        half = self.n_bins // 2
        qs = cp.linspace(0, 1, half + 1)[1:-1]
        for f in range(X.shape[1]):
            col = X[:, f]
            uniq = cp.unique(col)
            if uniq.size <= self.n_bins:
                edges = (uniq[:-1] + uniq[1:]) / 2
            else:
                e1 = cp.unique(cp.quantile(col, qs))
                n2 = self.n_bins - 1 - int(e1.size)
                idx = cp.rint(cp.linspace(0, uniq.size - 1, n2 + 2)[1:-1]).astype(cp.int64)
                edges = cp.unique(cp.concatenate([e1, uniq[idx]]))
            self.edges.append(edges)
        return self

    def transform(self, X):
        out = cp.empty(X.shape, dtype=cp.uint8)
        for f, edges in enumerate(self.edges):
            out[:, f] = cp.searchsorted(edges, X[:, f], side="right").astype(cp.uint8)
        return out


class CupyRandomForest:
   

    def __init__(self, ntree=20, mtry=2, n_bins=256, max_depth=None, skip_constant=True,
                 max_hist_elems=40_000_000, seed=None):
        self.ntree = ntree
        self.mtry = mtry
        self.n_bins = n_bins
        self.skip_constant = skip_constant
        self.max_depth = max_depth if max_depth is not None else 10_000
        self.max_hist_elems = max_hist_elems
        if seed is not None:
            cp.random.seed(seed)

    def prepare(self, xtrain, ytrain, xvalid):
        xtr = cp.asarray(xtrain, dtype=cp.float32)
        xva = cp.asarray(xvalid, dtype=cp.float32)
        binner = FeatureBinner(self.n_bins).fit(xtr)
        self.Xb_tr = binner.transform(xtr)
        self.Xb_va = binner.transform(xva)
        self.B = self.n_bins

        ytr = np.asarray(ytrain).ravel()
        self.classes = np.unique(ytr)
        self.C = len(self.classes)
        self.y_tr = cp.asarray(np.searchsorted(self.classes, ytr), dtype=cp.int32)
        self.n = self.Xb_tr.shape[0]
        self.m = self.Xb_va.shape[0]
        del xtr, xva
        cp.get_default_memory_pool().free_all_blocks()
        return self

    def _constant_mask(self, p_node, p_samp, feat_idx, n_nodes):
        
        k = int(feat_idx.size)
        mn = cp.full(n_nodes * k, 255, dtype=cp.int32)
        mx = cp.zeros(n_nodes * k, dtype=cp.int32)
        P = int(p_node.size)
        threads = 256
        blocks = (P * k + threads - 1) // threads
        _node_minmax_kernel((blocks,), (threads,),
                            (self.Xb_tr, p_samp, p_node, feat_idx, cp.int64(P), cp.int32(k),
                             cp.int32(self.Xb_tr.shape[1]), mn, mx))
        return (mn == mx).reshape(n_nodes, k)

    def _best_splits(self, p_node, p_samp, p_y, p_w, node_feats, n_nodes):
        
        mtry, B, C = node_feats.shape[1], self.B, self.C
        chunk = max(1, self.max_hist_elems // (mtry * B * C))
        best_crit = cp.empty(n_nodes, dtype=cp.float32)
        best_idx = cp.empty(n_nodes, dtype=cp.int64)
        j_off = cp.arange(mtry, dtype=cp.int64)

        for a in range(0, n_nodes, chunk):
            b = min(n_nodes, a + chunk)
            nn = b - a
            if a == 0 and b == n_nodes:
                q_node, q_samp, q_y, q_w = p_node, p_samp, p_y, p_w
            else:
                sel = (p_node >= a) & (p_node < b)
                q_node, q_samp, q_y, q_w = p_node[sel], p_samp[sel], p_y[sel], p_w[sel]
            local = (q_node - a).astype(cp.int64)
            pf = node_feats[q_node]
            pb = self.Xb_tr[q_samp[:, None], pf].astype(cp.int64)
            key = ((local[:, None] * mtry + j_off[None, :]) * B + pb) * C + q_y[:, None]
            H = cp.bincount(key.ravel(), weights=cp.repeat(q_w, mtry),
                            minlength=nn * mtry * B * C).astype(cp.float32).reshape(nn, mtry, B, C)

            L = cp.cumsum(H, axis=2)
            del H
            R = L[:, :, -1:, :] - L
            nL = L.sum(-1)
            nR = R.sum(-1)
            crit = (L * L).sum(-1) / cp.maximum(nL, 1e-6) + (R * R).sum(-1) / cp.maximum(nR, 1e-6)
            crit = cp.where((nL > 0) & (nR > 0), crit, -cp.inf).reshape(nn, mtry * B)
            idx = crit.argmax(axis=1)
            best_idx[a:b] = idx
            best_crit[a:b] = crit[cp.arange(nn), idx]
        return best_crit, best_idx

    def fit_predict(self, feat_idx):
        
        T, n, B, C = self.ntree, self.n, self.B, self.C
        feat_idx = cp.asarray(feat_idx, dtype=cp.int32)
        k = int(feat_idx.size)
        mtry = min(self.mtry, k)

        draws = cp.random.randint(0, n, size=(T, n), dtype=cp.int64)
        draws += cp.arange(T, dtype=cp.int64)[:, None] * n
        w = cp.bincount(draws.ravel(), minlength=T * n)
        del draws
        active = cp.nonzero(w)[0]
        p_w = w[active].astype(cp.float64)
        p_tree = (active // n).astype(cp.int32)
        p_samp = (active % n).astype(cp.int32)
        p_y = self.y_tr[p_samp]
        p_node = p_tree
        del w, active

        feats_lv, thr_lv, left_lv, right_lv, cls_lv = [], [], [], [], []
        n_nodes, offset, depth = T, 0, 0

        while n_nodes > 0:
            cnt = cp.bincount(p_node.astype(cp.int64) * C + p_y, weights=p_w,
                              minlength=n_nodes * C).reshape(n_nodes, C)
            tot = cnt.sum(1)
            leaf_cls = cnt.argmax(1).astype(cp.int32)
            can_split = (tot >= 2) & ((cnt > 0).sum(1) > 1)
            if depth >= self.max_depth:
                can_split[:] = False

            node_feat = cp.full(n_nodes, -1, dtype=cp.int32)
            node_thr = cp.zeros(n_nodes, dtype=cp.int32)
            node_left = cp.full(n_nodes, -1, dtype=cp.int32)
            node_right = cp.full(n_nodes, -1, dtype=cp.int32)

            split_ids = cp.nonzero(can_split)[0].astype(cp.int32)
            n_s = int(split_ids.size)
            n_new = 0
            if n_s > 0:
                rank = cp.full(n_nodes, -1, dtype=cp.int32)
                rank[split_ids] = cp.arange(n_s, dtype=cp.int32)
                keep = rank[p_node] >= 0
                p_node, p_samp, p_y, p_w = rank[p_node[keep]], p_samp[keep], p_y[keep], p_w[keep]

                keys = cp.random.random((n_s, k))
                if self.skip_constant and k > mtry:
                    keys[self._constant_mask(p_node, p_samp, feat_idx, n_s)] += 2.0
                choose = cp.argsort(keys, axis=1)[:, :mtry]
                node_feats = feat_idx[choose]

                best_crit, best_idx = self._best_splits(p_node, p_samp, p_y, p_w, node_feats, n_s)
                ok = cp.isfinite(best_crit)
                j = (best_idx // B).astype(cp.int32)
                thr = (best_idx % B).astype(cp.int32)
                sf = node_feats[cp.arange(n_s), j]

                ok_ids = cp.nonzero(ok)[0].astype(cp.int32)
                n_ok = int(ok_ids.size)
                if n_ok > 0:
                    r2 = cp.full(n_s, -1, dtype=cp.int32)
                    r2[ok_ids] = cp.arange(n_ok, dtype=cp.int32)
                    gid = split_ids[ok_ids]
                    node_feat[gid] = sf[ok_ids]
                    node_thr[gid] = thr[ok_ids]
                    child0 = offset + n_nodes + 2 * cp.arange(n_ok, dtype=cp.int32)
                    node_left[gid] = child0
                    node_right[gid] = child0 + 1

                    pr = r2[p_node]
                    keep = pr >= 0
                    p_node, p_samp, p_y, p_w = p_node[keep], p_samp[keep], p_y[keep], p_w[keep]
                    pr = pr[keep]
                    go_right = self.Xb_tr[p_samp, sf[p_node]] > thr[p_node].astype(cp.uint8)
                    p_node = 2 * pr + go_right.astype(cp.int32)
                    n_new = 2 * n_ok

            feats_lv.append(node_feat)
            thr_lv.append(node_thr)
            left_lv.append(node_left)
            right_lv.append(node_right)
            cls_lv.append(leaf_cls)
            offset += n_nodes
            n_nodes = n_new
            depth += 1

        feat_all = cp.concatenate(feats_lv)
        thr_all = cp.concatenate(thr_lv)
        left_all = cp.concatenate(left_lv)
        right_all = cp.concatenate(right_lv)
        cls_all = cp.concatenate(cls_lv)
        return self._predict(feat_all, thr_all, left_all, right_all, cls_all, depth)

    def _predict(self, feat_all, thr_all, left_all, right_all, cls_all, depth):
        T, m, C = self.ntree, self.m, self.C
        cur = cp.repeat(cp.arange(T, dtype=cp.int32)[:, None], m, axis=1)
        samp = cp.broadcast_to(cp.arange(m, dtype=cp.int32)[None, :], (T, m))
        for _ in range(depth):
            f = feat_all[cur]
            leaf = f < 0
            b = self.Xb_va[samp, cp.maximum(f, 0)].astype(cp.int32)
            nxt = cp.where(b <= thr_all[cur], left_all[cur], right_all[cur])
            cur = cp.where(leaf, cur, nxt)
        votes = cls_all[cur]
        key = (cp.arange(m, dtype=cp.int64)[None, :] * C + votes).ravel()
        counts = cp.bincount(key, minlength=m * C).reshape(m, C)
        return counts.argmax(1).astype(cp.int32)
