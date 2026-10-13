# PEBR-Net, the prior- and evidence-bounded reconstruction network.
# MIT License, see LICENSE.
"""Inference and TorchScript export.

This file is executed by pebrnet/__init__.py into the package namespace, in the order listed there;
names defined in the other files of the package are available here without imports.
"""
from __future__ import annotations

def resolve_inference_window(cfg: Config) -> int:
    """Window length for whole-run inference (0 = never window)."""
    w = int(getattr(cfg.data, "inference_window", 0) or 0)
    if w == 0:
        w = int(getattr(cfg.data, "profile_patch_len", 0) or 0)
    return max(0, w)


def _sliding_window_forward_core(model: "PEBRNet", x: torch.Tensor, nb: Optional[torch.Tensor],
                           geo: Optional[torch.Tensor], depth: Optional[torch.Tensor],
                           window: int, keys: Tuple[str, ...] = ("denoised",),
                           obs_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    n = int(x.shape[0])
    W = int(window)
    _kw = {"obs_mask": obs_mask} if obs_mask is not None else {}
    if W <= 0 or n <= W:
        o = model(x, nb, geo, depth_norm=depth, profile_len=(n if n > 2 else None), **_kw)
        return {k: o[k] for k in keys if k in o}
    starts = list(range(0, n - W + 1))
    idx = torch.tensor([[w + p for p in range(W)] for w in starts], device=x.device).reshape(-1)
    xw = x[idx]
    nbw = nb[idx] if nb is not None else None
    geow = geo[idx] if geo is not None else None
    dw = depth[idx] if depth is not None else None
    if obs_mask is not None:
        _kw = {"obs_mask": obs_mask[idx]}
    o = model(xw, nbw, geow, depth_norm=dw, profile_len=W, **_kw)
    pos = torch.arange(W, device=x.device, dtype=torch.float32)
    hann = 0.5 - 0.5 * torch.cos(2.0 * math.pi * (pos + 1.0) / float(W + 1))
    out: Dict[str, torch.Tensor] = {}
    for k in keys:
        if k not in o:
            continue
        v = o[k].float()
        vw = v.reshape(len(starts), W, *v.shape[1:])
        acc = torch.zeros((n,) + tuple(v.shape[1:]), device=x.device, dtype=torch.float32)
        wsum = torch.zeros(n, device=x.device, dtype=torch.float32)
        wv = hann.view(W, *([1] * (v.dim() - 1)))
        for j, w0 in enumerate(starts):
            acc[w0:w0 + W] += vw[j] * wv
            wsum[w0:w0 + W] += hann
        out[k] = (acc / wsum.view(n, *([1] * (v.dim() - 1)))).to(o[k].dtype)
    return out


def _reflect_extension_indices(n: int, k: int) -> List[int]:
    n = int(n)
    k = int(max(0, min(k, n - 1)))
    return list(range(k, 0, -1)) + list(range(n)) + list(range(n - 2, n - 2 - k, -1))


def resolve_edge_mode(cfg: Config) -> str:
    """The edge-extension mode in force: an explicit setting, or the mode the validation A/B selected (stored on
    cfg.data by select_edge_reflect_mode), else none.
    """
    m = str(getattr(cfg.data, "edge_reflect_mode", "auto") or "auto").strip().lower()
    if m in ("none", "window", "full"):
        return m
    r = str(getattr(cfg.data, "_edge_mode_resolved", "") or "").strip().lower()
    return r if r in ("none", "window", "full") else "none"


def sliding_window_forward(model: "PEBRNet", x: torch.Tensor, nb: Optional[torch.Tensor],
                           geo: Optional[torch.Tensor], depth: Optional[torch.Tensor],
                           window: int, keys: Tuple[str, ...] = ("denoised",),
                           edge_mode: str = "none", reflect_k: int = 0,
                           obs_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """Inference over a whole run of stations in training-length sliding windows."""
    n = int(x.shape[0])
    W = int(window)
    mode = str(edge_mode or "none").lower()
    if mode not in ("window", "full") or W <= 0 or n <= W or n < 3:
        return _sliding_window_forward_core(model, x, nb, geo, depth, W, keys, obs_mask=obs_mask)
    k = int(reflect_k) if int(reflect_k) > 0 else max(1, W // 2)
    k = max(1, min(k, n - 1))
    idx = torch.tensor(_reflect_extension_indices(n, k), device=x.device, dtype=torch.long)
    xe = x[idx]
    nbe = nb[idx].clone() if nb is not None else None
    geoe = geo[idx].clone() if geo is not None else None
    de = depth[idx] if depth is not None else None
    m = int(xe.shape[0])
    if mode == "full" and nbe is not None:
        K = int(nbe.shape[1])
        if geoe is not None and geoe.dim() == 3 and geoe.shape[1] == K and K >= 2:
            _stack = bool(float(geoe[:, K - 1, 3].max().item()) > 0.5) if geoe.shape[-1] >= 4 else False
            Kr = K - 1 if _stack else K
        else:
            Kr = K
        if Kr >= 1 and m > Kr:
            nbi = torch.tensor(build_neighbor_indices(m, Kr), device=x.device, dtype=torch.long)
            rows = torch.tensor([i for i in range(m) if i < 2 * k or i >= m - 2 * k],
                                device=x.device, dtype=torch.long)
            nbe[rows, :Kr, :] = xe[nbi[rows], 0, :].to(dtype=nbe.dtype)
            if geoe is not None and geoe.dim() == 3 and geoe.shape[-1] >= 3:
                off = (nbi[rows] - rows[:, None]).to(dtype=geoe.dtype)
                geoe[rows, :Kr, 0] = off.clamp(-16.0, 16.0)
                geoe[rows, :Kr, 1] = off.abs().clamp(max=16.0)
                geoe[rows, :Kr, 2] = 1.0
    out = _sliding_window_forward_core(model, xe, nbe, geoe, de, W, keys,
                                       obs_mask=(obs_mask[idx] if obs_mask is not None else None))
    return {kk: v[k:k + n] for kk, v in out.items()}


def select_edge_reflect_mode(model: "PEBRNet", cfg: Config, run_dir: Path, clean_cache: CleanCacheInfo,
                             noise_cache: NoiseCacheInfo, logger: logging.Logger) -> str:
    """A/B the three edge modes on the validation split (every area, whole-run forward), print the late NRMSE on edge
    stations / interior / by distance from the edge for each.
    """
    _cfgm = str(getattr(cfg.data, "edge_reflect_mode", "auto") or "auto").strip().lower()
    if _cfgm in ("none", "window", "full"):
        cfg.data._edge_mode_resolved = _cfgm
        logger.info("EDGE MODE fixed by configuration: %s (no A/B).", _cfgm)
        return _cfgm
    _done = str(getattr(cfg.data, "_edge_mode_resolved", "") or "").strip().lower()
    if _done in ("none", "window", "full"):
        logger.info("EDGE MODE already selected earlier in this run: %s.", _done)
        return _done
    pid = np.asarray(getattr(clean_cache, "province_id"))
    splits = build_splits(clean_cache, cfg)
    rows = np.sort(np.asarray(splits["val" if "val" in splits else "test"], dtype=np.int64))
    if rows.size == 0:
        cfg.data._edge_mode_resolved = "none"
        return "none"
    ls = int(cfg.data.late_start_index)
    K = int(max(1, cfg.data.num_neighbors))
    clean_mm = np.load(clean_cache.data_path, mmap_mode="r")
    areas = np.unique(pid[rows])
    dist = np.zeros(rows.size, dtype=np.int64)
    pos = {int(r): i for i, r in enumerate(rows)}
    for a in areas:
        r = rows[pid[rows] == a]
        nn_ = int(r.size)
        for i, rr in enumerate(r):
            dist[pos[int(rr)]] = min(i, nn_ - 1 - i)
    cl = np.asarray(clean_mm[rows], dtype=np.float64)[:, ls:]
    results: Dict[str, Dict[str, Any]] = {}
    for mode in ("none", "window", "full"):
        try:
            pred = _paired_test_forward(model, cfg, clean_cache, noise_cache, rows, logger, edge_mode=mode)
        except Exception as exc:
            logger.warning("edge mode %s failed: %r", mode, exc)
            continue
        d = np.asarray(pred, dtype=np.float64)[:, ls:]
        e2 = np.sum((d - cl) ** 2, axis=1)
        y2 = np.sum(cl ** 2, axis=1)
        def _nr(mask):
            return 100.0 * math.sqrt(float(e2[mask].sum()) / max(float(y2[mask].sum()), 1e-300)) if mask.any() else float("nan")
        edge = dist < K
        inter = ~edge
        by_d = {("d=%d" % q): _nr(dist == q) for q in range(4)}
        by_d["d=4-7"] = _nr((dist >= 4) & (dist < K))
        by_d["interior"] = _nr(inter)
        results[mode] = {"edge": _nr(edge), "interior": _nr(inter), "pooled": _nr(np.ones_like(edge)),
                         "by_distance": by_d, "finite": bool(np.isfinite(d).all())}
        logger.info("EDGE MODE A/B on val | mode=%-6s | late NRMSE edge(<=K=%d) %.2f%% | interior %.2f%% | "
                    "pooled %.2f%% | ratio %.2f | by distance %s",
                    mode, K, results[mode]["edge"], results[mode]["interior"], results[mode]["pooled"],
                    results[mode]["edge"] / max(results[mode]["interior"], 1e-9),
                    " ".join("%s:%.2f%%" % (q, v) for q, v in by_d.items()))
    if not results:
        cfg.data._edge_mode_resolved = "none"
        return "none"
    _best_int = min(v["interior"] for v in results.values() if math.isfinite(v["interior"]))
    cands = [mm for mm, v in results.items()
             if v["finite"] and math.isfinite(v["edge"]) and v["interior"] <= 1.02 * _best_int]
    best = min(cands, key=lambda mm: results[mm]["edge"]) if cands else "none"
    cfg.data._edge_mode_resolved = best
    logger.info("EDGE MODE SELECTED on val: %s (edge %.2f%% / interior %.2f%%); applied to test and "
                "field.", best,
                results[best]["edge"] if best in results else float("nan"),
                results[best]["interior"] if best in results else float("nan"))
    try:
        _rep = run_dir / "reports"
        _rep.mkdir(parents=True, exist_ok=True)
        atomic_json_dump({"selected": best, "results": results, "K": K, "window": resolve_inference_window(cfg)},
                         _rep / "edge_mode_ab.json")
    except Exception as exc:
        logger.warning("edge A/B report not written: %r", exc)
    return best


def _paired_test_forward(model: "PEBRNet", cfg: Config, clean_cache: CleanCacheInfo,
                         noise_cache: NoiseCacheInfo, rows: np.ndarray,
                         logger: logging.Logger, edge_mode: Optional[str] = None,
                         tail_cut: Optional[int] = None, tail_protocol: str = "mask") -> np.ndarray:
    device = resolve_device(cfg.runtime.device)
    ds = BTEMDenoisingDataset(
        clean_cache.data_path, noise_cache.data_path,
        np.sort(np.asarray(rows, dtype=np.int64)), clean_cache.global_scale, cfg.data,
        seed=cfg.train.seed, samples=int(len(rows)), training=False,
        noise_block_rows=noise_cache.block_rows, gate_times=clean_cache.target_time,
        real_neighbor_rows=province_real_neighbor_rows(clean_cache, cfg),
        real_neighbor_geometry=province_neighbor_geometry(clean_cache, cfg),
        province_id=np.asarray(clean_cache.province_id),
        paired_noisy_path=clean_cache.paired_noisy_path,
        paired_background_path=clean_cache.paired_background_path,
        paired_event_path=clean_cache.paired_event_path,
    )
    ds.inject_weak_bodies = False
    _cut = None
    if tail_cut is not None and 0 < int(tail_cut) < int(cfg.data.target_gates):
        _cut = int(tail_cut)
        if str(tail_protocol) == "noise_only":
            ds.emit_neighbor_clean = True
    _rows_sorted = np.sort(np.asarray(rows, dtype=np.int64))
    _pid_all = np.asarray(clean_cache.province_id)
    _runs: List[Tuple[int, int]] = []
    if _rows_sorted.size:
        _s = 0
        for _i in range(1, int(_rows_sorted.size) + 1):
            if (_i == _rows_sorted.size or _rows_sorted[_i] != _rows_sorted[_i - 1] + 1
                    or _pid_all[_rows_sorted[_i]] != _pid_all[_rows_sorted[_i - 1]]):
                _runs.append((_s, _i))
                _s = _i
    _cap = max(64, min(512, int(cfg.train.batch_size)))
    _batches: List[Tuple[List[int], Optional[int]]] = []
    _by_len: Dict[int, List[Tuple[int, int]]] = {}
    for _r in _runs:
        _by_len.setdefault(_r[1] - _r[0], []).append(_r)
    for _ln, _rs in sorted(_by_len.items()):
        _per = max(1, _cap // _ln) if _ln <= _cap else 1
        for _j in range(0, len(_rs), _per):
            _grp = _rs[_j:_j + _per]
            _pos = [p for (a, b) in _grp for p in range(a, b)]
            if _ln > _cap or _ln <= 2:
                for _q in range(0, len(_pos), _cap):
                    _batches.append((_pos[_q:_q + _cap], None))
            else:
                _batches.append((_pos, int(_ln)))
    was = model.training
    model.eval()
    outs: List[np.ndarray] = []
    idxs: List[np.ndarray] = []
    _collate = torch.utils.data.default_collate
    with torch.no_grad():
        for _pos, _pl in _batches:
            batch = _collate([ds[p] for p in _pos])
            nb = batch.get("neighbors")
            geo = batch.get("neighbor_geometry")
            _xin = batch["noisy"]
            _om = None
            if _cut is not None:
                if str(tail_protocol) == "noise_only":
                    _xin = _xin.clone()
                    _xin[..., _cut:] = (batch["noisy"] - batch["clean"])[..., _cut:]
                    if nb is not None and "neighbors_clean" in batch:
                        nb = nb.clone()
                        nb[..., _cut:] = (batch["neighbors"] - batch["neighbors_clean"])[..., _cut:]
                else:
                    _om = torch.ones((int(_xin.shape[0]), 1, int(_xin.shape[-1])), dtype=torch.float32)
                    _om[..., _cut:] = 0.0
                    _om = _om.to(device)
            _W = resolve_inference_window(cfg)
            if _pl is not None and _W > 0 and int(_pl) > _W:
                _xb = _xin.to(device)
                _nbb = nb.to(device) if nb is not None else None
                _gb = geo.to(device) if geo is not None else None
                _db = batch["depth_norm"].to(device) if "depth_norm" in batch else None
                _parts = []
                for _q in range(0, int(_xb.shape[0]), int(_pl)):
                    _sl = slice(_q, _q + int(_pl))
                    _parts.append(sliding_window_forward(
                        model, _xb[_sl], _nbb[_sl] if _nbb is not None else None,
                        _gb[_sl] if _gb is not None else None, _db[_sl] if _db is not None else None,
                        _W, edge_mode=(edge_mode if edge_mode is not None else resolve_edge_mode(cfg)),
                        reflect_k=int(getattr(cfg.data, "edge_reflect_k", 0) or 0),
                        obs_mask=(_om[_sl] if _om is not None else None))["denoised"])
                o = {"denoised": torch.cat(_parts, 0)}
            else:
                o = model(_xin.to(device),
                          nb.to(device) if nb is not None else None,
                          geo.to(device) if geo is not None else None,
                          depth_norm=(batch["depth_norm"].to(device)
                                      if "depth_norm" in batch else None),
                          profile_len=_pl,
                          **({"obs_mask": _om} if _om is not None else {}))
            d = o["denoised"].detach().float().cpu().numpy()
            outs.append((d[:, 0] if d.ndim == 3 else d) * clean_cache.global_scale)
            idxs.append(batch["clean_row"].numpy())
    if was:
        model.train()
    order = np.concatenate(idxs)
    flat = np.concatenate(outs, axis=0)
    pos = {int(r): k for k, r in enumerate(order)}
    return np.stack([flat[pos[int(r)]] for r in rows], axis=0)


def export_torchscript(model: PEBRNet, run_dir: Path, cfg: Config, logger: logging.Logger) -> None:
    if not cfg.runtime.export_torchscript:
        return
    try:
        import copy

        model_cpu = copy.deepcopy(model).to("cpu").eval()
        if hasattr(model_cpu, "cfg"):
            model_cpu.cfg.informative_continuation = "off"
        _T = int(cfg.data.target_gates)
        _K = int(getattr(cfg.model, "num_neighbors", 8))

        class _ScriptWrapper(nn.Module):
            def __init__(self, m: nn.Module) -> None:
                super().__init__()
                self.m = m

            def forward(self, x: torch.Tensor, neighbors: torch.Tensor, geometry: torch.Tensor,
                        depth_norm: torch.Tensor) -> torch.Tensor:
                return self.m(x, neighbors, geometry, depth_norm=depth_norm, profile_len=None)["denoised"]

        _w = _ScriptWrapper(model_cpu).eval()
        _gs = model_cpu.gate_scale_norm.detach().float().view(1, 1, -1)
        _ex_x = (_gs * (1.0 + 0.05 * torch.randn(1, 1, _T))).abs()
        _ex_nb = (_gs * (1.0 + 0.05 * torch.randn(1, _K, _T))).abs()
        _ex_geo = model_cpu._default_neighbor_geometry(_ex_nb) if hasattr(model_cpu, "_default_neighbor_geometry") else torch.zeros(1, _K, 4)
        _ex_dn = torch.full((1, 1), 0.5)
        with torch.no_grad():
            _ref = _w(_ex_x, _ex_nb, _ex_geo, _ex_dn)
            traced = torch.jit.trace(_w, (_ex_x, _ex_nb, _ex_geo, _ex_dn), strict=False, check_trace=False)
            _out = traced(_ex_x, _ex_nb, _ex_geo, _ex_dn)
        _rel = float((_out - _ref).abs().max() / (_ref.abs().max() + 1e-30))
        if not math.isfinite(_rel) or _rel > 1e-4:
            raise RuntimeError("traced forward differs from eager (max rel %.3e)" % _rel)
        traced.save(str(run_dir / "checkpoints" / tagged_name("pebrnet_torchscript.pt")))
        del traced, model_cpu, _w
        logger.info("TorchScript model exported (single-station, single-pass forward with neighbours/geometry/depth; eager "
                    "vs traced max rel diff %.2e; the whole-area sliding window and the informative continuation "
                    "remain eager operations).", _rel)
    except Exception as exc:
        logger.warning("TorchScript export failed: %s", exc)
