# PEBR-Net -- prior- and evidence-bounded reconstruction of noise-buried late-time borehole TEM responses.
# MIT License, see LICENSE.
"""Training and evaluation datasets.

This file is executed by pebrnet/__init__.py into the package namespace, in the order listed there;
names defined in the other files of the package are available here without imports.
"""
from __future__ import annotations

class BTEMDenoisingDataset(Dataset):
    """Patches of neighbouring stations drawn from a paired dataset, with the training augmentation."""
    def __init__(
        self,
        clean_cache_path: str,
        noise_cache_path: str,
        split_indices: np.ndarray,
        global_scale: float,
        data_cfg: DataConfig,
        seed: int,
        samples: int,
        training: bool,
        noise_block_rows: Optional[List[int]] = None,
        gate_times: Optional[np.ndarray] = None,
        real_neighbor_rows: Optional[np.ndarray] = None,
        province_id: Optional[np.ndarray] = None,
        real_neighbor_geometry: Optional[np.ndarray] = None,
        paired_noisy_path: str = "",
        paired_background_path: str = "",
        paired_event_path: str = "",
        measured_noise_path: str = "",
        extra_clean_path: str = "",
        survey_profile_path: str = "",
    ) -> None:
        self.paired_noisy_path = str(paired_noisy_path or "")
        self.paired_background_path = str(paired_background_path or "")
        self.paired_event_path = str(paired_event_path or "")
        self._paired_noisy: Optional[np.ndarray] = None
        self._paired_bg: Optional[np.ndarray] = None
        self._paired_ev: Optional[np.ndarray] = None
        self._anom_pool: Optional[np.ndarray] = None
        self._anom_anchors: Optional[np.ndarray] = None
        self._hard_pool: Optional[np.ndarray] = None
        self._hard_anchors: Optional[np.ndarray] = None
        self._replay_src: Optional[np.ndarray] = None
        self._replay_runs: Optional[List[Tuple[int, int]]] = None
        self._prov_ids: Optional[np.ndarray] = (
            np.asarray(province_id, dtype=np.int64) if province_id is not None else None)
        self.measured_noise_path = str(measured_noise_path or "")
        self.extra_clean_path = str(extra_clean_path or "")
        self._extra_clean = None
        self.survey_profile_path = str(survey_profile_path or "")
        self._survey: Optional[Dict[str, np.ndarray]] = None
        self._mnoise: Optional[np.ndarray] = None
        self._mnoise_nbr: Optional[np.ndarray] = None
        self.clean_cache_path = clean_cache_path
        self.noise_cache_path = noise_cache_path
        self.noise_block_rows = list(noise_block_rows) if noise_block_rows else None
        self._block_bounds: Optional[np.ndarray] = None
        if gate_times is not None:
            t = np.asarray(gate_times, dtype=np.float64).reshape(-1)
            self._gate_times_s = t if t[-1] < 1.0 else t / 1000.0
        else:
            self._gate_times_s = np.geomspace(0.054e-3, 16e-3, data_cfg.target_gates)
        self.split_indices = np.asarray(split_indices, dtype=np.int64)
        self.global_scale = float(global_scale)
        self.cfg = data_cfg
        self.seed = int(seed)
        self.samples = int(min(samples, len(split_indices)) if not training else samples)
        self.training = bool(training)
        self._patch_runs: List[Tuple[int, int]] = []
        _P0 = int(getattr(self.cfg, "profile_patch_len", 0) or 0)
        if self.training and _P0 > 1:
            _si = np.sort(np.asarray(split_indices, dtype=np.int64))
            if _si.size:
                _start = _prev = int(_si[0])
                for _v in _si[1:]:
                    _v = int(_v)
                    if _v == _prev + 1:
                        _prev = _v
                        continue
                    if _prev - _start + 1 >= _P0:
                        self._patch_runs.append((_start, _prev))
                    _start = _prev = _v
                if _prev - _start + 1 >= _P0:
                    self._patch_runs.append((_start, _prev))
        self.split_indices_full = np.asarray(split_indices, dtype=np.int64)
        self.eval_block_len = 0
        _Pe = int(getattr(self.cfg, "profile_patch_len", 0) or 0)
        if (not self.training and _Pe > 1
                and self.samples < len(split_indices)
                and bool(getattr(self.cfg, "eval_profile_blocks", True))):
            _si = np.sort(np.asarray(split_indices, dtype=np.int64))
            _pid_e = (np.asarray(province_id, dtype=np.int64)
                      if province_id is not None else None)
            _anchors: List[int] = []
            if _si.size:
                _s0 = _p0 = int(_si[0])
                for _v in _si[1:]:
                    _v = int(_v)
                    if _v == _p0 + 1 and (_pid_e is None or _pid_e[_v] == _pid_e[_p0]):
                        _p0 = _v
                        continue
                    _anchors.extend(range(_s0, _p0 - _Pe + 2, _Pe))
                    _s0 = _p0 = _v
                _anchors.extend(range(_s0, _p0 - _Pe + 2, _Pe))
            if _anchors:
                _an = np.asarray(_anchors, dtype=np.int64)
                _nb = int(max(1, min(_an.size, max(1, self.samples) // _Pe)))
                _perm = np.random.default_rng(self.seed + 5_000_011).permutation(_an.size)
                _chosen = np.sort(_an[_perm[:_nb]])
                self.split_indices = np.concatenate(
                    [np.arange(a, a + _Pe, dtype=np.int64) for a in _chosen])
                self.samples = int(self.split_indices.size)
                self.eval_block_len = _Pe
            else:
                logging.getLogger(PROGRAM_NAME).warning(
                    "Evaluation split has no contiguous run of %d rows; falling back to the scattered "
                    "representative subsample (no lateral operator in evaluation).",
                    _Pe)
        if (not self.training and self.eval_block_len == 0
                and self.samples < len(self.split_indices)):
            perm = np.random.default_rng(self.seed + 5_000_011).permutation(len(self.split_indices))
            self.split_indices = np.sort(self.split_indices[perm[: self.samples]])
        self.epoch = 0
        self.emit_neighbor_clean = False
        self.inject_weak_bodies = True
        self._aug_lams: Optional[np.ndarray] = None
        self.real_neighbor_rows = real_neighbor_rows
        self._epoch_shared = torch.zeros(1, dtype=torch.int64)
        try:
            self._epoch_shared.share_memory_()
        except Exception:
            pass
        self.real_neighbor_geometry = real_neighbor_geometry
        self._mix_field_idx: Optional[np.ndarray] = None
        self._mix_forward_idx: Optional[np.ndarray] = None
        if training and province_id is not None:
            _si = np.asarray(split_indices, dtype=np.int64)
            _pid = np.asarray(province_id, dtype=np.int64)[_si]
            _aux = _pid < 0
            if bool(_aux.any()) and bool((~_aux).any()):
                self._mix_field_idx = _si[~_aux]
                self._mix_forward_idx = _si[_aux]
        def _runs_of(_idx: Optional[np.ndarray]) -> List[Tuple[int, int]]:
            out: List[Tuple[int, int]] = []
            if _idx is None:
                return out
            _fi = np.sort(np.asarray(_idx, dtype=np.int64))
            if not _fi.size:
                return out
            _s = _p2 = int(_fi[0])
            for _v in _fi[1:]:
                _v = int(_v)
                if _v == _p2 + 1:
                    _p2 = _v
                    continue
                if _p2 - _s + 1 >= _P0:
                    out.append((_s, _p2))
                _s = _p2 = _v
            if _p2 - _s + 1 >= _P0:
                out.append((_s, _p2))
            return out
        self._field_patch_runs: List[Tuple[int, int]] = []
        self._forward_patch_runs: List[Tuple[int, int]] = []
        if self.training and _P0 > 1:
            self._field_patch_runs = _runs_of(self._mix_field_idx)
            self._forward_patch_runs = _runs_of(self._mix_forward_idx)
        self._aug_mats: Optional[np.ndarray] = None
        if self.training and bool(getattr(data_cfg, "clean_augment", False)) and gate_times is not None:
            dex = float(data_cfg.clean_aug_tau_dex)
            lams = np.power(10.0, np.linspace(-dex, dex, 21))
            tt = np.asarray(gate_times, dtype=np.float64).reshape(-1)
            self._aug_lams = lams
            self._aug_mats = np.stack([loglog_dilation_matrix(tt, float(lam_v)) for lam_v in lams])
        self._clean: Optional[np.ndarray] = None
        self._noise: Optional[np.ndarray] = None
        if self.training and _P0 > 1 and self.paired_noisy_path and province_id is not None:
            _pid_all = np.asarray(province_id, dtype=np.int64)
            _si = np.sort(np.asarray(split_indices, dtype=np.int64))
            _runs: List[Tuple[int, int]] = []
            if _si.size:
                _s = _p = int(_si[0])
                for _v in _si[1:]:
                    _v = int(_v)
                    if _v == _p + 1 and _pid_all[_v] == _pid_all[_p]:
                        _p = _v
                        continue
                    if _p - _s + 1 >= _P0:
                        _runs.append((_s, _p))
                    _s = _p = _v
                if _p - _s + 1 >= _P0:
                    _runs.append((_s, _p))
            self._patch_runs = _runs
            self._field_patch_runs = list(_runs)
            self._forward_patch_runs = []
        if len(self.split_indices) == 0:
            raise ValueError("Dataset split contains zero indices.")

    def _ensure_open(self) -> None:
        if self.paired_noisy_path and self._paired_noisy is None:
            self._paired_noisy = np.load(self.paired_noisy_path, mmap_mode="r")
            if self.paired_background_path:
                self._paired_bg = np.load(self.paired_background_path, mmap_mode="r")
            if self.paired_event_path:
                self._paired_ev = np.load(self.paired_event_path, mmap_mode="r")
            if (self.training and self._paired_ev is not None
                    and float(getattr(self.cfg, "paired_anomaly_oversample", 0.0)) > 0.0):
                n_all = int(self._paired_ev.shape[0])
                flag = np.zeros(n_all, dtype=bool)
                step = 65536
                for a in range(0, n_all, step):
                    b = min(a + step, n_all)
                    flag[a:b] = np.asarray(self._paired_ev[a:b]).max(axis=1) >= 1.0
                pool = self.split_indices[flag[self.split_indices]]
                self._anom_pool = np.sort(pool.astype(np.int64))
                _P = int(getattr(self.cfg, "profile_patch_len", 0) or 0)
                anchors: List[int] = []
                if _P > 1:
                    csum = np.concatenate([[0], np.cumsum(flag.astype(np.int64))])
                    for a0, b0 in (getattr(self, "_field_patch_runs", None)
                                   or self._patch_runs or []):
                        for k in range(a0, b0 - _P + 2):
                            if csum[k + _P] - csum[k] > 0:
                                anchors.append(k)
                self._anom_anchors = np.asarray(anchors, dtype=np.int64)
            if (self.training
                    and float(getattr(self.cfg, "paired_hard_snr_oversample", 0.0)) > 0.0):
                if self._clean is None:
                    self._clean = np.load(self.clean_cache_path, mmap_mode="r")
                n_all = int(self._clean.shape[0])
                ls0 = int(self.cfg.late_start_index)
                snr = np.full(n_all, np.inf, dtype=np.float64)
                step = 65536
                for a in range(0, n_all, step):
                    b = min(a + step, n_all)
                    c = np.asarray(self._clean[a:b], dtype=np.float64)[:, ls0:]
                    x = np.asarray(self._paired_noisy[a:b], dtype=np.float64)[:, ls0:]
                    cr = np.sqrt(np.mean(c ** 2, axis=1))
                    nr = np.sqrt(np.mean((x - c) ** 2, axis=1))
                    snr[a:b] = 20.0 * np.log10(np.maximum(cr, 1e-300)
                                               / np.maximum(nr, 1e-300))
                tr = self.split_indices
                thr = float(np.percentile(snr[tr], 10.0))
                hflag = snr <= thr
                self._hard_pool = np.sort(tr[hflag[tr]].astype(np.int64))
                _Ph = int(getattr(self.cfg, "profile_patch_len", 0) or 0)
                hanch: List[int] = []
                if _Ph > 1:
                    hcs = np.concatenate([[0], np.cumsum(hflag.astype(np.int64))])
                    for a0, b0 in (getattr(self, "_field_patch_runs", None)
                                   or self._patch_runs or []):
                        for k in range(a0, b0 - _Ph + 2):
                            if hcs[k + _Ph] - hcs[k] > 0:
                                hanch.append(k)
                self._hard_anchors = np.asarray(hanch, dtype=np.int64)
            if (self.training
                    and (float(getattr(self.cfg, "paired_residual_replay", 0.0)) > 0.0
                         or float(getattr(self.cfg, "paired_alt_corruption", 0.0)) > 0.0)):
                _sidx = np.sort(np.asarray(self.split_indices, dtype=np.int64))
                self._replay_src = _sidx
                _pid_r = self._prov_ids
                _runs_r: List[Tuple[int, int]] = []
                if _sidx.size:
                    _s = _p = int(_sidx[0])
                    for _v in _sidx[1:]:
                        _v = int(_v)
                        _same = (_v == _p + 1) and (
                            _pid_r is None or _pid_r[_v] == _pid_r[_p])
                        if _same:
                            _p = _v
                            continue
                        _runs_r.append((_s, _p))
                        _s = _p = _v
                    _runs_r.append((_s, _p))
                self._replay_runs = _runs_r
            _val_proto = ((not self.training)
                             and bool(getattr(self.cfg,
                                              "val_contract_snr_protocol", True)))
            if ((self.training or _val_proto) and self.measured_noise_path
                    and float(getattr(self.cfg, "measured_noise_inject", 0.0)) > 0.0
                    and self._mnoise is None):
                self._mnoise = np.load(self.measured_noise_path, mmap_mode="r")
                if self.extra_clean_path:
                    self._extra_clean = np.load(
                        self.extra_clean_path, mmap_mode="r")
                _nbp = Path(self.measured_noise_path).with_name(
                    "measured_noise_neighbors.npy")
                if _nbp.is_file():
                    self._mnoise_nbr = np.load(_nbp, mmap_mode="r")
                if (self._survey is None and self.survey_profile_path
                        and Path(self.survey_profile_path).is_file()):
                    with np.load(self.survey_profile_path, allow_pickle=False) as _z:
                        self._survey = {_k: np.asarray(_z[_k]) for _k in _z.files}
        if self._clean is None:
            self._clean = np.load(self.clean_cache_path, mmap_mode="r")
        if self._noise is None:
            self._noise = np.load(self.noise_cache_path, mmap_mode="r")
            n = len(self._noise)
            if self.noise_block_rows and sum(self.noise_block_rows) == n:
                self._block_bounds = np.concatenate([[0], np.cumsum(self.noise_block_rows)])
            else:
                self._block_bounds = np.asarray([0, n], dtype=np.int64)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        _sh = getattr(self, "_epoch_shared", None)
        if _sh is not None:
            _sh[0] = int(epoch)

    def _epoch_now(self) -> int:
        _sh = getattr(self, "_epoch_shared", None)
        if _sh is not None:
            try:
                return int(_sh[0].item())
            except Exception:
                pass
        return int(getattr(self, "epoch", 0))

    def forward_fraction_now(self) -> float:
        """The forward mixture fraction for the current epoch."""
        f0 = float(getattr(self.cfg, "mix_forward_fraction", 0.0) or 0.0)
        f1 = getattr(self.cfg, "forward_fraction_final", None)
        n = int(getattr(self.cfg, "forward_fraction_total_epochs", 0) or 0)
        if f1 is None or n <= 1:
            return f0
        f1 = float(f1)
        p = min(max(float(self.epoch) / float(max(n - 1, 1)), 0.0), 1.0)
        return float(f1 + (f0 - f1) * 0.5 * (1.0 + math.cos(math.pi * p)))

    def __len__(self) -> int:
        return self.samples

    def profile_position(self, row: int) -> float:
        """Normalised position of a cache row inside its work area: (row - first_row_of_area) / (n_area - 1); 0.5
        when no area structure is known.
        """
        if self._prov_ids is None:
            return 0.5
        pos = getattr(self, "_prov_pos", None)
        if pos is None:
            ids = np.asarray(self._prov_ids).reshape(-1)
            n = int(ids.size)
            pos = np.full(n, 0.5, dtype=np.float32)
            if n > 1:
                brk = np.flatnonzero(ids[1:] != ids[:-1]) + 1
                starts = np.concatenate([[0], brk])
                ends = np.concatenate([brk, [n]])
                for s, e in zip(starts, ends):
                    if e - s > 1:
                        pos[s:e] = np.arange(e - s, dtype=np.float32) / float(e - s - 1)
            self._prov_pos = pos
        r = int(row)
        return float(pos[r]) if 0 <= r < pos.shape[0] else 0.5

    def batch_profile_len(self) -> int:
        """Block length every item of this dataset belongs to (0 = none)."""
        if self.training:
            _P = int(getattr(self.cfg, "profile_patch_len", 0) or 0)
            return _P if (_P > 1 and (getattr(self, "_field_patch_runs", None)
                                      or self._patch_runs)) else 0
        return int(getattr(self, "eval_block_len", 0) or 0)

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_clean"] = None
        state["_noise"] = None
        state["_paired_noisy"] = None
        state["_paired_bg"] = None
        state["_paired_ev"] = None
        state["_anom_pool"] = None
        state["_anom_anchors"] = None
        state["_hard_pool"] = None
        state["_hard_anchors"] = None
        state["_replay_src"] = None
        state["_replay_runs"] = None
        state["_mnoise"] = None
        state["_mnoise_nbr"] = None
        return state

    def _uniform_anchor(self, rng: np.random.Generator, runs: List[Tuple[int, int]], P: int,
                           epoch: int, patch_slot: int) -> int:
        key = (id(runs), len(runs), int(P))
        cache = getattr(self, "_anchor_cache", None)
        if cache is None or cache.get("key") != key:
            starts: List[np.ndarray] = []
            ra: List[np.ndarray] = []
            rb: List[np.ndarray] = []
            for a, b in runs:
                a = int(a)
                b = int(b)
                hi = int(b - P + 2)
                if hi <= a:
                    hi = a + 1
                n = hi - a
                starts.append(np.arange(a, hi, dtype=np.int64))
                ra.append(np.full(n, a, dtype=np.int64))
                rb.append(np.full(n, b, dtype=np.int64))
            anchors = np.concatenate(starts) if starts else np.zeros(0, dtype=np.int64)
            cache = {"key": key, "anchors": anchors,
                     "ra": (np.concatenate(ra) if ra else anchors), "rb": (np.concatenate(rb) if rb else anchors),
                     "perm": None, "cycle": -1}
            self._anchor_cache = cache
        N = int(cache["anchors"].size)
        if N <= 0:
            a, b = runs[int(rng.integers(0, len(runs)))]
            return _draw_anchor(rng, a, b, P, float(getattr(self.cfg, "edge_patch_fraction", 0.0)))
        per_epoch = max(1, int(self.samples) // max(1, int(P)))
        g = int(epoch) * per_epoch + int(patch_slot)
        cycle, pos = divmod(g, N)
        if cache["cycle"] != cycle or cache["perm"] is None:
            cache["perm"] = np.random.default_rng(int(self.seed) + 7_919 * (int(cycle) + 1)).permutation(N)
            cache["cycle"] = cycle
        k = int(cache["perm"][pos])
        anchor = int(cache["anchors"][k])
        p_edge = float(getattr(self.cfg, "edge_patch_fraction", 0.0))
        if p_edge > 0.0 and float(rng.random()) < p_edge:
            a, b = int(cache["ra"][k]), int(cache["rb"][k])
            anchor = a if float(rng.random()) < 0.5 else max(a, b - P + 1)
        return anchor

    def _cycle_index(self, key: str, N: int, epoch: int, slot: int) -> int:
        N = int(max(1, N))
        P = int(max(1, int(getattr(self.cfg, "profile_patch_len", 0) or 1)))
        per_epoch = max(1, int(self.samples) // P)
        g = int(epoch) * per_epoch + int(slot)
        cycle, pos = divmod(g, N)
        cache = getattr(self, "_cycle_cache", None)
        if cache is None:
            cache = {}
            self._cycle_cache = cache
        ent = cache.get(key)
        if ent is None or ent[0] != cycle:
            ent = (cycle, np.random.default_rng(int(self.seed) + 104_729 * (int(cycle) + 1) + sum(ord(c) for c in key)).permutation(N))
            cache[key] = ent
        return int(ent[1][pos])

    def anchor_cycle_summary(self) -> Tuple[int, int]:
        """(number of anchors, patches per epoch) for the log."""
        P = int(getattr(self.cfg, "profile_patch_len", 0) or 0)
        runs = getattr(self, "_field_patch_runs", None) or getattr(self, "_patch_runs", None) or []
        if P <= 1 or not runs:
            return 0, 0
        n = 0
        for a, b in runs:
            n += max(1, int(b) - int(P) + 2 - int(a))
        return int(n), int(max(1, int(self.samples) // P))

    def _sim_noise_window(self, anchor: int, P: int, rng: np.random.Generator) -> Optional[Tuple[int, np.ndarray]]:
        if self._clean is None:
            return None
        n_rows = int(self._clean.shape[0])
        a = int(np.clip(int(anchor), 0, n_rows - 1))
        b = int(np.clip(a + int(P) - 1, 0, n_rows - 1))
        lo, hi = a, b
        if self.real_neighbor_rows is not None:
            try:
                _nr = np.asarray(self.real_neighbor_rows[a:b + 1]).reshape(-1)
                if _nr.size:
                    lo = int(min(lo, int(_nr.min())))
                    hi = int(max(hi, int(_nr.max())))
            except Exception:
                pass
        cw = np.asarray(self._clean[lo:hi + 1], dtype=np.float64)
        t_ms = np.asarray(self._gate_times_s, dtype=np.float64).reshape(-1) * 1000.0
        if t_ms.size != cw.shape[1]:
            t_ms = np.geomspace(1.0, 20.0, cw.shape[1])
        lo_db, hi_db = (tuple(getattr(self.cfg, "sim_noise_snr_db", (-30.0, 10.0))) + (10.0,))[:2]
        regimes = tuple(getattr(self.cfg, "sim_noise_regimes", SIM_NOISE_REGIMES)) or SIM_NOISE_REGIMES
        snr = float(rng.uniform(float(lo_db), float(hi_db)))
        regime = str(regimes[int(rng.integers(0, len(regimes)))])
        sub = np.random.default_rng(int(rng.integers(0, 2 ** 62)))
        noise = simulate_late_noise(cw, t_ms, sub, snr_gate_columns(cw.shape[1]), snr, regime)
        return lo, noise

    def _survey_curve(self, s: int, phi: float) -> Tuple[np.ndarray, float]:
        sv = self._survey
        c = sv["clean"]
        n = int(c.shape[0])
        s = int(min(max(int(s), 0), n - 1))
        if phi <= 0.0 or s >= n - 1 or bool(sv["hold"][s + 1]):
            g = float(sv["gain"][s])
            return np.asarray(c[s], dtype=np.float64) * g, g
        a = np.asarray(c[s], dtype=np.float64)
        b = np.asarray(c[s + 1], dtype=np.float64)
        if bool((a > 0.0).all() and (b > 0.0).all()):
            y = np.exp((1.0 - phi) * np.log(a) + phi * np.log(b))
        else:
            y = (1.0 - phi) * a + phi * b
        em = int(sv["em"])
        ag = float(sv["align_gain"])
        g = ag
        if int(sv["station_mode"]) == 1:
            r = float(np.sqrt(np.mean(y[:em] ** 2)))
            g = ag * float(np.clip((float(sv["lib_rms"]) / max(r, 1e-300)) / max(ag, 1e-300), 1.0 / 50.0, 50.0))
        return y * g, g

    def _survey_bgratio(self, s: int, phi: float) -> np.ndarray:
        sv = self._survey
        r = sv["bgratio"]
        n = int(r.shape[0])
        s = int(min(max(int(s), 0), n - 1))
        if phi <= 0.0 or s >= n - 1 or bool(sv["hold"][s + 1]):
            return np.asarray(r[s], dtype=np.float64)
        return np.exp((1.0 - phi) * np.log(np.asarray(r[s], dtype=np.float64))
                      + phi * np.log(np.asarray(r[s + 1], dtype=np.float64)))

    def _survey_neighbor(self, s: int, j: int) -> int:
        sv = self._survey
        nbr = sv["nbr"]
        if int(nbr.shape[1]) == 0:
            return int(s)
        t = int(nbr[int(s), min(int(j), int(nbr.shape[1]) - 1)])
        return int(s) if bool(sv["hold"][t]) else t

    def _getitem_paired(self, item: int) -> Dict[str, torch.Tensor]:
        self._ensure_open()
        assert self._clean is not None and self._paired_noisy is not None
        epoch_term = self._epoch_now() if self.training else 0
        _P = int(getattr(self.cfg, "profile_patch_len", 0) or 0)
        patch_offset = 0
        if self.training and _P > 1:
            patch_offset = int(item) % _P
            rng = np.random.default_rng(
                self.seed + 1_000_003 * epoch_term + 97_409 * (int(item) // _P))
        else:
            rng = np.random.default_rng(
                self.seed + 1_000_003 * epoch_term + 97_409 * int(item))

        if self.training:
            _p_over = float(getattr(self.cfg, "paired_anomaly_oversample", 0.0))
            _p_hard = float(getattr(self.cfg, "paired_hard_snr_oversample", 0.0))
            _r0 = rng.random()
            _aimed = _p_over > 0.0 and _r0 < _p_over
            _hard = (not _aimed) and _p_hard > 0.0 and _r0 < _p_over + _p_hard
            _runs = getattr(self, "_field_patch_runs", None) or self._patch_runs
            if _P > 1 and _runs:
                if (_aimed and self._anom_anchors is not None
                        and self._anom_anchors.size > 0):
                    anchor = int(self._anom_anchors[
                        rng.integers(0, self._anom_anchors.size)])
                elif (_hard and self._hard_anchors is not None
                        and self._hard_anchors.size > 0):
                    anchor = int(self._hard_anchors[
                        rng.integers(0, self._hard_anchors.size)])
                else:
                    anchor = self._uniform_anchor(rng, _runs, _P, epoch_term, int(item) // _P)
                clean_idx = anchor + patch_offset
            else:
                if (_aimed and self._anom_pool is not None
                        and self._anom_pool.size > 0):
                    clean_idx = int(self._anom_pool[
                        rng.integers(0, self._anom_pool.size)])
                elif (_hard and self._hard_pool is not None
                        and self._hard_pool.size > 0):
                    clean_idx = int(self._hard_pool[
                        rng.integers(0, self._hard_pool.size)])
                else:
                    clean_idx = int(self.split_indices[
                        rng.integers(0, len(self.split_indices))])
        else:
            clean_idx = int(self.split_indices[item % len(self.split_indices)])

        clean = np.asarray(self._clean[clean_idx], dtype=np.float64)
        noisy = np.asarray(self._paired_noisy[clean_idx], dtype=np.float64)
        _zm = np.zeros(noisy.shape[0], dtype=bool)
        if bool(getattr(self.cfg, "impute_zero_gates", True)):
            noisy, _zm = impute_zero_gates_row(noisy)
        scaled_noise = noisy - clean
        _route: Tuple[Any, ...] = ("pair",)
        _arm_id = 0.0
        _surv = None
        _sim = None
        _f2 = float(getattr(self.cfg, "sim_noise_fraction", 0.0) or 0.0)
        if self.training and _f2 > 0.0 and rng.random() < _f2:
            _sim = self._sim_noise_window(int(clean_idx) - int(patch_offset), max(1, _P), rng)
            if _sim is not None:
                noisy = clean + _sim[1][int(clean_idx) - int(_sim[0])]
                scaled_noise = noisy - clean
                _route = ("sim", int(_sim[0]))
                _arm_id = 2.0
        _p_rep = float(getattr(self.cfg, "paired_residual_replay", 0.0))
        if _sim is not None:
            pass
        elif self.training and _p_rep > 0.0 and self._replay_src is not None \
                and self._replay_src.size > 1 and rng.random() < _p_rep:
            _plen = max(1, int(getattr(self.cfg, "profile_patch_len", 0) or 1))
            _rr = [r for r in (self._replay_runs or []) if r[1] - r[0] + 1 >= _plen]
            if not _rr:
                _rr = list(self._replay_runs or [])
            if _rr:
                _a_r, _b_r = _rr[int(rng.integers(0, len(_rr)))]
                _hi = max(_a_r, _b_r - _plen + 1)
                _anchor_r = int(rng.integers(_a_r, _hi + 1))
                _don = int(min(_anchor_r + patch_offset, _b_r))
            else:
                _don = int(self._replay_src[
                    int(rng.integers(0, self._replay_src.size))])
            _dc = np.asarray(self._clean[_don], dtype=np.float64)
            _dn = np.asarray(self._paired_noisy[_don], dtype=np.float64)
            if bool(getattr(self.cfg, "impute_zero_gates", True)):
                _dn, _ = impute_zero_gates_row(_dn)
            scaled_noise = _dn - _dc
            _s3 = 1.0
            _pcl2 = float(getattr(self.cfg, "paired_clean_slice", 0.0) or 0.0)
            _pcur = float(getattr(self.cfg, "replay_snr_curriculum", 0.0) or 0.0)
            _u3 = float(rng.random())
            _ls4 = int(self.cfg.late_start_index)
            _ef = float(getattr(self.cfg, "deep_negative_early_floor_db", 12.0))
            _v = float(rng.random())
            _cr3 = float(np.sqrt(np.mean(clean[_ls4:] ** 2)))
            _nr3 = float(np.sqrt(np.mean((noisy - clean)[_ls4:] ** 2)))
            _nat = (20.0 * math.log10(_cr3 / _nr3) if (_cr3 > 1e-300 and _nr3 > 1e-300) else float("nan"))
            _drop = float(getattr(self.cfg, "replay_snr_drop_db", 0.0) or 0.0)
            if _pcl2 > 0.0 and _u3 < _pcl2:
                _s3, _ = late_scale_with_early_guard(clean, scaled_noise, _ls4, 80.0, _ef)
            elif _pcur > 0.0 and _u3 < _pcl2 + _pcur:
                _lo3 = float(self.cfg.measured_noise_snr_min_db)
                _hi4 = float(self.cfg.measured_noise_snr_max_db)
                if _drop > 0.0 and math.isfinite(_nat):
                    _tgt3 = max(_lo3, _nat - _drop * _v)
                else:
                    _foc2 = float(getattr(self.cfg, "measured_noise_snr_low_focus", 0.5))
                    _cap2 = min(10.0, _hi4)
                    _psz2 = float(getattr(self.cfg, "measured_noise_subzero_probability", 0.0) or 0.0)
                    _szmin2 = float(getattr(self.cfg, "measured_noise_subzero_min_db", -10.0))
                    if _psz2 > 0.0 and _szmin2 < _lo3 and rng.random() < _psz2:
                        _tgt3 = float(rng.uniform(_szmin2, _lo3))
                    elif _cap2 > _lo3 and rng.random() < _foc2:
                        _tgt3 = float(rng.uniform(_lo3, _cap2))
                    else:
                        _tgt3 = float(rng.uniform(_lo3, _hi4))
                _s3, _ = late_scale_with_early_guard(clean, scaled_noise, _ls4, _tgt3, _ef)
            elif bool(getattr(self.cfg, "replay_match_natural_snr", True)) and math.isfinite(_nat):
                _s3, _ = late_scale_with_early_guard(clean, scaled_noise, _ls4, _nat, _ef)
            _s3 = float(_s3) if math.isfinite(float(_s3)) else 1.0
            scaled_noise = scaled_noise * _s3
            noisy = clean + scaled_noise
            _route = ("replay", _don, _s3)
            _arm_id = 1.0
        elif ((not self.training) and self._mnoise is not None
              and bool(getattr(self.cfg, "val_contract_snr_protocol", True))
              and float(getattr(self.cfg, "measured_noise_inject", 0.0)) > 0.0):
            _S2 = int(self._mnoise.shape[1])
            _st2 = min(int(clean_idx) % _S2, _S2 - 1)
            _C2 = int(self._mnoise.shape[0])
            if bool(getattr(self.cfg, "measured_noise_component_mix", True))                     and _C2 > 2:
                _nl2 = np.zeros(int(self._mnoise.shape[2]), dtype=np.float64)
                _h = (int(clean_idx) * 2246822519) & 0xFFFFFFFF
                _K = _C2 - 1
                _e = np.zeros(_K, dtype=np.float64)
                for _k in range(_K):
                    _h = (_h * 2654435761 + 97) & 0xFFFFFFFF
                    _u1 = (((_h >> 7) % 65521) + 1) / 65522.0
                    _h = (_h * 2654435761 + 97) & 0xFFFFFFFF
                    _u2 = (((_h >> 7) % 65521) + 1) / 65522.0
                    _e[_k] = -np.log(_u1) - np.log(_u2)
                _w2 = _e / max(float(_e.sum()), 1e-30)
                _sel2 = list(range(1, _C2))
                _gain = []
                for _k in range(_K):
                    _g = float(np.sqrt(_K * _w2[_k]))
                    _gain.append(_g)
                    _nl2 += _g * np.asarray(
                        self._mnoise[1 + _k, _st2], dtype=np.float64)
            else:
                _nl2 = np.asarray(self._mnoise[0, _st2], dtype=np.float64)
                _sel2 = [0]
                _gain = [1.0]
            _lo2 = float(self.cfg.contract_test_snr_db_range[0])
            _hi3 = float(self.cfg.contract_test_snr_db_range[1])
            _u = float(((int(clean_idx) * 2654435761) % 1024)) / 1023.0
            _tgt2 = _lo2 + (_hi3 - _lo2) * _u
            _ls2 = int(self.cfg.late_start_index)
            _cr2 = float(np.sqrt(np.mean(clean[_ls2:] ** 2)))
            _nr2 = float(np.sqrt(np.mean(_nl2[_ls2:] ** 2)))
            _arm_id = 2.0
            _s2 = 0.0
            if _nr2 > 1e-300 and _cr2 > 1e-300:
                _s2, _tgt2 = late_scale_with_early_guard(
                    clean, _nl2, int(self.cfg.late_start_index), _tgt2,
                    float(getattr(self.cfg, "deep_negative_early_floor_db", 12.0)))
                scaled_noise = _nl2 * _s2
            else:
                scaled_noise = (np.asarray(self._paired_noisy[clean_idx],
                                           dtype=np.float64) - clean
                                if self._paired_noisy is not None
                                else np.zeros_like(clean))
            noisy = clean + scaled_noise
            _route = ("library", _st2, _sel2, float(_s2), tuple(_gain))
            _arm_id = 2.0
        elif (self.training and self._mnoise is not None
              and float(getattr(self.cfg, "measured_noise_inject", 0.0)) > 0.0
              and rng.random() < min(1.0, float(self.cfg.measured_noise_inject)
                                     / max(1e-9, 1.0 - min(0.999, float(getattr(
                                         self.cfg, "paired_residual_replay", 0.0)))))):
            _C, _S, _G = self._mnoise.shape
            _external_clean = False
            _sv = self._survey
            _f = float(getattr(self.cfg, "survey_profile_fraction", 0.0) or 0.0)
            if (_sv is not None and self.training and _f > 0.0 and int(_sv["starts"].size) > 0
                    and rng.random() < _f):
                _Ns = int(_sv["clean"].shape[0])
                _pl = max(1, int(getattr(self.cfg, "profile_patch_len", 0) or 1))
                _a = int(_sv["starts"][int(rng.integers(0, int(_sv["starts"].size)))])
                _ss = min(_a + patch_offset, _Ns - 1)
                _phi = (float(rng.random())
                           if rng.random() < float(getattr(self.cfg, "survey_shift_fraction", 0.0) or 0.0) else 0.0)
                if _phi > 0.0 and (_a + _pl >= _Ns
                                      or bool(_sv["hold"][min(_a + _pl, _Ns - 1)])):
                    _phi = 0.0
                _nat2 = bool(rng.random() < float(getattr(self.cfg, "survey_natural_fraction", 0.5)))
                _jdb = float(getattr(self.cfg, "survey_scale_jitter_db", 0.0) or 0.0)
                _jit = float(10.0 ** (float(rng.uniform(-_jdb, _jdb)) / 20.0)) if _jdb > 0.0 else 1.0
                clean, _g3 = self._survey_curve(_ss, _phi)
                _surv = (int(_ss), float(_phi), bool(_nat2), float(_jit), float(_g3))
                _external_clean = True
            if (self._extra_clean is not None and self.training and _surv is None
                    and rng.random() < float(getattr(
                        self.cfg, "extra_clean_fraction", 0.25))):
                _ei = self._cycle_index("extra_clean", int(self._extra_clean.shape[0]), epoch_term,
                                           (int(item) // _P) if _P > 1 else int(item))
                clean = np.asarray(self._extra_clean[_ei], dtype=np.float64)
                if _sv is not None and int(_sv["extra_align"]) == 1:
                    _r = float(np.sqrt(np.mean(clean[:int(_sv["em"])] ** 2)))
                    if _r > 1e-300:
                        clean = clean * (float(_sv["lib_rms"]) / _r)
                _external_clean = True
            _plen = max(1, int(getattr(self.cfg, "profile_patch_len", 0) or 1))
            _s0 = int(rng.integers(0, max(1, _S - _plen + 1)))
            _st = min(_s0 + patch_offset, _S - 1)
            if bool(getattr(self.cfg, "measured_noise_component_mix", True)) and _C > 2:
                _sel = rng.random(_C - 1) < 0.5
                if not _sel.any():
                    _sel[int(rng.integers(0, _C - 1))] = True
                _nl = np.zeros(_G, dtype=np.float64)
                _sel_idx = [int(_ci) + 1 for _ci in np.nonzero(_sel)[0]]
                _K2 = _C - 1
                _w3 = rng.gamma(2.0, 1.0, size=_K2)
                _w3 = _w3 / max(float(_w3.sum()), 1e-30)
                _sel_idx = list(range(1, _C))
                _gain2 = []
                for _k2, _ci in enumerate(_sel_idx):
                    _g2 = float(np.sqrt(_K2 * _w3[_k2]))
                    if bool(getattr(self.cfg, "measured_noise_random_sign", True)):
                        _g2 = -_g2 if rng.random() < 0.5 else _g2
                    _gain2.append(_g2)
                    _nl += _g2 * np.asarray(
                        self._mnoise[_ci, _st], dtype=np.float64)
            else:
                _sel_idx = [0]
                _gain2 = [1.0]
                _nl = np.asarray(self._mnoise[0, _st], dtype=np.float64)
            _ls = int(self.cfg.late_start_index)
            if bool(getattr(self.cfg, "injection_snr_calibrated_on_late_window", False)):
                _cr = float(np.sqrt(np.mean(clean[_ls:] ** 2)))
                _nr = float(np.sqrt(np.mean(_nl[_ls:] ** 2)))
            else:
                _cr = float(np.sqrt(np.mean(clean ** 2)))
                _nr = float(np.sqrt(np.mean(_nl ** 2)))
            if _nr > 1e-300 and _cr > 1e-300:
                _lo_n = float(self.cfg.measured_noise_snr_min_db)
                _hi_n = float(self.cfg.measured_noise_snr_max_db)
                _foc = float(getattr(self.cfg, "measured_noise_snr_low_focus", 0.5))
                _cap = min(10.0, _hi_n)
                _psz = float(getattr(self.cfg, "measured_noise_subzero_probability", 0.0) or 0.0)
                _szmin = float(getattr(self.cfg, "measured_noise_subzero_min_db", -10.0))
                if self.training and _psz > 0.0 and _szmin < _lo_n and rng.random() < _psz:
                    _tgt = float(rng.uniform(_szmin, _lo_n))
                elif _cap > _lo_n and rng.random() < _foc:
                    _tgt = float(rng.uniform(_lo_n, _cap))
                else:
                    _tgt = float(rng.uniform(_lo_n, _hi_n))
                _s, _tgt = late_scale_with_early_guard(
                    clean, _nl, int(self.cfg.late_start_index), _tgt,
                    float(getattr(self.cfg, "deep_negative_early_floor_db", 12.0)))
                scaled_noise = _nl * _s
                noisy = clean + scaled_noise
                _route = ("library", _st, _sel_idx, _s, tuple(_gain2))
                _arm_id = 2.0
                if _surv is not None and _surv[2]:
                    _nl = np.asarray(_sv["noise"][int(_surv[0])], dtype=np.float64) * float(_surv[4])
                    _s = float(_surv[3])
                    scaled_noise = _nl * _s
                    noisy = clean + scaled_noise
                    _tgt = float(20.0 * math.log10(max(float(np.sqrt(np.mean(clean[_ls:] ** 2))), 1e-300)
                                                   / max(float(np.sqrt(np.mean(scaled_noise[_ls:] ** 2))), 1e-300)))
                    _sel_idx = [0]
                    _gain2 = [1.0]
                    _route = ("survey", int(_surv[0]), float(_s))
                _pcl = float(getattr(self.cfg, "paired_clean_slice", 0.0) or 0.0)
                if _pcl > 0.0 and rng.random() < _pcl:
                    _s80, _tgt = late_scale_with_early_guard(
                        clean, _nl, int(self.cfg.late_start_index), 80.0,
                        float(getattr(self.cfg, "deep_negative_early_floor_db", 12.0)))
                    _s = float(_s80)
                    _tgt = 80.0
                    scaled_noise = _nl * _s
                    noisy = clean + scaled_noise
                    _route = ("library", _st, _sel_idx, _s, tuple(_gain2))
        if bool(locals().get("_external_clean", False)):
            bg = clean.copy()
        else:
            bg = (np.asarray(self._paired_bg[clean_idx], dtype=np.float64)
                  if self._paired_bg is not None else clean)
        ev_np = (np.zeros(clean.shape[0], dtype=np.float32)
                 if bool(locals().get("_external_clean", False))
                 else (np.asarray(self._paired_ev[clean_idx], dtype=np.float32)
                       if self._paired_ev is not None
                       else np.zeros(clean.shape[0], dtype=np.float32)))
        has_anom = float(ev_np.max() >= 1.0) if ev_np.size else 0.0
        if _surv is not None:
            bg = clean * self._survey_bgratio(int(_surv[0]), float(_surv[1]))
            _thr = float(getattr(self.cfg, "paired_event_threshold", 0.15))
            ev_np = np.clip(np.abs(clean - bg) / np.maximum(np.abs(bg), 1.0e-30) / max(_thr, 1.0e-9),
                            0.0, 1.0).astype(np.float32)
            ev_np[:int(self.cfg.late_start_index)] = 0.0
            has_anom = float(ev_np.max() >= 1.0)

        _wb = None
        _wbp = float(getattr(self.cfg, "weak_body_injection_prob", 0.0) or 0.0)
        _ext = bool(locals().get("_external_clean", False))
        _eval_inject_ok = (bool(getattr(self.cfg, "weak_body_eval_injection", False))
                           and int(getattr(self, "eval_block_len", 0) or 0) > 1
                           and bool(getattr(self, "inject_weak_bodies", True)))
        if (_wbp > 0.0 and not _ext
                and bool(getattr(self, "inject_weak_bodies", True))
                and (self.training or _eval_inject_ok)):
            if self.training and _P > 1:
                _wrng = np.random.default_rng(
                    self.seed + 5_550_001 + 1_000_003 * epoch_term + 97_409 * (int(item) // _P))
                _wpos, _wP = patch_offset, _P
            elif (not self.training) and int(getattr(self, "eval_block_len", 0) or 0) > 1:
                _Pe = int(self.eval_block_len)
                _wrng = np.random.default_rng(self.seed + 5_550_001 + 97_409 * (int(item) // _Pe))
                _wpos, _wP = int(item) % _Pe, _Pe
            else:
                _wrng = np.random.default_rng(self.seed + 5_550_001 + 97_409 * int(item))
                _wpos, _wP = 0, 1
            if int(_wP) > 9:
                _wbp = min(0.9, float(_wbp) * float(_wP) / 9.0)
            if _wrng.random() < _wbp:
                _lo = math.log10(max(1e-6, float(getattr(self.cfg, "weak_body_contrast_min", 0.005))))
                _hi = math.log10(max(1e-6, float(getattr(self.cfg, "weak_body_contrast_max", 0.20))))
                _A = 10.0 ** _wrng.uniform(min(_lo, _hi), max(_lo, _hi))
                if _wrng.random() >= float(getattr(self.cfg, "weak_body_positive_fraction", 0.7)):
                    _A = -_A
                _c = (_wrng.uniform(0.0, max(0.0, _wP - 1.0)) if _wP > 1 else _wrng.uniform(-4.0, 4.0))
                _w = _wrng.uniform(float(getattr(self.cfg, "weak_body_width_min", 1.0)),
                                   float(getattr(self.cfg, "weak_body_width_max", 4.0)))
                _g0 = max(0, int(self.cfg.late_start_index) // 2)
                _T = int(clean.shape[0])
                _env = np.clip((np.arange(_T, dtype=np.float64) - _g0) / max(1.0, float(_T - 1 - _g0)), 0.0, 1.0)
                _wb = (float(_A), float(_c), float(max(_w, 0.3)), _env, float(_wpos))
                _f0 = _A * math.exp(-0.5 * ((_wpos - _c) / max(_w, 0.3)) ** 2)
                _lift0 = 1.0 + _f0 * _env
                clean = clean * _lift0
                noisy = clean + scaled_noise
                _hit = (np.abs(_f0 * _env) >= 0.005).astype(np.float32)
                ev_np = np.maximum(ev_np, _hit)
                has_anom = float(ev_np.max() >= 1.0) if ev_np.size else 0.0

        _cut = None
        _p = float(getattr(self.cfg, "tail_loss_prob", 0.0) or 0.0)
        if self.training and _p > 0.0 and rng.random() < _p:
            _fr = tuple(getattr(self.cfg, "tail_loss_fraction", (0.05, 0.75))) + (0.75,)
            _T2 = int(clean.shape[0])
            _k4 = int(round(float(rng.uniform(float(_fr[0]), float(_fr[1]))) * _T2))
            _cut = int(np.clip(_T2 - _k4, 3, _T2 - 1))
        _ls3 = int(self.cfg.late_start_index)
        _crms = float(np.sqrt(np.mean(clean[_ls3:] ** 2)))
        _nrms = float(np.sqrt(np.mean(scaled_noise[_ls3:] ** 2)))
        snr_db = float(20.0 * np.log10(max(_crms, 1e-300) / max(_nrms, 1e-300)))
        if not np.isfinite(snr_db):
            snr_db = 0.0
        snr_db = min(snr_db, 80.0)

        q_target = compute_reliability_target_np(
            noisy, clean, kappa=self.cfg.reliability_kappa,
            window=self.cfg.reliability_window,
        )
        if bool(np.any(locals().get("_zm", False))):
            q_target = np.asarray(q_target, dtype=np.float32).copy()
            q_target[np.asarray(_zm, dtype=bool)] = 0.0
        scale = self.global_scale
        sample = {
            "noisy": torch.from_numpy((noisy / scale).astype(np.float32))[None, :],
            "clean": torch.from_numpy((clean / scale).astype(np.float32))[None, :],
            "noise": torch.from_numpy((scaled_noise / scale).astype(np.float32))[None, :],
            "q_target": torch.from_numpy(q_target)[None, :],
            "snr_db": torch.tensor(snr_db, dtype=torch.float32),
            "clean_index": torch.tensor(clean_idx, dtype=torch.int64),
            "noise_index": torch.tensor(clean_idx, dtype=torch.int64),
            "clean_bg": torch.from_numpy((bg / scale).astype(np.float32))[None, :],
            "noisy_bg": torch.from_numpy(
                ((bg + scaled_noise) / scale).astype(np.float32))[None, :],
            "has_anomaly": torch.tensor(has_anom, dtype=torch.float32),
            "depth_norm": torch.tensor(
                [((float(_surv[0]) + float(_surv[1])) / float(max(1, int(self._survey["clean"].shape[0]) - 1)))
                 if _surv is not None
                 else (0.5 if bool(locals().get("_external_clean", False))
                       else self.profile_position(clean_idx))],
                dtype=torch.float32),
            "profile_len": torch.tensor(int(self.batch_profile_len()), dtype=torch.int64),
            "event_mask": torch.from_numpy(
                np.ascontiguousarray(ev_np).copy())[None, :],
            "event_label": torch.tensor(has_anom, dtype=torch.float32),
            "is_forward": torch.tensor(0.0, dtype=torch.float32),
            "inject_snr_db": torch.tensor(
                (_tgt if (self.training and _arm_id == 2.0 and _sim is None) else float("nan"))
                if self.training else float("nan"), dtype=torch.float32),
            "sim_row": torch.tensor(1.0 if (self.training and _sim is not None) else 0.0, dtype=torch.float32),
            "arm_id": torch.tensor(_arm_id if self.training else 0.0,
                                   dtype=torch.float32),
            "contract_exempt": torch.tensor(
                1.0 if (self.training and _arm_id == 2.0
                        and bool(getattr(self.cfg, "library_contract_exempt", True))) else 0.0,
                dtype=torch.float32),
            "survey_row": torch.tensor((1.0 if _surv[2] else 2.0) if _surv is not None else 0.0,
                                       dtype=torch.float32),
        }
        _alt_on = False
        _alt_donor_row = -1
        if (self.training and self._replay_src is not None
                and self._replay_src.size > 1
                and float(getattr(self.cfg, "paired_alt_corruption", 0.0)) > 0.0
                and rng.random() < float(self.cfg.paired_alt_corruption)):
            _alt_on = True
        if self.training:
            _plen2 = max(1, int(getattr(self.cfg, "profile_patch_len", 0) or 1))
            _rr2 = [r for r in (self._replay_runs or []) if r[1] - r[0] + 1 >= _plen2]
            if not _rr2:
                _rr2 = list(self._replay_runs or [])
            if _alt_on and _rr2:
                _a2, _bb2 = _rr2[int(rng.integers(0, len(_rr2)))]
                _hi2 = max(_a2, _bb2 - _plen2 + 1)
                _anch2 = int(rng.integers(_a2, _hi2 + 1))
                _d2 = int(min(_anch2 + patch_offset, _bb2))
            else:
                _src2 = (self._replay_src if self._replay_src is not None
                         else np.asarray(self.split_indices, dtype=np.int64))
                _d2 = int(_src2[min(patch_offset, _src2.size - 1)])
            _alt = (clean + (np.asarray(self._paired_noisy[_d2], dtype=np.float64)
                             - np.asarray(self._clean[_d2], dtype=np.float64))
                    ) if _alt_on else noisy
            sample["noisy_alt"] = torch.from_numpy(
                (_alt / scale).astype(np.float32))[None, :]
            sample["has_alt"] = torch.tensor(1.0 if _alt_on else 0.0, dtype=torch.float32)
            _alt_donor_row = int(_d2) if _alt_on else -1

        k = int(self.cfg.num_neighbors)
        if k > 0:
            gates = clean.shape[0]
            _real = (self.real_neighbor_rows is not None
                     and bool(getattr(self.cfg, "paired_real_neighbors", True)))
            add_stack = bool(getattr(self.cfg, "coherent_stack_slot", False) and _real)
            k_slots = k + (1 if add_stack else 0)
            nb = np.empty((k_slots, gates), dtype=np.float32)
            nb_bg = np.empty((k_slots, gates), dtype=np.float32)
            nb_clean_out = (np.empty((k_slots, gates), dtype=np.float32)
                            if self.emit_neighbor_clean else None)
            geo = np.zeros((k_slots, NEIGHBOR_GEOMETRY_FEATURES), dtype=np.float32)
            for j in range(k):
                offset = ((j // 2) + 1) * (1 if j % 2 == 0 else -1)
                if _real and self.real_neighbor_geometry is not None:
                    _so = float(self.real_neighbor_geometry[clean_idx, j, 0])
                    _ir = float(self.real_neighbor_geometry[clean_idx, j, 1])
                elif _real:
                    _so, _ir = float(offset), 1.0
                else:
                    _so, _ir = float(offset), 0.0
                geo[j, 0] = _so
                geo[j, 1] = abs(_so)
                geo[j, 2] = _ir
                nb_row = int(self.real_neighbor_rows[clean_idx, j]) if _real else clean_idx
                _nbs = -1
                _nbg = 1.0
                if _surv is not None:
                    _nbs = self._survey_neighbor(int(_surv[0]), j)
                    nb_clean, _nbg = self._survey_curve(_nbs, float(_surv[1]))
                    if int(self._survey["geo"].shape[1]) > j:
                        geo[j, 0:3] = self._survey["geo"][int(_surv[0]), j, 0:3]
                else:
                    nb_clean = (clean.copy()
                                if bool(locals().get("_external_clean", False))
                                else np.asarray(self._clean[nb_row],
                                                dtype=np.float64))
                nb_noisy = np.asarray(self._paired_noisy[nb_row], dtype=np.float64)
                if bool(getattr(self.cfg, "impute_zero_gates", True)):
                    nb_noisy, _ = impute_zero_gates_row(nb_noisy)
                if _wb is not None and _real:
                    _Aj, _cj, _wj, _envj, _posj = _wb
                    _fj = _Aj * math.exp(-0.5 * ((_posj + (int(nb_row) - int(clean_idx)) - _cj) / _wj) ** 2)
                    _liftj = 1.0 + _fj * _envj
                    nb_noisy = nb_noisy + nb_clean * (_liftj - 1.0)
                    nb_clean = nb_clean * _liftj
                _rowstep = int(nb_row) - int(clean_idx)
                if _route[0] == "replay":
                    _nd = int(np.clip(int(_route[1]) + _rowstep,
                                      0, int(self._clean.shape[0]) - 1))
                    if (self._prov_ids is not None
                            and self._prov_ids[_nd] != self._prov_ids[int(_route[1])]):
                        _nd = int(_route[1])
                    nb_noisy = nb_clean + (
                        np.asarray(self._paired_noisy[_nd], dtype=np.float64)
                        - np.asarray(self._clean[_nd], dtype=np.float64)) * (float(_route[2]) if len(_route) > 2 else 1.0)
                elif _route[0] == "sim" and _sim is not None:
                    _k3 = int(nb_row) - int(_sim[0])
                    _k3 = int(np.clip(_k3, 0, int(_sim[1].shape[0]) - 1))
                    nb_noisy = nb_clean + _sim[1][_k3]
                elif _route[0] == "survey" and _surv is not None:
                    nb_noisy = nb_clean + (np.asarray(self._survey["noise"][_nbs], dtype=np.float64)
                                           * float(_nbg) * float(_route[2]))
                elif _route[0] == "library" and self._mnoise is not None:
                    _Sl = int(self._mnoise.shape[1])
                    if self._mnoise_nbr is not None and j < int(self._mnoise_nbr.shape[1]):
                        _nst = int(self._mnoise_nbr[int(_route[1]), j])
                    else:
                        _nst = int(np.clip(int(_route[1]) + _rowstep, 0, _Sl - 1))
                    _nnl = np.zeros(nb_clean.shape[0], dtype=np.float64)
                    _gains = _route[4] if len(_route) > 4 else [1.0] * len(_route[2])
                    for _ci, _gk in zip(_route[2], _gains):
                        _nnl += float(_gk) * np.asarray(self._mnoise[int(_ci), _nst], dtype=np.float64)
                    nb_noisy = nb_clean + _nnl * float(_route[3])
                nb_bgv = ((nb_clean * self._survey_bgratio(_nbs, float(_surv[1])))
                          if _surv is not None
                          else (nb_clean if bool(locals().get("_external_clean", False))
                                else (np.asarray(self._paired_bg[nb_row], dtype=np.float64)
                                      if self._paired_bg is not None else nb_clean)))
                if (self.training and _real
                        and rng.random() < float(getattr(self.cfg, "neighbor_channel_dropout", 0.0))):
                    _res = nb_noisy - nb_clean
                    nb[j] = (_res / scale).astype(np.float32)
                    nb_bg[j] = nb[j]
                else:
                    nb[j] = (nb_noisy / scale).astype(np.float32)
                    nb_bg[j] = ((nb_bgv + (nb_noisy - nb_clean)) / scale).astype(np.float32)
                if nb_clean_out is not None:
                    nb_clean_out[j] = (nb_clean / scale).astype(np.float32)
                if self.training:
                    if j == 0:
                        _nbn_sum = (nb_noisy - nb_clean).astype(np.float64)
                    else:
                        _nbn_sum += (nb_noisy - nb_clean)
            if add_stack:
                nb[k] = (nb[:k].sum(axis=0) / float(k)).astype(np.float32)
                nb_bg[k] = (nb_bg[:k].sum(axis=0) / float(k)).astype(np.float32)
                if nb_clean_out is not None:
                    nb_clean_out[k] = (
                        nb_clean_out[:k].sum(axis=0) / float(k)).astype(np.float32)
                geo[k, 3] = 1.0
            _adr = int(_alt_donor_row)
            if _adr >= 0:
                nb_alt = np.empty_like(nb)
                for j in range(k):
                    _nr = int(self.real_neighbor_rows[clean_idx, j]) if _real else clean_idx
                    _step = _nr - int(clean_idx)
                    _dn2 = int(np.clip(_adr + _step, 0, int(self._clean.shape[0]) - 1))
                    if (self._prov_ids is not None
                            and self._prov_ids[_dn2] != self._prov_ids[_adr]):
                        _dn2 = _adr
                    if _surv is not None:
                        _nc2 = self._survey_curve(self._survey_neighbor(int(_surv[0]), j),
                                                      float(_surv[1]))[0]
                    else:
                        _nc2 = (clean.copy() if bool(locals().get("_external_clean", False))
                                else np.asarray(self._clean[_nr], dtype=np.float64))
                    if _wb is not None and _real:
                        _Aa, _ca, _wa, _enva, _posa = _wb
                        _nc2 = _nc2 * (1.0 + _Aa * math.exp(
                            -0.5 * ((_posa + (int(_nr) - int(clean_idx)) - _ca) / _wa) ** 2) * _enva)
                    nb_alt[j] = ((_nc2
                                  + (np.asarray(self._paired_noisy[_dn2], dtype=np.float64)
                                     - np.asarray(self._clean[_dn2], dtype=np.float64)))
                                 / scale).astype(np.float32)
                if add_stack:
                    nb_alt[k] = (nb_alt[:k].sum(axis=0)
                                 / float(k)).astype(np.float32)
                sample["neighbors_alt"] = torch.from_numpy(nb_alt)
            elif self.training and "noisy_alt" in sample:
                sample["neighbors_alt"] = torch.from_numpy(nb.copy())
            sample["neighbor_geometry"] = torch.from_numpy(geo)
            if self.training and k > 0:
                sample["neighbor_noise_mean"] = torch.from_numpy(
                    (_nbn_sum / (float(k) * scale)).astype(np.float32))[None, :]
            sample["neighbors"] = torch.from_numpy(nb)
            sample["neighbors_bg"] = torch.from_numpy(nb_bg)
            if nb_clean_out is not None:
                sample["neighbors_clean"] = torch.from_numpy(nb_clean_out)
        sample["clean_row"] = torch.tensor(int(clean_idx), dtype=torch.long)
        if self.training and float(getattr(self.cfg, "tail_loss_prob", 0.0) or 0.0) > 0.0:
            _om = torch.ones_like(sample["noisy"])
            if _cut is not None:
                _om[..., _cut:] = 0.0
                sample["noisy"][..., _cut:] = sample["clean"][..., _cut:]
                sample["noise"][..., _cut:] = 0.0
                sample["q_target"][..., _cut:] = 1.0
                if "noisy_bg" in sample and "clean_bg" in sample:
                    sample["noisy_bg"][..., _cut:] = sample["clean_bg"][..., _cut:]
                if "noisy_alt" in sample:
                    sample["noisy_alt"][..., _cut:] = sample["clean"][..., _cut:]
            sample["obs_mask"] = _om
            sample["tail_loss"] = torch.tensor(0.0 if _cut is None else 1.0, dtype=torch.float32)
        return sample

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        if self.paired_noisy_path:
            return self._getitem_paired(item)
        _use_fwd = False
        self._ensure_open()
        assert self._clean is not None and self._noise is not None
        epoch_term = self._epoch_now() if self.training else 0
        _P = int(getattr(self.cfg, "profile_patch_len", 0) or 0)
        patch_offset = 0
        if self.training and _P > 1:
            patch_offset = int(item) % _P
            rng = np.random.default_rng(
                self.seed + 1_000_003 * epoch_term + 97_409 * (int(item) // _P))
        else:
            rng = np.random.default_rng(self.seed + 1_000_003 * epoch_term + 97_409 * int(item))
        if self.training:
            _runs = getattr(self, "_field_patch_runs", None) or self._patch_runs
            if _P > 1 and _runs:
                _use_fwd = (
                    self._mix_forward_idx is not None
                    and rng.random() < self.forward_fraction_now()
                )
                if _use_fwd:
                    _fr = getattr(self, "_forward_patch_runs", None)
                    if _fr:
                        fa, fb = _fr[int(rng.integers(0, len(_fr)))]
                        clean_idx = int(rng.integers(fa, fb - _P + 2)) + patch_offset
                    else:
                        clean_idx = int(self._mix_forward_idx[
                            rng.integers(0, len(self._mix_forward_idx))])
                else:
                    a, b = _runs[int(rng.integers(0, len(_runs)))]
                    anchor = _draw_anchor(rng, a, b, _P, float(getattr(self.cfg, "edge_patch_fraction", 0.0)))
                    clean_idx = anchor + patch_offset
            elif self._mix_field_idx is not None and self._mix_forward_idx is not None:
                _p = float(getattr(self.cfg, 'mix_forward_fraction', 0.15))
                _pool = self._mix_forward_idx if rng.random() < _p else self._mix_field_idx
                clean_idx = int(_pool[rng.integers(0, len(_pool))])
            elif _P > 1 and self._patch_runs:
                a, b = self._patch_runs[int(rng.integers(0, len(self._patch_runs)))]
                anchor = int(rng.integers(a, b - _P + 2))
                clean_idx = anchor + patch_offset
            else:
                clean_idx = int(self.split_indices[rng.integers(0, len(self.split_indices))])
        else:
            clean_idx = int(self.split_indices[item % len(self.split_indices)])
        clean = np.asarray(self._clean[clean_idx], dtype=np.float64).copy()
        aug_li = -1
        aug_amp = 1.0
        if (
            self.training
            and self._aug_mats is not None
            and rng.random() < float(self.cfg.clean_aug_probability)
            and bool(np.all(clean > 0.0))
        ):
            aug_li = int(rng.integers(0, len(self._aug_mats)))
            clean = np.exp(self._aug_mats[aug_li] @ np.log(clean))
            amp_dex = float(self.cfg.clean_aug_amp_dex)
            if amp_dex > 0.0:
                aug_amp = float(10.0 ** rng.uniform(-amp_dex, amp_dex))
                clean = clean * aug_amp
        bounds = self._block_bounds
        assert bounds is not None
        lengths = np.diff(bounds).astype(np.float64)
        block = int(rng.choice(len(lengths), p=lengths / lengths.sum()))
        bstart, blen = int(bounds[block]), int(lengths[block])
        center_local = int(rng.integers(0, blen))
        noise_idx = bstart + center_local
        sgn = -1.0 if (self.cfg.random_noise_sign_flip and rng.random() < 0.5) else 1.0
        roll_k = int(rng.integers(1, self.cfg.target_gates)) if rng.random() < self.cfg.noise_circular_shift_probability else 0
        mix_delta: Optional[int] = None
        mix_sgn = 1.0
        mix_beta = 0.0
        if rng.random() < self.cfg.noise_mixup_probability and blen > 1:
            mix_delta = int(rng.integers(1, blen))
            mix_sgn = -1.0 if (self.cfg.random_noise_sign_flip and rng.random() < 0.5) else 1.0
            mix_beta = float(rng.uniform(*self.cfg.noise_mixup_beta))
        white_ratio: Optional[float] = None
        if rng.random() < self.cfg.white_noise_probability:
            white_ratio = float(rng.uniform(*self.cfg.white_noise_relative_rms))

        def _composite(local_idx: int) -> np.ndarray:
            _row = self._noise[bstart + (local_idx % blen)]
            _mix = (self._noise[bstart + ((local_idx + mix_delta) % blen)]
                    if mix_delta is not None else None)
            return compose_noise_trace(_row, _mix, self.cfg, rng, sgn=sgn, roll_k=roll_k,
                                       mix_sgn=mix_sgn, mix_beta=mix_beta,
                                       white_ratio=white_ratio,
                                       gate_times_s=self._gate_times_s)

        clean_pre_anomaly = clean.copy()
        anomaly: Optional[Dict[str, float]] = None
        event_mask_np = np.zeros(clean.shape[0], dtype=np.float32)
        event_label = 0.0
        observed_clean = clean
        if self.training and rng.random() < float(getattr(self.cfg, "anomaly_profile_prob", 0.0)):
            if rng.random() < float(self.cfg.anomaly_singleton_prob):
                s_amp = float(rng.uniform(*self.cfg.anomaly_amp_range))
                if rng.random() < float(self.cfg.anomaly_negative_fraction):
                    s_amp = float(rng.uniform(*self.cfg.anomaly_suppress_range))
                _is_singleton_station = (_P <= 1) or (patch_offset == (_P - 1) // 2)
                observed_clean = clean * profile_anomaly_lift(
                    clean.shape[0], self.cfg.late_start_index, 1.0,
                    s_amp if _is_singleton_station else 1.0,
                    self.cfg.anomaly_onset_gates)
            else:
                a_amp = float(rng.uniform(*self.cfg.anomaly_amp_range))
                if rng.random() < float(self.cfg.anomaly_negative_fraction):
                    a_amp = float(rng.uniform(*self.cfg.anomaly_suppress_range))
                a_w = float(rng.uniform(*self.cfg.anomaly_width_range))
                if rng.random() < float(self.cfg.anomaly_skew_fraction):
                    a_wl = a_w * float(rng.uniform(0.45, 0.85))
                    a_wr = a_w * float(rng.uniform(1.15, 1.9))
                    if rng.random() < 0.5:
                        a_wl, a_wr = a_wr, a_wl
                else:
                    a_wl = a_wr = a_w
                a_c = float(rng.uniform(-self.cfg.anomaly_center_jitter,
                                        self.cfg.anomaly_center_jitter))
                anomaly = {"amp": a_amp, "w_l": a_wl, "w_r": a_wr, "c": a_c}
                _station_pos = float(patch_offset - (_P - 1) / 2.0) if _P > 1 else 0.0
                env_c = profile_anomaly_env(_station_pos, a_c, a_wl, a_wr)
                lift_c = profile_anomaly_lift(clean.shape[0], self.cfg.late_start_index,
                                              env_c, a_amp, self.cfg.anomaly_onset_gates)
                clean = clean * lift_c
                observed_clean = clean
                if env_c > 0.05:
                    event_label = 1.0
                    event_mask_np = np.clip(
                        (lift_c - 1.0) / (a_amp - 1.0 if abs(a_amp - 1.0) > 1e-9 else 1e-9),
                        0.0, 1.0).astype(np.float32)
        noise = _composite(center_local)
        clean_rms = float(np.sqrt(np.mean(observed_clean**2)))
        composite_rms = float(np.sqrt(np.mean(noise**2)))
        snr_db = sample_snr_db(rng, self.cfg, getattr(self, "_snr_override", None))
        gain = clean_rms / (
            (10.0 ** (snr_db / 20.0)) * max(composite_rms, self.cfg.minimum_noise_rms)
        )
        scaled_noise = noise * gain
        noisy = observed_clean + scaled_noise
        scaled_noise = noisy - clean
        q_target = compute_reliability_target_np(
            noisy,
            clean,
            kappa=self.cfg.reliability_kappa,
            window=self.cfg.reliability_window,
        )
        scale = self.global_scale
        sample = {
            "noisy": torch.from_numpy((noisy / scale).astype(np.float32))[None, :],
            "clean": torch.from_numpy((clean / scale).astype(np.float32))[None, :],
            "noise": torch.from_numpy((scaled_noise / scale).astype(np.float32))[None, :],
            "q_target": torch.from_numpy(q_target)[None, :],
            "snr_db": torch.tensor(snr_db, dtype=torch.float32),
            "clean_index": torch.tensor(clean_idx, dtype=torch.int64),
            "noise_index": torch.tensor(noise_idx, dtype=torch.int64),
            "clean_bg": torch.from_numpy(
                (clean_pre_anomaly / scale).astype(np.float32))[None, :],
            "noisy_bg": torch.from_numpy(
                (((clean_pre_anomaly + scaled_noise) if anomaly is not None else noisy)
                 / scale).astype(np.float32))[None, :],
            "has_anomaly": torch.tensor(1.0 if anomaly is not None else 0.0,
                                        dtype=torch.float32),
            "depth_norm": torch.tensor(
                [self.profile_position(clean_idx)],
                dtype=torch.float32),
            "profile_len": torch.tensor(int(self.batch_profile_len()), dtype=torch.int64),
            "event_mask": torch.from_numpy(event_mask_np)[None, :],
            "event_label": torch.tensor(event_label, dtype=torch.float32),
            "is_forward": torch.tensor(1.0 if _use_fwd else 0.0, dtype=torch.float32),
        }
        k = int(self.cfg.num_neighbors)
        if k > 0:
            gates = clean.shape[0]
            gate_axis = (np.arange(gates, dtype=np.float64) - 0.5 * (gates - 1)) / max(gates - 1, 1)
            add_stack = bool(
                getattr(self.cfg, "coherent_stack_slot", False)
                and self.real_neighbor_rows is not None
            )
            k_slots = k + (1 if add_stack else 0)
            neighbor_stack = np.empty((k_slots, gates), dtype=np.float32)
            neighbor_stack_bg = np.empty((k_slots, gates), dtype=np.float32)
            neighbor_clean_stack = (
                np.empty((k_slots, gates), dtype=np.float32) if self.emit_neighbor_clean else None
            )
            use_real_nb = self.real_neighbor_rows is not None
            if use_real_nb:
                row_nb = self.real_neighbor_rows[clean_idx]
                if bool(np.all(row_nb == clean_idx)):
                    use_real_nb = False
            geo = np.zeros((k + (1 if add_stack else 0), NEIGHBOR_GEOMETRY_FEATURES), dtype=np.float32)
            for j in range(k):
                offset = ((j // 2) + 1) * (1 if j % 2 == 0 else -1)
                if use_real_nb and self.real_neighbor_geometry is not None:
                    _so = float(self.real_neighbor_geometry[clean_idx, j, 0])
                    _ir = float(self.real_neighbor_geometry[clean_idx, j, 1])
                elif use_real_nb:
                    _so, _ir = float(offset), 1.0
                else:
                    _so, _ir = float(offset), 0.0
                geo[j, 0] = _so
                geo[j, 1] = abs(_so)
                geo[j, 2] = _ir
                if use_real_nb:
                    nb_row = int(self.real_neighbor_rows[clean_idx, j])
                    nb_clean = np.asarray(self._clean[nb_row], dtype=np.float64).copy()
                    if aug_li >= 0 and bool(np.all(nb_clean > 0.0)):
                        nb_clean = np.exp(self._aug_mats[aug_li] @ np.log(nb_clean)) * aug_amp
                    if anomaly is not None:
                        _env_j = profile_anomaly_env(_so, anomaly["c"], anomaly["w_l"], anomaly["w_r"])
                        nb_clean = nb_clean * profile_anomaly_lift(
                            nb_clean.shape[0], self.cfg.late_start_index, _env_j,
                            anomaly["amp"], self.cfg.anomaly_onset_gates)
                    nb_dropped = self.training and rng.random() < float(
                        getattr(self.cfg, "neighbor_channel_dropout", 0.0)
                    )
                else:
                    gain_nb = float(np.exp(rng.normal(0.0, self.cfg.neighbor_gain_std)))
                    slope = float(rng.normal(0.0, self.cfg.neighbor_tilt_std))
                    tilt = np.exp(slope * gate_axis)
                    nb_clean = clean_pre_anomaly * gain_nb * tilt
                    if anomaly is not None:
                        _env_j = profile_anomaly_env(_so, anomaly["c"], anomaly["w_l"], anomaly["w_r"])
                        nb_clean = nb_clean * profile_anomaly_lift(
                            nb_clean.shape[0], self.cfg.late_start_index, _env_j,
                            anomaly["amp"], self.cfg.anomaly_onset_gates)
                nb_noise = _composite(center_local + offset)
                nb_snr = snr_db + float(
                    rng.uniform(-self.cfg.neighbor_snr_jitter_db, self.cfg.neighbor_snr_jitter_db)
                )
                nb_gain = float(np.sqrt(np.mean(nb_clean**2))) / (
                    (10.0 ** (nb_snr / 20.0))
                    * max(float(np.sqrt(np.mean(nb_noise**2))), self.cfg.minimum_noise_rms)
                )
                if use_real_nb and nb_dropped:
                    nb_noisy = nb_noise * nb_gain
                else:
                    nb_noisy = nb_clean + nb_noise * nb_gain
                neighbor_stack[j] = (nb_noisy / scale).astype(np.float32)
                if anomaly is not None:
                    nb_clean_bg = nb_clean / np.maximum(
                        profile_anomaly_lift(
                            nb_clean.shape[0], self.cfg.late_start_index,
                            profile_anomaly_env(_so, anomaly["c"], anomaly["w_l"],
                                                anomaly["w_r"]),
                            anomaly["amp"], self.cfg.anomaly_onset_gates), 1e-30)
                    nb_noisy_bg = (nb_noise * nb_gain if (use_real_nb and nb_dropped)
                                   else nb_clean_bg + nb_noise * nb_gain)
                    neighbor_stack_bg[j] = (nb_noisy_bg / scale).astype(np.float32)
                else:
                    neighbor_stack_bg[j] = neighbor_stack[j]
                if neighbor_clean_stack is not None:
                    neighbor_clean_stack[j] = (nb_clean / scale).astype(np.float32)
            if add_stack:
                neighbor_stack[k] = (
                    neighbor_stack[:k].sum(axis=0) / float(k)
                ).astype(np.float32)
                neighbor_stack_bg[k] = (
                    neighbor_stack_bg[:k].sum(axis=0) / float(k)
                ).astype(np.float32)
                if neighbor_clean_stack is not None:
                    centre_clean_scaled = (clean / scale).astype(np.float32)
                    neighbor_clean_stack[k] = (
                        (centre_clean_scaled + neighbor_clean_stack[:k].sum(axis=0))
                        / float(k + 1)
                    ).astype(np.float32)
            if add_stack:
                geo[k, 3] = 1.0
            sample["neighbor_geometry"] = torch.from_numpy(geo)
            sample["neighbors"] = torch.from_numpy(neighbor_stack)
            sample["neighbors_bg"] = torch.from_numpy(neighbor_stack_bg)
            if neighbor_clean_stack is not None:
                sample["neighbors_clean"] = torch.from_numpy(neighbor_clean_stack)
        sample["clean_row"] = torch.tensor(int(clean_idx), dtype=torch.long)
        return sample


def profile_anomaly_env(offset: float, center: float, w_left: float, w_right: float) -> float:
    """Lateral envelope of a body at `center` with (possibly asymmetric) half-widths, evaluated at a station
    `offset`.
    """
    d = float(offset) - float(center)
    w = float(w_right if d >= 0.0 else w_left)
    return float(np.exp(-0.5 * (d / max(w, 1e-6)) ** 2))


def profile_anomaly_lift(gates: int, late_start: int, env: float, amp: float,
                         onset_gates: int = 3) -> np.ndarray:
    """Multiplicative late-window lift of a clean decay for a coherent geological anomaly: 1 + (amp-1)*env*ramp(t)"""
    g = np.arange(gates, dtype=np.float64)
    a = float(max(late_start - onset_gates, 0))
    b = float(max(gates - 1, 1))
    u = np.clip((g - a) / max(b - a, 1e-9), 0.0, 1.0)
    ramp = u * u * (3.0 - 2.0 * u)
    return 1.0 + (float(amp) - 1.0) * float(np.clip(env, 0.0, 1.0)) * ramp
