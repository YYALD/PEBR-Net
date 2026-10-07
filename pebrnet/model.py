# PEBR-Net, the prior- and evidence-bounded reconstruction network.
# MIT License, see LICENSE.
"""The PEBR-Net network (class PEBRNet).

This file is executed by pebrnet/__init__.py into the package namespace, in the order listed there;
names defined in the other files of the package are available here without imports.
"""
from __future__ import annotations

def signed_symlog(x: torch.Tensor) -> torch.Tensor:
    """Sign(x) * log(1+|x|): bounded working domain for equalized traces."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def signed_symexp(z: torch.Tensor) -> torch.Tensor:
    """Exact inverse of :func:`signed_symlog`."""
    return torch.sign(z) * torch.expm1(torch.abs(z))


def valid_group_count(channels: int, requested: int) -> int:
    g = min(channels, requested)
    while g > 1 and channels % g != 0:
        g -= 1
    return g


_NORM: Dict[str, bool] = {"gate_local": True}


class GateLocalNorm(nn.Module):
    """Normalisation over channels at each gate position (no statistic spans gates)."""

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.layer_norm(x.float().transpose(1, 2), (x.shape[1],), self.weight.float(), self.bias.float(), self.eps)
        return y.transpose(1, 2).to(dtype=x.dtype)


class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        groups: int = 8,
        dropout: float = 0.0,
        activation: bool = True,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        layers: List[nn.Module] = [
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            (GateLocalNorm(out_channels) if _NORM["gate_local"]
             else nn.GroupNorm(valid_group_count(out_channels, groups), out_channels)),
        ]
        if activation:
            layers.append(nn.SiLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DualDomainStem(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        c = cfg.base_channels
        g = cfg.group_norm_groups
        in_ch = 3 if cfg.use_position_encoding else 1
        self.linear_branch = nn.Sequential(
            ConvGNAct(in_ch, c // 2, 3, groups=g),
            ConvGNAct(c // 2, c, 5, groups=g),
        )
        self.symlog_branch = nn.Sequential(
            ConvGNAct(in_ch, c // 2, 3, groups=g),
            ConvGNAct(c // 2, c, 5, groups=g),
        )
        self.gate = nn.Sequential(
            nn.Conv1d(2 * c, c, 1),
            nn.Sigmoid(),
        )
        self.out = ConvGNAct(c, c, 3, groups=g)

    @staticmethod
    def signed_symlog(x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * torch.log1p(torch.abs(x))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f_lin = self.linear_branch(x)
        f_sym = self.symlog_branch(self.signed_symlog(x))
        gate = self.gate(torch.cat([f_lin, f_sym], dim=1))
        fused = gate * f_lin + (1.0 - gate) * f_sym
        return self.out(fused), gate


class MultiScaleTemporalBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if len(cfg.multiscale_kernels) != len(cfg.multiscale_dilations):
            raise ValueError("multiscale_kernels and multiscale_dilations must have equal length")
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    ConvGNAct(
                        cfg.base_channels,
                        cfg.branch_channels,
                        k,
                        dilation=d,
                        groups=cfg.group_norm_groups,
                        dropout=cfg.dropout,
                    ),
                    ConvGNAct(
                        cfg.branch_channels,
                        cfg.branch_channels,
                        3,
                        groups=cfg.group_norm_groups,
                    ),
                )
                for k, d in zip(cfg.multiscale_kernels, cfg.multiscale_dilations)
            ]
        )
        total = cfg.branch_channels * len(self.branches)
        self.project = ConvGNAct(total, cfg.feature_channels, 1, groups=cfg.group_norm_groups)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        branch_features = [branch(x) for branch in self.branches]
        stacked = torch.stack(branch_features, dim=1)
        scale_disagreement = stacked.var(dim=1, unbiased=False).mean(dim=1, keepdim=True)
        fused = self.project(torch.cat(branch_features, dim=1))
        return fused, scale_disagreement


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, cfg: ModelConfig) -> None:
        super().__init__()
        self.conv1 = ConvGNAct(
            channels,
            channels,
            3,
            dilation=dilation,
            groups=cfg.group_norm_groups,
            dropout=cfg.dropout,
        )
        self.conv2 = ConvGNAct(
            channels,
            channels,
            3,
            dilation=dilation,
            groups=cfg.group_norm_groups,
            activation=False,
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.conv2(self.conv1(x)))


class ResidualRefinementStack(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        ds = cfg.residual_dilations
        self.blocks = nn.ModuleList(
            [ResidualTemporalBlock(cfg.feature_channels, ds[i % len(ds)], cfg) for i in range(cfg.residual_blocks)]
        )
        self.checkpoint_blocks = (str(getattr(cfg, "activation_checkpointing", "off")).lower() == "on")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ck = (bool(getattr(self, "checkpoint_blocks", False)) and self.training
               and torch.is_grad_enabled() and bool(x.requires_grad)
               and not torch.jit.is_tracing() and not torch.jit.is_scripting())
        for block in self.blocks:
            if _ck:
                x = _torch_checkpoint_698.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x


class DegradationReliabilityRouter(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        c = cfg.feature_channels + 4
        self.net = nn.Sequential(
            ConvGNAct(c, cfg.base_channels, 5, groups=cfg.group_norm_groups),
            ConvGNAct(cfg.base_channels, cfg.base_channels // 2, 3, groups=cfg.group_norm_groups),
            nn.Conv1d(cfg.base_channels // 2, 1, 1),
        )

    @staticmethod
    def _diff1(x: torch.Tensor) -> torch.Tensor:
        d = x[..., 1:] - x[..., :-1]
        return F.pad(d, (1, 0), mode="replicate")

    @classmethod
    def _diff2(cls, x: torch.Tensor) -> torch.Tensor:
        return cls._diff1(cls._diff1(x))

    def forward(
        self,
        features: torch.Tensor,
        x: torch.Tensor,
        scale_disagreement: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return both reliability logits and probabilities."""
        d1 = torch.abs(self._diff1(x))
        d2 = torch.abs(self._diff2(x))
        local_mean = F.avg_pool1d(x, kernel_size=5, stride=1, padding=2)
        hf = torch.abs(x - local_mean)
        inputs = torch.cat([features, d1, d2, hf, scale_disagreement], dim=1)
        logits = self.net(inputs)
        probability = torch.sigmoid(logits)
        return logits, probability


class HighConfidenceReferenceAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=cfg.feature_channels,
            num_heads=cfg.attention_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(cfg.feature_channels)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.feature_channels, 2 * cfg.feature_channels),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(2 * cfg.feature_channels, cfg.feature_channels),
        )
        self.norm2 = nn.LayerNorm(cfg.feature_channels)

    def forward(self, features: torch.Tensor, reliability: torch.Tensor) -> torch.Tensor:
        f = features.transpose(1, 2)
        q = reliability.transpose(1, 2)
        high = f * q
        degraded = f * (1.0 - q)
        context, _ = self.attn(degraded, high, high, need_weights=False)
        z = self.norm(degraded + context)
        z = self.norm2(z + self.ffn(z))
        return z.transpose(1, 2)


class NeighborCrossAttention(nn.Module):
    """Cross-trace attention over K neighboring stations."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        c = cfg.feature_channels
        self.attn = nn.MultiheadAttention(
            embed_dim=c, num_heads=cfg.attention_heads, dropout=cfg.dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(c)
        self.ffn = nn.Sequential(
            nn.Linear(c, 2 * c), nn.SiLU(), nn.Dropout(cfg.dropout), nn.Linear(2 * c, c)
        )
        self.norm2 = nn.LayerNorm(c)
        self.gate = nn.Parameter(torch.full((1, c, 1), 0.5))
        self.heads = int(cfg.attention_heads)
        self.use_geometry = bool(getattr(cfg, "use_neighbor_geometry", False))
        if self.use_geometry:
            self.geo_bias = nn.Linear(NEIGHBOR_GEOMETRY_FEATURES, self.heads)
            nn.init.zeros_(self.geo_bias.weight)
            nn.init.zeros_(self.geo_bias.bias)
            p0 = float(max(getattr(cfg, "neighbor_distance_prior_init", 0.10), 0.0))
            rho0 = math.log(math.expm1(p0)) if p0 > 1e-6 else -12.0
            self.geo_prior_rho = nn.Parameter(torch.full((self.heads,), float(rho0)))

    def _geometry_biased_attention(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        bias_bhk: torch.Tensor,
        b: int,
        t: int,
        k: int,
    ) -> torch.Tensor:
        with amp_autocast_context(q.device, enabled=False):
            c = q.shape[-1]
            h = self.heads
            dh = c // h
            w = self.attn.in_proj_weight.float()
            bb = self.attn.in_proj_bias
            bb = bb.float() if bb is not None else None
            q32 = F.linear(q.float(), w[:c], bb[:c] if bb is not None else None)
            k32 = F.linear(kv.float(), w[c:2 * c], bb[c:2 * c] if bb is not None else None)
            v32 = F.linear(kv.float(), w[2 * c:], bb[2 * c:] if bb is not None else None)
            n = q32.shape[0]
            q32 = q32.view(n, 1, h, dh).permute(0, 2, 1, 3)
            k32 = k32.view(n, k, h, dh).permute(0, 2, 1, 3)
            v32 = v32.view(n, k, h, dh).permute(0, 2, 1, 3)
            scores = torch.matmul(q32, k32.transpose(-2, -1)) / math.sqrt(dh)
            scores = scores + (
                bias_bhk[:, None, :, None, :]
                .expand(b, t, h, 1, k)
                .reshape(n, h, 1, k)
            )
            probs = torch.softmax(scores, dim=-1)
            if self.training and float(self.attn.dropout) > 0.0:
                probs = F.dropout(probs, p=float(self.attn.dropout), training=True)
            ctx32 = torch.matmul(probs, v32)
            ctx32 = ctx32.permute(0, 2, 1, 3).reshape(n, 1, c)
            ctx32 = self.attn.out_proj(ctx32.to(self.attn.out_proj.weight.dtype))
        return ctx32.to(dtype=q.dtype)

    def forward(
        self,
        center: torch.Tensor,
        neighbors: torch.Tensor,
        geometry: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, c, t = center.shape
        k = neighbors.shape[1]
        q = center.permute(0, 2, 1).reshape(b * t, 1, c)
        kv = neighbors.permute(0, 3, 1, 2).reshape(b * t, k, c)
        bias_bhk = None
        if self.use_geometry and geometry is not None and geometry.shape[1] == k:
            with amp_autocast_context(center.device, enabled=False):
                g = geometry.float()
                adaptive = self.geo_bias(g)
                prior = -F.softplus(self.geo_prior_rho).view(1, 1, -1) * (
                    g[..., 1:2].clamp(max=16.0) * (1.0 - g[..., 3:4])
                )
                bias_bhk = (adaptive + prior).permute(0, 2, 1)
        if bias_bhk is None:
            ctx, _ = self.attn(q, kv, kv, need_weights=False)
        else:
            ctx = self._geometry_biased_attention(q, kv, bias_bhk, b, t, k)
        z = self.norm(q + ctx)
        z = self.norm2(z + self.ffn(z))
        context = z.reshape(b, t, c).permute(0, 2, 1)
        return torch.tanh(self.gate) * context


class ProfileJointEncoder(nn.Module):
    """Joint time-profile manifold code."""

    def __init__(self, in_dim: int, dim: int = 64, heads: int = 2, max_rel: int = 8) -> None:
        super().__init__()
        self.dim = int(dim)
        self.heads = max(1, int(heads))
        self.max_rel = max(1, int(max_rel))
        self.proj = nn.Linear(int(in_dim), self.dim)
        self.norm1 = nn.LayerNorm(self.dim)
        self.qkv = nn.Linear(self.dim, 3 * self.dim)
        self.out = nn.Linear(self.dim, self.dim)
        self.rel_bias = nn.Parameter(torch.zeros(self.heads, 2 * self.max_rel + 1))
        self.dw = nn.Conv1d(self.dim, self.dim, 3, padding=1, groups=self.dim)
        self.norm2 = nn.LayerNorm(self.dim)
        self.ff = nn.Sequential(nn.Linear(self.dim, 2 * self.dim), nn.GELU(),
                                nn.Linear(2 * self.dim, self.dim))
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        nn.init.zeros_(self.ff[-1].weight)
        nn.init.zeros_(self.ff[-1].bias)
        nn.init.zeros_(self.dw.weight)
        nn.init.zeros_(self.dw.bias)

    def forward(self, h: torch.Tensor, profile_len: int) -> torch.Tensor:
        B = int(h.shape[0])
        P = int(profile_len)
        n = B // P
        with amp_autocast_context(h.device, enabled=False):
            x = self.proj(h.float()).view(n, P, self.dim)
            y = self.norm1(x)
            q, k, v = self.qkv(y).chunk(3, dim=-1)
            dh = self.dim // self.heads
            q = q.view(n, P, self.heads, dh).transpose(1, 2)
            k = k.view(n, P, self.heads, dh).transpose(1, 2)
            v = v.view(n, P, self.heads, dh).transpose(1, 2)
            att = (q @ k.transpose(-1, -2)) / math.sqrt(float(dh))
            idx = torch.arange(P, device=h.device)
            rel = (idx[None, :] - idx[:, None]).clamp(-self.max_rel, self.max_rel) + self.max_rel
            att = att + self.rel_bias[:, rel].unsqueeze(0)
            att = torch.softmax(att, dim=-1)
            x = x + self.out((att @ v).transpose(1, 2).reshape(n, P, self.dim))
            x = x + self.dw(x.transpose(1, 2)).transpose(1, 2)
            x = x + self.ff(self.norm2(x))
            return x.reshape(B, self.dim)


class LateralFingerprintOperator(nn.Module):
    """Dynamic lateral kernels from a bank of lateral prototype signatures."""

    def __init__(self, ctx_dim: int, gates: int, late_start: int, n_proto: int = 32,
                 kernel: int = 9, key_dim: int = 32) -> None:
        super().__init__()
        self.k = int(max(3, kernel | 1))
        self.query = nn.Linear(int(ctx_dim), int(key_dim))
        self.keys = nn.Parameter(0.2 * torch.randn(int(n_proto), int(key_dim)))
        self.log_temp = nn.Parameter(torch.tensor(math.log(4.0)))
        self.shapes = nn.Parameter(0.3 * torch.randn(int(n_proto), self.k))
        self.env = nn.Parameter(torch.zeros(int(n_proto), int(gates)))
        mask = torch.ones(int(gates))
        mask[: max(0, int(late_start) // 2)] = 0.0
        self.register_buffer("gate_mask", mask, persistent=True)

    def kernels(self, ctx: torch.Tensor,
                strength: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Per-station, per-gate kernels [B, T, k]."""
        q = self.query(ctx.float())
        _temp = torch.exp(self.log_temp).clamp(0.25, 32.0)
        alpha = torch.softmax(_temp * (q @ self.keys.t()) / math.sqrt(float(q.shape[-1])), dim=-1)
        shp = torch.tanh(self.shapes)
        shp = shp - shp.mean(dim=-1, keepdim=True)
        env = torch.tanh(self.env) * self.gate_mask.view(1, -1)
        dyn = torch.einsum("bm,mt,mk->btk", alpha, env, shp)
        if strength is not None:
            dyn = dyn * strength.float().reshape(dyn.shape[0], -1, 1)
        delta = torch.zeros(self.k, device=ctx.device, dtype=dyn.dtype)
        delta[self.k // 2] = 1.0
        return delta.view(1, 1, -1) + dyn

    def forward(self, zc: torch.Tensor, ctx: torch.Tensor, profile_len: int,
                strength: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = int(zc.shape[0])
        P = int(profile_len)
        if P < 3 or B % P != 0:
            return zc
        n = B // P
        T = int(zc.shape[-1])
        c = self.k // 2
        with amp_autocast_context(zc.device, enabled=False):
            K = self.kernels(ctx, strength).view(n, P, T, self.k)
            Z = zc.float().reshape(n, P, T).permute(0, 2, 1)
            Zp = F.pad(Z, (c, c), mode="replicate")
            Zu = Zp.unfold(-1, self.k, 1)
            out = (Zu * K.permute(0, 2, 1, 3)).sum(dim=-1)
            delta = torch.zeros(self.k, device=zc.device, dtype=K.dtype)
            delta[c] = 1.0
            dev = (K - delta.view(1, 1, 1, -1)).abs().sum(-1)
            self._last_dev = dev.reshape(B, 1, T)
            return out.permute(0, 2, 1).reshape(B, 1, T).to(dtype=zc.dtype)


class LateralJointPrior(nn.Module):
    """Profile-level joint estimator of the fused z-curve along the station axis."""

    def __init__(self, gates: int, late_start: int, late_ramp: torch.Tensor,
                 lam_late: float = 1.0, lam_mid: float = 0.5, lam_early: float = 0.02,
                 order: int = 2) -> None:
        super().__init__()
        self.gates = int(gates)
        self.order = int(max(1, min(3, order)))
        ramp = late_ramp.detach().float().reshape(-1)
        if ramp.numel() != self.gates:
            ramp = torch.zeros(self.gates)
            ramp[int(late_start):] = 1.0
        mid0 = max(0, int(late_start) // 2)
        init = torch.full((self.gates,), float(max(lam_early, 1e-6)))
        init[mid0:] = float(max(lam_mid, 1e-6))
        init = init * (1.0 - ramp) + float(max(lam_late, 1e-6)) * ramp
        init[:mid0] = float(max(lam_early, 1e-6))
        theta = torch.log(torch.expm1(init.clamp_min(1e-6)))
        self.log_lambda = nn.Parameter(theta)
        self.log_lambda_wide = nn.Parameter(torch.log(torch.expm1((12.0 * init).clamp_min(1e-6))))
        self.mid_start = max(0, int(late_start) // 2)
        self.late_start = int(late_start)
        self.two_scale = True
        self.coh_kappa = 3.0
        self.coh_tau = 0.5
        self._dtd_cache: Dict[int, torch.Tensor] = {}
        self.lam_floor_late: float = 0.0
        self.boundary_reflect: bool = True

    def lambdas_wide(self) -> torch.Tensor:
        return F.softplus(self.log_lambda_wide)

    def lambdas(self) -> torch.Tensor:
        return F.softplus(self.log_lambda)

    def _dmat(self, P: int, device: torch.device) -> torch.Tensor:
        key = -int(P) - 1
        m = self._dtd_cache.get(key)
        if m is None or m.device != device:
            D = torch.eye(P, dtype=torch.float32)
            for _ in range(self.order):
                D = D[1:] - D[:-1]
            m = D.to(device)
            self._dtd_cache[key] = m
        return m

    def _regulariser(self, Ws: torch.Tensor, lam: torch.Tensor, Dm: torch.Tensor,
                        DtD: torch.Tensor) -> torch.Tensor:
        if (lam.dim() == 4 and lam.shape[-2] == Ws.shape[-1] and lam.shape[-2] > 1
                and bool(getattr(self, "symmetric_form", True))):
            k = int(self.order)
            lam_st = lam.squeeze(-1)
            n_, T_, Ps_ = lam_st.shape
            lam_d = F.avg_pool1d(lam_st.reshape(-1, 1, Ps_), kernel_size=k + 1, stride=1
                                 ).reshape(n_, T_, Ps_ - k)
            DtL = Dm.t().unsqueeze(0).unsqueeze(0) * lam_d.unsqueeze(-2)
            return torch.diag_embed(Ws) + DtL @ Dm
        return torch.diag_embed(Ws) + lam * DtD

    def _dtd(self, P: int, device: torch.device) -> torch.Tensor:
        key = int(P)
        m = self._dtd_cache.get(key)
        if m is None or m.device != device:
            D = torch.eye(P, dtype=torch.float32)
            for _ in range(self.order):
                D = D[1:] - D[:-1]
            m = (D.t() @ D).to(device)
            self._dtd_cache[key] = m
        return m

    def forward(self, z_hat: torch.Tensor, profile_len: Optional[int],
                weights: Optional[torch.Tensor] = None,
                noise_sigma: Optional[torch.Tensor] = None,
                noise_kappa: float = 0.0, noise_cap: float = 1.0,
                noise_sp0: float = 0.15,
                lam_factor: Optional[torch.Tensor] = None,
                floor_gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        if profile_len is None:
            return z_hat
        P = int(profile_len)
        B = int(z_hat.shape[0])
        if P < 3 or B % P != 0 or P <= self.order:
            return z_hat
        n = B // P
        T = int(z_hat.shape[-1])
        with amp_autocast_context(z_hat.device, enabled=False):
            Z = z_hat.float().reshape(n, P, T).permute(0, 2, 1).unsqueeze(-1)
            if weights is not None:
                W = weights.float().reshape(n, P, T).permute(0, 2, 1)
                W = W / W.mean(dim=-1, keepdim=True).clamp_min(1e-8)
                W = W.clamp(0.25, 4.0)
            else:
                W = torch.ones(n, T, P, device=z_hat.device, dtype=torch.float32)
            lam = self.lambdas().float().view(1, T, 1, 1)
            if noise_sigma is not None and noise_kappa > 0.0:
                _sm2 = noise_sigma.detach().float().reshape(n, P, T).pow(2).mean(dim=1)
                _fac = (1.0 + float(noise_kappa) * _sm2 / (float(noise_sp0) ** 2)
                        ).clamp(max=max(1.0, float(noise_cap)))
                lam = lam * _fac.view(n, T, 1, 1)
            if lam_factor is not None:
                _lf = lam_factor.detach().float().reshape(n, P, T).permute(0, 2, 1)
                lam = lam * _lf.unsqueeze(-1)
            _fl = float(getattr(self, "lam_floor_late", 0.0) or 0.0)
            if _fl > 0.0 and 0 <= int(self.late_start) < T:
                _fv = torch.zeros(1, T, 1, 1, device=lam.device, dtype=lam.dtype)
                _fv[:, int(self.late_start):] = _fl
                if floor_gate is not None:
                    _fg = floor_gate.detach().float().reshape(n, P, T).permute(0, 2, 1).unsqueeze(-1)
                    _fv = _fv * _fg.clamp(0.0, 1.0)
                lam = lam + _fv
            _k = int(self.order) if bool(getattr(self, "boundary_reflect", True)) else 0
            if _k > 0 and P > _k:
                Zs = F.pad(Z.squeeze(-1), (_k, _k), mode="reflect").unsqueeze(-1)
                Ws = F.pad(W, (_k, _k), mode="reflect")
                lam_s = (F.pad(lam.squeeze(-1), (_k, _k), mode="reflect").unsqueeze(-1)
                         if lam.dim() == 4 and lam.shape[-2] == P else lam)
            else:
                Zs, Ws, lam_s, _k = Z, W, lam, 0
            _Ps = P + 2 * _k
            _DtD = self._dtd(_Ps, z_hat.device).view(1, 1, _Ps, _Ps)
            _Dm = self._dmat(_Ps, z_hat.device)
            A = self._regulariser(Ws, lam_s, _Dm, _DtD)
            U = torch.linalg.solve(A, Ws.unsqueeze(-1) * Zs)
            Un = U.squeeze(-1)[..., _k:_k + P]
            if bool(getattr(self, "two_scale", False)) and P >= 5:
                lam_w = lam / self.lambdas().float().view(1, T, 1, 1).clamp_min(1e-8) \
                    * self.lambdas_wide().float().view(1, T, 1, 1)
                lam_ws = (F.pad(lam_w.squeeze(-1), (_k, _k), mode="reflect").unsqueeze(-1)
                          if (_k > 0 and lam_w.dim() == 4 and lam_w.shape[-2] == P) else lam_w)
                Aw = self._regulariser(Ws, lam_ws, _Dm, _DtD)
                Uw = torch.linalg.solve(Aw, Ws.unsqueeze(-1) * Zs).squeeze(-1)[..., _k:_k + P]
                d = (Un - Uw).detach()
                M = torch.ones(n, T, P, device=z_hat.device, dtype=torch.float32)
                for _a, _b in ((self.mid_start, self.late_start), (self.late_start, T)):
                    if _b - _a < 1:
                        continue
                    db = d[:, _a:_b, :]
                    sg = db.sum(dim=1, keepdim=True)
                    pad = F.pad(sg, (1, 1), mode="replicate")
                    s3 = pad[..., :-2] + pad[..., 1:-1] + pad[..., 2:]
                    sig3 = 1.4826 * torch.median(s3.abs(), dim=-1, keepdim=True).values + 1e-12
                    tstat = s3.abs() / sig3
                    Mb = torch.sigmoid((tstat - float(self.coh_kappa)) / max(float(self.coh_tau), 1e-3))
                    M[:, _a:_b, :] = Mb.expand(-1, _b - _a, -1)
                M[:, :self.mid_start, :] = 1.0
                Un = Uw + M * (Un - Uw)
                self._last_coherence_gate = M.mean(dim=1).reshape(B, 1)
            out = Un.permute(0, 2, 1).reshape(B, 1, T)
        return out.to(dtype=z_hat.dtype)


class PEBRNet(nn.Module):
    """PEBR-Net, the prior- and evidence-bounded reconstruction network."""
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        _NORM["gate_local"] = bool(getattr(cfg, "gate_local_norm", False))
        self.register_buffer("gate_scale_norm", torch.ones(1, 1, int(cfg.gates)), persistent=True)
        self.gl_departure_logit = (nn.Parameter(torch.full((int(cfg.gates),), float(getattr(cfg, "gate_loss_departure_init", -2.0))))
                                   if bool(getattr(cfg, "gate_loss_departure", True)) else None)
        self.register_buffer("gate_precision", torch.ones(int(cfg.gates)), persistent=True)
        self.register_buffer("gate_precision_ready", torch.zeros(()), persistent=True)
        self.register_buffer("seam_blend_floor", torch.zeros(int(cfg.gates)), persistent=True)
        self.register_buffer(
            "gate_position",
            self._build_position_channels(np.geomspace(0.054, 16.0, int(cfg.gates))),
            persistent=True,
        )
        default_times = np.geomspace(0.054, 16.0, int(cfg.gates))
        self.register_buffer(
            "gate_times", torch.as_tensor(default_times, dtype=torch.float32).view(1, -1), persistent=True
        )
        self._atom_stretches: Tuple[float, ...] = tuple(
            getattr(cfg, "relaxation_stretch_exponents", (1.0,)) or (1.0,)
        )
        self.register_buffer(
            "decay_atoms",
            self._build_decay_atoms(
                default_times, int(cfg.physics_atom_count), self._atom_stretches
            ),
            persistent=True,
        )
        if cfg.use_ip_head and cfg.use_physics_atoms:
            self.register_buffer(
                "ip_atoms",
                self._build_ip_atoms(
                    default_times,
                    int(cfg.late_start_index),
                    int(cfg.ip_atom_count),
                    float(cfg.ip_tau_min_factor),
                    float(cfg.ip_tau_max_factor),
                ),
                persistent=True,
            )
        floor = torch.zeros(1, 1, int(cfg.gates))
        _ls_f = int(cfg.late_start_index)
        _ramp_n = int(max(0, getattr(cfg, "blend_floor_ramp_gates", 6)))
        if cfg.use_manifold_branch:
            floor[..., _ls_f:] = float(cfg.manifold_blend_floor_late)
            if _ramp_n > 0:
                _a = max(0, _ls_f - _ramp_n)
                for _g in range(_a, _ls_f):
                    _u = (float(_g - _a) + 0.5) / float(max(_ls_f - _a, 1))
                    floor[..., _g] = (float(cfg.manifold_blend_floor_late)
                                      * (_u * _u * (3.0 - 2.0 * _u)))
        self.register_buffer("blend_floor", floor, persistent=True)
        _lr = (floor / max(float(cfg.manifold_blend_floor_late), 1e-6)).clamp(0.0, 1.0)
        if not cfg.use_manifold_branch:
            _lr = torch.zeros_like(floor)
            _lr[..., _ls_f:] = 1.0
        self.register_buffer("late_ramp", _lr, persistent=True)
        self._prior_band_g: int = int(max(1, _ls_f - _ramp_n))
        for _nm in ("profile_joint", "joint_inject", "lateral_fp_prior", "lateral_fp_out"):
            if not hasattr(self, _nm):
                setattr(self, _nm, None)
        self.innovation_warmup: float = 1.0
        if (bool(getattr(cfg, "lateral_joint_prior", True))
                and str(getattr(cfg, "lateral_joint_mode", "learned")) == "tikhonov"):
            self.lateral_joint = LateralJointPrior(
                int(cfg.gates), int(_ls_f), _lr,
                lam_late=float(getattr(cfg, "lateral_joint_lambda_late", 1.0)),
                lam_mid=float(getattr(cfg, "lateral_joint_lambda_mid", 0.5)),
                lam_early=float(getattr(cfg, "lateral_joint_lambda_early", 0.02)),
                order=int(getattr(cfg, "lateral_joint_order", 2)),
            )
        else:
            self.lateral_joint = None
        if self.lateral_joint is not None:
            self.lateral_joint.two_scale = bool(getattr(cfg, "lateral_two_scale", True))
            self.lateral_joint.coh_kappa = float(getattr(cfg, "lateral_coherence_kappa", 3.0))
            self.lateral_joint.coh_tau = float(getattr(cfg, "lateral_coherence_tau", 0.5))
            self.lateral_joint.lam_floor_late = float(max(0.0, getattr(cfg, "lateral_joint_lambda_floor_late", 0.0)))
            self.lateral_joint.boundary_reflect = bool(getattr(cfg, "lateral_joint_boundary_reflect", True))
            with torch.no_grad():
                _mult = float(max(1.0, getattr(cfg, "lateral_wide_multiplier", 12.0)))
                self.lateral_joint.log_lambda_wide.copy_(
                    torch.log(torch.expm1((_mult * F.softplus(self.lateral_joint.log_lambda)).clamp_min(1e-6))))
        _pos_floor = (floor > 0).reshape(-1)
        _ls_i = int(_pos_floor.float().argmax().item()) if bool(_pos_floor.any()) else 0
        self._late_start_gate = _ls_i
        _late_b = torch.zeros(1, 1, int(cfg.gates), dtype=torch.bool)
        _late_b[..., _ls_i:] = True
        self.register_buffer("nonlate_gate_mask", ~_late_b, persistent=False)
        _ev_acc = torch.zeros(1, 1, int(cfg.gates))
        _ev_acc[..., _ls_i // 2:] = 1.0
        self.register_buffer("evidence_accum_mask", _ev_acc, persistent=False)
        self.evidence_start_index = _ls_i // 2
        self.register_buffer("nonevidence_gate_mask", ~(_ev_acc > 0.5), persistent=False)
        self.register_buffer("blend_late_quality_cap", torch.full((), 0.999), persistent=True)
        self.register_buffer("late_prior_advantage", torch.zeros(()), persistent=True)
        self.register_buffer("event_expression_scale", torch.zeros(()), persistent=True)
        self.register_buffer("blend_late_val_floor", torch.zeros(()), persistent=True)
        self.register_buffer("blend_floor_scale", torch.ones(()), persistent=True)
        self.stem = DualDomainStem(cfg)
        self.multiscale = MultiScaleTemporalBlock(cfg)
        self.refinement_stack = ResidualRefinementStack(cfg)
        self.sigma_prior_713 = nn.Conv1d(cfg.feature_channels, 1, 1)
        self.sigma_meas_713 = nn.Conv1d(cfg.feature_channels, 1, 1)
        nn.init.zeros_(self.sigma_prior_713.weight)
        nn.init.constant_(self.sigma_prior_713.bias, -6.0)
        nn.init.zeros_(self.sigma_meas_713.weight)
        nn.init.constant_(self.sigma_meas_713.bias, -6.0)
        self.register_buffer("mmse_bstar_714", torch.full((int(cfg.gates),), float(getattr(cfg, "mmse_bstar_init", 0.65))), persistent=True)
        self.register_buffer("snr_ref_714", torch.full((int(cfg.gates),), 10.0), persistent=True)
        self.register_buffer("mmse_deploy_714", torch.zeros(()), persistent=True)
        self._force_mmse_blend = False
        self._force_no_anchors = False
        self._ablate_cdmr = False
        self._ablate_egsr = False
        self.register_buffer("row_completion_W_713", torch.zeros(0), persistent=False)
        self.register_buffer("row_completion_mx_713", torch.zeros(0), persistent=False)
        self.register_buffer("row_completion_my_713", torch.zeros(0), persistent=False)
        self.row_completion_gate_limit = 0
        self.router = DegradationReliabilityRouter(cfg)
        self.reference_attention = HighConfidenceReferenceAttention(cfg)
        self.neighbor_attention = (
            NeighborCrossAttention(cfg) if cfg.use_neighbor_attention else None
        )
        if cfg.use_manifold_branch:
            fc = cfg.feature_channels
            heads = int(cfg.manifold_pool_heads)
            self.manifold_score = nn.Conv1d(fc, heads, 1)
            self.decoder_gate_channel = bool(getattr(cfg, "decoder_gate_norm_channel", False))
            dec_in = heads * fc + 2 * int(cfg.gates) + (int(cfg.gates) if self.decoder_gate_channel else 0)
            _pw = max(1, int(round(2 * fc * float(
                getattr(cfg, "prior_width_mult", 1.0)))))
            self.manifold_decoder = nn.Sequential(
                nn.Linear(dec_in, _pw), nn.SiLU(),
                nn.Linear(_pw, _pw), nn.SiLU(),
                nn.Linear(_pw, _pw), nn.SiLU(),
            )
            self.manifold_head_free = nn.Linear(_pw, int(cfg.gates))
            _K = int(getattr(cfg, "signal_fingerprint_bank", 0) or 0)
            _D = int(getattr(cfg, "fingerprint_dim", 48))
            if _K > 0:
                self.fingerprint_query = nn.Linear(heads * fc + 1 + 16, _D)
                self.fingerprint_query_ctx = nn.Linear(
                    int(getattr(cfg, "joint_profile_dim", 64)), _D)
                nn.init.zeros_(self.fingerprint_query_ctx.weight)
                nn.init.zeros_(self.fingerprint_query_ctx.bias)
                self.lateral_encode = nn.Linear(int(cfg.gates), 16)
                self.lateral_inject = nn.Linear(16, _pw)
                nn.init.zeros_(self.lateral_inject.weight)
                nn.init.zeros_(self.lateral_inject.bias)
                self.fingerprint_keys = nn.Parameter(
                    0.02 * torch.randn(_K, _D))
                self.fingerprint_curves = nn.Parameter(
                    torch.zeros(_K, int(cfg.gates)))
                self.fingerprint_gate = nn.Linear(_pw, 1)
                nn.init.constant_(self.fingerprint_gate.bias, -2.0)
            else:
                self.fingerprint_query = None
                self.fingerprint_query_ctx = None
                self.fingerprint_keys = None
                self.fingerprint_curves = None
                self.fingerprint_gate = None
                self.lateral_encode = None
                self.lateral_inject = None
            if str(getattr(cfg, "lateral_joint_mode", "learned")) == "learned":
                _jd = int(getattr(cfg, "joint_profile_dim", 64))
                self.profile_joint = ProfileJointEncoder(
                    _pw + 2 + 16, dim=_jd,
                    heads=int(getattr(cfg, "joint_profile_heads", 2)),
                    max_rel=int(getattr(cfg, "joint_profile_max_rel", 8)))
                self.joint_inject = nn.Linear(_jd, _pw)
                nn.init.zeros_(self.joint_inject.weight)
                nn.init.zeros_(self.joint_inject.bias)
                _M = int(getattr(cfg, "lateral_fingerprint_bank", 32))
                _k = int(getattr(cfg, "lateral_fingerprint_kernel", 9))
                self.lateral_fp_prior = (LateralFingerprintOperator(
                    _jd, int(cfg.gates), int(cfg.late_start_index), _M, _k)
                    if bool(getattr(cfg, "lateral_fingerprint_on_prior", True)) else None)
                self.lateral_fp_out = (LateralFingerprintOperator(
                    _jd, int(cfg.gates), int(cfg.late_start_index), _M, _k)
                    if bool(getattr(cfg, "lateral_fingerprint_on_output", True)) else None)
                for _op in (self.lateral_fp_prior, self.lateral_fp_out):
                    if _op is not None:
                        with torch.no_grad():
                            _op.log_temp.fill_(math.log(max(
                                0.25, float(getattr(cfg, "lateral_retrieval_temperature", 4.0)))))
            else:
                self.profile_joint = None
                self.joint_inject = None
                self.lateral_fp_prior = None
                self.lateral_fp_out = None
            _n_atoms_total = int(cfg.physics_atom_count) * max(
                1, len(tuple(getattr(cfg, "relaxation_stretch_exponents", (1.0,)) or (1.0,)))
            )
            self.manifold_head_atoms = (
                nn.Linear(_pw, _n_atoms_total) if cfg.use_physics_atoms else None
            )
            if cfg.use_ip_head and cfg.use_physics_atoms:
                self.manifold_head_ip = nn.Linear(_pw, int(cfg.ip_atom_count))
                self.manifold_gate_ip = nn.Linear(_pw, 1)
            else:
                self.manifold_head_ip = None
                self.manifold_gate_ip = None
            if cfg.use_manifold_innovation:
                self.manifold_innovation = nn.Sequential(
                    ConvGNAct(fc, cfg.base_channels // 2, 3, groups=cfg.group_norm_groups),
                    nn.Conv1d(cfg.base_channels // 2, 1, 1),
                )
            else:
                self.manifold_innovation = None
            if cfg.use_manifold_innovation and cfg.measurement_conditioned_gates:
                self.prior_log_sigma = nn.Parameter(torch.full((int(cfg.gates),), -1.0))
            else:
                self.prior_log_sigma = None
            self.manifold_blend = nn.Sequential(
                ConvGNAct(fc, cfg.base_channels // 2, 3, groups=cfg.group_norm_groups),
                nn.Conv1d(cfg.base_channels // 2, 1, 1),
            )
            if bool(getattr(cfg, "use_library_subspace", True)):
                _r = int(max(2, getattr(cfg, "library_subspace_rank", 12)))
                self.register_buffer("lib_mu", torch.zeros(1, 1, int(cfg.gates)),
                                     persistent=True)
                self.register_buffer("lib_basis", torch.zeros(_r, int(cfg.gates)),
                                     persistent=True)
                self.register_buffer("lib_ready", torch.zeros(()), persistent=True)
                self._lib_ready_flag: Optional[bool] = None
                self.lib_coef_head = nn.Sequential(
                    nn.Linear(fc, max(16, fc // 2)), nn.GELU(),
                    nn.Linear(max(16, fc // 2), _r),
                )
                self.lib_gate_head = nn.Sequential(
                    ConvGNAct(fc, cfg.base_channels // 2, 3,
                              groups=cfg.group_norm_groups),
                    nn.Conv1d(cfg.base_channels // 2, 1, 1),
                )
            else:
                self.lib_coef_head = None
                self.lib_gate_head = None
                self._lib_ready_flag = False
            if bool(getattr(cfg, "use_ridge_anchor", True)):
                _lsR = int(getattr(cfg, "late_start_index", 20))
                _lateR = int(cfg.gates) - _lsR
                self.register_buffer("lib_ridge_W",
                                     torch.zeros(_lsR, _lateR), persistent=True)
                self.register_buffer("lib_ridge_b",
                                     torch.zeros(_lateR), persistent=True)
                self.register_buffer("lib_ridge_ready", torch.zeros(()),
                                     persistent=True)
                self._ridge_ready_flag: Optional[bool] = None
                self.ridge_gate_head = nn.Sequential(
                    ConvGNAct(fc, cfg.base_channels // 2, 3,
                              groups=cfg.group_norm_groups),
                    nn.Conv1d(cfg.base_channels // 2, 1, 1),
                )
                self.cont_head = nn.Sequential(
                    nn.Linear(2 * int(cfg.gates), 160), nn.GELU(),
                    nn.Linear(160, 160), nn.GELU(),
                    nn.Linear(160, _lateR))
            else:
                self.ridge_gate_head = None
                self._ridge_ready_flag = False
            self.register_buffer("lib_c2fam", torch.zeros(()), persistent=True)
            if bool(getattr(cfg, "use_profile_attention", True)):
                self.pa_value = nn.Conv1d(fc, fc, 1)
                self.pa_out = nn.Conv1d(fc, fc, 1)
                nn.init.zeros_(self.pa_out.weight)
                nn.init.zeros_(self.pa_out.bias)
                self.pa_log_tau = nn.Parameter(torch.zeros(int(cfg.gates)))
                self.pa_gate = nn.Parameter(torch.tensor(-2.0))
            else:
                self.pa_gate = None
            self.register_buffer("lib_resfam",
                                 torch.zeros(int(cfg.gates)), persistent=True)
            _rC = int(getattr(cfg, "library_subspace_rank", 12))
            self.register_buffer("lib_compA", torch.zeros(_rC + 1, _rC),
                                 persistent=True)
            self.register_buffer("lib_cvar", torch.ones(_rC), persistent=True)
            self.register_buffer("fam_compA_760", torch.zeros(_rC + 1, _rC), persistent=False)
            self.register_buffer("fam_mu_760", torch.zeros(_rC), persistent=False)
            self.register_buffer("fam_dir_760", torch.zeros(_rC), persistent=False)
            self.register_buffer("fam_dband_760", torch.zeros(4), persistent=False)
            self.register_buffer("fam_rho2_760", torch.zeros(int(cfg.gates)), persistent=False)
            self.register_buffer("fam_kappa_760", torch.ones(int(cfg.gates)), persistent=False)
            self.register_buffer("fam_ready_760", torch.zeros(()), persistent=False)
            _K2 = int(max(1, getattr(cfg, "cdmr_prior_components", 64)))
            self.register_buffer("cdmr_ncov", torch.zeros(int(cfg.gates), int(cfg.gates)), persistent=False)
            self.register_buffer("cdmr_pi", torch.zeros(_K2), persistent=False)
            self.register_buffer("cdmr_mu", torch.zeros(_K2, _rC), persistent=False)
            self.register_buffer("cdmr_var", torch.ones(_K2, _rC), persistent=False)
            self.register_buffer("cdmr_ready", torch.zeros(()), persistent=False)
            self.register_buffer("lib_logt", torch.zeros(int(cfg.gates)), persistent=True)
            self.register_buffer("lib_plbeta", torch.zeros(()), persistent=True)
            if bool(getattr(cfg, "use_relaxation_state_update", False)):
                self.relax_state_head = nn.Sequential(
                    ConvGNAct(fc + 2, cfg.base_channels // 2, 3, groups=cfg.group_norm_groups),
                    nn.Conv1d(cfg.base_channels // 2, 1, 3, padding=1),
                )
                self.register_buffer(
                    "relax_smooth_kernel",
                    torch.tensor([[[0.25, 0.50, 0.25]]], dtype=torch.float32),
                    persistent=False,
                )
            else:
                self.relax_state_head = None
            if bool(getattr(cfg, "use_event_residual", False)):
                self.event_gate_head = nn.Sequential(
                    ConvGNAct(fc + 3, cfg.base_channels // 2, 3, groups=cfg.group_norm_groups),
                    nn.Conv1d(cfg.base_channels // 2, 1, 3, padding=1),
                )
                self.struct_residual_head = nn.Sequential(
                    ConvGNAct(fc + 3, cfg.base_channels // 2, 3, groups=cfg.group_norm_groups),
                    nn.Conv1d(cfg.base_channels // 2, 1, 3, padding=1),
                )
                if not hasattr(self, "relax_smooth_kernel"):
                    self.register_buffer(
                        "relax_smooth_kernel",
                        torch.tensor([[[0.25, 0.50, 0.25]]], dtype=torch.float32),
                        persistent=False,
                    )
            else:
                self.event_gate_head = None
                self.struct_residual_head = None
        else:
            self.manifold_score = None
            self.manifold_decoder = None
            self.manifold_head_free = None
            self.manifold_head_atoms = None
            self.manifold_blend = None
            self.manifold_innovation = None
            self.manifold_head_ip = None
            self.manifold_gate_ip = None
            self.event_gate_head = None
            self.struct_residual_head = None
            self.relax_state_head = None
        c = cfg.feature_channels
        self.noise_head = nn.Sequential(
            ConvGNAct(c, cfg.base_channels, 3, groups=cfg.group_norm_groups),
            nn.Conv1d(cfg.base_channels, 1, 1),
        )
        self.refine_head = nn.Sequential(
            ConvGNAct(3 * c, c, 3, groups=cfg.group_norm_groups),
            ConvGNAct(c, cfg.base_channels, 3, groups=cfg.group_norm_groups),
            nn.Conv1d(cfg.base_channels, 1, 1),
        )
        self.uncertainty_head = (
            nn.Sequential(
                ConvGNAct(c, cfg.base_channels // 2, 3, groups=cfg.group_norm_groups),
                nn.Conv1d(cfg.base_channels // 2, 1, 1),
            )
            if cfg.use_uncertainty_head
            else None
        )
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.noise_head[-1].weight)
        nn.init.zeros_(self.noise_head[-1].bias)
        nn.init.zeros_(self.refine_head[-1].weight)
        nn.init.zeros_(self.refine_head[-1].bias)
        if getattr(self, "relax_state_head", None) is not None:
            nn.init.zeros_(self.relax_state_head[-1].weight)
            nn.init.zeros_(self.relax_state_head[-1].bias)
        if getattr(self, "event_gate_head", None) is not None:
            nn.init.normal_(self.event_gate_head[-1].weight, std=0.02)
            nn.init.constant_(self.event_gate_head[-1].bias, -4.0)
            nn.init.normal_(self.struct_residual_head[-1].weight, std=0.01)
            nn.init.zeros_(self.struct_residual_head[-1].bias)
            self.event_lat_gain = nn.Parameter(torch.tensor(1.5))
        if self.uncertainty_head is not None:
            nn.init.zeros_(self.uncertainty_head[-1].weight)
            nn.init.constant_(
                self.uncertainty_head[-1].bias,
                -4.0 if bool(getattr(self.cfg, "innovation_precision_weighting", True)) else 0.0)
        if getattr(self, "lib_gate_head", None) is not None:
            nn.init.normal_(self.lib_gate_head[-1].weight, std=0.02)
            nn.init.constant_(
                self.lib_gate_head[-1].bias,
                float(getattr(self.cfg, "library_gate_bias_init", -2.2)))
            if (bool(getattr(self.cfg, "use_coef_field", True))
                    and bool(getattr(self.cfg, "coef_field_learned_delta", True))
                    and getattr(self, "lib_coef_head", None) is not None):
                nn.init.zeros_(self.lib_coef_head[-1].weight)
                nn.init.zeros_(self.lib_coef_head[-1].bias)
        if getattr(self, "ridge_gate_head", None) is not None:
            nn.init.normal_(self.ridge_gate_head[-1].weight, std=0.02)
            nn.init.constant_(
                self.ridge_gate_head[-1].bias,
                float(getattr(self.cfg, "ridge_gate_bias_init", -1.8)))
        if getattr(self, "pa_out", None) is not None:
            nn.init.zeros_(self.pa_out.weight)
            nn.init.zeros_(self.pa_out.bias)
        if self.manifold_blend is not None:
            nn.init.zeros_(self.manifold_blend[-1].weight)
            nn.init.constant_(self.manifold_blend[-1].bias, -2.0)
        if self.manifold_innovation is not None:
            nn.init.zeros_(self.manifold_innovation[-1].weight)
            nn.init.constant_(
                self.manifold_innovation[-1].bias,
                2.0 if bool(getattr(self.cfg, "innovation_precision_weighting", True))
                else (-1.0 if bool(getattr(self.cfg, "innovation_evidence_gate", False)) else -4.0))

    def set_row_completion(self, W: np.ndarray, mx: np.ndarray, my: np.ndarray, gate_limit: int) -> None:
        """Install the early/mid -> mid/late completion operator (input-unit z domain: z = symlog(x /
        gate_scale_norm) on the pre-trace-scale input).
        """
        dev = self.gate_scale_norm.device
        self.register_buffer("row_completion_W_713", torch.as_tensor(np.asarray(W, dtype=np.float32), device=dev), persistent=False)
        self.register_buffer("row_completion_mx_713", torch.as_tensor(np.asarray(mx, dtype=np.float32).reshape(-1), device=dev), persistent=False)
        self.register_buffer("row_completion_my_713", torch.as_tensor(np.asarray(my, dtype=np.float32).reshape(-1), device=dev), persistent=False)
        self.row_completion_gate_limit = int(gate_limit)

    def set_gate_scale(self, gate_scale_norm: np.ndarray, *, recalibrate: bool = False) -> None:
        """Install the per-gate equalization vector (globally-scaled units)."""
        vec = torch.as_tensor(np.asarray(gate_scale_norm, dtype=np.float64), dtype=torch.float32)
        vec = vec.reshape(1, 1, -1)
        if vec.shape[-1] != self.gate_scale_norm.shape[-1]:
            raise ValueError(
                f"gate_scale length {vec.shape[-1]} does not match configured gates "
                f"{self.gate_scale_norm.shape[-1]}."
            )
        if not torch.isfinite(vec).all() or bool((vec <= 0).any()):
            raise ValueError("gate_scale must be finite and strictly positive.")
        with torch.no_grad():
            self.gate_scale_norm.copy_(vec.to(self.gate_scale_norm.device))
        if recalibrate:
            self._calibrate_manifold_atoms()

    @staticmethod
    def _build_position_channels(times: np.ndarray) -> torch.Tensor:
        t = np.asarray(times, dtype=np.float64).reshape(-1)
        logt = log_time_axis(t)
        logt = 2.0 * (logt - logt.min()) / max(logt.max() - logt.min(), 1e-12) - 1.0
        sq = np.sqrt(t)
        sq = 2.0 * (sq - sq.min()) / max(sq.max() - sq.min(), 1e-12) - 1.0
        return torch.as_tensor(np.stack([logt, sq]), dtype=torch.float32).unsqueeze(0)

    @staticmethod
    def _build_decay_atoms(
        times_ms: np.ndarray,
        atom_count: int,
        stretches: Tuple[float, ...] = (1.0,),
    ) -> torch.Tensor:
        t = np.asarray(times_ms, dtype=np.float64).reshape(-1)
        taus = decay_tau_grid(t, int(atom_count))
        blocks = []
        for c in stretches:
            cc = float(max(0.05, min(1.0, c)))
            blocks.append(np.exp(-np.power(t[None, :] / taus[:, None], cc)))
        A = np.concatenate(blocks, axis=0)
        return torch.as_tensor(A, dtype=torch.float32)

    @staticmethod
    def _build_ip_atoms(
        times_ms: np.ndarray,
        late_start: int,
        atom_count: int,
        tau_min_factor: float,
        tau_max_factor: float,
    ) -> torch.Tensor:
        t = np.asarray(times_ms, dtype=np.float64).reshape(-1)
        lo = float(t[int(np.clip(late_start, 0, len(t) - 1))]) * float(tau_min_factor)
        hi = float(t[-1]) * float(tau_max_factor)
        taus = np.geomspace(max(lo, 1e-9), max(hi, lo * 10.0), int(atom_count))
        A = np.exp(-t[None, :] / taus[:, None])
        return torch.as_tensor(A, dtype=torch.float32)

    def _calibrate_manifold_atoms(self) -> None:
        if getattr(self, "manifold_head_atoms", None) is None:
            return
        with torch.no_grad():
            gs = self.gate_scale_norm.view(-1).float().cpu()
            A = self.decay_atoms.float().cpu()
            B = A / gs.view(1, -1)
            target = torch.ones_like(gs)
            BBt = B @ B.T
            Bt = B @ target
            w = torch.full((A.shape[0],), 0.05)
            for _ in range(4000):
                w = w * Bt / torch.clamp(BBt @ w, min=1e-12)
            w = torch.clamp(w, min=1e-6, max=60.0)
            bias = torch.log(torch.expm1(w))
            bias = torch.where(torch.isfinite(bias), bias, torch.full_like(bias, -6.0))
            self.manifold_head_atoms.weight.zero_()
            self.manifold_head_atoms.bias.copy_(bias.to(self.manifold_head_atoms.bias.device))
            self.manifold_head_free.weight.zero_()
            self.manifold_head_free.bias.zero_()
            if self.manifold_head_ip is not None and self.manifold_gate_ip is not None:
                total = float(torch.clamp(w, min=1e-6).sum())
                amp = max(total * float(self.cfg.ip_init_scale) / max(int(self.cfg.ip_atom_count), 1), 1e-8)
                ip_bias = math.log(math.expm1(amp)) if amp > 1e-7 else math.log(amp)
                self.manifold_head_ip.weight.zero_()
                nn.init.constant_(self.manifold_head_ip.bias, float(ip_bias))
                self.manifold_gate_ip.weight.zero_()
                nn.init.constant_(self.manifold_gate_ip.bias, -4.0)

    def cdmr_calibrated_ready(self) -> bool:
        """The calibrated CDM-R is installed for the decay manifold in force."""
        rd = getattr(self, "cdmr_ready", None)
        if rd is None or not bool(getattr(self.cfg, "cdmr_calibrated", True)) or float(rd.item()) <= 0.5:
            return False
        if getattr(self, "lib_basis", None) is None or float(self.lib_ready.item()) <= 0.5:
            return False
        sig = getattr(self, "_cdmr_sig", None)
        return sig is not None and sig == cdmr_basis_signature(self)

    def _cdmr_mixture_posterior(self, za: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        B, K1, T = za.shape
        f64 = torch.float64
        Vf = self.lib_basis.to(dtype=f64)
        keep = Vf.abs().sum(dim=1) > 0
        V = Vf[keep]
        mu = self.lib_mu.to(dtype=f64).reshape(-1)
        C0 = self.cdmr_ncov.to(dtype=f64)
        Rm = torch.diag((self.lib_resfam.to(dtype=f64).reshape(-1) / 1.96).pow(2).clamp_min(1e-10))
        use = self.cdmr_pi > 0
        lpi = torch.log(self.cdmr_pi[use].to(dtype=f64))
        gm = self.cdmr_mu.to(dtype=f64)[use][:, keep]
        gv = self.cdmr_var.to(dtype=f64)[use][:, keep].clamp_min(1e-12)
        Pinv = 1.0 / gv
        logdetP = torch.log(gv).sum(dim=-1)
        levels = [float(s) for s in (getattr(self.cfg, "cdmr_noise_levels", (1.0,)) or (1.0,))]
        z = za.to(dtype=f64).reshape(B * K1, T)
        rec = (m[:, 0] > 0.5).to(torch.uint8).unsqueeze(1).expand(B, K1, T).reshape(B * K1, T)
        out = z.clone()
        pats, inv = torch.unique(rec.cpu(), dim=0, return_inverse=True)
        inv = inv.to(device=z.device)
        for p in range(int(pats.shape[0])):
            n_rec = int(pats[p].sum())
            if n_rec == T:
                continue
            mp = pats[p].to(device=z.device, dtype=f64)
            sel_all = torch.nonzero(inv == p).reshape(-1)
            if n_rec == 0:
                out[sel_all] = mu + (torch.softmax(lpi, dim=0).unsqueeze(1) * gm).sum(dim=0) @ V
                continue
            mm = mp.view(-1, 1) * mp.view(1, -1)
            facts = []
            for s in levels:
                L = torch.linalg.cholesky(((s * s) * C0 + Rm) * mm + torch.diag(1.0 - mp))
                Lam = torch.cholesky_inverse(L) * mm
                A = V @ Lam @ V.T
                Lg = torch.linalg.cholesky(A.unsqueeze(0) + torch.diag_embed(Pinv))
                facts.append((Lam, A, torch.cholesky_inverse(Lg),
                              2.0 * torch.log(torch.diagonal(L)).sum(),
                              2.0 * torch.log(torch.diagonal(Lg, dim1=-2, dim2=-1)).sum(dim=-1)))
            for c0 in range(0, int(sel_all.numel()), 4096):
                sel = sel_all[c0:c0 + 4096]
                d = (z[sel] - mu) * mp
                logws, means = [], []
                for Lam, A, Gi, logdetC, logdetG in facts:
                    b = d @ Lam @ V.T
                    dLd = ((d @ Lam) * d).sum(dim=-1)
                    ak = torch.einsum("kij,nkj->nki", Gi, b.unsqueeze(1) + (Pinv * gm).unsqueeze(0))
                    Am = gm @ A
                    eLe = dLd.unsqueeze(1) - 2.0 * (b @ gm.T) + (Am * gm).sum(dim=-1).unsqueeze(0)
                    cc = b.unsqueeze(1) - Am.unsqueeze(0)
                    cGc = torch.einsum("nki,kij,nkj->nk", cc, Gi, cc)
                    logws.append(lpi.unsqueeze(0) - 0.5 * (eLe - cGc + logdetC + logdetP.unsqueeze(0)
                                                            + logdetG.unsqueeze(0)))
                    means.append(ak)
                w = torch.softmax(torch.cat(logws, dim=1), dim=1)
                a = (w.unsqueeze(-1) * torch.cat(means, dim=1)).sum(dim=1)
                out[sel] = mu + a @ V
        return out.reshape(B, K1, T).to(dtype=za.dtype)

    def cdmr_continuation(self, za: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Conditional decay-manifold regeneration of every trace of a stack from its recorded gates."""
        B, K1, T = za.shape
        za = za.float()
        m = m.float()
        if self.cdmr_calibrated_ready():
            return self._cdmr_mixture_posterior(za, m)
        if (getattr(self, "lib_basis", None) is not None and float(self.lib_ready.item()) > 0.5
                and float(self.lib_basis.abs().sum().item()) > 0.0):
            V = self.lib_basis.float()
            mu = self.lib_mu.float().view(1, 1, T)
            cvar = self.lib_cvar.float().view(-1).clamp_min(1e-8)
            s2 = (self.lib_resfam.float().view(1, 1, T) / 1.96).pow(2).clamp_min(1e-6)
            if K1 > 1:
                med = za.median(dim=1, keepdim=True).values
                o2 = (1.4826 * (za - med).abs().median(dim=1, keepdim=True).values).pow(2)
            else:
                o2 = torch.zeros_like(za[:, :1])
            zc = za[:, :1]
            d2 = torch.zeros_like(zc)
            d2[..., 1:-1] = (zc[..., 2:] - 2.0 * zc[..., 1:-1] + zc[..., :-2]).pow(2) / 6.0
            d2[..., 0] = d2[..., 1]
            d2[..., -1] = d2[..., -2]
            c2f = float(self.lib_c2fam.item()) if getattr(self, "lib_c2fam", None) is not None else 0.0
            o2 = torch.maximum(o2, torch.relu(d2 - (c2f * c2f) / 6.0))
            w = m / (s2 + o2)
            G = torch.einsum("rt,bt,ut->bru", V, w[:, 0], V) + torch.diag_embed(1.0 / cvar).unsqueeze(0)
            rhs = torch.einsum("rt,bkt->bkr", V, w * (za - mu))
            a = torch.linalg.solve(G.unsqueeze(1).expand(B, K1, -1, -1).contiguous(), rhs.unsqueeze(-1)).squeeze(-1)
            return mu + torch.einsum("bkr,rt->bkt", a, V)
        j0 = (m[:, 0].sum(dim=-1).long() - 1).clamp(min=1)
        _tg = self.gate_times.float().view(-1)
        lt = torch.log(_tg.clamp_min(1e-12))
        if lt.numel() >= 3:
            lt = torch.cat([torch.where(_tg[0] > 0, lt[0], 2.0 * lt[1] - lt[2]).view(1), lt[1:]])
        out = za.clone()
        for b in range(B):
            j = int(j0[b].item())
            y1 = signed_symexp(za[b, :, j])
            y0 = signed_symexp(za[b, :, j - 1])
            ok = (y1 > 0) & (y0 > 0)
            sl = torch.where(ok, (torch.log(y1.clamp_min(1e-30)) - torch.log(y0.clamp_min(1e-30))) / (lt[j] - lt[j - 1]),
                             torch.zeros_like(y1)).clamp(max=0.0)
            ly = torch.log(y1.abs().clamp_min(1e-30)).view(-1, 1) + sl.view(-1, 1) * (lt[j + 1:] - lt[j]).view(1, -1)
            out[b, :, j + 1:] = signed_symlog(torch.sign(y1).view(-1, 1) * torch.exp(ly))
        return out

    def _rate_bound(self, zf: torch.Tensor, m: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        B, K1, T = zf.shape
        t = self.gate_times.float().view(-1)
        y = signed_symexp(zf) * g.view(1, 1, -1)
        out = zf.clone()
        nrec = m[:, 0].sum(dim=-1).long()
        for b in range(B):
            j0 = int(nrec[b].item()) - 1
            if j0 < 2 or j0 >= T - 1:
                continue
            yb = y[b, :, j0 - 2:]
            ok = (yb > 0).all(dim=-1)
            if not bool(ok.any()):
                continue
            ly = torch.log(yb.clamp_min(1e-30))
            dt = (t[j0 - 2 + 1:] - t[j0 - 2:-1]).clamp_min(1e-12)
            rate = -(ly[:, 1:] - ly[:, :-1]) / dt.view(1, -1)
            nu = rate[:, :2].mean(dim=-1, keepdim=True).clamp_min(0.0)
            r_tail = torch.minimum(rate[:, 2:].clamp_min(0.0), nu)
            ly_new = ly[:, 2:3] - torch.cumsum(r_tail * dt[2:].view(1, -1), dim=-1)
            zt = signed_symlog(torch.exp(ly_new) / g.view(-1)[j0 + 1:].view(1, -1))
            out[b, :, j0 + 1:] = torch.where(ok.view(-1, 1), zt, zf[b, :, j0 + 1:])
        return out

    def _gate_loss_fill(self, x: torch.Tensor, nb: Optional[torch.Tensor], obs_mask: torch.Tensor,
                            gs: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        m = obs_mask.to(device=x.device, dtype=torch.float32).reshape(x.shape[0], 1, -1)
        if m.shape[-1] != x.shape[-1] or not bool((m < 0.5).any()):
            return x, nb, None
        with torch.no_grad(), amp_autocast_context(x.device, enabled=False):
            g = gs.float().reshape(1, 1, -1)
            za = signed_symlog(x.float() / g)
            if nb is not None and nb.shape[1] > 0:
                za = torch.cat([za, signed_symlog(nb.float() / g)], dim=1)
            if (bool(getattr(self, "_ablate_cdmr", False))
                    or not bool(getattr(self.cfg, "gate_loss_continuation", True))):
                zf = torch.zeros_like(za)
            else:
                zf = self.cdmr_continuation(za, m)
                if str(getattr(self.cfg, "gate_loss_rate_bound", "off")).lower() in ("input", "on", "both"):
                    zf = self._rate_bound(zf, m, g)
            yf = signed_symexp(zf) * g
        withheld = m < 0.5
        x_new = torch.where(withheld, yf[:, :1].to(dtype=x.dtype), x)
        nb_new = nb
        if nb is not None and nb.shape[1] > 0:
            nb_new = torch.where(withheld.expand(-1, nb.shape[1], -1), yf[:, 1:].to(dtype=nb.dtype), nb)
        return x_new, nb_new, {"mask": m, "z_fill": zf[:, :1],
                               "cdmr": not (bool(getattr(self, "_ablate_cdmr", False))
                                            or not bool(getattr(self.cfg, "gate_loss_continuation", True)))}

    def _withheld_departure(self, z_hat: torch.Tensor, gl: Mapping[str, torch.Tensor],
                                trace_scale: Optional[torch.Tensor]) -> torch.Tensor:
        m = gl["mask"].to(device=z_hat.device, dtype=torch.float32)
        T = int(z_hat.shape[-1])
        zf = gl["z_fill"].to(device=z_hat.device, dtype=torch.float32)
        if trace_scale is not None:
            zf = signed_symlog(signed_symexp(zf) * trace_scale.float())
        n_rec = m.sum(dim=-1, keepdim=True)
        k = (torch.arange(T, device=z_hat.device).view(1, 1, -1).float() - n_rec).clamp(0, T - 1).long()
        g = torch.sigmoid(self.gl_departure_logit.float())[k]
        zh = z_hat.float()
        zw = zh + (g - 1.0) * (zh - zf).detach()
        return torch.where(m < 0.5, zw, zh).to(dtype=z_hat.dtype)

    def informative_mask(self, q: torch.Tensor) -> torch.Tensor:
        """Per trace, 1 up to the last informative gate and 0 after it."""
        qf = q.float().reshape(q.shape[0], -1)
        B, T = qf.shape
        thr = math.exp(-float(getattr(self.cfg, "informative_kappa", 1.5)))
        bad = qf < thr
        both = torch.zeros_like(bad)
        both[:, :-1] = bad[:, :-1] & bad[:, 1:]
        both[:, -1] = bad[:, -1]
        j_min = int(max(2, min(int(getattr(self.cfg, "informative_min_gates", 8)), T)))
        both[:, :j_min] = False
        idx = torch.arange(T, device=qf.device).view(1, -1).expand(B, T)
        first = torch.where(both, idx, torch.full_like(idx, T)).min(dim=1).values
        m = (idx < first.view(-1, 1)).to(dtype=torch.float32)
        return m.view(B, 1, T)

    def _informative_forward(self, x: torch.Tensor, neighbors: Optional[torch.Tensor],
                                 neighbor_geometry: Optional[torch.Tensor], depth_norm: Optional[torch.Tensor],
                                 profile_len: Optional[int]) -> Dict[str, torch.Tensor]:
        self._informative_pass = True
        try:
            o1 = self.forward(x, neighbors, neighbor_geometry, depth_norm=depth_norm, profile_len=profile_len)
            if "reliability" not in o1:
                return o1
            m = self.informative_mask(o1["reliability"].detach())
            if not bool((m < 0.5).any()):
                o1["informative_mask"] = m
                return o1
            o2 = self.forward(x, neighbors, neighbor_geometry, depth_norm=depth_norm, profile_len=profile_len,
                              obs_mask=m.to(device=x.device), record_evidence=True)
            o2["informative_mask"] = m
            return o2
        finally:
            self._informative_pass = False

    def set_gate_times(self, times_ms: np.ndarray, *, recalibrate: bool = False) -> None:
        """Install the resolved gate times and rebuild the decay-atom basis so the physics head extrapolates on the
        true time axis.
        """
        t = np.asarray(times_ms, dtype=np.float64).reshape(-1)
        if t.shape[0] != int(self.cfg.gates):
            raise ValueError(f"gate_times length {t.shape[0]} != configured gates {self.cfg.gates}.")
        try:
            validate_gate_axis(t)
        except ValueError as _e:
            raise ValueError("gate_times must be finite, strictly increasing and positive (the first gate may be "
                             "t = 0): %s" % _e) from None
        with torch.no_grad():
            self.gate_times.copy_(
                torch.as_tensor(t, dtype=torch.float32).view(1, -1).to(self.gate_times.device)
            )
            self.decay_atoms.copy_(
                self._build_decay_atoms(
                    t,
                    int(self.decay_atoms.shape[0]) // max(1, len(self._atom_stretches)),
                    self._atom_stretches,
                ).to(self.decay_atoms.device)
            )
            if getattr(self, "manifold_head_ip", None) is not None:
                self.ip_atoms.copy_(
                    self._build_ip_atoms(
                        t,
                        int(self.cfg.late_start_index),
                        int(self.ip_atoms.shape[0]),
                        float(self.cfg.ip_tau_min_factor),
                        float(self.cfg.ip_tau_max_factor),
                    ).to(self.ip_atoms.device)
                )
            self.gate_position.copy_(self._build_position_channels(t).to(self.gate_position.device))
        if recalibrate:
            self._calibrate_manifold_atoms()

    @torch.no_grad()
    def set_manifold_quality(self, manifold_late_error: float,
                             denoise_late_error: Optional[float] = None,
                             val_floor_enabled: bool = True,
                             val_floor_max: float = 0.93) -> Tuple[float, float, float]:
        """Slave the late blend ceiling to the prior's own measured error."""
        e = float(manifold_late_error)
        if not math.isfinite(e):
            return (float(self.blend_late_quality_cap.item()),
                    float(self.blend_floor_scale.item()),
                    float(self.blend_late_val_floor.item()))
        c_lo = float(self.cfg.manifold_cap_min)
        c_hi = float(self.cfg.manifold_cap_max)
        e0 = float(self.cfg.manifold_cap_error_midpoint)
        _s_log = max(float(getattr(self.cfg, "manifold_cap_error_log_scale", 0.55)),
                     1e-6)
        _r = max(e, 1e-9) / max(e0, 1e-9)
        cap = c_lo + (c_hi - c_lo) / (1.0 + math.exp(math.log(_r) / _s_log))
        fs = 0.5 + 0.5 / (1.0 + math.exp(-(0.20 - e) / 0.02))
        if not bool(getattr(self.cfg, "manifold_quality_gate", True)):
            cap, fs = float(self.cfg.manifold_cap_max), 1.0
        else:
            prev = float(self.blend_late_quality_cap.item())
            step = float(self.cfg.manifold_cap_max_step)
            if cap > prev + step:
                cap = prev + step
        self.blend_late_quality_cap.fill_(cap)
        self.blend_floor_scale.fill_(fs)
        vfl = float(self.blend_late_val_floor.item())
        if val_floor_enabled and denoise_late_error is not None:
            d = float(denoise_late_error)
            if math.isfinite(d) and d > 0.0:
                p = max(e, 1e-4)
                b_star = (d * d) / (p * p + d * d)
                tgt = min(max(b_star - 0.03, 0.0), cap - 4e-3, float(val_floor_max))
                vfl = tgt if tgt > vfl else max(vfl - 0.02, tgt)
                vfl = min(vfl, float(val_floor_max))
                self.blend_late_val_floor.fill_(max(vfl, 0.0))
        return cap, fs, vfl

    def set_library_subspace(self, mu: np.ndarray, basis: np.ndarray) -> None:
        """Install the survey-library subspace (mean + top-r rows of V)."""
        if getattr(self, "lib_coef_head", None) is None:
            return
        with torch.no_grad():
            g = int(self.lib_mu.shape[-1])
            self.lib_mu.copy_(torch.as_tensor(
                np.asarray(mu, dtype=np.float32).reshape(1, 1, g)))
            b = np.zeros((int(self.lib_basis.shape[0]), g), dtype=np.float32)
            r0 = min(int(basis.shape[0]), b.shape[0])
            b[:r0] = np.asarray(basis[:r0], dtype=np.float32)
            self.lib_basis.copy_(torch.as_tensor(b))
            self.lib_ready.fill_(1.0)
        self._lib_ready_flag = True

    def set_powerlaw(self, logt: np.ndarray, beta: float) -> None:
        """Install log gate times and the physics-extrapolation blend."""
        with torch.no_grad():
            self.lib_logt.copy_(torch.as_tensor(np.asarray(logt, dtype=np.float32).reshape(-1)))
            self.lib_plbeta.fill_(float(max(0.0, min(1.0, beta))))

    def set_seam_blend_floor(self, floor: np.ndarray, ema: float = 0.5) -> None:
        """Install (EMA-smoothed) the validation-calibrated per-gate blend floor; entries outside the seam band must
        be 0.
        """
        _f = np.clip(np.asarray(floor, dtype=np.float64).reshape(-1), 0.0, 0.95)
        if _f.size != int(self.seam_blend_floor.numel()) or not np.all(np.isfinite(_f)):
            return
        with torch.no_grad():
            _old = self.seam_blend_floor.detach().cpu().numpy().astype(np.float64)
            _new = np.where(_old > 0.0, ema * _old + (1.0 - ema) * _f, _f)
            _new = np.where(_f > 0.0, _new, 0.0)
            self.seam_blend_floor.copy_(torch.as_tensor(_new, dtype=self.seam_blend_floor.dtype))

    def set_mmse_bstar(self, bstar: np.ndarray, ema: float = 0.5) -> None:
        """Refresh the per-gate covariance-optimal prior share from a validation pass (EMA-smoothed), used by the
        witnessed MMSE blend on the late window.
        """
        v = torch.as_tensor(np.asarray(bstar, dtype=np.float32).reshape(-1), device=self.mmse_bstar_714.device)
        if v.numel() != self.mmse_bstar_714.numel():
            return
        ok = torch.isfinite(v)
        new = torch.where(ok, v.clamp(0.02, 0.98), self.mmse_bstar_714)
        self.mmse_bstar_714.mul_(1.0 - float(ema)).add_(float(ema) * new)

    def queue_seam_blend_floor(self, floor: np.ndarray) -> None:
        """Record a validation-calibrated floor without installing it."""
        self._queued_seam_floor = np.asarray(floor, dtype=np.float64).reshape(-1).copy()

    def apply_queued_seam_blend_floor(self, logger: Optional[logging.Logger] = None) -> bool:
        """Install the queued floor (EMA-smoothed by set_seam_blend_floor)."""
        _q = getattr(self, "_queued_seam_floor", None)
        if _q is None:
            return False
        self._queued_seam_floor = None
        self.set_seam_blend_floor(_q)
        if logger is not None:
            _cur = self.seam_blend_floor.detach().cpu().numpy()
            logger.info("queued seam blend floor INSTALLED for this training epoch: %s.",
                        " ".join("%.2f" % v for v in _cur))
        return True

    _CONTROL_BUFFERS = ("blend_late_quality_cap", "blend_floor_scale", "blend_late_val_floor",
                            "late_prior_advantage", "event_expression_scale")

    def snapshot_control_state(self) -> Dict[str, torch.Tensor]:
        return {n: getattr(self, n).detach().clone() for n in self._CONTROL_BUFFERS
                if isinstance(getattr(self, n, None), torch.Tensor)}

    def stage_control_updates(self, snapshot: Mapping[str, torch.Tensor], source: str = "") -> Dict[str, float]:
        pending: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for n, old in snapshot.items():
                buf = getattr(self, n, None)
                if not isinstance(buf, torch.Tensor):
                    continue
                new = buf.detach().clone()
                if not torch.equal(new, old.to(buf.device, buf.dtype)):
                    pending[n] = new
                    buf.copy_(old.to(buf.device, buf.dtype))
        if pending:
            self._pending_control = {"values": pending, "source": str(source)}
        return {n: float(v.float().mean()) for n, v in pending.items()}

    def commit_control_updates(self, logger: Optional[logging.Logger] = None) -> bool:
        did = self.apply_queued_seam_blend_floor(logger)
        p = getattr(self, "_pending_control", None)
        if p:
            with torch.no_grad():
                for n, v in p["values"].items():
                    buf = getattr(self, n, None)
                    if isinstance(buf, torch.Tensor):
                        buf.copy_(v.to(buf.device, buf.dtype))
            if logger is not None:
                logger.info("control updates COMMITTED at epoch start (source: %s): %s.",
                            p["source"], " ".join("%s=%.4g" % (n, float(v.float().mean())) for n, v in p["values"].items()))
            self._pending_control = None
            did = True
        return did

    def discard_pending_control_updates(self, logger: Optional[logging.Logger] = None, reason: str = "") -> None:
        had = bool(getattr(self, "_pending_control", None)) or getattr(self, "_queued_seam_floor", None) is not None
        self._pending_control = None
        self._queued_seam_floor = None
        if had and logger is not None:
            logger.info("pending control updates DISCARDED (%s): they belonged to a different model state.", reason)

    def export_pending_control(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        p = getattr(self, "_pending_control", None)
        if p:
            out["control"] = {n: v.detach().cpu().numpy().astype(np.float64).tolist() for n, v in p["values"].items()}
            out["source"] = str(p.get("source", ""))
        q = getattr(self, "_queued_seam_floor", None)
        if q is not None:
            out["seam_blend_floor_queued"] = np.asarray(q, dtype=np.float64).reshape(-1).tolist()
        return out

    def import_pending_control(self, d: Any) -> None:
        self._pending_control = None
        self._queued_seam_floor = None
        if not isinstance(d, dict):
            return
        vals: Dict[str, torch.Tensor] = {}
        for n, v in (d.get("control") or {}).items():
            buf = getattr(self, n, None)
            if isinstance(buf, torch.Tensor):
                try:
                    vals[n] = torch.as_tensor(np.asarray(v, dtype=np.float64), dtype=buf.dtype).reshape(buf.shape)
                except Exception:
                    continue
        if vals:
            self._pending_control = {"values": vals, "source": str(d.get("source", "checkpoint"))}
        q = d.get("seam_blend_floor_queued")
        if q is not None:
            self._queued_seam_floor = np.asarray(q, dtype=np.float64).reshape(-1)

    def set_gate_precision(self, w: np.ndarray) -> None:
        """Install the measured per-gate precision profile (any positive scale)."""
        _w = np.asarray(w, dtype=np.float64).reshape(-1)
        if _w.size != int(self.gate_precision.numel()) or not np.all(np.isfinite(_w)) or float(_w.max()) <= 0:
            return
        _w = np.clip(_w / float(_w.max()), 1e-4, 1.0)
        with torch.no_grad():
            self.gate_precision.copy_(torch.as_tensor(_w, dtype=self.gate_precision.dtype))
            self.gate_precision_ready.fill_(1.0)

    def describe_bands(self) -> Dict[str, int]:
        """The three gate boundaries by their physical names: physical_late_start (the contract late window,
        cfg.late_start_index), prior_support_start (the prior band / seam floor start)
        """
        return {"physical_late_start": int(self.cfg.late_start_index),
                "prior_support_start": int(getattr(self, "_prior_band_g", self.cfg.late_start_index)),
                "seam_start": int(getattr(self, "_prior_band_g", self.cfg.late_start_index)),
                "seam_end": int(self.cfg.late_start_index)}

    def seam_floor_mode(self) -> str:
        """The seam-floor mode in force: an explicit hard/soft/off is returned as is; 'auto' is 'hard' while the
        prior band [_prior_band_g, late start) spans at most seam_blend_floor_hard_max_gates gates.
        """
        m = str(getattr(self.cfg, "seam_blend_floor_mode", "hard"))
        if m != "auto":
            return m
        _w = int(self.cfg.late_start_index) - int(getattr(self, "_prior_band_g", int(self.cfg.late_start_index)))
        return "hard" if _w <= int(getattr(self.cfg, "seam_blend_floor_hard_max_gates", 8)) else "soft"

    def structure_witness(self, z: torch.Tensor, profile_len: Optional[int]) -> Optional[torch.Tensor]:
        """Per-station structure witness from the raw early gates (below the prior band), the statistic: f = 1 / (1 +
        (mean_g |D2_station z| / c0)^2) in [0, 1]; 1 = laterally smooth (pool)
        """
        c0 = float(getattr(self.cfg, "lateral_structure_witness_c0", 0.0) or 0.0)
        if profile_len is None or c0 <= 0.0:
            return None
        P = int(profile_len)
        B = int(z.shape[0])
        if P < 3 or B % P != 0:
            return None
        n = B // P
        T = int(z.shape[-1])
        g = int(min(max(1, int(getattr(self, "_prior_band_g", self.cfg.late_start_index))), T - 1))
        with amp_autocast_context(z.device, enabled=False), torch.no_grad():
            ze = z.detach().float().reshape(n, P, T)[:, :, :g]
            d2i = (ze[:, 2:] - 2.0 * ze[:, 1:-1] + ze[:, :-2]).abs().mean(dim=-1)
            d2 = torch.cat([d2i[:, :1], d2i, d2i[:, -1:]], dim=1)
            f = (1.0 / (1.0 + (d2 / c0) ** 2)).reshape(B, 1, 1)
            dz = float(min(0.9, max(0.0, getattr(self.cfg, "lateral_witness_dead_zone", 0.2))))
            f = (f / (1.0 - dz)).clamp(max=1.0)
        return f

    def set_prior_band_start(self, g: int) -> None:
        """Rebuild blend_floor / late_ramp so the prior floor ramps from gate g (the measured noise-jump gate) to the
        contract late start; the late window itself is unchanged.
        """
        cfg = self.cfg
        T = int(cfg.gates)
        ls = int(cfg.late_start_index)
        g = int(max(0, min(int(g), ls)))
        floor = torch.zeros(1, 1, T)
        if bool(cfg.use_manifold_branch):
            fl = float(cfg.manifold_blend_floor_late)
            floor[..., ls:] = fl
            for gg in range(g, ls):
                u = (float(gg - g) + 0.5) / float(max(ls - g, 1))
                floor[..., gg] = fl * (u * u * (3.0 - 2.0 * u))
            lr = (floor / max(fl, 1e-6)).clamp(0.0, 1.0)
        else:
            lr = torch.zeros_like(floor)
            lr[..., ls:] = 1.0
        with torch.no_grad():
            self.blend_floor.copy_(floor.to(dtype=self.blend_floor.dtype, device=self.blend_floor.device))
            self.late_ramp.copy_(lr.to(dtype=self.late_ramp.dtype, device=self.late_ramp.device))
        self._prior_band_g = int(max(1, g))

    def set_coef_prior_var(self, v: np.ndarray) -> None:
        """Library coefficient prior variances."""
        with torch.no_grad():
            self.lib_cvar.copy_(torch.as_tensor(np.asarray(v, dtype=np.float32).reshape(-1)).clamp_min(1e-8))

    def set_field_completion(self, A: np.ndarray) -> None:
        """Install the completion map."""
        with torch.no_grad():
            self.lib_compA.copy_(torch.as_tensor(np.asarray(A, dtype=np.float32)))

    def set_family_completion(self, A: Optional[np.ndarray], mu: Optional[np.ndarray] = None,
                                  direction: Optional[np.ndarray] = None, band: Optional[Sequence[float]] = None,
                                  rho2: Optional[np.ndarray] = None) -> None:
        """Install (A given) or clear (A None) the survey family's completion map, the membership score of the
        observed coefficients -- score = (c - mu) .
        """
        if getattr(self, "fam_ready_760", None) is None:
            return
        with torch.no_grad():
            if A is None:
                self.fam_ready_760.fill_(0.0)
                return
            _b = [float(v) for v in band]
            self.fam_compA_760.copy_(torch.as_tensor(np.asarray(A, dtype=np.float32)))
            self.fam_mu_760.copy_(torch.as_tensor(np.asarray(mu, dtype=np.float32).reshape(-1)))
            self.fam_dir_760.copy_(torch.as_tensor(np.asarray(direction, dtype=np.float32).reshape(-1)))
            self.fam_dband_760.copy_(torch.tensor(_b))
            self.fam_rho2_760.copy_(torch.as_tensor(np.asarray(rho2, dtype=np.float32).reshape(-1)).clamp(1e-6, 1.0e3))
            self.fam_ready_760.fill_(1.0 if (_b[0] < _b[1] <= _b[2] < _b[3]) else 0.0)

    def set_family_fusion(self, kappa: Optional[np.ndarray]) -> None:
        """The validated share of a buried gate the family completion carries for a member row, per gate (None = 1:
        no held-out evidence).
        """
        if getattr(self, "fam_kappa_760", None) is None:
            return
        with torch.no_grad():
            if kappa is None:
                self.fam_kappa_760.fill_(1.0)
            else:
                self.fam_kappa_760.copy_(torch.nan_to_num(torch.as_tensor(
                    np.asarray(kappa, dtype=np.float32).reshape(-1)), nan=1.0).clamp(0.0, 1.0))

    def lateral_common(self, inn: torch.Tensor, profile_len: Optional[int]) -> torch.Tensor:
        """The laterally common part of an innovation [B,1,T] whose rows are n profiles of profile_len consecutive
        stations: the running median over +-family_common_mode_stations stations.
        """
        h = int(getattr(self.cfg, "family_common_mode_stations", 2) or 0)
        if h <= 0 or profile_len is None:
            return inn
        P = int(profile_len)
        B = int(inn.shape[0])
        if P < 3 or B % P != 0:
            return inn
        v = inn.reshape(B // P, P, inn.shape[-1])
        return torch.stack([v[:, max(0, i - h): min(P, i + h + 1), :].median(dim=1).values for i in range(P)],
                           dim=1).reshape_as(inn)

    def set_family_residual(self, res: np.ndarray) -> None:
        """Install the per-gate rank-r residual tolerance."""
        with torch.no_grad():
            self.lib_resfam.copy_(torch.as_tensor(
                np.asarray(res, dtype=np.float32).reshape(-1)))

    def set_family_curvature(self, c2: float) -> None:
        """Install the family curvature bound used by the smoothness trust."""
        with torch.no_grad():
            self.lib_c2fam.fill_(float(max(c2, 0.0)))

    def set_library_ridge(self, W: np.ndarray, b: np.ndarray) -> None:
        """Install the library's closed-form early/mid->late ridge operator (computed by the audit on train rows)."""
        if getattr(self, "ridge_gate_head", None) is None:
            return
        with torch.no_grad():
            self.lib_ridge_W.copy_(torch.as_tensor(
                np.asarray(W, dtype=np.float32)))
            self.lib_ridge_b.copy_(torch.as_tensor(
                np.asarray(b, dtype=np.float32).reshape(-1)))
            self.lib_ridge_ready.fill_(1.0)
        self._ridge_ready_flag = True

    def set_late_path_advantage(self, prior_late_error: float,
                                denoise_late_error: float) -> float:
        """Which branch actually owns the late window, measured."""
        p, d = float(prior_late_error), float(denoise_late_error)
        if not (math.isfinite(p) and math.isfinite(d)) or p <= 0.0 or d <= 0.0:
            return float(self.late_prior_advantage.item())
        r = math.log10(max(d, 1e-12) / max(p, 1e-12))
        adv = 1.0 / (1.0 + math.exp(-(r - 0.30) / 0.15))
        self.late_prior_advantage.fill_(adv)
        return adv

    def set_event_expression(self, event_auroc: float,
                             open_auroc: float = 0.60,
                             chance_band: float = 0.55) -> float:
        """Map the probe's event AUROC to the residual's expression scale."""
        a = float(event_auroc)
        prev = float(self.event_expression_scale.item())
        if not math.isfinite(a):
            return prev
        lo = float(chance_band)
        hi = max(float(open_auroc), lo + 1e-6)
        x = min(max((a - lo) / (hi - lo), 0.0), 1.0)
        target = x * x * (3.0 - 2.0 * x)
        scale = target if target < prev else min(prev + 0.15, target)
        self.event_expression_scale.fill_(scale)
        return float(scale)

    def shared_parameters(self) -> List[nn.Parameter]:
        modules = [self.stem, self.multiscale, self.refinement_stack]
        params: List[nn.Parameter] = []
        for module in modules:
            params.extend([p for p in module.parameters() if p.requires_grad])
        return params

    def _default_neighbor_geometry(self, neighbors: torch.Tensor) -> torch.Tensor:
        b, k = int(neighbors.shape[0]), int(neighbors.shape[1])
        dev = neighbors.device
        j = torch.arange(k, device=dev, dtype=torch.float32)
        mag = torch.div(j, 2, rounding_mode="floor") + 1.0
        sign = torch.where(j.remainder(2.0) == 0, 1.0, -1.0)
        is_stack = (j >= float(int(self.cfg.num_neighbors))).float() if k > int(self.cfg.num_neighbors) else torch.zeros_like(j)
        off = mag * sign * (1.0 - is_stack)
        geo = torch.stack([off, off.abs(), 1.0 - is_stack, is_stack], dim=-1)
        return geo.unsqueeze(0).expand(b, k, geo.shape[-1]).to(dtype=torch.float32)

    def _encode_features(self, z: torch.Tensor) -> torch.Tensor:
        if self.cfg.use_position_encoding:
            pos = self.gate_position.to(dtype=z.dtype).expand(z.shape[0], -1, -1)
            z = torch.cat([z, pos], dim=1)
        stem, _ = self.stem(z)
        features, _ = self.multiscale(stem)
        return self.refinement_stack(features)

    def forward(
        self,
        x: torch.Tensor,
        neighbors: Optional[torch.Tensor] = None,
        neighbor_geometry: Optional[torch.Tensor] = None,
        depth_norm: Optional[torch.Tensor] = None,
        profile_len: Optional[int] = None,
        obs_mask: Optional[torch.Tensor] = None,
        record_evidence: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if (obs_mask is None and not self.training
                and str(getattr(self.cfg, "informative_continuation", "off")).lower() == "on"
                and not bool(getattr(self, "_informative_pass", False))):
            return self._informative_forward(x, neighbors, neighbor_geometry, depth_norm, profile_len)
        gs = self.gate_scale_norm.to(dtype=x.dtype)
        _gl2 = None
        _x_rec, _nb_rec = x, neighbors
        if obs_mask is not None:
            x, neighbors, _gl2 = self._gate_loss_fill(x, neighbors, obs_mask, gs)
        if not record_evidence:
            _x_rec, _nb_rec = x, neighbors
        _x_in = x
        _dec = bool(getattr(self.cfg, "decoupled_arms", False))
        _bnd = bool(getattr(self.cfg, "bounded_input", True))
        _zb = tuple(getattr(self.cfg, "input_z_bounds", (-1.0, 6.0)))
        trace_scale = None
        if self.cfg.per_trace_normalization:
            if str(getattr(self.cfg, "per_trace_norm_estimator", "centre_logmean")) == "stack_rms":
                ng = int(max(1, min(int(self.cfg.per_trace_norm_gates), x.shape[-1])))
                with amp_autocast_context(x.device, enabled=False):
                    xf = x[..., :ng].float()
                    m2 = torch.mean(xf ** 2, dim=-1, keepdim=True)
                    if neighbors is not None and neighbors.shape[1] > 0:
                        allt = torch.cat([x.float(), neighbors.float()], dim=1)[..., :ng]
                        n_tr = float(allt.shape[1])
                        stacked = allt.mean(dim=1, keepdim=True)
                        m2 = torch.mean(stacked ** 2, dim=-1, keepdim=True)
                        sig2 = torch.mean(
                            allt.var(dim=1, unbiased=True).clamp_min(0.0), dim=-1, keepdim=True
                        ).unsqueeze(1) / n_tr
                        m2 = torch.clamp(m2 - sig2, min=0.04 * m2)
                    amp_ref = torch.sqrt(torch.mean(gs[..., :ng].float() ** 2))
                    amp = torch.clamp(torch.sqrt(m2.clamp_min(0.0)), min=1e-6 * amp_ref)
                    trace_scale = torch.clamp(amp_ref / amp, 1e-4, 1e4).to(dtype=x.dtype)
            else:
                ng = int(max(2, min(int(self.cfg.per_trace_norm_gates), int(self.cfg.late_start_index) // 8, x.shape[-1])))
                with amp_autocast_context(x.device, enabled=False):
                    _u = (x[..., :ng].float().abs() / gs[..., :ng].float().clamp_min(1e-30)).clamp_min(1e-30)
                    trace_scale = torch.exp(-torch.quantile(torch.log(_u), 0.5, dim=-1, keepdim=True)).clamp(1e-4, 1e4).to(dtype=x.dtype)
            x = x * trace_scale
            if neighbors is not None:
                neighbors = neighbors * trace_scale
            if record_evidence:
                _x_rec = _x_rec * trace_scale
                if _nb_rec is not None:
                    _nb_rec = _nb_rec * trace_scale
        if not record_evidence:
            _x_rec, _nb_rec = x, neighbors
        x_eq = x / gs
        z = signed_symlog(x_eq)
        _fatt = (self.structure_witness(z, profile_len)
                    if bool(getattr(self.cfg, "lateral_witness_gates_attention", False)) else None)
        if _fatt is not None:
            self._last_attention_witness = _fatt.reshape(-1).detach()
        if _bnd:
            z = torch.clamp(z, float(_zb[0]), float(_zb[1]))
        _z_ev = z
        if record_evidence:
            _z_ev = signed_symlog(_x_rec / gs)
            if _bnd:
                _z_ev = torch.clamp(_z_ev, float(_zb[0]), float(_zb[1]))
        if self.cfg.use_position_encoding:
            pos = self.gate_position.to(dtype=z.dtype).expand(z.shape[0], -1, -1)
            stem_in = torch.cat([z, pos], dim=1)
        else:
            stem_in = z
        stem, domain_gate = self.stem(stem_in)
        features, scale_disagreement = self.multiscale(stem)
        features = self.refinement_stack(features)
        neighbor_context = None
        meas_sigma_z = None
        meas_reliability = None
        meas_sigma_u = None
        meas_snr_gate = None
        meas_n_eff = None
        _ctx2 = None
        _P = None
        _wkal = None
        lateral_strength = None
        strength_prior = None
        meas_agg_abs = None
        _rt2 = None
        _infl_rec = None
        _wout = None
        z_agg = z
        if neighbors is not None and self.neighbor_attention is not None and neighbors.shape[1] > 0:
            bn, kn, tn = neighbors.shape
            nb_eq = neighbors / gs
            all_eq = torch.cat([x_eq, nb_eq], dim=1)
            agg_mean = all_eq.mean(dim=1, keepdim=True)
            z_agg = signed_symlog(agg_mean)
            if _fatt is not None:
                z_agg = z + _fatt.to(z_agg.dtype) * (z_agg - z)
            agg_median = all_eq.median(dim=1).values.unsqueeze(1)
            with amp_autocast_context(x.device, enabled=False):
                all32 = all_eq.float()
                n_tr = float(all32.shape[1])
                agg32 = all32.mean(dim=1, keepdim=True)
                var32 = all32.var(dim=1, unbiased=True).clamp_min(0.0)
                sigma_u = torch.sqrt(var32 + 1e-12).unsqueeze(1)
                jac_z = 1.0 / (1.0 + agg32.abs())
                _rho = float(getattr(self.cfg, "neighbor_noise_equicorrelation", 0.5))
                _n_eff = n_tr / (1.0 + (n_tr - 1.0) * max(0.0, min(0.999, _rho)))
                meas_sigma_z = (sigma_u * jac_z / math.sqrt(_n_eff)).clamp(1e-4, 1e2)
                meas_sigma_u = sigma_u.clamp_min(1e-12)
                snr_gate = (agg32.abs() / (sigma_u / math.sqrt(_n_eff) + 1e-8)).clamp(max=1.0e3)
                meas_reliability = snr_gate.pow(2) / (snr_gate.pow(2) + 1.0)
                meas_snr_gate = snr_gate
                meas_n_eff = float(_n_eff)
                meas_agg_abs = agg32.abs()
            tokens_eq = torch.cat([nb_eq, agg_mean, agg_median], dim=1)
            kk = tokens_eq.shape[1]
            tokens_z = signed_symlog(tokens_eq).reshape(bn * kk, 1, tn)
            if _bnd:
                tokens_z = torch.clamp(tokens_z, float(_zb[0]), float(_zb[1]))
            tokens_features = self._encode_features(tokens_z)
            tokens_features = tokens_features.reshape(bn, kk, tokens_features.shape[1], tokens_features.shape[2])
            geo = neighbor_geometry
            if geo is None and bool(getattr(self.cfg, "use_neighbor_geometry", False)):
                geo = self._default_neighbor_geometry(neighbors)
            elif geo is not None and geo.shape[1] != neighbors.shape[1]:
                geo = self._default_neighbor_geometry(neighbors)
            if geo is not None:
                _pad_n = int(tokens_features.shape[1]) - int(geo.shape[1])
                if _pad_n > 0:
                    _pad = geo.new_zeros(geo.shape[0], _pad_n, geo.shape[2])
                    _pad[..., 3] = 1.0
                    geo = torch.cat([geo, _pad], dim=1)
                elif _pad_n < 0:
                    geo = geo[:, : tokens_features.shape[1]]
            neighbor_context = self.neighbor_attention(features, tokens_features, geo)
            if getattr(self, "pa_gate", None) is not None:
                try:
                    with amp_autocast_context(x.device, enabled=False):
                        _tf = tokens_features.float()
                        _Bq, _Kq, _Cq, _Tq = _tf.shape
                        _q = features.float().unsqueeze(1)
                        _d2 = ((_tf - _q) ** 2).mean(dim=2)
                        _tau = F.softplus(self.pa_log_tau).view(1, 1, -1) + 1e-6
                        _wat = torch.softmax(-_d2 / _tau, dim=1)
                        _v = self.pa_value(_tf.reshape(_Bq * _Kq, _Cq, _Tq)
                                           ).reshape(_Bq, _Kq, _Cq, _Tq)
                        _ctx = (_wat.unsqueeze(2) * _v).sum(dim=1)
                        _ctx = self.pa_out(_ctx) * torch.sigmoid(self.pa_gate)
                    neighbor_context = neighbor_context + _ctx.to(neighbor_context.dtype)
                except Exception:
                    pass
            if _fatt is not None:
                neighbor_context = neighbor_context * _fatt.to(neighbor_context.dtype)
            features = features + neighbor_context
        q_logits, q = self.router(features, z, scale_disagreement)
        noise_z = self.noise_head(features)
        base_z = torch.clamp(z - noise_z, -16.0, 16.0)
        with amp_autocast_context(x.device, enabled=False):
            _xeq = signed_symexp(z.float())
            _seq = signed_symexp(base_z.float())
            _neq = _xeq - _seq
            _ls3 = int(self.cfg.late_start_index)
            _ms2 = max(0, _ls3 // 2)
            _T2 = int(z.shape[-1])
            _rb = torch.ones(z.shape[0], 1, _T2, device=x.device, dtype=torch.float32)
            _pg = int(getattr(self, "_prior_band_g", -1))
            _cuts = sorted({0, _ms2, _ls3, _T2} | ({_pg} if 0 < _pg < _ls3 else set()))
            for _a, _b in zip(_cuts[:-1], _cuts[1:]):
                if _b - _a >= 1:
                    _sr = torch.sqrt(_seq[..., _a:_b].pow(2).mean(dim=-1, keepdim=True) + 1e-12)
                    _nr = torch.sqrt(_neq[..., _a:_b].pow(2).mean(dim=-1, keepdim=True) + 1e-12)
                    _sn = (_sr / (_nr + 1e-6)).clamp(max=1.0e3)
                    _rb[..., _a:_b] = (_sn.pow(2) / (_sn.pow(2) + 1.0)).expand(-1, -1, _b - _a)
            _rb = _rb.detach()
            noise_head_reliability = _rb
            lambda_factor = (1.0 - _rb).clamp(0.0, 1.0).pow(
                float(getattr(self.cfg, "lateral_gate_gamma", 1.0)))
        context = self.reference_attention(features, q)
        degraded = features * (1.0 - q)
        refine_raw = self.refine_head(torch.cat([features, context, degraded], dim=1))
        refinement_z = self.cfg.refinement_limit * torch.tanh(refine_raw)
        z_denoise = torch.clamp(base_z + (1.0 - q) * refinement_z, -16.0, 16.0)
        with amp_autocast_context(x.device, enabled=False):
            _ls2 = int(self.cfg.late_start_index)
            _ms = max(0, _ls2 // 2)
            _T = int(x_eq.shape[-1])
            _xe = x_eq.float()
            _sig_ref = (meas_agg_abs if meas_agg_abs is not None else _xe.abs())
            _rt = torch.zeros(_xe.shape[0], 1, _T, device=x.device, dtype=torch.float32)
            if bool(getattr(self.cfg, "lateral_gate_temporal_witness", True)):
                for _a, _b in ((0, _ms), (_ms, _ls2), (_ls2, _T)):
                    if _b - _a >= 3:
                        _seg = _xe[..., _a:_b]
                        _d2 = _seg[..., 2:] - 2.0 * _seg[..., 1:-1] + _seg[..., :-2]
                        _sig_t = torch.sqrt(_d2.pow(2).mean(dim=-1, keepdim=True) / 6.0 + 1e-12)
                        _lvl = torch.sqrt(_sig_ref[..., _a:_b].pow(2).mean(dim=-1, keepdim=True) + 1e-12)
                        _snr_t = (_lvl / _sig_t).clamp(max=1.0e3)
                        _rt[..., _a:_b] = (_snr_t.pow(2) / (_snr_t.pow(2) + 1.0)).expand(-1, -1, _b - _a)
            _reff = _rt if meas_reliability is None else torch.maximum(_rt, meas_reliability.float())
            _gam = float(getattr(self.cfg, "lateral_gate_gamma", 1.0))
            lateral_strength = (1.0 - _reff).clamp(0.0, 1.0).pow(_gam)
            lateral_strength[..., :_ms] = 0.0
            _rt2 = _rt
            if _ls2 - _ms >= 1:
                _rmid = _reff[..., _ms:_ls2].mean(dim=-1, keepdim=True)
                strength_prior = (1.0 - _rmid).clamp(0.0, 1.0).pow(_gam)
            else:
                strength_prior = torch.zeros(_xe.shape[0], 1, 1, device=x.device)
        _lv = (torch.clamp(self.uncertainty_head(features), -8.0, 5.0)
                  if self.uncertainty_head is not None else None)
        if self.manifold_decoder is not None:
            scores = torch.softmax(self.manifold_score(features), dim=-1)
            latent = torch.einsum("bct,bht->bhc", features, scores).flatten(1)
            if getattr(self, "decoder_gate_channel", False):
                _gsn_d = self.gate_scale_norm.to(dtype=x_eq.dtype).clamp_min(1e-30)
                _zg_d = signed_symlog(x_eq / _gsn_d).squeeze(1)
                dec_in = torch.cat([latent, z.squeeze(1), z_agg.squeeze(1), _zg_d], dim=1)
            else:
                dec_in = torch.cat([latent, z.squeeze(1), z_agg.squeeze(1)], dim=1)
            hidden = self.manifold_decoder(dec_in)
            if getattr(self, "lateral_encode", None) is not None:
                _lat2 = self.lateral_encode(z_agg.squeeze(1).to(hidden.dtype))
                hidden = hidden + self.lateral_inject(_lat2)
            else:
                _lat2 = None
            if (getattr(self, "profile_joint", None) is not None and profile_len is not None
                    and int(profile_len) > 2 and hidden.shape[0] % int(profile_len) == 0):
                _P = int(profile_len)
                _ls = int(self.cfg.late_start_index)
                if meas_reliability is not None:
                    _rm = meas_reliability[:, 0, _ls // 2:_ls].mean(dim=-1, keepdim=True)
                    _rl = meas_reliability[:, 0, _ls:].mean(dim=-1, keepdim=True)
                else:
                    _rm = torch.full((hidden.shape[0], 1), 0.5, device=hidden.device)
                    _rl = torch.full((hidden.shape[0], 1), 0.5, device=hidden.device)
                _l = (_lat2 if _lat2 is not None
                         else torch.zeros(hidden.shape[0], 16, device=hidden.device))
                _ctx2 = self.profile_joint(
                    torch.cat([hidden.float(), _rm.float(), _rl.float(), _l.float()], dim=1),
                    _P)
                hidden = hidden + self.joint_inject(_ctx2).to(dtype=hidden.dtype)
            if self.fingerprint_curves is not None:
                if depth_norm is None:
                    _dep = torch.full(
                        (latent.shape[0], 1), 0.5,
                        dtype=latent.dtype, device=latent.device)
                else:
                    _dep = depth_norm.reshape(latent.shape[0], 1).to(
                        dtype=latent.dtype, device=latent.device)
                _q2 = self.fingerprint_query(
                    torch.cat([latent, _dep,
                               _lat2.to(latent.dtype)], dim=1))
                if _ctx2 is not None and getattr(self, "fingerprint_query_ctx", None) is not None:
                    _q2 = _q2 + self.fingerprint_query_ctx(
                        _ctx2.to(dtype=latent.dtype))
                _a2 = torch.softmax(
                    _q2 @ self.fingerprint_keys.t()
                    / float(max(1, _q2.shape[1])) ** 0.5, dim=-1)
                _mem = _a2 @ self.fingerprint_curves
                _fp_mem = torch.sigmoid(self.fingerprint_gate(hidden)) * _mem
            else:
                _fp_mem = 0.0
            manifold_residual = None
            relax_update_rec = None
            if self.manifold_head_atoms is not None:
                atom_w = F.softplus(self.manifold_head_atoms(hidden))
                phys = atom_w.to(self.decay_atoms.dtype) @ self.decay_atoms
                ip_w = None
                ip_gate = None
                if self.manifold_head_ip is not None:
                    ip_w = F.softplus(self.manifold_head_ip(hidden))
                    ip_gate = torch.sigmoid(self.manifold_gate_ip(hidden))
                    ip_curve = (ip_w.to(self.ip_atoms.dtype) @ self.ip_atoms) * ip_gate.to(self.ip_atoms.dtype)
                    phys = phys - ip_curve
                if self.relax_state_head is not None and meas_reliability is not None:
                    with amp_autocast_context(x.device, enabled=False):
                        upd_in = torch.cat(
                            [features.float(), z_agg.float(), meas_reliability.float()],
                            dim=1,
                        )
                        delta = torch.tanh(self.relax_state_head(upd_in))
                    with amp_autocast_context(x.device, enabled=False):
                        d32 = F.pad(delta.float(), (1, 1), mode="replicate")
                        d32 = F.conv1d(d32, self.relax_smooth_kernel)
                        s = float(self.cfg.relaxation_update_scale)
                        relax_update_rec = (s * d32 * meas_reliability).squeeze(1)
                        phys = phys.float() * torch.exp(
                            torch.clamp(relax_update_rec, -abs(s), abs(s))
                        )
                    phys = phys.to(atom_w.dtype)
                z_phys = signed_symlog(phys.to(gs.dtype) / gs.squeeze(0).squeeze(0))
                manifold_residual = float(self.cfg.physics_residual_scale) * torch.tanh(
                    self.manifold_head_free(hidden)
                )
                manifold_z = torch.clamp(
                    z_phys + manifold_residual + _fp_mem, -16.0, 16.0
                ).unsqueeze(1)
                z_phys_record = z_phys.unsqueeze(1)
            else:
                manifold_z = torch.clamp(
                    self.manifold_head_free(hidden) + _fp_mem, -16.0, 16.0
                ).unsqueeze(1)
                z_phys_record = None
            if _ctx2 is not None and getattr(self, "lateral_fp_prior", None) is not None:
                manifold_z = self.lateral_fp_prior(manifold_z, _ctx2, _P,
                                                   strength=strength_prior)
            manifold_prior_z = manifold_z
            innovation_w = None
            kalman_gain = None
            prior_sp_record = None
            _gate = None
            mc = bool(self.cfg.measurement_conditioned_gates) and meas_sigma_z is not None
            if self.manifold_innovation is not None:
                late_gate = self.late_ramp.to(dtype=manifold_z.dtype)
                clip = float(self.cfg.manifold_innovation_clip)
                _tgt = z_agg
                _gate = None
                if meas_sigma_z is not None and meas_sigma_u is not None:
                    with amp_autocast_context(x.device, enabled=False):
                        _snr_c = (x_eq.float().abs() / (meas_sigma_u + 1e-8)).clamp(max=1.0e3)
                        _r_c = _snr_c.pow(2) / (_snr_c.pow(2) + 1.0)
                        _r_c = torch.maximum(_r_c, _rt2)
                        _tgt32 = _r_c * z.float() + (1.0 - _r_c) * z_agg.float()
                    _tgt = _tgt32.to(dtype=manifold_z.dtype)
                    if (_lv is not None
                            and bool(getattr(self.cfg, "innovation_precision_weighting", True))):
                        with amp_autocast_context(x.device, enabled=False):
                            _sig_p = torch.exp(0.5 * _lv.float()).detach()
                            _sig_stack = meas_sigma_z.float()
                            _sig_single = _sig_stack * math.sqrt(max(1.0, float(meas_n_eff or 1.0)))
                            _sig_m = _r_c * _sig_single + (1.0 - _r_c) * _sig_stack
                            _lm3 = self.late_ramp.float()
                            _nl2 = _lm3.sum().clamp_min(1.0)
                            _dbar2 = ((_tgt32 - manifold_z.float().detach()) * _lm3
                                        ).sum(dim=-1, keepdim=True) / _nl2
                            _vtot = ((_sig_p.pow(2) + _sig_m.pow(2)) * _lm3
                                        ).sum(dim=-1, keepdim=True) / _nl2 * (2.0 / float(_nl2))
                            _t2_505 = _dbar2.pow(2) / (_vtot + 1.0e-12)
                            _nu = max(1.0, float(getattr(self.cfg, "innovation_robust_nu", 3.0)))
                            _infl = ((_nu + _t2_505) / (_nu + 1.0)).clamp(min=1.0, max=1.0e3)
                            _sig_p2_eff = _sig_p.pow(2) * _infl
                            _wkal = (_sig_p2_eff
                                        / (_sig_p2_eff + _sig_m.pow(2) + 1.0e-12))
                            _infl_rec = _infl
                if (meas_sigma_z is not None and meas_sigma_u is not None
                        and bool(getattr(self.cfg, "innovation_evidence_gate", False))):
                    with amp_autocast_context(x.device, enabled=False):
                        _lm2 = self.late_ramp.float()
                        _nl = _lm2.sum().clamp_min(1.0)
                        _dbar = ((_tgt32 - manifold_z.float().detach()) * _lm2
                                 ).sum(dim=-1, keepdim=True) / _nl
                        _sbar = torch.sqrt(
                            (meas_sigma_z.float().pow(2) * _lm2).sum(dim=-1, keepdim=True)
                            / _nl) * math.sqrt(2.0 / float(_nl))
                        _sig = _dbar.abs() / (_sbar + 1.0e-6)
                        _gate = torch.sigmoid(
                            (_sig - float(getattr(self.cfg, "innovation_evidence_kappa", 3.0)))
                            / max(float(getattr(self.cfg, "innovation_evidence_tau", 0.5)), 1e-3)
                        )
                innovation = torch.clamp(_tgt - manifold_z, -clip, clip)
                raw = torch.sigmoid(self.manifold_innovation(features))
                if _gate is not None:
                    raw = raw * _gate.to(dtype=raw.dtype)
                if mc and self.prior_log_sigma is not None:
                    with amp_autocast_context(x.device, enabled=False):
                        sp = F.softplus(self.prior_log_sigma.float()).view(1, 1, -1)
                        sp = sp + float(self.cfg.prior_sigma_floor)
                        prior_sp_record = sp
                        sm = meas_sigma_z
                        w_kalman = sp.detach().pow(2) / (sp.detach().pow(2) + sm.pow(2) + 1e-6)
                        mod_floor = float(self.cfg.innovation_mod_floor)
                        mod = mod_floor + (1.0 - mod_floor) * raw.float()
                        innovation_w = (w_kalman * mod * late_gate.float()).to(dtype=manifold_z.dtype)
                    kalman_gain = w_kalman
                elif _wkal is not None:
                    with amp_autocast_context(x.device, enabled=False):
                        _mf = float(getattr(self.cfg, "innovation_precision_mod_floor", 0.25))
                        _mod = _mf + (1.0 - _mf) * raw.float()
                        _warm = float(min(1.0, max(0.0, float(getattr(self, "innovation_warmup", 1.0)))))
                        innovation_w = (_wkal * _mod * late_gate.float() * _warm).to(dtype=manifold_z.dtype)
                    kalman_gain = _wkal
                else:
                    innovation_w = raw * late_gate
                manifold_z = torch.clamp(manifold_z + innovation_w * innovation, -16.0, 16.0)
            _sp = bool(getattr(self.cfg, "supervised_primary_fusion", True))
            floor = self.blend_floor.to(dtype=z_denoise.dtype)
            raw_blend = torch.sigmoid(self.manifold_blend(features))
            if mc:
                with amp_autocast_context(x.device, enabled=False):
                    g = kalman_gain if kalman_gain is not None else meas_reliability
                    floor_eff = floor.float() * (1.0 - g)
                    ceil_eff = 1.0 - float(self.cfg.blend_ceiling_k) * g
                    blend = torch.clamp(raw_blend.float(), min=0.0, max=1.0)
                    blend = torch.maximum(torch.minimum(blend, ceil_eff), floor_eff)
                    late_mask_b = self.late_ramp.float()
                    cap = self.blend_late_quality_cap.float()
                blend = blend.to(dtype=z_denoise.dtype)
                if _sp:
                    blend = torch.clamp(raw_blend, 0.0, 1.0).to(dtype=z_denoise.dtype)
            else:
                floor_g = floor * self.blend_floor_scale.to(dtype=floor.dtype)
                if _sp:
                    blend = raw_blend
                else:
                    blend = floor_g + (1.0 - floor_g) * raw_blend
            late_noise_floor = None
            if (meas_snr_gate is not None
                    and bool(getattr(self.cfg, "late_measured_noise_floor", True))):
                with amp_autocast_context(x.device, enabled=False):
                    _srel = float(getattr(self.cfg, "late_noise_floor_prior_sigma_rel", 0.15))
                    _q3 = (_srel * meas_snr_gate.float()).pow(2)
                    late_noise_floor = (self.late_ramp.float()
                                        * float(getattr(self.cfg, "late_noise_floor_gamma", 0.98))
                                        / (1.0 + _q3))
                    _rnh = noise_head_reliability.float()
                    _snr_nh2 = (_rnh / (1.0 - _rnh).clamp_min(1.0e-6))
                    _floor_nh = (self.late_ramp.float()
                                 * float(getattr(self.cfg, "late_noise_floor_gamma", 0.98))
                                 / (1.0 + (_srel ** 2) * _snr_nh2))
                    late_noise_floor = torch.maximum(late_noise_floor, _floor_nh)
                if not _sp:
                    blend = torch.maximum(blend.float(), late_noise_floor).to(dtype=blend.dtype)
            late_mask_b = self.late_ramp.to(dtype=blend.dtype)
            cap = self.blend_late_quality_cap.to(dtype=blend.dtype)
            _vfl = torch.clamp(self.blend_late_val_floor.to(dtype=blend.dtype),
                               min=0.0)
            _vfl = torch.minimum(_vfl, torch.full_like(_vfl, 0.99))
            if not _sp:
                blend = torch.maximum(blend, late_mask_b * _vfl)
            blend_raw = blend
            _adv = self.late_prior_advantage.to(dtype=blend.dtype)
            _strength = cap + (1.0 - cap) * _adv
            if not _sp:
                blend = blend + late_mask_b * (1.0 - blend) * (1.0 - q) * _strength
            else:
                blend = torch.minimum(
                    blend.float(),
                    1.0 - late_mask_b.float() * (1.0 - cap.float()),
                ).to(dtype=z_denoise.dtype)
            if bool(getattr(self.cfg, "v1122_parity", False)):
                _sf = torch.zeros_like(self.blend_floor)
                _sf[..., int(self._late_start_gate):] = float(
                    self.cfg.manifold_blend_floor_late)
                _sf = _sf.to(dtype=blend.dtype)
                blend = _sf + (1.0 - _sf) * raw_blend.to(dtype=blend.dtype)
            if (bool(getattr(self.cfg, "seam_blend_floor_enabled", True))
                    and self.seam_floor_mode() == "hard"):
                blend = torch.maximum(blend, self.seam_blend_floor.to(dtype=blend.dtype).view(1, 1, -1))
            _use_mmse = (bool(getattr(self.cfg, "mmse_blend", True)) and not _dec and meas_snr_gate is not None
                            and self.mmse_bstar_714.numel() == z_denoise.shape[-1]
                            and ((self.training and bool(getattr(self.cfg, "mmse_blend_in_training", False)))
                                 or (not self.training and (bool(getattr(self, "_force_mmse_blend", False))
                                                            or float(self.mmse_deploy_714.item()) > 0.5))))
            if (self.training and meas_snr_gate is not None and bool(getattr(self.cfg, "mmse_blend", True))
                    and self.snr_ref_714.numel() == z_denoise.shape[-1]):
                with torch.no_grad():
                    _s = meas_snr_gate.float().clamp(1e-3, 1e3)
                    _href = torch.rsqrt((1.0 / _s ** 2).mean(dim=(0, 1)).clamp(1e-6, 1e6)).clamp(1e-3, 1e3)
                    self.snr_ref_714.mul_(0.99).add_(0.01 * _href)
            if _use_mmse:
                with amp_autocast_context(x.device, enabled=False):
                    _snr = meas_snr_gate.float().clamp(1e-3, 1e3)
                    _bs = self.mmse_bstar_714.float().clamp(0.02, 0.98).view(1, 1, -1)
                    _q5 = ((1.0 - _bs) / _bs) * (_snr / self.snr_ref_714.float().clamp(1e-3, 1e3).view(1, 1, -1)) ** 2
                    _bm = 1.0 / (1.0 + _q5)
                    _ramp = self.late_ramp.float().view(1, 1, -1)
                    _blend2 = _ramp * _bm + (1.0 - _ramp) * blend.float()
                blend = _blend2.clamp(0.0, 0.98).to(dtype=z_denoise.dtype)
            _gl = int(getattr(self, "row_completion_gate_limit", 0) or 0)
            _completion_z = None
            if _dec and _gl > 0 and self.row_completion_W_713.numel() > 0:
                with amp_autocast_context(x.device, enabled=False):
                    _gs32 = self.gate_scale_norm.float().view(1, 1, -1)
                    _zraw = signed_symlog(_x_in.float() / _gs32)
                    _xe = _zraw[:, 0, :_gl] - self.row_completion_mx_713.view(1, -1)
                    _zc_raw = _xe @ self.row_completion_W_713 + self.row_completion_my_713.view(1, -1)
                    _ts = trace_scale.float().view(-1, 1) if trace_scale is not None else torch.ones(_zc_raw.shape[0], 1, device=x.device)
                    _zc_model = signed_symlog(signed_symexp(_zc_raw) * _ts)
                    _zc_model = torch.clamp(_zc_model, float(_zb[0]), float(_zb[1]))
                    _prior = torch.cat([z.float()[:, 0, :_gl], _zc_model], dim=1).unsqueeze(1)
                    _lsp = torch.clamp(self.sigma_prior_713(features.float()).float(), -12.0, 6.0)
                    _lsm = torch.clamp(self.sigma_meas_713(features.float()).float(), -12.0, 6.0)
                    _wp = torch.sigmoid(_lsm - _lsp)
                    _blend = torch.zeros_like(_wp)
                    _blend[..., _gl:] = _wp[..., _gl:]
                _completion_z = _prior.to(dtype=z_denoise.dtype)
                manifold_z = _completion_z
                manifold_prior_z = _completion_z
                blend = _blend.detach().to(dtype=z_denoise.dtype)
                _log_sigma_prior = _lsp.to(dtype=z_denoise.dtype)
                _log_sigma_meas = _lsm.to(dtype=z_denoise.dtype)
            else:
                _log_sigma_prior = _log_sigma_meas = None
            if bool(getattr(self, "_ablate_cdmr", False)) and blend is not None:
                blend = torch.zeros_like(blend)
            z_hat = torch.clamp((1.0 - blend) * z_denoise + blend * manifold_z, -16.0, 16.0)
            z_hat_model = torch.clamp((1.0 - blend) * z_denoise
                                      + blend * manifold_prior_z.to(dtype=z_denoise.dtype),
                                      -16.0, 16.0)
            lib_bg_z = None
            lib_gate = None
            lib_coef = None
            _m = None
            _t = None
            _ff = None
            if getattr(self, "lib_coef_head", None) is not None:
                if self._lib_ready_flag is None:
                    self._lib_ready_flag = bool(float(self.lib_ready.item()) > 0.5)
                if self._lib_ready_flag:
                    with amp_autocast_context(x.device, enabled=False):
                        _ls4 = int(getattr(self.cfg, "late_start_index", 20))
                        _rel = noise_head_reliability.float().clamp(0.0, 1.0)
                        _w2 = torch.ones_like(features[:, :1, :].float())
                        _w2[..., _ls4:] = _rel[..., _ls4:]
                        _w2 = _w2.clamp_min(1e-3)
                        _pool = ((features.float() * _w2).sum(dim=-1)
                                    / _w2.sum(dim=-1))
                        lib_coef = self.lib_coef_head(_pool)
                        if bool(getattr(self.cfg, "use_coef_field", True)):
                            with torch.no_grad(), amp_autocast_context(
                                    x.device, enabled=False):
                                _Vb6 = self.lib_basis.float()
                                _mu6 = self.lib_mu.float().view(1, 1, -1)
                                _zc6 = torch.cat(
                                    [z.float(), signed_symlog(
                                        neighbors.float() / gs.float())], 1)
                                _w6 = torch.ones_like(_zc6[:, :1, :])
                                if (bool(getattr(self.cfg, "coef_field_gate_precision", True))
                                        and float(self.gate_precision_ready.item()) > 0.5):
                                    _w6 = _w6 * self.gate_precision.float().view(1, 1, -1)
                                    _q4 = locals().get("meas_reliability", None)
                                    if isinstance(_q4, torch.Tensor) and _q4.dim() == 3 \
                                            and _q4.shape[0] == _w6.shape[0] and _q4.shape[-1] == _w6.shape[-1]:
                                        _w6 = _w6 * (0.05 + 0.95 * _q4[:, :1, :].detach().float().clamp(0.0, 1.0))
                                _wcal = _w6.clone()
                                _med6 = _zc6.median(dim=1, keepdim=True).values
                                _var6 = ((_zc6 - _med6) ** 2).mean(
                                    dim=1, keepdim=True)
                                _sig6 = _med6.abs().clamp_min(1e-3) ** 2
                                _ws6 = _sig6 / (_sig6 + _var6)
                                _t = _ws6
                                _c2f = float(self.lib_c2fam.item())
                                if _c2f > 0.0:
                                    _zc0 = _zc6[:, :1, :]
                                    _d2 = (_zc0[..., 2:] - 2.0 * _zc0[..., 1:-1]
                                           + _zc0[..., :-2])
                                    _m2 = torch.zeros_like(_zc0)
                                    _m2[..., 1:-1] = (_d2 * _d2) / 6.0
                                    _m2[..., 0] = _m2[..., 1]
                                    _m2[..., -1] = _m2[..., -2]
                                    _sn6c = torch.relu(_m2 - (_c2f * _c2f) / 6.0)
                                    _wm6 = _sig6 / (_sig6 + _sn6c)
                                    _t = torch.minimum(_ws6, _wm6)
                                    _w6[..., _ls4:] = (_wcal * torch.minimum(
                                        _ws6, _wm6))[..., _ls4:]
                                else:
                                    _w6[..., _ls4:] = (_wcal * _ws6)[..., _ls4:]
                                _w6 = _w6.clamp_min(1e-6)
                                _rhs6 = torch.einsum(
                                    "rt,bkt->bkr", _Vb6,
                                    _w6 * (_zc6 - _mu6))
                                _G6 = torch.einsum(
                                    "rt,bt,ut->bru", _Vb6, _w6[:, 0], _Vb6)
                                _d6 = torch.diagonal(_G6, dim1=-2, dim2=-1)
                                _sr2 = self.lib_cvar.float().view(1, -1)
                                _sn2 = torch.zeros(_zc6.shape[0], 1, device=_zc6.device)
                                _mu0 = float(getattr(self.cfg,
                                                     "coef_field_mu", 1e-3))
                                _lam6 = _mu0 + 8.0 * torch.relu(0.5 - _d6.clamp(0.0, 1.0))
                                _G6 = _G6 + torch.diag_embed(_lam6)
                                _c6 = torch.linalg.solve(
                                    _G6.float().unsqueeze(1),
                                    _rhs6.float().unsqueeze(-1)
                                ).squeeze(-1)
                                _cf6 = _c6.median(dim=1).values
                                _rf6 = self.lib_resfam.float().view(1, 1, -1)
                                if float(_rf6.abs().max().item()) > 0.0:
                                    _fit6 = (_mu6 + torch.einsum(
                                        "br,rg->bg", _cf6, _Vb6).unsqueeze(1))
                                    _res6 = (_zc6[:, :1, :] - _fit6).abs()
                                    _sn2 = (1.4826 * _res6[..., :_ls4].median(
                                        dim=-1).values) ** 2
                                    _sn2 = _sn2.view(-1, 1)
                                    _tol6 = _rf6.clamp_min(1e-4)
                                    _s6m = (_res6 / _tol6).median(
                                        dim=-1, keepdim=True).values
                                    _tolE = torch.sqrt((_tol6 * _s6m.clamp_min(1.0)) ** 2 + _sn2.view(-1, 1, 1))
                                    _wr6 = (_tolE * _tolE) / (
                                        _tolE * _tolE + _res6 * _res6)
                                    _w6b = _w6.clone()
                                    _w6b[..., _ls4:] = (_w6 * _wr6)[..., _ls4:]
                                    _w6b = _w6b.clamp_min(1e-6)
                                    _rhs6 = torch.einsum(
                                        "rt,bkt->bkr", _Vb6,
                                        _w6b * (_zc6 - _mu6))
                                    _G6b = torch.einsum(
                                        "rt,bt,ut->bru", _Vb6,
                                        _w6b[:, 0], _Vb6)
                                    _d6b = torch.diagonal(
                                        _G6b, dim1=-2, dim2=-1)
                                    _lamb = (_mu0 + _sn2 / _sr2 + 8.0 * torch.relu(
                                        0.5 - _d6b.clamp(0.0, 1.0)))
                                    _G6b = _G6b + torch.diag_embed(_lamb)
                                    _c6 = torch.linalg.solve(
                                        _G6b.float().unsqueeze(1),
                                        _rhs6.float().unsqueeze(-1)
                                    ).squeeze(-1)
                                    _cf6 = _c6.median(dim=1).values
                                _A7 = self.lib_compA.float()
                                if float(_A7.abs().sum().item()) > 0.0:
                                    _dobs7 = (_d6b if "_d6b" in dir() else _d6)
                                    _tr7 = _dobs7.clamp(0.0, 1.0)
                                    _wB7 = torch.as_tensor(completion_observation_weight(int(_w6.shape[-1]), _ls4),
                                                           dtype=_w6.dtype, device=_w6.device).view(1, 1, -1).expand_as(_w6).clone()
                                    _rhsB = torch.einsum("rt,bkt->bkr", _Vb6,
                                                         _wB7 * (_zc6 - _mu6))
                                    _GB7 = torch.einsum("rt,bt,ut->bru", _Vb6,
                                                        _wB7[:, 0], _Vb6)
                                    _dB7 = torch.diagonal(_GB7, dim1=-2, dim2=-1)
                                    _GB7 = _GB7 + torch.diag_embed(
                                        _mu0 + 4.0 * (1.0 - _dB7.clamp(0.0, 1.0)))
                                    _cB7 = torch.linalg.solve(
                                        _GB7.float().unsqueeze(1),
                                        _rhsB.float().unsqueeze(-1)
                                    ).squeeze(-1).median(dim=1).values
                                    _cx7 = torch.cat(
                                        [_cB7, torch.ones_like(_cB7[:, :1])], 1)
                                    _cc7 = _cx7 @ _A7
                                    if float(self.fam_ready_760.item()) > 0.5:
                                        _ds = ((_cB7 - self.fam_mu_760.float().view(1, -1))
                                                  * self.fam_dir_760.float().view(1, -1)).sum(dim=-1)
                                        _db = self.fam_dband_760.float()
                                        _mm = torch.nan_to_num(torch.minimum(
                                            ((_ds - _db[0]) / (_db[1] - _db[0]).clamp_min(1e-6)).clamp(0.0, 1.0),
                                            ((_db[3] - _ds) / (_db[3] - _db[2]).clamp_min(1e-6)).clamp(0.0, 1.0)),
                                            nan=0.0).view(-1, 1)
                                        if bool((_mm > 0).any()):
                                            _m = _mm
                                    if bool(getattr(self.cfg, "coef_field_trust_mix", True)):
                                        _cf6 = _tr7 * _cf6 + (1.0 - _tr7) * _cc7
                                    else:
                                        _cf6 = _cc7
                                    if _m is not None:
                                        _cf6 = torch.where(_m > 0, _cf6 + _m * (_cx7 @ self.fam_compA_760.float() - _cf6), _cf6)
                            if bool(getattr(self.cfg, "coef_field_learned_delta", True)):
                                _cf6 = _cf6.detach() + (lib_coef.float() if _m is None
                                                        else lib_coef.float() * (1.0 - _m))
                                lib_coef = _cf6
                            _bg = _mu6.squeeze(1) + torch.einsum(
                                "br,rg->bg", _cf6, _Vb6).unsqueeze(1)
                            _beta4 = float(self.lib_plbeta.item())
                            if _beta4 > 0.0 and float(self.lib_logt.abs().sum().item()) > 0.0:
                                with torch.no_grad():
                                    _M4 = 8
                                    _lt4 = self.lib_logt.float()
                                    _zf4 = _bg[:, 0, :]
                                    _yf4 = signed_symexp(_zf4) * gs.float().view(1, -1)
                                    _ly4 = torch.log(_yf4[:, _ls4 - _M4:_ls4].abs().clamp_min(1e-30))
                                    _ltm = _lt4[_ls4 - _M4:_ls4]
                                    _ltc = _ltm - _ltm.mean()
                                    _b4 = ((_ltc.view(1, -1) * (_ly4 - _ly4.mean(1, keepdim=True))).sum(1)
                                           / (_ltc * _ltc).sum().clamp_min(1e-12))
                                    _a4 = _ly4.mean(1) - _b4 * _ltm.mean()
                                    _lyl = _a4.view(-1, 1) + _b4.view(-1, 1) * _lt4[_ls4:].view(1, -1)
                                    _sg4 = torch.sign(_yf4[:, _ls4 - 1:_ls4]).clamp(min=-1.0)
                                    _sg4 = torch.where(_sg4 == 0, torch.ones_like(_sg4), _sg4)
                                    _ypl = _sg4 * torch.exp(_lyl.clamp(max=60.0))
                                    _zpl = signed_symlog(_ypl / gs.float().view(1, -1)[:, _ls4:])
                                    _wl4 = (_w6b if "_w6b" in dir() else _w6)
                                    _tr4 = _wl4[..., _ls4:].mean(dim=(-2, -1)).clamp(0.0, 1.0).view(-1, 1)
                                    _mix4 = (1.0 - _tr4) * _beta4
                                    if _m is not None:
                                        _mix4 = _mix4 * (1.0 - _m)
                                    _zl4 = _zf4[:, _ls4:] + _mix4 * (_zpl - _zf4[:, _ls4:])
                                    _bg = torch.cat([_zf4[:, :_ls4], _zl4], dim=1).unsqueeze(1)
                        else:
                            _bg = self.lib_mu.float() + torch.einsum(
                                "br,rg->bg", lib_coef, self.lib_basis.float()
                            ).unsqueeze(1)
                        lib_bg_z = torch.clamp(_bg, -16.0, 16.0)
                        _lg = torch.sigmoid(
                            self.lib_gate_head(features.float()))
                        lib_gate = _lg * self.late_ramp.float()
                        if _lv is not None:
                            _sig2 = torch.exp(0.5 * _lv.float()).clamp_min(0.1)
                            _dev = (lib_bg_z - z_hat.float()).abs()
                            _cons = torch.exp(-0.5 * (_dev / (3.0 * _sig2)) ** 2)
                            lib_gate = lib_gate * _cons.to(lib_gate.dtype)
                        _xg = None
                        if (_m is not None and _t is not None
                                and bool(getattr(self.cfg, "family_gate_floor", True))):
                            _tt = _t.float().clamp(0.0, 1.0)
                            _fl = (1.0 - _tt) / ((1.0 - _tt) + self.fam_rho2_760.float().view(1, 1, -1) * _tt
                                                       + 1.0e-12)
                            _ff = (_m.view(-1, 1, 1) * self.fam_kappa_760.float().view(1, 1, -1)
                                      * _fl)
                            lib_gate = lib_gate * (1.0 - _m.view(-1, 1, 1)).to(lib_gate.dtype)
                            _xg = torch.relu(_ff * self.late_ramp.float() - lib_gate.float())
                        if (not self.training) and (bool(getattr(self, "_force_no_anchors", False))
                                                    or bool(getattr(self, "_ablate_cdmr", False))):
                            lib_gate = torch.zeros_like(lib_gate)
                            _xg = None
                        _inn = lib_bg_z - z_hat.float()
                        _zh = z_hat.float() + lib_gate * _inn
                        if _xg is not None:
                            _zh = _zh + _xg * self.lateral_common(_inn, profile_len)
                            lib_gate = (lib_gate.float() + _xg).to(lib_gate.dtype)
                    z_hat = torch.clamp(_zh, -16.0, 16.0).to(dtype=z_denoise.dtype)
                    lib_bg_z = lib_bg_z.to(dtype=z_denoise.dtype)
                    lib_gate = lib_gate.to(dtype=z_denoise.dtype)
            ridge_bg_z = None
            ridge_gate = None
            if getattr(self, "ridge_gate_head", None) is not None:
                _learnedC = (bool(getattr(self.cfg, "learned_continuation", True))
                             and getattr(self, "cont_head", None) is not None)
                if self._ridge_ready_flag is None:
                    self._ridge_ready_flag = _learnedC or bool(
                        float(self.lib_ridge_ready.item()) > 0.5)
                if self._ridge_ready_flag:
                    with amp_autocast_context(x.device, enabled=False):
                        _lsR = int(self.lib_ridge_W.shape[0])
                        _zbR = z_hat.float().clone()
                        _emR = _zbR[..., :_lsR]
                        if _learnedC:
                            _wF = torch.ones_like(_zbR)
                            if noise_head_reliability is not None:
                                _wF[..., _lsR:] = noise_head_reliability.float(
                                    )[..., _lsR:].clamp(0.0, 1.0)
                            _zin = torch.cat([_zbR * _wF, _wF], dim=-1)
                            _rlR = self.cont_head(_zin.reshape(-1, _zin.shape[-1])
                                                  ).reshape(_zbR.shape[0], _zbR.shape[1], -1)
                        else:
                            _rlR = (torch.einsum("bce,el->bcl", _emR,
                                                 self.lib_ridge_W.float())
                                    + self.lib_ridge_b.float().view(1, 1, -1))
                        ridge_bg_z = torch.clamp(
                            torch.cat([_emR, _rlR], dim=-1), -16.0, 16.0)
                        _rgR = torch.sigmoid(
                            self.ridge_gate_head(features.float()))
                        ridge_gate = _rgR * self.late_ramp.float()
                        if _lv is not None:
                            _sigR = torch.exp(0.5 * _lv.float()).clamp_min(0.1)
                            _devR = (ridge_bg_z - _zbR).abs()
                            ridge_gate = ridge_gate * torch.exp(
                                -0.5 * (_devR / (3.0 * _sigR)) ** 2).to(ridge_gate.dtype)
                        if _ff is not None:
                            ridge_gate = ridge_gate * (1.0 - _ff).to(ridge_gate.dtype)
                        if (not self.training) and (bool(getattr(self, "_force_no_anchors", False))
                                                    or bool(getattr(self, "_ablate_cdmr", False))):
                            ridge_gate = torch.zeros_like(ridge_gate)
                        _zhR = _zbR + ridge_gate * (ridge_bg_z - _zbR)
                    z_hat = torch.clamp(_zhR, -16.0, 16.0).to(dtype=z_denoise.dtype)
                    ridge_bg_z = ridge_bg_z.to(dtype=z_denoise.dtype)
                    ridge_gate = ridge_gate.to(dtype=z_denoise.dtype)
            if (bool(getattr(self.cfg, "output_measurement_fusion", True))
                    and _lv is not None and meas_sigma_z is not None
                    and meas_sigma_u is not None and _rt2 is not None):
                with amp_autocast_context(x.device, enabled=False):
                    _snr_o = (x_eq.float().abs() / (meas_sigma_u + 1e-8)).clamp(max=1.0e3)
                    _rc_o = torch.maximum(_snr_o.pow(2) / (_snr_o.pow(2) + 1.0), _rt2)
                    _tgt_o = _rc_o * z.float() + (1.0 - _rc_o) * z_agg.float()
                    _sp_o = torch.exp(0.5 * _lv.float()).detach()
                    _sst_o = meas_sigma_z.float()
                    _ssg_o = _sst_o * math.sqrt(max(1.0, float(meas_n_eff or 1.0)))
                    _sm_o = _rc_o * _ssg_o + (1.0 - _rc_o) * _sst_o
                    _d_o = (_tgt_o - z_hat.float().detach())
                    _t2_o = _d_o.pow(2) / (_sp_o.pow(2) + _sm_o.pow(2) + 1.0e-12)
                    _nu_o = max(1.0, float(getattr(self.cfg, "innovation_robust_nu", 3.0)))
                    _infl_o = ((_nu_o + _t2_o) / (_nu_o + 1.0)).clamp(min=1.0, max=1.0e3)
                    _sm2_o = _sm_o.pow(2) * _infl_o
                    _sp2_o = _sp_o.pow(2)
                    _warm_o = float(min(1.0, max(0.0, float(getattr(self, "innovation_warmup", 1.0)))))
                    _prec = (self.gate_precision.float().view(1, 1, -1)
                                if float(self.gate_precision_ready.item()) > 0.5
                                else torch.ones_like(self.late_ramp.float()))
                    _wout = (_sp2_o / (_sp2_o + _sm2_o + 1.0e-12)
                                * (1.0 - self.late_ramp.float()) * _warm_o * _prec)
                    _zo = z_hat.float() + _wout * (_tgt_o - z_hat.float())
                z_hat = torch.clamp(_zo, -16.0, 16.0).to(dtype=z_hat.dtype)
            z_hat_pre_event = z_hat
            if self.event_gate_head is not None:
                with amp_autocast_context(x.device, enabled=False):
                    _mis = (_z_ev.float() - manifold_z.float()).detach()
                    _mis = torch.tanh(_mis / max(float(self.cfg.event_misfit_scale), 1e-6))
                    _lm = self.evidence_accum_mask.to(dtype=_mis.dtype)
                    _cnt = torch.cumsum(_lm, dim=-1).clamp_min(1.0)
                    _acc = torch.cumsum(_mis * _lm, dim=-1) / _cnt
                    _mis = _mis * (1.0 - _lm) + _acc * _lm
                    if _nb_rec is not None and _nb_rec.shape[1] > 0:
                        _z_nb = signed_symlog(_nb_rec.float() / gs.float())
                        _lat = (_z_ev.float() - _z_nb.median(dim=1, keepdim=True).values).detach()
                        _lat = torch.tanh(_lat / max(float(self.cfg.event_misfit_scale), 1e-6))
                        _lat = torch.cumsum(_lat, dim=-1) / torch.arange(
                            1, _lat.shape[-1] + 1, device=_lat.device,
                            dtype=_lat.dtype).view(1, 1, -1)
                    else:
                        _lat = torch.zeros_like(_mis)
                    _ev_in = torch.cat([features.float(), _mis, _lat, q.float()], dim=1)
                    event_logits = self.event_gate_head(_ev_in)
                    event_logits = event_logits + (
                        F.softplus(self.event_lat_gain) * _lat.abs()
                    ).to(dtype=event_logits.dtype)
                    event_prob = torch.sigmoid(event_logits)
                    _r = float(self.cfg.struct_residual_scale) * torch.tanh(
                        self.struct_residual_head(_ev_in)
                    )
                    _k = self.relax_smooth_kernel.float()
                    _r = F.conv1d(F.conv1d(_r, _k, padding=1), _k, padding=1)
                    struct_residual = _r
                _tc = torch.sigmoid(
                    event_logits.float()
                    .masked_fill(self.nonevidence_gate_mask, -1.0e4)
                    .amax(dim=-1, keepdim=True)).to(dtype=z_hat.dtype)
                if not bool(getattr(self, "_ablate_egsr", False)):
                    z_hat = torch.clamp(
                        z_hat + (self.event_expression_scale.to(dtype=z_hat.dtype)
                                 * _tc * event_prob * struct_residual
                                 ).to(dtype=z_hat.dtype),
                        -16.0, 16.0
                    )
            else:
                event_logits = None
                event_prob = None
                struct_residual = None
        else:
            manifold_z = None
            blend = None
            z_hat = z_denoise
            z_hat_model = z_denoise
            z_hat_pre_event = z_hat
            event_logits = None
            event_prob = None
            struct_residual = None
        z_hat_post_event = z_hat
        z_hat_pre_lateral = z_hat
        lateral_applied = False
        out_lambda_factor = None
        _kdev = None
        if _ctx2 is not None and getattr(self, "lateral_fp_out", None) is not None:
            z_hat = self.lateral_fp_out(z_hat, _ctx2, _P,
                                        strength=lateral_strength)
            _kdev = getattr(self.lateral_fp_out, "_last_dev", None)
            lateral_applied = True
        if (self.lateral_joint is not None and profile_len is not None
                and int(profile_len) > 2 and z_hat.shape[0] % int(profile_len) == 0):
            _w = None
            if (str(getattr(self.cfg, "lateral_joint_weighting", "uniform")) == "sigma"
                    and _lv is not None):
                _w = torch.exp(-_lv.float()).detach()
            _wit = str(getattr(self.cfg, "lateral_lambda_witness", "noise_head"))
            if _wit == "noise_head":
                _lf = lambda_factor
                if (lateral_strength is not None and meas_reliability is not None
                        and bool(getattr(self.cfg, "lateral_witness_measurement_floor", True))):
                    _lf = torch.maximum(lambda_factor.float(), lateral_strength.detach().float())
            elif _wit == "scatter" and lateral_strength is not None:
                _lf = lateral_strength
            else:
                _lf = None
            _c0634 = float(getattr(self.cfg, "lateral_structure_witness_c0", 0.0) or 0.0)
            if _c0634 > 0.0:
                try:
                    _P2 = int(profile_len)
                    _n = int(z_hat.shape[0]) // _P2
                    _T3 = int(z_hat.shape[-1])
                    _g = int(min(max(1, int(getattr(self, "_prior_band_g", 1))), _T3 - 1))
                    _ze = z.detach().float().reshape(_n, _P2, _T3)[:, :, :_g]
                    if _P2 >= 3:
                        _zp = F.pad(_ze.permute(0, 2, 1), (1, 1), mode="reflect").permute(0, 2, 1)
                        _d2634 = (_zp[:, 2:] - 2.0 * _zp[:, 1:-1] + _zp[:, :-2]).abs().mean(dim=-1)
                        _f = (1.0 / (1.0 + (_d2634 / _c0634) ** 2)).reshape(_n * _P2, 1, 1)
                        _f = _f.expand(-1, 1, _T3).clone()
                        _f[..., :_g] = 1.0
                        _lf = (_f if _lf is None
                                  else _lf.reshape(_n * _P2, 1, _T3) * _f.to(dtype=_lf.dtype))
                        self._last_structure_witness = _f[..., -1].reshape(-1).detach()
                except Exception:
                    pass
            _fg = None
            _kl = float(getattr(self.cfg, "lateral_event_licence", 0.0) or 0.0)
            if _kl > 0.0 and event_prob is not None and event_prob.shape[0] == z_hat.shape[0]:
                _T4 = int(z_hat.shape[-1])
                _g2 = int(min(max(1, int(getattr(self, "_prior_band_g", self.cfg.late_start_index))), _T4 - 1))
                _ep = event_prob.detach().float().reshape(z_hat.shape[0], 1, -1)
                if _ep.shape[-1] == _T4:
                    _P3 = int(profile_len)
                    _k2 = int(getattr(self.lateral_joint, "order", 3))
                    _n2 = int(z_hat.shape[0]) // _P3
                    _e = _ep.reshape(_n2, _P3, _T4).permute(0, 2, 1).reshape(_n2 * _T4, 1, _P3)
                    _e = F.max_pool1d(_e, kernel_size=2 * _k2 + 1, stride=1, padding=_k2)
                    _ep = _e.reshape(_n2, _T4, _P3).permute(0, 2, 1).reshape(z_hat.shape[0], 1, _T4)
                    _p0748 = float(min(0.9, max(0.0, getattr(self.cfg, "lateral_event_licence_p0", 0.1))))
                    _ep = ((_ep - _p0748) / (1.0 - _p0748)).clamp(0.0, 1.0)
                    _fg = (1.0 - _kl * _ep).clamp(0.0, 1.0)
                    _fg[..., :_g2] = 1.0
                    _lf = (_fg if _lf is None
                              else _lf.float().reshape(z_hat.shape[0], 1, -1) * _fg)
                    self._last_event_licence = {"mean": float((1.0 - _fg[..., _g2:]).mean()),
                                                "licensed_share": float(((1.0 - _fg[..., _g2:]) > 0.5).float().mean())}
            if _lf is not None:
                _lf = _lf.float().clamp(0.0, 1.0)
                try:
                    _lsn = (lateral_strength.detach().float() if lateral_strength is not None else None)
                    self._last_lambda_factor_stats = {
                        "mean": float(_lf.mean()),
                        "below_measurement_share": (float((_lf < _lsn - 1e-6).float().mean()) if _lsn is not None else float("nan")),
                        "structure_witness_c0": float(_c0634)}
                except Exception:
                    pass
            self.lateral_joint.symmetric_form = bool(getattr(self.cfg, "lateral_joint_symmetric", True))
            z_hat = self.lateral_joint(
                z_hat, int(profile_len), _w,
                noise_sigma=meas_sigma_z,
                noise_kappa=float(getattr(self.cfg, "lateral_joint_noise_kappa", 0.0)),
                noise_cap=float(getattr(self.cfg, "lateral_joint_noise_cap", 3.0)),
                noise_sp0=float(getattr(self.cfg, "late_noise_floor_prior_sigma_z", 0.15)),
                lam_factor=_lf, floor_gate=_fg)
            lateral_applied = True
            out_lambda_factor = _lf
        if (_gl2 is not None and bool(_gl2.get("cdmr", False))
                and getattr(self, "gl_departure_logit", None) is not None):
            z_hat = self._withheld_departure(z_hat, _gl2, trace_scale)
        denoised_eq = signed_symexp(z_hat.float())
        base_eq = signed_symexp(base_z.float())
        inv = (1.0 / trace_scale) if trace_scale is not None else 1.0
        out: Dict[str, torch.Tensor] = {
            "denoised": denoised_eq * gs * inv,
            "base": base_eq * gs * inv,
            "predicted_noise": (x_eq - base_eq) * gs * inv,
            "refinement": refinement_z,
            "denoised_eq": denoised_eq,
            "base_eq": base_eq,
            "z_noisy": z,
            "z_hat": z_hat,
            **({"fp_curves": self.fingerprint_curves}
               if self.fingerprint_curves is not None else {}),
            "base_z": base_z,
            "noise_z": noise_z,
            "refinement_z": refinement_z,
            "reliability": q,
            "reliability_logits": q_logits,
            "domain_gate": domain_gate,
            "z_hat_pre_event": z_hat_pre_event,
            "z_hat_post_event": z_hat_post_event,
            "scale_disagreement": scale_disagreement,
            "z_denoise": z_denoise,
        }
        if trace_scale is not None:
            out["trace_scale"] = trace_scale
        if meas_sigma_z is not None:
            out["meas_sigma_z"] = meas_sigma_z
            out["meas_reliability"] = meas_reliability
        if _lv is not None:
            out["log_variance"] = _lv
        out["z_hat_pre_lateral"] = z_hat_pre_lateral
        out["z_hat_model"] = z_hat_model
        if _gl2 is not None:
            out["gate_loss_fill_z"] = _gl2["z_fill"].to(dtype=z_hat_model.dtype)
        out["lateral_applied"] = torch.tensor(1.0 if lateral_applied else 0.0, device=z_hat.device)
        if self.lateral_joint is not None:
            out["lateral_lambda"] = self.lateral_joint.lambdas()
        if _kdev is not None:
            out["lateral_kernel_dev"] = _kdev
        if lateral_strength is not None:
            out["lateral_strength"] = lateral_strength
            out["lateral_strength_prior"] = strength_prior
        out["noise_head_reliability"] = noise_head_reliability
        if lib_bg_z is not None:
            out["library_bg_z"] = lib_bg_z
            out["library_bg"] = (signed_symexp(lib_bg_z.float()) * gs * inv
                                 ).to(dtype=z_denoise.dtype)
            out["library_gate"] = lib_gate
            if getattr(self, "fam_ready_760", None) is not None and float(self.fam_ready_760.item()) > 0.5:
                out["family_membership_map"] = (
                    _m.view(-1, 1, 1).expand(-1, 1, int(lib_gate.shape[-1])).to(lib_gate.dtype)
                    if _m is not None else torch.zeros_like(lib_gate))
            out["library_coef"] = lib_coef
            out["library_mu"] = self.lib_mu
            out["library_basis"] = self.lib_basis
        if ridge_bg_z is not None:
            out["ridge_bg_z"] = ridge_bg_z
            out["ridge_gate"] = ridge_gate
            out["ridge_bg"] = (signed_symexp(ridge_bg_z.float()) * gs * inv
                               ).to(dtype=z_denoise.dtype)
        if out_lambda_factor is not None:
            out["lateral_lambda_factor"] = out_lambda_factor
        if _ctx2 is not None:
            out["joint_profile_code"] = _ctx2
        if manifold_z is not None:
            out["manifold_z"] = manifold_z
            out["denoise_arm_z"] = z_denoise
            if _completion_z is not None:
                out["completion_z"] = _completion_z
                out["log_sigma_prior"] = _log_sigma_prior
                out["log_sigma_meas"] = _log_sigma_meas
                out["completion_gate_end"] = torch.tensor(int(_gl), device=z_hat.device)
            out["manifold_blend"] = blend
            out["manifold_blend_raw"] = blend_raw
            out["manifold_blend_head"] = raw_blend
            if self.seam_floor_mode() == "soft":
                out["seam_blend_floor"] = self.seam_blend_floor.detach().to(dtype=raw_blend.dtype).view(1, 1, -1)
            if event_prob is not None:
                out["event_logits"] = event_logits
                out["event_prob"] = event_prob
                out["struct_residual"] = struct_residual
            out["manifold_prior_z"] = manifold_prior_z
            out["z_agg"] = z_agg
            if z_phys_record is not None:
                out["z_phys"] = z_phys_record
            if innovation_w is not None:
                out["manifold_innovation_w"] = innovation_w
            if late_noise_floor is not None:
                out["late_noise_floor"] = late_noise_floor
            if _gate is not None:
                out["innovation_evidence_gate"] = _gate
            if _wkal is not None:
                out["innovation_precision_w"] = _wkal
            if _infl_rec is not None:
                out["innovation_robust_inflation"] = _infl_rec
            if _wout is not None:
                out["output_fusion_w"] = _wout
            if kalman_gain is not None:
                out["kalman_gain"] = kalman_gain
            if prior_sp_record is not None:
                out["prior_log_sigma_sp"] = prior_sp_record
            if manifold_residual is not None:
                out["manifold_residual"] = manifold_residual
            if relax_update_rec is not None:
                out["relaxation_update"] = relax_update_rec
            if self.manifold_head_atoms is not None:
                out["manifold_atom_w"] = atom_w
                out["phys_linear"] = phys
                if ip_w is not None and ip_gate is not None:
                    out["ip_atom_w"] = ip_w
                    out["ip_gate"] = ip_gate
        return out


def sync_model_gates(cfg: Config, logger: Optional[logging.Logger] = None) -> None:
    """Every model construction re-syncs from DataConfig first, or set_gate_scale raises 'gate_scale length 256 does
    not match configured gates 31'.
    """
    if int(cfg.model.gates) != int(cfg.data.target_gates) or int(cfg.model.late_start_index) != int(cfg.data.late_start_index):
        if logger is not None:
            logger.info("model config synced to the data: gates %d -> %d, late_start_index %d -> %d.",
                        int(cfg.model.gates), int(cfg.data.target_gates), int(cfg.model.late_start_index), int(cfg.data.late_start_index))
        cfg.model.gates = int(cfg.data.target_gates)
        cfg.model.late_start_index = int(cfg.data.late_start_index)
    scale_receptive_field(cfg, logger)
    _m = str(getattr(cfg.model, "amplitude_conditioning", "auto")).lower()
    _want = (True if _m == "on" else False if _m == "off"
                else (True if int(cfg.model.gates) > int(RF_DESIGN_GATES) else bool(cfg.model.per_trace_normalization)))
    if bool(cfg.model.per_trace_normalization) != _want:
        cfg.model.per_trace_normalization = _want
    if logger is not None:
        logger.info("per-trace amplitude conditioning %s (mode %s, %d gates, estimator %s, first %d gates). ", "ON" if _want else "OFF", _m, int(cfg.model.gates),
                    str(getattr(cfg.model, "per_trace_norm_estimator", "centre_logmean")),
                    int(max(2, min(int(cfg.model.per_trace_norm_gates), int(cfg.model.late_start_index) // 8))))


RF_DESIGN_GATES = 31
RF_DESIGN_RESIDUAL: Tuple[int, ...] = (1, 2, 4)
RF_DESIGN_MULTISCALE: Tuple[int, ...] = (1, 1, 2, 2)


def receptive_field_for_gates(gates: int) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """(residual_dilations, multiscale_dilations) that carry the 31-gate design's span in NATIVE-gate units to an
    axis of `gates` samples.
    """
    s = float(gates) / float(RF_DESIGN_GATES)
    if s < 1.75:
        return RF_DESIGN_RESIDUAL, RF_DESIGN_MULTISCALE
    top = float(max(RF_DESIGN_RESIDUAL)) * s
    n = max(len(RF_DESIGN_RESIDUAL), int(round(math.log2(top))) + 1)
    cap = max(max(RF_DESIGN_RESIDUAL), int(gates) // 8)
    res = tuple(int(2 ** k) for k in range(n) if int(2 ** k) <= cap)
    m = max(2, int(2 ** int(round(math.log2(s)))))
    ms = (1, max(1, m // 4), max(2, m // 2), m)
    return res, ms


def scale_receptive_field(cfg: Config, logger: Optional[logging.Logger] = None) -> None:
    """Apply receptive_field_for_gates when ModelConfig.receptive_field_scaling is "auto" and the dilations are
    still the 31-gate design.
    """
    m = cfg.model
    if str(getattr(m, "receptive_field_scaling", "auto")).lower() != "auto":
        return
    res_now = tuple(int(v) for v in m.residual_dilations)
    ms_now = tuple(int(v) for v in m.multiscale_dilations)
    if res_now != RF_DESIGN_RESIDUAL or ms_now != RF_DESIGN_MULTISCALE:
        m.receptive_field_scaling = "custom"
        if logger is not None:
            logger.info("receptive field: operator-chosen dilations kept (residual %s, multiscale %s).", res_now, ms_now)
        return
    res, ms = receptive_field_for_gates(int(m.gates))
    if res == res_now and ms == ms_now:
        return
    if len(ms) != len(tuple(m.multiscale_kernels)):
        return
    m.residual_dilations = res
    m.multiscale_dilations = ms
    m.receptive_field_scaling = "applied"
    if logger is not None:
        _k = 3
        _sig_old = math.sqrt(sum(2.0 * (2.0 / 3.0) * float(RF_DESIGN_RESIDUAL[i % len(RF_DESIGN_RESIDUAL)]) ** 2
                                 for i in range(int(m.residual_blocks))))
        _sig_new = math.sqrt(sum(2.0 * (2.0 / 3.0) * float(res[i % len(res)]) ** 2 for i in range(int(m.residual_blocks))))
        _per = float(m.gates) / float(RF_DESIGN_GATES)
        logger.info("RECEPTIVE FIELD RESCALED to the %d-gate axis | residual dilations %s -> %s, "
                    "multiscale %s -> %s (kernel %d, %d blocks) | residual-stack effective sigma %.1f -> %.1f gates "
                    "= %.1f -> %.1f native gates (one native gate = %.2f model gates) | parameter shapes unchanged.",
                    int(m.gates), RF_DESIGN_RESIDUAL, res, RF_DESIGN_MULTISCALE, ms, _k, int(m.residual_blocks),
                    _sig_old, _sig_new, _sig_old / _per, _sig_new / _per, _per)


def receptive_field_from_checkpoint(path: Path, cfg: Config, logger: logging.Logger) -> None:
    """Dilations do not change any tensor shape, so weights trained with one set load silently into a model built
    with another.
    """
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        logger.warning("could not pre-read %s for its receptive field (%r); load_checkpoint will verify it.", path, exc)
        return
    saved = (state.get("config") or {}).get("model", {}) or {}
    del state
    _apply_saved_receptive_field(saved, cfg, logger, str(path))


def _apply_saved_receptive_field(saved: Mapping[str, Any], cfg: Config, logger: logging.Logger, what: str) -> List[str]:
    out: List[str] = []
    if not saved:
        res, ms, mode = RF_DESIGN_RESIDUAL, RF_DESIGN_MULTISCALE, "off"
    else:
        res = tuple(int(v) for v in (saved.get("residual_dilations") or RF_DESIGN_RESIDUAL))
        ms = tuple(int(v) for v in (saved.get("multiscale_dilations") or RF_DESIGN_MULTISCALE))
        mode = str(saved.get("receptive_field_scaling", "off"))
        if mode == "auto":
            mode = "off"
    if tuple(int(v) for v in cfg.model.residual_dilations) != res:
        out.append("residual_dilations: %r -> %r" % (tuple(cfg.model.residual_dilations), res))
    if tuple(int(v) for v in cfg.model.multiscale_dilations) != ms:
        out.append("multiscale_dilations: %r -> %r" % (tuple(cfg.model.multiscale_dilations), ms))
    cfg.model.residual_dilations = res
    cfg.model.multiscale_dilations = ms
    cfg.model.receptive_field_scaling = mode
    for _nm2 in ("gate_local_norm", "decoder_gate_norm_channel", "coef_field_trust_mix",
                   "amplitude_conditioning", "per_trace_norm_estimator"):
        if not hasattr(cfg.model, _nm2):
            continue
        if _nm2 == "amplitude_conditioning":
            _d = "on" if (saved and bool(saved.get("per_trace_normalization", False))) else "off"
        else:
            _d = _PRE_135_ARCH_DEFAULTS.get(_nm2, getattr(cfg.model, _nm2))
        _v = saved.get(_nm2, _d) if saved else _d
        if isinstance(_d, bool):
            _v = bool(_v)
        if getattr(cfg.model, _nm2) != _v:
            out.append("%s: %r -> %r" % (_nm2, getattr(cfg.model, _nm2), _v))
        setattr(cfg.model, _nm2, _v)
    logger.info("receptive field taken from %s: residual %s, multiscale %s (scaling=%s).",
                what, res, ms, mode)
    return out
