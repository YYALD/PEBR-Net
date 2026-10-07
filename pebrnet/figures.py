# PEBR-Net -- prior- and evidence-bounded reconstruction of noise-buried late-time borehole TEM responses.
# MIT License, see LICENSE.
"""Figure style, export and renderers.

This file is executed by pebrnet/__init__.py into the package namespace, in the order listed there;
names defined in the other files of the package are available here without imports.
"""
from __future__ import annotations

_MM_PER_INCH = 25.4
NATURE_DOUBLE_MM = 183.0


OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#999999",
}


def _mm(width_mm: float, height_mm: float) -> Tuple[float, float]:
    return width_mm / _MM_PER_INCH, height_mm / _MM_PER_INCH


def nature_rc() -> Dict[str, Any]:
    """Nature-leaning Matplotlib rcParams: Arial-first sans-serif, 5-7 pt text, hairline axes, no top/right spines,
    vector-friendly font embedding.
    """
    return {
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.5,
        "grid.linewidth": 0.3,
        "lines.linewidth": 0.8,
        "lines.markersize": 2.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "legend.frameon": False,
        "figure.constrained_layout.use": True,
    }


def _thin_symlog_ticks(ax: Any, linthresh: float, vmax: float, step: int = 2) -> None:
    import numpy as _np

    lo = int(math.ceil(math.log10(max(linthresh, 1e-30))))
    hi = int(math.floor(math.log10(max(vmax, linthresh * 10.0))))
    decades = list(range(lo, hi + 1, step)) or [hi]
    y0, y1 = ax.get_ylim()
    cand = [-(10.0**p) for p in reversed(decades)] + [0.0] + [10.0**p for p in decades]
    ticks = [t for t in cand if y0 <= t <= y1] or [t for t in cand if t > 0][:1]
    ax.set_yticks(ticks)
    ax.set_ylim(y0, y1)
    ax.yaxis.set_minor_locator(__import__("matplotlib").ticker.NullLocator())


def _panel_label(ax: Any, text: str) -> None:
    ax.text(
        -0.16,
        1.06,
        text,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _symlog_linthresh(*arrays: np.ndarray) -> float:
    mags = np.concatenate([np.abs(np.asarray(a, dtype=np.float64)).ravel() for a in arrays])
    mags = mags[np.isfinite(mags) & (mags > 0)]
    if mags.size == 0:
        return 1e-6
    return float(max(np.median(mags) * 1e-3, np.min(mags), 1e-12))


def _svg_parse_translate(transform: str) -> Optional[Tuple[float, float]]:
    t = transform.strip()
    m = re.fullmatch(r"translate\(([-\d.eE+]+)[,\s]+([-\d.eE+]+)\)", t)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.fullmatch(r"translate\(([-\d.eE+]+)\)", t)
    if m:
        return float(m.group(1)), 0.0
    return None


def _svg_shift_path_d(d: str, dx: float, dy: float) -> str:
    out: List[str] = []
    i = 0
    tokens = re.findall(r"[A-Za-z]|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", d)
    cmd = ""
    while i < len(tokens):
        tok = tokens[i]
        if re.fullmatch(r"[A-Za-z]", tok):
            cmd = tok
            out.append(tok)
            i += 1
            continue
        if cmd in ("M", "L", "T"):
            n = 2
        elif cmd in ("C",):
            n = 6
        elif cmd in ("Q", "S"):
            n = 4
        elif cmd in ("H",):
            out.append("%.4f" % (float(tok) + dx))
            i += 1
            continue
        elif cmd in ("V",):
            out.append("%.4f" % (float(tok) + dy))
            i += 1
            continue
        elif cmd in ("Z", "z", ""):
            out.append(tok)
            i += 1
            continue
        else:
            raise ValueError("unsupported path command %r" % cmd)
        vals = [float(v) for v in tokens[i : i + n]]
        if len(vals) < n:
            raise ValueError("truncated path data")
        for j in range(0, n, 2):
            out.append("%.4f" % (vals[j] + dx))
            out.append("%.4f" % (vals[j + 1] + dy))
        i += n
    return " ".join(out)


def _svg_clip_polyline(d: str, rect: Tuple[float, float, float, float]) -> Optional[str]:
    x0, y0, x1, y1 = rect
    x0, y0, x1, y1 = x0 - 0.5, y0 - 0.5, x1 + 0.5, y1 + 0.5
    toks = re.findall(r"[A-Za-z]|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", d)
    if any(t in ("C", "Q", "S", "T", "A", "a", "c", "q", "s", "t", "H", "V") for t in toks):
        return None
    pts: List[Optional[Tuple[float, float]]] = []
    i = 0
    _start: Optional[Tuple[float, float]] = None
    while i < len(toks):
        t = toks[i]
        if t in ("M", "L"):
            x = float(toks[i + 1])
            y = float(toks[i + 2])
            if t == "M" and pts:
                pts.append(None)
            if t == "M":
                _start = (x, y)
            pts.append((x, y))
            i += 3
        elif t in ("Z", "z"):
            if _start is not None and pts and pts[-1] is not None and pts[-1] != _start:
                pts.append(_start)
            i += 1
        else:
            return None
    segs: List[List[Tuple[float, float]]] = []
    run: List[Tuple[float, float]] = []
    for p in pts:
        if p is None:
            if len(run) > 1:
                segs.append(run)
            run = []
        else:
            run.append(p)
    if len(run) > 1:
        segs.append(run)
    if not segs:
        return None

    def _liang_barsky(p: Tuple[float, float], q: Tuple[float, float]):
        dx = q[0] - p[0]
        dy = q[1] - p[1]
        t0, t1 = 0.0, 1.0
        for pk, qk in ((-dx, p[0] - x0), (dx, x1 - p[0]), (-dy, p[1] - y0), (dy, y1 - p[1])):
            if abs(pk) < 1e-12:
                if qk < 0:
                    return None
                continue
            r = qk / pk
            if pk < 0:
                if r > t1:
                    return None
                t0 = max(t0, r)
            else:
                if r < t0:
                    return None
                t1 = min(t1, r)
        return (p[0] + t0 * dx, p[1] + t0 * dy), (p[0] + t1 * dx, p[1] + t1 * dy)

    parts: List[str] = []
    for run in segs:
        pen_up = True
        for a, b in zip(run[:-1], run[1:]):
            clipped = _liang_barsky(a, b)
            if clipped is None:
                pen_up = True
                continue
            ca, cb = clipped
            if pen_up:
                parts.append("M %.4f %.4f" % ca)
                pen_up = False
            elif abs(ca[0] - float(parts[-1].split()[-2])) > 1e-3 or abs(ca[1] - float(parts[-1].split()[-1])) > 1e-3:
                parts.append("M %.4f %.4f" % ca)
            parts.append("L %.4f %.4f" % cb)
    return " ".join(parts) if parts else ""


def _svg_clip_polygon(d: str, rect: Tuple[float, float, float, float]) -> Optional[str]:
    x0, y0, x1, y1 = rect
    x0, y0, x1, y1 = x0 - 0.5, y0 - 0.5, x1 + 0.5, y1 + 0.5
    toks = re.findall(r"[A-Za-z]|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", d)
    if any(t in ("C", "Q", "S", "T", "A", "a", "c", "q", "s", "t", "H", "V") for t in toks):
        return None
    polys: List[List[Tuple[float, float]]] = []
    cur: List[Tuple[float, float]] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t == "M":
            if len(cur) > 2:
                polys.append(cur)
            cur = [(float(toks[i + 1]), float(toks[i + 2]))]
            i += 3
        elif t == "L":
            cur.append((float(toks[i + 1]), float(toks[i + 2])))
            i += 3
        elif t in ("Z", "z"):
            i += 1
        else:
            return None
    if len(cur) > 2:
        polys.append(cur)

    def _clip_edge(poly, inside, intersect):
        out = []
        if not poly:
            return out
        prev = poly[-1]
        for p in poly:
            if inside(p):
                if not inside(prev):
                    out.append(intersect(prev, p))
                out.append(p)
            elif inside(prev):
                out.append(intersect(prev, p))
            prev = p
        return out

    def _ix(a, b, axis, val):
        (ax, ay), (bx, by) = a, b
        if axis == 0:
            t = (val - ax) / (bx - ax) if bx != ax else 0.0
            return (val, ay + t * (by - ay))
        t = (val - ay) / (by - ay) if by != ay else 0.0
        return (ax + t * (bx - ax), val)

    parts: List[str] = []
    for poly in polys:
        q = poly
        q = _clip_edge(q, lambda p: p[0] >= x0, lambda a, b: _ix(a, b, 0, x0))
        q = _clip_edge(q, lambda p: p[0] <= x1, lambda a, b: _ix(a, b, 0, x1))
        q = _clip_edge(q, lambda p: p[1] >= y0, lambda a, b: _ix(a, b, 1, y0))
        q = _clip_edge(q, lambda p: p[1] <= y1, lambda a, b: _ix(a, b, 1, y1))
        if len(q) < 3:
            continue
        parts.append("M %.4f %.4f " % q[0] + " ".join("L %.4f %.4f" % p for p in q[1:]) + " Z")
    return " ".join(parts) if parts else ""


_SVG_VISIO_SELFTEST_DONE: bool = False
_SVG_VISIO_SELFTEST_OK: bool = False


def _svg_path_is_filled(node: Any) -> bool:
    st = node.get("style") or ""
    m = re.search(r"(?:^|;)\s*fill\s*:\s*([^;]*)", st)
    if m is not None:
        v = m.group(1).strip().lower()
        if v:
            return v != "none"
    v = (node.get("fill") or "").strip().lower()
    if not v:
        return True
    return v != "none"


def _svg_stroke_polylines(svg_text: str, stroke_hex: str) -> List[Tuple[List[List[Tuple[float, float]]], bool]]:
    import xml.etree.ElementTree as ET
    root = ET.fromstring(svg_text)
    parent = {c: p for p in root.iter() for c in p}
    key = "stroke:" + stroke_hex.strip().lower()
    out: List[Tuple[List[List[Tuple[float, float]]], bool]] = []
    for el in root.iter():
        if el.tag.split("}")[-1] != "path":
            continue
        st = (el.get("style") or "").replace(" ", "").lower()
        if key not in st:
            continue
        toks = re.findall(r"[A-Za-z]|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", el.get("d", ""))
        if any(t in ("C", "Q", "S", "T", "A", "a", "c", "q", "s", "t", "H", "V") for t in toks):
            continue
        tx = ty = 0.0
        nd = el
        while nd is not None:
            tr = nd.get("transform")
            if tr:
                sh = _svg_parse_translate(tr)
                if sh is None:
                    break
                tx += sh[0]
                ty += sh[1]
            nd = parent.get(nd)
        runs: List[List[Tuple[float, float]]] = []
        cur: List[Tuple[float, float]] = []
        closed = False
        i = 0
        while i < len(toks):
            t = toks[i]
            if t == "M":
                if len(cur) > 1:
                    runs.append(cur)
                cur = [(float(toks[i + 1]) + tx, float(toks[i + 2]) + ty)]
                i += 3
            elif t == "L":
                cur.append((float(toks[i + 1]) + tx, float(toks[i + 2]) + ty))
                i += 3
            elif t in ("Z", "z"):
                closed = True
                i += 1
            else:
                i += 1
        if len(cur) > 1:
            runs.append(cur)
        if runs:
            out.append((runs, closed))
    return out


def _svg_polyline_fidelity(raw_svg: str, out_svg: str, stroke_hex: str, tol: float = 2e-3) -> Tuple[bool, int, int, int, int, int]:
    src = _svg_stroke_polylines(raw_svg, stroke_hex)
    dst = _svg_stroke_polylines(out_svg, stroke_hex)
    segs = [(a, b) for runs, _ in src for run in runs for a, b in zip(run[:-1], run[1:])]

    def _on(p, a, b):
        dx, dy = b[0] - a[0], b[1] - a[1]
        l2 = dx * dx + dy * dy
        if l2 < 1e-18:
            return abs(p[0] - a[0]) <= tol and abs(p[1] - a[1]) <= tol
        t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2))
        return abs(p[0] - (a[0] + t * dx)) <= tol and abs(p[1] - (a[1] + t * dy)) <= tol

    n_closed = sum(1 for _, c in dst if c)
    n_fab = 0
    for runs, _ in dst:
        for run in runs:
            for a, b in zip(run[:-1], run[1:]):
                if not any(_on(a, s, e) and _on(b, s, e) for s, e in segs):
                    n_fab += 1
    n_in = sum(len(run) for runs, _ in src for run in runs)
    n_out = sum(len(run) for runs, _ in dst for run in runs)
    ok = bool(dst) and n_closed == 0 and n_fab == 0
    return ok, len(dst), n_closed, n_fab, n_in, n_out


def _svg_text_positions(svg_text: str) -> Tuple[List[Tuple[str, float, float]], int]:
    import xml.etree.ElementTree as ET
    root = ET.fromstring(svg_text)
    parent = {c: p for p in root.iter() for c in p}
    out: List[Tuple[str, float, float]] = []
    skipped = 0

    def _chain(e):
        tx = ty = 0.0
        ok = True
        n = e
        while n is not None:
            tr = n.get("transform")
            if tr:
                sh = _svg_parse_translate(tr)
                if sh is None:
                    ok = False
                    break
                tx += sh[0]
                ty += sh[1]
            n = parent.get(n)
        return ok, tx, ty

    for el in root.iter():
        if el.tag.split("}")[-1] != "text":
            continue
        ok, tx, ty = _chain(el)
        bx = float(el.get("x", 0) or 0)
        by = float(el.get("y", 0) or 0)
        runs = [(el.text or "").strip()] if (el.text or "").strip() else []
        for ts in el.iter():
            if ts.tag.split("}")[-1] != "tspan":
                continue
            txt = (ts.text or "").strip()
            if not txt:
                continue
            if not ok:
                skipped += 1
                continue
            x = float((ts.get("x") or str(bx)).replace(",", " ").split()[0])
            y = float((ts.get("y") or str(by)).replace(",", " ").split()[0])
            out.append((txt, round(tx + x, 2), round(ty + y, 2)))
        for txt in runs:
            if ok:
                out.append((txt, round(tx + bx, 2), round(ty + by, 2)))
            else:
                skipped += 1
    return out, skipped


def _svg_visio_self_test(logger: logging.Logger) -> bool:
    global _SVG_VISIO_SELFTEST_DONE, _SVG_VISIO_SELFTEST_OK
    if _SVG_VISIO_SELFTEST_DONE:
        return bool(_SVG_VISIO_SELFTEST_OK)
    _SVG_VISIO_SELFTEST_DONE = True
    try:
        import matplotlib
        if matplotlib.get_backend().lower() != "agg":
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import xml.etree.ElementTree as ET
        with plt.rc_context(nature_rc()):
            fig, (a0, a1, a2, a3) = plt.subplots(1, 4, figsize=(8.0, 1.8))
            a0.semilogy([1, 2, 3, 4], [1e-3, 1e-1, 1e1, 1e3], "-")
            _x = np.linspace(0.0, 1.0, 61)
            a2.plot(_x, np.exp(-3.0 * _x) + 0.05 * np.sin(40.0 * _x), color="#123456", lw=0.7)
            a3.plot(_x, 2.0 * np.sin(25.0 * _x), color="#654321", lw=0.7)
            a3.set_ylim(-0.5, 0.5)
            a0.set_xlabel("x")
            a0.set_ylabel("amplitude")
            a1.bar([0, 1, 2], [1.0, 0.5, 0.25], width=0.6)
            a1.hist([3.1, 3.2, 5.0, 5.1, 7.0], bins=5)
            raw = io.StringIO()
            fig.savefig(raw, format="svg")
            plt.close(fig)
        out = svg_to_visio(raw.getvalue())
        pos_raw, sk_raw = _svg_text_positions(raw.getvalue())
        pos_out, sk_out = _svg_text_positions(out)
        set_raw = sorted(pos_raw)
        set_out = sorted(pos_out)
        n_math = sum(1 for t, _, _ in pos_raw if t in ("−", "-"))
        def _match(pa: List[Tuple[str, float, float]], pb: List[Tuple[str, float, float]], tol: float = 0.05) -> bool:
            if len(pa) != len(pb):
                return False
            used = [False] * len(pb)
            for t, x, y in pa:
                hit = False
                for j, (t2, x2, y2) in enumerate(pb):
                    if not used[j] and t2 == t and abs(x2 - x) <= tol and abs(y2 - y) <= tol:
                        used[j] = True
                        hit = True
                        break
                if not hit:
                    return False
            return True
        _pos_ok = bool(set_raw == set_out) or _match(pos_raw, pos_out)
        labels_ok = (len(set_raw) > 0 and _pos_ok and n_math >= 1
                     and sk_raw == sk_out)
        root = ET.fromstring(out)
        n_fill = n_ok = 0
        for el in root.iter():
            if el.tag.split("}")[-1] != "path":
                continue
            st = "%s fill:%s" % (el.get("style") or "", el.get("fill") or "")
            if not re.search(r"fill:\s*#", st):
                continue
            dd = el.get("d", "").rstrip()
            nL = dd.count("L")
            if nL >= 3 and dd.endswith(("Z", "z")):
                n_fill += 1
                n_ok += 1
            elif nL >= 2:
                n_fill += 1
        bars_ok = (n_fill >= 8 and n_ok == n_fill)
        with plt.rc_context(nature_rc()):
            f2, (b0, b1) = plt.subplots(1, 2, figsize=(4.0, 1.6), layout="constrained")
            b0.plot([0, 1], [0, 1])
            b1.plot([0, 1], [0, 1])
            tt = b0.text(0.02, 0.03, "x" * 90, transform=b0.transAxes, fontsize=4.0)
            tt.set_in_layout(False)
            f2.canvas.draw()
            w0 = b0.get_position().width
            w1 = b1.get_position().width
            plt.close(f2)
        layout_ok = (w0 >= 0.9 * w1)
        _f_in = _svg_polyline_fidelity(raw.getvalue(), out, "#123456")
        _f_out = _svg_polyline_fidelity(raw.getvalue(), out, "#654321")
        lines_ok = bool(_f_in[0] and _f_out[0] and _f_in[4] == _f_in[5] and _f_in[4] >= 61)
        ok = bool(labels_ok and bars_ok and layout_ok and lines_ok)
        _SVG_VISIO_SELFTEST_OK = ok
        (logger.info if ok else logger.error)(
            "SVG->Visio self-test %s | label glyph runs %d (positions preserved: %s, "
            "mathtext runs %d, rotated runs %d/%d) | filled polygons %d (closed >=4-vertex %d) | "
            "panel width with long caption / without = %.2f | data polylines: "
            "inside curve closed %d / fabricated segments %d / vertices %d->%d, box-leaving curve closed %d / "
            "fabricated segments %d. ",
            "PASSED" if ok else "FAILED", len(set_raw), "yes" if _pos_ok else "NO",
            n_math, sk_raw, sk_out, n_fill, n_ok, w0 / max(w1, 1e-9),
            _f_in[2], _f_in[3], _f_in[4], _f_in[5], _f_out[2], _f_out[3])
        return ok
    except Exception as exc:
        _SVG_VISIO_SELFTEST_OK = False
        logger.warning("SVG->Visio self-test skipped: %r", exc)
        return False


def svg_to_visio(svg_text: str, font_family: str = "Arial", font_size_pt: float = 7.0) -> str:
    """Rewrite a Matplotlib SVG into one that Microsoft Visio can open, render completely and edit."""
    import xml.etree.ElementTree as ET

    SVG = "http://www.w3.org/2000/svg"
    XLINK = "http://www.w3.org/1999/xlink"
    ET.register_namespace("", SVG)
    root = ET.fromstring(svg_text)

    def _tag(e) -> str:
        return e.tag.split("}")[-1] if isinstance(e.tag, str) else ""

    defs: Dict[str, Any] = {}
    clips: Dict[str, Tuple[float, float, float, float]] = {}
    for parent in root.iter():
        for child in list(parent):
            t = _tag(child)
            if t == "defs":
                for node in child.iter():
                    nid = node.get("id")
                    if nid and _tag(node) == "path":
                        defs[nid] = node
                    if nid and _tag(node) == "clipPath":
                        rect = node.find("{%s}rect" % SVG)
                        if rect is not None:
                            x = float(rect.get("x", 0))
                            y = float(rect.get("y", 0))
                            w = float(rect.get("width", 0))
                            h = float(rect.get("height", 0))
                            clips[nid] = (x, y, x + w, y + h)
                parent.remove(child)
            elif t in ("metadata",):
                parent.remove(child)

    flat: List[Any] = []

    def _walk(node, dx: float, dy: float, clip: Optional[str]) -> None:
        t = _tag(node)
        cp = node.get("clip-path")
        if cp:
            m = re.match(r"url\(#([^)]+)\)", cp)
            if m:
                clip = m.group(1)
        if t == "g":
            tr = node.get("transform", "")
            shift = _svg_parse_translate(tr) if tr else (0.0, 0.0)
            if tr and shift is None:
                node.set("transform", ("translate(%.4f %.4f) %s" % (dx, dy, tr))
                         if (dx or dy) else tr)
                flat.append(("group", node, 0.0, 0.0, clip))
                return
            ndx, ndy = (dx + shift[0], dy + shift[1]) if shift else (dx, dy)
            for ch in list(node):
                _walk(ch, ndx, ndy, clip)
            return
        flat.append(("leaf", node, dx, dy, clip))

    for child in list(root):
        _walk(child, 0.0, 0.0, None)
    for child in list(root):
        root.remove(child)

    def _merge_style(base: Optional[str], extra: Optional[str]) -> Optional[str]:
        if not base:
            return extra
        if not extra:
            return base
        return base.rstrip("; ") + "; " + extra

    n_use = 0
    n_clipped = 0
    for kind, node, dx, dy, clip in flat:
        t = _tag(node)
        if t == "use":
            ref = node.get("{%s}href" % XLINK) or node.get("href") or ""
            src = defs.get(ref.lstrip("#"))
            if src is None:
                continue
            ux = float(node.get("x", 0)) + dx
            uy = float(node.get("y", 0)) + dy
            new = ET.Element("{%s}path" % SVG)
            try:
                new.set("d", _svg_shift_path_d(src.get("d", ""), ux, uy))
            except ValueError:
                continue
            st = _merge_style(src.get("style"), node.get("style"))
            if st:
                new.set("style", st)
            if src.get("fill"):
                new.set("fill", src.get("fill"))
            root.append(new)
            n_use += 1
            continue
        if t == "path":
            d = node.get("d", "")
            if d:
                try:
                    d = _svg_shift_path_d(d, dx, dy)
                except ValueError:
                    continue
                if clip and clip in clips:
                    _filled = _svg_path_is_filled(node)
                    cd = (_svg_clip_polygon(d, clips[clip]) if _filled
                          else _svg_clip_polyline(d, clips[clip]))
                    if cd is not None:
                        if cd == "":
                            continue
                        d = cd
                        n_clipped += 1
                node.set("d", d)
        elif t in ("rect", "image"):
            node.set("x", "%.4f" % (float(node.get("x", 0)) + dx))
            node.set("y", "%.4f" % (float(node.get("y", 0)) + dy))
        elif t in ("text",):
            node.set("x", "%.4f" % (float(node.get("x", 0)) + dx))
            node.set("y", "%.4f" % (float(node.get("y", 0)) + dy))
            node.set("font-family", font_family)
            if not node.get("font-size"):
                node.set("font-size", "%gpx" % font_size_pt)
            for sub in node.iter():
                if _tag(sub) == "tspan":
                    sub.set("font-family", font_family)
                    for _ax, _dd in (("x", dx), ("y", dy)):
                        _v = sub.get(_ax)
                        if _v is not None:
                            try:
                                sub.set(_ax, " ".join(
                                    "%.4f" % (float(_t) + _dd) for _t in _v.replace(",", " ").split()))
                            except ValueError:
                                pass
        elif t == "g":
            tr = node.get("transform", "")
            if dx or dy:
                node.set("transform", "translate(%.4f %.4f) %s" % (dx, dy, tr))
        if node.get("clip-path"):
            del node.attrib["clip-path"]
        root.append(node)

    root.set("visio-compatible", "true")
    body = ET.tostring(root, encoding="unicode")
    body = body.replace(' visio-compatible="true"', "")
    return '<?xml version="1.0" encoding="utf-8" standalone="no"?>\n' + body


_UNQUALIFIED_BANNER: str = ""


def set_unqualified_banner(text: str) -> None:
    """The sentence stamped across every figure of a non-qualified run."""
    global _UNQUALIFIED_BANNER
    _UNQUALIFIED_BANNER = str(text or "")


def _stamp_unqualified(fig: Any) -> None:
    if not _UNQUALIFIED_BANNER:
        return
    try:
        fig.text(0.5, 0.5, "NOT QUALIFIED\nDIAGNOSTIC ONLY", transform=fig.transFigure,
                 fontsize=52, color="#d62728", alpha=0.22, rotation=32,
                 ha="center", va="center", zorder=1000, fontweight="bold")
        fig.text(0.5, 0.008, _UNQUALIFIED_BANNER, transform=fig.transFigure, fontsize=7.5,
                 color="#b00020", ha="center", va="bottom", zorder=1000, wrap=True)
    except Exception:
        pass


FIGURE_PNG_DPI = 600


def _save_figure(fig: Any, figure_dir: Path, stem: str) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = tagged_name(stem + ".x")[:-2]
    _stamp_unqualified(fig)
    fig.savefig(figure_dir / f"{stem}.pdf")
    fig.savefig(figure_dir / f"{stem}.png", dpi=FIGURE_PNG_DPI)
    raw = io.StringIO()
    fig.savefig(raw, format="svg")
    svg_dir = figure_dir / "svg_visio"
    svg_dir.mkdir(parents=True, exist_ok=True)
    _ok = _svg_visio_self_test(logging.getLogger(PROGRAM_NAME))
    try:
        if not _ok:
            raise RuntimeError("svg_to_visio self-test FAILED; the raw Matplotlib SVG is written instead")
        (svg_dir / f"{stem}.svg").write_text(svg_to_visio(raw.getvalue()), encoding="utf-8")
    except Exception as exc:
        (svg_dir / f"{stem}.svg").write_text(raw.getvalue(), encoding="utf-8")
        logging.getLogger(PROGRAM_NAME).warning(
            "SVG Visio post-processing failed for %s (%s); the raw Matplotlib SVG was written instead.",
            stem, exc,
        )


def _pick_snr_percentile_rows(snr: np.ndarray) -> Tuple[List[int], List[Tuple[float, str]]]:
    snr = np.asarray(snr, dtype=np.float64)
    order = np.argsort(snr)
    n = int(snr.size)
    ranks = [max(0, min(n - 1, int(round(q * (n - 1))))) for q in (0.05, 0.50, 0.95)]
    for i in range(1, 3):
        if ranks[i] <= ranks[i - 1]:
            ranks[i] = min(n - 1, ranks[i - 1] + 1)
    picks = [int(order[r]) for r in ranks]
    labels = [(float(snr[p]), lab) for p, lab in
              zip(picks, ("low SNR (p5)", "mid SNR (p50)", "high SNR (p95)"))]
    return picks, labels


def _injected_snr_panels(model: "PEBRNet", cfg: Config,
                         clean_cache: CleanCacheInfo, noise_cache: NoiseCacheInfo,
                         targets_db: Tuple[float, ...],
                         device: torch.device) -> List[Tuple[float, np.ndarray,
                                                             np.ndarray, np.ndarray,
                                                             float]]:
    splits = build_splits(clean_cache, cfg)
    low_cfg = dataclasses.replace(
        cfg.data,
        paired_residual_replay=0.0, paired_alt_corruption=0.0,
        measured_noise_inject=1.0,
        measured_noise_snr_min_db=min(targets_db) - 1.5,
        measured_noise_snr_max_db=max(targets_db) + 1.5,
        measured_noise_snr_low_focus=0.0,
        paired_anomaly_oversample=0.0, paired_hard_snr_oversample=0.0,
        neighbor_channel_dropout=0.0,
        survey_profile_fraction=0.0)
    ds = BTEMDenoisingDataset(
        clean_cache.data_path, noise_cache.data_path, splits["val"],
        clean_cache.global_scale, low_cfg, 4431, 240, True,
        noise_block_rows=noise_cache.block_rows,
        gate_times=noise_cache.target_time,
        real_neighbor_rows=province_real_neighbor_rows(clean_cache, cfg),
        real_neighbor_geometry=province_neighbor_geometry(clean_cache, cfg),
        province_id=getattr(clean_cache, "province_id", None),
        paired_noisy_path=getattr(clean_cache, "paired_noisy_path", ""),
        paired_background_path=getattr(clean_cache, "paired_background_path", ""),
        paired_event_path=getattr(clean_cache, "paired_event_path", ""),
        measured_noise_path=getattr(clean_cache, "measured_noise_path", ""),
        extra_clean_path=getattr(clean_cache, "extra_clean_path", ""),
        survey_profile_path=getattr(clean_cache, "survey_profile_path", ""))
    rows = []
    for i in range(len(ds)):
        s = ds[i]
        sv = float(s.get("inject_snr_db", torch.tensor(float("nan"))))
        if math.isfinite(sv):
            rows.append((sv, s))
    if not rows:
        raise RuntimeError("no injected samples produced")
    was_training = model.training
    model.eval()
    out: List[Tuple[float, np.ndarray, np.ndarray, np.ndarray, float]] = []
    gscale = float(clean_cache.global_scale)
    with torch.no_grad():
        for t in targets_db:
            sv, s = min(rows, key=lambda r: abs(r[0] - t))
            x = s["noisy"][None].to(device)
            nb = s.get("neighbors")
            geo = s.get("neighbor_geometry")
            _dn = s.get("depth_norm")
            pred = model(x, nb[None].to(device) if nb is not None else None,
                         geo[None].to(device) if geo is not None else None,
                         depth_norm=(_dn.reshape(1, 1).to(device) if isinstance(_dn, torch.Tensor) else None))["denoised"]
            out.append((float(t),
                        (s["noisy"][0].cpu().numpy() * gscale),
                        (s["clean"][0].cpu().numpy() * gscale),
                        (pred[0, 0].float().cpu().numpy() * gscale),
                        sv))
    if was_training:
        model.train()
    logging.getLogger(PROGRAM_NAME).info(
        "injected-SNR decay panels: SINGLE-STATION diagnostic forward (neighbours + depth, no area window); "
        "targets/realized SNR: %s. Not the deliverable path -- do not read it as such.",
        " ".join("%.0f->%.1f dB" % (r_[0], r_[4]) for r_ in out))
    return out


def render_simulation_denoising_figure(
    model: PEBRNet,
    cfg: Config,
    run_dir: Path,
    clean_cache: CleanCacheInfo,
    noise_cache: NoiseCacheInfo,
    logger: logging.Logger,
) -> None:
    """Nature-style before/after figure on held-out *simulation* data."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception as exc:
        logger.warning("Simulation figure skipped (matplotlib unavailable): %s", exc)
        return

    device = resolve_device(cfg.runtime.device)
    splits = build_splits(clean_cache, cfg)
    n_examples = int(min(2048, max(cfg.data.val_samples, 512), len(splits["val"]) * 8))
    ds = BTEMDenoisingDataset(
        clean_cache.data_path,
        noise_cache.data_path,
        splits["val"],
        clean_cache.global_scale,
        cfg.data,
        cfg.train.seed + 101,
        n_examples,
        False,
        noise_block_rows=noise_cache.block_rows,
        gate_times=noise_cache.target_time,
        real_neighbor_rows=province_real_neighbor_rows(clean_cache, cfg),
        real_neighbor_geometry=province_neighbor_geometry(clean_cache, cfg),
        province_id=getattr(clean_cache, "province_id", None),
        paired_noisy_path=getattr(clean_cache, "paired_noisy_path", ""),
        paired_background_path=getattr(clean_cache, "paired_background_path", ""),
        paired_event_path=getattr(clean_cache, "paired_event_path", ""),
        measured_noise_path=getattr(clean_cache, "measured_noise_path", ""),
        extra_clean_path=getattr(clean_cache, "extra_clean_path", ""),
    )
    ds.set_epoch(0)
    loader = DataLoader(ds, batch_size=block_aligned_batch_size(
        ds, min(cfg.train.batch_size, n_examples)), shuffle=False)
    scale = clean_cache.global_scale
    late = cfg.data.late_start_index
    eps = cfg.loss.eps

    cleans, noisys, denoiseds, rels, snrs = [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for raw in loader:
            batch = move_batch(raw, device)
            out = model(batch["noisy"], batch.get("neighbors"), batch.get("neighbor_geometry"),
                  depth_norm=batch.get("depth_norm"), profile_len=batch_profile_len(batch))
            cleans.append(batch["clean"].cpu().numpy()[:, 0, :] * scale)
            noisys.append(batch["noisy"].cpu().numpy()[:, 0, :] * scale)
            denoiseds.append(out["denoised"].cpu().numpy()[:, 0, :] * scale)
            rels.append(out["reliability"].cpu().numpy()[:, 0, :])
            snrs.append(batch["snr_db"].cpu().numpy().ravel())
    clean = np.concatenate(cleans, 0)
    noisy = np.concatenate(noisys, 0)
    denoised = np.concatenate(denoiseds, 0)
    reliability = np.concatenate(rels, 0)
    snr = np.concatenate(snrs, 0)
    time_axis = np.asarray(clean_cache.target_time, dtype=np.float64)

    def per_sample_nrmse(a: np.ndarray, sl: slice) -> np.ndarray:
        num = np.sqrt(np.mean((a[:, sl] - clean[:, sl]) ** 2, axis=1))
        den = np.sqrt(np.mean(clean[:, sl] ** 2, axis=1)) + eps
        return 100.0 * num / den

    late_err_out = per_sample_nrmse(denoised, slice(late, None))
    late_err_in = per_sample_nrmse(noisy, slice(late, None))

    picks, targets = _pick_snr_percentile_rows(snr)
    _inj_panels = None
    _mn_path = str(getattr(clean_cache, "measured_noise_path", "") or "")
    if _mn_path:
        try:
            _inj_panels = _injected_snr_panels(
                model, cfg, clean_cache, noise_cache,
                targets_db=(-40.0, -25.0, -10.0, -5.0, 5.0, 15.0),
                device=next(model.parameters()).device)
        except Exception as _iexc:
            logger.warning(
                "Injected-SNR panels unavailable (%s); falling back to the natural-distribution "
                "percentiles.", _iexc)

    _ncol = int(len(_inj_panels)) if _inj_panels else 3
    _ncol = max(3, _ncol)
    _t = max(1, _ncol // 3)
    with plt.rc_context(nature_rc()):
        _nat = native_presentation_axis(cfg, time_axis)
        _time_p = _nat if _nat is not None else time_axis
        _noisy_p, _clean_p, _den_p = (to_presentation_axis(noisy, time_axis, _nat), to_presentation_axis(clean, time_axis, _nat),
                                      to_presentation_axis(denoised, time_axis, _nat))
        _rel_p = to_presentation_axis(reliability, time_axis, _nat)
        _late_p = native_late_start(late, time_axis, _nat)
        fig = plt.figure(figsize=_mm(NATURE_DOUBLE_MM, 120.0))
        gs = fig.add_gridspec(2, _ncol, hspace=0.42, wspace=0.34)
        lin = _symlog_linthresh(clean, denoised)
        if _inj_panels is not None:
            _cols = [(xi, yi, di,
                      "injected %+.0f dB (library noise)" % t, s)
                     for (t, xi, yi, di, s) in _inj_panels]
        else:
            _cols = [(noisy[idx], clean[idx], denoised[idx],
                      label + " (natural)", float(snr[idx]))
                     for idx, (_, label) in zip(picks, targets)]
        for col, (_xn, _yc, _dn, _lab, _sr) in enumerate(_cols):
            ax = fig.add_subplot(gs[0, col])
            _xn, _yc, _dn = (to_presentation_axis(_xn, time_axis, _nat), to_presentation_axis(_yc, time_axis, _nat),
                             to_presentation_axis(_dn, time_axis, _nat))
            _mk = _nat is not None
            ax.plot(_time_p, _xn, color=OKABE_ITO["grey"], lw=0.6, alpha=0.9, zorder=1,
                    marker="o" if _mk else None, ms=1.6, mew=0)
            ax.plot(_time_p, _yc, color=OKABE_ITO["black"], lw=1.0, zorder=3)
            ax.plot(_time_p, _dn, color=OKABE_ITO["vermillion"], lw=0.9, ls=(0, (4, 1.5)), zorder=4,
                    marker="s" if _mk else None, ms=1.4, mew=0)
            set_gate_time_xscale(ax, _time_p)
            ax.set_yscale("symlog", linthresh=lin)
            ax.axvline(_time_p[min(_late_p, _time_p.size - 1)], color=OKABE_ITO["sky"], lw=0.5, ls=":", zorder=0)
            ax.set_title(_lab.replace("injected ", "inj. ").replace(
                             " (library noise)", "")
                         + "\nreal. %+.1f dB" % _sr, fontsize=5.2, pad=1.6)
            _thin_symlog_ticks(ax, lin, float(np.nanmax(np.abs(_yc))))
            ax.tick_params(labelleft=(col == 0), labelsize=5)
            ax.set_xlabel("Time (s)")
            if col == 0:
                ax.set_ylabel(r"$\mathrm{d}B/\mathrm{d}t$ (a.u.)")
            _panel_label(ax, chr(ord("a") + col))

        legend_handles = [
            Line2D([0], [0], color=OKABE_ITO["black"], lw=1.0, label="Clean (truth)"),
            Line2D([0], [0], color=OKABE_ITO["grey"], lw=0.8, label="Noisy input"),
            Line2D([0], [0], color=OKABE_ITO["vermillion"], lw=0.9, ls=(0, (4, 1.5)), label="Denoised"),
            Line2D([0], [0], color=OKABE_ITO["sky"], lw=0.5, ls=":", label="Late-window start"),
        ]
        fig.legend(
            handles=legend_handles,
            loc="outside upper center",
            ncol=4,
            columnspacing=1.4,
            handlelength=1.8,
        )

        axd = fig.add_subplot(gs[1, 0:_t])
        rel_in = 100.0 * np.median(np.abs(_noisy_p - _clean_p) / (np.abs(_clean_p) + eps), axis=0)
        rel_out = 100.0 * np.median(np.abs(_den_p - _clean_p) / (np.abs(_clean_p) + eps), axis=0)
        gate_idx = np.arange(1, _clean_p.shape[1] + 1)
        axd.plot(gate_idx, rel_in, color=OKABE_ITO["grey"], marker="o", ms=2.0, lw=0.7, label="Noisy")
        axd.plot(gate_idx, rel_out, color=OKABE_ITO["blue"], marker="s", ms=2.0, lw=0.8, label="Denoised")
        axd.axvspan(_late_p + 0.5, _clean_p.shape[1] + 0.5, color=OKABE_ITO["yellow"], alpha=0.18, lw=0)
        axd.set_yscale("log")
        axd.set_xlabel("Gate index")
        axd.set_ylabel("Median rel. error (%)")
        axd.legend(loc="upper left")
        _panel_label(axd, "d")

        axe = fig.add_subplot(gs[1, _t:2 * _t])
        axe.scatter(snr, late_err_in, s=3, color=OKABE_ITO["grey"], alpha=0.5, edgecolors="none", label="Noisy")
        axe.scatter(snr, late_err_out, s=3, color=OKABE_ITO["vermillion"], alpha=0.6, edgecolors="none", label="Denoised")
        edges = np.arange(cfg.data.snr_min_db, cfg.data.snr_max_db + 5.0, 5.0)
        centers = 0.5 * (edges[:-1] + edges[1:])
        binned = [np.median(late_err_out[(snr >= edges[i]) & (snr < edges[i + 1])]) if np.any((snr >= edges[i]) & (snr < edges[i + 1])) else np.nan for i in range(len(edges) - 1)]
        axe.plot(centers, binned, color=OKABE_ITO["blue"], lw=1.0, marker="D", ms=2.5, label="Denoised (median)")
        axe.axhline(4.0, color=OKABE_ITO["green"], lw=0.6, ls="--")
        axe.set_yscale("log")
        axe.set_xlabel("Input SNR (dB)")
        axe.set_ylabel("Late-window NRMSE (%)")
        axe.legend(loc="upper right")
        _panel_label(axe, "e")

        axf = fig.add_subplot(gs[1, 2 * _t:_ncol])
        q50 = np.median(_rel_p, axis=0)
        q25 = np.percentile(_rel_p, 25, axis=0)
        q75 = np.percentile(_rel_p, 75, axis=0)
        axf.fill_between(gate_idx, q25, q75, color=OKABE_ITO["sky"], alpha=0.3, lw=0)
        axf.plot(gate_idx, q50, color=OKABE_ITO["blue"], lw=1.0, marker="o" if _nat is not None else None, ms=1.8, mew=0)
        axf.axvspan(_late_p + 0.5, _clean_p.shape[1] + 0.5, color=OKABE_ITO["yellow"], alpha=0.18, lw=0)
        axf.set_ylim(-0.02, 1.02)
        axf.set_xlabel("Gate index")
        axf.set_ylabel("Reliability q")
        _panel_label(axf, "f")

        figure_dir = run_dir / "figures"
        _save_figure(fig, figure_dir, "fig_simulation_denoising")
        plt.close(fig)

    summary = {
        "examples": int(clean.shape[0]),
        "late_nrmse_median_in_percent": float(np.median(late_err_in)),
        "late_nrmse_median_out_percent": float(np.median(late_err_out)),
        "late_nrmse_p90_out_percent": float(np.percentile(late_err_out, 90)),
        "global_nrmse_median_out_percent": float(np.median(per_sample_nrmse(denoised, slice(None)))),
    }
    atomic_json_dump(summary, run_dir / "metrics" / "simulation_figure_summary.json")
    logger.info(
        "Simulation figure written | held-out late NRMSE median %.2f%% (noisy %.2f%%) over %d traces",
        summary["late_nrmse_median_out_percent"],
        summary["late_nrmse_median_in_percent"],
        summary["examples"],
    )


def render_late_window_figure(payload: Dict[str, Any], cfg: "Config", run_dir: Path, logger: logging.Logger) -> None:
    """The six-panel late-window group figure."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception as exc:
        logger.warning("Late-window figure skipped (matplotlib unavailable): %s", exc)
        return
    rows = payload["rows"]
    if not rows:
        return
    tau = float(payload["tau"])
    raw = payload["raw"]
    x = np.arange(len(rows))
    xt = [r["snr_bin"] for r in rows]
    figure_dir = run_dir / "figures"

    def col(name: str, default=np.nan) -> np.ndarray:
        return np.asarray([r.get(name, default) for r in rows], dtype=np.float64)

    with plt.rc_context(nature_rc()):
        fig, axes = plt.subplots(2, 3, figsize=_mm(NATURE_DOUBLE_MM, 104), layout="constrained")
        ax = axes[0][0]
        ax.plot(x, col("late_nrmse_percent"), marker="o", color=OKABE_ITO["blue"], label="Full model")
        ax.plot(x, col("manifold_late_nrmse_percent"), marker="s", color=OKABE_ITO["vermillion"],
                ls=(0, (4, 1.5)), label="Prior alone (manifold)")
        ax.axhline(tau, color=OKABE_ITO["sky"], lw=0.7, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels(xt, rotation=45, ha="right", fontsize=5)
        ax.set_ylabel("Late NRMSE (%)")
        ax.set_xlabel("Input SNR (dB)")
        ax.legend(fontsize=5, loc="upper right", framealpha=0.9)
        ax.set_title("Full model vs. the prior alone")
        _panel_label(ax, "a")

        ax = axes[0][1]
        ax.plot(x, col("blend_late"), marker="o", color=OKABE_ITO["vermillion"], label=r"Blend $\alpha$ (prior weight)")
        if np.isfinite(col("measured_reliability")).any():
            ax.plot(x, col("measured_reliability"), marker="^", color=OKABE_ITO["green"],
                    ls=(0, (1.5, 1.2)), label="Measured reliability $r$")
        ax.axhline(0.95, color=OKABE_ITO["grey"], lw=0.6, ls="--")
        ax.set_ylim(-0.03, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels(xt, rotation=45, ha="right", fontsize=5)
        ax.set_ylabel("Gate value")
        ax.set_xlabel("Input SNR (dB)")
        ax.legend(fontsize=5, loc="lower left")
        ax.set_title("Blend gate against the measured\nreliability of the same gates")
        _panel_label(ax, "b")

        ax = axes[0][2]
        iw = col("innovation_weight")
        ax.plot(x, iw, marker="o", color=OKABE_ITO["blue"])
        ax.axhline(0.02, color=OKABE_ITO["vermillion"], lw=0.7, ls="--")
        ax.text(0.03, 0.04, "below the dashed line the\nmeasurement channel is dead",
                transform=ax.transAxes, fontsize=4.6, va="bottom")
        ax.set_ylim(bottom=0.0)
        ax.set_xticks(x)
        ax.set_xticklabels(xt, rotation=45, ha="right", fontsize=5)
        ax.set_ylabel("Innovation gain $w$")
        ax.set_xlabel("Input SNR (dB)")
        ax.set_title("Kalman gain: alive or nominal?")
        _panel_label(ax, "c")

        ax = axes[1][0]
        keys = [("share_denoise", "Denoising path (measurement)", OKABE_ITO["sky"]),
                ("share_innovation", "Innovation (measurement)", OKABE_ITO["green"]),
                ("share_prior", "Learned prior", OKABE_ITO["vermillion"])]
        base = np.zeros(len(rows))
        for k, lab, c in keys:
            v = 100.0 * np.nan_to_num(col(k))
            ax.bar(x, v, 0.65, bottom=base, color=c, alpha=0.85, edgecolor="white", lw=0.4, label=lab)
            base = base + v
        ax.set_ylim(0, 100)
        ax.set_xticks(x)
        ax.set_xticklabels(xt, rotation=45, ha="right", fontsize=5)
        ax.set_ylabel("Share of the late output (%)")
        ax.set_xlabel("Input SNR (dB)")
        ax.legend(fontsize=4.4, loc="upper center", ncol=3, framealpha=0.9,
                  handlelength=1.0, columnspacing=0.6)
        ax.set_ylim(0, 128)
        ax.set_title("Where the late window comes from")
        _panel_label(ax, "d")

        ax = axes[1][1]
        if np.isfinite(col("late_measurement_snr_db")).any():
            ax.plot(x, col("late_measurement_snr_db"), marker="o", color=OKABE_ITO["orange"])
            ax.axhline(20.0 * math.log10(1.0 / (tau / 100.0)), color=OKABE_ITO["sky"], lw=0.7, ls=":")
            ax.text(0.02, 0.98, "dotted line: SNR the late measurement\nneeds to meet the %.0f%% contract alone" % tau,
                    transform=ax.transAxes, fontsize=4.4, va="top")
        ax.set_xticks(x)
        ax.set_xticklabels(xt, rotation=45, ha="right", fontsize=5)
        ax.set_ylabel("Late-gate measurement SNR after stacking (dB)")
        ax.set_xlabel("Input SNR (dB)")
        ax.set_title("How much the data actually carries")
        _panel_label(ax, "e")

        ax = axes[1][2]
        for vals, c, lab in ((raw["late_err"], OKABE_ITO["blue"], "Full model"),
                             (raw["man_err"], OKABE_ITO["vermillion"], "Prior alone")):
            if vals.size:
                vx, vy = _ecdf(vals)
                ax.step(vx, vy, where="post", color=c, label=lab)
        ax.axvline(tau, color=OKABE_ITO["sky"], lw=0.7, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("Per-curve late NRMSE (%)")
        ax.set_ylabel("Empirical CDF")
        ax.legend(fontsize=5, loc="lower right")
        ax.set_title("Per-curve late error distribution")
        _panel_label(ax, "f")
        _save_figure(fig, figure_dir, "fig_late_window_diagnosis")
        plt.close(fig)
    logger.info("Late-window group figure rendered: fig_late_window_diagnosis (PDF/PNG/Visio-SVG) -> %s", figure_dir)


def render_diagnosis_figure(payload: Dict[str, Any], cfg: "Config", run_dir: Path, logger: logging.Logger) -> None:
    """One figure that shows which of the five causes is operating."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception as exc:
        logger.warning("Diagnosis figure skipped (matplotlib unavailable): %s", exc)
        return

    figure_dir = run_dir / "figures"
    tau = float(payload["tau"])
    with plt.rc_context(nature_rc()):
        fig, axes = plt.subplots(1, 4, figsize=_mm(NATURE_DOUBLE_MM, 52), layout="constrained")
        ax = axes[0]
        names = ["TRAIN\n(eval)", "VAL", "TEST"]
        keys = ["m_tr", "m_va", "m_te"]
        w = 0.26
        metric_cols = [OKABE_ITO["grey"], OKABE_ITO["sky"], OKABE_ITO["vermillion"]]
        for j, (lab, met) in enumerate((("G", "global_error_percent"),
                                        ("EM", "early_mid_error_percent"),
                                        ("L", "late_error_percent"))):
            vals = [payload[k][met] for k in keys]
            ax.bar(np.arange(3) + (j - 1) * w, vals, w * 0.92,
                   color=metric_cols[j], alpha=0.85, label=lab)
        ax.axhline(tau, color=OKABE_ITO["vermillion"], lw=0.7, ls="--")
        ax.set_xticks(range(3))
        ax.set_xticklabels(names, fontsize=5)
        ax.set_ylabel("NRMSE (%)")
        ax.set_title("Identical protocol")
        ax.legend(fontsize=5, ncol=3, loc="upper left", framealpha=0.9)
        _panel_label(ax, "a")

        ax = axes[1]
        for vals, col, lab in ((payload["late_tr"], OKABE_ITO["grey"], "Train (eval)"),
                               (payload["late_va"], OKABE_ITO["blue"], "Validation"),
                               (payload["late_te"], OKABE_ITO["green"], "Test")):
            vx, vy = _ecdf(vals)
            ax.step(vx, vy, where="post", color=col, label=lab)
        ax.axvline(tau, color=OKABE_ITO["vermillion"], lw=0.7, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("Per-curve late NRMSE (%)")
        ax.set_ylabel("Empirical CDF")
        ax.legend(loc="lower right", fontsize=5)
        ax.set_title("A gap here is variance;\nno gap and both high is bias")
        _panel_label(ax, "b")

        ax = axes[2]
        mem = payload["mem"]
        if "_arrays" in mem:
            dd = mem["_arrays"]["d"]
            ee = mem["_arrays"]["err"]
            ax.scatter(dd, ee, s=2.5, lw=0, color=OKABE_ITO["blue"], alpha=0.35)
            qs = mem["quintiles"]
            ax.plot([q["nn_distance_median"] for q in qs],
                    [q["late_error_median_percent"] for q in qs],
                    marker="o", color=OKABE_ITO["vermillion"], lw=1.0)
            ax.set_yscale("log")
            ax.text(0.03, 0.95, r"Spearman $\rho$ = %.2f" % mem["spearman_error_vs_nn_distance"],
                    transform=ax.transAxes, fontsize=5, va="top")
        ax.axhline(tau, color=OKABE_ITO["vermillion"], lw=0.6, ls=":")
        ax.set_xlabel("Distance to the nearest TRAINING curve")
        ax.set_ylabel("Validation late NRMSE (%)")
        ax.set_title("Memorization probe:\na rising trend = library too sparse")
        _panel_label(ax, "c")

        ax = axes[3]
        attr = payload["attr"]
        items = [(k, v) for k, v in attr.items() if isinstance(v, float) and abs(v) <= 1.0]
        if items:
            labels_a = [k.replace("_", " ")[:22] for k, _ in items]
            vals = [v for _, v in items]
            colors = [OKABE_ITO["vermillion"] if abs(v) > 0.25 else OKABE_ITO["grey"] for v in vals]
            ax.barh(range(len(items)), vals, 0.6, color=colors, alpha=0.85)
            ax.set_yticks(range(len(items)))
            ax.set_yticklabels(labels_a, fontsize=4.6)
            ax.axvline(0.0, color=OKABE_ITO["black"], lw=0.5)
            ax.set_xlim(-1, 1)
        ax.set_xlabel(r"Spearman $\rho$ with the validation error")
        ax.set_title("What the failures have in common")
        _panel_label(ax, "d")
        _save_figure(fig, figure_dir, "fig_generalization_diagnosis")
        plt.close(fig)
    logger.info("Diagnosis figure rendered: fig_generalization_diagnosis -> %s", figure_dir)


def verify_figures(
    run_dir: Path,
    expected: Sequence[str],
    skipped: Mapping[str, str],
    logger: logging.Logger,
    deployment: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Check every expected figure exists in PDF, PNG and Visio-SVG, print an itemized table, and fail loudly on
    anything missing.
    """
    figure_dir = run_dir / "figures"
    svg_dir = figure_dir / "svg_visio"
    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    by_stem = {f["stem"]: f for f in FIGURE_MANIFEST}
    for stem in expected:
        tagged = tagged_name(stem + ".x")[:-2]
        _tag = str(output_tag() or "")
        _parts = [p for p in _tag.split("__") if p]
        _cands = [stem + ("__" + "__".join(_parts[:k]) if k else "") for k in range(len(_parts), -1, -1)]
        pdf = png = svg = False
        for _c in _cands:
            _p, _g, _s = (figure_dir / f"{_c}.pdf").exists(), (figure_dir / f"{_c}.png").exists(), (svg_dir / f"{_c}.svg").exists()
            if _p and _g and _s:
                tagged, pdf, png, svg = _c, True, True, True
                break
            if (_p or _g or _s) and not (pdf or png or svg):
                tagged, pdf, png, svg = _c, _p, _g, _s
        meta = by_stem.get(stem, {})
        ok = pdf and png and svg
        reason = "" if ok else (skipped.get(stem) or "renderer did not run")
        if not ok:
            missing.append(stem)
        _dep_ok = True if (deployment is None or deployment.get("verdict_enforced") is False) \
            else bool(deployment.get("deployment_allowed", True))
        status = ("OK" if _dep_ok else "RENDERED_BUT_MODEL_UNQUALIFIED") if ok else "MISSING"
        rows.append({
            "figure": tagged, "group": meta.get("group", ""), "pdf": pdf, "png": png,
            "svg_visio": svg, "status": status,
            "reason_if_missing": reason, "claim_supported": meta.get("claim", ""),
            "checkpoint_feasible": None if deployment is None else deployment.get("checkpoint_feasible"),
            "test_global_nrmse": None if deployment is None else deployment.get("test_global_nrmse"),
            "test_late_nrmse": None if deployment is None else deployment.get("test_late_nrmse"),
            "test_late_cvar": None if deployment is None else deployment.get("test_late_cvar"),
            "deployment_allowed": None if deployment is None else deployment.get("deployment_allowed"),
        })
    df = pd.DataFrame(rows)
    (run_dir / "reports").mkdir(exist_ok=True)
    df.to_csv(run_dir / "reports" / tagged_name("figure_index.csv"), index=False, encoding="utf-8-sig")
    lines = ["FIGURE PACKAGE (%d expected) | PDF + 600-dpi PNG + Visio-editable SVG" % len(expected),
             "  %-46s %-22s %s" % ("figure", "group", "status")]
    for r in rows:
        lines.append("  %-46s %-22s %s%s" % (
            r["figure"], r["group"], r["status"],
            "" if r["status"] == "OK" else "  <- " + r["reason_if_missing"]))
    logger.info("\n".join(lines))
    if missing:
        logger.error(
            "%d FIGURE(S) MISSING: %s. Every one of them is listed above with the reason. This is an "
            "error, not a warning: a figure that silently fails to appear cannot be noticed in a "
            "hundred-line log.", len(missing), ", ".join(missing),
        )
    else:
        logger.info(
            "All %d figures written in all three formats. Index: reports/%s | Visio-editable SVG: "
            "figures/svg_visio/", len(expected), tagged_name("figure_index.csv"),
        )
    _lowres: List[str] = []
    _checked = 0
    try:
        from PIL import Image as _PILImage
    except Exception:
        _PILImage = None
        logger.warning(
            "Pillow is unavailable, so PNG resolution could NOT be verified "
            "on disk.")
    if _PILImage is not None:
        for _png in sorted(figure_dir.glob("*.png")):
            try:
                with _PILImage.open(_png) as _im:
                    _w, _h = _im.size
                    _dpi = float((_im.info.get("dpi") or (0.0, 0.0))[0])
            except Exception as _exc:
                _lowres.append("%s (unreadable: %s)" % (_png.name, _exc))
                continue
            _checked += 1
            if _dpi < float(FIGURE_PNG_DPI) - 1.0:
                _lowres.append("%s (%.0f dpi, %dx%d px)" % (_png.name, _dpi, _w, _h))
        if _lowres:
            logger.error(
                "%d of %d PNG figure(s) are BELOW the %d-dpi contract: %s. A renderer that draws "
                "outside plt.rc_context(nature_rc()) inherits Matplotlib's 100-dpi default; _save_figure now "
                "passes the dpi explicitly, so a file listed here means a NEW export path was added that "
                "bypasses it.",
                len(_lowres), _checked, int(FIGURE_PNG_DPI), "; ".join(_lowres))
        else:
            logger.info(
                "PNG resolution verified on disk: %d/%d figures at >= %d dpi.",
                _checked, _checked, int(FIGURE_PNG_DPI))
    return {"expected": len(expected), "missing": missing, "index": rows,
            "png_below_dpi": _lowres, "png_checked": _checked}


def render_test_profile_comparison(model: "PEBRNet", cfg: Config, run_dir: Path,
                                   clean_cache: CleanCacheInfo,
                                   noise_cache: NoiseCacheInfo,
                                   logger: logging.Logger,
                                   split: str = "test") -> None:
    """Up to four random held-out test work areas, each as the triptych the field figure cannot give: measured-noisy
    | clean reference | denoised, one curve per gate.
    """
    if not getattr(clean_cache, "paired_noisy_path", ""):
        logger.info("Skipped: not a paired run (no reference exists).")
        return
    try:
        import matplotlib
        if matplotlib.get_backend().lower() != "agg":
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import LogNorm, Normalize
    except Exception as exc:
        logger.warning("matplotlib unavailable: %s", exc)
        return
    splits = build_splits(clean_cache, cfg)
    pid = np.asarray(clean_cache.province_id)
    test_rows = np.asarray(splits[split if split in splits else "test"], dtype=np.int64)
    areas = np.unique(pid[test_rows])
    _fig_stem = "fig_%s_profile_comparison" % ("val" if split == "val" else "test")
    _extras = bool(getattr(cfg.runtime, "profile_extras", True))
    _unit = str(getattr(cfg.data, "paired_unit", "nT/s") or "nT/s")
    clean_mm = np.load(clean_cache.data_path, mmap_mode="r")
    noisy_mm = np.load(clean_cache.paired_noisy_path, mmap_mode="r")
    station_mm = np.load(clean_cache.station_path, mmap_mode="r")
    t = np.asarray(clean_cache.target_time, dtype=np.float64)
    names = list(clean_cache.province_names or [])
    ls = int(cfg.data.late_start_index)
    _nat = native_presentation_axis(cfg, t)
    _tp = _nat if _nat is not None else t
    _tpms = _tp * (1000.0 if float(np.max(_tp)) < 1.0 else 1.0)
    bg_mm = None
    try:
        if getattr(clean_cache, "paired_background_path", None):
            bg_mm = np.load(clean_cache.paired_background_path, mmap_mode="r")
    except Exception:
        bg_mm = None
    _pred_cache: Dict[int, np.ndarray] = {}
    if bg_mm is not None and areas.size >= 6:
        _pred_all = _paired_test_forward(model, cfg, clean_cache, noise_cache, test_rows, logger)
        _pos_all = {int(r): k for k, r in enumerate(test_rows)}
        _contrast = []
        _laterr = []
        _recov = []
        _quiet = []
        _nst = []
        _rawerr = []
        for _a in areas:
            _r = test_rows[pid[test_rows] == _a]
            _c = np.asarray(clean_mm[_r], dtype=np.float64)[:, ls:]
            _b = np.asarray(bg_mm[_r], dtype=np.float64)[:, ls:]
            _d = _pred_all[[_pos_all[int(x)] for x in _r]][:, ls:]
            _den = max(float(np.linalg.norm(_b)), 1e-300)
            _con = float(np.linalg.norm(_c - _b) / _den)
            _contrast.append(_con)
            _laterr.append(float(np.linalg.norm(_d - _c) / max(float(np.linalg.norm(_c)), 1e-300)))
            _nz = np.asarray(noisy_mm[_r], dtype=np.float64)[:, ls:]
            _rawerr.append(float(np.linalg.norm(_nz - _c) / max(float(np.linalg.norm(_c)), 1e-300)))
            _at = (_c - _b).ravel()
            _ah = (_d - _b).ravel()
            _recov.append(float(np.dot(_ah, _at) / max(float(np.dot(_at, _at)), 1e-300)) if _con > 1e-3 else float("nan"))
            _quiet.append(float(np.linalg.norm(_d - _b) / _den) if _con <= 2e-2 else float("nan"))
            _nst.append(int(_r.size))
            _pred_cache[int(_a)] = _pred_all[[_pos_all[int(x)] for x in _r]]
        _contrast = np.asarray(_contrast)
        _laterr = np.asarray(_laterr)
        _recov = np.asarray(_recov)
        _quiet = np.asarray(_quiet)
        try:
            _K = int(max(1, cfg.data.num_neighbors))
            _eE = _eR = _iE = _iR = 0.0
            for _a in areas:
                _r = test_rows[pid[test_rows] == _a]
                _c = np.asarray(clean_mm[_r], dtype=np.float64)[:, ls:]
                _d = np.asarray(_pred_cache[int(_a)], dtype=np.float64)[:, ls:]
                _n = int(_r.size)
                _edge = np.zeros(_n, dtype=bool)
                _edge[:min(_K, _n)] = True
                _edge[max(0, _n - _K):] = True
                _eE += float(np.sum((_d[_edge] - _c[_edge]) ** 2))
                _eR += float(np.sum(_c[_edge] ** 2))
                if (~_edge).any():
                    _iE += float(np.sum((_d[~_edge] - _c[~_edge]) ** 2))
                    _iR += float(np.sum(_c[~_edge] ** 2))
            _le = 100.0 * math.sqrt(_eE / max(_eR, 1e-300))
            _li = 100.0 * math.sqrt(_iE / max(_iR, 1e-300))
            logger.info("PROFILE-EDGE AUDIT (%s, %d areas) | late NRMSE on edge stations "
                        "(<= K=%d from either end) = %.2f%% | interior = %.2f%% | ratio %.2f (>1.5 = the "
                        "boundary mechanism -- one-sided neighbour window / free lateral boundary -- is "
                        "the defect).",
                        split, int(areas.size), _K, _le, _li, _le / max(_li, 1e-9))
            try:
                _dE = {}
                _dR = {}
                for _a in areas:
                    _r = test_rows[pid[test_rows] == _a]
                    _c = np.asarray(clean_mm[_r], dtype=np.float64)[:, ls:]
                    _d = np.asarray(_pred_cache[int(_a)], dtype=np.float64)[:, ls:]
                    _n = int(_r.size)
                    for _i in range(_n):
                        _q = min(_i, _n - 1 - _i)
                        _key = ("d=%d" % _q) if _q < 4 else ("d=4-7" if _q < _K else "interior")
                        _dE[_key] = _dE.get(_key, 0.0) + float(np.sum((_d[_i] - _c[_i]) ** 2))
                        _dR[_key] = _dR.get(_key, 0.0) + float(np.sum(_c[_i] ** 2))
                _order = ["d=0", "d=1", "d=2", "d=3", "d=4-7", "interior"]
                logger.info("EDGE PROFILE (%s) | edge mode in force: %s | late NRMSE by distance from the "
                            "area edge: %s.",
                            split, resolve_edge_mode(cfg),
                            " | ".join("%s %.2f%%" % (_k, 100.0 * math.sqrt(_dE[_k] / max(_dR[_k], 1e-300)))
                                       for _k in _order if _k in _dE))
            except Exception as _e:
                logger.warning("edge profile skipped: %r", _e)
        except Exception as _e:
            logger.warning("edge audit skipped: %r", _e)
        try:
            _rep_dir = run_dir / "reports"
            _rep_dir.mkdir(parents=True, exist_ok=True)
            with open(_rep_dir / ("%s_sections_metrics.csv" % split), "w", newline="", encoding="utf-8") as _fh:
                _w = csv.writer(_fh)
                _w.writerow(["area_id", "area_name", "n_stations", "late_error_percent", "raw_late_error_percent",
                             "body_contrast", "recovery", "quiet_false_level_percent"])
                for _k, _a in enumerate(areas):
                    _w.writerow([int(_a), names[int(_a)] if int(_a) < len(names) else "", _nst[_k],
                                 "%.4f" % (100 * _laterr[_k]), "%.4f" % (100 * _rawerr[_k]), "%.5f" % _contrast[_k],
                                 "" if not np.isfinite(_recov[_k]) else "%.4f" % _recov[_k],
                                 "" if not np.isfinite(_quiet[_k]) else "%.4f" % (100 * _quiet[_k])])
        except Exception as _e:
            logger.warning("metrics export skipped: %r", _e)
        try:
            _le2 = 100.0 * np.asarray(_laterr, dtype=np.float64)
            _re = 100.0 * np.asarray(_rawerr, dtype=np.float64)
            _ok = np.isfinite(_le2) & np.isfinite(_re)
            if bool(_ok.any()):
                _le2, _re = _le2[_ok], _re[_ok]

                def _pc(v: np.ndarray) -> Dict[str, float]:
                    return {"p25": float(np.percentile(v, 25)), "median": float(np.median(v)),
                            "p75": float(np.percentile(v, 75))}
                _sum = {"split": split, "profiles": int(_le2.size),
                           "late_nrmse_percent_reconstruction": _pc(_le2),
                           "late_nrmse_percent_raw": _pc(_re),
                           "profiles_below_the_best_raw_profile": int(np.sum(_le2 < float(np.min(_re)))),
                           "profiles_better_than_their_raw_record": int(np.sum(_le2 < _re))}
                atomic_json_dump(_sum, run_dir / "reports" / ("%s_profile_summary.json" % split))
                logger.info(
                    "PER-PROFILE LATE NRMSE (Eq. 7, %s, %d profiles) | reconstruction median %.2f%% "
                    "(IQR %.2f-%.2f%%) | raw record median %.2f%% (IQR %.2f-%.2f%%) | below the best raw profile: %d "
                    "| better than their own raw record: %d",
                    split, int(_le2.size), _sum["late_nrmse_percent_reconstruction"]["median"],
                    _sum["late_nrmse_percent_reconstruction"]["p25"], _sum["late_nrmse_percent_reconstruction"]["p75"],
                    _sum["late_nrmse_percent_raw"]["median"], _sum["late_nrmse_percent_raw"]["p25"],
                    _sum["late_nrmse_percent_raw"]["p75"], _sum["profiles_below_the_best_raw_profile"],
                    _sum["profiles_better_than_their_raw_record"])
        except Exception as _e:
            logger.warning("per-profile summary skipped: %r", _e)
        if _extras:
            try:
                with plt.rc_context(nature_rc()):
                    _fq, _axq = plt.subplots(2, 2, figsize=_mm(NATURE_DOUBLE_MM, 120.0), layout="constrained")
                    _cx = np.maximum(_contrast, 2e-4)
                    _axq[0, 0].scatter(_cx, 100 * _laterr, s=6, alpha=0.5, color="#0072B2")
                    _axq[0, 0].set_xscale("log")
                    _axq[0, 0].set_xlabel("late-window body contrast")
                    _axq[0, 0].set_ylabel("late error (%)")
                    _axq[0, 0].axhline(100 * float(cfg.contract.late_threshold), color="0.4", lw=0.6, ls="--")
                    _axq[0, 0].set_title("(a) late error vs body contrast", fontsize=7)
                    _m = np.isfinite(_recov)
                    _axq[0, 1].scatter(_cx[_m], _recov[_m], s=6, alpha=0.5, color="#D55E00")
                    _axq[0, 1].set_xscale("log")
                    _axq[0, 1].axhline(0.8, color="0.4", lw=0.6, ls="--")
                    _axq[0, 1].axhline(1.0, color="0.7", lw=0.5)
                    _axq[0, 1].set_ylim(-1.0, 2.0)
                    _axq[0, 1].set_xlabel("late-window body contrast")
                    _axq[0, 1].set_ylabel("recovery <d-b, c-b>/<c-b, c-b>")
                    _axq[0, 1].set_title("(b) body recovery vs contrast (0.8 line)", fontsize=7)
                    for _axL in (_axq[0, 0], _axq[0, 1]):
                        _axL.xaxis.set_major_locator(
                            __import__("matplotlib").ticker.LogLocator(numticks=6))
                        _axL.xaxis.set_major_formatter(
                            __import__("matplotlib").ticker.LogFormatterMathtext())
                        _axL.tick_params(axis="x", labelsize=5.5)
                    _q = 100 * _quiet[np.isfinite(_quiet)]
                    if _q.size:
                        _axq[1, 0].hist(_q, bins=30, color="#009E73")
                    _axq[1, 0].set_xlabel("quiet-area false level (% of background, late)")
                    _axq[1, 0].set_ylabel("areas")
                    _axq[1, 0].set_title("(c) false-anomaly level on quiet areas (n=%d)" % _q.size, fontsize=7)
                    _edges = [0.0, 0.01, 0.02, 0.05, 0.20, np.inf]
                    _lab = ["<1%", "1-2%", "2-5%", "5-20%", ">=20%"]
                    _means = []
                    _cnts = []
                    for _lo, _hi in zip(_edges[:-1], _edges[1:]):
                        _mm_ = _m & (_contrast >= _lo) & (_contrast < _hi)
                        _means.append(float(np.nanmean(_recov[_mm_])) if _mm_.any() else np.nan)
                        _cnts.append(int(_mm_.sum()))
                    _axq[1, 1].bar(range(5),
                                   [0 if not np.isfinite(v) else v for v in _means],
                                   width=0.62, color="#CC79A7",
                                   edgecolor="black", linewidth=0.4)
                    _axq[1, 1].set_ylim(0.0, 1.3)
                    _axq[1, 1].set_yticks([0.0, 0.4, 0.8, 1.2])
                    _axq[1, 1].set_xticks(range(5))
                    _axq[1, 1].set_xticklabels(["%s\nn=%d" % (lab_v, c) for lab_v, c in zip(_lab, _cnts)], fontsize=5.5)
                    _axq[1, 1].axhline(0.8, color="0.4", lw=0.6, ls="--")
                    _axq[1, 1].set_ylabel("mean recovery")
                    _axq[1, 1].set_title("(d) recovery by contrast stratum", fontsize=7)
                    _save_figure(_fq, run_dir / "figures", "fig_%s_weak_anomaly_quantification" % split)
                logger.info("%s weak-anomaly quantification | areas %d | recovery by stratum %s | quiet false level "
                            "median %.2f%% (n=%d).", split, areas.size,
                            ", ".join("%s: %s (n=%d)" % (lab_v, "n/a" if not np.isfinite(v) else "%.2f" % v, c) for lab_v, v, c in zip(_lab, _means, _cnts)),
                            float(np.median(_q)) if _q.size else float("nan"), int(_q.size))
            except Exception as _e528b:
                logger.warning("quantification figure skipped: %r", _e528b)
        _by_err = np.argsort(_laterr)
        for _i2 in list(_by_err[::-1][:(int(max(0, getattr(cfg.runtime, "section_autopsy_count", 3))) if _extras else 0)]):
            try:
                _a2 = areas[int(_i2)]
                worst_section_autopsy(model, cfg, run_dir, clean_cache, noise_cache,
                                          test_rows[pid[test_rows] == _a2], split,
                                          names[int(_a2)] if int(_a2) < len(names) else str(int(_a2)), logger)
            except Exception as _e:
                logger.warning("section autopsy skipped: %r", _e)
        _nonzero = [i for i in np.argsort(_contrast) if _contrast[i] > 1e-3]
        _sel = representative_profiles(_laterr)
        pick = np.asarray([areas[i] for i in _sel])
        pick_contrast = [float(_contrast[i]) for i in _sel]
        pick_role = list(PROFILE_ROLES[:len(_sel)])
        _extra_figs = [
            (np.asarray([areas[i] for i in _by_err[:5]]), [float(_contrast[i]) for i in _by_err[:5]],
             ["best #%d" % (k + 1) for k in range(min(5, areas.size))], "fig_%s_profile_best5" % split),
            (np.asarray([areas[i] for i in _by_err[::-1][:5]]), [float(_contrast[i]) for i in _by_err[::-1][:5]],
             ["worst #%d" % (k + 1) for k in range(min(5, areas.size))], "fig_%s_profile_worst5" % split),
        ]
        if len(_nonzero) >= 2:
            _bi = list(_nonzero[::-1][:3]) + [i for i in _nonzero[:3] if i not in _nonzero[::-1][:3]]
            _extra_figs.append((np.asarray([areas[i] for i in _bi]), [float(_contrast[i]) for i in _bi],
                                ["strongest #%d" % (k + 1) for k in range(min(3, len(_nonzero)))]
                                + ["weakest #%d" % (k + 1) for k in range(len(_bi) - min(3, len(_nonzero)))],
                                "fig_%s_profile_bodies" % split))
        if not _extras:
            _extra_figs = []
        logger.info("PROFILE CONTRACT %s | areas with late error <= 4%%: %.1f%% of %d | <= 3%%: %.1f%% | worst-decile mean %.2f%%. ", split, 100.0 * float(np.mean(np.asarray(_laterr) <= 0.04)), int(areas.size),
                    100.0 * float(np.mean(np.asarray(_laterr) <= 0.03)),
                    100.0 * float(np.mean(np.sort(np.asarray(_laterr))[::-1][:max(1, int(round(0.1 * len(_laterr))))])))
        logger.info("%s sections | late error over %d areas: median %.2f%% p10 %.2f%% p90 %.2f%% "
                    "| shown: %s.", split, areas.size, 100 * float(np.median(_laterr)),
                    100 * float(np.percentile(_laterr, 10)), 100 * float(np.percentile(_laterr, 90)),
                    ", ".join("%s=%s (late %.2f%%)" % (pick_role[k], names[int(areas[i])] if int(areas[i]) < len(names) else int(areas[i]), 100 * _laterr[i]) for k, i in enumerate(_sel)))
    else:
        _pred_all = _paired_test_forward(model, cfg, clean_cache, noise_cache, test_rows, logger)
        _pos_all = {int(r): k for k, r in enumerate(test_rows)}
        _laterr = []
        for _a in areas:
            _r = test_rows[pid[test_rows] == _a]
            _c = np.asarray(clean_mm[_r], dtype=np.float64)[:, ls:]
            _d = _pred_all[[_pos_all[int(x)] for x in _r]]
            _laterr.append(float(np.linalg.norm(_d[:, ls:] - _c) / max(float(np.linalg.norm(_c)), 1e-300)))
            _pred_cache[int(_a)] = _d
        _sel = representative_profiles(_laterr)
        pick = np.asarray([areas[i] for i in _sel])
        pick_contrast = [float("nan")] * len(pick)
        pick_role = list(PROFILE_ROLES[:len(_sel)])
        _extra_figs = []

    _shown_npz: Dict[str, np.ndarray] = {}
    _main = _fig_stem
    _lab2 = "Validation" if split == "val" else "Test"
    for pick, pick_contrast, pick_role, _fig_stem in ([(pick, pick_contrast, pick_role, _fig_stem)] + _extra_figs):
      n = len(pick)
      ncol = 4 if (bg_mm is not None and _fig_stem != _main) else 3
      with plt.rc_context(nature_rc()):
          fig, axes = plt.subplots(n, ncol, figsize=_mm(NATURE_DOUBLE_MM, 48.0 * n),
                                   sharex="row", sharey=False, layout="constrained")
          axes = np.atleast_2d(axes)
          cmap = plt.get_cmap("turbo")
          tnorm = (LogNorm(vmin=float(_tpms.min()), vmax=float(_tpms.max()))
                   if float(_tpms.min()) > 0 else Normalize(float(_tpms.min()), float(_tpms.max())))
          _lsp = native_late_start(ls, t, _nat)
          for r, area in enumerate(pick):
              rows = test_rows[pid[test_rows] == area]
              _perm = np.argsort(np.asarray(station_mm[rows]), kind="stable")
              rows = rows[_perm]
              st = np.asarray(station_mm[rows], dtype=np.float64)
              cl = np.asarray(clean_mm[rows], dtype=np.float64)
              nz = np.asarray(noisy_mm[rows], dtype=np.float64)
              dn = (_pred_cache[int(area)][_perm] if int(area) in _pred_cache
                    else _paired_test_forward(model, cfg, clean_cache, noise_cache, rows, logger))
              viol = 0.0
              _viol_measured = False
              if bool(getattr(cfg.data, "gate_monotonic_projection", True)):
                  dn, viol = project_gate_monotonic(dn)
                  _viol_measured = True
              else:
                  try:
                      _, viol = project_gate_monotonic(np.array(dn, copy=True))
                      _viol_measured = True
                  except Exception:
                      pass
              rel_late = 100.0 * float(np.linalg.norm(dn[:, ls:] - cl[:, ls:])
                                       / max(np.linalg.norm(cl[:, ls:]), 1e-300))
              _key = "%s__%s__area%d" % (_fig_stem, pick_role[r].replace(" ", "_").replace("#", ""), int(area))
              _shown_npz[_key + "__station"] = st
              _shown_npz[_key + "__clean"] = cl
              _shown_npz[_key + "__noisy"] = nz
              _shown_npz[_key + "__denoised"] = dn
              try:
                  _dig = hashlib.sha1(np.ascontiguousarray(nz.astype(np.float64)).tobytes()).hexdigest()[:10]
                  _nzl = nz[:, ls:] - cl[:, ls:]
                  _si = np.arange(nz.shape[0], dtype=np.float64)
                  _si = (_si - _si.mean()) / max(_si.std(), 1e-9)
                  _nc = _nzl - _nzl.mean(axis=0, keepdims=True)
                  _r2 = np.abs((_nc * _si[:, None]).mean(axis=0) / np.maximum(_nc.std(axis=0), 1e-30))
                  _h = nz.shape[0] // 2
                  _amp2 = (np.sqrt(np.mean(_nzl[_h:] ** 2)) / max(np.sqrt(np.mean(_nzl[:_h] ** 2)), 1e-30))
                  logger.info("%s area %d | noisy sha1 %s | late-noise drift |r| median %.2f | right/left late-noise amplitude %.2fx",
                              split, int(area), _dig, float(np.median(_r2)), float(_amp2))
              except Exception:
                  pass
              if bg_mm is not None:
                  _shown_npz[_key + "__background"] = np.asarray(bg_mm[rows], dtype=np.float64)
              _pid = _area_sample_id(names[area] if area < len(names) else str(int(area)))
              _lc = log10_amplitude(to_presentation_axis(cl, t, _nat))
              _ln = log10_amplitude(to_presentation_axis(nz, t, _nat))
              _ylo = float(np.nanmin(_lc)) - 0.35
              _yhi = max(float(np.nanmax(_lc)), float(np.nanpercentile(_ln, 99.5))) + 0.25
              for c, (mat, title) in enumerate(
                      ((nz, "%s profile %s: noisy record" % (_lab2, _pid)), (cl, "clean reference"),
                       (dn, "reconstruction, late NRMSE %.2f%%" % rel_late))):
                  ax = axes[r, c]
                  _lg = log10_amplitude(to_presentation_axis(mat, t, _nat))
                  for g in range(_lg.shape[1]):
                      ax.plot(st, _lg[:, g], color=cmap(tnorm(_tpms[g])), lw=0.6)
                  ax.set_ylim(_ylo, _yhi)
                  ax.set_title("(%s) %s" % ("abcdefghijklmnopqrstuvwxyz"[(r * ncol + c) % 26], title), loc="left",
                               fontsize=6.5)
                  if c == 0:
                      ax.set_ylabel("Amplitude (log10(%s))" % _unit, fontsize=6.5)
                  else:
                      ax.tick_params(labelleft=False)
                  if r == n - 1:
                      ax.set_xlabel("Depth (m)", fontsize=6.5)
                  ax.tick_params(labelsize=5.5)
              if _fig_stem != _main:
                  axes[r, 2].text(
                      0.02, 0.03,
                      "%s | order-viol %s" % (
                          pick_role[r], ("%.2f%%" % viol) if _viol_measured else "not measured (projection off)"),
                      transform=axes[r, 2].transAxes, fontsize=5.5, va="bottom")
              if ncol == 4:
                  axb = axes[r, 3]
                  bgm = np.asarray(bg_mm[rows], dtype=np.float64)
                  _amp = max(float(np.max(np.abs(cl[:, ls:] - bgm[:, ls:]))),
                             0.05 * float(np.sqrt(np.mean(bgm[:, ls:] ** 2))), 1e-300)
                  _clp, _dnp, _bgp = (to_presentation_axis(cl, t, _nat), to_presentation_axis(dn, t, _nat),
                                               to_presentation_axis(bgm, t, _nat))
                  for g in range(_lsp, _clp.shape[1]):
                      axb.plot(st, (_clp[:, g] - _bgp[:, g]) / _amp, color=cmap(tnorm(_tpms[g])), lw=0.6)
                      axb.plot(st, (_dnp[:, g] - _bgp[:, g]) / _amp, color=cmap(tnorm(_tpms[g])), lw=0.6,
                               ls="--")
                  axb.axhline(0.0, color="0.5", lw=0.4)
                  axb.set_ylim(-1.6, 1.6)
                  _rec = float(np.dot((dn[:, ls:] - bgm[:, ls:]).ravel(), (cl[:, ls:] - bgm[:, ls:]).ravel())
                               / max(float(np.sum((cl[:, ls:] - bgm[:, ls:]) ** 2)), 1e-300))
                  axb.text(0.02, 0.03, "body contrast %.3f | recovery %.2f" % (pick_contrast[r], _rec),
                           transform=axb.transAxes, fontsize=5.5, va="bottom")
                  if r == 0:
                      axb.set_title("Late-window anomaly response\n(solid ref, dashed denoised)",
                                    fontsize=6.5)
                  if r == n - 1:
                      axb.set_xlabel("Depth (m)", fontsize=6.5)
                  axb.set_ylabel("normalised", fontsize=6)
                  axb.tick_params(labelsize=5.5)
          sm = ScalarMappable(norm=tnorm, cmap=cmap)
          fig.colorbar(sm, ax=axes[:, -1], label="Gate time (ms)", shrink=0.85)
          for _ax in np.asarray(axes).ravel():
              for _t in list(getattr(_ax, "texts", [])):
                  try:
                      _t.set_in_layout(False)
                  except Exception:
                      pass
          _save_figure(fig, run_dir / "figures", _fig_stem)
    try:
        _rep_dir = run_dir / "reports"
        _rep_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(_rep_dir / ("%s_sections_shown.npz" % split), **_shown_npz)
    except Exception as _e528c:
        logger.warning("section export skipped: %r", _e528c)
    logger.info(
        "%s profile sections rendered for %d work area(s): %s.",
        split, n, [names[a] if a < len(names) else int(a) for a in pick])
