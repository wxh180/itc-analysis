#!/usr/bin/env python3
"""
itc_analysis.py - process and fit MicroCal iTC200 (.itc) titrations.

What it does
  1. Reads a MicroCal .itc file (iTC200 / VP-ITC text format): run settings,
     injection schedule and the raw thermogram (DP in ucal/s vs time).
  2. Integrates each injection peak against a linear local baseline
     (median of the last few seconds before this injection and before the next).
  3. Optionally subtracts a control titration (ligand into buffer), injection by injection.
  4. Converts heats to kcal per mol of injectant and fits a one-set-of-sites model
     (the MicroCal "OneSites" model, with the standard displaced-volume correction)
     for N, Ka, dH (+ an optional constant heat-of-dilution offset).
  5. Writes a figure (thermograms, binding isotherm with fit, residuals), a CSV of
     integrated heats and a text report with Kd, dG, dH, -TdS.

Usage
  python itc_analysis.py SAMPLE.itc [--blank BUFFER.itc] \
      --syringe-mM 10 --cell-uM 100 [--fix-n 1] [--drop 1] [--out results/AKR1A1_CoA]

  Concentrations: the .itc header stores whatever was typed into the instrument
  software, which is often a default. The values found are printed; pass
  --syringe-mM / --cell-uM to override (recommended).

Sign convention: MicroCal DP drops for an exothermic injection, so integrated heats
and dH are negative for exothermic binding.

Requires: numpy, scipy, matplotlib  (pip install numpy scipy matplotlib)
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

R_KCAL = 1.98720425864083e-3  # kcal / (mol K)


# --------------------------------------------------------------------------- parsing
@dataclass
class Injection:
    index: int
    volume_uL: float
    duration_s: float
    spacing_s: float
    filter_s: float
    start_s: float = math.nan  # time of the @n marker in the data


@dataclass
class ItcRun:
    path: Path
    n_injections: int
    temperature_C: float
    initial_delay_s: float
    stir_rpm: float
    ref_power: float
    syringe_mM: float | None
    cell_mM: float | None
    cell_volume_mL: float
    injections: list[Injection]
    time: np.ndarray
    dp: np.ndarray            # ucal/s
    cell_temp: np.ndarray
    header_hash: list[float] = field(default_factory=list)
    instrument: str = ""


def _nums(line: str) -> list[float]:
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)]


def read_itc(path: str | Path) -> ItcRun:
    """Parse a MicroCal .itc file.

    Layout (iTC200):
      $ITC / $ n_inj / $NOT / $ temp / $ initial delay / $ stir / $ ref power / ...
      $ vol, duration, spacing, filter     (one line per injection)
      # ...                                 (experiment values: see below)
      % ...                                 (instrument calibration block)
      @0 / data rows / @1,vol,dur,time / data rows / ...
      data row: time, DP, cell T, ..., jacket T, ...
    The '#' block on iTC200 files is: #0, #syringe mM, #cell mM, #cell volume mL,
    #temperature, ... These are printed so you can check them.
    """
    path = Path(path)
    lines = path.read_text(errors="replace").splitlines()
    dollars, hashes, injections = [], [], []
    rows, markers = [], {}
    instrument = ""
    current = None
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("$"):
            dollars.append(s)
            v = _nums(s)
            if len(v) == 4 and "," in s:
                injections.append(Injection(len(injections) + 1, *v))
        elif s.startswith("#"):
            v = _nums(s)
            hashes.append(v[0] if v else math.nan)
        elif s.startswith("%"):
            if "Ver" in s or "ITC" in s:
                instrument = instrument or s.lstrip("% ").strip()
        elif s.startswith("@"):
            v = _nums(s)
            current = int(v[0])
            markers[current] = v[3] if len(v) >= 4 else None
        elif current is not None and s[0] in "-+.0123456789":
            v = _nums(s)
            if len(v) >= 3:
                rows.append(v)

    def dollar_val(i, default=math.nan):
        try:
            return _nums(dollars[i])[0]
        except (IndexError, ValueError):
            return default

    n_inj = int(dollar_val(1, len(injections)))
    data = np.array([r[:3] for r in rows], dtype=float)
    time, dp, cell_t = data[:, 0], data[:, 1], data[:, 2]
    for inj in injections:
        t = markers.get(inj.index)
        if t is None:  # fall back to the schedule
            t = dollar_val(4, 0) + sum(i.spacing_s for i in injections[: inj.index - 1])
        inj.start_s = float(t)

    syringe = hashes[1] if len(hashes) > 1 else None
    cell = hashes[2] if len(hashes) > 2 else None
    cell_vol = hashes[3] if len(hashes) > 3 and 0.1 < hashes[3] < 2 else 0.2029
    return ItcRun(
        path=path,
        n_injections=n_inj,
        temperature_C=dollar_val(3),
        initial_delay_s=dollar_val(4),
        stir_rpm=dollar_val(5),
        ref_power=dollar_val(6),
        syringe_mM=syringe,
        cell_mM=cell,
        cell_volume_mL=cell_vol,
        injections=injections[:n_inj],
        time=time,
        dp=dp,
        cell_temp=cell_t,
        header_hash=hashes,
        instrument=instrument,
    )


# --------------------------------------------------------------------------- integration
@dataclass
class PeakResult:
    heat_ucal: np.ndarray          # integrated heat per injection (ucal)
    baselines: list[tuple[np.ndarray, np.ndarray]]  # (t, baseline) per injection for plotting


def integrate_peaks(run: ItcRun, base_window_s: float = 20.0, skip_s: float = 0.0) -> PeakResult:
    """Integrate DP - baseline over each injection interval.

    Baseline = straight line between the median DP in the `base_window_s` seconds
    before this injection and the median DP in the `base_window_s` seconds before the
    next injection (or the end of the run).
    """
    t, dp = run.time, run.dp
    heats, bases = [], []
    starts = [inj.start_s for inj in run.injections]
    for k, t0 in enumerate(starts):
        t1 = starts[k + 1] if k + 1 < len(starts) else t[-1] + 1e-9
        pre = (t > t0 - base_window_s) & (t <= t0)
        post = (t > t1 - base_window_s) & (t <= t1)
        b0 = np.median(dp[pre]) if pre.any() else dp[np.searchsorted(t, t0) - 1]
        b1 = np.median(dp[post]) if post.any() else b0
        tm0 = t0 - base_window_s / 2
        tm1 = t1 - base_window_s / 2
        seg = (t >= t0 + skip_s) & (t < t1)
        ts = t[seg]
        base = b0 + (b1 - b0) * (ts - tm0) / max(tm1 - tm0, 1e-9)
        heats.append(np.trapezoid(dp[seg] - base, ts) if hasattr(np, "trapezoid") else np.trapz(dp[seg] - base, ts))
        bases.append((ts, base))
    return PeakResult(np.array(heats), bases)


# --------------------------------------------------------------------------- model
def cell_concentrations(vols_uL: np.ndarray, V0_mL: float, M0: float, Xsyr: float):
    """Total macromolecule (Mt) and ligand (Xt) in the cell after each injection,
    in the same units as M0/Xsyr, using the MicroCal overflow (displaced volume) model."""
    V0 = V0_mL * 1000.0  # uL
    v = np.cumsum(vols_uL)
    f = v / (2 * V0)
    Mt = M0 * (1 - f) / (1 + f)
    Xt = Xsyr * (v / V0) * (1 - f)
    return Mt, Xt


def onesite_heats(params, vols_uL, V0_mL, M0_M, Xsyr_M):
    """Heat per injection (ucal) for n identical independent sites.
    params: n, logKa (log10 M^-1), dH (kcal/mol), offset (kcal/mol injectant)."""
    n, logK, dH, off = params
    K = 10.0 ** logK
    Mt, Xt = cell_concentrations(vols_uL, V0_mL, M0_M, Xsyr_M)
    V0_L = V0_mL / 1000.0
    nM = n * Mt
    b = 1 + Xt / nM + 1 / (K * nM)
    Q = (nM * dH * 1e3 * V0_L / 2) * (b - np.sqrt(np.maximum(b * b - 4 * Xt / nM, 0)))  # cal
    Qprev = np.concatenate([[0.0], Q[:-1]])
    dQ = Q + (vols_uL / (V0_mL * 1000.0)) * (Q + Qprev) / 2 - Qprev                       # cal
    dQ_ucal = dQ * 1e6
    inj_nmol = vols_uL * Xsyr_M * 1e3  # uL * M -> umol ; *1e3 -> nmol
    return dQ_ucal + off * inj_nmol    # kcal/mol * nmol = ucal


def fit_onesite(heats_ucal, vols_uL, V0_mL, M0_M, Xsyr_M, use, fix_n=None, fit_offset=True,
                guess=None):
    from scipy.optimize import least_squares

    names = ["n", "logKa", "dH", "offset"]
    inj_nmol = vols_uL * Xsyr_M * 1e3
    norm = heats_ucal / inj_nmol  # kcal/mol injectant
    if guess is None:
        dH0 = norm[use][0] if use.any() else -5.0
        guess = [1.0, 4.0, float(dH0) if abs(dH0) > 0.1 else -5.0, float(norm[use][-1]) if use.any() else 0.0]
    free = [True, True, True, fit_offset]
    if fix_n is not None:
        guess[0], free[0] = fix_n, False
    if not fit_offset:
        guess[3] = 0.0
    p0 = np.array(guess, float)
    idx = [i for i, f in enumerate(free) if f]
    lo = np.array([0.01, 0.0, -200.0, -50.0])[idx]
    hi = np.array([50.0, 12.0, 200.0, 50.0])[idx]

    def full(x):
        p = p0.copy()
        p[idx] = x
        return p

    def resid(x):
        model = onesite_heats(full(x), vols_uL, V0_mL, M0_M, Xsyr_M) / inj_nmol
        return (model - norm)[use]

    best = None
    for logK_start in (2.5, 3.5, 4.5, 5.5, 6.5, 7.5):   # multistart over affinity
        x0 = p0[idx].copy()
        if 1 in idx:
            x0[idx.index(1)] = logK_start
        x0 = np.clip(x0, lo + 1e-6, hi - 1e-6)
        r = least_squares(resid, x0, bounds=(lo, hi), x_scale="jac")
        if best is None or r.cost < best.cost:
            best = r
    p = full(best.x)
    dof = max(int(use.sum()) - len(idx), 1)
    s2 = 2 * best.cost / dof
    try:
        J = best.jac
        cov = np.linalg.pinv(J.T @ J) * s2
        err_free = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        err_free = np.full(len(idx), np.nan)
    err = np.zeros(4)
    err[idx] = err_free
    return dict(zip(names, p)), dict(zip(names, err)), math.sqrt(s2), free


def flag_failed(q_ucal: np.ndarray, vols_uL: np.ndarray, frac: float = 0.3) -> list[int]:
    """Injections whose heat per uL is < frac of the median of their neighbours (+-2),
    a sign of a bubble or a missed injection. The first injection is left to --drop."""
    per = np.abs(q_ucal) / np.where(vols_uL > 0, vols_uL, 1)
    out = []
    for i in range(1, len(per)):
        nb = np.delete(per[max(0, i - 2):i + 3], min(i, 2))
        if len(nb) and per[i] < frac * np.median(nb):
            out.append(i)
    return out


# --------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sample", help="sample titration .itc (ligand into protein)")
    ap.add_argument("--blank", help="control titration .itc (ligand into buffer), subtracted per injection")
    ap.add_argument("--syringe-mM", type=float, help="ligand concentration in the syringe (mM); default = header value")
    ap.add_argument("--cell-uM", type=float, help="macromolecule concentration in the cell (uM); default = header value")
    ap.add_argument("--cell-volume-mL", type=float, help="cell volume (mL); default = header (iTC200: 0.2029)")
    ap.add_argument("--drop", type=int, nargs="*", default=[1],
                    help="injection numbers to exclude from the fit (default: 1, the small first injection)")
    ap.add_argument("--keep-all", action="store_true",
                    help="don't auto-exclude injections that look failed (heat < 30%% of their neighbours)")
    ap.add_argument("--fix-n", type=float, help="fix the stoichiometry N (use for weak binding, c < 1)")
    ap.add_argument("--no-offset", action="store_true", help="don't fit a constant heat-of-dilution offset")
    ap.add_argument("--base-window", type=float, default=20.0, help="seconds averaged for each baseline point (default 20)")
    ap.add_argument("--out", help="output prefix (default: next to the sample file)")
    a = ap.parse_args(argv)

    run = read_itc(a.sample)
    print(f"Sample : {run.path.name}  ({run.instrument or 'MicroCal'})")
    print(f"  {len(run.injections)} injections, {run.temperature_C:g} °C, stir {run.stir_rpm:g} rpm, "
          f"ref power {run.ref_power:g}, cell volume {run.cell_volume_mL:g} mL")
    print(f"  header concentrations: syringe {run.syringe_mM} mM, cell {run.cell_mM} mM  "
          f"(raw '#' values: {run.header_hash})")
    syr_mM = a.syringe_mM if a.syringe_mM is not None else run.syringe_mM
    cell_uM = a.cell_uM if a.cell_uM is not None else (run.cell_mM * 1000 if run.cell_mM else None)
    V0 = a.cell_volume_mL or run.cell_volume_mL
    if not syr_mM or not cell_uM:
        sys.exit("Need concentrations: pass --syringe-mM and --cell-uM")
    if a.syringe_mM is None or a.cell_uM is None:
        print("  ! using header concentration(s) - check they are what was actually loaded")
    print(f"  using: syringe {syr_mM:g} mM, cell {cell_uM:g} uM, V0 {V0:g} mL")

    vols = np.array([i.volume_uL for i in run.injections])
    pk = integrate_peaks(run, a.base_window)
    q_sample = pk.heat_ucal

    q_blank = np.zeros_like(q_sample)
    blank = pk_b = None
    if a.blank:
        blank = read_itc(a.blank)
        print(f"Blank  : {blank.path.name}  ({len(blank.injections)} injections)")
        bvols = np.array([i.volume_uL for i in blank.injections])
        pk_b = integrate_peaks(blank, a.base_window)
        n = min(len(bvols), len(vols))
        if not np.allclose(bvols[:n], vols[:n]):
            print("  ! blank injection volumes differ from the sample; scaling blank heats by volume")
        q_blank[:n] = pk_b.heat_ucal[:n] * vols[:n] / np.where(bvols[:n] > 0, bvols[:n], 1)
        if n < len(vols):
            q_blank[n:] = np.mean(q_blank[max(0, n - 3):n])
    q_corr = q_sample - q_blank

    Xsyr_M = syr_mM * 1e-3
    M0_M = cell_uM * 1e-6
    inj_nmol = vols * syr_mM  # uL * mM = nmol
    kcal = q_corr / inj_nmol
    Mt, Xt = cell_concentrations(vols, V0, M0_M, Xsyr_M)
    ratio = Xt / Mt

    use = np.ones(len(vols), bool)
    for d in a.drop or []:
        if 1 <= d <= len(vols):
            use[d - 1] = False
    failed = [] if a.keep_all else flag_failed(q_sample, vols)
    for i in failed:
        use[i] = False
        nb = np.median(np.abs(np.delete(q_sample, i)[max(0, i - 2):i + 2]))
        print(f"  ! injection {i + 1} looks failed: {q_sample[i]:.3g} ucal vs ~{nb:.3g} for its neighbours "
              f"(bubble or missed injection) - excluded. Later concentrations assume it was delivered.")

    p, e, rmsd, free = fit_onesite(q_corr, vols, V0, M0_M, Xsyr_M, use,
                                   fix_n=a.fix_n, fit_offset=not a.no_offset)
    # Is there a binding isotherm at all? Compare with a flat line (heat of dilution only).
    k1 = sum(free)
    rss1 = (rmsd ** 2) * max(int(use.sum()) - k1, 1)
    flat = float(np.mean(kcal[use])) if not a.no_offset else 0.0
    rss0 = float(np.sum((kcal[use] - flat) ** 2))
    k0 = 0 if a.no_offset else 1
    dof1 = max(int(use.sum()) - k1, 1)
    F = ((rss0 - rss1) / max(k1 - k0, 1)) / (rss1 / dof1) if rss1 > 0 else math.inf
    from scipy.stats import f as fdist
    p_bind = float(fdist.sf(F, max(k1 - k0, 1), dof1)) if math.isfinite(F) else 0.0
    at_bound = p["logKa"] <= 0.01 or p["logKa"] >= 11.99 or p["n"] >= 49.9
    binding = p_bind < 0.05 and not at_bound
    T = run.temperature_C + 273.15
    Ka = 10 ** p["logKa"]
    Kd = 1 / Ka
    Kd_err = Kd * math.log(10) * e["logKa"]
    dG = -R_KCAL * T * math.log(Ka)
    TdS = p["dH"] - dG
    c_value = Ka * p["n"] * M0_M

    out = Path(a.out) if a.out else run.path.with_suffix("")
    out.parent.mkdir(parents=True, exist_ok=True)

    # ---- CSV
    with open(f"{out}_heats.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["inj", "volume_uL", "time_s", "Mt_uM", "Xt_uM", "molar_ratio",
                    "Q_sample_ucal", "Q_blank_ucal", "Q_corrected_ucal", "dH_kcal_per_mol_injectant",
                    "fit_kcal_per_mol", "used_in_fit"])
        model = onesite_heats([p["n"], p["logKa"], p["dH"], p["offset"]], vols, V0, M0_M, Xsyr_M) / inj_nmol
        for i, inj in enumerate(run.injections):
            w.writerow([inj.index, f"{vols[i]:.3f}", f"{inj.start_s:.1f}", f"{Mt[i]*1e6:.3f}", f"{Xt[i]*1e6:.3f}",
                        f"{ratio[i]:.4f}", f"{q_sample[i]:.4f}", f"{q_blank[i]:.4f}", f"{q_corr[i]:.4f}",
                        f"{kcal[i]:.4f}", f"{model[i]:.4f}", int(use[i])])

    # ---- report
    def pm(name, unit, fmt="{:.3g}"):
        val, err = p[name], e[name]
        fixed = "" if free[["n", "logKa", "dH", "offset"].index(name)] else "  (fixed)"
        return f"{fmt.format(val)} ± {fmt.format(err)} {unit}{fixed}"

    report_nb = [
        f"ITC analysis: {run.path.name}" + (f" minus {blank.path.name}" if blank else ""),
        f"{int(use.sum())} of {len(use)} injections used (excluded: {', '.join(str(i + 1) for i in np.where(~use)[0]) or 'none'})",
        f"T = {run.temperature_C:g} °C; syringe {syr_mM:g} mM; cell {cell_uM:g} uM; V0 {V0:g} mL",
        "",
        "NO SIGNIFICANT BINDING HEAT above the control.",
        f"  corrected heats: {np.mean(kcal[use]):.3g} ± {np.std(kcal[use], ddof=1):.2g} kcal/mol injectant "
        f"(mean ± SD), no significant trend with molar ratio",
        f"  one-site fit vs flat line: F-test p = {p_bind:.2g}" + ("; fit ran to a parameter bound" if at_bound else ""),
        f"  what remains after subtracting the control is {100 * np.mean(q_corr[use]) / np.mean(q_sample[use]):.2g}% "
        "of the raw sample heat (the rest is ligand dilution)" if blank is not None else "",
        "  Either no binding under these conditions, or binding with dH close to 0 at this temperature.",
        "  Check with an orthogonal assay or another temperature before concluding no binding.",
    ]
    report = [
        f"ITC analysis: {run.path.name}" + (f" minus {blank.path.name}" if blank else ""),
        f"Model: one set of sites (MicroCal OneSites), {int(use.sum())} of {len(use)} injections fitted"
        f" (excluded: {', '.join(str(i + 1) for i in np.where(~use)[0]) or 'none'})",
        f"T = {run.temperature_C:g} °C; syringe {syr_mM:g} mM; cell {cell_uM:g} uM; V0 {V0:g} mL",
        "",
        f"N        = {pm('n', '')}",
        f"Ka       = {Ka:.3g} M^-1   (log10 Ka = {p['logKa']:.3f} ± {e['logKa']:.3f})",
        f"Kd       = {Kd*1e6:.3g} ± {Kd_err*1e6:.2g} uM",
        f"dH       = {pm('dH', 'kcal/mol')}",
        f"dG       = {dG:.3g} kcal/mol",
        f"-TdS     = {-TdS:.3g} kcal/mol",
        f"offset   = {pm('offset', 'kcal/mol injectant')}",
        f"c-value  = {c_value:.3g}   (N·Ka·[M]; reliable Kd and N need roughly 1 < c < 1000)",
        f"fit RMSD = {rmsd:.3g} kcal/mol injectant",
    ]
    if c_value < 1 and a.fix_n is None:
        report.append("! c < 1: N, Ka and dH are strongly correlated. Re-run with --fix-n 1 "
                      "(or the known stoichiometry) and report Kd with that assumption.")
    if blank is not None:
        report.append(f"Blank heats: mean {np.mean(q_blank):.3g} ucal per injection "
                      f"({np.mean(q_blank / inj_nmol):.3g} kcal/mol injectant)")
    if not binding:
        report = report_nb + (["", f"Blank heats: mean {np.mean(q_blank):.3g} ucal per injection "
                                   f"({np.mean(q_blank / inj_nmol):.3g} kcal/mol injectant)"] if blank is not None else [])
    else:
        report.append(f"F-test vs flat line: p = {p_bind:.2g}")
    text = "\n".join(r for r in report if r is not None)
    Path(f"{out}_report.txt").write_text(text + "\n")
    print("\n" + text)

    # ---- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7.5, 9))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.1, 1.4, 0.45], hspace=0.08)
    ax0 = fig.add_subplot(gs[0])
    tmin = run.time / 60
    ax0.plot(tmin, run.dp, lw=0.8, color="#1f4e79", label=run.path.stem)
    for ts, b in pk.baselines:
        ax0.plot(ts / 60, b, lw=0.6, color="#c00000")
    if blank is not None:
        shift = np.median(run.dp) - np.median(blank.dp) - 0.5 * np.ptp(run.dp[run.time > run.injections[0].start_s])
        ax0.plot(blank.time / 60, blank.dp + shift, lw=0.8, color="#7f7f7f", label=f"{blank.path.stem} (offset)")
    ax0.set_ylabel("DP (µcal/s)")
    ax0.set_title(f"{run.path.stem}  ·  {run.temperature_C:g} °C", fontsize=11)
    ax0.legend(fontsize=8, frameon=False, loc="lower right")
    ax0.xaxis.set_label_position("top")
    ax0.xaxis.tick_top()
    ax0.set_xlabel("Time (min)")

    ax1 = fig.add_subplot(gs[1])
    xs = np.linspace(ratio.min(), ratio.max(), 300)
    # smooth model curve: evaluate on a fine grid of equal injections spanning the same total volume
    fine_v = np.full(300, vols.sum() / 300)
    Mt_f, Xt_f = cell_concentrations(fine_v, V0, M0_M, Xsyr_M)
    curve = onesite_heats([p["n"], p["logKa"], p["dH"], p["offset"]], fine_v, V0, M0_M, Xsyr_M) / (fine_v * syr_mM)
    if binding:
        ax1.plot(Xt_f / Mt_f, curve, color="#c00000", lw=1.4, label="one-site fit")
    else:
        ax1.axhline(flat, color="#c00000", lw=1.2, ls="--", label="flat (no binding)")
    ax1.plot(ratio[use], kcal[use], "s", ms=5, mfc="#1f4e79", mec="#1f4e79", label="sample − blank" if blank else "sample")
    ax1.plot(ratio[~use], kcal[~use], "s", ms=5, mfc="white", mec="#1f4e79", label="excluded")
    if blank is not None:
        ax1.plot(ratio, q_blank / inj_nmol, "o", ms=3.5, mfc="none", mec="#7f7f7f", label="blank")
    ax1.axhline(0, color="k", lw=0.5)
    ax1.set_ylabel("kcal/mol of injectant")
    ax1.legend(fontsize=8, frameon=False)
    ax1.text(0.98, 0.04,
             (f"No significant binding\n(F-test p = {p_bind:.2g})\nmean {np.mean(kcal[use]):.2f} kcal/mol") if not binding else
             f"N = {p['n']:.2f}{'' if free[0] else ' (fixed)'}\nKd = {Kd*1e6:.3g} ± {Kd_err*1e6:.2g} µM\n"
             f"ΔH = {p['dH']:.2f} ± {e['dH']:.2f} kcal/mol\nΔG = {dG:.2f}  −TΔS = {-TdS:.2f} kcal/mol",
             transform=ax1.transAxes, ha="right", va="bottom", fontsize=8.5,
             bbox=dict(fc="white", ec="#cccccc", lw=0.6))
    ax1.tick_params(labelbottom=False)

    ax2 = fig.add_subplot(gs[2], sharex=ax1)
    model_pts = onesite_heats([p["n"], p["logKa"], p["dH"], p["offset"]], vols, V0, M0_M, Xsyr_M) / inj_nmol
    ax2.axhline(0, color="k", lw=0.5)
    ax2.plot(ratio[use], (kcal - (model_pts if binding else flat))[use], "s", ms=4, color="#1f4e79")
    ax2.set_ylabel("resid.")
    ax2.set_xlabel("Molar ratio (ligand / macromolecule)")
    fig.savefig(f"{out}_fit.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{out}_fit.pdf", bbox_inches="tight")
    print(f"\nWrote {out}_fit.png/.pdf, {out}_heats.csv, {out}_report.txt")


if __name__ == "__main__":
    main()
