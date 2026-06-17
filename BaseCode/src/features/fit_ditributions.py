#!/usr/bin/env python3
"""
Ajusta distribuciones estadísticas a las métricas de comportamiento grupal
(cardinalidad, permanencia, pausa, longitud de vuelo, desplazamiento neto,
rectitud, velocidad promedio, velocidad máxima) usando los datos ya
almacenados en la base de datos.

Para cada métrica:
  - Variables continuas positivas -> compara Log-normal, Gamma, Weibull, Exponencial
  - Variables acotadas en [0,1]    -> ajusta Beta
  - Cardinalidad (discreta, >=2)   -> compara Geométrica desplazada y Binomial Negativa

Genera:
  - fit_report.txt   : resumen de parámetros ajustados y comparación de modelos
  - plots/*.png       : histograma + curvas de densidad ajustadas por métrica

Uso:
    python fit_distributions.py --pause_threshold 0.5 --output_dir reportes/distribution_fits
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict

from src.database.connection import SessionLocal
from src.database.models import VideoMetadata, GroupDetection, FrameObjectDetection


# ──────────────────────────────────────────────────────────────────────────────
# RECONSTRUCCIÓN DE MÉTRICAS DESDE LA BD (misma lógica que recompute_consolidated)
# ──────────────────────────────────────────────────────────────────────────────

def compute_group_stats(trajectory, pause_threshold, fps):
    if len(trajectory) < 2:
        return None

    first_frame = trajectory[0][0]
    last_frame = trajectory[-1][0]
    duration_frames = last_frame - first_frame + 1
    duration_seconds = duration_frames / fps

    cardinality = max(size for _, _, _, size in trajectory)

    velocities = []
    pause_frames = 0
    for i in range(1, len(trajectory)):
        f0, x0, y0, _ = trajectory[i - 1]
        f1, x1, y1, _ = trajectory[i]
        frame_delta = f1 - f0
        if frame_delta <= 0:
            continue
        displacement = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        velocity = displacement / frame_delta
        velocities.append(velocity)
        if velocity < pause_threshold:
            pause_frames += 1

    if not velocities:
        return None

    avg_velocity_px_frame = float(np.mean(velocities))
    max_velocity_px_frame = float(np.max(velocities))

    coords = np.array([(cx, cy) for _, cx, cy, _ in trajectory])
    diffs = np.diff(coords, axis=0)
    flight_length_px = float(np.sum(np.linalg.norm(diffs, axis=1)))

    x_start, y_start = coords[0]
    x_end, y_end = coords[-1]
    net_displacement_px = float(np.sqrt((x_end - x_start) ** 2 + (y_end - y_start) ** 2))

    straightness = (net_displacement_px / flight_length_px) if flight_length_px > 0 else 0.0

    pause_seconds = pause_frames / fps
    pause_ratio = pause_frames / duration_frames if duration_frames > 0 else 0.0

    return {
        'cardinality': int(cardinality),
        'duration_seconds': duration_seconds,
        'pause_seconds': pause_seconds,
        'pause_ratio': pause_ratio,
        'flight_length_px': flight_length_px,
        'net_displacement_px': net_displacement_px,
        'straightness_index': straightness,
        'avg_velocity_px_per_sec': avg_velocity_px_frame * fps,
        'max_velocity_px_per_frame': max_velocity_px_frame,
    }


def load_all_metrics(pause_threshold):
    """Recorre la BD y devuelve un dict metric_name -> np.array de valores."""
    db = SessionLocal()
    metrics = defaultdict(list)

    try:
        videos = db.query(VideoMetadata).all()
        print(f"Cargando métricas de {len(videos)} videos...")

        for i, video in enumerate(videos, 1):
            max_frame_row = db.query(FrameObjectDetection.frame_number) \
                .filter_by(video_id=video.video_id) \
                .order_by(FrameObjectDetection.frame_number.desc()) \
                .first()
            last_frame = max_frame_row[0] if max_frame_row else 0
            fps = (last_frame / video.duration) if (video.duration and video.duration > 0) else 30.0
            if fps <= 0:
                fps = 30.0

            rows = db.query(
                GroupDetection.group_id,
                GroupDetection.frame_number,
                GroupDetection.center_x,
                GroupDetection.center_y,
                GroupDetection.size
            ).filter_by(video_id=video.video_id) \
             .order_by(GroupDetection.group_id, GroupDetection.frame_number) \
             .all()

            trajectories = defaultdict(list)
            for gid, frame_number, cx, cy, size in rows:
                trajectories[gid].append((frame_number, cx, cy, size))

            for gid, traj in trajectories.items():
                stats_dict = compute_group_stats(traj, pause_threshold, fps)
                if stats_dict:
                    for key, val in stats_dict.items():
                        metrics[key].append(val)

            if i % 20 == 0 or i == len(videos):
                print(f"  [{i}/{len(videos)}] procesados")

    finally:
        db.close()

    return {k: np.array(v, dtype=float) for k, v in metrics.items()}


# ──────────────────────────────────────────────────────────────────────────────
# AJUSTE DE DISTRIBUCIONES CONTINUAS (positivas, sin cota superior)
# ──────────────────────────────────────────────────────────────────────────────

CONTINUOUS_CANDIDATES = {
    'lognorm':    stats.lognorm,
    'gamma':      stats.gamma,
    'weibull_min': stats.weibull_min,
    'expon':      stats.expon,
}


def fit_continuous(data, candidates=CONTINUOUS_CANDIDATES):
    """
    Ajusta varias distribuciones continuas (loc fijo en 0) y las compara
    por log-likelihood / AIC y estadístico KS.

    Returns:
        list de dicts ordenados por AIC ascendente (mejor primero)
    """
    results = []
    n = len(data)

    for name, dist in candidates.items():
        try:
            # floc=0 ancla la distribución en cero (variables físicas positivas)
            params = dist.fit(data, floc=0)
            loglik = np.sum(dist.logpdf(data, *params))
            k = len(params) - 1  # no contamos loc, que está fijo
            aic = 2 * k - 2 * loglik

            ks_stat, ks_pvalue = stats.kstest(data, name, args=params)

            results.append({
                'name': name,
                'params': params,
                'loglik': loglik,
                'aic': aic,
                'ks_stat': ks_stat,
                'ks_pvalue': ks_pvalue,
            })
        except Exception as e:
            results.append({
                'name': name,
                'params': None,
                'error': str(e),
            })

    valid = [r for r in results if r.get('params') is not None]
    valid.sort(key=lambda r: r['aic'])
    return valid


# ──────────────────────────────────────────────────────────────────────────────
# AJUSTE DE DISTRIBUCIONES ACOTADAS [0,1] (Beta)
# ──────────────────────────────────────────────────────────────────────────────

def fit_bounded(data, eps=1e-6):
    """
    Ajusta una distribución Beta a datos en [0,1].
    Los valores exactamente 0 o 1 se recortan ligeramente porque la
    log-verosimilitud de Beta es indefinida en los bordes.
    """
    clipped = np.clip(data, eps, 1 - eps)
    try:
        a, b, loc, scale = stats.beta.fit(clipped, floc=0, fscale=1)
        loglik = np.sum(stats.beta.logpdf(clipped, a, b, loc=0, scale=1))
        aic = 2 * 2 - 2 * loglik
        ks_stat, ks_pvalue = stats.kstest(clipped, 'beta', args=(a, b, 0, 1))
        return {
            'name': 'beta',
            'params': (a, b, 0, 1),
            'loglik': loglik,
            'aic': aic,
            'ks_stat': ks_stat,
            'ks_pvalue': ks_pvalue,
        }
    except Exception as e:
        return {'name': 'beta', 'params': None, 'error': str(e)}


# ──────────────────────────────────────────────────────────────────────────────
# AJUSTE DE CARDINALIDAD (discreta, soporte {2, 3, 4, ...})
# ──────────────────────────────────────────────────────────────────────────────

def fit_cardinality(data):
    """
    Ajusta Geométrica desplazada y Binomial Negativa (método de momentos)
    al conteo de personas por grupo (mínimo 2).

    Returns:
        list de dicts con parámetros y chi-cuadrado de bondad de ajuste
    """
    data = data.astype(int)
    n = len(data)
    mean = np.mean(data)
    var = np.var(data)

    results = []

    # ── Geométrica desplazada: soporte {2,3,...} -> geom(p, loc=1) ──────────
    # Para geom estándar (soporte {1,2,...}): mean = 1/p  =>  p = 1/mean(Y)
    # Y = X - 1 (para que el soporte de Y empiece en 1, dado X >= 2)
    p_geom = 1.0 / (mean - 1.0)
    p_geom = min(max(p_geom, 1e-6), 1 - 1e-6)

    # ── Binomial Negativa (método de momentos), loc=2 ───────────────────────
    # Y = X - 2 >= 0.  mean_Y = n*(1-p)/p ; var_Y = n*(1-p)/p^2
    mean_y = mean - 2
    var_y = var
    nbinom_params = None
    if var_y > mean_y > 0:
        p_nb = mean_y / var_y
        n_nb = mean_y * p_nb / (1 - p_nb)
        if n_nb > 0:
            nbinom_params = (n_nb, p_nb)

    # ── Tabla de frecuencias observadas para chi-cuadrado ───────────────────
    max_k = int(np.max(data))
    # Agrupar la cola larga en un bin "k+" para evitar celdas con frecuencia esperada ~0
    tail_cutoff = int(np.percentile(data, 99))
    tail_cutoff = max(tail_cutoff, 3)

    bins = list(range(2, tail_cutoff + 1)) + [tail_cutoff + 1]  # último = "tail_cutoff+1 o más"
    observed = []
    for k in bins[:-1]:
        observed.append(np.sum(data == k))
    observed.append(np.sum(data >= bins[-1]))
    observed = np.array(observed, dtype=float)

    # --- Geométrica: probabilidades esperadas ---
    geom_probs = []
    for k in bins[:-1]:
        geom_probs.append(stats.geom.pmf(k - 1, p_geom))  # Y=k-1, soporte {1,2,...}
    geom_probs.append(stats.geom.sf(bins[-1] - 1 - 1, p_geom))  # P(Y >= bins[-1]-1)
    geom_probs = np.array(geom_probs)
    geom_expected = geom_probs * n
    chi2_geom = np.sum((observed - geom_expected) ** 2 / np.where(geom_expected > 0, geom_expected, 1))
    dof_geom = len(observed) - 1 - 1  # -1 por parámetro estimado
    pvalue_geom = 1 - stats.chi2.cdf(chi2_geom, max(dof_geom, 1))

    results.append({
        'name': 'geometric (shifted, loc=2)',
        'params': {'p': p_geom},
        'chi2': chi2_geom,
        'dof': dof_geom,
        'pvalue': pvalue_geom,
        'observed': observed,
        'expected': geom_expected,
        'bins': bins,
    })

    # --- Binomial Negativa: probabilidades esperadas ---
    if nbinom_params:
        n_nb, p_nb = nbinom_params
        nb_probs = []
        for k in bins[:-1]:
            nb_probs.append(stats.nbinom.pmf(k - 2, n_nb, p_nb))
        nb_probs.append(stats.nbinom.sf(bins[-1] - 2 - 1, n_nb, p_nb))
        nb_probs = np.array(nb_probs)
        nb_expected = nb_probs * n
        chi2_nb = np.sum((observed - nb_expected) ** 2 / np.where(nb_expected > 0, nb_expected, 1))
        dof_nb = len(observed) - 1 - 2  # -2 por parámetros estimados (n, p)
        pvalue_nb = 1 - stats.chi2.cdf(chi2_nb, max(dof_nb, 1))

        results.append({
            'name': 'negative binomial (method of moments, loc=2)',
            'params': {'n': n_nb, 'p': p_nb},
            'chi2': chi2_nb,
            'dof': dof_nb,
            'pvalue': pvalue_nb,
            'observed': observed,
            'expected': nb_expected,
            'bins': bins,
        })

    results.sort(key=lambda r: r['chi2'])
    return results


# ──────────────────────────────────────────────────────────────────────────────
# GRAFICACIÓN
# ──────────────────────────────────────────────────────────────────────────────

def plot_continuous_fit(data, fits, metric_label, unit, outpath, top_n=3, clip_percentile=99):
    """Histograma + curvas de densidad de las top_n distribuciones ajustadas."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Recortar la cola extrema solo para visualización (no afecta el ajuste)
    x_max = np.percentile(data, clip_percentile)
    plot_data = data[data <= x_max]

    ax.hist(plot_data, bins=60, density=True, alpha=0.4, color='steelblue',
            label=f'Datos (hasta P{clip_percentile})')

    x = np.linspace(max(plot_data.min(), 1e-6), x_max, 500)
    colors = ['crimson', 'darkorange', 'forestgreen']

    for i, fit in enumerate(fits[:top_n]):
        dist = CONTINUOUS_CANDIDATES[fit['name']]
        y = dist.pdf(x, *fit['params'])
        ax.plot(x, y, color=colors[i % len(colors)], lw=2,
                label=f"{fit['name']} (AIC={fit['aic']:.1f})")

    ax.set_title(f"{metric_label}")
    ax.set_xlabel(f"{metric_label}" + (f" ({unit})" if unit else ""))
    ax.set_ylabel("Densidad")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


def plot_bounded_fit(data, fit, metric_label, outpath):
    """Histograma + curva Beta ajustada para variables en [0,1]."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(data, bins=50, density=True, alpha=0.4, color='steelblue', label='Datos')

    if fit.get('params') is not None:
        a, b, loc, scale = fit['params']
        x = np.linspace(1e-4, 1 - 1e-4, 500)
        y = stats.beta.pdf(x, a, b, loc, scale)
        ax.plot(x, y, color='crimson', lw=2,
                label=f"beta(a={a:.2f}, b={b:.2f}) AIC={fit['aic']:.1f}")

    ax.set_title(metric_label)
    ax.set_xlabel(metric_label)
    ax.set_ylabel("Densidad")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


def plot_cardinality_fit(data, fits, outpath):
    """Barras observadas vs esperadas para Geométrica y Binomial Negativa."""
    best = fits[0]
    bins = best['bins']
    observed = best['observed']

    fig, ax = plt.subplots(figsize=(9, 5))
    x_labels = [str(b) for b in bins[:-1]] + [f"{bins[-1]}+"]
    x_pos = np.arange(len(x_labels))
    width = 0.25

    ax.bar(x_pos - width, observed, width=width, label='Observado', color='steelblue')

    colors = ['crimson', 'darkorange']
    for i, fit in enumerate(fits[:2]):
        ax.bar(x_pos + i * width, fit['expected'], width=width,
               label=f"{fit['name']} (χ²={fit['chi2']:.1f})", color=colors[i % len(colors)])

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Cardinalidad (personas por grupo)")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Cardinalidad — Observado vs. Ajustes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# REPORTE
# ──────────────────────────────────────────────────────────────────────────────

DIST_FORMULAS = {
    'lognorm':     "f(x) = 1 / (x*sigma*sqrt(2*pi)) * exp(-(ln(x)-mu)^2 / (2*sigma^2))",
    'gamma':       "f(x) = x^(a-1) * exp(-x/scale) / (Gamma(a) * scale^a)",
    'weibull_min': "f(x) = (k/scale) * (x/scale)^(k-1) * exp(-(x/scale)^k)",
    'expon':       "f(x) = (1/scale) * exp(-x/scale)",
    'beta':        "f(x) = x^(a-1) * (1-x)^(b-1) / B(a,b)",
}


def write_continuous_section(f, label, unit, data, fits, zero_fraction=None):
    f.write(f"{label}\n")
    f.write("-" * 60 + "\n")
    f.write(f"  n = {len(data)}\n")
    if zero_fraction is not None:
        f.write(f"  Fracción de valores en 0 (excluidos del ajuste): {zero_fraction*100:.1f}%\n")
    f.write(f"  Media = {np.mean(data):.4f} {unit}  |  Mediana = {np.median(data):.4f} {unit}\n\n")

    f.write(f"  {'Distribución':<14} {'AIC':>12} {'KS stat':>10} {'KS p-valor':>12}\n")
    for fit in fits:
        f.write(f"  {fit['name']:<14} {fit['aic']:>12.2f} {fit['ks_stat']:>10.4f} {fit['ks_pvalue']:>12.4g}\n")

    best = fits[0]
    f.write(f"\n  >> Mejor ajuste (menor AIC): {best['name']}\n")
    f.write(f"     Fórmula: {DIST_FORMULAS.get(best['name'], 'N/D')}\n")

    # scipy devuelve params en orden: (shape1, [shape2,...], loc, scale)
    params = best['params']
    if best['name'] == 'lognorm':
        f.write(f"     Parámetros: sigma(shape)={params[0]:.4f}, loc={params[1]:.4f}, scale={params[2]:.4f}\n")
        f.write(f"     (mu = ln(scale) = {np.log(params[2]):.4f})\n")
    elif best['name'] == 'gamma':
        f.write(f"     Parámetros: a(shape)={params[0]:.4f}, loc={params[1]:.4f}, scale={params[2]:.4f}\n")
    elif best['name'] == 'weibull_min':
        f.write(f"     Parámetros: k(shape)={params[0]:.4f}, loc={params[1]:.4f}, scale={params[2]:.4f}\n")
    elif best['name'] == 'expon':
        f.write(f"     Parámetros: loc={params[0]:.4f}, scale={params[1]:.4f}\n")

    f.write("\n  Nota: con n grande, el p-valor de KS casi siempre será ~0\n")
    f.write("  (cualquier desviación, aunque mínima, se vuelve 'significativa').\n")
    f.write("  Usa el AIC y el estadístico KS para comparación RELATIVA entre\n")
    f.write("  modelos, y la gráfica para evaluar el ajuste visualmente.\n")
    f.write("\n\n")


def write_bounded_section(f, label, data, fit):
    f.write(f"{label}\n")
    f.write("-" * 60 + "\n")
    f.write(f"  n = {len(data)}\n")
    f.write(f"  Media = {np.mean(data):.4f}  |  Mediana = {np.median(data):.4f}\n\n")

    if fit.get('params') is None:
        f.write(f"  No se pudo ajustar Beta: {fit.get('error')}\n\n\n")
        return

    a, b, loc, scale = fit['params']
    f.write(f"  Distribución: Beta(a={a:.4f}, b={b:.4f}) en [0,1]\n")
    f.write(f"     Fórmula: {DIST_FORMULAS['beta']}\n")
    f.write(f"  AIC = {fit['aic']:.2f}  |  KS stat = {fit['ks_stat']:.4f}  |  KS p-valor = {fit['ks_pvalue']:.4g}\n")

    mean_beta = a / (a + b)
    f.write(f"  Media teórica de Beta(a,b) = a/(a+b) = {mean_beta:.4f}\n")
    f.write("\n\n")


def write_cardinality_section(f, data, fits):
    f.write("Cardinalidad (personas por grupo)\n")
    f.write("-" * 60 + "\n")
    f.write(f"  n = {len(data)}\n")
    f.write(f"  Media = {np.mean(data):.4f}  |  Mediana = {np.median(data):.1f}  |  Máximo = {int(np.max(data))}\n\n")

    f.write(f"  {'Modelo':<45} {'Chi2':>10} {'dof':>6} {'p-valor':>10}\n")
    for fit in fits:
        f.write(f"  {fit['name']:<45} {fit['chi2']:>10.2f} {fit['dof']:>6} {fit['pvalue']:>10.4g}\n")

    best = fits[0]
    f.write(f"\n  >> Mejor ajuste (menor Chi2): {best['name']}\n")
    if 'p' in best['params']:
        p = best['params']['p']
        if 'n' in best['params']:
            f.write(f"     Parámetros: n={best['params']['n']:.4f}, p={p:.4f}\n")
            f.write(f"     pmf: P(X=k) = C(k-2+n-1, k-2) * p^n * (1-p)^(k-2),  k=2,3,...\n")
        else:
            f.write(f"     Parámetro: p={p:.4f}\n")
            f.write(f"     pmf: P(X=k) = (1-p)^(k-2) * p,  k=2,3,...\n")
            f.write(f"     Media teórica = 1 + 1/p = {1 + 1/p:.4f}\n")

    f.write("\n  Nota: con n grande, el chi-cuadrado también tiende a rechazar\n")
    f.write("  cualquier modelo. Usa el valor de Chi2 para comparación RELATIVA\n")
    f.write("  y la gráfica de barras para evaluar el ajuste visualmente.\n")
    f.write("\n\n")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

METRIC_DEFINITIONS = [
    # (key, label, unit, kind)
    ('cardinality',             "Cardinalidad",                       "personas", 'discrete'),
    ('duration_seconds',        "Tiempo de permanencia",              "seg",      'continuous'),
    ('pause_seconds',           "Tiempo de pausa",                    "seg",      'continuous_zeros'),
    ('pause_ratio',             "Ratio de pausa",                     "",         'bounded'),
    ('flight_length_px',        "Longitud de vuelo (recorrido total)","px",       'continuous'),
    ('net_displacement_px',     "Desplazamiento neto (inicio→fin)",   "px",       'continuous'),
    ('straightness_index',      "Índice de rectitud",                 "",         'bounded'),
    ('avg_velocity_px_per_sec', "Velocidad promedio",                 "px/seg",   'continuous'),
    ('max_velocity_px_per_frame', "Velocidad máxima",                 "px/frame", 'continuous'),
]


def main():
    parser = argparse.ArgumentParser(
        description="Ajusta distribuciones estadísticas a las métricas de grupos"
    )
    parser.add_argument("--pause_threshold", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default="reportes/distribution_fits")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    metrics = load_all_metrics(args.pause_threshold)

    report_path = os.path.join(args.output_dir, "fit_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(" AJUSTE DE DISTRIBUCIONES ESTADÍSTICAS\n")
        f.write(f" pause_threshold = {args.pause_threshold} px/frame\n")
        f.write("=" * 70 + "\n\n")

        for key, label, unit, kind in METRIC_DEFINITIONS:
            data = metrics.get(key)
            if data is None or len(data) == 0:
                f.write(f"{label}: sin datos\n\n")
                continue

            print(f"Ajustando: {label} (n={len(data)})...")

            if kind == 'discrete':
                fits = fit_cardinality(data)
                write_cardinality_section(f, data, fits)
                plot_cardinality_fit(data, fits, os.path.join(plots_dir, f"{key}.png"))

            elif kind == 'bounded':
                fit = fit_bounded(data)
                write_bounded_section(f, label, data, fit)
                if fit.get('params') is not None:
                    plot_bounded_fit(data, fit, label, os.path.join(plots_dir, f"{key}.png"))

            elif kind == 'continuous_zeros':
                zero_mask = data == 0
                zero_fraction = float(np.mean(zero_mask))
                nonzero = data[~zero_mask]
                if len(nonzero) < 10:
                    f.write(f"{label}: insuficientes valores distintos de cero\n\n")
                    continue
                fits = fit_continuous(nonzero)
                write_continuous_section(f, label, unit, nonzero, fits, zero_fraction=zero_fraction)
                plot_continuous_fit(nonzero, fits, label, unit, os.path.join(plots_dir, f"{key}.png"))

            else:  # continuous
                # Excluir ceros si los hay (lognorm/gamma no admiten 0)
                nonzero = data[data > 0]
                fits = fit_continuous(nonzero)
                write_continuous_section(f, label, unit, nonzero, fits)
                plot_continuous_fit(nonzero, fits, label, unit, os.path.join(plots_dir, f"{key}.png"))

        f.write("=" * 70 + "\n")
        f.write(" FIN DEL REPORTE\n")
        f.write("=" * 70 + "\n")

    print(f"\nReporte de ajuste generado en: {report_path}")
    print(f"Gráficas guardadas en: {plots_dir}")


if __name__ == "__main__":
    main()