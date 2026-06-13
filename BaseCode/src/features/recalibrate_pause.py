#!/usr/bin/env python3
"""
Recalibra el parámetro `pause_threshold` reutilizando las trayectorias de
grupos ya almacenadas en la base de datos (tabla group_detections), sin
necesidad de reprocesar los videos con YOLO/DeepSort.

Para cada grupo, reconstruye su trayectoria (frame_number, center_x, center_y)
y calcula las velocidades frame-a-frame. Luego, para cada umbral candidato,
calcula qué fracción de esos pasos caería por debajo del umbral (i.e., se
clasificaría como "pausa"), y agrega la distribución del pause_ratio
resultante sobre todos los grupos.

Uso:
    python recalibrate_pause_threshold.py
    python recalibrate_pause_threshold.py --thresholds 1 2 3 4 5 6 8 10
    python recalibrate_pause_threshold.py --output reportes/PAUSE_CALIBRATION.txt
"""

import argparse
import numpy as np
from collections import defaultdict

from src.database.connection import SessionLocal
from src.database.models import VideoMetadata, GroupDetection


DEFAULT_THRESHOLDS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]


def compute_group_velocities(rows):
    """
    Calcula las velocidades frame-a-frame (px/frame) de un grupo
    a partir de su trayectoria ordenada por frame_number.

    Args:
        rows: lista de tuplas (frame_number, center_x, center_y) ordenada

    Returns:
        (velocities, duration_frames)
            velocities: lista de velocidades por paso (puede estar vacía)
            duration_frames: first_frame -> last_frame inclusive
    """
    if len(rows) < 2:
        return [], (1 if rows else 0)

    velocities = []
    for i in range(1, len(rows)):
        f0, x0, y0 = rows[i - 1]
        f1, x1, y1 = rows[i]
        frame_delta = f1 - f0
        if frame_delta > 0:
            displacement = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
            velocities.append(displacement / frame_delta)

    duration_frames = rows[-1][0] - rows[0][0] + 1
    return velocities, duration_frames


def distribution_summary(values):
    """Resumen estadístico rápido de una lista de valores."""
    if not values:
        return None
    arr = np.array(values, dtype=float)
    return {
        'n':      len(arr),
        'mean':   float(np.mean(arr)),
        'median': float(np.median(arr)),
        'std':    float(np.std(arr)),
        'p25':    float(np.percentile(arr, 25)),
        'p75':    float(np.percentile(arr, 75)),
        'p90':    float(np.percentile(arr, 90)),
    }


def analyze_video(db, video, thresholds):
    """
    Procesa todos los grupos de un video y retorna, para cada threshold,
    la lista de pause_ratio resultantes (uno por grupo).
    """
    rows = db.query(
        GroupDetection.group_id,
        GroupDetection.frame_number,
        GroupDetection.center_x,
        GroupDetection.center_y
    ).filter_by(video_id=video.video_id) \
     .order_by(GroupDetection.group_id, GroupDetection.frame_number) \
     .all()

    # Agrupar por group_id manteniendo el orden por frame
    trajectories = defaultdict(list)
    for gid, frame_number, cx, cy in rows:
        trajectories[gid].append((frame_number, cx, cy))

    # Para cada threshold, acumular el pause_ratio de cada grupo de este video
    ratios_by_threshold = {th: [] for th in thresholds}
    # También guardamos las velocidades crudas para estadísticas globales
    all_velocities = []

    for gid, traj in trajectories.items():
        velocities, duration_frames = compute_group_velocities(traj)
        if not velocities or duration_frames <= 0:
            continue

        all_velocities.extend(velocities)

        for th in thresholds:
            pause_frames = sum(1 for v in velocities if v < th)
            ratio = pause_frames / duration_frames
            ratios_by_threshold[th].append(ratio)

    return ratios_by_threshold, all_velocities


def main():
    parser = argparse.ArgumentParser(
        description="Recalibra pause_threshold usando datos ya existentes en la BD"
    )
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS,
        help=f"Lista de umbrales candidatos en px/frame (default: {DEFAULT_THRESHOLDS})"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Ruta de archivo para guardar el reporte (opcional; si no se da, solo imprime en consola)"
    )
    args = parser.parse_args()

    thresholds = sorted(args.thresholds)

    db = SessionLocal()
    try:
        videos = db.query(VideoMetadata).all()
        if not videos:
            print("No hay videos en la base de datos.")
            return

        print(f"Procesando {len(videos)} videos...\n")

        # Acumuladores globales
        global_ratios = {th: [] for th in thresholds}
        global_velocities = []

        for i, video in enumerate(videos, 1):
            ratios_by_threshold, velocities = analyze_video(db, video, thresholds)

            n_groups = len(ratios_by_threshold[thresholds[0]]) if ratios_by_threshold else 0
            print(f"  [{i}/{len(videos)}] {video.title} — {n_groups} grupos")

            for th in thresholds:
                global_ratios[th].extend(ratios_by_threshold[th])
            global_velocities.extend(velocities)

    finally:
        db.close()

    # ── Construir reporte ────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 70)
    lines.append(" RECALIBRACIÓN DE pause_threshold")
    lines.append("=" * 70)
    lines.append("")

    # Distribución general de velocidades (sin threshold aplicado)
    vel_summary = distribution_summary(global_velocities)
    if vel_summary:
        lines.append("Distribución de velocidades frame-a-frame (todos los pasos, px/frame):")
        lines.append(f"  n      : {vel_summary['n']:,}")
        lines.append(f"  Media  : {vel_summary['mean']:.3f}")
        lines.append(f"  Mediana: {vel_summary['median']:.3f}")
        lines.append(f"  Std    : {vel_summary['std']:.3f}")
        lines.append(f"  P25    : {vel_summary['p25']:.3f}")
        lines.append(f"  P75    : {vel_summary['p75']:.3f}")
        lines.append(f"  P90    : {vel_summary['p90']:.3f}")
        lines.append("")

    lines.append("Ratio de pausa resultante por umbral candidato")
    lines.append("(fracción de tiempo que un grupo se clasificaría como 'en pausa'):")
    lines.append("")
    header = f"{'Umbral (px/frame)':>18} | {'Media':>8} | {'Mediana':>8} | {'P25':>8} | {'P75':>8} | {'P90':>8}"
    lines.append(header)
    lines.append("-" * len(header))

    for th in thresholds:
        summary = distribution_summary(global_ratios[th])
        if summary is None:
            continue
        lines.append(
            f"{th:>18.2f} | {summary['mean']:>8.3f} | {summary['median']:>8.3f} | "
            f"{summary['p25']:>8.3f} | {summary['p75']:>8.3f} | {summary['p90']:>8.3f}"
        )

    lines.append("")
    lines.append("=" * 70)
    lines.append("Interpretación:")
    lines.append("  - pause_ratio cercano a 1.0  -> el umbral es muy alto, casi todo")
    lines.append("    el movimiento se clasifica como pausa (poco útil).")
    lines.append("  - pause_ratio cercano a 0.0  -> el umbral es muy bajo, casi nada")
    lines.append("    se clasifica como pausa (también poco útil).")
    lines.append("  - Busca un umbral donde la mediana del pause_ratio esté en un")
    lines.append("    rango intermedio (ej. 0.3 - 0.6), o que coincida con tu")
    lines.append("    expectativa de campo sobre cuánto tiempo la gente realmente")
    lines.append("    se detiene a conversar vs. está caminando.")
    lines.append("=" * 70)

    report = "\n".join(lines)
    print("\n" + report)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(f"\nReporte guardado en: {args.output}")


if __name__ == "__main__":
    main()