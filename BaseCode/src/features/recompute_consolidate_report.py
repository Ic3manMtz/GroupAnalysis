#!/usr/bin/env python3
"""
Recalcula el reporte consolidado completo directamente desde la base de datos,
usando un `pause_threshold` arbitrario — sin necesidad de reprocesar los videos.

Todas las métricas (cardinalidad, permanencia, pausa, longitud de vuelo,
desplazamiento neto, rectitud, velocidad) se recalculan a partir de las
trayectorias guardadas en `group_detections`. Esto garantiza consistencia
total entre métricas y permite experimentar con distintos valores de
pause_threshold sin volver a correr YOLO/DeepSort.

Uso:
    python recompute_consolidated_report.py --pause_threshold 0.5
    python recompute_consolidated_report.py --pause_threshold 0.5 --output reportes/CONSOLIDADO_v2.txt
"""

import os
import argparse
import numpy as np
from collections import defaultdict

from src.database.connection import SessionLocal
from src.database.models import VideoMetadata, GroupDetection, FrameObjectDetection


# ──────────────────────────────────────────────────────────────────────────────
# CÁLCULO DE ESTADÍSTICAS POR GRUPO
# ──────────────────────────────────────────────────────────────────────────────

def compute_group_stats(trajectory, pause_threshold, fps):
    """
    Calcula todas las métricas de un grupo a partir de su trayectoria.

    Args:
        trajectory: lista de tuplas (frame_number, center_x, center_y, size)
                    ordenada por frame_number
        pause_threshold: velocidad en px/frame por debajo de la cual se
                         considera "pausa"
        fps: frames por segundo del video

    Returns:
        dict con todas las métricas, o None si la trayectoria es insuficiente
    """
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

    # Longitud de vuelo: suma de todos los desplazamientos
    coords = np.array([(cx, cy) for _, cx, cy, _ in trajectory])
    diffs = np.diff(coords, axis=0)
    flight_length_px = float(np.sum(np.linalg.norm(diffs, axis=1)))

    # Desplazamiento neto: inicio -> fin
    x_start, y_start = coords[0]
    x_end, y_end = coords[-1]
    net_displacement_px = float(np.sqrt((x_end - x_start) ** 2 + (y_end - y_start) ** 2))

    straightness = (net_displacement_px / flight_length_px) if flight_length_px > 0 else 0.0

    pause_seconds = pause_frames / fps
    pause_ratio = pause_frames / duration_frames if duration_frames > 0 else 0.0

    return {
        'cardinality': int(cardinality),
        'duration_frames': duration_frames,
        'duration_seconds': duration_seconds,
        'pause_frames': pause_frames,
        'pause_seconds': pause_seconds,
        'pause_ratio': pause_ratio,
        'flight_length_px': flight_length_px,
        'net_displacement_px': net_displacement_px,
        'straightness_index': straightness,
        'avg_velocity_px_per_frame': avg_velocity_px_frame,
        'max_velocity_px_per_frame': max_velocity_px_frame,
        'avg_velocity_px_per_sec': avg_velocity_px_frame * fps,
    }


# ──────────────────────────────────────────────────────────────────────────────
# PROCESAMIENTO POR VIDEO
# ──────────────────────────────────────────────────────────────────────────────

def process_video(db, video, pause_threshold):
    """
    Procesa un video: reconstruye trayectorias de grupos desde GroupDetection
    y calcula estadísticas para cada uno.

    Returns:
        dict con metadatos del video y lista de stats por grupo
    """
    # FPS estimado: max frame_number de detecciones individuales / duración (seg)
    max_frame_row = db.query(FrameObjectDetection.frame_number) \
        .filter_by(video_id=video.video_id) \
        .order_by(FrameObjectDetection.frame_number.desc()) \
        .first()
    last_frame = max_frame_row[0] if max_frame_row else 0

    fps = (last_frame / video.duration) if (video.duration and video.duration > 0) else 30.0
    if fps <= 0:
        fps = 30.0

    unique_people = db.query(FrameObjectDetection.track_id) \
        .filter_by(video_id=video.video_id).distinct().count()

    # Trayectorias de grupos
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

    group_stats_list = []
    for gid, traj in trajectories.items():
        stats = compute_group_stats(traj, pause_threshold, fps)
        if stats:
            group_stats_list.append(stats)

    return {
        'video_name': video.title,
        'fps': fps,
        'last_frame': last_frame,
        'unique_people': unique_people,
        'n_groups': len(group_stats_list),
        'groups': group_stats_list,
    }


# ──────────────────────────────────────────────────────────────────────────────
# ESTADÍSTICAS AGREGADAS Y ESCRITURA DEL REPORTE
# ──────────────────────────────────────────────────────────────────────────────

def distribution_summary(values):
    if not values:
        return None
    arr = np.array(values, dtype=float)
    return {
        'n': len(arr),
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'p25': float(np.percentile(arr, 25)),
        'p75': float(np.percentile(arr, 75)),
        'p90': float(np.percentile(arr, 90)),
        'p95': float(np.percentile(arr, 95)),
    }


def write_distribution(f, dist, unit=""):
    if dist is None:
        f.write("  Sin datos suficientes\n")
        return
    u = f" {unit}" if unit else ""
    f.write(f"  n (grupos)   : {dist['n']}\n")
    f.write(f"  Media        : {dist['mean']:.3f}{u}\n")
    f.write(f"  Mediana      : {dist['median']:.3f}{u}\n")
    f.write(f"  Desv. Std    : {dist['std']:.3f}{u}\n")
    f.write(f"  Mínimo       : {dist['min']:.3f}{u}\n")
    f.write(f"  Máximo       : {dist['max']:.3f}{u}\n")
    f.write(f"  P25          : {dist['p25']:.3f}{u}\n")
    f.write(f"  P75          : {dist['p75']:.3f}{u}\n")
    f.write(f"  P90          : {dist['p90']:.3f}{u}\n")
    f.write(f"  P95          : {dist['p95']:.3f}{u}\n")


def generate_report(videos_data, pause_threshold, output_path):
    all_groups = []
    for v in videos_data:
        all_groups.extend(v['groups'])

    total_videos = len(videos_data)
    total_groups = len(all_groups)
    total_people = sum(v['unique_people'] for v in videos_data)
    total_frames = sum(v['last_frame'] for v in videos_data)
    total_seconds = sum(v['last_frame'] / v['fps'] for v in videos_data)

    cardinalities  = [g['cardinality']             for g in all_groups]
    durations      = [g['duration_seconds']        for g in all_groups]
    pause_secs     = [g['pause_seconds']           for g in all_groups]
    pause_ratios   = [g['pause_ratio']             for g in all_groups]
    flight_lengths = [g['flight_length_px']        for g in all_groups]
    net_disps      = [g['net_displacement_px']     for g in all_groups]
    straightness   = [g['straightness_index']      for g in all_groups]
    avg_velocities = [g['avg_velocity_px_per_sec']  for g in all_groups]
    max_velocities = [g['max_velocity_px_per_frame'] for g in all_groups]

    groups_per_video = [v['n_groups'] for v in videos_data]

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(" REPORTE CONSOLIDADO — ANÁLISIS DE GRUPOS (recalculado desde BD)\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"  pause_threshold usado: {pause_threshold} px/frame\n\n")

        f.write(f"  Videos analizados    : {total_videos}\n")
        f.write(f"  Grupos totales       : {total_groups}\n")
        f.write(f"  Personas únicas      : {total_people:,}\n")
        f.write(f"  Frames totales       : {total_frames:,}\n")
        f.write(f"  Duración total       : {total_seconds/3600:.2f} horas "
                f"({total_seconds:.0f} segundos)\n")
        f.write(f"  Promedio grupos/video: {np.mean(groups_per_video):.1f}\n")
        f.write("\n")

        # Tabla por video
        f.write("=" * 70 + "\n")
        f.write(" RESUMEN POR VIDEO\n")
        f.write("=" * 70 + "\n")
        f.write(f"  {'Video':<40} {'Grupos':>7} {'Personas':>9} {'Duración':>10}\n")
        f.write(f"  {'-'*40} {'-'*7} {'-'*9} {'-'*10}\n")
        for v in sorted(videos_data, key=lambda x: x['video_name']):
            dur = v['last_frame'] / v['fps']
            f.write(f"  {v['video_name']:<40} {v['n_groups']:>7} "
                    f"{v['unique_people']:>9} {dur:>9.1f}s\n")
        f.write("\n")

        # Distribuciones globales
        f.write("=" * 70 + "\n")
        f.write(" DISTRIBUCIONES ESTADÍSTICAS GLOBALES (todos los grupos)\n")
        f.write("=" * 70 + "\n\n")

        sections = [
            ("Cardinalidad",                        cardinalities,  "personas"),
            ("Tiempo de permanencia",                durations,      "seg"),
            ("Tiempo de pausa",                      pause_secs,     "seg"),
            ("Ratio de pausa (fracción 0-1)",        pause_ratios,   ""),
            ("Longitud de vuelo (recorrido total)",  flight_lengths, "px"),
            ("Desplazamiento neto (inicio→fin)",     net_disps,      "px"),
            ("Índice de rectitud",                   straightness,   ""),
            ("Velocidad promedio",                   avg_velocities, "px/seg"),
            ("Velocidad máxima",                     max_velocities, "px/frame"),
        ]

        for label, values, unit in sections:
            dist = distribution_summary(values)
            f.write(f"  {label}\n")
            f.write(f"  {'-' * 50}\n")
            write_distribution(f, dist, unit)
            f.write("\n")

        # Distribución de grupos por video
        f.write("=" * 70 + "\n")
        f.write(" DISTRIBUCIÓN DE GRUPOS POR VIDEO\n")
        f.write("=" * 70 + "\n\n")
        write_distribution(f, distribution_summary(groups_per_video), "grupos")
        f.write("\n")

        # Tabla de frecuencias de cardinalidad
        f.write("=" * 70 + "\n")
        f.write(" TABLA DE FRECUENCIAS — CARDINALIDAD\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"  {'Tamaño':>8}  {'Frecuencia':>12}  {'Porcentaje':>12}\n")
        f.write(f"  {'-'*8}  {'-'*12}  {'-'*12}\n")

        unique_cards = sorted(set(cardinalities))
        for size in unique_cards:
            count = cardinalities.count(size)
            pct = count / len(cardinalities) * 100
            f.write(f"  {size:>8}  {count:>12}  {pct:>11.1f}%\n")
        f.write("\n")

        f.write("=" * 70 + "\n")
        f.write(" FIN DEL REPORTE CONSOLIDADO\n")
        f.write("=" * 70 + "\n")

    print(f"Reporte consolidado generado: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Recalcula el reporte consolidado desde la BD con un pause_threshold ajustable"
    )
    parser.add_argument(
        "--pause_threshold", type=float, default=0.5,
        help="Velocidad en px/frame por debajo de la cual se considera pausa (default: 0.5)"
    )
    parser.add_argument(
        "--output", type=str, default="reportes/CONSOLIDADO.txt",
        help="Ruta del reporte de salida (default: reportes/CONSOLIDADO.txt)"
    )
    args = parser.parse_args()

    db = SessionLocal()
    try:
        videos = db.query(VideoMetadata).all()
        if not videos:
            print("No hay videos en la base de datos.")
            return

        print(f"Procesando {len(videos)} videos con pause_threshold={args.pause_threshold}...\n")

        videos_data = []
        for i, video in enumerate(videos, 1):
            data = process_video(db, video, args.pause_threshold)
            print(f"  [{i}/{len(videos)}] {data['video_name']} — {data['n_groups']} grupos")
            videos_data.append(data)

    finally:
        db.close()

    total_groups = sum(v['n_groups'] for v in videos_data)
    print(f"\nGenerando reporte con {total_groups} grupos totales...")

    generate_report(videos_data, args.pause_threshold, args.output)


if __name__ == "__main__":
    main()