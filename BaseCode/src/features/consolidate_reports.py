#!/usr/bin/env python3
"""
Consolida todos los reportes .txt generados por pipeline_concurrent.py
en un único reporte agregado con distribuciones estadísticas globales.

Uso:
    python consolidate_reports.py --reports_dir /app/reportes --output /app/reportes/CONSOLIDADO.txt
"""

import os
import re
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


# ──────────────────────────────────────────────────────────────────────────────
# PARSEO DE REPORTES INDIVIDUALES
# ──────────────────────────────────────────────────────────────────────────────

def _extract_float(pattern, text, default=None):
    """Extrae el primer float que coincida con el patrón."""
    m = re.search(pattern, text)
    return float(m.group(1)) if m else default

def _extract_int(pattern, text, default=None):
    m = re.search(pattern, text)
    return int(m.group(1)) if m else default


def parse_group_block(block_text):
    """
    Parsea el bloque de texto de un grupo individual dentro del reporte.
    Retorna un dict con todas las métricas o None si el bloque está incompleto.
    """
    group_id = _extract_int(r'Grupo\s+(\d+)', block_text)
    if group_id is None:
        return None

    return {
        'group_id':              group_id,
        'cardinality':           _extract_int(r'Cardinalidad\s+:\s+(\d+)', block_text),
        'duration_seconds':      _extract_float(r'(\d+\.\d+)\s+seg\)', block_text),
        'duration_frames':       _extract_int(r'\((\d+)\s+frames\s*/', block_text),
        'pause_seconds':         _extract_float(r'Tiempo en pausa\s+:.*?(\d+\.\d+)\s+seg', block_text),
        'pause_ratio':           _extract_float(r'\((\d+\.\d+)%\s+del tiempo\)', block_text),
        'flight_length_px':      _extract_float(r'Longitud de vuelo\s+:\s+([\d.]+)\s+px', block_text),
        'net_displacement_px':   _extract_float(r'Desplazamiento neto\s+:\s+([\d.]+)\s+px', block_text),
        'straightness_index':    _extract_float(r'Índice de rectitud\s+:\s+([\d.]+)', block_text),
        'avg_velocity_px_sec':   _extract_float(r'Velocidad promedio\s+:.*?\|\s+([\d.]+)\s+px/seg', block_text),
        'max_velocity_px_frame': _extract_float(r'Velocidad máxima\s+:\s+([\d.]+)\s+px/frame', block_text),
        'pause_ratio_pct':       _extract_float(r'\(([\d.]+)%\s+del tiempo\)', block_text),
    }


def parse_report_file(filepath):
    """
    Parsea un archivo .txt de reporte completo.
    Retorna un dict con metadatos del video y lista de grupos.
    """
    text = Path(filepath).read_text(encoding='utf-8')

    # ── Metadatos del video ──────────────────────────────────────────────────
    video_name = re.search(r'REPORTE ESTADÍSTICO - (.+)', text)
    fps         = _extract_float(r'FPS del video\s+:\s+([\d.]+)', text, default=30.0)
    last_frame  = _extract_int(r'Duración total\s+:\s+(\d+)\s+frames', text, default=0)
    unique_ppl  = _extract_int(r'Personas únicas\s+:\s+(\d+)', text, default=0)
    n_groups    = _extract_int(r'Grupos detectados\s+:\s+(\d+)', text, default=0)

    # ── Extraer bloques por grupo ────────────────────────────────────────────
    # Cada bloque empieza en "Grupo N" y termina antes del siguiente o en "==="
    group_blocks = re.split(r'\n(?=Grupo \d+\n)', text)

    groups = []
    for block in group_blocks:
        if not re.match(r'Grupo \d+', block.strip()):
            continue
        parsed = parse_group_block(block)
        if parsed and parsed['cardinality'] is not None:
            groups.append(parsed)

    return {
        'filename':    os.path.basename(filepath),
        'video_name':  video_name.group(1).strip() if video_name else os.path.basename(filepath),
        'fps':         fps,
        'last_frame':  last_frame,
        'unique_ppl':  unique_ppl,
        'n_groups':    n_groups,
        'groups':      groups,
    }


# ──────────────────────────────────────────────────────────────────────────────
# ESTADÍSTICAS
# ──────────────────────────────────────────────────────────────────────────────

def distribution_summary(values, label, unit=""):
    """Calcula estadísticas descriptivas de una lista de valores."""
    if not values:
        return None
    arr = np.array([v for v in values if v is not None], dtype=float)
    if len(arr) == 0:
        return None
    return {
        'label':  label,
        'unit':   unit,
        'n':      len(arr),
        'mean':   float(np.mean(arr)),
        'median': float(np.median(arr)),
        'std':    float(np.std(arr)),
        'min':    float(np.min(arr)),
        'max':    float(np.max(arr)),
        'p25':    float(np.percentile(arr, 25)),
        'p75':    float(np.percentile(arr, 75)),
        'p90':    float(np.percentile(arr, 90)),
        'p95':    float(np.percentile(arr, 95)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# ESCRITURA DEL REPORTE CONSOLIDADO
# ──────────────────────────────────────────────────────────────────────────────

def write_distribution(f, dist):
    """Escribe un bloque de distribución formateado."""
    if dist is None:
        f.write(f"  Sin datos suficientes\n")
        return
    u = f" {dist['unit']}" if dist['unit'] else ""
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


def generate_consolidated_report(reports_data, output_path):
    """
    Genera el reporte consolidado a partir de la lista de reportes parseados.
    """
    # ── Recolectar todos los valores de todos los grupos de todos los videos ──
    all_groups = []
    for r in reports_data:
        all_groups.extend(r['groups'])

    total_videos  = len(reports_data)
    total_groups  = len(all_groups)
    total_people  = sum(r['unique_ppl'] for r in reports_data)
    total_frames  = sum(r['last_frame'] for r in reports_data)
    total_seconds = sum(r['last_frame'] / r['fps'] for r in reports_data)

    # Extraer vectores de cada métrica
    cardinalities    = [g['cardinality']           for g in all_groups]
    durations        = [g['duration_seconds']      for g in all_groups]
    pause_secs       = [g['pause_seconds']         for g in all_groups]
    pause_ratios     = [g['pause_ratio'] / 100.0   for g in all_groups
                        if g['pause_ratio'] is not None]
    flight_lengths   = [g['flight_length_px']      for g in all_groups]
    net_disps        = [g['net_displacement_px']   for g in all_groups]
    straightness     = [g['straightness_index']    for g in all_groups]
    avg_velocities   = [g['avg_velocity_px_sec']   for g in all_groups]
    max_velocities   = [g['max_velocity_px_frame'] for g in all_groups]

    # Estadísticas por video (para la tabla de resumen)
    groups_per_video   = [r['n_groups']   for r in reports_data]
    people_per_video   = [r['unique_ppl'] for r in reports_data]
    duration_per_video = [r['last_frame'] / r['fps'] for r in reports_data]

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:

        # ── ENCABEZADO ───────────────────────────────────────────────────────
        f.write("=" * 70 + "\n")
        f.write(" REPORTE CONSOLIDADO — ANÁLISIS DE GRUPOS\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"  Videos analizados    : {total_videos}\n")
        f.write(f"  Grupos totales       : {total_groups}\n")
        f.write(f"  Personas únicas      : {total_people:,}\n")
        f.write(f"  Frames totales       : {total_frames:,}\n")
        f.write(f"  Duración total       : {total_seconds/3600:.2f} horas "
                f"({total_seconds:.0f} segundos)\n")
        f.write(f"  Promedio grupos/video: {np.mean(groups_per_video):.1f}\n")
        f.write("\n")

        # ── TABLA DE VIDEOS ──────────────────────────────────────────────────
        f.write("=" * 70 + "\n")
        f.write(" RESUMEN POR VIDEO\n")
        f.write("=" * 70 + "\n")
        f.write(f"  {'Video':<40} {'Grupos':>7} {'Personas':>9} {'Duración':>10}\n")
        f.write(f"  {'-'*40} {'-'*7} {'-'*9} {'-'*10}\n")
        for r in sorted(reports_data, key=lambda x: x['video_name']):
            dur = r['last_frame'] / r['fps']
            f.write(f"  {r['video_name']:<40} {r['n_groups']:>7} "
                    f"{r['unique_ppl']:>9} {dur:>9.1f}s\n")
        f.write("\n")

        # ── DISTRIBUCIONES GLOBALES ──────────────────────────────────────────
        f.write("=" * 70 + "\n")
        f.write(" DISTRIBUCIONES ESTADÍSTICAS GLOBALES (todos los grupos)\n")
        f.write("=" * 70 + "\n\n")

        metrics = [
            distribution_summary(cardinalities,  "Cardinalidad",                       "personas"),
            distribution_summary(durations,       "Tiempo de permanencia",              "seg"),
            distribution_summary(pause_secs,      "Tiempo de pausa",                    "seg"),
            distribution_summary(pause_ratios,    "Ratio de pausa (fracción 0-1)",      ""),
            distribution_summary(flight_lengths,  "Longitud de vuelo (recorrido total)","px"),
            distribution_summary(net_disps,       "Desplazamiento neto (inicio→fin)",   "px"),
            distribution_summary(straightness,    "Índice de rectitud",                 ""),
            distribution_summary(avg_velocities,  "Velocidad promedio",                 "px/seg"),
            distribution_summary(max_velocities,  "Velocidad máxima",                   "px/frame"),
        ]

        for dist in metrics:
            if dist is None:
                continue
            f.write(f"  {dist['label']}\n")
            f.write(f"  {'-' * 50}\n")
            write_distribution(f, dist)
            f.write("\n")

        # ── DISTRIBUCIÓN DE GRUPOS POR VIDEO ─────────────────────────────────
        f.write("=" * 70 + "\n")
        f.write(" DISTRIBUCIÓN DE GRUPOS POR VIDEO\n")
        f.write("=" * 70 + "\n\n")
        gv = distribution_summary(groups_per_video, "Grupos por video", "grupos")
        write_distribution(f, gv)
        f.write("\n")

        # ── DISTRIBUCIÓN DE CARDINALIDAD — TABLA DE FRECUENCIAS ──────────────
        f.write("=" * 70 + "\n")
        f.write(" TABLA DE FRECUENCIAS — CARDINALIDAD\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"  {'Tamaño':>8}  {'Frecuencia':>12}  {'Porcentaje':>12}\n")
        f.write(f"  {'-'*8}  {'-'*12}  {'-'*12}\n")

        card_arr = [c for c in cardinalities if c is not None]
        unique_cards = sorted(set(card_arr))
        for size in unique_cards:
            count = card_arr.count(size)
            pct   = count / len(card_arr) * 100
            f.write(f"  {size:>8}  {count:>12}  {pct:>11.1f}%\n")
        f.write("\n")

        f.write("=" * 70 + "\n")
        f.write(" FIN DEL REPORTE CONSOLIDADO\n")
        f.write("=" * 70 + "\n")

    print(f"Reporte consolidado generado: {output_path}")
    return {m['label']: m for m in metrics if m}


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Consolida reportes .txt de pipeline_concurrent en un único reporte agregado"
    )
    parser.add_argument(
        "--reports_dir",
        type=str,
        default="reportes",
        help="Carpeta con los archivos REPORTE_*.txt (default: reportes/)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Ruta del reporte consolidado de salida (default: <reports_dir>/CONSOLIDADO.txt)"
    )
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    output_path = args.output or str(reports_dir / "CONSOLIDADO.txt")

    # Buscar archivos de reporte
    report_files = sorted(reports_dir.glob("REPORTE_*.txt"))

    if not report_files:
        print(f"No se encontraron archivos REPORTE_*.txt en '{reports_dir}'")
        return

    print(f"Encontrados {len(report_files)} reportes en '{reports_dir}'")

    # Parsear cada reporte
    reports_data = []
    errores = []
    for filepath in report_files:
        try:
            data = parse_report_file(filepath)
            reports_data.append(data)
            n = len(data['groups'])
            print(f"  OK  {filepath.name} — {n} grupos parseados")
        except Exception as e:
            errores.append((filepath.name, str(e)))
            print(f"  ERR {filepath.name} — {e}")

    if not reports_data:
        print("No se pudo parsear ningún reporte.")
        return

    if errores:
        print(f"\nAdvertencia: {len(errores)} archivo(s) con errores fueron omitidos.")

    print(f"\nConsolidando {len(reports_data)} videos ({sum(len(r['groups']) for r in reports_data)} grupos)...")
    generate_consolidated_report(reports_data, output_path)


if __name__ == "__main__":
    main()