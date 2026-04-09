#!/usr/bin/env python3
"""
Script para procesar reportes de detección de grupos y generar estadísticas
para el documento de resultados de la investigación.

Uso:
    python procesar_reportes.py

Requisitos:
    - Carpeta 'Reportes' con archivos .txt en el mismo directorio
"""

import os
import re
from collections import defaultdict
from pathlib import Path


class ProcesadorReportes:
    """Clase para procesar reportes de detección grupal."""
    
    def __init__(self, carpeta_reportes="Reportes"):
        self.carpeta_reportes = carpeta_reportes
        self.reportes_data = []
        self.stats_por_fecha = defaultdict(lambda: {
            'videos': 0,
            'personas_unicas': 0,
            'grupos_detectados': 0,
            'duracion_frames': 0,
            'top_grupos': []
        })
        
    def extraer_fecha_video(self, nombre_archivo):
        """
        Extrae la fecha del nombre del archivo.
        Ejemplos: '11-14-2025-V3.mp4' -> '11-14', '05-02-2025-V1.mp4' -> '05-02'
        """
        match = re.search(r'(\d{2})-(\d{2})-(\d{4})', nombre_archivo)
        if match:
            mes, dia, año = match.groups()
            return f"{mes}-{dia}"
        return None
    
    def parsear_reporte(self, contenido, nombre_archivo):
        """
        Parsea el contenido de un reporte individual.
        
        Args:
            contenido: Texto del reporte
            nombre_archivo: Nombre del archivo TXT
            
        Returns:
            Dict con los datos extraídos
        """
        data = {
            'archivo': nombre_archivo,
            'nombre_video': None,
            'fecha': None,
            'duracion_frames': 0,
            'personas_unicas': 0,
            'grupos_detectados': 0,
            'top_grupos': []
        }
        
        # Extraer nombre del video
        video_match = re.search(r'REPORTE AUTOMÁTICO - (.+\.mp4)', contenido)
        if video_match:
            data['nombre_video'] = video_match.group(1)
            data['fecha'] = self.extraer_fecha_video(data['nombre_video'])
        
        # Extraer duración en frames
        duracion_match = re.search(r'Duración frames:\s*(\d+)', contenido)
        if duracion_match:
            data['duracion_frames'] = int(duracion_match.group(1))
        
        # Extraer personas únicas
        personas_match = re.search(r'Personas únicas:\s*(\d+)', contenido)
        if personas_match:
            data['personas_unicas'] = int(personas_match.group(1))
        
        # Extraer grupos detectados
        grupos_match = re.search(r'Grupos detectados:\s*(\d+)', contenido)
        if grupos_match:
            data['grupos_detectados'] = int(grupos_match.group(1))
        
        # Extraer top 5 grupos
        grupos_pattern = r'- Grupo (\d+):\s*(\d+)\s*frames'
        grupos_matches = re.findall(grupos_pattern, contenido)
        for grupo_id, frames in grupos_matches:
            data['top_grupos'].append({
                'id': int(grupo_id),
                'frames': int(frames)
            })
        
        return data
    
    def procesar_todos_reportes(self):
        """Procesa todos los archivos .txt en la carpeta de reportes."""
        carpeta = Path(self.carpeta_reportes)
        
        if not carpeta.exists():
            print(f"❌ Error: La carpeta '{self.carpeta_reportes}' no existe.")
            print(f"   Asegúrate de que la carpeta esté en el mismo directorio que este script.")
            return False
        
        archivos_txt = sorted(carpeta.glob('*.txt'))
        
        if not archivos_txt:
            print(f"❌ Error: No se encontraron archivos .txt en '{self.carpeta_reportes}'")
            return False
        
        print(f"📂 Procesando {len(archivos_txt)} reportes...\n")
        
        for archivo in archivos_txt:
            try:
                with open(archivo, 'r', encoding='utf-8') as f:
                    contenido = f.read()
                
                data = self.parsear_reporte(contenido, archivo.name)
                self.reportes_data.append(data)
                
                # Agregar a estadísticas por fecha
                if data['fecha']:
                    self.stats_por_fecha[data['fecha']]['videos'] += 1
                    self.stats_por_fecha[data['fecha']]['personas_unicas'] += data['personas_unicas']
                    self.stats_por_fecha[data['fecha']]['grupos_detectados'] += data['grupos_detectados']
                    self.stats_por_fecha[data['fecha']]['duracion_frames'] += data['duracion_frames']
                    self.stats_por_fecha[data['fecha']]['top_grupos'].extend(data['top_grupos'])
                
            except Exception as e:
                print(f"⚠️  Error procesando {archivo.name}: {e}")
        
        print(f"✅ {len(self.reportes_data)} reportes procesados exitosamente\n")
        return True
    
    def calcular_estadisticas_globales(self):
        """Calcula estadísticas agregadas de todos los reportes."""
        total_videos = len(self.reportes_data)
        total_frames = sum(r['duracion_frames'] for r in self.reportes_data)
        total_personas = sum(r['personas_unicas'] for r in self.reportes_data)
        total_grupos = sum(r['grupos_detectados'] for r in self.reportes_data)
        
        # Calcular horas estimadas (asumiendo 30 FPS)
        fps = 30
        total_segundos = total_frames / fps
        total_horas = total_segundos / 3600
        
        return {
            'total_videos': total_videos,
            'total_frames': total_frames,
            'total_horas': total_horas,
            'total_personas': total_personas,
            'total_grupos': total_grupos
        }
    
    def obtener_top_grupos_globales(self, n=10):
        """Obtiene los N grupos más duraderos de todos los videos."""
        todos_grupos = []
        
        for reporte in self.reportes_data:
            for grupo in reporte['top_grupos']:
                todos_grupos.append({
                    'video': reporte['nombre_video'],
                    'fecha': reporte['fecha'],
                    'grupo_id': grupo['id'],
                    'frames': grupo['frames'],
                    'segundos': grupo['frames'] / 30,
                    'minutos': (grupo['frames'] / 30) / 60
                })
        
        # Ordenar por duración descendente
        todos_grupos.sort(key=lambda x: x['frames'], reverse=True)
        
        return todos_grupos[:n]
    
    def obtener_grupo_mas_corto(self):
        """Obtiene el grupo con menor duración."""
        todos_grupos = []
        
        for reporte in self.reportes_data:
            for grupo in reporte['top_grupos']:
                todos_grupos.append({
                    'video': reporte['nombre_video'],
                    'fecha': reporte['fecha'],
                    'grupo_id': grupo['id'],
                    'frames': grupo['frames']
                })
        
        if todos_grupos:
            return min(todos_grupos, key=lambda x: x['frames'])
        return None
    
    def generar_reporte_latex(self):
        """Genera un reporte con todos los valores necesarios para LaTeX."""
        stats = self.calcular_estadisticas_globales()
        
        print("=" * 70)
        print("📊 ESTADÍSTICAS GLOBALES")
        print("=" * 70)
        print(f"Total de Videos Analizados: {stats['total_videos']}")
        print(f"Duración Total (Frames): {stats['total_frames']:,}")
        print(f"Duración Total (Horas): {stats['total_horas']:.2f} horas")
        print(f"Total Personas Únicas: {stats['total_personas']:,}")
        print(f"Total Grupos Detectados: {stats['total_grupos']:,}")
        print()
        
        print("=" * 70)
        print("📅 ESTADÍSTICAS POR FECHA")
        print("=" * 70)
        
        for fecha in sorted(self.stats_por_fecha.keys()):
            data = self.stats_por_fecha[fecha]
            ratio = data['personas_unicas'] / data['grupos_detectados'] if data['grupos_detectados'] > 0 else 0
            
            print(f"\nFecha: {fecha}")
            print(f"  Videos: {data['videos']}")
            print(f"  Personas Únicas: {data['personas_unicas']:,}")
            print(f"  Grupos Detectados: {data['grupos_detectados']}")
            print(f"  Ratio Grupos/Personas: 1:{ratio:.1f}")
            print(f"  Duración Total: {data['duracion_frames']:,} frames")
        
        print()
        print("=" * 70)
        print("🏆 TOP 10 GRUPOS MÁS DURADEROS (Global)")
        print("=" * 70)
        
        top_grupos = self.obtener_top_grupos_globales(10)
        for i, grupo in enumerate(top_grupos, 1):
            print(f"{i:2d}. Grupo {grupo['grupo_id']} ({grupo['fecha']}, {grupo['video']})")
            print(f"    └─ {grupo['frames']:,} frames (~{grupo['minutos']:.1f} minutos)")
        
        print()
        print("=" * 70)
        print("⚡ GRUPO MÁS CORTO")
        print("=" * 70)
        
        grupo_corto = self.obtener_grupo_mas_corto()
        if grupo_corto:
            print(f"Grupo {grupo_corto['grupo_id']} ({grupo_corto['fecha']}, {grupo_corto['video']})")
            print(f"  └─ {grupo_corto['frames']} frames")
        
        print()
        
        # Calcular ratio promedio
        ratio_total = stats['total_personas'] / stats['total_grupos'] if stats['total_grupos'] > 0 else 0
        print(f"📈 Ratio Promedio General: 1:{ratio_total:.1f}")
        print()
    
    def exportar_a_csv(self, archivo_salida="estadisticas_reportes.csv"):
        """Exporta las estadísticas a un archivo CSV."""
        import csv
        
        with open(archivo_salida, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Escribir encabezado
            writer.writerow([
                'Archivo', 'Video', 'Fecha', 'Duración (frames)', 
                'Personas Únicas', 'Grupos Detectados', 'Top Grupo (frames)'
            ])
            
            # Escribir datos
            for reporte in self.reportes_data:
                top_grupo = reporte['top_grupos'][0]['frames'] if reporte['top_grupos'] else 0
                writer.writerow([
                    reporte['archivo'],
                    reporte['nombre_video'],
                    reporte['fecha'],
                    reporte['duracion_frames'],
                    reporte['personas_unicas'],
                    reporte['grupos_detectados'],
                    top_grupo
                ])
        
        print(f"💾 Datos exportados a: {archivo_salida}")
        print()


def main():
    """Función principal."""
    print("\n" + "=" * 70)
    print("🔬 PROCESADOR DE REPORTES - ANÁLISIS GRUPAL")
    print("=" * 70)
    print()
    
    procesador = ProcesadorReportes()
    
    if procesador.procesar_todos_reportes():
        procesador.generar_reporte_latex()
        procesador.exportar_a_csv()
        
        print("=" * 70)
        print("✅ Proceso completado exitosamente")
        print("=" * 70)
        print()
        print("💡 Los valores mostrados arriba pueden ser usados para actualizar")
        print("   tu documento LaTeX (resultados.tex)")
        print()
    else:
        print("\n❌ No se pudo completar el procesamiento")
        print("   Verifica que la carpeta 'Reportes' exista y contenga archivos .txt\n")


if __name__ == "__main__":
    main()
