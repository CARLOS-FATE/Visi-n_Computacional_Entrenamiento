# report_generator.py

"""
Módulo de Generación de Reportes

Autor: [Tu Nombre]
Fecha: [Fecha Actual]
Versión: 1.0
Descripción:
    - Genera un reporte detallado en formato CSV basado en las detecciones realizadas.
    - Incluye análisis del uso de implementos de seguridad y carga de trabajo.
"""

import os
import pandas as pd
import datetime

def generate_report(data, output_path):
    """
    Genera un reporte en formato CSV con los resultados del análisis.

    Args:
        data (list): Lista de detecciones en formato [(ID, Objeto, Tiempo Detectado, Duración)]
        output_path (str): Ruta donde se guardará el reporte.

    Returns:
        None
    """
    # Crear dataframe
    df = pd.DataFrame(data, columns=['ID', 'Objeto', 'Tiempo Detectado', 'Duración (s)'])

    # Cálculo del total de objetos detectados por tipo
    summary = df.groupby('Objeto').size().reset_index(name='Total Detectado')

    # Cálculo del tiempo total de detección por objeto
    duration_summary = df.groupby('Objeto')['Duración (s)'].sum().reset_index()
    duration_summary.columns = ['Objeto', 'Tiempo Total (s)']

    # Unir ambos resúmenes
    report = pd.merge(summary, duration_summary, on='Objeto')

    # Agregar timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report['Fecha y Hora'] = timestamp

    # Guardar reporte en CSV
    report_path = os.path.join(output_path, f"reporte_{timestamp.replace(':', '-')}.csv")
    report.to_csv(report_path, index=False)
    print(f"Reporte generado y guardado en: {report_path}")

    return report_path
