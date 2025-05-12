import os
import pandas as pd
import datetime

"""
Módulo de Generación de Reportes

Autor: [Tu Nombre]
Fecha: [Fecha Actual]
Versión: 2.0
Descripción:
    - Genera reportes detallados y resumen en formato CSV con detecciones y duraciones.
    - Compatible con el flujo optimizado del sistema con YOLOv11.
"""

def generate_report(data, output_path, class_names=None):
    """
    Genera reportes detallado y resumen en formato CSV.

    Args:
        data (list): Lista de tuplas (ID, Objeto, Tiempo Detectado en ms, duración=0).
        output_path (str): Carpeta donde se guardarán los archivos.
        class_names (list or dict): Lista o diccionario con los nombres de clases.

    Returns:
        str: Ruta del archivo principal de reporte generado.
    """
    if not data:
        raise ValueError("No hay datos de detección para generar el reporte.")

    os.makedirs(output_path, exist_ok=True)

    # Convertir IDs numéricos en nombres de clase si es necesario
    processed_data = []
    for row in data:
        obj_id, clase, tiempo, duracion = row

        # Si clase es un número y tenemos nombres, convertirlo
        if isinstance(clase, int) and class_names:
            try:
                clase = class_names[clase]
            except (IndexError, KeyError):
                clase = f"clase_{clase}"

        processed_data.append((obj_id, clase, tiempo, duracion))

    df = pd.DataFrame(processed_data, columns=['ID', 'Objeto', 'Tiempo Detectado (ms)', 'Duración Placeholder'])

    # Reporte detallado por objeto/ID
    detalle = df.groupby(['ID', 'Objeto'])['Tiempo Detectado (ms)'].agg(['min', 'max']).reset_index()
    detalle['Duración Estimada (s)'] = (detalle['max'] - detalle['min']) / 1000.0
    detalle = detalle.rename(columns={
        'min': 'Primer Detección (ms)',
        'max': 'Última Detección (ms)'
    })

    # Resumen por tipo de objeto
    resumen = detalle.groupby('Objeto').agg({
        'ID': 'count',
        'Duración Estimada (s)': 'sum'
    }).reset_index().rename(columns={
        'ID': 'Total Detectado',
        'Duración Estimada (s)': 'Tiempo Total (s)'
    })

    # Timestamp actual para nombres de archivos
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Guardar archivos
    reporte_detalle_path = os.path.join(output_path, f"reporte_detallado_{timestamp}.csv")
    reporte_resumen_path = os.path.join(output_path, f"reporte_resumen_{timestamp}.csv")

    detalle.to_csv(reporte_detalle_path, index=False)
    resumen.to_csv(reporte_resumen_path, index=False)

    print(f"✅ Reporte detallado guardado: {reporte_detalle_path}")
    print(f"✅ Reporte resumen guardado: {reporte_resumen_path}")

    return reporte_detalle_path
