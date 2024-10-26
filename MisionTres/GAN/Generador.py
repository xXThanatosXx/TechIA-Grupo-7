import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Simulación de datos
n = 1000  # Número de días de datos simulados

# Fechas
start_date = datetime(2020, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(n)]

# Consumo diario en metros cúbicos (simulado)
consumo_diario_m3 = np.random.randint(1000, 5000, size=n)

# Temperatura media en grados Celsius (simulado)
temperatura_media_C = np.random.uniform(15, 40, size=n)

# Precipitación diaria en milímetros (simulado)
precipitacion_mm = np.random.uniform(0, 20, size=n)

# Humedad relativa en porcentaje (simulado)
humedad_relativa_ = np.random.uniform(30, 90, size=n)

# Día de la semana
dia_semana = [date.weekday() for date in dates]

# Eventos especiales (binario, 1 si hay evento, 0 si no)
eventos_especiales = [random.choice([0, 0, 0, 1]) for _ in range(n)]

# Nivel del reservorio en metros (simulado)
nivel_reservorio_metros = np.random.uniform(5, 15, size=n)

# Población aproximada en el sector (simulado)
poblacion_sector = np.random.randint(5000, 15000, size=n)

# Tarifa de agua por m³ (simulado)
tarifa_agua = np.random.uniform(0.5, 2.0, size=n)

# Racionamiento previo (binario, 1 si hubo racionamiento, 0 si no)
racionamiento_previo = [random.choice([0, 1]) for _ in range(n)]

# Racionamiento necesario (target) - Simulado basado en consumo y nivel del reservorio
racionamiento_necesario = [1 if consumo_diario_m3[i] > 4000 and nivel_reservorio_metros[i] < 7 else 0 for i in range(n)]

# Crear el DataFrame
data = {
    'Fecha': dates,
    'Consumo_diario_m3': consumo_diario_m3,
    'Temperatura_media_°C': temperatura_media_C,
    'Precipitación_mm': precipitacion_mm,
    'Humedad_relativa_%': humedad_relativa_,
    'Día_semana': dia_semana,
    'Eventos_especiales': eventos_especiales,
    'Nivel_reservorio_metros': nivel_reservorio_metros,
    'Población_sector': poblacion_sector,
    'Tarifa_agua': tarifa_agua,
    'Racionamiento_previo': racionamiento_previo,
    'Racionamiento_necesario': racionamiento_necesario
}

df = pd.DataFrame(data)

# Mostrar las primeras filas del DataFrame
print(df.head())

# Guardar el DataFrame generado en un archivo CSV
file_path = 'consumo_agua_racionamiento.csv'
df.to_csv(file_path, index=False)

print(f"Archivo CSV guardado como: {file_path}")