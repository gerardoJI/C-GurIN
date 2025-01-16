# estados.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Variables globales
sensor_names = ['co_2000', 'co_1950', 'co_1900', 'co_1850', 'co_1810', 'co_1750']
ventilator_names = ['v1', 'v2', 'v3']
columns = sensor_names + ventilator_names + ['timestamp']

df_all = pd.DataFrame(columns=columns)  # DataFrame global
last_update = {}  # Diccionario de última actualización
ventilator_states = {'v1': 1, 'v2': 1, 'v3': 1}  # Estado inicial de los ventiladores
ventilator_off_time = {'v1': None, 'v2': None, 'v3': None}  # Hora en que se apagan los ventiladores

# Función para generar los valores de los sensores de CO
def generate_co_values():
    prob = np.random.rand()
    if prob < 0.85:
        return np.round(np.random.uniform(0.01, 0.05), 4)
    else:
        return np.round(np.random.choice([0.00, 0.10]), 4)

# Función para generar los valores de los ventiladores
def generate_ventilator_values():
    global ventilator_states, ventilator_off_time
    current_time = datetime.now()
    for ventilator in ventilator_names:
        if ventilator_states[ventilator] == 0:
            if current_time - ventilator_off_time[ventilator] >= timedelta(minutes=30):
                ventilator_states[ventilator] = 1
                ventilator_off_time[ventilator] = None
        else:
            if np.random.rand() < 0.08:
                ventilator_states[ventilator] = 0
                ventilator_off_time[ventilator] = current_time
    return list(ventilator_states.values())

# Esta es la función que actualiza last_update y df_all
def update_and_return_data():
    global df_all, last_update, ventilator_states, ventilator_off_time
    
    # Generar los valores de los sensores y ventiladores
    co_values = [generate_co_values() for _ in sensor_names]
    ventilator_values = generate_ventilator_values()
    
    # Obtener la fecha y hora de la actualización
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Crear una fila con los datos generados
    data = co_values + ventilator_values + [timestamp]
    
    # Agregar la nueva fila al DataFrame global
    new_row = pd.DataFrame([data], columns=columns)
    global df_all
    df_all = pd.concat([df_all, new_row], ignore_index=True)
    
    # Actualizar el diccionario last_update con los últimos valores
    global last_update
    last_update = {columns[i]: data[i] for i in range(len(columns))}
    
    return df_all, last_update

# Función para obtener la última actualización
def get_last_update():
    return last_update

# Función para obtener el historial completo de datos
def get_df_all():
    return df_all