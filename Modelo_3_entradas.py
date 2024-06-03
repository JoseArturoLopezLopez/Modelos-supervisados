# Reestructuracion del codigo
# ********** Liberias necesarias **********
# Manejo de archivos
import os
import csv
import shutil
import pandas

# Manejo del ente
import tensorflow as tf
import numpy as np
# Dice type: ignore porque no quiero que me marque lo de que no se usa de esta manera :[
from tensorflow.keras.layers import Input, Dense # type: ignore
from tensorflow.keras.models import Model # type: ignore

# Graficacion
import matplotlib.pyplot as plt

# Configuraciones para control de datos
dir_entrada = "entradas"; # Debe de existir
dir_salida = "datos" # Existir es opcional
normalizacion = 10 # a= 1, b=2, c=3,d = 4, e = 5,f = 6, g = 7, h = 8, i = 9, j = 10
epocas = 50
entrenamiento = 0.80
outlayers = 0.1
size = [13, 6]
entradas = ['NH3-N', 'TN', 'TKN']

# Directorios y archivos

# Crear directorio de salidas en caso de no existir

if not os.path.exists(dir_salida):
    os.makedirs(dir_salida)
    
# Crear una copia del archivo de entradas para evitar reescribirlo
shutil.copyfile(os.path.join(dir_entrada, "entrada.csv"), os.path.join(dir_salida, "copia.csv"))

# Crear acceso limpio a los documentos con sus rutas y preparar los archivos para su uso
archivo_entrada = os.path.join(dir_salida, "copia.csv")
archivo_salida = os.path.join(dir_salida, "normalizado.csv")

# Abrir el archivo de entrada para lectura y el archivo de salida para escritura para eliminar las filas que contenga almenos un elemento en blanco
with open(archivo_entrada, mode='r', newline='') as file_in, \
     open(archivo_salida, mode='w', newline='') as file_out:

    # Crear objetos para leer y escribir archivos CSV
    reader = csv.reader(file_in)
    writer = csv.writer(file_out)

    # Iterar sobre cada fila en el archivo de entrada
    for row in reader:
        # Verificar si hay algún cuadro vacío en las columnas de a hasta j
        if any(cell == '' for cell in row[0:normalizacion]):
            # Si hay algún cuadro vacío, no escribir esta fila en el archivo de salida
            continue
        # Si no hay cuadros vacíos, escribir la fila en el archivo de salida
        writer.writerow(row)
        
# Normalizar los datos con la manera del profe (dato / valor absoluto (Valor mayor - Valor Menor))
# Leer el archivo CSV excluyendo la primera línea
df = pandas.read_csv(archivo_salida)

# Seleccionar solo las columnas de la 'a' a la 'j' que son las que tienen informacion (Si no esto sera eterno :/)
columnas_a_normalizar = df.columns[0:10]  

# Iterar sobre las columnas seleccionadas y normalizar manualmente cada dato
for columna in columnas_a_normalizar:
    max_valor = df[columna].max()
    min_valor = df[columna].min()
    # Iterar sobre cada dato de la columna
    for i in range(len(df[columna])):
        dato = df.at[i, columna]
        df.at[i, columna] = dato / abs(max_valor - min_valor)

#revolver los datos
df = df.sample(frac=1, random_state=42);

# Guardar el DataFrame con los datos normalizados en un nuevo archivo CSV
df.to_csv(archivo_salida, index=False)

# Tenemos 5 entradas asi que tomamos los valores de los datos de las primeras 5 columnas
datos_entrada = df[entradas].values

# Nos toco salida sectima asi que obtenemos sus datos
datos_salida = df['WATER TEMP.'].values

# Dividir los datos en Training set y Validation set (70% y 30% respectivamente creo)
cantidad_datos = len(df)
entrenamiento = int(cantidad_datos * entrenamiento)

training_input = datos_entrada[:entrenamiento] # Asignar porcentaje definido hacia atras
training_output = datos_salida[:entrenamiento] # Asignar porcentaje definido hacia atras
validation_input = datos_entrada[entrenamiento:] # Asignar porcentaje restante hacia adelante
validation_output = datos_salida[entrenamiento:] # Asignar porcentaje restante hacia adelante

# Convertir los valores a tensores (Cuerdas creo)
inputs_np = np.array(training_input, dtype=np.float32)
outputs_np = np.array(training_output, dtype=np.float32)

train_inputs = tf.convert_to_tensor(inputs_np, dtype=tf.float32)
train_outputs = tf.convert_to_tensor(outputs_np, dtype=tf.float32)

inputs_np = np.array(validation_input, dtype=np.float32)
outputs_np = np.array(validation_output, dtype=np.float32)

val_inputs = tf.convert_to_tensor(inputs_np, dtype=tf.float32)
val_outputs = tf.convert_to_tensor(outputs_np, dtype=tf.float32)


# Definir el modelo
input_layer = Input(shape=(len(entradas),))
hidden_layer = Dense(16, activation='relu')(input_layer)
hidden_layer = Dense(8, activation='relu')(hidden_layer)
output_layer = Dense(1, activation='linear')(hidden_layer)  # Dos salidas lineales

model = Model(inputs=input_layer, outputs=output_layer)

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo con datos de entrenamiento y validación, ademas de almacenar el historial de entrenamiento
history = model.fit(train_inputs, train_outputs, epochs=epocas,validation_split=0.2, batch_size=32, verbose=1)

# Evaluar el modelo con los datos de validación
loss = model.evaluate(train_inputs, train_outputs)
print(f'Loss en el conjunto de validación: {loss}')

# Predecir con nuevas entradas
predicciones = model.predict(val_inputs)

print('Predicciones:', predicciones)
print('Valores reales: ', val_outputs)

filtered_predicciones = []
filtered_val_outputs = []

for pred, val in zip(predicciones, val_outputs):
    if abs(pred - val) <= outlayers:
        filtered_predicciones.append(pred)
        filtered_val_outputs.append(val)

# Crear una figura y dos subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(size[0], size[1]))

# Obtener la pérdida  del historial de entrenamiento
loss = history.history['loss']

# Graficar la pérdida en el primer subplot
epochs = range(1, len(loss) + 1)
ax1.plot(epochs, loss, 'b', label='Pérdida de entrenamiento')
# ax1.plot(epochs, history.history['val_loss'], 'r', label='Pérdida de validación')
ax1.set_title('Pérdida durante entrenamiento del modelo')
ax1.set_xlabel('Épocas')
ax1.set_ylabel('Pérdida')
ax1.legend()
# Graficar las salidas esperadas y las predicciones en el segundo subplot
print(f"1: {len(filtered_predicciones)}, 2: {len(filtered_val_outputs)}")
ax2.plot(range(1, len(filtered_val_outputs) + 1), filtered_val_outputs, 'bo-', label='Salida esperada')
ax2.plot(range(1, len(filtered_predicciones) + 1), filtered_predicciones, 'ro-', label='Predicciones')
ax2.set_title('Salidas esperadas y predicciones')
ax2.set_xlabel('Muestras')
ax2.set_ylabel('Valor')
ax2.legend()

# Ajustar los subplots para evitar solapamiento
plt.tight_layout()

# Mostrar la figura con los dos subplots
plt.show()
