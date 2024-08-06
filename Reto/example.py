# Importar las bibliotecas necesarias
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from google.colab import files

# Cargar el archivo CSV desde la computadora local
uploaded = files.upload()

# Leer el archivo CSV
data = pd.read_csv(next(iter(uploaded)))

# Preprocesamiento de los datos
# Separar características (X) y la etiqueta (y)
X = data.drop(columns=['Afluencia_Turistas', 'Fecha'])
y = data['Afluencia_Turistas']

# Identificar las columnas categóricas y numéricas
categorical_features = ['Condicion_Climatica', 'Trafico_Transporte']
numeric_features = ['Evento_Importante', 'Festividad', 'Reservas_Alojamiento']

# Crear transformadores para datos numéricos y categóricos
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder()

# Crear un ColumnTransformer para aplicar las transformaciones adecuadas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de regresión lineal
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression())])

# Entrenar el modelo
model.fit(X_train, y_train)

# Realizar predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
