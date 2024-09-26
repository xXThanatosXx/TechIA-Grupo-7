import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score
import config
from data_processing import preprocess_data

# Cargar los datos de prueba (ajusta la ruta y el nombre del archivo de acuerdo a tus datos)
data = pd.read_excel(config.DATA_PATH)

# Preprocesar los datos
X_train, X_test, y_train, y_test = preprocess_data(data)

# Cargar el modelo guardado
modelo = joblib.load('.\\MisionTres\\ChurnPrediction\\random_forest_model.joblib')
print("Modelo cargado correctamente")

# Hacer predicciones con el modelo cargado
predicciones = modelo.predict(X_test)

# Evaluar las predicciones
accuracy = accuracy_score(y_test, predicciones)
print(f"Exactitud del modelo cargado: {accuracy:.2f}")

# Calcular la matriz de confusión
matriz_confusion = confusion_matrix(y_test, predicciones)
print("Matriz de confusión:")
print(matriz_confusion)

# Calcular el AUC
proba_predicciones = modelo.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, proba_predicciones)
print(f"AUC: {auc:.2f}")

# Calcular el F1 Score
f1 = f1_score(y_test, predicciones)
print(f"F1 Score: {f1:.2f}")
