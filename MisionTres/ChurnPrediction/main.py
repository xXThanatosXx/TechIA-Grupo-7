import pandas as pd
from data_processing import preprocess_data
from model import build_and_train_model
from evaluation import evaluate_model
import config
from joblib import dump, load

def main():
    # Cargar datos desde un archivo Excel
    data = pd.read_excel(config.DATA_PATH)
    
    # Preprocesar los datos
    X_train, X_test, y_train, y_test = preprocess_data(data)
    
    # Construir y entrenar el modelo
    model = build_and_train_model(X_train, y_train)
    
    # Evaluar el modelo
    evaluate_model(model, X_test, y_test, X_train, y_train)

    # Guardar el modelo
    dump(model, '.\\MisionTres\\ChurnPrediction\\random_forest_model.joblib')

    # # Cargar el modelo
    # loaded_model = load('random_forest_model.joblib')


    
if __name__ == "__main__":
    main()
