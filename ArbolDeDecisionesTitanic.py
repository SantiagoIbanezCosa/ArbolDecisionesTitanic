import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import pandas as pd
import os

def check_file_exists(path):
    if os.path.exists(path):
        return True
    else:
        print(f"Archivo no encontrado: {path}")
        return False

def save_predictions(output_path, passenger_ids, predictions):
    output = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': predictions})
    output.to_csv(output_path, index=False)
    print(f"Predicciones guardadas en {output_path}")

def preprocess_data(df):
    df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    return df

# Cargar el conjunto de datos del Titanic
train_file_path = 'C:\\Users\\Santiago\\Desktop\\PruebasTecnias\\titanic\\train.csv'
test_file_path = 'C:\\Users\\Santiago\\Desktop\\PruebasTecnias\\titanic\\test.csv'

if check_file_exists(train_file_path) and check_file_exists(test_file_path):
    train_data = pd.read_csv(train_file_path)
    test_data = pd.read_csv(test_file_path)

    train_data = preprocess_data(train_data)
    test_data = preprocess_data(test_data)

    X_train = train_data.drop('Survived', axis=1)
    y_train = train_data['Survived']
    X_test = test_data

    # Asegurarse de que todos los datos sean numéricos
    X_train = X_train.apply(pd.to_numeric)
    X_test = X_test.apply(pd.to_numeric)

    # Ajuste de hiperparámetros
    param_grid = {'max_depth': [3, 5, 7, 10], 'min_samples_split': [2, 5, 10]}
    grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_clf = grid_search.best_estimator_

    # Validación cruzada
    scores = cross_val_score(best_clf, X_train, y_train, cv=5)
    print(f"Puntuaciones de validación cruzada: {scores}")
    print(f"Puntuación media de validación cruzada: {scores.mean()}")

    # Entrenar el clasificador de árbol de decisión con los mejores parámetros
    best_clf.fit(X_train, y_train)

    # Predecir la respuesta para el conjunto de datos de prueba
    y_pred = best_clf.predict(X_test)

    # Guardar las predicciones en un archivo CSV
    save_predictions('C:\\Users\\Santiago\\Desktop\\PruebasTecnias\\titanic\\predictions.csv', test_data['PassengerId'], y_pred)