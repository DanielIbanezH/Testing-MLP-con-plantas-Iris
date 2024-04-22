import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, LeaveOneOut, LeavePOut, cross_val_score
from sklearn.metrics import accuracy_score

# Cargar los datos del archivo CSV
data = pd.read_csv("irisbin.csv", header=None)

# Dividir las características (entradas) y las etiquetas (salidas)
X = data.iloc[:, :-3].values
y = data.iloc[:, -3:].values

# Mapear las etiquetas binarias a números enteros (0, 1, 2)
y_mapped = np.argmax(y, axis=1)
print(y_mapped)
# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_mapped, test_size=0.2, random_state=42)

# Construir y entrenar el modelo de perceptrón multicapa
model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Evaluar el modelo en el conjunto de prueba
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on test set:", accuracy)

# Validación cruzada leave-one-out
loo = LeaveOneOut()
loo_scores = []
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_mapped[train_index], y_mapped[test_index]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    loo_scores.append(accuracy_score(y_test, y_pred))
loo_avg_accuracy = np.mean(loo_scores)
loo_std_accuracy = np.std(loo_scores)
print("Leave-One-Out Cross Validation - Average Accuracy:", loo_avg_accuracy)
print("Leave-One-Out Cross Validation - Standard Deviation of Accuracy:", loo_std_accuracy)


# Validación cruzada leave-p-out (p = 2)
lpo = LeavePOut(p=2)
lpo_scores = cross_val_score(model, X, y_mapped, cv=lpo, n_jobs=-1)  # Utiliza todos los núcleos de CPU disponibles
lpo_avg_accuracy = np.mean(lpo_scores)
lpo_std_accuracy = np.std(lpo_scores)
print("Leave-P-Out Cross Validation (P = 2) - Average Accuracy:", lpo_avg_accuracy)
print("Leave-P-Out Cross Validation (P = 2) - Standard Deviation of Accuracy:", lpo_std_accuracy)
