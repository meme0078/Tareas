import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# CARGA DE DATOS
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# CODIFICACIÓN CATEGÓRICA
cat_cols = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']
encoder = LabelEncoder()

for col in cat_cols:
    train[col] = encoder.fit_transform(train[col].astype(str))
    test[col] = test[col].apply(lambda x: x if x in encoder.classes_ else encoder.classes_[0])
    test[col] = encoder.transform(test[col].astype(str))

# PREPARACIÓN PARA ENTRENAMIENTO
X = train.drop(['Transported', 'PassengerId', 'Name'], axis=1)
y = train['Transported'].astype(int)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ENTRENAMIENTO
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# EVALUACIÓN
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Precisión en validación: {accuracy:.4f}")

# MATRIZ DE CONFUSIÓN
cm = confusion_matrix(y_val, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap='Blues')
plt.title("Matriz de Confusión (Validación)")
plt.savefig("matriz_confusion.png")
plt.close()

# IMPORTANCIA DE CARACTERÍSTICAS
importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
plt.barh(features, importances)
plt.xlabel("Importancia")
plt.title("Importancia de Características del Modelo")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("importancia_caracteristicas.png")
plt.close()

# PREDICCIÓN SOBRE TEST
X_test = test.drop(['PassengerId', 'Name'], axis=1)
predictions = model.predict(X_test)

# DISTRIBUCIÓN DE CLASES PREDICHAS
plt.figure()
plt.hist(predictions, bins=2, edgecolor='black')
plt.xticks([0, 1], ['No Transported', 'Transported'])
plt.title("Distribución de Clases Predichas en Test")
plt.xlabel("Clase")
plt.ylabel("Frecuencia")
plt.savefig("distribucion_predicciones.png")
plt.close()

# CREAR ARCHIVO DE SUBMISIÓN
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Transported': predictions
})

submission.to_csv('submission.csv', index=False)
print("Archivo 'submission.csv' guardado correctamente.")
print("Gráficas guardadas: matriz_confusion.png, importancia_caracteristicas.png, distribucion_predicciones.png")
