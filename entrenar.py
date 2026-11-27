import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#    Leer y Limpiar
df = pd.read_csv("sensores_planta.csv").dropna()

#    Variables de entrada y objetivo
columnas = ["temperatura_c", "vibracion_mms", "horas_operacion", "presion_psi"]
X = df[columnas]
y = df["estado"]

#    Dividir 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#    Entrenar
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

#    Evaluar
y_pred = modelo.predict(X_test)
precision = accuracy_score(y_test, y_pred)
print(f"🎯 Precisión del modelo (Accuracy): {precision * 100:.2f}%")

#    Exportar
joblib.dump(modelo, "modelo_mantenimiento.pkl")
print("✅ Modelo guardado como modelo_mantenimiento.pkl")