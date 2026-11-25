import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay


df = pd.read_csv('./base_ajustada.csv')
cols = df.columns.values
X = df[cols[:-1]]
y = df['target']


X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=30, stratify=y)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=30)


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("--- Divisão dos Dados ---")
print(f"Amostras de Treino: {X_train_scaled.shape[0]} (70%)")
print(f"Amostras de Validação: {X_val_scaled.shape[0]} (15%)")
print(f"Amostras de Teste: {X_test_scaled.shape[0]} (15%)")


# y_train = y_train - 1
# y_val = y_val - 1
# y_test = y_test - 1

model = Sequential()

model.add(Input(shape=(X_train_scaled.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='softmax'))

print("\n--- Arquitetura da Rede Neural ---")
model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=20,
                           restore_best_weights=True)


model.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy'],
)

print("\n--- Iniciando o Treinamento ---")

history = model.fit(
    X_train_scaled, 
    y_train,
    validation_data=(X_val_scaled, y_val), 
    epochs=100,      
    batch_size=32,
    callbacks=[early_stop]
)

# --- ETAPA 8: Validação (Curvas de Aprendizado) ---

plt.figure(figsize=(12, 5))


plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='--')
plt.title('Train vs Validation Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linestyle='--')
plt.title('Train vs Validation Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("aprendizado.png")

# --- ETAPA 9: Avaliação Final ---

print("\n--- Realizando Previsões nos Dados de Teste ---")
y_pred_prob = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_prob, axis=1)


print("\n--- Relatório de Classificação  ---")

print(classification_report(y_test, y_pred))

plt.figure(figsize=(12, 10))
disp = ConfusionMatrixDisplay.from_predictions(
    y_test, 
    y_pred, 
    cmap='Blues', 
    colorbar=True
)
disp.ax_.set_title('Matriz de Confusão - Dados de Teste (15%)')
plt.savefig("confusao.png")