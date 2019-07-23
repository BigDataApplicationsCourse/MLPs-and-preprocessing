# Creación de una MLP en Keras, utilizando sequential

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy

# fijamos la semilla random para reproducibilidad
seed = 7
numpy.random.seed(seed)

# Cargamos el dataset Pima Indians - sobre diabetis

dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

# separamos los datos en variables de entrada (X) y en salida (Y)

X = dataset[:,0:8]
Y = dataset[:,8]

# dividimos el data set en entrenamiento y prueba

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)

# creamos el modelo

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# Compilamos el model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# entrenamos el modelo (Fit) con todos los datos

#model.fit(X, Y, epochs=150, batch_size=10)

# Entrenamos el modelo con el 20% para validación

#model.fit(X, Y, validation_split=0.20, epochs=10, batch_size=10)

# Entrenamos el modelo definiendo manualmente el set de validación

#model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=150, batch_size=10)

# evaluamos el modelo y extraemos las métricas

scores = model.evaluate(X_train, y_train)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# con el modelo entrenado ya podemos realizar predicciones.
# para predicciones binarias

#predictions = model.predict(X_test)

#predicciones multiclase

predictions = model.predict_classes(X_test)

print ('Predicciones ', predictions)
