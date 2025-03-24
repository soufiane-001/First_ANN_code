# First_ANN_code
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

L=10
np.random.seed(42)
theta_degrees=np.linspace(0,180,100)
theta_radians=np.radians(theta_degrees)
x=L*np.cos(theta_radians)
X_train, X_test, y_train, y_test = train_test_split(theta_radians.reshape(-1,1), x, test_size=0.2, random_state=42)
model=MLPRegressor(hidden_layer_sizes=(10,10),activation='relu',max_iter=5000,random_state=42)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
print(f"Mean Squared Error: {mse:.5f}")
plt.scatter(np.degrees(X_test), y_pred, label="Prédictions", color="red", marker='x')
plt.scatter(theta_degrees,x,label='données réelles',color="blue",alpha=0.5)
plt.xlabel("Theta (degrés)")
plt.ylabel("X (cm)")
plt.legend()
plt.title("Prédiction de X en fonction de Theta")
plt.show()
#print(x)
#plt.figure(figsize=(10, 5))
#plt.plot(theta_degrees, x, color='blue', linewidth=2, label=r'$x = L \cdot \cos(\theta)$')

