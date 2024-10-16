#   Codigo que implementa el metodo de gradiente 
#   para minimizar una funcion de una variable
# 
#           Autor:
#   Dr. Ivan de Jesus May-Cen
#   imaycen@hotmail.com
#   Version 1.0 : 16/10/2024
#
import numpy as np
import matplotlib.pyplot as plt

# Función objetivo f(x) = (x-3)^2
def objective_function(x):
    return (x - 3)**2

# Gradiente de la función objetivo: f'(x) = 2*(x-3)
def gradient(x):
    return 2 * (x - 3)

# Parámetros de la optimización
learning_rate = 0.1  # Tasa de aprendizaje
max_iters = 50  # Número máximo de iteraciones
tolerance = 1e-6  # Tolerancia para detener el algoritmo

# Inicialización
x = np.random.randn()  # Punto inicial aleatorio
history = []  # Para guardar el valor de la función en cada iteración
x_values = []  # Para guardar los valores de x

# Método del gradiente descendente
for i in range(max_iters):
    # Guardamos el valor de la función objetivo y el valor de x
    history.append(objective_function(x))
    x_values.append(x)
    
    # Calculamos el gradiente
    grad = gradient(x)
    
    # Actualizamos el valor de x
    x_new = x - learning_rate * grad
    
    # Si la diferencia es menor que la tolerancia, terminamos
    if np.abs(x_new - x) < tolerance:
        print(f"Convergencia alcanzada en {i+1} iteraciones.")
        break
    
    x = x_new

# Si no alcanzamos convergencia
if i == max_iters - 1:
    print("Se alcanzó el número máximo de iteraciones.")

# Mostrar resultados finales
print(f"Valor mínimo encontrado: x = {x:.4f}")
print(f"Valor de la función objetivo en el mínimo: f(x) = {objective_function(x):.4f}")

# Graficar la convergencia
plt.figure(figsize=(10, 5))

# Subplot 1: Convergencia de la función objetivo
plt.subplot(1, 2, 1)
plt.plot(history, marker='o')
plt.title("Convergencia del valor de la función objetivo")
plt.xlabel("Iteraciones")
plt.ylabel("f(x)")
plt.grid(True)

# Subplot 2: Función objetivo con la ubicación de la solución
x_range = np.linspace(-2, 8, 100)  # Rango de x para graficar la función
y_range = objective_function(x_range)

plt.subplot(1, 2, 2)
plt.plot(x_range, y_range, label="f(x) = (x-3)^2")
plt.scatter(x_values, objective_function(np.array(x_values)), color='red', label="Iteraciones")
plt.scatter(x, objective_function(x), color='green', zorder=5, label=f"Solución: x = {x:.4f}")
plt.title("Función objetivo y ubicación de la solución")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)

# Mostrar gráficas
plt.tight_layout()
plt.savefig("convergencia-y-solucion-1-variable.eps")
plt.show()


