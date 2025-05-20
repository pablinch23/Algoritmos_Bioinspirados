# ----------------------------- Metadatos y referencias -----------------------------
# Python code of Gravitational Search Algorithm (GSA)
# Reference: Rashedi, Esmat, Hossein Nezamabadi-Pour, and Saeid Saryazdi. 
#            "GSA: a gravitational search algorithm." Information sciences 179.13 (2009): 2232-2248.
# Coded by: Mukesh Saraswat, Himanshu Mittal y Raju Pal (emails en cabecera original)
# Este archivo es el principal del algoritmo GSA para minimizar una función objetivo.

import numpy as np                           # Importa NumPy para cálculos numéricos y manejo de arreglos
import matplotlib.pyplot as plt              # Importa Matplotlib para graficar resultados
import seaborn as sns                        # Importa Seaborn para gráficos estadísticos (heatmap)
from sklearn.datasets import load_breast_cancer        # Carga un dataset de cáncer de mama de scikit-learn
from sklearn.model_selection import train_test_split   # Función para dividir datos en entrenamiento y prueba
from sklearn.neighbors import KNeighborsClassifier     # Importa el clasificador k-NN
from sklearn.metrics import accuracy_score             # Métrica de precisión para evaluar el modelo
from sklearn.decomposition import PCA                  # PCA para reducción de dimensión en animación
from matplotlib.animation import FuncAnimation, PillowWriter  # Para crear y guardar animaciones GIF

# ==============================
# Cargar dataset
# ==============================
data = load_breast_cancer()                # Carga el dataset y lo guarda en la variable `data`
X_data = data.data                         # Extrae las características (matriz n_samples × n_features)
y_data = data.target                       # Extrae las etiquetas (vector de clases)
num_features = X_data.shape[1]             # Calcula cuántas características hay (dimensión)

# Divide los datos en conjunto de entrenamiento (70%) y prueba (30%), con semilla fija
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.3, random_state=42
)

# ==============================
# Función de evaluación
# ==============================
def evaluate_solution(binary_vector):
    """
    Evalúa un agente (vector binario) entrenando k-NN
    con las características seleccionadas y mide precisión.
    """
    if np.count_nonzero(binary_vector) == 0:
        return 1.0                      # Si no selecciona ninguna característica, retorna peor fitness

    selected = np.where(binary_vector == 1)[0]  # Índices de características marcadas con 1
    clf = KNeighborsClassifier(n_neighbors=5)  # Crea un clasificador k-NN con k=5
    clf.fit(X_train[:, selected], y_train)     # Entrena el clasificador solo con características seleccionadas
    pred = clf.predict(X_test[:, selected])    # Predice sobre el conjunto de prueba
    acc = accuracy_score(y_test, pred)         # Calcula la precisión del clasificador

    penalty = np.count_nonzero(binary_vector) / len(binary_vector)  
    # Penaliza soluciones que usan muchas características
    return 1.0 - (acc - 0.01 * penalty)        # Return fitness: cuanto menor, mejor

# ==============================
# Algoritmo GSA con animación
# ==============================
def GSA_feature_selection(num_agents=30, max_iter=40, dim=num_features):
    # Inicializa posiciones binarias de agentes y velocidades
    X = np.random.randint(0, 2, (num_agents, dim))  # Población: matriz (agentes × características)
    V = np.zeros((num_agents, dim))                 # Velocidades iniciales = 0

    best_fitness = float('inf')          # Mejor fitness encontrado (infinito al inicio)
    best_position = None                 # Mejor posición (vector) hasta ahora

    history_fitness = []                 # Historial de mejor fitness por iteración
    history_num_features = []            # Historial de cuántas características usa la mejor posición
    feature_selection_counter = np.zeros(dim)  
    # Contador de cuántas veces cada característica fue seleccionada en el mejor global
    all_positions = []                   # Guarda todas las posiciones para la animación

    for t in range(max_iter):            # Bucle principal por número de iteraciones
        fitness = np.array([evaluate_solution(agent) for agent in X])
        # Calcula fitness de cada agente en la población

        best = np.min(fitness)           # Fitness más bajo (mejor) de la iteración
        worst = np.max(fitness)          # Fitness más alto (peor) de la iteración

        if best < best_fitness:          # Si se mejora el mejor global
            best_fitness = best
            best_position = X[np.argmin(fitness)].copy()  
            # Actualiza la mejor posición global

        eps = 1e-20                       # Epsilon para evitar divisiones por cero
        m = (fitness - worst) / (best - worst + eps)  
        # Calcula masa relativa de cada agente
        M = m / (np.sum(m) + eps)        # Normaliza masas para que sumen 1

        G = 100 * np.exp(-20 * t / max_iter)  
        # Calcula constante gravitacional decreciente

        a = np.zeros((num_agents, dim))  # Aceleraciones vacías
        for i in range(num_agents):      # Para cada agente i
            total_force = np.zeros(dim)
            for j in range(num_agents):  # Para cada agente j≠i
                if i != j:
                    R = np.linalg.norm(X[i] - X[j])  
                    # Distancia Euclidiana entre agentes i y j
                    force = G * M[i] * M[j] * (X[j] - X[i]) / (R + eps)
                    # Ley de gravitación modificada en espacio binario
                    total_force += np.random.rand() * force  
                    # Aleatoriza contribución de la fuerza
            a[i] = total_force / (M[i] + eps)  
            # Aceleración proporcional a la masa del agente

        V = np.random.rand(num_agents, dim) * V + a  
        # Actualiza velocidades con componente aleatorio
        S = 1 / (1 + np.exp(-V))               
        # Función sigmoide para convertir velocidades a probabilidades
        X = (S > np.random.rand(num_agents, dim)).astype(int)  
        # Nuevas posiciones: 1 si probabilidad > aleatorio, 0 si no

        feature_selection_counter += np.sum(X, axis=0)  
        # Actualiza contador de cuántas veces cada característica aparece
        history_fitness.append(best_fitness)  
        history_num_features.append(np.count_nonzero(best_position))
        all_positions.append(X.copy())  
        # Guarda estados para graficar y animar

        print(f"Iteración {t+1}/{max_iter} | Mejor fitness: {best_fitness:.4f} | Características: {np.count_nonzero(best_position)}")

    return best_position, 1.0 - best_fitness, history_fitness, history_num_features, feature_selection_counter, all_positions
    # Devuelve: mejor subconjunto, precisión asociada, historiales y datos de animación

# ==============================
# Ejecutar y graficar
# ==============================
if __name__ == "__main__":
    # Llama al algoritmo con parámetros por defecto
    best_subset, best_acc, fitness_hist, feature_hist, feature_usage, all_positions = GSA_feature_selection()

    selected = np.where(best_subset == 1)[0]  # Índices de características finalmente seleccionadas

    # Imprime resultados finales en consola
    print("\n===== RESULTADOS =====")
    print(f"Accuracy final: {best_acc:.4f}")
    print(f"Número de características seleccionadas: {len(selected)}")
    print(f"Índices seleccionados: {selected}")

    # 1. Graficar evolución del fitness
    plt.figure()
    plt.plot(fitness_hist, marker='o')
    plt.title("Mejor fitness por iteración")
    plt.xlabel("Iteración")
    plt.ylabel("Fitness (menor es mejor)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 2. Graficar número de características de la mejor solución
    plt.figure()
    plt.plot(feature_hist, marker='s', color='orange')
    plt.title("Número de características seleccionadas por iteración")
    plt.xlabel("Iteración")
    plt.ylabel("Características seleccionadas")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 3. Heatmap de frecuencia de uso de cada característica
    plt.figure(figsize=(10, 2))
    sns.heatmap(
        [feature_usage],
        cmap="YlGnBu",
        cbar=True,
        xticklabels=[f'F{i}' for i in range(num_features)],
        yticklabels=['Uso']
    )
    plt.title("Frecuencia de uso de cada característica")
    plt.xlabel("Características")
    plt.tight_layout()
    plt.show()

    # 4. Crear y guardar animación GIF del movimiento de partículas
    fig, ax = plt.subplots(figsize=(7, 6))
    pca = PCA(n_components=2)  # Reduce posiciones a 2D para visualizar

    def update(frame):
        ax.clear()
        reduced = pca.fit_transform(all_positions[frame])
        ax.scatter(reduced[:, 0], reduced[:, 1], c='purple', alpha=0.6, s=30)
        ax.set_title(f"Movimiento de partículas (Iteración {frame + 1})")
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.grid(True)

    anim = FuncAnimation(fig, update, frames=len(all_positions), interval=500)
    anim.save("gsa_movimiento.gif", writer=PillowWriter(fps=2))
    print("🎥 Animación guardada como 'gsa_movimiento.gif'")
