"""
Python code of Gravitational Search Algorithm (GSA)
Reference: Rashedi, Esmat, Hossein Nezamabadi-Pour, and Saeid Saryazdi. "GSA: a gravitational search algorithm." 
           Information sciences 179.13 (2009): 2232-2248.	
Coded by: Mukesh Saraswat (saraswatmukesh@gmail.com), Himanshu Mittal (emailid: himanshu.mittal224@gmail.com) and Raju Pal (emailid: raju3131.pal@gmail.com)
The code template used is similar given at link: https://github.com/7ossam81/EvoloPy and matlab version of GSA at mathworks.

Purpose: Main file of Gravitational Search Algorithm(GSA) 
            for minimizing of the Objective Function

Code compatible:
 -- Python: 2.* or 3.*
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from matplotlib.animation import FuncAnimation, PillowWriter

# ==============================
# Cargar dataset
# ==============================
data = load_breast_cancer()
X_data = data.data
y_data = data.target
num_features = X_data.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42)

# ==============================
# Función de evaluación
# ==============================
def evaluate_solution(binary_vector):
    if np.count_nonzero(binary_vector) == 0:
        return 1.0

    selected = np.where(binary_vector == 1)[0]
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train[:, selected], y_train)
    pred = clf.predict(X_test[:, selected])
    acc = accuracy_score(y_test, pred)

    penalty = np.count_nonzero(binary_vector) / len(binary_vector)
    return 1.0 - (acc - 0.01 * penalty)

# ==============================
# Algoritmo GSA con animación
# ==============================
def GSA_feature_selection(num_agents=30, max_iter=40, dim=num_features):
    X = np.random.randint(0, 2, (num_agents, dim))
    V = np.zeros((num_agents, dim))

    best_fitness = float('inf')
    best_position = None

    history_fitness = []
    history_num_features = []
    feature_selection_counter = np.zeros(dim)
    all_positions = []

    for t in range(max_iter):
        fitness = np.array([evaluate_solution(agent) for agent in X])

        best = np.min(fitness)
        worst = np.max(fitness)

        if best < best_fitness:
            best_fitness = best
            best_position = X[np.argmin(fitness)].copy()

        eps = 1e-20
        m = (fitness - worst) / (best - worst + eps)
        M = m / (np.sum(m) + eps)

        G = 100 * np.exp(-20 * t / max_iter)

        a = np.zeros((num_agents, dim))
        for i in range(num_agents):
            total_force = np.zeros(dim)
            for j in range(num_agents):
                if i != j:
                    R = np.linalg.norm(X[i] - X[j])
                    force = G * M[i] * M[j] * (X[j] - X[i]) / (R + eps)
                    total_force += np.random.rand() * force
            a[i] = total_force / (M[i] + eps)

        V = np.random.rand(num_agents, dim) * V + a
        S = 1 / (1 + np.exp(-V))
        X = (S > np.random.rand(num_agents, dim)).astype(int)

        feature_selection_counter += np.sum(X, axis=0)
        history_fitness.append(best_fitness)
        history_num_features.append(np.count_nonzero(best_position))
        all_positions.append(X.copy())

        print(f"Iteración {t+1}/{max_iter} | Mejor fitness: {best_fitness:.4f} | Características: {np.count_nonzero(best_position)}")

    return best_position, 1.0 - best_fitness, history_fitness, history_num_features, feature_selection_counter, all_positions

# ==============================
# Ejecutar y graficar
# ==============================
if __name__ == "__main__":
    best_subset, best_acc, fitness_hist, feature_hist, feature_usage, all_positions = GSA_feature_selection()
    selected = np.where(best_subset == 1)[0]

    print("\n===== RESULTADOS =====")
    print(f"Accuracy final: {best_acc:.4f}")
    print(f"Número de características seleccionadas: {len(selected)}")
    print(f"Índices seleccionados: {selected}")

    # 1. Fitness
    plt.figure()
    plt.plot(fitness_hist, marker='o')
    plt.title("Mejor fitness por iteración")
    plt.xlabel("Iteración")
    plt.ylabel("Fitness (menor es mejor)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 2. Número de características seleccionadas
    plt.figure()
    plt.plot(feature_hist, marker='s', color='orange')
    plt.title("Número de características seleccionadas por iteración")
    plt.xlabel("Iteración")
    plt.ylabel("Características seleccionadas")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 3. Heatmap de uso de características
    plt.figure(figsize=(10, 2))
    sns.heatmap([feature_usage], cmap="YlGnBu", cbar=True, xticklabels=[f'F{i}' for i in range(num_features)], yticklabels=['Uso'])
    plt.title("Frecuencia de uso de cada característica")
    plt.xlabel("Características")
    plt.tight_layout()
    plt.show()

    # 4. Animación de movimiento de partículas
    fig, ax = plt.subplots(figsize=(7, 6))
    pca = PCA(n_components=2)

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
