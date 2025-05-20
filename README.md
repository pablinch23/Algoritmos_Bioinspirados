# Algoritmos_Bioinspirados

Este script implementa el Gravitational Search Algorithm (GSA) para el problema de selección de características en un conjunto de datos de cáncer de mama. Su función principal es hallar un subconjunto de variables que maximice la precisión de un clasificador k-NN (k = 5), minimizando simultáneamente el número de características (mediante una penalización en el fitness).

¿Cómo funciona?

Representación: cada agente es un vector binario que indica qué características usar.

Evaluación: para cada vector, se entrena un k-NN y se mide la precisión; el fitness combina precisión y tamaño del subconjunto.

Gravitación: agentes con mejor fitness tienen “mayor masa” y atraen a otros, modelando fuerzas gravitacionales en el espacio de soluciones.

Actualización: posiciones y velocidades se actualizan con ruido y función sigmoide para mantener valores binarios
