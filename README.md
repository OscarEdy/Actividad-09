[README.md](https://github.com/user-attachments/files/32635511/README.md)
# Reducción de Dimensiones y Detección de Anomalías

**Maestría en Ciencia de Datos**

**Integrantes:** Oscar Edy Vilca Quispe · Jose Jhonanatan Quispe Cartajena

Este repositorio contiene la resolución de tres ejercicios sobre reducción de dimensiones y detección de anomalías: un sistema de reconocimiento facial con Eigenfaces (PCA), un recomendador de artículos basado en NMF, y un modelo ensemble para detectar anomalías en series temporales.

**Notebook de Google Colab:** [`Eigenfaces_NMF_Anomalias.ipynb`](./Eigenfaces_NMF_Anomalias.ipynb)

---

## 1. Eigenfaces — Reconocimiento facial con PCA

### Problema

Construir un sistema de reconocimiento de rostros usando el dataset **Olivetti Faces** de `scikit-learn` (400 imágenes en escala de grises de 64×64 píxeles, correspondientes a 40 personas), sin emplear redes neuronales.

### Enfoque

El enfoque clásico de **Eigenfaces** (Turk & Pentland) consiste en:

1. **Representar cada imagen como un vector** de 4096 valores (64×64 aplanado).
2. **Aplicar PCA** sobre el conjunto de entrenamiento para encontrar las direcciones de máxima varianza del espacio de rostros. Los componentes principales, reinterpretados como imágenes de 64×64, son las *eigenfaces*: patrones de variación facial (iluminación, forma de ojos, contorno, etc.).
3. **Proyectar** cada rostro (train y test) a ese espacio de baja dimensión (150 componentes en lugar de 4096 píxeles), lo que retiene ~90-95% de la varianza total.
4. **Clasificar** los vectores proyectados con un modelo clásico — se usó **SVM con kernel RBF** — para no violar la restricción de no usar redes neuronales.

### Pasos implementados en el notebook

- Carga de `fetch_olivetti_faces` y partición estratificada 75/25 (train/test) por persona.
- Ajuste de `PCA(n_components=150, whiten=True)` solo sobre el train (se evita fuga de información del test).
- Visualización de la cara promedio y las primeras eigenfaces.
- Proyección de train/test al espacio PCA y entrenamiento de un `SVC(kernel='rbf')`.
- Evaluación con **accuracy** y **matriz de confusión**.
- Reconstrucción de rostros de test a partir de sus 150 componentes, para verificar cuánta información visual conserva la compresión.

### Resultado esperado

Con 150 componentes (~93% de varianza explicada) y SVM-RBF, este pipeline típicamente alcanza una **accuracy entre 0.94 y 0.98** sobre el set de test (40 clases, 10 imágenes por persona) — comparable al desempeño de una red convolucional simple, pero usando únicamente álgebra lineal (PCA) y un clasificador de márgenes (SVM). La reconstrucción muestra que, aun descartando ~96% de las dimensiones originales (de 4096 a 150), el rostro sigue siendo reconocible visualmente, lo que confirma que la mayor parte de la información relevante está concentrada en pocas direcciones principales.

---

## 2. Recomendador de artículos con NMF

### Problema

Dado un artículo que un lector está consultando, recomendar otros artículos con temática similar — caso típico de un sistema de recomendación de contenido editorial.

### Enfoque

1. **Vectorización TF-IDF** de cada artículo (dataset `20 Newsgroups` de `scikit-learn`, ~18 000 artículos de noticias reales agrupados en 20 categorías temáticas), limitando el vocabulario a las 5000 palabras más informativas y filtrando términos demasiado frecuentes o demasiado raros.
2. **NMF (Non-negative Matrix Factorization)** sobre la matriz TF-IDF para reducir la dimensión de miles de términos a **20 tópicos latentes**. A diferencia de PCA, NMF impone no-negatividad, lo que produce factores interpretables como "mezclas de tópicos" (cada artículo es una combinación no-negativa de tópicos, y cada tópico una combinación no-negativa de palabras).
3. **Similitud de coseno** entre los vectores de tópicos (`W`, de dimensión artículo × 20) para medir cercanía temática entre artículos, en vez de comparar directamente los miles de términos originales.
4. Función `recomendar(idx, top_n)` que, dado el índice de un artículo, devuelve los `top_n` artículos con mayor similitud de coseno en el espacio de tópicos.

### Resultado esperado

Al inspeccionar los tópicos obtenidos, cada uno agrupa términos coherentes (por ejemplo, términos de deportes, de política, de tecnología, de religión, etc., dado que 20 Newsgroups cubre justamente esas categorías). Al pedir recomendaciones para un artículo de una categoría dada, los artículos recomendados pertenecen mayoritariamente a la **misma categoría real** (`news.target`), aunque esa etiqueta nunca se usa para entrenar el modelo — es puramente un chequeo de validación. Esto confirma que la reducción de dimensión con NMF preserva la señal temática relevante para la recomendación.

---

## 3. Detección de anomalías en series temporales (ensemble)

### Problema

Entrenar un modelo ensemble que detecte anomalías (valores atípicos puntuales) dentro de una serie temporal.

### Enfoque

Se generó una serie temporal sintética con un patrón cíclico (simulando estacionalidad diaria) más ruido gaussiano, y se inyectaron 40 anomalías puntuales (picos/caídas fuera de rango) en posiciones conocidas, lo que permite **evaluar objetivamente** el desempeño de los detectores (ground truth disponible).

**Feature engineering:** cada punto se representa con 3 características: su valor, la media móvil y la desviación estándar móvil de una ventana de 24 pasos anteriores. Esto captura el contexto temporal reciente sin usar modelos secuenciales complejos.

**Ensemble de 3 detectores no supervisados, con supuestos distintos y complementarios:**

| Modelo | Supuesto | Cómo detecta |
|---|---|---|
| **Isolation Forest** | Las anomalías son fáciles de "aislar" con pocos cortes aleatorios | Puntúa según profundidad promedio de aislamiento en árboles aleatorios |
| **Local Outlier Factor (LOF)** | Las anomalías tienen densidad local mucho menor que sus vecinos | Compara densidad local de cada punto contra la de sus vecinos |
| **One-Class SVM** | Los puntos normales están dentro de una frontera de soporte aprendida | Puntúa la distancia a la frontera de la clase "normal" |

**Fusión (ensemble):** cada detector produce un *score* de anomalía; los tres scores se normalizan a `[0,1]` con `MinMaxScaler` y se **promedian**. Se marca como anomalía todo punto cuyo score promedio supere el percentil 98. Se optó por fusión de scores en lugar de voto por mayoría simple, porque promediar scores normalizados aprovecha *cuán* anómalo considera cada modelo a un punto, no solo si lo cruza o no un umbral individual.

### Resultado esperado

Se reporta el F1-score de cada detector individual y del ensemble frente a las anomalías reales inyectadas. En la práctica:

- Cada detector individual tiene puntos ciegos distintos (LOF es más sensible a anomalías por densidad local, Isolation Forest a valores extremos globales, One-Class SVM a desviaciones de la forma general de la distribución).
- El ensemble por fusión de scores generalmente iguala o se acerca al mejor modelo individual en F1, pero es **más robusto entre distintas corridas/datasets**, porque no depende de que un único supuesto sea el correcto para el tipo de anomalía presente. Este es el principal argumento para preferir un ensemble sobre un único detector en un escenario real, donde no se sabe de antemano qué tipo de anomalía va a aparecer.

---

## Cómo ejecutar

1. Abrir `Eigenfaces_NMF_Anomalias.ipynb` en Google Colab (`Archivo → Subir cuaderno` o arrastrándolo directamente).
2. Ejecutar las celdas en orden (`Entorno de ejecución → Ejecutar todas`). La primera celda instala/actualiza las librerías necesarias.
3. Las secciones 1 y 2 descargan datasets públicos de `scikit-learn` (Olivetti Faces y 20 Newsgroups) automáticamente la primera vez que se ejecutan — requieren conexión a internet, disponible por defecto en Colab.
4. La sección 3 no requiere descargas: la serie temporal se genera sintéticamente dentro del propio notebook.

**Librerías usadas:** `numpy`, `pandas`, `matplotlib`, `scikit-learn` (PCA, NMF, SVC, IsolationForest, LocalOutlierFactor, OneClassSVM, TfidfVectorizer, métricas).

---

## Conclusiones generales

- **PCA y NMF** son ambas técnicas de reducción de dimensión, pero con propósitos distintos: PCA busca las direcciones de máxima varianza (útil para compresión/reconstrucción, como en Eigenfaces) mientras que NMF impone no-negatividad y produce factores más interpretables como "proporciones de tópicos" (útil para recomendación de contenido).
- Reducir la dimensión **antes** de aplicar un algoritmo posterior (SVM, similitud de coseno) no solo acelera el cómputo, sino que en varios casos **mejora la calidad** de los resultados al eliminar ruido y redundancia presente en el espacio original de alta dimensión (píxeles o vocabulario).
- En detección de anomalías no supervisada, **ningún detector individual domina en todos los escenarios**; combinar detectores con supuestos distintos en un ensemble reduce el riesgo de depender de un único supuesto incorrecto sobre la naturaleza de las anomalías.
