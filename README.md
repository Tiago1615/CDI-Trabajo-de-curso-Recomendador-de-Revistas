# Recomendador Inteligente de Revistas Científicas

Este proyecto desarrolla un **sistema inteligente de recomendación de revistas científicas** a partir del contenido textual de un artículo (título, resumen y palabras clave), utilizando técnicas de **Ciencia de Datos** y **Procesamiento de Lenguaje Natural (NLP)**.

El sistema aprende a recomendar la revista más adecuada a partir de artículos publicados previamente, siguiendo dos enfoques diferenciados:

- **Aproximación clásica** basada en representaciones TF-IDF y modelos de clasificación tradicionales.
- **Aproximación conexionista** basada en redes neuronales recurrentes (BiLSTM) y embeddings Word2Vec.

El proyecto ha sido desarrollado íntegramente en Python, empleando librerías estándar del ecosistema de Machine Learning y Deep Learning.

---

## 🧠 Conjunto de Datos

El conjunto de datos está formado por **artículos científicos publicados en revistas del ámbito de la Inteligencia Artificial y áreas afines**, obtenidos de ScienceDirect(Elsevier).

Cada instancia contiene:
- Título del artículo
- Resumen (abstract)
- Palabras clave
- Revista de publicación (etiqueta)

Debido a un fuerte desbalance inicial entre revistas, se aplicó un filtrado por frecuencia, eliminando aquellas revistas con menos de **20 artículos**, con el objetivo de:
- Garantizar significancia estadística
- Evitar ruido por clases extremadamente pequeñas
- Prescindir de técnicas artificiales de rebalanceo

---

## 🔧 Preprocesamiento del Texto

Las principales etapas de preprocesamiento incluyen:

- Conversión del texto a minúsculas
- Eliminación de caracteres especiales y signos de puntuación
- Tokenización del texto

En la aproximación clásica se delega gran parte del preprocesamiento al propio `TfidfVectorizer` de *scikit-learn*.  Adicionalmente, se evalúa el impacto de la lematización mediante *NLTK*.

---

## 📊 Aproximación Clásica

### Representación
- TF-IDF con n-gramas (1,2)
- Stopwords en inglés

### Modelos evaluados
- Regresión Logística
- Linear SVM
- Random Forest

Cada modelo se entrena y evalúa:
- **Con y sin lematización**
- Mediante **validación cruzada estratificada**
- Sobre un **conjunto de test independiente**

### Métricas
- Accuracy
- Precision, Recall y F1-score
- Matrices de confusión

📌 El modelo **Linear SVM con TF-IDF sin lematización** obtiene el mejor rendimiento global (≈ 72% de accuracy en test).

---

## 🤖 Aproximación Conexionista

### Representación distribucional
- Entrenamiento de un modelo **Word2Vec** propio sobre el corpus
- Tokenización simple (lowercase + limpieza básica)
- Construcción de una **matriz de embeddings**

### Modelo
- Arquitectura **BiLSTM (Bidirectional LSTM)** implementada en PyTorch
- Capa de embeddings inicializada con Word2Vec
- Capa recurrente bidireccional
- Capa densa final multiclase

### Regularización
- Dropout
- Penalización L2 (*weight decay*)
- Early stopping

El entrenamiento se realizó utilizando **aceleración por GPU** con **CUDA 13.1**.

📌 El modelo BiLSTM alcanza un rendimiento cercano al 70% en test, comparable a los modelos clásicos, aunque presenta mayor tendencia al sobreajuste y mayor coste computacional.

---

## ⚙️ Entorno de Ejecución

El entorno se gestiona mediante **Anaconda**.

### Crear el entorno
```bash
conda env create -f environment.yml
```

### Activar el entorno
```bash
conda activate journal-recommender-nlp
```

### Dependencias principales

- Python 3.10
- NumPy
- Pandas
- Scikit-learn
- NLTK
- PyTorch
- Gensim
- Matplotlib / Seaborn

---

## Ejecución del Proyecto

1. Activar el entorno de Anaconda

2. Abrir el notebook principal: ``Recomendador-Revistas.ipynb``

3. Seleccionar el entorno creado con Anaconda como kernel, haciendo clic en la parte superior derecha ``select kernel``, o alternativamente:
   - Presionar las teclas ``Control + Shift + P``.
   - Escribir ``Python: Select Interpreter``.
   - Seleccionar el entorno creado con Anaconda.

4. Ejecutar las celdas secuencialmente para:
   - Cargar y preprocesar los datos.
   - Entrenar los modelos clásicos.
   - Entrenar el modelo BiLSTM.
   - Evaluar resultados y generar gráficas.

---

## Conclusiones Principales

- Los modelos clásicos basados en TF-IDF + Linear SVM ofrecen el mejor equilibrio entre rendimiento, estabilidad y coste computacional.
- La lematización no aporta mejoras significativas en este problema concreto.
- La aproximación conexionista es competitiva, pero requiere mayor regularización y más datos para superar claramente a los enfoques clásicos.
