PRA2

Proyecto de limpieza y análisis de datos

Por Andrés Egea Tripiana y José Miguel Cuenca Fernández

Descripción general

Este repositorio contiene el código desarrollado para la práctica de limpieza y análisis de datos basada en el conjunto Default of Credit Card Clients (UCI Machine Learning Repository).

El objetivo del proyecto es analizar el riesgo de impago de clientes de tarjetas de crédito mediante:

técnicas de aprendizaje supervisado (clasificación),

técnicas de aprendizaje no supervisado (clustering),

y un contraste de hipótesis estadístico que valide los resultados obtenidos.

Todo el flujo ha sido diseñado de forma modular, reproducible y secuencial, generando archivos intermedios en cada etapa del proceso.

Estructura del repositorio
PRA2/
│
├── data/
│   ├── UCI_Credit_Card.csv
│   ├── credit_selected.csv
│   ├── credit_clean_step1.csv
│   ├── credit_clean_step2.csv
│   ├── credit_clean_step3.csv
│   └── credit_with_clusters.csv
│
├── reports/
│   ├── pr_curve_xgb.png
│   ├── feature_importance.png
│   ├── clustering_silhouette.png
│   ├── clustering_pca.png
│   ├── default_rate_by_cluster.png
│   └── predicted_default_by_cluster.png
│
├── src/
│   ├── 01_overview.py
│   ├── 02_selection.py
│   ├── 03_clean_invalid_codes.py
│   ├── 04_fix_dtypes.py
│   ├── 05_handle_outliers.py
│   ├── 06_final_sanity_checks.py
│   ├── 07_models_cv.py
│   ├── 08_xgb_gridsearch_prcurve.py
│   ├── 09_feature_importance.py
│   ├── 10_clustering_profiles.py
│   ├── 11_cluster_default_plots.py
│   └── 12_hypothesis_test_clusters.py
│
└── README.txt
└── requirements.txt

Ejecución del proyecto

Importante:
Los scripts deben ejecutarse en orden numérico, ya que muchos de ellos utilizan archivos .csv generados en pasos anteriores.