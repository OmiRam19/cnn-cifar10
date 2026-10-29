# CNN sobre CIFAR-10 — Evaluación Segunda Unidad IA

Universidad Andina del Cusco · Código 022100269G · Semilla individual: **69**

## Pregunta 2 — CNN frente a MLP

### 1. Justificación teórica

Una imagen de CIFAR-10 tiene 32×32×3 = 3 072 valores. Un MLP la aplana en un vector y conecta cada píxel con cada neurona oculta, lo que genera tres problemas:

1. **Explosión de parámetros.** Una sola capa oculta de 1 024 neuronas ya necesita 3 072 × 1 024 ≈ 3,1 M de pesos, y crece cuadráticamente con la resolución. Esto vuelve el entrenamiento costoso y favorece el sobreajuste.
2. **Pérdida de la estructura espacial.** Al aplanar, píxeles vecinos y lejanos reciben el mismo trato; la red debe aprender desde cero que la vecindad importa.
3. **Sin invariancia a la traslación.** Si el objeto se desplaza unos píxeles, el vector de entrada cambia por completo y el MLP debe aprender cada posición por separado.

Las propiedades de las CNN resuelven estas limitaciones:

- **Conexiones locales:** cada neurona ve solo un campo receptivo pequeño (3×3), capta bordes y texturas, y las capas profundas combinan esos patrones en formas y objetos.
- **Weight sharing:** el mismo filtro se aplica en toda la imagen, por lo que `Conv2d(3, 32, 3)` usa solo 3·3·3·32 + 32 = 896 parámetros, no millones.
- **Invarianza espacial:** un patrón se detecta sin importar dónde aparezca; además `MaxPool2d` aporta tolerancia a pequeños desplazamientos y reduce la resolución.

### 2. Especificación de la arquitectura

| Capa | Salida (B = batch) | Parámetros |
|---|---|---|
| Entrada | (B, 3, 32, 32) | – |
| Bloque 1: Conv2d(3→32, 3×3, pad 1) + ReLU + MaxPool2d(2) | (B, 32, 16, 16) | 896 |
| Bloque 2: Conv2d(32→64, 3×3, pad 1) + ReLU + MaxPool2d(2) | (B, 64, 8, 8) | 18 496 |
| Bloque 3: Conv2d(64→128, 3×3, pad 1) + ReLU + MaxPool2d(2) | (B, 128, 4, 4) | 73 856 |
| Flatten | (B, 2048) | – |
| Linear(2048→256) + ReLU + Dropout(0.5) | (B, 256) | 524 544 |
| Linear(256→10) | (B, 10) | 2 570 |

Total: 620 362 parámetros entrenables. Cumple con al menos dos bloques convolucionales (Conv2d + ReLU + MaxPool2d) y capas `Linear` hacia las 10 clases.

## Pregunta 4 — Análisis crítico de métricas (semilla 69)

Configuración: `CrossEntropyLoss`, `Adam` (lr = 1e-3), batch 128, 30 épocas. Se conserva el modelo con mejor accuracy de validación (época 27) y se evalúa en el test independiente.

![Curvas de aprendizaje](outputs/loss_accuracy.png)
![Matriz de confusión](outputs/confusion_matrix.png)

| Métrica | Train | Validación | Test |
|---|---|---|---|
| Accuracy (época 30) | 81,06 % | 81,04 % | – |
| Loss (época 30) | 0,552 | 0,567 | – |
| Mejor modelo (época 27) | – | 82,42 % | **81,33 %** (loss 0,552) |

- **Overfitting: no.** La brecha entre entrenamiento y validación en la última época es de apenas 0,02 puntos de accuracy (81,06 % vs. 81,04 %) y 0,015 en loss. Durante casi todo el entrenamiento la validación estuvo por encima del entrenamiento, algo esperable porque el train usa data augmentation y Dropout (0,5) y la validación no. La loss de validación sube ligeramente en las últimas 3 épocas (0,536 → 0,567), pero sin abrir una brecha sostenida con el entrenamiento.
- **Underfitting: leve.** La accuracy de entrenamiento es solo ~81 % y las curvas seguían mejorando lentamente al terminar las 30 épocas. Más épocas, BatchNorm, más filtros o un *scheduler* de la tasa de aprendizaje probablemente superarían el 85 %.
- **Generalización:** el accuracy de test (81,33 %) es consistente con el de validación (82,42 % del mejor checkpoint; ~1 punto de diferencia por seleccionar el mejor modelo sobre validación).
- **Errores por clase:** las mejores clases son *automóvil* (F1 0,921) y *barco* (0,904); las peores, *gato* (0,654) y *pájaro* (0,733). Las confusiones más frecuentes son perro → gato (147 casos), gato → perro (104), gato → pájaro (76), pájaro → ciervo (69) y gato → ciervo (67): animales con formas y texturas similares a 32×32 píxeles.
