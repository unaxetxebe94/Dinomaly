# Dinomaly sobre imágenes grandes troceadas (Top-Bottom / Left-Right, COCO)

Pipeline de detección no supervisada de anomalías con
[Dinomaly (CVPR 2025)](https://github.com/guojiajeremy/Dinomaly) adaptado a los
escaneos grandes del dataset (`16384 x 21000/27000` píxeles) con anotaciones
COCO de tipo polígono.

```
imágenes gigantes + _annotations.coco.json
        │
        ├─ 01 build_tiles  → tiles JPEG 1024² + máscaras PNG + manifest.parquet
        │
        ├─ 02 train        → Dinomaly (DINOv2 congelado + bottleneck + decoder)
        │
        ├─ 03 evaluate     → AUROC/AP/F1 a nivel de píxel, tile e imagen
        │
        └─ 04 predict      → heatmap por imagen completa
```

## 1. Requisitos

- Linux con `libvips` (para leer los TIFF sin cargarlos en RAM):

  ```bash
  sudo pacman -S libvips        # Arch / Omarchy
  ```

- [uv](https://docs.astral.sh/uv/) y una GPU NVIDIA. El proyecto fija Python
  3.11 y `torch>=2.7` (ruedas CUDA 12.8+, compatible con RTX 50xx).

  ```bash
  uv sync
  uv run python -c "import torch; print(torch.cuda.is_available())"
  ```

## 2. Configuración

Hay un config por grupo de caras: [`configs/top-bottom.yaml`](configs/top-bottom.yaml)
(fuentes `Top` + `Bottom`) y [`configs/left-right.yaml`](configs/left-right.yaml)
(`Left` + `Right`). Los ejemplos usan `top-bottom`; cambia el config para
entrenar el otro grupo. Lo importante:

| Clave | Valor por defecto | Significado |
|---|---|---|
| `data.sources` | carpetas `Top` + `Bottom` | Lista de `{name, root, annotations}`. Acepta varias carpetas. |
| `data.tiles_dir` | `data/tiles_top-bottom` | Salida de tiles, máscaras y manifest. |
| `data.tile_size` / `tile_overlap` | `1024` / `128` | Stride = 896 px (12.5 % solape). |
| `data.min_std` | `3.0` | Descarta tiles de fondo casi uniforme (los que tienen defecto nunca se descartan). |
| `data.roi.enabled` | `true` | Solo genera tiles alrededor del objeto detectado (ver abajo). |
| `data.roi.threshold` | `10` | Brillo mínimo (0–255) para separar el objeto del fondo negro. |
| `data.roi.select` | `area` | Qué zona conectada se queda: `area`, `width` o `height`. |
| `data.roi.margin` | `0` | Píxeles extra alrededor de la caja. |
| `data.roi.preview` | `1024` | Resolución de trabajo del detector (no afecta a los tiles). |
| `data.split` | 70/15/15 | Split **por imagen original**, estratificado por presencia de defecto. |
| `data.crop_size` | `392` | Entrada del ViT (392 / 14 = 28×28 tokens). |
| `model.name` | `dinov2reg_vit_base_14` | `small` / `base` / `large`, con o sin registros. |
| `train.total_iters` | `10000` | Iteraciones (no epochs). |
| `train.batch_size` | `16` | ~5.7 GB de VRAM con ViT-B/392 y AMP. |
| `train.checkpoint_dir` | `runs/top-bottom` | `best.pt`, `last.pt`, `history.csv`, `test_metrics.json`. |

> **Nota sobre las rutas**: cada config combina dos fuentes (dos caras); para
> añadir otra basta con añadir otra entrada a `data.sources`. Las imágenes con el mismo
> `image_id` en fuentes distintas no colisionan (`image_uid = source:id`).
>
> El reparto train/val/test se calcula sobre el conjunto completo de imágenes
> con una semilla fija. Si cambias las anotaciones, las fuentes, `tile_size`,
> `tile_overlap` o cualquier valor de `data.roi`, los tiles existentes dejan de
> ser válidos: reconstruye con `--overwrite` en lugar de `--resume`. El
> manifiesto guarda la firma del troceado y `--resume` se niega a mezclar
> configuraciones distintas.

## 3. Crear los tiles

```bash
uv run python scripts/build_tiles.py --config configs/top-bottom.yaml --workers 6
```

- Lee los TIFF con `libvips` (solo decodifica las ventanas necesarias).
- Cada `--workers` es un **proceso** independiente: libvips/libtiff falla
  (`tiffload: load error`) al decodificar varios TIFF gigantes desde hilos del
  mismo proceso. Si el proceso se interrumpe, relánzalo con `--resume` para
  completar solo las imágenes que faltan sin repetir las ya hechas.
- Genera `tiles/<split>/<tile_id>.jpg` y `masks/<split>/<tile_id>.png`.
  En la máscara cada píxel vale el índice compacto de la clase (0 = fondo);
  [`classes.json`](data/tiles_top-bottom/classes.json) mapea índice → nombre.
- Escribe `manifest.parquet` (una fila por tile: ruta, split, coordenadas,
  `has_defect`, píxeles de defecto, categorías) y `build_info.json`.
- Los tiles que contienen **cualquier** píxel anotado se marcan como
  defectuosos y se excluyen del entrenamiento.

### Recorte al objeto (ROI)

Los escaneos son un lienzo enorme con el objeto en el centro y un borde negro
alrededor. Para no trocear el fondo, `data.roi` activa este proceso por imagen:

1. Se crea un thumbnail de `preview` px.
2. Se binariza el brillo con `threshold` (0–255): píxel = objeto si supera el
   umbral.
3. Se etiquetan las **zonas conectadas** y se queda la mayor (`select: area`);
   también puedes elegir la más ancha (`width`) o la más alta (`height`).
4. Solo se generan los tiles de la cuadrícula habitual que **intersecan la caja**
   de esa zona; el resto del lienzo se ignora.

La cuadrícula no se realinea: los tiles siguen midiendo exactamente
`tile_size` y pueden sobresalir un poco de la caja (unas pocas franjas de
fondo, que `min_std` descarta). Si la detección falla (imagen muy oscura o sin
píxeles por encima del umbral) se trocea la imagen completa y se anota el aviso
en `build_info.json`.

Antes de lanzar horas de troceado, comprueba el recorte en una muestra:

```bash
uv run python scripts/inspect_roi.py --config configs/top-bottom.yaml --limit 10
```

Escribe en `roi_preview/` un JPG por imagen con la caja dibujada, la máscara
binarizada y `summary.txt` con la comparación de tiles (ROI vs imagen
completa). Si la caja no encaja, ajusta `threshold` (súbelo si entra fondo,
bájalo si se corta el objeto). El manifiesto guarda además `roi_x0/roi_y0/
roi_x1/roi_y1` por tile, y `build_info.json` la cobertura media.

Si un run anterior quedó incompleto:

```bash
uv run python scripts/build_tiles.py --config configs/top-bottom.yaml --workers 6 --resume
```

También puedes forzar la reconstrucción completa con `--overwrite` (borra
`tiles/`, `masks/` y el manifiesto anteriores para no dejar JPEG huérfanos).

Prueba rápida con 4 imágenes (sin llenar el disco):

```bash
uv run python scripts/build_tiles.py --config configs/top-bottom.yaml \
    --tiles-dir data/tiles_debug --limit-images 4 --workers 4 --overwrite
```

**Órdenes de magnitud** (dependen de las fuentes y del JSON de anotaciones;
en el `_annotations.coco.json` original de `Top`: 607 imágenes anotadas,
579 presentes y 1386 polígonos):

| Concepto | Estimación |
|---|---|
| Tiles a 1024/128 | ~170–220 k por cada ~600 imágenes (sin ROI) |
| Tiles con `data.roi` activo | ~40–50 % menos cuando el objeto no llena el lienzo |
| Tamaño JPEG medio | ~200 KB |
| Disco | ~30–40 GB (menos con ROI) |
| Tiempo con 6 workers | ~45–70 min (menos con ROI) |

Las imágenes anotadas que no existen en disco y los TIFF corruptos se
registran en `build_info.json` y no detienen el proceso. En `Top` el archivo
corrupto conocido es `1377_26944_top_native.tiff`.

## 4. Entrenar

```bash
uv run python scripts/train.py --config configs/top-bottom.yaml
```

- Entrena **solo con tiles normales** del split `train` (Dinomaly es
  no supervisado). Los tiles con defecto de imágenes de train no se usan.
- Congela DINOv2, entrena bottleneck + 8 bloques de decoder (~61 M parámetros).
- Pérdida: coseno con *hard mining* percentil (`p=0.9`, `factor=0.1`),
  igual que el paper.
- Cada `eval_every` iters evalúa un subconjunto de validación (todos los tiles
  defectuosos + normales aleatorios) y guarda `best.pt` según `image.auroc`
  (con fallback a `tile.auroc`).
- Guarda `last.pt` cada `save_every` iteraciones, además de en cada evaluación.
- Si una iteración produce pérdida o gradiente no finito, se **descarta ese
  step** (con aviso en el log) en lugar de corromper los pesos con NaN. Tras
  25 steps no finitos seguidos desactiva AMP automáticamente; si el problema
  persiste en fp32, aborta para que reanudes desde el último checkpoint bueno.
- Por defecto entrena en **fp32** (`train.amp: false`), como el código oficial.
  `train.amp: true` usa bfloat16 y es ~2.5× más rápido, pero en una ejecución
  larga puede desestabilizarse; el guard de NaN evita que eso arruine el run.
- Al terminar, si añades `--eval-test`, evalúa `best.pt` en el split de test y
  escribe `runs/top-bottom/test_metrics.json`.

Reanudar:

```bash
uv run python scripts/train.py --config configs/top-bottom.yaml --resume runs/top-bottom/last.pt
```

Para un run largo es más cómodo el lanzador con auto-reanudación: si el
proceso muere, vuelve a arrancar desde `last.pt` (hasta 5 intentos) y lanza
la evaluación de test al terminar.

```bash
./scripts/train_resilient.sh configs/top-bottom.yaml
```

Prueba de humo (20 iteraciones):

```bash
uv run python scripts/train.py --config configs/top-bottom.yaml \
    --tiles-dir data/tiles_debug --run-dir runs/debug \
    --set train.total_iters=20 --set train.eval_every=10 --set train.max_eval_tiles=300
```

En una RTX 5060 Ti: ~1.4 it/s en fp32 (batch 16), 10 000 iteraciones ≈ 2 h;
con `--set train.amp=true` sube a ~3–3.5 it/s (≈50 min).

### Salidas

```
runs/top-bottom/
├── best.pt              # bottleneck + decoder + umbrales de val
├── last.pt              # último checkpoint (reanudable)
├── history.csv
├── train.log
├── train_driver.log     # solo con scripts/train_resilient.sh
├── config.json
├── val_metrics.json
├── test_metrics.json    # si se usa --eval-test
└── metrics_iter_N.json
```

Los checkpoints guardan `pixel_threshold` e `image_threshold` calculados en
validación (mejor F1) para aplicarlos al test sin reajustar.

## 5. Evaluar

```bash
uv run python scripts/evaluate.py --config configs/top-bottom.yaml \
    --checkpoint runs/top-bottom/best.pt --split test --max-tiles 8000
```

`--max-tiles 0` evalúa todos los tiles (lento). El script imprime y guarda
`eval_metrics.json` junto al checkpoint.

### Inspeccionar los errores del test

Por defecto se guardan los **50 falsos positivos y 50 falsos negativos** más
severos de cada split evaluado, con su heatmap, en
`runs/top-bottom/<split>_errors/`:

| Archivo | Contenido |
|---|---|
| `false_positives.jpg` | mosaico de tiles normales marcados como anómalos (borde rojo) |
| `false_negatives.jpg` | mosaico de tiles con defecto que pasaron como normales (borde azul) |
| `fp_<tile_id>.jpg` / `fn_<tile_id>.jpg` | overlay del heatmap sobre el tile; en los FN el polígono anotado se dibuja en **verde** |
| `*_map.png` | heatmap puro, sin la imagen debajo |
| `*_mask.png` | máscara GT (solo FN) |
| `errors.json` | score, umbral de decisión, categorías, rutas y recuento total de FP/FN del split |

La decisión es a nivel de **tile**: score = media del 1 % de píxeles más altos,
anómalo si supera el umbral de tile de mejor F1. El heatmap usa una escala fija
(`4 × umbral`) para que todos los tiles sean comparables; el valor se anota en
`errors.json`.

Ajustes:

```bash
# más casos, y en otra carpeta
uv run python scripts/evaluate.py --config configs/top-bottom.yaml \
    --checkpoint runs/top-bottom/best.pt --split test --error-tiles 150 \
    --errors-dir runs/top-bottom/mis_errores

# desactivar
uv run python scripts/evaluate.py ... --error-tiles 0
```

El entrenamiento con `--eval-test` los genera automáticamente en
`runs/top-bottom/test_errors/` (tamaño controlado por `train.save_error_tiles`).

### Métricas

Se calculan a tres niveles, todas usando las máscaras de polígono rasterizadas:

| Nivel | Qué mide | Cómo |
|---|---|---|
| **Píxel** | localización | AUROC, AP, F1 (mejor umbral y F1 al umbral de validación) y PRO sobre los mapas de anomalía vs. máscaras. |
| **Tile** | detección local | Score del tile = media del 1 % de píxeles más anómalos; etiqueta = ¿tiene algún píxel de defecto? |
| **Imagen** | detección global | Score de la imagen = máximo (o top-k) de sus tiles; etiqueta = ¿tiene algún defecto? |

También hay desglose **por categoría** de AUROC/AP de píxel y AP de tile
(p. ej. `Damaged face`, `Slip skin`, `Broken lipping`, …). El AP de píxel por
categoría usa como negativos todos los píxeles de fondo.

Detalles de implementación:

- Los scores de píxel se acumulan en un histograma de 4096 bins sobre `[0, 2]`
  (distancia coseno), así que la memoria es constante aunque el test tenga
  decenas de miles de tiles. La precisión frente a sklearn es ~1e-4.
- PRO se calcula sobre hasta `train.pro_max_tiles` tiles defectuosos,
  reescalados a `pro_size` (128). Es una aproximación por tile, no sobre la
  imagen completa de 16k×27k.
- El F1 de test usa el umbral elegido en validación (`f1_at_threshold`), además
  del F1 oráculo (`f1`).

## 6. Predecir una imagen nueva

```bash
uv run python scripts/predict.py --config configs/top-bottom.yaml \
    --checkpoint runs/top-bottom/best.pt \
    --image "/run/media/.../Top/1008_111213_top_native.tiff" \
    --out-dir predictions/1008_111213 --downsample 8
```

Genera:

- `anomaly_map.npy` / `anomaly_map.png`: mapa a resolución reducida
  (`downsample=8` → 2048×2625 para una imagen de 16384×21000).
- `overlay.jpg`: heatmap superpuesto a la imagen.
- `prediction.json`: score global, umbral, caja ROI usada y top-20 tiles.

Los tiles se solapan y se combinan con una ventana de Hann para eliminar las
costuras del ViT. Se aplica el mismo recorte `data.roi` que en el entrenamiento
(solo se infiere dentro de la caja del objeto).

## 7. Estructura del código

```
dinomaly/
├── coco.py       # lectura COCO + polígonos
├── roi.py        # detección del objeto (binarizado + componentes conexas)
├── tiling.py     # tiles con libvips + rasterizado de máscaras + splits
├── data.py       # Dataset/DataLoader, augmentations, muestreo de eval
├── model.py      # DINOv2 (timm) + bottleneck + decoder lineal (ViTill)
├── losses.py     # cosine loss con hard mining
├── metrics.py    # AUROC/AP/F1/PRO por píxel, tile e imagen
├── engine.py     # ciclo de entrenamiento, scheduler, checkpoints
└── utils.py
scripts/
├── build_tiles.py
├── inspect_roi.py
├── train.py
├── evaluate.py
└── predict.py
```

## 8. Problemas frecuentes

| Síntoma | Causa / solución |
|---|---|
| `unable to call jpegsave` y `tiffload: load error` al trocear | Varios hilos decodificando TIFF gigantes en el mismo proceso. Ya está resuelto usando procesos; vuelve a lanzar con `--resume` para completar las imágenes que fallaron. |
| `loss=nan` | Inestabilidad numérica (típica con bf16). El entrenamiento descarta ese step y sigue; revisa `skipped=` en el log. Si se repite mucho, usa `train.amp: false`. |
| `I-AUROC= n/a` | El subconjunto de validación no contiene imágenes normales. Usa el split completo o selecciona `tile.auroc` como métrica. |
| El proceso se corta a mitad de entrenamiento | `last.pt` se guarda cada `train.save_every` iteraciones. Reanuda con `--resume runs/top-bottom/last.pt` o usa `./scripts/train_resilient.sh`. |
| El proceso termina sin traza (`EXIT=120`) | Suele ser falta de espacio/RAM: `/tmp` es tmpfs (cuenta como RAM) y los logs o checkpoints ahí compiten con el entrenamiento. Guarda runs y logs en el proyecto, no en `/tmp`. |
| Cambio de anotaciones o de fuentes | Los splits se recalculan; reconstruye los tiles con `--overwrite` (no uses `--resume`). |
| `--resume` avisa de "different tiling settings" | El manifiesto se construyó con otro `tile_size`, anotaciones o `data.roi`. Usa `--overwrite` o un `--tiles-dir` nuevo. |
| La caja del ROI corta el objeto o incluye fondo | Ajusta `data.roi.threshold` y compruébalo con `scripts/inspect_roi.py --limit 10`. |

## 9. Notas y limitaciones

- **Solo polígonos**: las anotaciones RLE se ignoran (en este dataset todas son
  polígonos; se avisa por consola si aparece alguna).
- **Entrenamiento no supervisado**: los tiles con defecto se excluyen, no se
  usan como ejemplos negativos. Si en el futuro quieres aprovecharlos, se puede
  añadir una máscara de pérdida para ignorar los píxeles anómalos.
- **Solape de polígonos**: si dos categorías se solapan, se dibuja primero el
  polígono de mayor área; el píxel queda con la categoría más pequeña (más
  específica).
- **Tiles de borde**: la última fila/columna se desplaza hacia dentro para que
  todos los tiles midan exactamente `tile_size` (sin padding); esos tiles se
  solapan con el anterior y el manifest guarda sus coordenadas reales.
- **Reproducibilidad**: se fija la semilla, pero `cudnn.benchmark=True` puede
  introducir pequeñas diferencias no deterministas entre ejecuciones.
- **Aviso de libvips sobre OpenSlide**: es inocuo (falta un módulo opcional para
  imágenes de microscopía); no afecta a los TIFF.
