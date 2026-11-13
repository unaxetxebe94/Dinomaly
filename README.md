# Dinomaly sobre imágenes grandes troceadas (Top / COCO)

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

Todo se controla desde [`configs/top.yaml`](configs/top.yaml). Lo importante:

| Clave | Valor por defecto | Significado |
|---|---|---|
| `data.sources` | carpeta `Top` | Lista de `{name, root, annotations}`. Acepta varias carpetas. |
| `data.tiles_dir` | `data/tiles_top` | Salida de tiles, máscaras y manifest. |
| `data.tile_size` / `tile_overlap` | `1024` / `128` | Stride = 896 px (12.5 % solape). |
| `data.min_std` | `3.0` | Descarta tiles de fondo casi uniforme (los que tienen defecto nunca se descartan). |
| `data.split` | 70/15/15 | Split **por imagen original**, estratificado por presencia de defecto. |
| `data.crop_size` | `392` | Entrada del ViT (392 / 14 = 28×28 tokens). |
| `model.name` | `dinov2reg_vit_base_14` | `small` / `base` / `large`, con o sin registros. |
| `train.total_iters` | `10000` | Iteraciones (no epochs). |
| `train.batch_size` | `16` | ~5.7 GB de VRAM con ViT-B/392 y AMP. |
| `train.checkpoint_dir` | `runs/top` | `best.pt`, `last.pt`, `history.csv`, `test_metrics.json`. |

> **Nota sobre las rutas**: en la petición aparecía dos veces la misma carpeta
> `Top`. El config usa una sola fuente; para añadir otra (p. ej. `Front`)
> basta con añadir otra entrada a `data.sources`. Las imágenes con el mismo
> `image_id` en fuentes distintas no colisionan (`image_uid = source:id`).
>
> El reparto train/val/test se calcula sobre el conjunto completo de imágenes
> con una semilla fija. Si cambias las anotaciones, las fuentes, `tile_size` o
> `tile_overlap`, los tiles existentes dejan de ser válidos: reconstruye con
> `--overwrite` en lugar de `--resume`.

## 3. Crear los tiles

```bash
uv run python scripts/build_tiles.py --config configs/top.yaml --workers 6
```

- Lee los TIFF con `libvips` (solo decodifica las ventanas necesarias).
- Cada `--workers` es un **proceso** independiente: libvips/libtiff falla
  (`tiffload: load error`) al decodificar varios TIFF gigantes desde hilos del
  mismo proceso. Si el proceso se interrumpe, relánzalo con `--resume` para
  completar solo las imágenes que faltan sin repetir las ya hechas.
- Genera `tiles/<split>/<tile_id>.jpg` y `masks/<split>/<tile_id>.png`.
  En la máscara cada píxel vale el índice compacto de la clase (0 = fondo);
  [`classes.json`](data/tiles_top/classes.json) mapea índice → nombre.
- Escribe `manifest.parquet` (una fila por tile: ruta, split, coordenadas,
  `has_defect`, píxeles de defecto, categorías) y `build_info.json`.
- Los tiles que contienen **cualquier** píxel anotado se marcan como
  defectuosos y se excluyen del entrenamiento.

Si un run anterior quedó incompleto:

```bash
uv run python scripts/build_tiles.py --config configs/top.yaml --workers 6 --resume
```

También puedes forzar la reconstrucción completa con `--overwrite`.

Prueba rápida con 4 imágenes (sin llenar el disco):

```bash
uv run python scripts/build_tiles.py --config configs/top.yaml \
    --tiles-dir data/tiles_debug --limit-images 4 --workers 4 --overwrite
```

**Órdenes de magnitud** (dependen de las fuentes y del JSON de anotaciones;
en el `_annotations.coco.json` original de `Top`: 607 imágenes anotadas,
579 presentes y 1386 polígonos):

| Concepto | Estimación |
|---|---|
| Tiles a 1024/128 | ~170–220 k por cada ~600 imágenes |
| Tamaño JPEG medio | ~200 KB |
| Disco | ~30–40 GB |
| Tiempo con 6 workers | ~45–70 min |

Las imágenes anotadas que no existen en disco y los TIFF corruptos se
registran en `build_info.json` y no detienen el proceso. En `Top` el archivo
corrupto conocido es `1377_26944_top_native.tiff`.

## 4. Entrenar

```bash
uv run python scripts/train.py --config configs/top.yaml
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
  escribe `runs/top/test_metrics.json`.

Reanudar:

```bash
uv run python scripts/train.py --config configs/top.yaml --resume runs/top/last.pt
```

Prueba de humo (20 iteraciones):

```bash
uv run python scripts/train.py --config configs/top.yaml \
    --tiles-dir data/tiles_debug --run-dir runs/debug \
    --set train.total_iters=20 --set train.eval_every=10 --set train.max_eval_tiles=300
```

En una RTX 5060 Ti: ~1.4 it/s en fp32 (batch 16), 10 000 iteraciones ≈ 2 h;
con `--set train.amp=true` sube a ~3–3.5 it/s (≈50 min).

### Salidas

```
runs/top/
├── best.pt              # bottleneck + decoder + umbrales de val
├── last.pt              # último checkpoint (reanudable)
├── history.csv
├── train.log
├── config.json
├── val_metrics.json
├── test_metrics.json    # si se usa --eval-test
└── metrics_iter_N.json
```

Los checkpoints guardan `pixel_threshold` e `image_threshold` calculados en
validación (mejor F1) para aplicarlos al test sin reajustar.

## 5. Evaluar

```bash
uv run python scripts/evaluate.py --config configs/top.yaml \
    --checkpoint runs/top/best.pt --split test --max-tiles 8000
```

`--max-tiles 0` evalúa todos los tiles (lento). El script imprime y guarda
`eval_metrics.json` junto al checkpoint.

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
uv run python scripts/predict.py --config configs/top.yaml \
    --checkpoint runs/top/best.pt \
    --image "/run/media/.../Top/1008_111213_top_native.tiff" \
    --out-dir predictions/1008_111213 --downsample 8
```

Genera:

- `anomaly_map.npy` / `anomaly_map.png`: mapa a resolución reducida
  (`downsample=8` → 2048×2625 para una imagen de 16384×21000).
- `overlay.jpg`: heatmap superpuesto a la imagen.
- `prediction.json`: score global, umbral, top-20 tiles.

Los tiles se solapan y se combinan con una ventana de Hann para eliminar las
costuras del ViT.

## 7. Estructura del código

```
dinomaly/
├── coco.py       # lectura COCO + polígonos
├── tiling.py     # tiles con libvips + rasterizado de máscaras + splits
├── data.py       # Dataset/DataLoader, augmentations, muestreo de eval
├── model.py      # DINOv2 (timm) + bottleneck + decoder lineal (ViTill)
├── losses.py     # cosine loss con hard mining
├── metrics.py    # AUROC/AP/F1/PRO por píxel, tile e imagen
├── engine.py     # ciclo de entrenamiento, scheduler, checkpoints
└── utils.py
scripts/
├── build_tiles.py
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
| El proceso se corta a mitad de entrenamiento | `last.pt` se guarda cada `train.save_every` iteraciones. Reanuda con `--resume runs/top/last.pt`. |
| Cambio de anotaciones o de fuentes | Los splits se recalculan; reconstruye los tiles con `--overwrite` (no uses `--resume`). |

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
