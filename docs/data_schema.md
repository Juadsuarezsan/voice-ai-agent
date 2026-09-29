# Esquema de datos

Este proyecto no entrena modelos; sus datos son (a) el eval set sintético de
conversaciones que sí está commiteado, (b) un fixture manual para el script de
WER y (c) los datasets públicos de voz/diálogo que `scripts/download_data.py`
descarga bajo `data/raw/` (ignorado por git; ninguno cabe en el repo).

## `data/eval/conversations.jsonl` (commiteado, sintético)

Un objeto JSON por línea. Generado por `scripts/build_eval_set.py` con semilla
`20260516` a partir de bancos de frases escritas a mano; la verdad de terreno se
deriva de los valores muestreados, así que es correcta por construcción. Todos
los registros llevan `"synthetic": true` y no se mezclan con datos reales.

| Campo | Tipo | Rango / valores | Descripción |
|---|---|---|---|
| `id` | string | `conv-001` … `conv-100` | Identificador estable. |
| `synthetic` | bool | siempre `true` | Marca explícita de dato sintético. |
| `generator` | string | `scripts/build_eval_set.py` | Procedencia. |
| `seed` | int | `20260516` | Semilla usada. |
| `scenario` | string | `incremental` (40), `one_shot` (20), `correction` (12), `transfer_human` (10), `out_of_scope` (8), `spanish` (10) | Familia de escenario. |
| `turns` | list[string] | 1–7 enunciados | Turnos del usuario en orden; el simulador los envía uno a uno hasta que el agente termina. |
| `expected.slots` | object | vacío para transferencias; para reservas `party_size` (int 1–10), `date` (string tal como se dijo, p. ej. `tomorrow`, `friday`, `2026-10-03`, `october 12`), `time` (`HH:MM` 24 h), `name` (string) | Slots que debe tener el estado final. |
| `expected.final_action` | string | `book` \| `transfer_to_human` | Acción con la que debe terminar la conversación. |

Notas:

- Los 10 diálogos en español existen para medir la limitación conocida del
  fallback determinista (solo inglés). No se retiran del eval set.
- El escenario `correction` incluye una enmienda del `party_size` durante la
  lectura de confirmación; el slot esperado es el valor corregido.

## `data/eval/wer_fixture/` (commiteado, escrito a mano)

`references.txt` y `hypotheses.txt`: 10 líneas alineadas. Solo sirven para probar
`scripts/wer_benchmark.py`; su WER (0.049) **no** es un resultado de Whisper.

## `data/raw/` (descargado, ignorado por git)

| Dataset | URL exacta | Licencia | Tamaño | Uso |
|---|---|---|---|---|
| LibriSpeech test-clean | https://www.openslr.org/resources/12/test-clean.tar.gz | CC BY 4.0 | 346 MB | WER en inglés (`scripts/wer_benchmark.py --librispeech-dir`). |
| LibriSpeech test-other | https://www.openslr.org/resources/12/test-other.tar.gz | CC BY 4.0 | 328 MB | WER en condiciones difíciles. |
| MultiWOZ 2.4 | https://github.com/smartyfh/MultiWOZ2.4/archive/refs/heads/main.zip | MIT | ~30 MB | Diálogos task-oriented reales (dominio `restaurant`) para contrastar el eval set sintético. |
| Mozilla Common Voice (es) | https://commonvoice.mozilla.org/es/datasets | CC0 (dataset gated: requiere cuenta) | varios GB | WER en español; descarga manual, no automatizada. |

Estructura de LibriSpeech tras extraer: `LibriSpeech/<split>/<speaker>/<chapter>/<speaker>-<chapter>-<utt>.flac`
más `<speaker>-<chapter>.trans.txt` con líneas `ID TRANSCRIPCIÓN EN MAYÚSCULAS`.
El script de WER normaliza a minúsculas y sin puntuación (jiwer).

## `data/MANIFEST.txt`

Una línea por archivo: `sha256  archivo  dataset  licencia  bytes  timestamp`.
Los fixtures commiteados se listan con su hash real; los datasets descargados se
añaden automáticamente por `scripts/download_data.py` cuando se ejecuta en una
red que permita `openslr.org`.

## Reproducibilidad

```bash
python scripts/build_eval_set.py           # regenera conversations.jsonl (idéntico byte a byte)
python scripts/download_data.py --dataset all
python -m eval.run                         # escribe eval/runs/*.json y eval/RESULTS.md
```

No hay split train/val/test: el eval set es solo de evaluación (ningún componente
se ajusta con él) y el orden de los 100 registros es fijo por la semilla.
