# Resultados de evaluación — Voice AI Conversational Agent

Generado por `python -m eval.run` el 2026-09-29T01:23:45Z a partir de `eval/runs/2026-09-29-offline-stub.json`.

**Aviso obligatorio.** Todas las cifras medidas en este archivo provienen del **fallback determinista, sin LLM** (`StubReasoner`, STT/TTS stub). No representan el rendimiento del sistema con Claude, Whisper y ElevenLabs; esas celdas quedan como `pendiente (...)` hasta que exista una corrida con llaves guardada en `eval/runs/`.

## Tabla comparativa obligatoria

| Componente | Métrica | Valor |
|---|---|---|
| STT | WER sobre LibriSpeech test-clean | pendiente (LibriSpeech: openslr.org bloqueado; requiere extra `[stt]` y `scripts/wer_benchmark.py --librispeech-dir`) |
| STT | WER en español Common Voice | pendiente (Common Voice es gated en HF; requiere token y `--common-voice-dir`) |
| LLM | Resolution rate sobre 100 conversaciones | 90.0 % (90/100) — **fallback determinista, sin LLM**; con Claude: pendiente (requiere ANTHROPIC_API_KEY) |
| TTS | MOS automated | pendiente (NISQA + ELEVENLABS_API_KEY / XTTS-v2) |
| Sistema | Latencia p95 end-to-end por turno | 5 ms — **solo stubs (sin STT/TTS/LLM reales)**; con componentes reales: pendiente (llaves + GPU) |
| Sistema | Costo por minuto de conversación | pendiente (medición real requiere ANTHROPIC_API_KEY + ELEVENLABS_API_KEY; estimación con supuestos en docs/scalability.md) |

## Baselines (100 conversaciones sintéticas, `data/eval/conversations.jsonl`)

| Baseline | Resolution rate | Reservas resueltas | Transferencias correctas | Slots exactos | Turnos medios | p95 ms | Tokens in/out | Costo USD |
|---|---|---|---|---|---|---|---|---|
| stub_heuristic (fallback determinista, sin LLM) | 90.0 % | 87.8 % (82) | 100.0 % (18) | 87.8 % | 4.47 | 5 | 0/0 | 0.0 |
| claude_zero_shot_stateless | pendiente (requiere ANTHROPIC_API_KEY) | – | – | – | – | – | – | – |
| claude_pipeline (producción) | pendiente (requiere ANTHROPIC_API_KEY) | – | – | – | – | – | – | – |

### Por escenario (stub_heuristic)

| Escenario | n | Resueltas | Tasa |
|---|---|---|---|
| correction | 12 | 12 | 100.0 % |
| incremental | 40 | 40 | 100.0 % |
| one_shot | 20 | 20 | 100.0 % |
| out_of_scope | 8 | 8 | 100.0 % |
| spanish | 10 | 0 | 0.0 % |
| transfer_human | 10 | 10 | 100.0 % |

## LLM-as-judge (rúbrica numerada en `src/eval/judge.py`)

| Juez | n | Pass rate (crit. 1-3) | 1 Tarea | 2 Slots | 3 Sin transferencia innecesaria | 4 Eficiencia | 5 Naturalidad (1-5) |
|---|---|---|---|---|---|---|---|
| heurístico (proxy offline, no es LLM) | 100 | 90.0 % | 90.0 % | 90.0 % | 100.0 % | 100.0 % | 3.56 |
| Claude (LLM-as-judge) | pendiente (requiere ANTHROPIC_API_KEY) | – | – | – | – | – | – |

## Ablaciones

### VAD on/off (audio sintético: 10 turnos con voz + 10 de silencio, STT stub)

| VAD | Silencios rechazados | Voz que pasa | Llamadas STT ahorradas | Sobrecarga VAD media (ms) |
|---|---|---|---|---|
| energy | 10/10 (100.0 %) | 10/10 (100.0 %) | 10 | 0.0 |
| off | 0/10 (0.0 %) | 10/10 (100.0 %) | 0 | 0.0 |

### Whisper tiny vs base (CPU)

pendiente (requiere extra `[stt]` con torch y LibriSpeech; openslr.org bloqueado en este entorno)

## Latencia por nodo (stubs, media sobre todos los turnos)

| Nodo | Media ms |
|---|---|
| llm | 0.07 |
| stt | 0.01 |
| tts | 0.25 |
| vad | 0.0 |

## Peores casos

Análisis con hipótesis en `docs/error_analysis.md`. Fallos de la corrida:

- `conv-091` (spanish): esperado `book`, obtenido `ask_clarification`; slots {'party_size': 5, 'time': '21:00'}
- `conv-092` (spanish): esperado `book`, obtenido `ask_clarification`; slots {'party_size': 2, 'time': '18:45'}
- `conv-093` (spanish): esperado `book`, obtenido `ask_clarification`; slots {'party_size': 4, 'time': '19:00'}
- `conv-094` (spanish): esperado `book`, obtenido `ask_clarification`; slots {'party_size': 8, 'time': '21:00'}
- `conv-095` (spanish): esperado `book`, obtenido `ask_clarification`; slots {'party_size': 4, 'time': '21:00'}
- `conv-096` (spanish): esperado `book`, obtenido `ask_clarification`; slots {'party_size': 6, 'time': '12:00'}
- `conv-097` (spanish): esperado `book`, obtenido `ask_clarification`; slots {'party_size': 4, 'time': '20:00'}
- `conv-098` (spanish): esperado `book`, obtenido `ask_clarification`; slots {'party_size': 5, 'time': '21:00'}
- `conv-099` (spanish): esperado `book`, obtenido `ask_clarification`; slots {'party_size': 5, 'time': '21:00'}
- `conv-100` (spanish): esperado `book`, obtenido `ask_clarification`; slots {'party_size': 4, 'time': '20:00'}

## Reproducir

```bash
python scripts/build_eval_set.py   # regenera data/eval/conversations.jsonl (seed fijo)
python -m eval.run                 # escribe eval/runs/<fecha>-<nombre>.json y este archivo
python -m eval.run --check         # verifica que las métricas deterministas coinciden
```
