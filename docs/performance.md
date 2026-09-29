# Performance — Voice AI Conversational Agent

## Qué se ha medido (stubs, sin llaves, CPU compartida)

Fuente: `eval/runs/2026-09-29-offline-stub.json`, 100 conversaciones (447
turnos) más 20 turnos de audio sintético para la ablación de VAD.

| Métrica | Valor | Condición |
|---|---|---|
| Latencia media por turno | 2–3 ms | texto → StubSTT → StubReasoner → StubTTS → logger en memoria |
| Latencia p95 por turno | 5 ms | ídem |
| `llm` (StubReasoner, regex) | 0.07 ms | media por nodo |
| `stt` (StubSTT) | 0.01 ms | media por nodo |
| `tts` (StubTTS, tono de relleno) | 0.25 ms | media por nodo; genera y codifica un WAV |
| `vad` (EnergyVAD) | <0.05 ms | sobre 1.2–1.5 s de audio a 16 kHz |
| Sobrecarga del grafo + logger | ≈ 2 ms | diferencia entre el total y la suma de nodos |

Conclusión: la orquestación (FastAPI, LangGraph, medición, redacción de PII,
persistencia en memoria) cuesta ~2 ms por turno. El objetivo de <800 ms
depende íntegramente de STT, LLM y TTS reales.

## Lo que falta medir (pendiente de llaves y GPU)

| Medición | Cómo se obtiene | Bloqueo |
|---|---|---|
| p95 end-to-end real | `python -m eval.run` con `ANTHROPIC_API_KEY`, `TTS_BACKEND=elevenlabs`, `WHISPER_BACKEND=api` o `local` | llaves; GPU para whisper-large-v3 |
| Latencia de Whisper `tiny` vs `base` en CPU | `scripts/wer_benchmark.py --librispeech-dir … --whisper-model tiny|base` (reporta `transcription_seconds`) | extra `[stt]` (torch) y LibriSpeech (openslr bloqueado aquí) |
| Tiempo al primer byte de TTS | streaming por chunks de ElevenLabs | llave |

## Dónde optimizar cuando existan medidas

1. **STT incremental**: transcribir mientras el usuario habla (Silero VAD en
   streaming + Whisper por ventanas) para que al final del turno solo quede el
   último fragmento. Es el cambio con mayor impacto esperado.
2. **TTS por chunks**: emitir el audio de la primera oración mientras se
   sintetiza la segunda. La API ya separa `response_text` en oraciones cortas
   (prompt: "one or two short spoken sentences").
3. **Prompt del LLM**: el sistema de prompt es estable y cacheable; el historial
   se recorta a 12 mensajes y el estado viaja como JSON compacto para acotar
   tokens de entrada (y con ello latencia y costo).
4. **Persistencia fuera del camino crítico**: `logger_node` es el último nodo y
   con Postgres añadiría un round-trip; mover la escritura a una cola.

## Perfil de memoria

Instalación base (sin torch): el proceso de la API con stubs ocupa decenas de
MB. Los extras `[stt]`/`[vad]`/`[tts]` cargan torch y modelos (whisper `base`
≈ 150 MB, `large-v3` ≈ 3 GB de pesos; XTTS-v2 ≈ 2 GB), por eso la imagen
Docker se divide en `base` y `audio`.

## Cómo reproducir estos números

```bash
make eval                      # regenera eval/runs/*.json y eval/RESULTS.md
curl localhost:8000/api/metrics | jq .latency_ms   # últimos 100 turnos en vivo
```
