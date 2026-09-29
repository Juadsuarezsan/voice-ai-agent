# Arquitectura — Voice AI Conversational Agent

![Diagrama](architecture.svg)

## Resumen

Un turno de conversación entra por HTTP (texto o WAV en base64), atraviesa un
`StateGraph` de LangGraph con cinco nodos (`vad_node → stt_node → reasoner_node
→ tts_node → logger_node`) y vuelve como texto + audio + metadatos (acción,
slots, latencia por nodo, tokens, costo, `trace_id`). Cada etapa tiene un
backend real y un stub determinista intercambiables por variable de entorno, de
modo que la misma ruta de código corre en CI sin llaves y en producción con
Claude, Whisper y ElevenLabs.

## Flujo de un turno

1. **`POST /api/turn`** (`src/api/main.py`). Pydantic valida `session_id`
   (`[A-Za-z0-9_-]{1,64}`), tamaño del texto, tamaño y cabecera RIFF/WAVE del
   audio y que exista al menos uno de los dos → 422 en caso contrario. `slowapi`
   aplica `RATE_LIMIT` por IP. CORS solo admite `CORS_ORIGINS`.
2. **`VoiceLoop.turn`** (`src/agent/loop.py`) busca o crea el `SessionState`
   (slots, historial, `awaiting_confirmation`), genera un `trace_id`, lo liga al
   contexto de loguru y construye el `TurnState` inicial del grafo.
3. **`vad_node`**: si hay audio, lo decodifica a PCM float32 y corre el detector
   (`EnergyVAD` por defecto; `SileroVAD` con import perezoso de torch). Sin voz,
   la arista condicional salta a `no_speech_node`, que responde "Sorry, I didn't
   catch that" sin tocar STT ni el reasoner ni el estado de la sesión.
4. **`stt_node`**: `StubSTT` (eco del texto), `LocalWhisperSTT` (remuestreo a
   16 kHz, ejecución en hilo con `asyncio.wait_for`) o `WhisperAPISTT` (httpx,
   timeout, tenacity). Devuelve transcript, idioma, confianza y segundos.
5. **`reasoner_node`**: `ClaudeReasoner` envía el historial (últimos 12
   mensajes) + `<state>` con los slots + `<transcript>` y exige un JSON validado
   contra `LLMTurnOutput`; `StubReasoner` hace slot filling por regex con
   lectura de confirmación antes de reservar. Ambos cumplen el `Protocol`
   `Reasoner` y producen `ReasonResult` (acción, slots, `finished`, tokens).
6. **`tts_node`**: `StubTTS` (tono de relleno), `ElevenLabsTTS` (REST con httpx,
   MP3) o `XTTSLocalTTS`. Devuelve base64, MIME y caracteres facturables.
7. **`logger_node`**: calcula `cost_usd` (tokens × precio del modelo, caracteres
   × precio de ElevenLabs, minutos × precio de Whisper API), redacta PII
   (`src/security/pii.py`: seudónimo estable del nombre, máscara de teléfonos,
   correos y tarjetas) y persiste un `TurnRecord` en el `ConversationLogger`.
8. `VoiceLoop` actualiza la sesión (solo si hubo voz), arma `TurnResponse` y la
   API añade `X-Trace-Id`.

Cada nodo está envuelto por `_timed`, que registra entrada/salida con el
`trace_id` y escribe su latencia en `state["latency"]`.

## Separación de capas

| Capa | Módulos | Sabe de |
|---|---|---|
| Configuración | `src/config.py` | Solo variables de entorno (`Settings`, pydantic-settings). |
| Dominio | `src/agent/reasoner.py`, `src/agent/vad.py`, `src/security/pii.py`, `src/eval/*` | Tipos propios; nada de HTTP. |
| I/O externo | `src/stt/*`, `src/tts/*`, `src/storage/*` | Clientes HTTP/SDK con timeout y tenacity; stubs para tests. |
| Orquestación | `src/agent/graph.py`, `src/agent/loop.py` | Compone dominio e I/O en el grafo. |
| Transporte | `src/api/main.py`, `src/api/schemas.py` | FastAPI, validación, mapeo de errores. |
| Observabilidad | `src/observability.py` | loguru, `trace_id`, costos, LangSmith por env. |

## Backends por variable de entorno

| Etapa | Variable | Valores | Stub |
|---|---|---|---|
| VAD | `VAD_BACKEND` | `energy` \| `silero` | `energy` (numpy) |
| STT | `WHISPER_BACKEND` | `stub` \| `local` \| `api` | `stub` |
| LLM | `ANTHROPIC_API_KEY` | vacío → stub | `StubReasoner` |
| TTS | `TTS_BACKEND` | `stub` \| `elevenlabs` \| `xtts` | `stub` |
| Persistencia | `LOGGER_BACKEND` | `memory` \| `sqlite` \| `postgres` | `memory` |

Las dependencias pesadas (`openai-whisper`, `silero-vad`, `coqui-tts`, todas
arrastran torch) son extras opcionales (`pip install ".[stt]"`, `[vad]`, `[tts]`
o `[audio]`) y se importan dentro del constructor de cada clase. Pedir un
backend cuya librería no está instalada lanza `ConfigurationError` con el
comando de instalación: no hay degradación silenciosa al stub. La única
excepción es el VAD, que es consultivo en el flujo HTTP (el cliente ya segmenta
turnos) y cae a `energy` con un `WARNING`.

## Manejo de errores

| Excepción | Origen | HTTP |
|---|---|---|
| `ValidationError` (Pydantic) | esquema de entrada | 422 con detalle por campo |
| `AudioDecodeError` | WAV ilegible tras pasar el esquema | 422 `invalid_audio` |
| `ReasonerError` | API de Anthropic agotó reintentos, 4xx o JSON inválido | 502 `reasoner_unavailable` |
| `STTError` / `TTSError` | Whisper API / ElevenLabs tras reintentos | 502 |
| `RateLimitExceeded` | slowapi | 429 |

No hay `except Exception`: cada backend captura las excepciones de su cliente
(`anthropic.*`, `httpx.*`) y las convierte en su error tipado.

## Persistencia

`TurnRecord` guarda `session_id`, `trace_id`, turno, transcript y respuesta ya
redactados, acción, slots, latencia por nodo, uso/costo y backends. Los tres
backends comparten el `Protocol` `ConversationLogger`; el de Postgres usa
`psycopg_pool.ConnectionPool` (inyectable para tests) y JSONB.

## Qué no está aquí

- El canal telefónico/WebRTC de producción (LiveKit o Twilio) no está
  implementado: el demo usa Web Audio en el navegador y HTTP por turno. Las
  variables `LIVEKIT_*`/`TWILIO_*` existen en `.env.example` como reserva.
- No hay RAG sobre una base de conocimiento: la tarea de reserva no lo necesita
  y se documenta como trabajo futuro en el README.
