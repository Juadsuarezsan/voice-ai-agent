# Escalabilidad y costo — Voice AI Conversational Agent

## Dónde está el cuello de botella

Un turno de voz es una cadena serial: STT → LLM → TTS. Con stubs el p95 es
5 ms (`eval/RESULTS.md`), es decir, la orquestación (FastAPI + LangGraph +
logger) no aporta latencia apreciable. Todo el presupuesto de <800 ms se lo
reparten los tres servicios externos, y ninguno se ha medido aquí todavía
(requiere llaves y GPU). Orden de magnitud público de cada proveedor, que no
sustituye a la medición:

| Etapa | Componente | Escala con | Notas |
|---|---|---|---|
| STT | whisper-large-v3 GPU / Whisper API | segundos de audio | En CPU `base` tarda varios segundos por turno; inviable para <800 ms. |
| LLM | Claude Sonnet 4.5, ~150 tokens de salida | tokens de salida y longitud del historial | Se recorta el historial a 12 mensajes y el estado va como JSON compacto. |
| TTS | ElevenLabs / XTTS-v2 | caracteres | El streaming por chunks es la única forma de que el primer byte llegue antes de que termine la síntesis. |

## Estimación de costo por minuto (supuestos explícitos, no medición)

Supuestos: un minuto de conversación tiene ~6 turnos; cada turno envía
~700 tokens de entrada (sistema + historial + estado) y recibe ~60 tokens; la
respuesta hablada tiene ~90 caracteres; el cliente habla ~25 s por minuto.
Precios de lista configurados en `.env.example`.

| Partida | Cálculo | USD/min |
|---|---|---|
| Claude Sonnet 4.5 | 6 × (700 × 3 + 60 × 15) / 1e6 | 0.018 |
| ElevenLabs | 6 × 90 chars × 0.30 / 1000 | 0.162 |
| Whisper API | 25 s / 60 × 0.006 | 0.003 |
| **Total (todo hospedado)** | | **≈ 0.18** |
| Con Whisper local + XTTS-v2 en GPU propia | solo Claude | ≈ 0.02 + amortización de GPU |

El TTS premium domina el costo (≈ 90 %). Con 1.000 usuarios/mes y 5 minutos
por usuario: ≈ 900 USD/mes todo hospedado, ≈ 100 USD/mes más una GPU si STT y
TTS son locales. La medición real (`usage.cost_usd` acumulado en
`GET /api/metrics`) sustituirá estas cifras cuando existan llaves.

## Capacidad actual (una réplica, stubs)

- El proceso es asíncrono; STT local corre en el executor de hilos para no
  bloquear el event loop.
- El estado de sesión vive en memoria del proceso (`VoiceLoop._sessions`): una
  sola réplica o afinidad de sesión.
- `RATE_LIMIT=30/minute` por IP con `slowapi` en memoria.

## 100× (miles de llamadas concurrentes)

- **Estado de sesión** → Redis con TTL (la `SessionState` cabe en <5 KB). El
  `VoiceLoop` ya aísla `_get_or_create`, así que es un cambio local.
- **Rate limit** → `slowapi` con almacenamiento Redis para que sea global.
- **STT/TTS** → servicios separados con autoscaling por GPU; la API solo
  orquesta. Whisper por lotes no aplica en tiempo real; sí aplica *batching*
  de síntesis para respuestas idénticas (caché por hash de texto + voz).
- **Postgres** → `psycopg_pool` ya está; pasar a `AsyncConnectionPool` y a
  particionado por mes de la tabla `turns` (solo escritura, lectura por
  `session_id` indexada).

## 1.000×

- Canal telefónico real (LiveKit/Twilio) con VAD Silero en streaming y STT
  incremental: el turno no espera al fin del audio para transcribir.
- Un modelo más pequeño (Haiku) para los turnos de puro slot filling y Sonnet
  solo cuando el estado cambia de fase; la rúbrica del juez mide si baja la
  resolución.
- Colas para el `logger_node` (escribir el turno fuera del camino crítico).

## Lo que no escala y se acepta

- El stub determinista es un baseline y no debe recibir tráfico real.
- El eval set sintético mide regresiones, no calidad absoluta: la
  validación real es el LLM-as-judge sobre conversaciones con Claude (pendiente
  de llave) y, después, transcripciones de llamadas reales anonimizadas.
