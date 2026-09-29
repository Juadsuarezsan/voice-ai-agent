# Decisiones técnicas — Voice AI Conversational Agent

Cada entrada registra la alternativa rechazada y el criterio. Si una decisión
cambia, se añade una entrada nueva en lugar de editar la anterior.

## 1. Whisper local como camino principal, Whisper API como opción medida

Rechazado: usar solo la API hospedada. Motivo: la spec exige WER sobre
LibriSpeech y Common Voice con un modelo controlado; `whisper-large-v3` local
en GPU permite reproducir el benchmark sin costo por minuto y sin enviar audio
a un tercero. La API (`WHISPER_BACKEND=api`, `whisper-1`, $0.006/min) queda
implementada con httpx + tenacity para la comparativa de costo del DoD. En CPU
solo son razonables `tiny`/`base`; por eso el extra `[stt]` no se instala en la
imagen base y el eval marca la fila como pendiente.

## 2. ElevenLabs por REST directo, sin el SDK del proveedor

Rechazado: `elevenlabs` (SDK) como dependencia base. Motivo: se usa un único
endpoint (`/v1/text-to-speech/{voice_id}`); httpx ya está en el proyecto y con
él el timeout, los reintentos (429/5xx/transporte) y el mock con `respx` son
explícitos y verificables. XTTS-v2 (coqui) se mantiene como alternativa local
para la comparativa de MOS, tras el extra `[tts]`.

## 3. LangGraph `StateGraph` en lugar de un bucle secuencial

Rechazado: el bucle `for` original (STT → reasoner → TTS). Motivo: el grafo
hace explícita la arista condicional VAD → `no_speech_node`, nombra los nodos
(lo que permite loguear entrada/salida y latencia por nodo con un solo
decorador) y deja preparado el punto de extensión para RAG o `tool_call` como
nodos adicionales sin reescribir el orquestador. Regla defensiva: los nodos se
llaman `*_node` para no colisionar nunca con una clave del estado.

## 4. VAD por energía como predeterminado, Silero opcional

Rechazado: exigir `silero-vad` siempre. Motivo: arrastra torch (GB de disco) y
en el flujo HTTP el cliente ya delimita el turno al soltar el botón; el VAD
sirve para descartar silencios y ahorrar llamadas a STT, algo que el detector
por RMS resuelve (ablación: 10/10 silencios rechazados, 10/10 voces aceptadas
en audio sintético). Silero es el camino para streaming real (LiveKit/Twilio),
donde hay que detectar fin de turno dentro de un flujo continuo.

## 5. Slot filling determinista como fallback, no como producto

Rechazado: eliminar el stub y exigir llave de API. Motivo: el stub hace que CI,
tests, demo y `python -m eval.run` funcionen sin secretos y da un baseline
honesto para el LLM. Se etiqueta siempre como "fallback determinista, sin LLM";
sus 90/100 no se presentan como resolution rate del sistema. Consecuencia: el
stub es solo inglés y falla 10/10 en español por diseño (ver
`docs/error_analysis.md`); no se amplía para maquillar la métrica.

## 6. Confirmación explícita antes de reservar

Rechazado: reservar en cuanto se conocen los cuatro slots. Motivo: en voz los
errores de STT en nombres y horas son los más frecuentes; una lectura de
confirmación cuesta un turno y evita reservas mal hechas. Se modela con
`awaiting_confirmation` en el estado y con acciones de corrección (`No, make
it 6 people`).

## 7. Salida del LLM como JSON validado con Pydantic y tenacity fuera del SDK

Rechazado: confiar en los reintentos automáticos del SDK y en `json.loads`.
Motivo: el SDK reintenta con política propia y opaca; con `max_retries=0` y
tenacity el backoff, los tipos de error reintentables y el número de intentos
están en el código y se prueban con mocks. La respuesta se valida contra
`LLMTurnOutput`; cualquier desvío es `ReasonerError` → 502, nunca un fallback
silencioso al stub (que enmascararía un incidente de producción).

## 8. Modelo pinneado por fecha

`ANTHROPIC_MODEL=claude-sonnet-4-5-20250929`. Rechazado: alias sin fecha.
Motivo: reproducibilidad de `eval/runs/` y del costo por token.

## 9. Redacción de PII en el límite de persistencia, no en el turno

Rechazado: anonimizar antes del reasoner. Motivo: el agente necesita el nombre
para completar la reserva. Se seudonimiza (`NAME_<hash>`) y se enmascaran
teléfonos, correos y tarjetas justo antes de escribir en el
`ConversationLogger`, con tests.

## 10. `python -m eval.run --check` en CI

Rechazado: publicar métricas escritas a mano en el README. Motivo: la tabla
de `eval/RESULTS.md` es una función pura del JSON de la corrida; `--check`
vuelve a ejecutar la parte determinista y falla si los números cambian sin
regenerar el archivo. Un test adicional comprueba que `RESULTS.md` es
exactamente el render del último run.

## 11. Umbral de cobertura 70 % con cobertura real 97 %

El umbral es el del DoD; la cobertura alta viene de que cada backend externo
tiene un doble (mock, `respx` o módulo falso) y no de tests triviales.
