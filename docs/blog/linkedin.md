# Borrador de post para LinkedIn

Acabo de publicar el noveno proyecto de mi portafolio de AI Engineering: un agente conversacional por voz para reservas de restaurante.

Qué hace: el cliente habla, un VAD decide cuándo terminó, Whisper transcribe, Claude decide el siguiente paso (preguntar un dato, confirmar, reservar o pasar a una persona) y ElevenLabs responde con voz. Todo orquestado como un grafo de LangGraph con latencia, tokens y costo medidos por nodo.

Tres cosas que aprendí construyéndolo:

1. En voz, el turno de confirmación no es opcional. Los errores de STT en nombres y horas son los más comunes; leer la reserva antes de cerrarla cuesta un turno y evita reservas mal hechas.

2. La instalación base no debe arrastrar torch. Whisper, Silero y XTTS viven en extras opcionales e imports perezosos; la API corre en un contenedor de ~300 MB y crece hacia GPU solo donde hace falta.

3. Una métrica sin corrida guardada no existe. `python -m eval.run` regenera la tabla de resultados desde un JSON versionado, y la CI falla si los números cambian sin regenerarla. Hoy la tabla dice con claridad qué está medido (el fallback determinista, sin LLM: 90/100 conversaciones sintéticas, 0/10 en español) y qué está pendiente de llaves y GPU (WER, MOS, latencia real).

Stack: Python 3.11, FastAPI, LangGraph, SDK de Anthropic con salida JSON validada y reintentos con tenacity, httpx para Whisper API y ElevenLabs, PostgreSQL/SQLite para los transcripts con PII seudonimizada, pytest (149 tests, 97 % de cobertura), mypy --strict, gitleaks en CI.

Repo: https://github.com/Juadsuarezsan/voice-ai-agent
Demo (casos pre-generados y modo en vivo con micrófono): https://juadsuarezsan.github.io/voice-ai-agent/demo/

#AIEngineering #VoiceAI #LangGraph #Claude #Whisper #Python
