# Un agente de voz que reserva mesas: VAD, Whisper, Claude y TTS en un grafo de LangGraph

*Borrador para Medium / Dev.to. Repo: https://github.com/Juadsuarezsan/voice-ai-agent*

## Por qué voz

Los asistentes de texto ya son rutina; la voz sigue siendo el canal donde
más se nota la ingeniería. Un turno hablado encadena tres modelos distintos
(reconocimiento de voz, lenguaje y síntesis), cada uno con su latencia, su
costo y su modo de fallo, y el usuario espera una respuesta en menos de un
segundo porque así funcionan las conversaciones humanas. Quería un proyecto
que obligara a resolver esa cadena completa, medirla de punta a punta y dejar
un sistema que alguien pudiera clonar y ejecutar sin pedirme una llave de API.

La tarea elegida es deliberadamente acotada: reservar una mesa en un
restaurante por teléfono. Cuatro datos (personas, fecha, hora, nombre), una
confirmación y dos salidas alternativas (pasar a una persona, o rechazar con
elegancia lo que no es una reserva). Es lo bastante pequeña para evaluarla con
rigor y lo bastante real para que aparezcan todos los problemas de la voz.

## La forma del sistema

Cada turno entra por `POST /api/turn` como texto o como WAV en base64 y
recorre un `StateGraph` de LangGraph con cinco nodos:

```
vad_node -> stt_node -> reasoner_node -> tts_node -> logger_node
     \-- (sin voz) --> no_speech_node ---------^
```

El VAD decide si hay voz; si no la hay, el grafo salta directamente a una
respuesta de "no te escuché" sin gastar una llamada de STT ni una de LLM. El
STT transcribe (Whisper local, Whisper API o un stub que devuelve el texto tal
cual). El reasoner recibe el historial, el estado de los slots y el transcript,
y devuelve un JSON con la frase a decir, la acción (`respond`,
`ask_clarification`, `book`, `transfer_to_human`), los slots actualizados y si
la conversación terminó. El TTS convierte la frase en audio. El logger calcula
el costo del turno, seudonimiza el nombre del cliente y enmascara teléfonos y
correos antes de escribir el transcript en PostgreSQL.

Lo que más valor me ha dado del grafo no es la orquestación en sí (un bucle
`for` haría lo mismo con cinco líneas), sino que cada nodo tiene nombre. Un
decorador de diez líneas envuelve todos los nodos, registra entrada y salida
con el `trace_id` del turno y guarda la latencia de cada uno en el estado. Al
final del turno la API devuelve `latency.vad_ms`, `stt_ms`, `llm_ms`, `tts_ms`
y `total_ms`, y con eso el perfil de rendimiento sale gratis.

## Backends intercambiables, instalación ligera

El primer obstáculo fue práctico: `pip install` fallaba porque `openai-whisper`
no construía, y aunque construyera, arrastra torch (gigabytes) a cualquier
contenedor. La regla que adopté es que la instalación base no incluye ningún
modelo local. Whisper, Silero VAD y XTTS-v2 son extras opcionales
(`pip install ".[stt]"`, `[vad]`, `[tts]`) y sus imports ocurren dentro del
constructor de la clase que los usa. La imagen Docker tiene dos targets:
`base` (~300 MB, corre con Claude, Whisper API y ElevenLabs) y `audio` (añade
ffmpeg, torch para CPU y los tres extras).

Cada etapa tiene además un stub determinista: `StubSTT` devuelve el texto que
le pasan, `StubReasoner` hace slot filling por expresiones regulares, `StubTTS`
genera un tono cuya duración depende del texto. Con los stubs, la misma ruta de
código corre en CI, en la demo estática y en la evaluación sin ninguna llave.

Hay una decisión que tomé al revés de lo habitual: pedir un backend cuya
librería no está instalada no degrada al stub, lanza `ConfigurationError` con
el comando de instalación. La versión anterior del proyecto hacía
`except Exception: return StubSTT()`, lo que significaba que un despliegue mal
configurado respondía con transcripciones vacías y nadie se enteraba.

## El reasoner: JSON validado y reintentos explícitos

`ClaudeReasoner` usa el SDK oficial de Anthropic con `max_retries=0` y un
timeout explícito; los reintentos los gestiona tenacity con backoff exponencial
solo para los errores que merecen reintento (429, conexión, timeout, 5xx). La
respuesta del modelo tiene que ser un objeto JSON que valide contra un modelo
de Pydantic (`LLMTurnOutput`); cualquier otra cosa es un `ReasonerError` que la
API convierte en 502. Nada de "si el LLM falla, uso el stub": eso escondería un
incidente detrás de un comportamiento aparentemente normal.

El prompt es corto y prescriptivo: un slot por pregunta, en orden fijo; cuando
están los cuatro, leerlos y pedir confirmación; reservar solo tras un sí
explícito; transferir si piden una persona o algo que no es una reserva; frases
de una o dos oraciones habladas, sin markdown. El modelo está pinneado por
fecha (`claude-sonnet-4-5-20250929`) para que las corridas de evaluación sean
comparables.

El stub sigue exactamente el mismo protocolo, incluida la confirmación. Eso
tiene una consecuencia que me pareció importante: es el baseline contra el que
se medirá Claude, y no lo he "mejorado" para que quede bien en la tabla. Es
solo inglés y falla en las diez conversaciones en español del conjunto de
evaluación. Está documentado, no arreglado.

## Evaluar sin inventar

La spec pide una tabla con WER, resolution rate sobre cien conversaciones,
MOS, latencia p95 y costo por minuto. En el entorno donde construí esto no
había llaves de API, ni GPU, y los hosts de LibriSpeech y Hugging Face estaban
bloqueados. La tentación es escribir números "razonables". La regla del
portafolio es que un número solo existe si lo produjo una corrida guardada en
`eval/runs/`, y esa regla está codificada:

- `python -m eval.run` ejecuta todo lo que puede ejecutar sin llaves (el stub
  sobre las cien conversaciones, un juez heurístico que aplica la rúbrica desde
  la verdad de terreno, y una ablación de VAD encendido/apagado sobre audio
  sintético), escribe un JSON con fecha y regenera `eval/RESULTS.md` desde ese
  JSON. Con `ANTHROPIC_API_KEY` presente, añade a la misma corrida las dos
  líneas base con Claude y el LLM-as-judge.
- `python -m eval.run --check` vuelve a correr la parte determinista y falla si
  los números difieren del JSON guardado. Corre en CI.
- Un test comprueba que `RESULTS.md` es exactamente el render del último JSON.

La tabla resultante dice hoy: resolution rate 90 % (90/100), **fallback
determinista, sin LLM**; latencia p95 5 ms, **solo stubs**; WER, MOS y costo,
pendientes con el motivo exacto. No es la tabla que quiero mostrar en una
entrevista, pero es una tabla que puedo defender.

El conjunto de cien conversaciones es sintético y lo dice en cada registro.
Las frases están escritas a mano (bancos de aperturas, formas de decir el
número de personas, la fecha, la hora, el nombre, confirmaciones, correcciones,
peticiones de hablar con alguien, peticiones fuera de alcance, y diez diálogos
en español); un generador con semilla fija las combina y deriva la verdad de
terreno de los valores elegidos, así que los slots esperados son correctos por
construcción. Un dataset real de diálogos (MultiWOZ 2.4) queda referenciado y
descargable, pero no sustituye a este: sus diálogos son de texto y de otro
dominio de reserva.

La rúbrica del juez tiene cinco criterios numerados con ejemplos: tarea
completada, slots correctos, sin transferencia innecesaria, eficiencia (cada
slot se pregunta como máximo una vez) y naturalidad de 1 a 5. El juez
heurístico puntúa los cuatro primeros desde la verdad de terreno y el quinto
con una regla superficial (más de dos oraciones o markdown penalizan); el juez
con Claude recibe la misma rúbrica y devuelve el mismo JSON. Curiosamente el
proxy ya detecta un defecto real del stub: su frase final de reserva tiene tres
oraciones y saca 3/5.

## Lo que la demo enseñó

La demo estática tiene cinco conversaciones pre-generadas de seis o más turnos
y un modo en vivo que graba con Web Audio, convierte a WAV mono de 16 kHz en el
navegador, lo envía a `/api/turn` y reproduce el audio de vuelta. Los casos
pre-generados los produce un script que ejecuta el pipeline real con los stubs;
no se escriben a mano.

Generarlos destapó un bug: en "For Friday, please." el extractor de nombres
capturaba "Friday" porque la regla era "palabra con mayúscula después de
*for*". La corrección fue añadir días, meses y palabras de cortesía a la lista
de exclusión y cubrirlo con un test de regresión. Otro caso, "Actually, could
it be Priya Patel instead?" durante la confirmación, no cambiaba el nombre;
ahora sí. Que la demo sea salida del código y no texto decorativo es lo que
permite que sirva de prueba.

## Seguridad y observabilidad sin ceremonia

La validación de entrada es de Pydantic y devuelve 422 con detalle: petición
sin texto ni audio, texto de más de 2.000 caracteres, audio de más de 5 MB,
base64 malformado, cabecera que no es RIFF/WAVE, `session_id` con caracteres
raros. `slowapi` limita por IP y CORS solo admite los orígenes de la variable
de entorno. `gitleaks` corre sobre el historial completo en CI; encontró un
falso positivo (el identificador público de una voz de ElevenLabs parece una
clave) que queda documentado en la allowlist.

Cada línea de log lleva el `trace_id` del turno, que también viaja en la
cabecera `X-Trace-Id`; `GET /api/metrics` resume los últimos cien turnos con
p95 y costo acumulado. LangSmith se activa con dos variables de entorno; no lo
he ejecutado porque requiere llave.

## Lo que falta, con nombre

- Medir con componentes reales: resolution rate con Claude y el juez LLM sobre
  las cien conversaciones, WER de whisper `tiny`/`base`/`large-v3` sobre
  LibriSpeech y Common Voice en español, MOS con NISQA para ElevenLabs y XTTS,
  latencia p95 real y costo por minuto medido en lugar de estimado.
- STT incremental y TTS por chunks: con stubs el grafo cuesta 2 ms; el
  presupuesto de 800 ms se lo llevan los tres modelos y solo el solapamiento
  los hace caber.
- Canal telefónico (LiveKit o Twilio) con Silero en streaming, y estado de
  sesión en Redis para más de una réplica.
- Diez llamadas reales grabadas como evidencia.

## Qué me llevo

Tres cosas. Primero, que un fallback determinista es útil justo mientras se le
llame por su nombre; en cuanto se cuela en una tabla como si fuera el sistema,
deja de serlo. Segundo, que los imports perezosos y los extras opcionales no
son una manía de empaquetado: son la diferencia entre un contenedor de 300 MB
que arranca en CI y uno de varios gigas que nadie prueba. Y tercero, que la
observabilidad por nodo cuesta un decorador y devuelve el perfil de latencia,
el costo por turno y el análisis de errores sin trabajo adicional. El resto
(WER, MOS, latencia real) llegará con las llaves, y la tabla ya sabe dónde
ponerlo.
