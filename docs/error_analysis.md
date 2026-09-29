# Análisis de errores

Fuente: `eval/runs/2026-09-29-offline-stub.json` (100 conversaciones sintéticas,
**fallback determinista sin LLM**). El sistema falla en 10 de 100; las diez están
en el mismo escenario. No existe todavía una corrida con Claude, así que este
análisis describe el comportamiento del stub y anticipa los modos de fallo que
esperamos del reasoner real.

## Los diez peores casos

| # | id | Escenario | Esperado | Obtenido | Slots capturados | Causa raíz |
|---|---|---|---|---|---|---|
| 1 | conv-091 | spanish | book | ask_clarification | party_size=5, time=21:00 | `_extract_date` no reconoce "mañana"; `_extract_name` no ancla "Mi nombre es". |
| 2 | conv-092 | spanish | book | ask_clarification | party_size=2, time=18:45 | "El viernes" no está en la lista de días; nombre no anclado. |
| 3 | conv-093 | spanish | book | ask_clarification | party_size=4, time=19:00 | Igual que 2. |
| 4 | conv-094 | spanish | book | ask_clarification | party_size=8, time=21:00 | "A nombre de" no es un anclaje conocido. |
| 5 | conv-095 | spanish | book | ask_clarification | party_size=4, time=21:00 | "Para mañana" y "A nombre de". |
| 6 | conv-096 | spanish | book | ask_clarification | party_size=6, time=12:00 | Igual que 2. |
| 7 | conv-097 | spanish | book | ask_clarification | party_size=4, time=20:00 | Igual que 2. |
| 8 | conv-098 | spanish | book | ask_clarification | party_size=5, time=21:00 | Igual que 5. |
| 9 | conv-099 | spanish | book | ask_clarification | party_size=5, time=21:00 | Igual que 2. |
| 10 | conv-100 | spanish | book | ask_clarification | party_size=4, time=20:00 | Igual que 2. |

Patrón: el stub sí captura `party_size` (los dígitos son universales) y `time`
(formato `HH:MM`), pero nunca `date` ni `name` porque sus expresiones regulares
son léxicas y en inglés. El agente se queda preguntando la fecha en bucle hasta
que se acaban los turnos guionizados.

## Hipótesis y qué haríamos

1. **Cobertura léxica del stub.** Añadir sinónimos en español (`mañana`, `hoy`,
   días de la semana, anclajes `a nombre de` / `mi nombre es`) llevaría el stub
   a ~100 % en este eval set, pero sería ajustar el baseline al conjunto de
   evaluación. Lo dejamos como está: el stub es el *fallback*, no el producto.
2. **El reasoner real resuelve esto por diseño.** `ClaudeReasoner` recibe el
   transcript sin ningún supuesto de idioma y el prompt pide slots normalizados
   (`time` en `HH:MM`, `date` tal como lo dijo el usuario). La comprobación
   pendiente (requiere `ANTHROPIC_API_KEY`) es que `date` quede como `mañana`
   y no como `tomorrow`: el ground truth sintético espera la forma inglesa,
   así que el comparador de slots deberá aceptar equivalencias antes de la
   corrida con Claude.
3. **Detección de idioma en STT.** Whisper devuelve `language`; hoy la API lo
   pasa a TTS pero el reasoner no lo usa. Propagar `language` al prompt evita
   respuestas en inglés a un cliente que habló en español.

## Modos de fallo que esperamos con componentes reales (aún sin medir)

- **Alucinación de slots por el LLM**: el criterio 2 de la rúbrica
  (`slot_correctness`) y el criterio 6 del prompt ("never invent slot values")
  existen para detectarlo; la corrida con `claude_zero_shot_stateless` mide
  cuánto empeora sin estado de slots.
- **Errores de STT en nombres propios**: en el fixture manual de WER el único
  error de sustitución "grave" es `Lopez → Lopes`; en producción el read-back
  antes de reservar (turno de confirmación) es la mitigación.
- **Confirmaciones ambiguas** ("yeah, but…"): `AFFIRM_RE` y `NEGATE_RE` son
  mutuamente excluyentes por el primer token; un "yes, but change the time" se
  reservaría con la hora vieja. Pendiente: tratar cualquier corrección detectada
  como negación aunque empiece por "yes".
- **Latencia**: con stubs el p95 es 5 ms; el objetivo <800 ms depende por
  completo de STT y TTS reales (ver `docs/performance.md`).

## Casos que el stub resuelve pero con calidad discutible

- Naturalidad 3/5 en el juez heurístico: la frase de reserva final tiene tres
  oraciones ("Great… We'll send… Goodbye!"); el criterio 5 penaliza más de dos.
- `one_shot`: si el usuario dice todo en una frase el agente igual lee la
  reserva y pide confirmación, lo que añade un turno (aceptado por diseño:
  criterio 4 permite `turnos guionizados + 1`).
