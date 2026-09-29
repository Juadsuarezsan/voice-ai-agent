"""Render ``eval/RESULTS.md`` from a saved run.

Only numbers present in the run JSON are printed. Anything that needs an API
key, a GPU or a blocked dataset stays ``pendiente (...)``.
"""

from __future__ import annotations

from typing import Any

PENDING_WER_EN = "pendiente (LibriSpeech: openslr.org bloqueado; requiere extra `[stt]` y `scripts/wer_benchmark.py --librispeech-dir`)"
PENDING_WER_ES = "pendiente (Common Voice es gated en HF; requiere token y `--common-voice-dir`)"
PENDING_MOS = "pendiente (NISQA + ELEVENLABS_API_KEY / XTTS-v2)"
PENDING_KEY = "pendiente (requiere ANTHROPIC_API_KEY)"
PENDING_COST = "pendiente (medición real requiere ANTHROPIC_API_KEY + ELEVENLABS_API_KEY; estimación con supuestos en docs/scalability.md)"


def _pct(x: float) -> str:
    return f"{x * 100:.1f} %"


def render_results(run: dict[str, Any]) -> str:
    """Build the Markdown report for one run dictionary."""
    stub = run["baselines"]["stub_heuristic"]
    judge = run["judge"]["heuristic"]
    vad = {r["vad"]: r for r in run["ablations"]["vad"]}
    claude = run["baselines"].get("claude_pipeline")
    stateless = run["baselines"].get("claude_zero_shot_stateless")
    lines: list[str] = []
    lines.append("# Resultados de evaluación — Voice AI Conversational Agent")
    lines.append("")
    lines.append(
        f"Generado por `python -m eval.run` el {run['generated_at']} a partir de `{run['run_file']}`."
    )
    lines.append("")
    lines.append(
        "**Aviso obligatorio.** Todas las cifras medidas en este archivo provienen del "
        "**fallback determinista, sin LLM** (`StubReasoner`, STT/TTS stub). No representan "
        "el rendimiento del sistema con Claude, Whisper y ElevenLabs; esas celdas quedan como "
        "`pendiente (...)` hasta que exista una corrida con llaves guardada en `eval/runs/`."
    )
    lines.append("")
    lines.append("## Tabla comparativa obligatoria")
    lines.append("")
    lines.append("| Componente | Métrica | Valor |")
    lines.append("|---|---|---|")
    lines.append(f"| STT | WER sobre LibriSpeech test-clean | {PENDING_WER_EN} |")
    lines.append(f"| STT | WER en español Common Voice | {PENDING_WER_ES} |")
    resolution_cell = (
        f"{_pct(stub['resolution_rate'])} ({stub['resolved']}/{stub['n']}) — "
        "**fallback determinista, sin LLM**; con Claude: "
        + (_pct(claude["resolution_rate"]) if claude else PENDING_KEY)
    )
    lines.append(f"| LLM | Resolution rate sobre 100 conversaciones | {resolution_cell} |")
    lines.append(f"| TTS | MOS automated | {PENDING_MOS} |")
    lines.append(
        f"| Sistema | Latencia p95 end-to-end por turno | {stub['latency_p95_ms']} ms — "
        "**solo stubs (sin STT/TTS/LLM reales)**; con componentes reales: pendiente (llaves + GPU) |"
    )
    lines.append(f"| Sistema | Costo por minuto de conversación | {PENDING_COST} |")
    lines.append("")
    lines.append("## Baselines (100 conversaciones sintéticas, `data/eval/conversations.jsonl`)")
    lines.append("")
    lines.append(
        "| Baseline | Resolution rate | Reservas resueltas | Transferencias correctas | Slots exactos | Turnos medios | p95 ms | Tokens in/out | Costo USD |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")

    def _row(name: str, r: dict[str, Any] | None, pending: str) -> str:
        if not r:
            return f"| {name} | {pending} | – | – | – | – | – | – | – |"
        return (
            f"| {name} | {_pct(r['resolution_rate'])} | {_pct(r['booking_resolution_rate'])} "
            f"({r['booking_n']}) | {_pct(r['transfer_accuracy'])} ({r['transfer_n']}) | "
            f"{_pct(r['slot_exact_match_rate'])} | {r['avg_turns_to_finish']} | {r['latency_p95_ms']} | "
            f"{r['total_input_tokens']}/{r['total_output_tokens']} | {r['total_cost_usd']} |"
        )

    lines.append(_row("stub_heuristic (fallback determinista, sin LLM)", stub, ""))
    lines.append(_row("claude_zero_shot_stateless", stateless, PENDING_KEY))
    lines.append(_row("claude_pipeline (producción)", claude, PENDING_KEY))
    lines.append("")
    lines.append("### Por escenario (stub_heuristic)")
    lines.append("")
    lines.append("| Escenario | n | Resueltas | Tasa |")
    lines.append("|---|---|---|---|")
    for scenario, m in stub["per_scenario"].items():
        lines.append(
            f"| {scenario} | {int(m['n'])} | {int(m['resolved'])} | {_pct(m['resolution_rate'])} |"
        )
    lines.append("")
    lines.append("## LLM-as-judge (rúbrica numerada en `src/eval/judge.py`)")
    lines.append("")
    lines.append(
        "| Juez | n | Pass rate (crit. 1-3) | 1 Tarea | 2 Slots | 3 Sin transferencia innecesaria | 4 Eficiencia | 5 Naturalidad (1-5) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    lines.append(
        f"| heurístico (proxy offline, no es LLM) | {judge['n']} | {_pct(judge['pass_rate'])} | "
        f"{_pct(judge['task_completion'])} | {_pct(judge['slot_correctness'])} | "
        f"{_pct(judge['no_unnecessary_transfer'])} | {_pct(judge['efficiency'])} | {judge['naturalness_mean']} |"
    )
    cj = run["judge"].get("claude")
    if cj:
        lines.append(
            f"| Claude | {cj['n']} | {_pct(cj['pass_rate'])} | {_pct(cj['task_completion'])} | "
            f"{_pct(cj['slot_correctness'])} | {_pct(cj['no_unnecessary_transfer'])} | "
            f"{_pct(cj['efficiency'])} | {cj['naturalness_mean']} |"
        )
    else:
        lines.append(f"| Claude (LLM-as-judge) | {PENDING_KEY} | – | – | – | – | – | – |")
    lines.append("")
    lines.append("## Ablaciones")
    lines.append("")
    lines.append("### VAD on/off (audio sintético: 10 turnos con voz + 10 de silencio, STT stub)")
    lines.append("")
    lines.append(
        "| VAD | Silencios rechazados | Voz que pasa | Llamadas STT ahorradas | Sobrecarga VAD media (ms) |"
    )
    lines.append("|---|---|---|---|---|")
    for key in ("energy", "off"):
        r = vad[key]
        lines.append(
            f"| {key} | {r['silence_rejected']}/{r['n_silence']} ({_pct(r['silence_rejection_rate'])}) | "
            f"{r['speech_passed']}/{r['n_speech']} ({_pct(r['speech_pass_rate'])}) | {r['stt_calls_saved']} | "
            f"{r['vad_overhead_ms_mean']} |"
        )
    lines.append("")
    lines.append("### Whisper tiny vs base (CPU)")
    lines.append("")
    lines.append(f"{run['ablations']['whisper']}")
    lines.append("")
    lines.append("## Latencia por nodo (stubs, media sobre todos los turnos)")
    lines.append("")
    lines.append("| Nodo | Media ms |")
    lines.append("|---|---|")
    for node, ms in run["latency_per_node_ms"].items():
        lines.append(f"| {node} | {ms} |")
    lines.append("")
    lines.append("## Peores casos")
    lines.append("")
    lines.append("Análisis con hipótesis en `docs/error_analysis.md`. Fallos de la corrida:")
    lines.append("")
    for f in run["failures"][:10]:
        lines.append(
            f"- `{f['id']}` ({f['scenario']}): esperado `{f['expected_action']}`, obtenido `{f['final_action']}`; slots {f['slots']}"
        )
    if not run["failures"]:
        lines.append("- ninguno")
    lines.append("")
    lines.append("## Reproducir")
    lines.append("")
    lines.append("```bash")
    lines.append(
        "python scripts/build_eval_set.py   # regenera data/eval/conversations.jsonl (seed fijo)"
    )
    lines.append(
        "python -m eval.run                 # escribe eval/runs/<fecha>-<nombre>.json y este archivo"
    )
    lines.append(
        "python -m eval.run --check         # verifica que las métricas deterministas coinciden"
    )
    lines.append("```")
    lines.append("")
    return "\n".join(lines)
