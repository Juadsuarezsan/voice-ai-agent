# Project 09 — Voice AI Conversational Agent

> Telephony customer service over voice: Whisper STT → Claude reasoning → ElevenLabs TTS. Target under-800ms per turn for natural conversation. The pattern behind Bland AI, Vapi, Retell, Hume, Pindrop.

[![Status](https://img.shields.io/badge/status-planned-fbbf24)]()
[![STT](https://img.shields.io/badge/STT-Whisper--large--v3-22d3ee)]()
[![TTS](https://img.shields.io/badge/TTS-ElevenLabs%20%2F%20XTTS--v2-7c5cff)]()

**Industrial use case:** Voice AI is one of the fastest-growing AI verticals — restaurant booking, medical scheduling, first-line customer support, conversational commerce.

## What this project does

Closes the voice loop: caller speaks → VAD detects end of turn → Whisper transcribes → Claude reasons (with optional KB retrieval) → ElevenLabs synthesizes voice response → stream back to caller. Target latency under 800 ms end-to-end per turn for natural conversation.

## Architecture

```
Audio input (WebRTC / phone / file)
   │
   ▼
[VAD] Silero VAD detects turn end
   │
   ▼
[STT] Whisper-large-v3
   │ → text + confidence + detected language
   │
   ▼
[Intent + Context] LangGraph state
   │ ├─ conversation history
   │ ├─ slots completed (date, party_size, etc.)
   │ └─ current task type
   │
   ▼
[Reasoner] Claude → next action: respond | ask_clarification | book | transfer_to_human
   ├──► if needs info: RAG over KB
   └──► if needs action: tool call (booking API, etc.)
   │
   ▼
[Response Generator] Claude → natural conversational text
   │
   ▼
[TTS] ElevenLabs or XTTS-v2 → streamed audio
   │
   ▼
[Audio Stream] back to caller
   │
   ▼
[Conversation Logger] PostgreSQL — transcript, audio refs, latencies per turn
```

## Roadmap to v1.0.0

1. [ ] Whisper-large-v3 local setup (GPU) or Whisper API integration
2. [ ] Silero VAD for end-of-turn detection
3. [ ] LangGraph state with conversation history + slots
4. [ ] Claude reasoning with structured output (action + parameters)
5. [ ] Tool integrations (mock booking API for demo)
6. [ ] ElevenLabs TTS with streaming
7. [ ] XTTS-v2 local fallback for cost-free demos
8. [ ] WER eval on LibriSpeech test-clean + test-other + Spanish Common Voice
9. [ ] Resolution rate eval over 100 simulated conversations (LLM-as-judge)
10. [ ] Next.js demo with WebRTC for browser audio
11. [ ] 10 recorded conversations (MP3 in repo) of real task completions

## Stack

| Layer | Technology |
|---|---|
| STT | openai-whisper `whisper-large-v3` (local, GPU) or Whisper API ($0.006/min) |
| LLM | Claude Sonnet 4.5 |
| TTS | ElevenLabs API ($0.30/1K chars) or coqui-tts XTTS-v2 (local) |
| Orchestration | LangGraph with turn detection |
| Streaming | LiveKit Cloud or Twilio Voice for production |
| VAD | silero-vad |
| Storage | PostgreSQL for transcripts, S3/filesystem for audio |
| Frontend | Next.js + WebRTC for browser audio |
| Observability | LangSmith + custom per-component latency metrics |

## Definition of Done — project-specific

- [ ] WER reported on at least 2 benchmarks (LibriSpeech + Spanish dataset)
- [ ] End-to-end per-turn latency measured + optimized under 1 second
- [ ] 100 simulated conversations evaluated with LLM-as-judge for resolution rate
- [ ] Demo: WebRTC interface — speak in browser, receive voice response
- [ ] 10 real recorded conversations (MP3) included in repo as evidence
- [ ] Whisper open vs API comparison with costs documented
- [ ] XTTS-v2 vs ElevenLabs comparison with MOS documented
- [ ] Demo conversation lasts ≥6 turns completing a real task

Plus the 12 universal DoD blocks.

## License

MIT.
