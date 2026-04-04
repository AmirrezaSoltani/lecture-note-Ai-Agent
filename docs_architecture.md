# Architecture Notes

## MVP implemented here
- FastAPI web panel
- SQLite persistence
- Background thread processing
- Rule-based emphasis detection
- Structured notes renderer

## Next production steps
1. Replace in-process worker with Redis queue
2. Move ASR to dedicated GPU workers
3. Store transcripts / notes in PostgreSQL + object storage
4. Add word timestamps and speaker diarization as optional stages
5. Add constrained LLM post-processing only for title cleanup and ambiguity resolution
6. Add edit history and reviewer workflow in the web panel
