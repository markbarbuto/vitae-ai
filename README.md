# Vitae

Vitae is a self-hosted chat engine that answers questions from a folder of documents you wrote, plus a React chat component you add to your own site.

Each owner runs one backend for one site. The backend holds one knowledge pack and one profile. The profile sets the assistant's voice, the topics it may discuss, its refusal wording, and the suggested questions. Every question passes through guards before the model sees it, and every sentence of the answer is checked before it reaches the screen. Models run locally on CPU through llama.cpp. Nothing is sent to a hosted AI service.

Vitae is in early development. Setup and usage instructions will be added once the backend and the chat component can run.

## Repository layout

- `backend/` is the Python backend: the API contract, the server settings, the HTTP API, and the guard pipeline every request passes through. It also holds an example profile (`backend/profiles/`) and the benchmarks that measure the guards against public datasets (`backend/bench/`).
- `contract/` holds files generated from that contract: a JSON Schema, TypeScript types for the chat component, and example streams. Regenerate them with `npm run contract`.
- `docs/learn/` explains how the system works, with diagrams. Start with `00-system-design.md`. `docs/adr/` records the design decisions.
- `THIRD_PARTY_NOTICES.md` lists the models the server runs and the datasets used to evaluate it, with their licences.
- `tests/evals/` is a black-box eval suite and latency probe built on promptfoo. It sends questions to a running backend over HTTP and checks what a visitor would see.
- `tests/visual/` holds the reference screenshots the chat component must match, and the script that captures and compares them.

## License

MIT. See `LICENSE`.
