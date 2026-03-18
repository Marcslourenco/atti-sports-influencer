# atti-sports-influencer

**Domain-specific sports AI influencers powered by ATTI core infrastructure.**

## Architecture

This module is the first vertical of the ATTI Influencer platform. It consumes ATTI core services via adapters and never duplicates existing infrastructure.

### Principles
1. **Maximum Reuse** — Uses SoulXEngine, PersonaManager, RAG FAISS, WorkerRouter, Multi-Tenant Core
2. **Extension by Adapters** — All ATTI integration via `src/adapters/`
3. **Isolated Domain** — All sports logic in `src/ingestion/`, `src/engines/`, `src/pipelines/`
4. **Configuration over Code** — Agents defined as YAML in `agents/`

### Structure

```
atti-sports-influencer/
├── src/
│   ├── ingestion/          # Sports data ingestion (football-data.org, RSS, scraping)
│   ├── engines/            # Sentiment, Highlight, Rivalry engines
│   ├── pipelines/          # Content generation pipelines
│   ├── registry/           # Agent Registry (dynamic YAML loader)
│   ├── workers/            # Stateless Worker Pool (text, media)
│   ├── adapters/           # ATTI Core integration adapters
│   ├── social/             # Social media publishing
│   └── api/                # FastAPI endpoints
├── agents/                 # Agent YAML configurations
├── config/                 # Data sources and personas
├── data/                   # Knowledge base
├── tests/                  # Validation tests
└── docker/                 # Docker Compose with Redis
```

### ATTI Integration Points

| ATTI Component | Integration Method | Purpose |
|---|---|---|
| PersonaManager | Adapter (extends) | Load sports personas |
| Knowledge Packages | Adapter (reads) | Sports knowledge domain |
| SoulXEngine | Adapter (consumes) | Personality and tone control |
| WorkerRouter | Extension (subclass) | Route generation tasks |
| Multi-Tenant Core | Adapter (maps) | Each influencer = tenant |

## Quick Start

```bash
pip install -r requirements.txt
# Start Redis
docker-compose -f docker/docker-compose.yml up -d redis
# Run API
python -m src.api.main
```

## License

Proprietary — ATTI Ecosystem
