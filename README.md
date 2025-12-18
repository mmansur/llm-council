# LLM Council

A **LangGraph** implementation of [Andrej Karpathy's LLM Council](https://github.com/karpathy/llm-council) concept.

## What is LLM Council?

Instead of asking a question to a single LLM, you can group multiple LLMs into a "council". The council works together through a 3-stage process:

1. **Stage 1: First Opinions** - The user query is given to all LLMs individually, and responses are collected
2. **Stage 2: Peer Review** - Each LLM reviews and ranks the other LLMs' anonymized responses
3. **Stage 3: Chairman Synthesis** - A designated Chairman LLM synthesizes all responses and rankings into a final answer

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER QUERY                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 1: First Opinions                       │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                   │
│  │  GPT-4o  │    │  Claude  │    │  Gemini  │   (Parallel)      │
│  └──────────┘    └──────────┘    └──────────┘                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 2: Peer Review                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                   │
│  │  GPT-4o  │    │  Claude  │    │  Gemini  │   (Parallel)      │
│  │  ranks   │    │  ranks   │    │  ranks   │                   │
│  └──────────┘    └──────────┘    └──────────┘                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 STAGE 3: Chairman Synthesis                      │
│                    ┌──────────────┐                              │
│                    │   Chairman   │                              │
│                    │   (GPT-4o)   │                              │
│                    └──────────────┘                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FINAL ANSWER                                │
└─────────────────────────────────────────────────────────────────┘
```

## Setup

### Prerequisites

- Python 3.11+
- [OpenRouter API Key](https://openrouter.ai/keys) (provides access to multiple LLM providers)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/llm-council.git
cd llm-council
```

2. Install dependencies:
```bash
pip install -r llm_council/requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

### Running Locally

```bash
python -m llm_council.agent "What is the meaning of life?"
```

## Deploy to LangSmith

This repository is configured for deployment to [LangSmith](https://smith.langchain.com/).

### Repository Structure

```
llm-council/
├── llm_council/                 # Main package
│   ├── utils/                   # Utilities
│   │   ├── __init__.py
│   │   ├── nodes.py            # Node functions
│   │   └── state.py            # State definitions
│   ├── requirements.txt        # Dependencies
│   ├── __init__.py
│   └── agent.py                # Graph definition
├── .env.example                # Environment variables template
├── langgraph.json              # LangGraph configuration
└── README.md
```

### Deployment Steps

1. Push this repository to GitHub
2. Go to [LangSmith](https://smith.langchain.com/)
3. Navigate to **Deployments** → **New Deployment**
4. Connect your GitHub repository
5. Set the required environment variable:
   - `OPENROUTER_API_KEY`: Your OpenRouter API key
6. Deploy!

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENROUTER_API_KEY` | Yes | - | Your OpenRouter API key |
| `COUNCIL_MODELS` | No | `openai/gpt-4o,anthropic/claude-3.5-sonnet,google/gemini-2.0-flash-exp:free` | Comma-separated list of council models |
| `CHAIRMAN_MODEL` | No | `openai/gpt-4o` | Model to use as chairman |

### Available Models (OpenRouter)

| Model ID | Provider | Description |
|----------|----------|-------------|
| `openai/gpt-4o` | OpenAI | Most capable, multimodal |
| `openai/gpt-4o-mini` | OpenAI | Smaller, faster, cheaper |
| `anthropic/claude-3.5-sonnet` | Anthropic | Excellent reasoning |
| `anthropic/claude-3-opus` | Anthropic | Most capable Claude |
| `google/gemini-2.0-flash-exp:free` | Google | Free tier Gemini |
| `meta-llama/llama-3.1-70b-instruct` | Meta | Open source, powerful |
| `mistralai/mistral-large` | Mistral | European flagship |

See [OpenRouter Models](https://openrouter.ai/models) for the complete list.

## How It Works

### State Schema

The workflow uses a `CouncilState` TypedDict with the following fields:

- `user_query`: The original question
- `stage1_responses`: List of responses from each council member
- `stage2_rankings`: List of rankings from each council member
- `label_to_model`: Mapping from anonymized labels to model names
- `final_response`: The chairman's synthesized answer
- `aggregate_rankings`: Calculated average rankings for each model

### Graph Structure

The LangGraph workflow consists of:

1. **Parallel Stage 1 nodes** - One per council model, collecting initial responses
2. **Stage 1 collector** - Synchronization point
3. **Parallel Stage 2 nodes** - One per council model, performing peer review
4. **Stage 2 collector** - Synchronization point
5. **Chairman synthesis** - Final answer generation

## References

- [Original LLM Council by Andrej Karpathy](https://github.com/karpathy/llm-council)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [OpenRouter](https://openrouter.ai/)

## License

MIT License
