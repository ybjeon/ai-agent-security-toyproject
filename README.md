# ai-agent-security-toyproject

## about AI agent security
- [Blog link]()
 
## Test files
- test_llm.py : make basic LLM model
- test_rag.py: make LLM response using RAG
- test_toolcall.py: make agent with pseudo tool call
- test_planner.py: basic planner & executor 
- test_dpi.py: simulate direct prompt injection
- test_ipi.py: simulate indirect prompt injection
- test_origintracing.py: defending with instruction extraction & origin tracing
- test_guardrail.py: input/output filter with guardrail
- TBD


## System configuration
- LLM model served by Ollama

## Basic python framework
- langchain
- fastapi
- langserve
- uv
- ollama (only for model)
- openai

## LLM model specs
| Model | Ollama Name | Size | Parameters | Memory (RAM) | VRAM (GPU) |
|-------|------------|------|------------|--------------|-----------|
| Mistral | mistral:7b | 7B | 7 Billion | 16GB | 6GB |
| Qwen 3.6 | qwen:7b | 7B | 7 Billion | 16GB | 6GB |
| Qwen 3.6 | qwen:14b | 14B | 14 Billion | 32GB | 10GB |
| Llama 3.2 | llama3.2:1b | 1B | 1 Billion | 4GB | 2GB |
| Llama 3.2 | llama3.2:3b | 3B | 3 Billion | 8GB | 4GB |
| Llama 3.1 | llama3.1:8b | 8B | 8 Billion | 16GB | 6GB |
| Llama 3.1 | llama3.1:70b | 70B | 70 Billion | 64GB | 40GB |

## Installation

### Virtual Environment (uv)
```bash
# install uv 
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install packages
uv pip install ollama openai langchain langserve
```

### Ollama Installation and Running
```bash
# Install Ollama (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Run Ollama server
ollama serve
```

### Download Model
```bash
# Default model (test_llm.py default)
ollama pull mistral
```

### Run
```bash
python test_llm.py
```