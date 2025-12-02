# Active Dreaming Memory (ADM)

A biologically-inspired dual-store memory system for lifelong learning in LLM-based autonomous agents.

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install openai chromadb tiktoken requests
   ```

2. **Set your Groq API key:**
   ```bash
   export GROQ_API_KEY="your-groq-api-key-here"
   ```

3. **Run the demo:**
   ```bash
   python -m scalable_agent.main
   ```

## 📂 Project Structure

- **`scalable_agent/`** - Main implementation
  - `agent.py` - Lifelong learning agent
  - `core/` - Dreamer, Reflector, Executor modules
  - `memory/` - Vector store and hybrid retrieval
  - `baselines.py` - Baseline agents for comparison
  - `ablation.py` - Ablation study experiments

## 🧪 Running Experiments

### Demo
```bash
python -m scalable_agent.main
```

### Ablation Study
```bash
python -m scalable_agent.ablation
```

## 🔑 Key Features

- **Active Dreaming**: Counterfactual verification of learned rules
- **Hybrid Retrieval**: Dense + symbolic memory search
- **Dual-Store Architecture**: Episodic + semantic memory
- **Production-Ready**: ChromaDB backend with sub-second retrieval

## 📄 License

MIT License
