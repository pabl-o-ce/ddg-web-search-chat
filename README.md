---
title: Chat with DuckDuckGo Agent
emoji: 🦆
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 4.36.1
app_file: app.py
pinned: true
license: apache-2.0
header: mini
fullWidth: true
short_description: Chat llama-cpp-agent that can search the web.
models:
- mistralai/Mistral-7B-Instruct-v0.3
- meta-llama/Meta-Llama-3-8B-Instruct
---

# Chat with DuckDuckGo Agent 🦆

[![Open In Spaces](https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue.svg)](https://huggingface.co/spaces/poscye/ddg-web-search-chat)
[![Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

A Gradio-based web interface that allows users to chat with an AI agent capable of searching the web using DuckDuckGo. The agent uses llama.cpp to run various open-source language models locally.

## Features

- Web search capabilities using DuckDuckGo
- Support for multiple LLM models:
  - Mistral 7B Instruct v0.3
  - Mixtral 8x7B Instruct v0.1
  - Meta Llama 3 8B Instruct
- Real-time chat interface
- Source citation for responses
- Customizable model parameters
- Dark mode support

## Technical Details

- **Framework**: Gradio 4.36.1
- **Models**: Uses GGUF quantized models
- **Agent**: Powered by llama-cpp-agent
- **Web Scraping**: Uses trafilatura for content extraction
- **Context Window**: 
  - 32k tokens for Mistral and Mixtral
  - 8k tokens for Meta Llama 3

## Usage

The chat interface provides several customizable parameters:

- Model selection
- System message
- Max tokens (1-4096)
- Temperature (0.1-1.0)
- Top-p (0.1-1.0)
- Top-k (0-100)
- Repetition penalty (0.0-2.0)

### Example Queries

```
- "latest news about Yann LeCun"
- "Latest news site:github.blog"
- "Where I can find best hotel in Galapagos, Ecuador intitle:hotel"
- "filetype:pdf intitle:python"
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/pabl-o-ce/ddg-web-search-chat
cd ddg-web-search-chat
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the required model files:
The application will automatically download the following models:
- Mistral-7B-Instruct-v0.3-Q6_K.gguf
- Meta-Llama-3-8B-Instruct-Q6_K.gguf
- mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf

4. Run the application:
```bash
python app.py
```

## Configuration

The application can be configured through various parameters in `settings.py`:
- Model context limits
- Message formatter types for different models
- System prompts and templates

## UI Customization

The interface uses a custom theme with:
- Orange primary color
- Amber secondary color
- Dark mode support
- Custom CSS for message bubbles and layout

## License

Apache 2.0

## Links

- [Discord Community](https://discord.gg/fgr5RycPFP)
- [GitHub Repository](https://github.com/Maximilian-Winter/llama-cpp-agent)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

*Note: This project is powered by llama-cpp-agent and uses Hugging Face Spaces for deployment.*