from llama_cpp_agent import MessagesFormatterType

def get_context_by_model(model_name):
    model_context_limits = {
        "Mistral-7B-Instruct-v0.3-Q6_K.gguf": 32768,
        "mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf": 32768,
        "Meta-Llama-3-8B-Instruct-Q6_K.gguf": 8192
    }
    return model_context_limits.get(model_name, None)

def get_messages_formatter_type(model_name):
    model_name = model_name.lower()
    if any(keyword in model_name for keyword in ["meta", "aya"]):
        return MessagesFormatterType.LLAMA_3
    elif any(keyword in model_name for keyword in ["mistral", "mixtral"]):
        return MessagesFormatterType.MISTRAL
    elif any(keyword in model_name for keyword in ["einstein", "dolphin"]):
        return MessagesFormatterType.CHATML
    elif "phi" in model_name:
        return MessagesFormatterType.PHI_3
    else:
        return MessagesFormatterType.CHATML