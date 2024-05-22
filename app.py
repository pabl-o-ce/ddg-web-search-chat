import spaces
import json
import subprocess
import gradio as gr
from huggingface_hub import hf_hub_download

subprocess.run('pip install llama-cpp-python==0.2.75 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124', shell=True)
subprocess.run('pip install llama-cpp-agent==0.2.10', shell=True)

hf_hub_download(
    repo_id="bartowski/Meta-Llama-3-70B-Instruct-GGUF",
    filename="Meta-Llama-3-70B-Instruct-Q3_K_M.gguf",
    local_dir = "./models"
)
hf_hub_download(
    repo_id="bartowski/Llama-3-8B-Synthia-v3.5-GGUF",
    filename="Llama-3-8B-Synthia-v3.5-f16.gguf",
    local_dir = "./models"
)
hf_hub_download(
    repo_id="bartowski/Mistral-7B-Instruct-v0.3-GGUF",
    filename="Mistral-7B-Instruct-v0.3-f32.gguf",
    local_dir = "./models"
)

css = """
.message-row {
    justify-content: space-evenly !important;
}
.message-bubble-border {
    border-radius: 6px !important;
}
.dark.message-bubble-border {
    border-color: #343140 !important;
}
.dark.user {
    background: #1e1c26 !important;
}
.dark.assistant.dark, .dark.pending.dark {
    background: #111111 !important;
}
"""

def get_messages_formatter_type(model_name):
    from llama_cpp_agent import MessagesFormatterType
    if "Llama" in model_name:
        return MessagesFormatterType.LLAMA_3
    elif "Mistral" in model_name:
        return MessagesFormatterType.MISTRAL
    else:
        raise ValueError(f"Unsupported model: {model_name}")

@spaces.GPU(duration=120)
def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
    top_k,
    repeat_penalty,
    model,
):
    from llama_cpp import Llama
    from llama_cpp_agent import LlamaCppAgent
    from llama_cpp_agent.providers import LlamaCppPythonProvider
    from llama_cpp_agent.chat_history import BasicChatHistory
    from llama_cpp_agent.chat_history.messages import Roles

    chat_template = get_messages_formatter_type(model)

    llm = Llama(
        model_path=f"models/{model}",
        flash_attn=True,
        n_threads=40,
        n_gpu_layers=81,
        n_batch=1024,
        n_ctx=8192,
    )
    provider = LlamaCppPythonProvider(llm)

    agent = LlamaCppAgent(
        provider,
        system_prompt=f"{system_message}",
        predefined_messages_formatter_type=chat_template,
        debug_output=True
    )
    
    settings = provider.get_provider_default_settings()
    settings.temperature = temperature
    settings.top_k = top_k
    settings.top_p = top_p
    settings.max_tokens = max_tokens
    settings.repeat_penalty = repeat_penalty
    settings.stream = True

    messages = BasicChatHistory()

    for msn in history:
        user = {
            'role': Roles.user,
            'content': msn[0]
        }
        assistant = {
            'role': Roles.assistant,
            'content': msn[1]
        }
        messages.add_message(user)
        messages.add_message(assistant)
    
    stream = agent.get_chat_response(
        message,
        llm_sampling_settings=settings,
        chat_history=messages,
        returns_streaming_generator=True,
        print_output=False
    )
    
    outputs = ""
    for output in stream:
        outputs += output
        yield outputs

demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are a helpful assistant.", label="System message"),
        gr.Slider(minimum=1, maximum=4096, value=2048, step=1, label="Max tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p",
        ),
        gr.Slider(
            minimum=0,
            maximum=100,
            value=40,
            step=1,
            label="Top-k",
        ),
        gr.Slider(
            minimum=0.0,
            maximum=2.0,
            value=1.1,
            step=0.1,
            label="Repetition penalty",
        ),
        gr.Dropdown([
                'Meta-Llama-3-70B-Instruct-Q3_K_M.gguf',
                'Llama-3-8B-Synthia-v3.5-f16.gguf',
                'Mistral-7B-Instruct-v0.3-f32.gguf'
            ],
            value="Meta-Llama-3-70B-Instruct-Q3_K_M.gguf",
            label="Model"
        ),
    ],
    theme=gr.themes.Soft(primary_hue="violet", secondary_hue="violet", neutral_hue="gray",font=[gr.themes.GoogleFont("Exo"), "ui-sans-serif", "system-ui", "sans-serif"]).set(
        body_background_fill_dark="#111111",
        block_background_fill_dark="#111111",
        block_border_width="1px",
        block_title_background_fill_dark="#1e1c26",
        input_background_fill_dark="#292733",
        button_secondary_background_fill_dark="#24212b",
        border_color_primary_dark="#343140",
        background_fill_secondary_dark="#111111",
        color_accent_soft_dark="transparent"
    ),
    css=css,
    retry_btn="Retry",
    undo_btn="Undo",
    clear_btn="Clear",
    submit_btn="Send",
    description="Llama-cpp-agent: Chat multi llm selection"
)

if __name__ == "__main__":
    demo.launch()