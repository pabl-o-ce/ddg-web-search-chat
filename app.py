import spaces
import logging
import gradio as gr
from huggingface_hub import hf_hub_download

from llama_cpp import Llama
from llama_cpp_agent.providers import LlamaCppPythonProvider
from llama_cpp_agent import LlamaCppAgent, MessagesFormatterType
from llama_cpp_agent.chat_history import BasicChatHistory
from llama_cpp_agent.chat_history.messages import Roles
from llama_cpp_agent.llm_output_settings import (
    LlmStructuredOutputSettings,
    LlmStructuredOutputType,
)
from llama_cpp_agent.tools import WebSearchTool
from llama_cpp_agent.prompt_templates import web_search_system_prompt, research_system_prompt
from ui import css, PLACEHOLDER
from utils import CitingSources
from settings import get_context_by_model, get_messages_formatter_type

hf_hub_download(
    repo_id="bartowski/Mistral-7B-Instruct-v0.3-GGUF",
    filename="Mistral-7B-Instruct-v0.3-Q6_K.gguf",
    local_dir="./models"
)
hf_hub_download(
    repo_id="bartowski/Meta-Llama-3-8B-Instruct-GGUF",
    filename="Meta-Llama-3-8B-Instruct-Q6_K.gguf",
    local_dir="./models"
)
hf_hub_download(
    repo_id="TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF",
    filename="mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf",
    local_dir="./models"
)

examples = [
    ["latest news about Yann LeCun"],
    ["Latest news site:github.blog"],
    ["Where I can find best hotel in Galapagos, Ecuador intitle:hotel"],
    ["filetype:pdf intitle:python"]
]

def write_message_to_user():
    """
    Let you write a message to the user.
    """
    return "Please write the message to the user."


@spaces.GPU(duration=120)
def respond(
    message,
    history: list[tuple[str, str]],
    model,
    system_message,
    max_tokens,
    temperature,
    top_p,
    top_k,
    repeat_penalty,
):
    chat_template = get_messages_formatter_type(model)
    llm = Llama(
        model_path=f"models/{model}",
        flash_attn=True,
        n_gpu_layers=81,
        n_batch=1024,
        n_ctx=get_context_by_model(model),
    )
    provider = LlamaCppPythonProvider(llm)
    logging.info(f"Loaded chat examples: {chat_template}")
    search_tool = WebSearchTool(
        llm_provider=provider,
        message_formatter_type=chat_template,
        max_tokens_search_results=12000,
        max_tokens_per_summary=2048,
    )

    web_search_agent = LlamaCppAgent(
        provider,
        system_prompt=web_search_system_prompt,
        predefined_messages_formatter_type=chat_template,
        debug_output=True,
    )

    answer_agent = LlamaCppAgent(
        provider,
        system_prompt=research_system_prompt,
        predefined_messages_formatter_type=chat_template,
        debug_output=True,
    )

    settings = provider.get_provider_default_settings()
    settings.stream = False
    settings.temperature = temperature
    settings.top_k = top_k
    settings.top_p = top_p

    settings.max_tokens = max_tokens
    settings.repeat_penalty = repeat_penalty

    output_settings = LlmStructuredOutputSettings.from_functions(
        [search_tool.get_tool()]
    )

    messages = BasicChatHistory()

    for msn in history:
        user = {"role": Roles.user, "content": msn[0]}
        assistant = {"role": Roles.assistant, "content": msn[1]}
        messages.add_message(user)
        messages.add_message(assistant)

    result = web_search_agent.get_chat_response(
        message,
        llm_sampling_settings=settings,
        structured_output_settings=output_settings,
        add_message_to_chat_history=False,
        add_response_to_chat_history=False,
        print_output=False,
    )

    outputs = ""

    settings.stream = True
    response_text = answer_agent.get_chat_response(
        f"Write a detailed and complete research document that fulfills the following user request: '{message}', based on the information from the web below.\n\n" +
        result[0]["return_value"],
        role=Roles.tool,
        llm_sampling_settings=settings,
        chat_history=messages,
        returns_streaming_generator=True,
        print_output=False,
    )

    for text in response_text:
        outputs += text
        yield outputs

    output_settings = LlmStructuredOutputSettings.from_pydantic_models(
        [CitingSources], LlmStructuredOutputType.object_instance
    )

    citing_sources = answer_agent.get_chat_response(
        "Cite the sources you used in your response.",
        role=Roles.tool,
        llm_sampling_settings=settings,
        chat_history=messages,
        returns_streaming_generator=False,
        structured_output_settings=output_settings,
        print_output=False,
    )
    outputs += "\n\nSources:\n"
    outputs += "\n".join(citing_sources.sources)
    yield outputs


demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Dropdown([
            'Mistral-7B-Instruct-v0.3-Q6_K.gguf',
            'mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf',
            'Meta-Llama-3-8B-Instruct-Q6_K.gguf'
        ],
            value="Mistral-7B-Instruct-v0.3-Q6_K.gguf",
            label="Model"
        ),
        gr.Textbox(value=web_search_system_prompt, label="System message"),
        gr.Slider(minimum=1, maximum=4096, value=2048, step=1, label="Max tokens"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.45, step=0.1, label="Temperature"),
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
    ],
    theme=gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="amber",
        neutral_hue="gray",
        font=[gr.themes.GoogleFont("Exo"), "ui-sans-serif", "system-ui", "sans-serif"]).set(
            body_background_fill_dark="#0c0505",
            block_background_fill_dark="#0c0505",
            block_border_width="1px",
            block_title_background_fill_dark="#1b0f0f",
            input_background_fill_dark="#140b0b",
            button_secondary_background_fill_dark="#140b0b",
            border_color_accent_dark="#1b0f0f",
            border_color_primary_dark="#1b0f0f",
            background_fill_secondary_dark="#0c0505",
            color_accent_soft_dark="transparent",
            code_background_fill_dark="#140b0b"
        ),
        css=css,
        retry_btn="Retry",
        undo_btn="Undo",
        clear_btn="Clear",
        submit_btn="Send",
        cache_examples=False,
        examples = (examples),
        description="Llama-cpp-agent: Chat with DuckDuckGo Agent",
        analytics_enabled=False,
        chatbot=gr.Chatbot(
            scale=1,
            placeholder=PLACEHOLDER,
            show_copy_button=True
        )
    )

if __name__ == "__main__":
    demo.launch()
