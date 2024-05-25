import spaces
import json
import subprocess
import time
import gradio as gr
from llama_cpp import Llama
from llama_cpp_agent import LlamaCppAgent
from llama_cpp_agent.providers import LlamaCppPythonProvider
from llama_cpp_agent.chat_history import BasicChatHistory
from llama_cpp_agent.chat_history.messages import Roles
from llama_cpp_agent.llm_output_settings import LlmStructuredOutputSettings
from huggingface_hub import hf_hub_download
from web_search import WebSearchTool

model_selected = "Mistral-7B-Instruct-v0.3-Q6_K.gguf"
examples = [
    ["latest news about Yann LeCun"],
    ["Latest news site:github.blog"],
    ["Where I can find best hotel in Galapagos, Ecuador intitle:hotel"],
    ["filetype:pdf intitle:python"]
]

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

css = """
.message-row {
    justify-content: space-evenly !important;
}
.message-bubble-border {
    border-radius: 6px !important;
}
.dark.message-bubble-border {
    border-color: #1b0f0f !important;
}
.dark.user {
    background: #140b0b !important;
}
.dark.assistant.dark, .dark.pending.dark {
    background: #0c0505 !important;
}
"""

PLACEHOLDER = """
<div class="message-bubble-border" style="display:flex; max-width: 600px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); backdrop-filter: blur(10px);">
    <figure style="margin: 0;">
        <img src="https://huggingface.co/spaces/poscye/ddg-web-search-chat/resolve/main/logo.jpg" alt="Logo" style="width: 100%; height: 100%; border-radius: 8px;">
    </figure>
    <div style="padding: .5rem 1.5rem;">
        <h2 style="text-align: left; font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem;">llama-cpp-agent</h2>
        <p style="text-align: left; font-size: 16px; line-height: 1.5; margin-bottom: 15px;">DDG Agent allows users to interact with it using natural language, making it easier for them to find the information they need. Offers a convenient and secure way for users to access web-based information.</p>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="display: flex; flex-flow: column; justify-content: space-between;">
                <span style="display: inline-flex; align-items: center; border-radius: 0.375rem; background-color: rgba(229, 70, 77, 0.1); padding: 0.1rem 0.75rem; font-size: 0.75rem; font-weight: 500; color: #f88181; margin-bottom: 2.5px;">
                    Mistral 7B Instruct v0.3
                </span>
                <span style="display: inline-flex; align-items: center; border-radius: 0.375rem; background-color: rgba(79, 70, 229, 0.1); padding: 0.1rem 0.75rem; font-size: 0.75rem; font-weight: 500; color: #60a5fa; margin-top: 2.5px;">
                    Meta Llama 3 8B Instruct
                </span>
            </div>
            <div style="display: flex; justify-content: flex-end; align-items: center;">
                <a href="https://discord.gg/sRMvWKrh" target="_blank" rel="noreferrer" style="padding: .5rem;">
                    <svg width="24" height="24" fill="currentColor" xmlns="http://www.w3.org/2000/svg" viewBox="0 5 30.67 23.25">
                        <title>Discord</title>
                        <path d="M26.0015 6.9529C24.0021 6.03845 21.8787 5.37198 19.6623 5C19.3833 5.48048 19.0733 6.13144 18.8563 6.64292C16.4989 6.30193 14.1585 6.30193 11.8336 6.64292C11.6166 6.13144 11.2911 5.48048 11.0276 5C8.79575 5.37198 6.67235 6.03845 4.6869 6.9529C0.672601 12.8736 -0.41235 18.6548 0.130124 24.3585C2.79599 26.2959 5.36889 27.4739 7.89682 28.2489C8.51679 27.4119 9.07477 26.5129 9.55525 25.5675C8.64079 25.2265 7.77283 24.808 6.93587 24.312C7.15286 24.1571 7.36986 23.9866 7.57135 23.8161C12.6241 26.1255 18.0969 26.1255 23.0876 23.8161C23.3046 23.9866 23.5061 24.1571 23.7231 24.312C22.8861 24.808 22.0182 25.2265 21.1037 25.5675C21.5842 26.5129 22.1422 27.4119 22.7621 28.2489C25.2885 27.4739 27.8769 26.2959 30.5288 24.3585C31.1952 17.7559 29.4733 12.0212 26.0015 6.9529ZM10.2527 20.8402C8.73376 20.8402 7.49382 19.4608 7.49382 17.7714C7.49382 16.082 8.70276 14.7025 10.2527 14.7025C11.7871 14.7025 13.0425 16.082 13.0115 17.7714C13.0115 19.4608 11.7871 20.8402 10.2527 20.8402ZM20.4373 20.8402C18.9183 20.8402 17.6768 19.4608 17.6768 17.7714C17.6768 16.082 18.8873 14.7025 20.4373 14.7025C21.9717 14.7025 23.2271 16.082 23.1961 17.7714C23.1961 19.4608 21.9872 20.8402 20.4373 20.8402Z"></path>
                    </svg>
                </a>
                <a href="https://github.com/Maximilian-Winter/llama-cpp-agent" target="_blank" rel="noreferrer" style="padding: .5rem;">
                    <svg width="24" height="24" fill="currentColor" viewBox="3 3 18 18">
                        <title>GitHub</title>
                        <path d="M12 3C7.0275 3 3 7.12937 3 12.2276C3 16.3109 5.57625 19.7597 9.15374 20.9824C9.60374 21.0631 9.77249 20.7863 9.77249 20.5441C9.77249 20.3249 9.76125 19.5982 9.76125 18.8254C7.5 19.2522 6.915 18.2602 6.735 17.7412C6.63375 17.4759 6.19499 16.6569 5.8125 16.4378C5.4975 16.2647 5.0475 15.838 5.80124 15.8264C6.51 15.8149 7.01625 16.4954 7.18499 16.7723C7.99499 18.1679 9.28875 17.7758 9.80625 17.5335C9.885 16.9337 10.1212 16.53 10.38 16.2993C8.3775 16.0687 6.285 15.2728 6.285 11.7432C6.285 10.7397 6.63375 9.9092 7.20749 9.26326C7.1175 9.03257 6.8025 8.08674 7.2975 6.81794C7.2975 6.81794 8.05125 6.57571 9.77249 7.76377C10.4925 7.55615 11.2575 7.45234 12.0225 7.45234C12.7875 7.45234 13.5525 7.55615 14.2725 7.76377C15.9937 6.56418 16.7475 6.81794 16.7475 6.81794C17.2424 8.08674 16.9275 9.03257 16.8375 9.26326C17.4113 9.9092 17.76 10.7281 17.76 11.7432C17.76 15.2843 15.6563 16.0687 13.6537 16.2993C13.98 16.5877 14.2613 17.1414 14.2613 18.0065C14.2613 19.2407 14.25 20.2326 14.25 20.5441C14.25 20.7863 14.4188 21.0746 14.8688 20.9824C16.6554 20.364 18.2079 19.1866 19.3078 17.6162C20.4077 16.0457 20.9995 14.1611 21 12.2276C21 7.12937 16.9725 3 12 3Z"></path>
                    </svg>
                </a>
            </div>
        </div>
    </div>
</div>
"""

def get_context_by_model(model_name):
    model_context_limits = {
        "Mistral-7B-Instruct-v0.3-Q6_K.gguf": 32768,
        "Meta-Llama-3-8B-Instruct-Q6_K.gguf": 8192
    }
    return model_context_limits.get(model_name, None)

def get_messages_formatter_type(model_name):
    from llama_cpp_agent import MessagesFormatterType
    if "Meta" in model_name or "aya" in model_name:
        return MessagesFormatterType.LLAMA_3
    elif "Mistral" in model_name:
        return MessagesFormatterType.MISTRAL
    elif "Einstein-v6-7B" in model_name or "dolphin" in model_name:
        return MessagesFormatterType.CHATML
    elif "Phi" in model_name:
        return MessagesFormatterType.PHI_3
    else:
        return MessagesFormatterType.CHATML


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
    model_selected = model

    system_message += f" {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}"

    llm = Llama(
        model_path=f"models/{model}",
        flash_attn=True,
        n_threads=40,
        n_gpu_layers=81,
        n_batch=1024,
        n_ctx=get_context_by_model(model),
    )
    provider = LlamaCppPythonProvider(llm)

    agent = LlamaCppAgent(
        provider,
        system_prompt=f"{system_message}",
        predefined_messages_formatter_type=chat_template,
        debug_output=True
    )
    search_tool = WebSearchTool(provider, chat_template, get_context_by_model(model))
    settings = provider.get_provider_default_settings()
    settings.temperature = temperature
    settings.top_k = top_k
    settings.top_p = top_p
    settings.max_tokens = max_tokens
    settings.repeat_penalty = repeat_penalty
    settings.stream = True
    output_settings = LlmStructuredOutputSettings.from_functions(
        [search_tool.get_tool(), write_message_to_user])
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
    result = agent.get_chat_response(message, llm_sampling_settings=settings, structured_output_settings=output_settings,
                                     chat_history=messages,
                                     print_output=False)
    while True:
        if result[0]["function"] == "write_message_to_user":
            break
        else:
            result = agent.get_chat_response(result[0]["return_value"], role=Roles.tool, chat_history=messages,structured_output_settings=output_settings,
                                             print_output=False)

    stream = agent.get_chat_response(
        result[0]["return_value"], role=Roles.tool, llm_sampling_settings=settings, chat_history=messages, returns_streaming_generator=True,
        print_output=False
    )

    outputs = ""
    for output in stream:
        outputs += output
        yield outputs


demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Dropdown([
            'Mistral-7B-Instruct-v0.3-Q6_K.gguf',
            'Meta-Llama-3-8B-Instruct-Q6_K.gguf'
        ],
            value="Mistral-7B-Instruct-v0.3-Q6_K.gguf",
            label="Model"
        ),
        gr.Textbox(value="You are a helpful assistant. Use additional available information you have access to when giving a response. Always give detailed and long responses. Format your response, well structured in markdown format.", label="System message"),
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
            border_color_primary_dark="#1b0f0f",
            background_fill_secondary_dark="#0c0505",
            color_accent_soft_dark="transparent"
        ),
        css=css,
        retry_btn="Retry",
        undo_btn="Undo",
        clear_btn="Clear",
        submit_btn="Send",
        examples = (examples),
        description="Llama-cpp-agent: Chat Web Search DDG Agent",
        chatbot=gr.Chatbot(scale=1, placeholder=PLACEHOLDER)
    )

if __name__ == "__main__":
    demo.launch()
