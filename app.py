import spaces
import json
import subprocess
import gradio as gr
from huggingface_hub import hf_hub_download

from duckduckgo_search import DDGS

from trafilatura import fetch_url, extract

subprocess.run(
    'pip install llama-cpp-python==0.2.75 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124',
    shell=True)
subprocess.run('pip install llama-cpp-agent==0.2.10', shell=True)

hf_hub_download(
    repo_id="bartowski/Mistral-7B-Instruct-v0.3-GGUF",
    filename="Mistral-7B-Instruct-v0.3-f32.gguf",
    local_dir="./models"
)
hf_hub_download(
    repo_id="bartowski/Llama-3-8B-Instruct-262k-GGUF",
    filename="Llama-3-8B-Instruct-262k-Q6_K.gguf",
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
    border-color: #343140 !important;
}
.dark.user {
    background: #1e1c26 !important;
}
.dark.assistant.dark, .dark.pending.dark {
    background: #111111 !important;
}
"""


def get_website_content_from_url(url: str) -> str:
    """
    Get website content from a URL using Selenium and BeautifulSoup for improved content extraction and filtering.

    Args:
        url (str): URL to get website content from.

    Returns:
        str: Extracted content including title, main text, and tables.
    """

    try:
        downloaded = fetch_url(url)

        result = extract(downloaded, include_formatting=True, include_links=True, output_format='json', url=url)

        if result:
            result = json.loads(result)
            return f'=========== Website Title: {result["title"]} ===========\n\n=========== Website URL: {url} ===========\n\n=========== Website Content ===========\n\n{result["raw_text"]}\n\n=========== Website Content End ===========\n\n'
        else:
            return ""
    except Exception as e:
        return f"An error occurred: {str(e)}"


def search_web(search_query: str):
    """
    Search the web for information.
    Args:
        search_query (str): Search query to search for.
    """
    results = DDGS().text(search_query, region='wt-wt', safesearch='off', timelimit='y', max_results=3)
    result_string = ''
    for res in results:
        web_info = get_website_content_from_url(res['href'])
        if web_info != "":
            result_string += web_info

    res = result_string.strip()
    return "Based on the following results, answer the previous user query:\nResults:\n\n" + res


def get_messages_formatter_type(model_name):
    from llama_cpp_agent import MessagesFormatterType
    if "Llama" in model_name:
        return MessagesFormatterType.LLAMA_3
    elif "Mistral" in model_name:
        return MessagesFormatterType.MISTRAL
    elif "mixtral" in model_name:
        return MessagesFormatterType.MISTRAL
    elif "Phi" in model_name:
        return MessagesFormatterType.PHI_3
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def write_message_to_user():
    """
    Let you write a message to the user.
    """
    return "Please write the message to the user."


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
    from llama_cpp_agent.llm_output_settings import LlmStructuredOutputSettings
    chat_template = get_messages_formatter_type(model)

    llm = Llama(
        model_path=f"models/{model}",
        flash_attn=True,
        n_threads=40,
        n_gpu_layers=81,
        n_batch=1024,
        n_ctx=32768,
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
    output_settings = LlmStructuredOutputSettings.from_functions(
        [search_web, write_message_to_user])
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
            'Mistral-7B-Instruct-v0.3-f32.gguf',
            'Llama-3-8B-Instruct-262k-Q6_K.gguf'
        ],
            value="Mistral-7B-Instruct-v0.3-f32.gguf",
            label="Model"
        ),
    ],
    theme=gr.themes.Soft(
        primary_hue="violet",
        secondary_hue="violet",
        neutral_hue="gray",
        font=[gr.themes.GoogleFont("Exo"), "ui-sans-serif", "system-ui", "sans-serif"]).set(
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
        description="Llama-cpp-agent: Chat Web Search DDG Agent"
    )

if __name__ == "__main__":
    demo.launch()
