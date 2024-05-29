css = """
.message-row {
    justify-content: space-evenly !important;
}
.message-bubble-border {
    border-radius: 6px !important;
}
.message-buttons-bot, .message-buttons-user {
    right: 10px !important;
    left: auto !important;
    bottom: 2px !important;
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
<div class="message-bubble-border" style="display:flex; max-width: 600px; border-width: 1px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); backdrop-filter: blur(10px);">
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
                <span style="display: inline-flex; align-items: center; border-radius: 0.375rem; background-color: rgba(229, 70, 77, 0.1); padding: 0.1rem 0.75rem; font-size: 0.75rem; font-weight: 500; color: #f88181; margin-bottom: 2.5px;">
                    Mixtral 8x7B Instruct v0.1
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