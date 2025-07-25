import random
import json
import asyncio
import os
import io
import tempfile
import time
import zipfile                               # ---------- Batch-TTS additions ----------
import numpy as np
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
import gradio as gr

load_dotenv()

# Global state
is_playing = False
LANGUAGES = [
    "English",
    "Arabic",
    "Bulgarian",
    "Chinese, Traditional",
    "Czech",
    "Danish",
    "Dutch",
    "Finnish",
    "French",
    "German",
    "Greek",
    "Hebrew",
    "Hungarian",
    "Italian",
    "Japanese",
    "Korean",
    "Malay",
    "Norwegian Bokmål",
    "Polish",
    "Portuguese",
    "Portuguese (Brazil)",
    "Romanian",
    "Russian",
    "Slovak",
    "Spanish",
    "Spanish (Latin America)",
    "Swedish",
    "Thai",
    "Turkish",
    "Ukrainian",
    "Vietnamese",
    "Indonesian",
    "Latvijas",
    "Kazakh",
]
BATCH_PARALLELISM = 8  # 控制批量生成的并发数

# Create a temporary directory to store audio files
temp_dir = tempfile.mkdtemp()

# openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
azure = AsyncAzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2025-03-01-preview",
)


async def translate_text(text, language):
    """
    Translate `text` into `language` asynchronously with AsyncAzureOpenAI Chat-Completions.
    Only returns the translated text.
    """
    completion = await azure.chat.completions.create(
        model="gpt-4o-001",
        messages=[
            {
                "role": "system",
                "content": f"Translate the user's input into {language}. "
                           f"Only respond with the translated text, nothing else.",
            },
            {"role": "user", "content": text},
        ],
    )
    return completion.choices[0].message.content.strip()


# -------------------- Voice related helpers -------------------- #
VOICE_OPTIONS = ["Alloy", "Ash", "Ballad", "Coral", "Echo", "Sage", "Shimmer", "Verse"]  # Available voices


def update_voice(selected_voice):
    return selected_voice, gr.Button(variant="primary")


def reset_buttons():
    """Return a list of secondary-variant buttons equal to VOICE_OPTIONS length"""
    return [gr.Button(variant="secondary") for _ in range(len(VOICE_OPTIONS))]


def update_random_button():
    """Highlight a random voice button"""
    buttons = reset_buttons()
    random_index = random.randint(0, len(VOICE_OPTIONS) - 1)
    buttons[random_index] = gr.Button(variant="primary")
    selected_voice = VOICE_OPTIONS[random_index]
    return f"# Voice: {selected_voice}", *buttons


# -------------------- Vibe helpers -------------------- #
def load_vibes():
    with open("vibe.json", "r", encoding="utf-8") as file:
        vibes = json.load(file)
    return [vibe["Vibe"] for vibe in vibes]


def get_vibe_description(vibe_name):
    with open("vibe.json", "r", encoding="utf-8") as file:
        vibes = json.load(file)
        for vibe in vibes:
            if vibe["Vibe"] == vibe_name:
                # Replace escaped newlines with actual newlines while preserving other characters
                description = vibe["Description"].replace('\\n', '\n')
                return description
    return ""


def get_vibe_info(vibe_name):
    with open("vibe.json", "r", encoding="utf-8") as file:
        vibes = json.load(file)
        for vibe in vibes:
            if vibe["Vibe"] == vibe_name:
                # Replace escaped newlines with actual newlines while preserving other characters
                description = vibe["Description"].replace('\\n', '\n')
                script = vibe["Script"].replace('\\n', '\n')
                return description, script
    return "", ""


# -------------------- Audio helpers -------------------- #
def check_api_key():
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Azure OpenAI API key not found. Please set the AZURE_OPENAI_API_KEY environment variable.")
    return api_key


async def generate_streaming_audio(voice_name, text, instructions):
    """Generate audio chunks from OpenAI TTS API"""
    async with azure.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice=voice_name,
        input=text,
        instructions=instructions,
        response_format="mp3"  # MP3 format works better for streaming
    ) as response:
        async for chunk in response.iter_bytes():
            yield chunk


async def generate_audio_file(input, output_path, voice_name="coral", instructions=None):
    """Generate audio file from OpenAI TTS and save to the given path"""
    async with azure.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice=voice_name,
        input=input,
        instructions=instructions,
        response_format="mp3",
    ) as response:
        await response.stream_to_file(output_path)


def stream_audio(voice_name, text, instructions):
    """Stream audio chunks from OpenAI TTS to the Gradio Audio component"""

    def generate():
        """Generator function that yields audio chunks as they're received"""
        # We need to accumulate bytes for proper MP3 playback
        audio_bytes = b""

        # Run the async function and collect audio chunks
        for chunk in asyncio.run(generate_streaming_audio(voice_name, text, instructions)):
            # Add the new chunk to our accumulated audio
            audio_bytes += chunk
            # Yield the accumulated audio so far
            yield audio_bytes

    # Return the generator function itself
    return generate


def stop_audio():
    global is_playing
    if is_playing:
        is_playing = False
        yield "Audio stopped."
    yield "No audio is playing."


# -------------------- Custom CSS -------------------- #
css = """
/* -------------------- voice buttons grid -------------------- */
.voice-buttons {
    display: grid;
    grid-template-columns: repeat(9, 1fr);
    gap: 0.75rem;
    width: 100%;
    padding: 0.5rem;
}

.voice-button {
    aspect-ratio: 1/1;
    min-height: 60px;
    max-height: 100px;
    display: flex !important;
    flex-direction: column;
    position: relative;
    border-radius: 0.5rem;
    transition: all 0.2s ease-in-out;
    padding: 0.93rem;
}

.voice-button:hover {
    transform: scale(1.02);
}

/* Target both direct span and span within button content div */
.voice-button span,
.voice-button > div > span {
    font-size: 1rem;
    font-weight: 500;
    margin: 0;
    padding: 0;
    position: relative;
    left: 0;
    top: 0;
}

/* Container for button content */
.voice-button > div {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    justify-content: flex-start;
    width: 100%;
    margin: 0;
    padding: 0;
}

/* Fix icon alignment */
.voice-button svg {
    position: absolute;
    right: 0.93rem;
    top: 0.93rem;
    width: 1.25rem;
    height: 1.25rem;
}

.voice-button[data-variant="primary"]::after {
    content: "";
    position: absolute;
    left: 0.93rem;
    bottom: 0.93rem;
    width: 6px;
    height: 6px;
    background-color: #10B981;
    border-radius: 50%;
    box-shadow: 0 0 8px #10B981;
}

/* -------------------- vibe buttons -------------------- */
.vibe-buttons {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.75rem;
    width: 100%;
    padding: 0.5rem;
}

.vibe-buttons button {
    min-height: 60px;
    max-height: 100px;
}

/* Responsive vibes grid */
@media (max-width: 1024px) {
    .vibe-buttons {
        grid-template-columns: repeat(2, 1fr);
    }
}
@media (max-width: 768px) {
    .vibe-buttons {
        grid-template-columns: repeat(3, 1fr);
    }
}
@media (max-width: 640px) {
    .vibe-buttons {
        grid-template-columns: repeat(2, 1fr);
    }
}

/* Responsive voice grid */
@media (max-width: 1280px) {
    .voice-buttons {
        grid-template-columns: repeat(6, 1fr);
    }
    .voice-button {
        aspect-ratio: 2.5/1;
    }
}
@media (max-width: 1024px) {
    .voice-buttons {
        grid-template-columns: repeat(4, 1fr);
    }
    .voice-button  {
        aspect-ratio: 2/1;
    }
}
@media (max-width: 768px) {
    .voice-buttons {
        grid-template-columns: repeat(3, 1fr);
    }
    .voice-button  {
        aspect-ratio: 2/1;
    }
}
@media (max-width: 640px) {
    .voice-buttons {
        grid-template-columns: repeat(3, 1fr);
    }
    .voice-button {
        aspect-ratio: 4/3;
    }
}

/* -------------------- misc -------------------- */
footer {
    visibility: hidden;
}
"""

brand_theme = gr.themes.Default(
    primary_hue="blue",
    secondary_hue="blue",
    neutral_hue="gray",
    font=["Segoe UI", "Arial", "sans-serif"],
    font_mono=["Courier New", "monospace"]).set(
    button_primary_background_fill="#0f6cbd",
    button_primary_background_fill_hover="#115ea3",
    button_primary_background_fill_hover_dark="#4f52b2",
    button_primary_background_fill_dark="#5b5fc7",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#e0e0e0",
    button_secondary_background_fill_hover="#c0c0c0",
    button_secondary_background_fill_hover_dark="#a0a0a0",
    button_secondary_background_fill_dark="#808080",
    button_secondary_text_color="#000000", body_background_fill="#f5f5f5",
    block_background_fill="#ffffff",
    body_text_color="#242424",
    body_text_color_subdued="#616161",
    block_border_color="#d1d1d1",
    block_border_color_dark="#333333",
    input_background_fill="#ffffff",
    input_border_color="#d1d1d1",
    input_border_color_focus="#0f6cbd"
)

with gr.Blocks(
    css=css,
    theme=brand_theme,
    title="文本转语音Demo"
) as demo:
    with gr.Row():
        voice_label = gr.Markdown("# Voice: ")

    # -------------------- voice selection buttons -------------------- #
    with gr.Row(elem_classes="voice-buttons"):
        alloy = gr.Button("Alloy", variant="secondary",
                          icon="assets/ic_fluent_sound_wave_circle_sparkle_24_regular.svg", elem_classes="voice-button")
        ash = gr.Button("Ash", variant="secondary",
                        icon="assets/ic_fluent_sound_wave_circle_sparkle_24_regular.svg", elem_classes="voice-button")
        ballad = gr.Button("Ballad", variant="secondary",
                           icon="assets/ic_fluent_sound_wave_circle_sparkle_24_regular.svg", elem_classes="voice-button")
        coral = gr.Button("Coral", variant="secondary",
                          icon="assets/ic_fluent_sound_wave_circle_sparkle_24_regular.svg", elem_classes="voice-button")
        echo = gr.Button("Echo", variant="secondary",
                         icon="assets/ic_fluent_sound_wave_circle_sparkle_24_regular.svg", elem_classes="voice-button")
        sage = gr.Button("Sage", variant="secondary",
                         icon="assets/ic_fluent_sound_wave_circle_sparkle_24_regular.svg", elem_classes="voice-button")
        shimmer = gr.Button("Shimmer", variant="secondary",
                            icon="assets/ic_fluent_sound_wave_circle_sparkle_24_regular.svg", elem_classes="voice-button")
        verse = gr.Button("Verse", variant="secondary",
                          icon="assets/ic_fluent_sound_wave_circle_sparkle_24_regular.svg", elem_classes="voice-button")
        random_btn = gr.Button("Random", variant="huggingface", elem_classes="voice-button")

    # helper to reset + highlight selected voice
    def update_button_and_reset(selected_voice):
        buttons = reset_buttons()
        buttons[VOICE_OPTIONS.index(selected_voice)] = gr.Button(variant="primary")
        return f"# Voice: {selected_voice}", *buttons

    alloy.click(lambda: update_button_and_reset("Alloy"),
                outputs=[voice_label, alloy, ash, ballad, coral, echo, sage, shimmer, verse])
    ash.click(lambda: update_button_and_reset("Ash"),
              outputs=[voice_label, alloy, ash, ballad, coral, echo, sage, shimmer, verse])
    ballad.click(lambda: update_button_and_reset("Ballad"),
                 outputs=[voice_label, alloy, ash, ballad, coral, echo, sage, shimmer, verse])
    coral.click(lambda: update_button_and_reset("Coral"),
                outputs=[voice_label, alloy, ash, ballad, coral, echo, sage, shimmer, verse])
    echo.click(lambda: update_button_and_reset("Echo"),
               outputs=[voice_label, alloy, ash, ballad, coral, echo, sage, shimmer, verse])
    sage.click(lambda: update_button_and_reset("Sage"),
               outputs=[voice_label, alloy, ash, ballad, coral, echo, sage, shimmer, verse])
    shimmer.click(lambda: update_button_and_reset("Shimmer"),
                  outputs=[voice_label, alloy, ash, ballad, coral, echo, sage, shimmer, verse])
    verse.click(lambda: update_button_and_reset("Verse"),
                outputs=[voice_label, alloy, ash, ballad, coral, echo, sage, shimmer, verse])

    random_btn.click(update_random_button,
                     outputs=[voice_label, alloy, ash, ballad, coral, echo, sage, shimmer, verse])

    # -------------------- vibe area -------------------- #
    with gr.Row():
        with gr.Column():
            gr.Markdown("# Vibe")
            all_vibes = load_vibes()
            vibe_dropdown = gr.Dropdown(choices=all_vibes, value=all_vibes[0], label="Select a Vibe")
            shuffle_btn = gr.Button("Shuffle", variant="huggingface", visible=True)
            vibe_desc = gr.Textbox(show_label=False, container=False, lines=15, max_lines=15)

        with gr.Column():
            gr.Markdown("# Script")

            translation_toggle = gr.Radio(
                choices=["Original", "Translation"],
                value="Original",
                show_label=False
            )
            vibe_script = gr.Textbox(show_label=False, container=False, lines=9, max_lines=9)

            with gr.Row():
                translate_btn = gr.Button("Translate to", variant="secondary")
                language_dd = gr.Dropdown(
                    show_label=False,
                    choices=LANGUAGES,
                    value=LANGUAGES[0]
                )
            original_state = gr.State("")
            translated_state = gr.State("")

            # -------------------- 修复：仅在“Original”模式下同步 -------------------- #
            def sync_original(text, mode):
                if mode == "Original":
                    return text
                return gr.update()

            vibe_script.change(
                sync_original,
                inputs=[vibe_script, translation_toggle],
                outputs=[original_state]
            )

            audio_output = gr.Audio(autoplay=True, streaming=True)
            play_btn = gr.Button(value="Play", variant="primary", icon=os.path.join("assets", "ic_fluent_play_24_filled.svg"), visible=True)
            stop_btn = gr.Button(value="Stop", variant="stop", icon=os.path.join("assets", "ic_fluent_stop_24_filled.svg"), visible=False)

            # ---------- Batch-TTS additions (UI) ----------
            gr.Markdown("---")
            with gr.Accordion("Batch TTS", open=False):  # 折叠以简化布局
                multi_lang_btn = gr.Button("Generate All Languages Audio (Original)", variant="primary")
                multi_lang_progress = gr.HTML()
                multi_lang_zip = gr.File(label="All Languages ZIP", visible=False)
                with gr.Row():
                    txt_file = gr.File(label="Upload .txt", file_types=["text"])
                    batch_btn = gr.Button("Batch Generate TTS from txt", variant="primary")
                progress_bar = gr.HTML()     # 新增 progress 组件
                download_zip = gr.File(label="Download ZIP", visible=False)
            # ---------- End Batch-TTS additions (UI) ----------

    # 新增：dropdown 选择 vibe 时，更新描述、脚本、状态
    def on_vibe_change(vibe):
        desc, script = get_vibe_info(vibe)
        return desc, script, vibe, script, ""  # desc, script, original_state, script(原文), 清空翻译

    vibe_dropdown.change(
        on_vibe_change,
        inputs=[vibe_dropdown],
        outputs=[vibe_desc, vibe_script, original_state, vibe_script, translated_state]
    )

    # Shuffle 逻辑：随机选一个vibe
    def shuffle_vibe_dropdown(vibe_list):
        vibe = random.choice(vibe_list)
        desc, script = get_vibe_info(vibe)
        return vibe, desc, script, vibe, script, ""  # dropdown值, desc, script, original_state, script, 清空翻译

    shuffle_btn.click(
        shuffle_vibe_dropdown,
        inputs=[gr.State(all_vibes)],
        outputs=[vibe_dropdown, vibe_desc, vibe_script, original_state, vibe_script, translated_state]
    )

    # ---------- Translation button逻辑 ---------- #
    def run_translation(orig_text, tgt_lang):
        """
        点击“Translate to”按钮后执行翻译。
        1. 保留已翻译内容（再次点击会覆盖）。
        2. 不自动在开关时翻译或清理内容。
        """
        translated = asyncio.run(translate_text(orig_text, tgt_lang))
        # 默认切换到“Translation”视图
        return translated, translated, "Translation", orig_text

    translate_btn.click(
        run_translation,
        inputs=[vibe_script, language_dd],
        outputs=[vibe_script, translated_state, translation_toggle, original_state]
    )

    def switch_text(orig, trans, mode):
        """
        Switch textbox content based on toggle.
        """
        if mode == "Translation" and trans:
            return trans
        return orig

    translation_toggle.change(
        switch_text,
        inputs=[original_state, translated_state, translation_toggle],
        outputs=[vibe_script]
    )

    # -------------------- play / stop handlers -------------------- #
    def toggle_play_stop(voice_name, vibe_desc, script_text):
        """Handle the play button click and toggle button visibility"""
        global is_playing
        is_playing = True
        try:
            api_key = check_api_key()
            # Make sure we have a voice name and properly format it
            if not voice_name or not isinstance(voice_name, str):
                raise ValueError("No voice selected. Please select a voice first.")
            voice_name = voice_name.replace("# Voice: ", "").strip().lower()
            if not voice_name:
                raise ValueError("Invalid voice name. Please select a valid voice.")

            # Use whatever text is in the shared textbox
            text_for_tts = script_text

            # Create a temporary file path
            temp_file = os.path.join(temp_dir, f"{voice_name}_{int(time.time())}.mp3")

            # Generate and save audio to temp file
            asyncio.run(generate_audio_file(text_for_tts, temp_file, voice_name, vibe_desc))

            gr.Info("Audio playing...")
            play_btn = gr.Button(value="Play", variant="primary", icon=os.path.join("assets", "ic_fluent_play_24_filled.svg"), visible=False)
            stop_btn = gr.Button(value="Stop", variant="stop", icon=os.path.join("assets", "ic_fluent_stop_24_filled.svg"), visible=True)
            return play_btn, stop_btn, temp_file  # Return the path to the temp file

        except Exception as e:
            is_playing = False
            play_btn = gr.Button(value="Play", variant="primary", icon=os.path.join("assets", "ic_fluent_play_24_filled.svg"), visible=True)
            stop_btn = gr.Button(value="Stop", variant="stop", icon=os.path.join("assets", "ic_fluent_stop_24_filled.svg"), visible=False)
            raise gr.Error(f"Error playing audio: {str(e)}")

    def handle_stop():
        """Handle the stop button click"""
        global is_playing
        is_playing = False
        gr.Info("Audio stopped")
        play_btn = gr.Button(value="Play", variant="primary", icon=os.path.join("assets", "ic_fluent_play_24_filled.svg"), visible=True)
        stop_btn = gr.Button(value="Stop", variant="stop", icon=os.path.join("assets", "ic_fluent_stop_24_filled.svg"), visible=False)
        return play_btn, stop_btn, None

    play_btn.click(
        toggle_play_stop,
        inputs=[voice_label, vibe_desc, vibe_script],
        outputs=[play_btn, stop_btn, audio_output]
    )

    stop_btn.click(
        handle_stop,
        outputs=[play_btn, stop_btn, audio_output]
    )

    # ---------- Batch-TTS additions (logic) ----------
    async def batch_generate_tts(txt_file_obj, voice_name_md, vibe_instruction, progress=gr.Progress()):
        """
        1. 读取上传的 txt，按行（非空行）拆分。
        2. 使用当前 voice 与 vibe 指令批量异步合成 MP3。
        3. 实时更新进度条，最后打包 ZIP 返回。
        """
        if txt_file_obj is None:
            raise gr.Error("Please upload a .txt file first.")
        voice_name = voice_name_md.replace("# Voice: ", "").strip().lower()
        if not voice_name:
            raise gr.Error("Please select a voice before generating.")

        with open(txt_file_obj.name, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        if not lines:
            raise gr.Error("TXT file is empty.")

        total = len(lines)
        zip_path = os.path.join(temp_dir, f"{voice_name}_{int(time.time())}.zip")
        yield gr.update(visible=False), gr.update(value="Processing...", interactive=False), gr.update(value="")

        semaphore = asyncio.Semaphore(BATCH_PARALLELISM)  # 控制最大并发量，比如 5

        # 存每条任务的mp3路径
        mp3_paths = [os.path.join(temp_dir, f"{voice_name}_{idx+1}.mp3") for idx in range(total)]

        async def synth_one(idx, line):
            # 用信号量保证并发控制
            async with semaphore:
                await generate_audio_file(line, mp3_paths[idx], voice_name, vibe_instruction)
            return idx

        # 启动所有语音合成任务
        tasks = [asyncio.create_task(synth_one(idx, line)) for idx, line in enumerate(lines)]

        finished = 0
        for coro in asyncio.as_completed(tasks):
            try:
                idx = await coro  # 获取已完成的 idx
                finished += 1
                progress(finished / total, desc=f"Generating {finished}/{total}")
                progress_html = f"<progress value='{finished}' max='{total}' style='width:100%'></progress>"
                yield gr.update(visible=False), gr.update(), gr.update(value=progress_html)
            except Exception as e:
                # 这里你可以加异常处理逻辑，或者把失败任务补录下来
                pass

        # 打包 ZIP
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for mp3_path in mp3_paths:
                zipf.write(mp3_path, arcname=os.path.basename(mp3_path))
        gr.Info(f"Generated {total} audio files.")
        final_html = f"<progress value='{total}' max='{total}' style='width:100%'></progress>"
        yield gr.update(value=zip_path, visible=True), gr.update(value="Batch Generate TTS", interactive=True), gr.update(value=final_html)

    async def generate_all_languages_audio(original_text, voice_name_md, vibe_instruction, progress=gr.Progress()):
        """
        1. 把 original_text 依次翻译成所有 LANGUAGES。
        2. 每个翻译用同一 voice/vibe 合成 mp3。
        3. 全部 mp3 打包 zip 返回。
        """
        if not original_text or not original_text.strip():
            raise gr.Error("No text to synthesize.")
        voice_name = voice_name_md.replace("# Voice: ", "").strip().lower()
        if not voice_name:
            raise gr.Error("Please select a voice.")

        total = len(LANGUAGES)
        zip_path = os.path.join(temp_dir, f"{voice_name}_alllangs_{int(time.time())}.zip")
        yield gr.update(visible=False), gr.update(value="Processing...", interactive=False), gr.update(value="")

        semaphore = asyncio.Semaphore(BATCH_PARALLELISM)  # 并发数自己调，通常 5-10 足够
        mp3_paths = [os.path.join(temp_dir, f"{voice_name}_{lang.replace(' ', '_')}.mp3") for lang in LANGUAGES]

        async def translate_and_synth(idx, lang):
            async with semaphore:
                # 翻译
                try:
                    if lang == "English":
                        translated = original_text
                    else:
                        translated = await translate_text(original_text, lang)
                except Exception as e:
                    translated = f"[Translation Error: {str(e)}]"
                # 合成
                try:
                    await generate_audio_file(translated, mp3_paths[idx], voice_name, vibe_instruction)
                except Exception as e:
                    # 你可以写个特殊音频，或者啥都不做
                    pass
                return idx, lang

        # 创建全部并发任务
        tasks = [asyncio.create_task(translate_and_synth(idx, lang)) for idx, lang in enumerate(LANGUAGES)]

        finished = 0
        for coro in asyncio.as_completed(tasks):
            idx, lang = await coro
            finished += 1
            progress(finished / total, desc=f"{lang}: finished")
            progress_html = (
                f"<progress value='{finished}' max='{total}' style='width:100%'></progress> "
                f"<div>正在处理: {lang}</div>"
            )
            yield gr.update(visible=False), gr.update(), gr.update(value=progress_html)

        # 打包 ZIP
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for mp3_path in mp3_paths:
                zipf.write(mp3_path, arcname=os.path.basename(mp3_path))
        gr.Info(f"已生成 {total} 个音频，打包完成。")
        final_html = (
            f"<progress value='{total}' max='{total}' style='width:100%'></progress> "
            f"<div>所有语言批量生成完成！</div>"
        )
        yield gr.update(value=zip_path, visible=True), gr.update(value="Generate All Languages Audio (Original)", interactive=True), gr.update(value=final_html)

    # 三步输出： [download_zip, batch_btn, progress_bar]
    batch_btn.click(
        batch_generate_tts,
        inputs=[txt_file, voice_label, vibe_desc],
        outputs=[download_zip, batch_btn, progress_bar],
        show_progress="none"   # 使用自定义 progress 对象，禁用默认 spinner
    )
    multi_lang_btn.click(
        generate_all_languages_audio,
        inputs=[original_state, voice_label, vibe_desc],
        outputs=[multi_lang_zip, multi_lang_btn, multi_lang_progress],
        show_progress="none"
    )
    # ---------- End Batch-TTS additions (logic) ----------

if __name__ == "__main__":
    asyncio.run(
        demo.launch(
            favicon_path="assets/system_head.png",
            auth=("admin", "tplinkcnai"),
            server_name="0.0.0.0",
            server_port=7861,
            root_path="/tts-web"
        )
    )
