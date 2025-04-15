import gradio as gr
import random
import json
import asyncio
import os
import io
import tempfile
import time
import numpy as np
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI

load_dotenv()

# Global state
is_playing = False

# Create a temporary directory to store audio files
temp_dir = tempfile.mkdtemp()

#openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
azure = AsyncAzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2025-03-01-preview",
)

def update_voice(selected_voice):
    return selected_voice, gr.Button(variant="primary")

def reset_buttons():
    return [gr.Button(variant="secondary") for _ in range(11)]

def update_random_button():
    buttons = reset_buttons()
    random_index = random.randint(0, 10)  # Randomly select one button index
    buttons[random_index] = gr.Button(variant="primary")
    selected_voice = "Alloy Ash Ballad Coral Echo Fable Nova Onyx Sage Shimmer Verse".split()[random_index]
    return f"Voice: {selected_voice}", *buttons

def load_vibes():
    with open("vibe.json", "r", encoding="utf-8") as file:
        vibes = json.load(file)
    return [vibe["Vibe"] for vibe in vibes]

def update_vibe_buttons(all_vibes=None):
    if all_vibes is None:
        all_vibes = load_vibes()
    visible_vibes = random.sample(all_vibes, 5)
    return [gr.Button(vibe, variant="secondary", visible=(vibe in visible_vibes)) for vibe in all_vibes], visible_vibes

def get_vibe_description(vibe_name):
    with open("vibe.json", "r", encoding="utf-8") as file:
        vibes = json.load(file)
        for vibe in vibes:
            if vibe["Vibe"] == vibe_name:
                # Replace escaped newlines with actual newlines while preserving other characters
                description = vibe["Description"].replace('\\n', '\n')
                return description
    return ""

def update_selected_vibe(selected_vibe, visible_vibes):
    all_vibes = load_vibes()
    description = get_vibe_description(selected_vibe)
    return [gr.Button(vibe, 
            variant="primary" if vibe == selected_vibe else "secondary",
            visible=(vibe in visible_vibes)) 
            for vibe in all_vibes], visible_vibes

def shuffle_vibes():
    all_vibes = load_vibes()
    visible_vibes = random.sample(all_vibes, 5)
    buttons = [gr.Button(vibe, variant="secondary", visible=(vibe in visible_vibes)) for vibe in all_vibes]
    return *buttons, visible_vibes

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
    if (is_playing):
        is_playing = False
        yield "Audio stopped."
    yield "No audio is playing."

css = """
.voice-buttons {
    display: grid;
    grid-template-columns: repeat(12, 1fr);
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

/* Vibe buttons layout */
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
    .voice-button {
        aspect-ratio: 2/1;
    }
}

@media (max-width: 768px) {
    .voice-buttons {
        grid-template-columns: repeat(3, 1fr);
    }
    .voice-button {
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
) as demo:
    with gr.Row():
        gr.HTML(
            '''
            <div style="text-align: center;"><h1>Azure OpenAI TTS Audio Soundboard</h1></div>
            '''
        )
    with gr.Row():
        voice_label = gr.Label("Voice: ", show_label=False, container=False)    
    with gr.Row(elem_classes="voice-buttons"):
        alloy = gr.Button("Alloy", variant="secondary", elem_classes="voice-button")
        ash = gr.Button("Ash", variant="secondary", icon="assets/ic_fluent_sound_wave_circle_sparkle_24_regular.svg", elem_classes="voice-button")
        ballad = gr.Button("Ballad", variant="secondary", icon="assets/ic_fluent_sound_wave_circle_sparkle_24_regular.svg", elem_classes="voice-button")
        coral = gr.Button("Coral", variant="secondary", icon="assets/ic_fluent_sound_wave_circle_sparkle_24_regular.svg", elem_classes="voice-button")
        echo = gr.Button("Echo", variant="secondary", elem_classes="voice-button")
        fable = gr.Button("Fable", variant="secondary", elem_classes="voice-button")
        nova = gr.Button("Nova", variant="secondary", elem_classes="voice-button")
        onyx = gr.Button("Onyx", variant="secondary", elem_classes="voice-button")
        sage = gr.Button("Sage", variant="secondary", icon="assets/ic_fluent_sound_wave_circle_sparkle_24_regular.svg", elem_classes="voice-button")
        shimmer = gr.Button("Shimmer", variant="secondary", elem_classes="voice-button")
        verse = gr.Button("Verse", variant="secondary", icon="assets/ic_fluent_sound_wave_circle_sparkle_24_regular.svg", elem_classes="voice-button")
        random_btn = gr.Button("Random", variant="huggingface", elem_classes="voice-button")

    def update_button_and_reset(selected_voice):
        buttons = reset_buttons()
        buttons["Alloy Ash Ballad Coral Echo Fable Nova Onyx Sage Shimmer Verse".split().index(selected_voice)] = gr.Button(variant="primary")
        return f"Voice: {selected_voice}", *buttons

    alloy.click(lambda: update_button_and_reset("Alloy"), outputs=[voice_label, alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer, verse])
    ash.click(lambda: update_button_and_reset("Ash"), outputs=[voice_label, alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer, verse])
    ballad.click(lambda: update_button_and_reset("Ballad"), outputs=[voice_label, alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer, verse])
    coral.click(lambda: update_button_and_reset("Coral"), outputs=[voice_label, alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer, verse])
    echo.click(lambda: update_button_and_reset("Echo"), outputs=[voice_label, alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer, verse])
    fable.click(lambda: update_button_and_reset("Fable"), outputs=[voice_label, alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer, verse])
    nova.click(lambda: update_button_and_reset("Nova"), outputs=[voice_label, alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer, verse])
    onyx.click(lambda: update_button_and_reset("Onyx"), outputs=[voice_label, alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer, verse])
    sage.click(lambda: update_button_and_reset("Sage"), outputs=[voice_label, alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer, verse])
    shimmer.click(lambda: update_button_and_reset("Shimmer"), outputs=[voice_label, alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer, verse])
    verse.click(lambda: update_button_and_reset("Verse"), outputs=[voice_label, alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer, verse])

    random_btn.click(update_random_button, outputs=[voice_label, alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer, verse])
    with gr.Row():
        with gr.Column():
            gr.Label("Vibe", container=False)
            with gr.Row(elem_classes="vibe-buttons"):
                all_vibes = load_vibes()
                vibe_buttons, visible_vibes = update_vibe_buttons(all_vibes)
                visible_vibes_state = gr.State(value=visible_vibes)
                shuffle_btn = gr.Button("Shuffle", variant="huggingface", visible=True)
            vibe_desc = gr.Textbox(show_label=False, container=False, lines=8, max_lines=20)

        with gr.Column():
            gr.Label("Script", container=False)
            vibe_script = gr.Textbox(show_label=False, container=False, lines=8, max_lines=20)
            audio_output = gr.Audio(autoplay=True, streaming=True)
            play_btn = gr.Button(value="Play", variant="primary", icon=os.path.join("assets", "ic_fluent_play_24_filled.svg"), visible=True)
            stop_btn = gr.Button(value="Stop", variant="stop", icon=os.path.join("assets", "ic_fluent_stop_24_filled.svg"), visible=False)

        for vibe_button in vibe_buttons:            vibe_button.click(
                lambda vibe, current_vibes: (get_vibe_info(vibe)[0], *update_selected_vibe(vibe, current_vibes)[0], current_vibes, get_vibe_info(vibe)[1]),
                inputs=[vibe_button, visible_vibes_state],
                outputs=[vibe_desc, *vibe_buttons, visible_vibes_state, vibe_script]
            )
            
        shuffle_btn.click(shuffle_vibes,
            outputs=[*vibe_buttons, visible_vibes_state]        )

    def toggle_play_stop(voice_name, vibe_desc, vibe_script):
        """Handle the play button click and toggle button visibility"""
        global is_playing
        is_playing = True
        try:            
            api_key = check_api_key()
            # Make sure we have a voice name and properly format it
            if not voice_name or not isinstance(voice_name, str):
                raise ValueError("No voice selected. Please select a voice first.")
            voice_name = voice_name.replace("Voice: ", "").strip().lower()
            if not voice_name:
                raise ValueError("Invalid voice name. Please select a valid voice.")
            
            # Create a temporary file path
            temp_file = os.path.join(temp_dir, f"{voice_name}_{int(time.time())}.mp3")
            
            # Generate and save audio to temp file
            asyncio.run(generate_audio_file(vibe_script, temp_file, voice_name, vibe_desc))
            
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
        
if __name__ == "__main__":
    asyncio.run(demo.launch(favicon_path="assets/ai_studio_icon_color.png"))