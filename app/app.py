import multiprocessing
import os
import tempfile
import time

import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import streamlit as st
import torch
from langdetect import detect_langs
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.io.VideoFileClip import VideoFileClip

from .config import LANGUAGE_CONFIG
from .models import device, load_asr_model, load_emotion_model, load_english_tts_models, load_indic_tts
from .utils import (
    detect_speech_segments,
    diarize_speakers,
    distribute_sentences_by_duration,
    extract_audio_from_video,
    generate_gtts_fallback,
    generate_indic_tts,
    get_indic_description,
    get_speaker_gender_map,
    normalize_math_and_numbers,
    speech_to_text,
    split_into_sentences,
    split_text_for_speecht5,
    translate_text,
    get_cache_state,
)

multiprocessing.freeze_support()

st.set_page_config(page_title="AI4Bharat Neural TTS", layout="centered")
st.title("AI Based Multilingual Neural Text to Speech Platform️")

st.write("⚙️ Running on:", device.upper())
all_cached, _ = get_cache_state()
if all_cached:
    st.success("✅ All models cached locally — fully offline mode. Click Generate to load.")
else:
    st.info("📥 Some models not yet cached — first-time run will download them.")

st.divider()

uploaded = st.file_uploader(
    "📁 Upload TXT / Audio (WAV, MP3) / Video (MP4, AVI, MOV)",
    type=["txt", "wav", "mp3", "mp4", "avi", "mov"],
)
text_input = st.text_area(
    "✏️ Or enter text directly",
    height=150,
    placeholder="Type in any Indian language or English...",
)

lang_names = list(LANGUAGE_CONFIG.keys())
output_language = st.selectbox("🌐 Output Language", lang_names, index=0)

col1, col2, col3 = st.columns(3)
with col2:
    pitch = st.slider("🎵 Pitch shift (English only)", -5, 5, 0)
with col3:
    speed = st.slider("⚡ Speed (non-video only)", 0.7, 1.5, 1.0, step=0.05)

if output_language != "English":
    single_gender = st.radio(
        "🗣️ Voice Gender (for single-speaker preview)",
        ["female", "male", "neutral"],
        horizontal=True,
        index=0,
    )
else:
    single_gender = "neutral"

video_file = uploaded if uploaded and uploaded.name.endswith(("mp4", "avi", "mov")) else None
audio_file = uploaded if uploaded and uploaded.name.endswith(("wav", "mp3")) else None

generate_video = False
speaker_gender_overrides: dict = {}

if video_file:
    generate_video = st.checkbox("🎬 Generate translated MP4 with synced multi-speaker audio", value=True)
    with st.expander("🎚️ Manual Speaker Gender Override (optional)"):
        st.caption("Override auto-detected gender per speaker.")
        for spk_id in range(4):
            col_a, col_b = st.columns([1, 3])
            with col_a:
                st.write(f"Speaker {spk_id}")
            with col_b:
                choice = st.radio(
                    f"Gender for Speaker {spk_id}",
                    ["auto", "male", "female", "neutral"],
                    horizontal=True,
                    key=f"spk_gender_{spk_id}",
                    label_visibility="collapsed",
                )
                if choice != "auto":
                    speaker_gender_overrides[spk_id] = choice

if st.button("🔊 Generate & Play", type="primary"):
    lang_cfg = LANGUAGE_CONFIG[output_language]
    lang_code = lang_cfg["code"]
    nemo_id = lang_cfg["nemo_id"]

    with st.status("⏳ Loading models…", expanded=True) as status_box:
        t0 = time.time()
        st.write("🔊 Loading English TTS (SpeechT5)…")
        processor, en_tts_model, vocoder, voice_options = load_english_tts_models()
        st.write(f"   ✅ English TTS ready ({time.time() - t0:.1f}s)")

        voice_choice = "Male-like" if single_gender == "male" else ("Female-like" if single_gender == "female" else "Neutral")

        if output_language != "English":
            st.write("🇮🇳 Loading IndicParler-TTS…")
            t1 = time.time()
            indic_tts_model, indic_tts_tokenizer, indic_tts_type = load_indic_tts()
            st.write(f"   ✅ Indic TTS — backend: `{indic_tts_type}` ({time.time() - t1:.1f}s)")
        else:
            indic_tts_model, indic_tts_tokenizer, indic_tts_type = None, None, "gtts"

        if video_file or audio_file:
            st.write("🎙️ Loading ASR (IndicConformer)…")
            t1 = time.time()
            asr_model, asr_type = load_asr_model()
            st.write(f"   ✅ ASR — backend: `{asr_type}` ({time.time() - t1:.1f}s)")
        else:
            asr_model, asr_type = None, "none"

        st.write("🧠 Loading emotion classifier…")
        t1 = time.time()
        emotion_classifier = load_emotion_model()
        st.write(f"   ✅ Emotion model ready ({time.time() - t1:.1f}s)")

        status_box.update(label="✅ Models loaded!", state="complete", expanded=False)

    video_path = None
    audio_path = None

    if video_file:
        video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        with open(video_path, "wb") as f:
            f.write(video_file.read())
        with st.spinner("🎬 Extracting audio from video…"):
            audio_path = extract_audio_from_video(video_path)
        with st.spinner(f"🎙️ Transcribing with {asr_type}…"):
            final_text = speech_to_text(audio_path, lang_code, nemo_id, asr_model, asr_type)
    elif audio_file:
        audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]).name
        with open(audio_path, "wb") as f:
            f.write(audio_file.read())
        with st.spinner(f"🎙️ Transcribing with {asr_type}…"):
            final_text = speech_to_text(audio_path, lang_code, nemo_id, asr_model, asr_type)
    elif uploaded and uploaded.name.endswith(".txt"):
        final_text = uploaded.read().decode("utf-8").strip()
    else:
        final_text = text_input.strip()

    if not final_text:
        st.warning("⚠️ Please provide some input text or upload a file.")
        st.stop()

    st.subheader("📝 Input Text")
    st.write(final_text)

    final_text = normalize_math_and_numbers(final_text)

    try:
        detected_lang = detect_langs(final_text)[0].lang
    except Exception:
        detected_lang = lang_code

    if detected_lang != lang_code:
        with st.spinner(f"🌐 Translating to {output_language}…"):
            final_text = translate_text(final_text, lang_code)
        st.subheader("🌐 Translated Text")
        st.write(final_text)

    try:
        ei = final_text if detected_lang == "en" else translate_text(final_text, "en")
        emotion_scores = emotion_classifier(ei, truncation=True, max_length=512)[0]
        best_emotion = max(emotion_scores, key=lambda x: x["score"])["label"]
    except Exception:
        best_emotion = "neutral"

    with st.spinner("🔊 Generating speech preview…"):
        if output_language == "English":
            audio_chunks = []
            for chunk in split_text_for_speecht5(final_text):
                inputs = processor(text=chunk, return_tensors="pt").to(device)
                with torch.no_grad():
                    speech = en_tts_model.generate_speech(
                        inputs["input_ids"],
                        speaker_embeddings=voice_options[voice_choice],
                        vocoder=vocoder,
                    )
                audio_chunks.append(speech.cpu().numpy())
            audio = np.concatenate(audio_chunks)
            sr = 16000
            if pitch != 0:
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch)
            if speed != 1.0:
                audio = librosa.effects.time_stretch(audio, rate=speed)
        else:
            if indic_tts_type == "indicparler":
                audio, sr = generate_indic_tts(
                    final_text,
                    output_language,
                    single_gender,
                    indic_tts_model,
                    indic_tts_tokenizer,
                    device,
                    speed=speed,
                )
            else:
                audio, sr = generate_gtts_fallback(final_text, lang_code)
                if speed != 1.0:
                    audio = librosa.effects.time_stretch(audio, rate=speed)

    audio = audio.astype(np.float32)
    mx = np.max(np.abs(audio))
    if mx > 0:
        audio /= mx
    wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    sf.write(wav_path, audio, sr)

    st.success(f"✅ Speech generated in **{output_language}** | Emotion: `{best_emotion}`")
    st.audio(wav_path)

    st.subheader("🧪 Speech Quality Metrics")
    rms = np.mean(librosa.feature.rms(y=audio))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio))
    col_m1, col_m2 = st.columns(2)
    col_m1.metric("🔊 RMS Energy", round(float(rms), 4))
    col_m2.metric("✂️ Zero Crossing Rate", round(float(zcr), 4))

    st.subheader("📊 Prosody Visualization")
    fig1, ax1 = plt.subplots()
    librosa.display.waveshow(audio, sr=sr, ax=ax1)
    ax1.set_title("Waveform")
    st.pyplot(fig1)

    f0, _, _ = librosa.pyin(audio, fmin=50, fmax=300)
    fig2, ax2 = plt.subplots()
    ax2.plot(f0)
    ax2.set_title("Pitch Contour (F0)")
    st.pyplot(fig2)

    S = librosa.feature.melspectrogram(y=audio, sr=sr)
    fig3, ax3 = plt.subplots()
    img = librosa.display.specshow(librosa.power_to_db(S), sr=sr, x_axis="time", y_axis="mel", ax=ax3)
    plt.colorbar(img, ax=ax3)
    ax3.set_title("Mel Spectrogram")
    st.pyplot(fig3)

    st.subheader("🧠 AI Understanding Summary")
    st.markdown(f"""
    🌍 **Detected Language:** `{detected_lang}`  
    🗣️ **Output Language:** `{output_language}`  
    🧠 **Emotion:** `{best_emotion}`  
    🎙️ **ASR Backend:** `{asr_type}`  
    🔊 **TTS Backend:** `{indic_tts_type}`  
    💻 **Device:** `{device}`
    """)
    if output_language != "English" and indic_tts_type == "indicparler":
        st.markdown(f"🗣️ **Voice Description Used:** _{get_indic_description(output_language, single_gender)}_")

    st.subheader("📝 AI Processing Report")
    report = f"""
INPUT TEXT:
{final_text}

Detected Language : {detected_lang}
Output Language   : {output_language}
Detected Emotion  : {best_emotion}
ASR Backend       : {asr_type}
TTS Backend       : {indic_tts_type}
Voice Style       : {voice_choice if output_language == 'English' else f'IndicParler-TTS ({single_gender}) Indian Accent'}
Voice Description : {get_indic_description(output_language, single_gender) if output_language != 'English' else 'N/A'}
RMS Energy        : {round(float(rms), 4)}
Zero Crossing Rate: {round(float(zcr), 4)}
Device Used       : {device}
"""
    st.text_area("📄 System Report", report, height=260)
    st.download_button("📥 Download Report", report, "ai4bharat_tts_report.txt")

    if generate_video and video_file and video_path and audio_path:
        st.divider()
        st.subheader("🎬 Generating Multi-Speaker Synchronized Video…")

        with st.spinner("🔍 Running VAD + speaker diarization…"):
            orig_audio_np, orig_sr = librosa.load(audio_path, sr=16000, mono=True)
            segments = detect_speech_segments(orig_audio_np, orig_sr)
            speaker_ids = diarize_speakers(orig_audio_np, orig_sr, segments)
            unique_spk = sorted(set(speaker_ids))
            spk_gender_map = get_speaker_gender_map(speaker_ids, speaker_gender_overrides)

            st.write(
                f"📍 **{len(segments)}** speech segment(s) | 🗣️ **{len(unique_spk)}** distinct speaker group(s): {unique_spk}"
            )

            COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
            fig_d, ax_d = plt.subplots(figsize=(10, 1.8))
            seen_labels = set()
            for i, (start, end) in enumerate(segments):
                spk = speaker_ids[i]
                g = spk_gender_map.get(spk, "neutral")
                label = f"Speaker {spk} ({g})" if spk not in seen_labels else "_nolegend_"
                seen_labels.add(spk)
                ax_d.barh(0, (end - start) / orig_sr, left=start / orig_sr, height=0.5, color=COLORS[spk % len(COLORS)], label=label)
            ax_d.set_xlabel("Time (s)")
            ax_d.set_yticks([])
            ax_d.set_title("Speaker Diarization Timeline")
            ax_d.legend(loc="upper right", fontsize=7)
            st.pyplot(fig_d)

            sentences = split_into_sentences(final_text)
            text_chunks = distribute_sentences_by_duration(sentences, segments)
            with st.expander("📋 Sentence-to-Segment Assignment Preview"):
                for i, (chunk, (s, e)) in enumerate(zip(text_chunks, segments)):
                    spk = speaker_ids[i]
                    g = spk_gender_map.get(spk, "neutral")
                    st.write(
                        f"**Seg {i+1}** [{s/orig_sr:.2f}s–{e/orig_sr:.2f}s] Spk {spk} ({g}): {chunk[:80]}{'…' if len(chunk) > 80 else ''}"
                    )

        st.info("ℹ️ Using preview-quality generated audio for MP4 output.")
        synced_audio = audio.astype(np.float32).copy()
        synced_sr = sr

        synced_wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        sf.write(synced_wav_path, synced_audio, synced_sr)
        st.write("🎧 Synced translated audio preview:")
        st.audio(synced_wav_path, format="audio/wav")

        with st.spinner("🎬 Compositing final video…"):
            video_clip = VideoFileClip(video_path)
            new_audio_clip = AudioFileClip(synced_wav_path)
            vid_dur = video_clip.duration
            aud_dur = new_audio_clip.duration

            if aud_dur > vid_dur:
                new_audio_clip = new_audio_clip.with_duration(vid_dur - 0.05)
            elif aud_dur < vid_dur:
                pad_np = np.zeros(int((vid_dur - aud_dur) * synced_sr), dtype=np.float32)
                padded = np.concatenate([synced_audio, pad_np])
                pad_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
                sf.write(pad_path, padded, synced_sr)
                new_audio_clip = AudioFileClip(pad_path).with_duration(vid_dur - 0.05)

            final_video = video_clip.with_audio(new_audio_clip)
            output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            final_video.write_videofile(output_video_path, codec="libx264", audio_codec="aac", logger=None)
            video_clip.close()
            new_audio_clip.close()
            final_video.close()

        st.video(output_video_path)
        st.success(f"✅ Multi-speaker synchronized video ready! ({len(unique_spk)} distinct TTS voice(s) used)")
        with open(output_video_path, "rb") as f:
            st.download_button(
                "📥 Download Translated Video",
                f.read(),
                file_name="translated_multispeaker_video.mp4",
                mime="video/mp4",
            )
