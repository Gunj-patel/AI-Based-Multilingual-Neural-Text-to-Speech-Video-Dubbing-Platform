import os
import re
import tempfile
import warnings
import logging

import librosa
import numpy as np
import soundfile as sf
import streamlit as st
import torch
from deep_translator import GoogleTranslator
from langdetect import detect_langs, DetectorFactory
from num2words import num2words
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip

from .config import INDIC_VOICE_POOL, INDIC_VOICE_DEFAULT, HF_ROOT

DetectorFactory.seed = 0
warnings.filterwarnings("ignore")
for _log in ("nemo_logger", "huggingface_hub", "huggingface_hub.file_download", "transformers", "datasets", "nemo", "root"):
    logging.getLogger(_log).setLevel(logging.ERROR)


def normalize_math_and_numbers(text):
    def replace_numbers(match):
        num = match.group()
        try:
            return num2words(float(num)) if "." in num else num2words(int(num))
        except Exception:
            return num

    text = re.sub(r"\d+(\.\d+)?", replace_numbers, text)
    replacements = {
        r"\+": " plus ",
        r"-": " minus ",
        r"\*": " multiplied by ",
        r"/": " divided by ",
        r"=": " equals ",
        r"%": " percent ",
    }
    for sym, word in replacements.items():
        text = re.sub(sym, word, text)
    return re.sub(r"\s+", " ", text).strip()


def split_text_for_speecht5(text, max_chars=350):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) <= max_chars:
            current += " " + s
        else:
            if current.strip():
                chunks.append(current.strip())
            current = s
    if current.strip():
        chunks.append(current.strip())
    return chunks or [text]


def split_into_sentences(text: str) -> list:
    parts = re.split(r"(?<=[.!.?])\s+", text.strip())
    return [s.strip() for s in parts if s.strip()] or [text.strip()]


def detect_speech_segments(audio_np, sr, top_db=28, min_silence_len=0.25):
    intervals = librosa.effects.split(audio_np, top_db=top_db)
    if len(intervals) == 0:
        return [[0, len(audio_np)]]
    min_gap = int(min_silence_len * sr)
    merged = [intervals[0].tolist()]
    for start, end in intervals[1:]:
        if start - merged[-1][1] < min_gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return merged


def diarize_speakers(audio_np, sr, segments):
    pitches = []
    for start, end in segments:
        chunk = audio_np[start:end]
        if len(chunk) < int(sr * 0.1):
            pitches.append(None)
            continue
        try:
            f0, voiced, _ = librosa.pyin(
                chunk,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=sr,
            )
            voiced_f0 = f0[voiced] if voiced is not None else np.array([])
            pitches.append(float(np.median(voiced_f0)) if len(voiced_f0) > 0 else None)
        except Exception:
            pitches.append(None)

    valid = [p for p in pitches if p is not None]
    global_median = float(np.median(valid)) if valid else 150.0
    pitches = [p if p is not None else global_median for p in pitches]

    all_p = np.array(pitches)
    q33 = float(np.percentile(all_p, 33))
    q66 = float(np.percentile(all_p, 66))

    speaker_ids = []
    for p in pitches:
        if p <= q33:
            speaker_ids.append(0)
        elif p <= q66:
            speaker_ids.append(1)
        else:
            speaker_ids.append(2)
    return speaker_ids


def get_speaker_gender_map(speaker_ids, user_gender_overrides: dict) -> dict:
    auto_map = {}
    for sid in sorted(set(speaker_ids)):
        auto_map[sid] = "male" if sid in (0, 1) else "female"
    for sid, gender in user_gender_overrides.items():
        auto_map[sid] = gender
    return auto_map


def distribute_sentences_by_duration(sentences: list, segments: list) -> list:
    num_seg = len(segments)
    if num_seg == 0:
        return []
    if num_seg == 1:
        return [" ".join(sentences)]

    seg_durations = [seg[1] - seg[0] for seg in segments]
    total_dur = max(sum(seg_durations), 1)
    n_sent = len(sentences)
    targets = [max(1, round((d / total_dur) * n_sent)) for d in seg_durations]

    text_chunks = []
    cursor = 0
    for i, target in enumerate(targets):
        remaining_segs = num_seg - i
        remaining_sents = n_sent - cursor
        take = min(max(1, target), remaining_sents - (remaining_segs - 1))
        take = max(1, take)
        chunk = sentences[cursor: cursor + take]
        text_chunks.append(" ".join(chunk).strip() if chunk else "")
        cursor += take
        if cursor >= n_sent:
            while len(text_chunks) < num_seg:
                text_chunks.append(sentences[-1])
            break

    while len(text_chunks) < num_seg:
        text_chunks.append(sentences[-1] if sentences else "")

    return text_chunks[:num_seg]


def get_indic_description(language_name: str, gender: str) -> str:
    desc = INDIC_VOICE_POOL.get(
        (language_name, gender),
        INDIC_VOICE_POOL.get((language_name, "neutral"), INDIC_VOICE_DEFAULT),
    )
    if gender == "male":
        return "A male speaker with a deep masculine voice. " + desc
    if gender == "female":
        return "A female speaker with a clear feminine voice. " + desc
    return desc


def _hf_model_cached(model_slug: str) -> bool:
    folder = os.path.join(HF_ROOT, "models--" + model_slug.replace("/", "--"))
    snaps = os.path.join(folder, "snapshots")
    if not os.path.isdir(snaps):
        return False
    return any(os.path.isdir(os.path.join(snaps, s)) for s in os.listdir(snaps))


def _hf_dataset_cached(dataset_slug: str) -> bool:
    folder = os.path.join(HF_ROOT, "datasets--" + dataset_slug.replace("/", "--"))
    return os.path.isdir(folder)


def get_cache_state() -> tuple[bool, bool]:
    xvec_cached = _hf_dataset_cached("Matthijs/cmu-arctic-xvectors") or os.path.isdir(
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "datasets", "Matthijs___cmu-arctic-xvectors")
    )
    all_cached = (
        _hf_model_cached("microsoft/speecht5_tts")
        and _hf_model_cached("microsoft/speecht5_hifigan")
        and _hf_model_cached("j-hartmann/emotion-english-distilroberta-base")
        and xvec_cached
        and os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "indic_conformer_600m.nemo"))
    )
    return all_cached, xvec_cached


def extract_audio_from_video(video_path):
    clip = VideoFileClip(video_path)
    if clip.audio is None:
        clip.close()
        st.error("❌ This video has no audio track. Please upload a video with audio.")
        st.stop()
    audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    clip.audio.write_audiofile(audio_path, fps=16000, codec="pcm_s16le", logger=None)
    clip.close()
    return audio_path


def speech_to_text(audio_path, lang_code, nemo_id, asr_model, asr_type):
    try:
        audio_np, sr = librosa.load(audio_path, sr=16000, mono=True)
        if len(audio_np) < 1600:
            st.warning("⚠️ Audio too short or empty — please check your video has audio.")
            return ""
        audio_np = audio_np.astype(np.float32)
        if len(audio_np) < 16000:
            audio_np = np.pad(audio_np, (0, 16000 - len(audio_np)))
        validated_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        sf.write(validated_path, audio_np, 16000)
        result = asr_model.transcribe(validated_path)
        return result["text"]
    except Exception as e:
        st.error(f"❌ Transcription failed: {e}")
        return ""


def translate_text(text, target):
    try:
        return GoogleTranslator(source="auto", target=target).translate(text)
    except Exception:
        return text


def generate_gtts_fallback(text, lang_code):
    from gtts import gTTS

    tts = gTTS(text=text, lang=lang_code)
    mp3 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(mp3.name)
    audio, sr = librosa.load(mp3.name, sr=16000)
    return audio, sr


def generate_indic_tts(text, language_name, gender, model, tokenizer, device, speed=1.0):
    description = get_indic_description(language_name, gender)
    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_ids)
    audio = generation.cpu().numpy().squeeze().astype(np.float32)
    sr = model.config.sampling_rate
    gender_shift = -4 if gender == "male" else (3 if gender == "female" else 0)
    if gender_shift != 0:
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=float(gender_shift))
    if speed != 1.0:
        audio = librosa.effects.time_stretch(audio, rate=speed)
    return audio, sr
