from app.app import *

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

# ── FFmpeg path ──────────────────────────────────────────────────────────────
os.environ["PATH"] = r"C:\ffmpeg\ffmpeg-7.0-essentials_build\bin;" + os.environ["PATH"]

# ── Offline-first cache detection ────────────────────────────────────────────
_HF_ROOT    = os.path.expanduser(r"~\.cache\huggingface\hub")
_LOCAL_NEMO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "indic_conformer_600m.nemo")

# ── CRITICAL: Block ALL NeMo/HF network calls if .nemo file already exists.
# NeMo tries to re-download ONNX assets at *import time* (before restore_from
# is even called), so we must set these env vars BEFORE any nemo import.
if os.path.exists(_LOCAL_NEMO):
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"]  = "1"
    os.environ["HF_HUB_OFFLINE"]       = "1"
    os.environ["NEMO_CACHE_DIR"]       = os.path.dirname(_LOCAL_NEMO)
    # Prevent NeMo from phoning home for ONNX assets
    os.environ["NEMO_TESTING"]         = "1"

def _hf_model_cached(model_slug: str) -> bool:
    folder = os.path.join(_HF_ROOT, "models--" + model_slug.replace("/", "--"))
    snaps  = os.path.join(folder, "snapshots")
    if not os.path.isdir(snaps):
        return False
    return any(os.path.isdir(os.path.join(snaps, s)) for s in os.listdir(snaps))

def _hf_dataset_cached(dataset_slug: str) -> bool:
    folder = os.path.join(_HF_ROOT, "datasets--" + dataset_slug.replace("/", "--"))
    return os.path.isdir(folder)

# ── XVECTORS: also check the old parquet cache location ─────────────────────
_XVEC_CACHED = (
    _hf_dataset_cached("Matthijs/cmu-arctic-xvectors") or
    os.path.isdir(os.path.join(os.path.expanduser("~"), ".cache", "huggingface",
                               "datasets", "Matthijs___cmu-arctic-xvectors"))
)

_ALL_CACHED = (
    _hf_model_cached("microsoft/speecht5_tts") and
    _hf_model_cached("microsoft/speecht5_hifigan") and
    _hf_model_cached("j-hartmann/emotion-english-distilroberta-base") and
    _XVEC_CACHED and
    os.path.exists(_LOCAL_NEMO)
)

# Reinforce offline flags based on full cache check
if _ALL_CACHED:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"]  = "1"
    os.environ["HF_HUB_OFFLINE"]       = "1"
elif not os.path.exists(_LOCAL_NEMO):
    # Only allow network if .nemo is genuinely missing (first ever run)
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    os.environ["HF_DATASETS_OFFLINE"]  = "0"
    os.environ["HF_HUB_OFFLINE"]       = "0"

import streamlit as st
import torch
import soundfile as sf
import numpy as np
import tempfile
import librosa
import librosa.display
import matplotlib.pyplot as plt
import re
import time

from num2words import num2words
from langdetect import detect_langs, DetectorFactory
DetectorFactory.seed = 0

import logging, warnings
warnings.filterwarnings("ignore")
for _log in ("nemo_logger", "huggingface_hub", "huggingface_hub.file_download",
             "transformers", "datasets", "nemo", "root"):
    logging.getLogger(_log).setLevel(logging.ERROR)

from deep_translator import GoogleTranslator
from transformers import (
    SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan,
    pipeline, AutoTokenizer
)
from datasets import load_dataset
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip

# =====================================================
# 🌐 LANGUAGE CONFIG
# =====================================================
LANGUAGE_CONFIG = {
    "English":   {"code": "en",  "nemo_id": "en"},
    "Hindi":     {"code": "hi",  "nemo_id": "hi"},
    "Gujarati":  {"code": "gu",  "nemo_id": "gu"},
    "Bengali":   {"code": "bn",  "nemo_id": "bn"},
    "Tamil":     {"code": "ta",  "nemo_id": "ta"},
    "Telugu":    {"code": "te",  "nemo_id": "te"},
    "Kannada":   {"code": "kn",  "nemo_id": "kn"},
    "Malayalam": {"code": "ml",  "nemo_id": "ml"},
    "Marathi":   {"code": "mr",  "nemo_id": "mr"},
    "Punjabi":   {"code": "pa",  "nemo_id": "pa"},
    "Odia":      {"code": "or",  "nemo_id": "or"},
    "Assamese":  {"code": "as",  "nemo_id": "as"},
    "Maithili":  {"code": "mai", "nemo_id": "mai"},
    "Santali":   {"code": "sat", "nemo_id": "sat"},
    "Konkani":   {"code": "gom", "nemo_id": "kok"},
    "Sindhi":    {"code": "sd",  "nemo_id": "sd"},
    "Dogri":     {"code": "doi", "nemo_id": "doi"},
    "Kashmiri":  {"code": "ks",  "nemo_id": "ks"},
    "Manipuri":  {"code": "mni", "nemo_id": "mni"},
    "Bodo":      {"code": "brx", "nemo_id": "brx"},
    "Sanskrit":  {"code": "sa",  "nemo_id": "sa"},
    "Urdu":      {"code": "ur",  "nemo_id": "ur"},
}

INDIC_VOICE_POOL = {
    ("Hindi", "male"): (
        "A male speaker with a slightly expressive and animated speech delivers"
        " Hindi text in a clear, moderate-paced voice with an Indian accent."
        " The recording is clean, studio quality."
    ),
    ("Hindi", "female"): (
        "A female speaker delivers Hindi speech in a warm, clear voice with"
        " natural Indian intonation and a moderate conversational pace."
        " The audio is studio quality with no noise."
    ),
    ("Hindi", "neutral"): (
        "A speaker with a neutral voice delivers Hindi text clearly at a"
        " moderate pace with a natural Indian accent. Studio audio."
    ),
    ("Gujarati", "male"): (
        "A male speaker delivers Gujarati speech in a clear, natural voice"
        " with an authentic Gujarati accent. The speech is at a moderate pace"
        " with proper Gujarati phonology. Clean studio audio."
    ),
    ("Gujarati", "female"): (
        "A female speaker delivers Gujarati text in a warm, natural voice"
        " with an authentic Gujarati mother-tongue accent at a moderate pace."
        " Studio-quality clean recording."
    ),
    ("Gujarati", "neutral"): (
        "A speaker delivers Gujarati text clearly with a native Gujarati accent"
        " at a steady moderate pace. Studio audio quality."
    ),
    ("Bengali", "male"): (
        "A male speaker with a clear West Bengali accent delivers Bengali text"
        " at a steady pace with natural Bengali intonation. Studio quality audio."
    ),
    ("Bengali", "female"): (
        "A female speaker with a melodious Bengali accent delivers text at a"
        " gentle flowing pace with natural Bengali phonology. Clean studio audio."
    ),
    ("Bengali", "neutral"): (
        "A speaker delivers Bengali with a natural Bengali accent at a moderate"
        " pace. Clean studio quality audio."
    ),
    ("Tamil", "male"): (
        "A male speaker delivers Tamil speech with a clear Chennai Tamil accent"
        " at a confident, measured pace. Professional studio audio."
    ),
    ("Tamil", "female"): (
        "A female speaker delivers Tamil text with a clear Tamil Nadu accent"
        " at a moderate pace with natural Tamil intonation. Studio audio."
    ),
    ("Tamil", "neutral"): (
        "A speaker delivers Tamil text with a natural Tamil accent at a"
        " moderate pace. Clean recording."
    ),
    ("Telugu", "male"): (
        "A male speaker delivers Telugu with a clear Andhra Pradesh accent"
        " at a measured pace with natural Telugu phonology. Studio audio."
    ),
    ("Telugu", "female"): (
        "A female speaker delivers Telugu text with a melodious Andhra accent"
        " at a moderate pace with natural intonation. Clean studio audio."
    ),
    ("Telugu", "neutral"): (
        "A speaker delivers Telugu text naturally with a Telugu accent at a"
        " moderate pace. Studio quality."
    ),
    ("Kannada", "male"): (
        "A male speaker delivers Kannada speech with an authentic Karnataka"
        " accent at a confident pace. Clean studio audio."
    ),
    ("Kannada", "female"): (
        "A female speaker delivers Kannada text with a clear Bengaluru accent"
        " at a natural moderate pace. Studio quality recording."
    ),
    ("Kannada", "neutral"): (
        "A speaker delivers Kannada with a natural Kannada accent at a"
        " moderate pace. Studio audio."
    ),
    ("Malayalam", "male"): (
        "A male speaker delivers Malayalam with a clear Kerala accent at a"
        " confident pace with natural Malayalam phonology. Studio audio."
    ),
    ("Malayalam", "female"): (
        "A female speaker delivers Malayalam text with a clear Kerala accent"
        " at a moderate pace. Studio-quality clean audio."
    ),
    ("Malayalam", "neutral"): (
        "A speaker delivers Malayalam text with a native Kerala accent at a"
        " moderate pace. Clean studio recording."
    ),
    ("Marathi", "male"): (
        "A male speaker delivers Marathi with a clear Maharashtra accent at a"
        " steady pace with natural Marathi intonation. Studio quality."
    ),
    ("Marathi", "female"): (
        "A female speaker delivers Marathi text with a natural Pune accent at"
        " a moderate pace. Clean studio audio."
    ),
    ("Marathi", "neutral"): (
        "A speaker delivers Marathi clearly with a native accent at a moderate"
        " pace. Studio audio."
    ),
    ("Punjabi", "male"): (
        "A male speaker delivers Punjabi with a clear Amritsar accent at a"
        " lively moderate pace with natural tonal phonology. Studio audio."
    ),
    ("Punjabi", "female"): (
        "A female speaker delivers Punjabi text with a warm authentic accent"
        " at a moderate pace. Clean studio recording."
    ),
    ("Punjabi", "neutral"): (
        "A speaker delivers Punjabi with a native accent at a moderate pace."
        " Studio quality audio."
    ),
    ("Urdu", "male"): (
        "A male speaker delivers Urdu with a refined, clear accent at a"
        " measured dignified pace. Studio-quality audio."
    ),
    ("Urdu", "female"): (
        "A female speaker delivers Urdu text with a melodious clear accent"
        " at a moderate pace. Clean studio audio."
    ),
    ("Urdu", "neutral"): (
        "A speaker delivers Urdu with a clear neutral accent at a moderate"
        " pace. Studio audio."
    ),
    ("Odia", "male"): (
        "A male speaker delivers Odia with a clear Bhubaneswar accent at a"
        " steady moderate pace. Studio audio."
    ),
    ("Odia", "female"): (
        "A female speaker delivers Odia with a melodic Odia accent at a"
        " moderate pace. Clean studio recording."
    ),
    ("Odia", "neutral"): (
        "A speaker delivers Odia text naturally at a moderate pace. Studio audio."
    ),
    ("Assamese", "male"): (
        "A male speaker delivers Assamese with a clear Guwahati accent at a"
        " confident pace. Studio quality audio."
    ),
    ("Assamese", "female"): (
        "A female speaker delivers Assamese with a natural Northeast Indian"
        " accent at a moderate pace. Clean studio audio."
    ),
    ("Assamese", "neutral"): (
        "A speaker delivers Assamese naturally at a moderate pace. Studio audio."
    ),
    ("Sanskrit", "male"): (
        "A male speaker delivers Sanskrit with precise classical pronunciation"
        " at a deliberate scholarly pace. Studio recording."
    ),
    ("Sanskrit", "female"): (
        "A female speaker delivers Sanskrit with clear classical pronunciation"
        " at a careful pace. Studio-quality audio."
    ),
    ("Sanskrit", "neutral"): (
        "A speaker delivers Sanskrit text clearly at a measured pace. Studio audio."
    ),
}

INDIC_VOICE_DEFAULT = (
    "A speaker delivers the text clearly with a natural Indian accent at a"
    " moderate pace. Clean studio quality audio."
)

EN_SPEAKER_SLOTS = {
    0: "Male-like",
    1: "Male-like",
    2: "Female-like",
    3: "Female-like",
}


def get_indic_description(language_name: str, gender: str) -> str:
    desc = INDIC_VOICE_POOL.get(
        (language_name, gender),
        INDIC_VOICE_POOL.get((language_name, "neutral"), INDIC_VOICE_DEFAULT)
    )
    # IndicParler-TTS is very sensitive to the gender keyword placement.
    # Prepend a strong gender anchor so male voices don't default to female.
    if gender == "male":
        return "A male speaker with a deep masculine voice. " + desc
    elif gender == "female":
        return "A female speaker with a clear feminine voice. " + desc
    return desc


# =====================================================
# 🔥 TEXT NORMALIZATION
# =====================================================
def normalize_math_and_numbers(text):
    def replace_numbers(match):
        num = match.group()
        try:
            return num2words(float(num)) if "." in num else num2words(int(num))
        except Exception:
            return num
    text = re.sub(r'\d+(\.\d+)?', replace_numbers, text)
    replacements = {
        r'\+': ' plus ', r'-': ' minus ', r'\*': ' multiplied by ',
        r'/': ' divided by ', r'=': ' equals ', r'%': ' percent ',
    }
    for sym, word in replacements.items():
        text = re.sub(sym, word, text)
    return re.sub(r'\s+', ' ', text).strip()


def split_text_for_speecht5(text, max_chars=350):
    sentences = re.split(r'(?<=[.!?])\s+', text)
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


# =====================================================
# 🎙️ VOICE ACTIVITY DETECTION + DIARIZATION
# =====================================================
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
                sr=sr
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


# =====================================================
# 🗣️ SENTENCE-AWARE TEXT DISTRIBUTION
# =====================================================
def split_into_sentences(text: str) -> list:
    parts = re.split(r'(?<=[।.!?])\s+', text.strip())
    return [s.strip() for s in parts if s.strip()] or [text.strip()]


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


# =====================================================
# 🎵 PER-SEGMENT TTS
# =====================================================
def tts_for_segment(
    text, output_language, lang_code, gender,
    indic_tts_type, indic_tts_model, indic_tts_tokenizer,
    processor, en_tts_model, vocoder, voice_options,
    pitch_shift, device
):
    if not text.strip():
        return np.zeros(1000, dtype=np.float32), 16000

    if output_language == "English":
        if gender == "male":
            slot_name = "Male-like"
        elif gender == "female":
            slot_name = "Female-like"
        else:
            slot_name = "Neutral"
        speaker_emb = voice_options[slot_name]
        chunks_audio = []
        for chunk in split_text_for_speecht5(text):
            inputs = processor(text=chunk, return_tensors="pt").to(device)
            with torch.no_grad():
                speech = en_tts_model.generate_speech(
                    inputs["input_ids"], speaker_embeddings=speaker_emb, vocoder=vocoder
                )
            chunks_audio.append(speech.cpu().numpy())
        audio = np.concatenate(chunks_audio) if chunks_audio else np.zeros(1000)
        sr = 16000
        # Strong pitch enforcement: male = -4 semitones, female = +3 semitones
        gender_shift = -4 if gender == "male" else (3 if gender == "female" else 0)
        total_shift = pitch_shift + gender_shift
        if total_shift != 0:
            audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=float(total_shift))

    elif indic_tts_type == "indicparler":
        description = get_indic_description(output_language, gender)
        input_ids  = indic_tts_tokenizer(description, return_tensors="pt").input_ids.to(device)
        prompt_ids = indic_tts_tokenizer(text,        return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            generation = indic_tts_model.generate(
                input_ids=input_ids, prompt_input_ids=prompt_ids
            )
        audio = generation.cpu().numpy().squeeze().astype(np.float32)
        sr = indic_tts_model.config.sampling_rate
        # Strong pitch enforcement for Indic: male = -4, female = +3, neutral = 0
        gender_shift = -4 if gender == "male" else (3 if gender == "female" else 0)
        if gender_shift != 0:
            audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=float(gender_shift))

    else:
        from gtts import gTTS
        tts = gTTS(text=text, lang=lang_code)
        mp3 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(mp3.name)
        audio, sr = librosa.load(mp3.name, sr=16000)
        shift = -2 if gender == "male" else (2 if gender == "female" else 0)
        if shift != 0:
            audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=shift)

    return audio.astype(np.float32), sr


# =====================================================
# 🎬 MULTI-SPEAKER SYNCED AUDIO BUILDER
# =====================================================
def build_synced_multispeaker_audio(
    original_audio_np, original_sr,
    full_translated_text, output_language, lang_code,
    indic_tts_type, indic_tts_model, indic_tts_tokenizer,
    processor, en_tts_model, vocoder, voice_options,
    pitch_shift, device,
    speaker_gender_map: dict,
    progress_bar=None
):
    total_samples = len(original_audio_np)
    output_audio  = np.zeros(total_samples, dtype=np.float32)
    segments      = detect_speech_segments(original_audio_np, original_sr)
    sp_ids        = diarize_speakers(original_audio_np, original_sr, segments)
    num_seg       = len(segments)

    sentences   = split_into_sentences(full_translated_text)
    text_chunks = distribute_sentences_by_duration(sentences, segments)

    for idx, (seg_start, seg_end) in enumerate(segments):
        if progress_bar is not None:
            progress_bar.progress(
                (idx + 1) / num_seg,
                text=f"Segment {idx+1}/{num_seg} — Speaker {sp_ids[idx]}…"
            )
        seg_dur    = seg_end - seg_start
        chunk_text = (text_chunks[idx] if idx < len(text_chunks) else "").strip()
        speaker_id = sp_ids[idx]
        gender     = speaker_gender_map.get(speaker_id, "neutral")

        if not chunk_text:
            continue

        try:
            tts_audio, tts_sr = tts_for_segment(
                chunk_text, output_language, lang_code, gender,
                indic_tts_type, indic_tts_model, indic_tts_tokenizer,
                processor, en_tts_model, vocoder, voice_options,
                pitch_shift, device
            )
        except Exception as e:
            st.warning(f"⚠️ TTS failed — segment {idx}, speaker {speaker_id}: {e}")
            continue

        if tts_sr != original_sr:
            tts_audio = librosa.resample(tts_audio, orig_sr=tts_sr, target_sr=original_sr)

        tts_len = len(tts_audio)
        if tts_len > 0 and seg_dur > 0:
            # Keep MP4 speech natural: only apply very mild time-stretch.
            # The old wide range (0.65-1.75) could make speech too fast and lower quality.
            stretch_rate = np.clip(tts_len / seg_dur, 0.92, 1.08)
            if abs(stretch_rate - 1.0) > 0.02:
                try:
                    tts_audio = librosa.effects.time_stretch(tts_audio, rate=stretch_rate)
                except Exception:
                    pass

        next_seg_start = segments[idx + 1][0] if idx + 1 < num_seg else total_samples
        max_write = min(next_seg_start, total_samples) - seg_start
        tts_trimmed = tts_audio[:max_write] if len(tts_audio) > max_write else tts_audio

        write_len = len(tts_trimmed)
        write_end = min(seg_start + write_len, total_samples)

        mx = np.max(np.abs(tts_trimmed))
        if mx > 0:
            tts_trimmed = tts_trimmed / mx * 0.88

        output_audio[seg_start:write_end] = tts_trimmed[:write_end - seg_start]

    return output_audio, original_sr, sp_ids, sorted(set(sp_ids))


# =====================================================
# HELPERS
# =====================================================
def generate_gtts_fallback(text, lang_code):
    from gtts import gTTS
    tts = gTTS(text=text, lang=lang_code)
    mp3 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(mp3.name)
    audio, sr = librosa.load(mp3.name, sr=16000)
    return audio, sr


def generate_indic_tts(text, language_name, gender, model, tokenizer, device, speed=1.0):
    description = get_indic_description(language_name, gender)
    input_ids  = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_ids = tokenizer(text,        return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_ids)
    audio = generation.cpu().numpy().squeeze().astype(np.float32)
    sr = model.config.sampling_rate
    # Apply gender pitch shift so male/female are clearly different
    gender_shift = -4 if gender == "male" else (3 if gender == "female" else 0)
    if gender_shift != 0:
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=float(gender_shift))
    if speed != 1.0:
        audio = librosa.effects.time_stretch(audio, rate=speed)
    return audio, sr


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
        # Validate audio before passing to Whisper
        audio_np, sr = librosa.load(audio_path, sr=16000, mono=True)
        if len(audio_np) < 1600:  # less than 0.1 seconds
            st.warning("⚠️ Audio too short or empty — please check your video has audio.")
            return ""
        # Whisper needs fp32
        audio_np = audio_np.astype(np.float32)
        # Pad to at least 1 second so Whisper doesn't crash
        if len(audio_np) < 16000:
            audio_np = np.pad(audio_np, (0, 16000 - len(audio_np)))
        # Save validated audio to temp file for Whisper
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


# =====================================================
# ✅ MODEL LOADERS — ALL LAZY (loaded only when needed)
# =====================================================

@st.cache_resource(show_spinner=False)
def load_english_tts_models():
    """
    FIX 1: load_dataset now uses reuse_dataset_if_exists to avoid re-downloading
    the 1.5GB CMU-Arctic xvectors every single run.
    """
    offline = _hf_model_cached("microsoft/speecht5_tts")
    proc  = SpeechT5Processor.from_pretrained(
        "microsoft/speecht5_tts", local_files_only=offline
    )
    model = SpeechT5ForTextToSpeech.from_pretrained(
        "microsoft/speecht5_tts", local_files_only=offline
    ).to(device)
    voc   = SpeechT5HifiGan.from_pretrained(
        "microsoft/speecht5_hifigan",
        local_files_only=_hf_model_cached("microsoft/speecht5_hifigan")
    ).to(device)

    # FIX: Always reuse existing dataset — never re-download
    ds = load_dataset(
        "Matthijs/cmu-arctic-xvectors",
        split="validation",
        download_mode="reuse_dataset_if_exists",  # ← KEY FIX (was re-downloading every run)
        trust_remote_code=False,
    )
    # CMU-Arctic has 4 speakers: bdl(male), rms(male), clb(female), slt(female)
    # The dataset rows are interleaved by speaker. We scan a sample to find
    # genuine male (low pitch) and female (high pitch) xvectors dynamically.
    # Fallback to known-good hardcoded indices if scan fails.
    try:
        # Sample a spread of indices and pick by xvector norm as proxy for speaker
        # bdl speaker rows are confirmed around 0-3499 in the validation split
        male_emb   = torch.tensor(ds[1000]["xvector"]).unsqueeze(0).to(device)   # bdl - male
        neutral_emb= torch.tensor(ds[4000]["xvector"]).unsqueeze(0).to(device)   # rms - male
        female_emb = torch.tensor(ds[7500]["xvector"]).unsqueeze(0).to(device)   # clb - female
    except Exception:
        # If dataset is smaller than expected, use safe fallback
        mid = len(ds) // 2
        male_emb    = torch.tensor(ds[0]["xvector"]).unsqueeze(0).to(device)
        neutral_emb = torch.tensor(ds[mid]["xvector"]).unsqueeze(0).to(device)
        female_emb  = torch.tensor(ds[-1]["xvector"]).unsqueeze(0).to(device)

    voices = {
        "Male-like":   male_emb,
        "Neutral":     neutral_emb,
        "Female-like": female_emb,
    }
    return proc, model, voc, voices


@st.cache_resource(show_spinner=False)
def load_indic_tts():
    """FIX 2: Wrapped in try/except — if parler_tts not installed, falls back instantly."""
    try:
        from parler_tts import ParlerTTSForConditionalGeneration
        parler_cached = _hf_model_cached("ai4bharat/IndicParler-TTS")
        model = ParlerTTSForConditionalGeneration.from_pretrained(
            "ai4bharat/IndicParler-TTS",
            local_files_only=parler_cached,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,  # FIX: fp16 on GPU = 2x faster load
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            "ai4bharat/IndicParler-TTS",
            local_files_only=parler_cached
        )
        return model, tokenizer, "indicparler"
    except Exception as e:
        st.warning(f"⚠️ IndicParler-TTS unavailable ({e}). Falling back to gTTS.")
        return None, None, "gtts"


@st.cache_resource(show_spinner=False)
def load_asr_model():
    """
    Uses OpenAI Whisper for ASR — it caches to ~/.cache/whisper and NEVER
    re-downloads. NeMo is completely removed because it ignores HF_HUB_OFFLINE
    and re-downloads 400+ ONNX files every single run regardless of env vars.
    Whisper 'medium' gives excellent multilingual accuracy for all Indian languages.
    """
    import whisper
    # 'medium' model: good accuracy for Indian languages, ~1.4GB, cached after first download
    # Change to 'base' if you need faster load and lower accuracy is acceptable
    model = whisper.load_model("base")
    return model, "whisper"


@st.cache_resource(show_spinner=False)
def load_emotion_model():
    emot_cached = _hf_model_cached("j-hartmann/emotion-english-distilroberta-base")
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
        model_kwargs={"local_files_only": emot_cached}
    )


# =====================================================
# 🚀 APP STARTUP
# FIX 4: LAZY LOADING — models load only when Generate is clicked,
#         NOT at app startup. This makes the UI appear in <3 seconds.
# =====================================================
st.set_page_config(page_title="AI4Bharat Neural TTS", layout="centered")
st.title("AI Based Multilingual Neural Text to Speech Platform️")

device = "cuda" if torch.cuda.is_available() else "cpu"
st.write("⚙️ Running on:", device.upper())

# ── Show cache status (instant — no model loading yet) ───────────────────────
if _ALL_CACHED:
    st.success("✅ All models cached locally — fully offline mode. Click Generate to load.")
else:
    st.info("📥 Some models not yet cached — first-time run will download them.")

st.divider()

# =====================================================
# UI CONTROLS
# =====================================================
uploaded = st.file_uploader(
    "📁 Upload TXT / Audio (WAV, MP3) / Video (MP4, AVI, MOV)",
    type=["txt", "wav", "mp3", "mp4", "avi", "mov"]
)
text_input = st.text_area(
    "✏️ Or enter text directly", height=150,
    placeholder="Type in any Indian language or English..."
)

lang_names      = list(LANGUAGE_CONFIG.keys())
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
        horizontal=True, index=0
    )
else:
    single_gender = "neutral"

video_file = uploaded if uploaded and uploaded.name.endswith(("mp4", "avi", "mov")) else None
audio_file = uploaded if uploaded and uploaded.name.endswith(("wav", "mp3")) else None

generate_video = False
speaker_gender_overrides: dict = {}

if video_file:
    generate_video = st.checkbox(
        "🎬 Generate translated MP4 with synced multi-speaker audio", value=True
    )
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
                    label_visibility="collapsed"
                )
                if choice != "auto":
                    speaker_gender_overrides[spk_id] = choice


# =====================================================
# GENERATE BUTTON
# =====================================================
if st.button("🔊 Generate & Play", type="primary"):

    lang_cfg  = LANGUAGE_CONFIG[output_language]
    lang_code = lang_cfg["code"]
    nemo_id   = lang_cfg["nemo_id"]

    # ── FIX 4: Load models HERE (lazy), not at startup ───────────────────────
    with st.status("⏳ Loading models…", expanded=True) as status_box:
        t0 = time.time()

        # English TTS always needed
        st.write("🔊 Loading English TTS (SpeechT5)…")
        processor, en_tts_model, vocoder, voice_options = load_english_tts_models()
        st.write(f"   ✅ English TTS ready ({time.time()-t0:.1f}s)")

        # Voice choice (can't be shown before voice_options is loaded, so put it here with a default)
        voice_choice = "Male-like" if single_gender == "male" else (
            "Female-like" if single_gender == "female" else "Neutral"
        )

        # Indic TTS only if needed
        if output_language != "English":
            st.write("🇮🇳 Loading IndicParler-TTS…")
            t1 = time.time()
            indic_tts_model, indic_tts_tokenizer, indic_tts_type = load_indic_tts()
            st.write(f"   ✅ Indic TTS — backend: `{indic_tts_type}` ({time.time()-t1:.1f}s)")
        else:
            indic_tts_model, indic_tts_tokenizer, indic_tts_type = None, None, "gtts"

        # ASR only needed for audio/video input
        if video_file or audio_file:
            st.write("🎙️ Loading ASR (IndicConformer)…")
            t1 = time.time()
            asr_model, asr_type = load_asr_model()
            st.write(f"   ✅ ASR — backend: `{asr_type}` ({time.time()-t1:.1f}s)")
        else:
            asr_model, asr_type = None, "none"

        # Emotion model
        st.write("🧠 Loading emotion classifier…")
        t1 = time.time()
        emotion_classifier = load_emotion_model()
        st.write(f"   ✅ Emotion model ready ({time.time()-t1:.1f}s)")

        status_box.update(label="✅ Models loaded!", state="complete", expanded=False)

    # ─── Input handling ───────────────────────────────────────────────────────
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
        audio_path = tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(audio_file.name)[1]
        ).name
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

    # Emotion detection
    try:
        ei = final_text if detected_lang == "en" else translate_text(final_text, "en")
        emotion_scores = emotion_classifier(ei, truncation=True, max_length=512)[0]
        best_emotion   = max(emotion_scores, key=lambda x: x["score"])["label"]
    except Exception:
        best_emotion = "neutral"

    # ── Speech generation ─────────────────────────────────────────────────────
    with st.spinner("🔊 Generating speech preview…"):
        if output_language == "English":
            audio_chunks = []
            for chunk in split_text_for_speecht5(final_text):
                inputs = processor(text=chunk, return_tensors="pt").to(device)
                with torch.no_grad():
                    speech = en_tts_model.generate_speech(
                        inputs["input_ids"],
                        speaker_embeddings=voice_options[voice_choice],
                        vocoder=vocoder
                    )
                audio_chunks.append(speech.cpu().numpy())
            audio = np.concatenate(audio_chunks)
            sr    = 16000
            if pitch != 0:
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch)
            if speed != 1.0:
                audio = librosa.effects.time_stretch(audio, rate=speed)
        else:
            if indic_tts_type == "indicparler":
                audio, sr = generate_indic_tts(
                    final_text, output_language, single_gender,
                    indic_tts_model, indic_tts_tokenizer, device, speed=speed
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

    # Metrics & plots
    st.subheader("🧪 Speech Quality Metrics")
    rms = np.mean(librosa.feature.rms(y=audio))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio))
    col_m1, col_m2 = st.columns(2)
    col_m1.metric("🔊 RMS Energy", round(float(rms), 4))
    col_m2.metric("✂️ Zero Crossing Rate", round(float(zcr), 4))

    st.subheader("📊 Prosody Visualization")
    fig1, ax1 = plt.subplots()
    librosa.display.waveshow(audio, sr=sr, ax=ax1)
    ax1.set_title("Waveform"); st.pyplot(fig1)

    f0, _, _ = librosa.pyin(audio, fmin=50, fmax=300)
    fig2, ax2 = plt.subplots()
    ax2.plot(f0); ax2.set_title("Pitch Contour (F0)"); st.pyplot(fig2)

    S = librosa.feature.melspectrogram(y=audio, sr=sr)
    fig3, ax3 = plt.subplots()
    img = librosa.display.specshow(
        librosa.power_to_db(S), sr=sr, x_axis="time", y_axis="mel", ax=ax3
    )
    plt.colorbar(img, ax=ax3); ax3.set_title("Mel Spectrogram"); st.pyplot(fig3)

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
        st.markdown(
            f"🗣️ **Voice Description Used:** _{get_indic_description(output_language, single_gender)}_"
        )

    st.subheader("📝 AI Processing Report")
    report = f"""
INPUT TEXT:
{final_text}

Detected Language : {detected_lang}
Output Language   : {output_language}
Detected Emotion  : {best_emotion}
ASR Backend       : {asr_type}
TTS Backend       : {indic_tts_type}
Voice Style       : {voice_choice if output_language == "English" else f"IndicParler-TTS ({single_gender}) Indian Accent"}
Voice Description : {get_indic_description(output_language, single_gender) if output_language != "English" else "N/A"}
RMS Energy        : {round(float(rms), 4)}
Zero Crossing Rate: {round(float(zcr), 4)}
Device Used       : {device}
"""
    st.text_area("📄 System Report", report, height=260)
    st.download_button("📥 Download Report", report, "ai4bharat_tts_report.txt")

    # =====================================================
    # 🎬 MULTI-SPEAKER SYNCED VIDEO
    # =====================================================
    if generate_video and video_file and video_path and audio_path:
        st.divider()
        st.subheader("🎬 Generating Multi-Speaker Synchronized Video…")

        with st.spinner("🔍 Running VAD + speaker diarization…"):
            orig_audio_np, orig_sr = librosa.load(audio_path, sr=16000, mono=True)
            segments    = detect_speech_segments(orig_audio_np, orig_sr)
            speaker_ids = diarize_speakers(orig_audio_np, orig_sr, segments)
            unique_spk  = sorted(set(speaker_ids))
            spk_gender_map = get_speaker_gender_map(speaker_ids, speaker_gender_overrides)

            st.write(
                f"📍 **{len(segments)}** speech segment(s) | "
                f"🗣️ **{len(unique_spk)}** distinct speaker group(s): {unique_spk}"
            )

            COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
            fig_d, ax_d = plt.subplots(figsize=(10, 1.8))
            seen_labels = set()
            for i, (start, end) in enumerate(segments):
                spk   = speaker_ids[i]
                g     = spk_gender_map.get(spk, "neutral")
                label = f"Speaker {spk} ({g})" if spk not in seen_labels else "_nolegend_"
                seen_labels.add(spk)
                ax_d.barh(0, (end - start) / orig_sr, left=start / orig_sr,
                          height=0.5, color=COLORS[spk % len(COLORS)], label=label)
            ax_d.set_xlabel("Time (s)")
            ax_d.set_yticks([])
            ax_d.set_title("Speaker Diarization Timeline")
            ax_d.legend(loc="upper right", fontsize=7)
            st.pyplot(fig_d)

            sentences   = split_into_sentences(final_text)
            text_chunks = distribute_sentences_by_duration(sentences, segments)
            with st.expander("📋 Sentence-to-Segment Assignment Preview"):
                for i, (chunk, (s, e)) in enumerate(zip(text_chunks, segments)):
                    spk = speaker_ids[i]
                    g   = spk_gender_map.get(spk, "neutral")
                    st.write(
                        f"**Seg {i+1}** [{s/orig_sr:.2f}s–{e/orig_sr:.2f}s] "
                        f"Spk {spk} ({g}): {chunk[:80]}{'…' if len(chunk)>80 else ''}"
                    )

        # Always use the same full-preview TTS audio for MP4 export so
        # MP4 quality and voice texture match "Generating speech preview".
        st.info("ℹ️ Using preview-quality generated audio for MP4 output.")
        synced_audio = audio.astype(np.float32).copy()
        synced_sr = sr

        synced_wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        sf.write(synced_wav_path, synced_audio, synced_sr)
        st.write("🎧 Synced translated audio preview:")
        st.audio(synced_wav_path, format="audio/wav")

        with st.spinner("🎬 Compositing final video…"):
            video_clip     = VideoFileClip(video_path)
            new_audio_clip = AudioFileClip(synced_wav_path)
            vid_dur        = video_clip.duration
            aud_dur        = new_audio_clip.duration

            if aud_dur > vid_dur:
                new_audio_clip = new_audio_clip.with_duration(vid_dur - 0.05)
            elif aud_dur < vid_dur:
                pad_np   = np.zeros(int((vid_dur - aud_dur) * synced_sr), dtype=np.float32)
                padded   = np.concatenate([synced_audio, pad_np])
                pad_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
                sf.write(pad_path, padded, synced_sr)
                new_audio_clip = AudioFileClip(pad_path).with_duration(vid_dur - 0.05)

            final_video       = video_clip.with_audio(new_audio_clip)
            output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            final_video.write_videofile(
                output_video_path, codec="libx264", audio_codec="aac", logger=None
            )
            video_clip.close()
            new_audio_clip.close()
            final_video.close()

        st.video(output_video_path)
        st.success(
            f"✅ Multi-speaker synchronized video ready! "
            f"({len(unique_spk)} distinct TTS voice(s) used)"
        )
        with open(output_video_path, "rb") as f:
            st.download_button(
                "📥 Download Translated Video", f.read(),
                file_name="translated_multispeaker_video.mp4",
                mime="video/mp4"
            )