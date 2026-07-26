import os

import streamlit as st
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor, pipeline

from .config import HF_ROOT
from .utils import _hf_model_cached


def load_english_tts_models():
    offline = _hf_model_cached("microsoft/speecht5_tts")
    proc = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts", local_files_only=offline)
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts", local_files_only=offline).to(device)
    voc = SpeechT5HifiGan.from_pretrained(
        "microsoft/speecht5_hifigan",
        local_files_only=_hf_model_cached("microsoft/speecht5_hifigan"),
    ).to(device)

    ds = load_dataset(
        "Matthijs/cmu-arctic-xvectors",
        split="validation",
        download_mode="reuse_dataset_if_exists",
        trust_remote_code=False,
    )
    try:
        male_emb = torch.tensor(ds[1000]["xvector"]).unsqueeze(0).to(device)
        neutral_emb = torch.tensor(ds[4000]["xvector"]).unsqueeze(0).to(device)
        female_emb = torch.tensor(ds[7500]["xvector"]).unsqueeze(0).to(device)
    except Exception:
        mid = len(ds) // 2
        male_emb = torch.tensor(ds[0]["xvector"]).unsqueeze(0).to(device)
        neutral_emb = torch.tensor(ds[mid]["xvector"]).unsqueeze(0).to(device)
        female_emb = torch.tensor(ds[-1]["xvector"]).unsqueeze(0).to(device)

    voices = {
        "Male-like": male_emb,
        "Neutral": neutral_emb,
        "Female-like": female_emb,
    }
    return proc, model, voc, voices


@st.cache_resource(show_spinner=False)
def load_indic_tts():
    try:
        from parler_tts import ParlerTTSForConditionalGeneration

        parler_cached = _hf_model_cached("ai4bharat/IndicParler-TTS")
        model = ParlerTTSForConditionalGeneration.from_pretrained(
            "ai4bharat/IndicParler-TTS",
            local_files_only=parler_cached,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            "ai4bharat/IndicParler-TTS",
            local_files_only=parler_cached,
        )
        return model, tokenizer, "indicparler"
    except Exception as e:
        st.warning(f"⚠️ IndicParler-TTS unavailable ({e}). Falling back to gTTS.")
        return None, None, "gtts"


@st.cache_resource(show_spinner=False)
def load_asr_model():
    import whisper

    model = whisper.load_model("base")
    return model, "whisper"


@st.cache_resource(show_spinner=False)
def load_emotion_model():
    emot_cached = _hf_model_cached("j-hartmann/emotion-english-distilroberta-base")
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
        model_kwargs={"local_files_only": emot_cached},
    )


device = "cuda" if torch.cuda.is_available() else "cpu"
