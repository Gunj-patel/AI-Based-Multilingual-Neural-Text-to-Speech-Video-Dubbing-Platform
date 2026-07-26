import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PATH"] = r"C:\ffmpeg\ffmpeg-7.0-essentials_build\bin;" + os.environ.get("PATH", "")

HF_ROOT = os.path.expanduser(r"~\.cache\huggingface\hub")
LOCAL_NEMO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "indic_conformer_600m.nemo")

LANGUAGE_CONFIG = {
    "English": {"code": "en", "nemo_id": "en"},
    "Hindi": {"code": "hi", "nemo_id": "hi"},
    "Gujarati": {"code": "gu", "nemo_id": "gu"},
    "Bengali": {"code": "bn", "nemo_id": "bn"},
    "Tamil": {"code": "ta", "nemo_id": "ta"},
    "Telugu": {"code": "te", "nemo_id": "te"},
    "Kannada": {"code": "kn", "nemo_id": "kn"},
    "Malayalam": {"code": "ml", "nemo_id": "ml"},
    "Marathi": {"code": "mr", "nemo_id": "mr"},
    "Punjabi": {"code": "pa", "nemo_id": "pa"},
    "Odia": {"code": "or", "nemo_id": "or"},
    "Assamese": {"code": "as", "nemo_id": "as"},
    "Maithili": {"code": "mai", "nemo_id": "mai"},
    "Santali": {"code": "sat", "nemo_id": "sat"},
    "Konkani": {"code": "gom", "nemo_id": "kok"},
    "Sindhi": {"code": "sd", "nemo_id": "sd"},
    "Dogri": {"code": "doi", "nemo_id": "doi"},
    "Kashmiri": {"code": "ks", "nemo_id": "ks"},
    "Manipuri": {"code": "mni", "nemo_id": "mni"},
    "Bodo": {"code": "brx", "nemo_id": "brx"},
    "Sanskrit": {"code": "sa", "nemo_id": "sa"},
    "Urdu": {"code": "ur", "nemo_id": "ur"},
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
