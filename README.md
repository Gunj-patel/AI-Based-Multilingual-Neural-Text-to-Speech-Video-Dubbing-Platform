# AI4Bharat Multilingual TTS

This project has been reorganized into a small package structure for easier GitHub hosting and maintenance.

## Structure

- app/config.py: language and voice configuration
- app/utils.py: text normalization, audio processing, translation helpers
- app/models.py: model loading and caching helpers
- app/app.py: Streamlit UI entry point
- streamlit_tts2.py: compatibility wrapper that runs the app

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_tts2.py
```
