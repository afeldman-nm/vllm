from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
from vllm import LLM
import torch
"""
A simple script to run vLLM inference using the Whisper model.
The goal is to prototype and debug
"""
hf_model_id = "openai/whisper-tiny"

## Load model and processor
processor = WhisperProcessor.from_pretrained(hf_model_id)
model = WhisperForConditionalGeneration.from_pretrained(
    hf_model_id, attn_implementation="eager")
model.config.forced_decoder_ids = None

## Load dummy dataset and read audio files
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy",
                  "clean",
                  split="validation")
sample = ds[0]["audio"]
input_features = processor(sample["array"],
                           sampling_rate=sample["sampling_rate"],
                           return_tensors="pt").input_features
print(input_features.shape)
# generate token ids
predicted_ids = model.generate(input_features)
