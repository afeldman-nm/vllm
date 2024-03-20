import pytest
import torch
from datasets import load_dataset
from vllm.config import AudioFeaturesConfig

@pytest.fixture()
def model_id():
    return "openai/whisper-tiny"

@pytest.fixture()
def audio_features_config():
    return AudioFeaturesConfig()

def sample_from_librispeech():
    dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    return dataset[0]

audio_sample = sample_from_librispeech()["audio"]









@pytest.mark.parametrize("worker_use_ray", [True])
@pytest.mark.parametrize("dtype", ["float"]) # TODO fix that
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("prompts, audio_samples", [([""], [audio_sample])])
def test_text_to_audio_scenario(hf_runner, vllm_runner, model_id, prompts, audio_samples, dtype: str, max_tokens: int, worker_use_ray: bool) -> None:
    hf_model = hf_runner(model_id, dtype=dtype)
    hf_outputs = hf_model.generate_greedy(prompts=prompts,
                                          audio_samples=audio_samples,
                                          max_tokens=max_tokens)
    del hf_model

    # Truly cleans up GPU memory.
    torch.cuda.empty_cache()

    vllm_model = vllm_runner(model_id,
                             dtype=dtype,
                             worker_use_ray=worker_use_ray,
                             #**as_dict(vision_language_config)
                             )
    vllm_outputs = vllm_model.generate_greedy(vllm_image_prompts,
                                              max_tokens,
                                              images=vllm_images)
    del vllm_model
    # Truly cleans up GPU memory.
    torch.cuda.empty_cache()

    for i in range(len(hf_image_prompts)):
        hf_output_ids, hf_output_str = hf_outputs[i]
        vllm_output_ids, vllm_output_str = sanitize_vllm_output(
            vllm_outputs[i], vision_language_config, model_id)
        assert hf_output_str == vllm_output_str, (
            f"Test{i}:\nHF: {hf_output_str!r}\nvLLM: {vllm_output_str!r}")
        assert hf_output_ids == vllm_output_ids, (
            f"Test{i}:\nHF: {hf_output_ids}\nvLLM: {vllm_output_ids}")