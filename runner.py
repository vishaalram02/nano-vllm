"""
Run nano-vllm example in Modal
"""

import modal

app = modal.App("nano-vllm-example")
volumes = {"/cache": modal.Volume.from_name("nano-vllm-models", create_if_missing=True)}

MODEL_ID = "Qwen/Qwen3-0.6B"

def download_model():
    from huggingface_hub import snapshot_download
    import os
    # Download model to a subdirectory with the model name
    model_dir = os.path.join("/cache/models", MODEL_ID.replace("/", "_"))
    snapshot_download(MODEL_ID, local_dir=model_dir)

cuda_version = "12.8.1"  # should be no greater than host CUDA version
flavor = "devel"  # includes full CUDA toolkit
operating_sys = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .uv_sync()
    .run_function(download_model, volumes=volumes)
    .add_local_dir("./nanovllm", remote_path="/root/nanovllm")
)

@app.function(
    image=image,
    gpu="A100",
    volumes=volumes,
    timeout=600,
)
def run_inference():
    import sys
    import os
    
    sys.path.insert(0, "/root")
    
    from nanovllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    
    
    model_path = os.path.join("/cache/models", MODEL_ID.replace("/", "_"))
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)
    
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
        "why is the sky blue?",
        "what is the capital of France?",
        "how do you bake a cake?",
    ]
    
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    
    outputs = llm.generate(prompts, sampling_params)
    
    results = []
    for prompt, output in zip(prompts, outputs):
        result = {
            "prompt": prompt,
            "completion": output['text']
        }
        results.append(result)
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")
    
    return results


@app.local_entrypoint()
def main():
    """
    Local entrypoint to run the Modal function
    """
    print("Starting nano-vllm inference on Modal...")
    print("=" * 50)
    
    results = run_inference.remote()
    
    print("\n" + "=" * 50)
    print("Inference completed successfully!")
    print("=" * 50)
    
    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(f"Prompt: {result['prompt']}")
        print(f"Completion: {result['completion']}")


if __name__ == "__main__":
    main()
