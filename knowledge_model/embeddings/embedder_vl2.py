"""
embedder_vl2.py
Provides functionality to load DeepSeek-VL2 from Hugging Face or local,
and generate embeddings from text (and optionally images).
"""

import torch
import numpy as np

from deepseek_vl2.serve.inference import load_model
# ^ This is from 'inference.py' (which loads the processor, tokenizer, model)

class DeepSeekVL2Embedder:
    def __init__(
        self,
        model_path: str = "deepseek-ai/deepseek-vl2-tiny",
        dtype=torch.bfloat16,
        device: str = None,
    ):
        """
        :param model_path: Hugging Face repo or local directory
        :param dtype: PyTorch dtype, e.g. torch.bfloat16 or torch.float16
        :param device: 'cuda' or 'cpu'. If None, auto-detects 'cuda' if available.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        # load_model is from inference.py. It returns (tokenizer, model, processor)
        self.tokenizer, self.model, self.processor = load_model(model_path, dtype=dtype)
        self.model.eval().to(self.device)

    @torch.no_grad()
    def embed_text(self, text: str, max_new_tokens: int = 0, pooling: str = "mean") -> np.ndarray:
        """
        Generate an embedding for text alone, ignoring images.
        :param text: input text
        :param max_new_tokens: 0 => no generation, just hidden states (not used in forward pass)
        :param pooling: 'mean' or 'last_token'
        :return: 1D NumPy array (embedding)
        """
        # Create a minimal conversation
        conversation = [
            {"role": "<|User|>", "content": text, "images": []},
            {"role": "<|Assistant|>", "content": ""}
        ]

        # Prepare inputs (tokenization + some internal logic from the processor)
        inputs = self.processor(
            conversations=conversation,
            images=[],
            force_batchify=True,
            inference_mode=True,
            system_prompt=""
        ).to(self.device, dtype=self.model.dtype)

        # Directly call self.model(...) without 'max_new_tokens'
        outputs = self.model(
            inputs_embeds=self.model.prepare_inputs_embeds(**inputs),
            attention_mask=inputs.attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        # outputs.hidden_states[-1]: shape [batch_size, seq_len, hidden_dim]
        final_hidden = outputs.hidden_states[-1]
        hidden_seq = final_hidden[0]  # first batch

        if pooling == "mean":
            embedding = hidden_seq.mean(dim=0)
        elif pooling == "last_token":
            embedding = hidden_seq[-1, :]
        else:
            embedding = hidden_seq.mean(dim=0)

        return embedding.cpu().float().numpy()

    @torch.no_grad()
    def embed_multimodal(
        self,
        text: str,
        image_paths: list[str],
        max_new_tokens: int = 0,
        pooling: str = "mean"
    ) -> np.ndarray:
        """
        Embed text + images together. The model merges text and image tokens in the final hidden states.
        :param text: user text
        :param image_paths: local file paths to images
        :param max_new_tokens: 0 => no generation, just hidden states (not used in forward pass)
        :param pooling: 'mean' or 'last_token'
        :return: 1D NumPy array
        """
        from PIL import Image

        pil_images = [Image.open(p).convert("RGB") for p in image_paths]

        conversation = [
            {"role": "<|User|>", "content": text, "images": pil_images},
            {"role": "<|Assistant|>", "content": ""}
        ]

        inputs = self.processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            inference_mode=True,
            system_prompt=""
        ).to(self.device, dtype=self.model.dtype)

        # Same fix: omit 'max_new_tokens' in self.model(...)
        outputs = self.model(
            inputs_embeds=self.model.prepare_inputs_embeds(**inputs),
            attention_mask=inputs.attention_mask,
            output_hidden_states=True,
            use_cache=False
        )

        final_hidden = outputs.hidden_states[-1]  # shape [batch_size, seq_len, hidden_dim]
        hidden_seq = final_hidden[0]

        if pooling == "mean":
            embedding = hidden_seq.mean(dim=0)
        elif pooling == "last_token":
            embedding = hidden_seq[-1, :]
        else:
            embedding = hidden_seq.mean(dim=0)

        return embedding.cpu().float().numpy()

def main():
    embedder = DeepSeekVL2Embedder()
    emb = embedder.embed_text("Hello world!")
    print("Embedding shape:", emb.shape)

if __name__ == "__main__":
    main()
