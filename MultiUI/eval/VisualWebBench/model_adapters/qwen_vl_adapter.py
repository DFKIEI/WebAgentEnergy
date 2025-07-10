import re

from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_adapters import BaseAdapter
from utils.constants import *

class QwenVLAdapter(BaseAdapter):
    def __init__(
        self, 
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
    ):
        super().__init__(model, tokenizer)

    def generate(
        self,
        query: str,
        image: Image,
        task_type: str,
        max_new_tokens: int = 512
    ) -> str:
        # If the model supports images, convert image properly
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        else:
            image = image.convert('RGB')

        # If model requires formatted multimodal input, replace this with appropriate tokens.
        # Since UIX-Qwen2-Mind2Web is based on Qwen2, use prompt format manually:
        prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Generate output
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id
        )

        response = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Post-processing based on task type
        if task_type == CAPTION_TASK:
            pattern = re.compile(r"<meta name=\"description\" content=\"(.*)\">")
            cur_meta = re.findall(pattern, response)
            return cur_meta[0] if cur_meta else response

        elif task_type == ACTION_PREDICTION_TASK:
            return response.strip()[0].upper()

        elif task_type in [WEBQA_TASK, ELEMENT_OCR_TASK]:
            if ":" in response:
                response = ":".join(response.split(":", 1)[1:])
            return response.strip().strip('"').strip("'")

        else:
            return response.strip()
