"""Text generation: vLLM when available, plain transformers otherwise."""
from __future__ import annotations


class Generator:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        g = cfg["gen"]
        self.backend = g["backend"]
        if self.backend == "vllm":
            from vllm import LLM
            self.llm = LLM(model=cfg["model"], dtype="bfloat16", seed=cfg["seed"],
                           max_model_len=g["max_model_len"], gpu_memory_utilization=g["gpu_mem"])
            self.tokenizer = self.llm.get_tokenizer()
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg["model"], torch_dtype=torch.bfloat16, device_map="cuda").eval()

    def generate(self, prompts: list[str], n: int, temperature: float, max_tokens: int,
                 seed: int = 0) -> list[list[str]]:
        """Returns one list of `n` completions per prompt. temperature 0 means greedy."""
        if self.backend == "vllm":
            from vllm import SamplingParams
            sp = SamplingParams(n=n, temperature=temperature,
                                top_p=self.cfg["gen"]["top_p"] if temperature > 0 else 1.0,
                                max_tokens=max_tokens, seed=seed)
            outs = self.llm.generate(prompts, sp)
            return [[o.text for o in r.outputs] for r in outs]
        return self._hf_generate(prompts, n, temperature, max_tokens, seed)

    def _hf_generate(self, prompts, n, temperature, max_tokens, seed, bs=8):
        import torch
        torch.manual_seed(seed)
        out = []
        for i in range(0, len(prompts), bs):
            batch = prompts[i:i + bs]
            enc = self.tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            kw = dict(max_new_tokens=max_tokens, num_return_sequences=n,
                      pad_token_id=self.tokenizer.pad_token_id)
            if temperature > 0:
                kw.update(do_sample=True, temperature=temperature, top_p=self.cfg["gen"]["top_p"])
            else:
                kw.update(do_sample=False)
            with torch.no_grad():
                ids = self.model.generate(**enc, **kw)
            ids = ids[:, enc["input_ids"].shape[1]:]
            texts = self.tokenizer.batch_decode(ids, skip_special_tokens=True)
            out.extend(texts[j * n:(j + 1) * n] for j in range(len(batch)))
        return out
