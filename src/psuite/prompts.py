"""Prompts.  Two prompt types per problem:
  plain   the problem + the usual step-by-step instruction
  con     the SAME text + ONE deterministic output constraint (see verify.py), assigned per problem by a seeded hash (a manipulated variable)
CUE is appended after an arbitrary prefix of the model's own solution to FORCE its answer (readiness / settledness labels)."""
from .verify import CONSTRAINT_TEXT, constraint_for

INSTR = {"int": "Please reason step by step, and put your final answer within \\boxed{}.",
         "mcq": "Please reason step by step, and put the letter of your final answer within \\boxed{}."}
CUE = "\n\nTherefore, the final answer is \\(\\boxed{"


def plain_text(p: dict) -> str:
    return f"{p['question']}\n\n{INSTR[p['kind']]}"


def con_text(p: dict) -> str:
    return f"{plain_text(p)}\n\n{CONSTRAINT_TEXT[constraint_for(p['id'])]}"


def chat_prompt(tok, text: str) -> str:
    if hasattr(tok, "apply_chat_template"):
        return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
