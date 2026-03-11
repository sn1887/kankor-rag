from __future__ import annotations
from collections.abc import Iterator, Sequence
from threading import Thread

from rag_core.contracts.llm import LLMProvider
from rag_core.types import ChatTurn


class TransformersLLMProvider(LLMProvider):
    def __init__(self, model_name: str, demo_mode: bool = False) -> None:
        self.model_name = model_name
        self.demo_mode = demo_mode
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self.demo_mode or self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            device_map='cpu',
        )

    def _build_prompt(self, messages: Sequence[ChatTurn], system_prompt: str):
        payload = [{'role': 'system', 'content': system_prompt}] + [
            {'role': item.role, 'content': item.content} for item in messages
        ]

        if hasattr(self._tokenizer, 'apply_chat_template'):
            text = self._tokenizer.apply_chat_template(
                payload,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = '\n'.join(f"{item['role']}: {item['content']}" for item in payload)

        model_inputs = self._tokenizer(text, return_tensors='pt')
        return model_inputs


    def _demo_stream(self, messages: Sequence[ChatTurn]) -> Iterator[str]:
        user_message = next((item.content for item in reversed(messages) if item.role == 'user'), '')
        response = (
            'Demo mode is enabled, so this answer is generated without a local transformer model. '
            'The retrieval pipeline still ran normally. '
            f'Use the cited sources to verify the explanation, and replace demo mode with {self.model_name} for full local generation. '
            f'Your latest question was: {user_message}'
        )
        for token in response.split(' '):
            yield token + ' '

    def stream_chat(
        self,
        *,
        messages: Sequence[ChatTurn],
        system_prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> Iterator[str]:
        if self.demo_mode:
            yield from self._demo_stream(messages)
            return

        self._ensure_loaded()
        assert self._tokenizer is not None and self._model is not None

        from transformers import TextIteratorStreamer

        input_ids = self._build_prompt(messages, system_prompt)
        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
        model_inputs = self._build_prompt(messages, system_prompt)
        model_inputs = {k: v.to(self._model.device) for k, v in model_inputs.items()}

        generate_kwargs = {
            **model_inputs,
            'streamer': streamer,
            'max_new_tokens': max_new_tokens,
            'do_sample': temperature > 0,
            'temperature': max(temperature, 0.01),
        }

        thread = Thread(target=self._model.generate, kwargs=generate_kwargs, daemon=True)
        thread.start()
        for text in streamer:
            yield text
