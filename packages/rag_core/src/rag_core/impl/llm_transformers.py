from __future__ import annotations
from collections.abc import Iterator, Sequence
from threading import Lock, Thread

from rag_core.contracts.llm import LLMProvider
from rag_core.types import ChatTurn


class TransformersLLMProvider(LLMProvider):
    def __init__(
        self,
        model_name: str,
        demo_mode: bool = False,
        generation_mode: str = 'sample',
        contrastive_penalty_alpha: float = 0.6,
        contrastive_top_k: int = 4,
    ) -> None:
        self.model_name = model_name
        self.demo_mode = demo_mode
        self.generation_mode = generation_mode.lower().strip()
        if self.generation_mode not in {'sample', 'contrastive', 'greedy'}:
            self.generation_mode = 'sample'
        self.contrastive_penalty_alpha = max(0.0, float(contrastive_penalty_alpha))
        self.contrastive_top_k = max(1, int(contrastive_top_k))
        self._tokenizer = None
        self._model = None
        self._generation_lock = Lock()

    def _ensure_loaded(self) -> None:
        if self.demo_mode or self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        has_accelerator = torch.cuda.is_available() or (getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            device_map='auto' if has_accelerator else 'cpu',
            torch_dtype='auto' if has_accelerator else None,
            low_cpu_mem_usage=True,
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

        # Local transformers generation is not thread-safe on a shared model instance;
        # serialize requests to avoid interleaved streamer output and device contention.
        with self._generation_lock:
            streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
            model_inputs = self._build_prompt(messages, system_prompt)
            model_inputs = {k: v.to(self._model.device) for k, v in model_inputs.items()}

            generate_kwargs = {
                **model_inputs,
                'streamer': streamer,
                'max_new_tokens': max_new_tokens,
            }
            if self.generation_mode == 'contrastive':
                generate_kwargs.update(
                    {
                        'do_sample': False,
                        'penalty_alpha': self.contrastive_penalty_alpha,
                        'top_k': self.contrastive_top_k,
                        'custom_generate': 'transformers-community/contrastive-search',
                        'trust_remote_code': True,
                    }
                )
            elif self.generation_mode == 'greedy':
                generate_kwargs.update({'do_sample': False})
            else:
                do_sample = temperature > 0
                generate_kwargs.update({'do_sample': do_sample})
                if do_sample:
                    generate_kwargs.update({'temperature': max(temperature, 0.01)})

            generation_error: list[BaseException] = []

            def _run_generate() -> None:
                try:
                    self._model.generate(**generate_kwargs)
                except BaseException as exc:  # pragma: no cover - defensive runtime safety
                    generation_error.append(exc)
                finally:
                    # Ensure the iterator terminates even when generation crashes.
                    streamer.on_finalized_text("", stream_end=True)

            thread = Thread(target=_run_generate, daemon=True)
            thread.start()
            for text in streamer:
                yield text
            thread.join()

            if generation_error:
                raise RuntimeError("Transformers generation failed") from generation_error[0]
