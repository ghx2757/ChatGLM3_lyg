from __future__ import annotations

import os
import streamlit as st
import torch

from collections.abc import Iterable
from typing import Any, Protocol
from huggingface_hub.inference._text_generation import TextGenerationStreamResponse, Token
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList

from conversation import Conversation

TOOL_PROMPT = 'Answer the following questions as best as you can. You have access to the following tools:'

MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/chatglm3-6b')
print(f'MODEL_PATH==>{MODEL_PATH}')

# PT_PATH = os.environ.get('PT_PATH', None)
PT_PATH = '/home/lyg/code/ChatGLM3_lyg/finetune_chatmodel_demo/output/data_hsw_m-20231212-122142-128-2e-2'
PRE_SEQ_LEN = int(os.environ.get("PRE_SEQ_LEN", 128))
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)


@st.cache_resource
def get_client() -> Client:
    client = HFClient(MODEL_PATH, TOKENIZER_PATH, PT_PATH)
    return client


class Client(Protocol):
    def generate_stream(self,
                        system: str | None,
                        tools: list[dict] | None,
                        history: list[Conversation],
                        **parameters: Any
                        ) -> Iterable[TextGenerationStreamResponse]:
        ...

# 最后的推理部分代码，涉及到底层模型输入部分需要深入理解
def stream_chat(
        self, tokenizer, query: str, # 这里的self传入的是modle，我推测
        history: list[tuple[str, str]] = None,
        role: str = "user",
        past_key_values=None,
        max_new_tokens: int = 256,
        do_sample=True, top_p=0.8,
        temperature=0.8,
        repetition_penalty=1.0,
        length_penalty=1.0, num_beams=1,
        logits_processor=None,
        return_past_key_values=False,
        **kwargs
):
    # 这段代码的目的是在检测到scores中存在NaN或无穷大的值时，将所有的scores设置为0，并将第五个元素设置为一个非常大的值。这样做的原因可能是为了在出现这种异常情况时，强制模型选择第五个词汇作为输出，或者防止模型产生无效的输出。
    class InvalidScoreLogitsProcessor(LogitsProcessor):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
            if torch.isnan(scores).any() or torch.isinf(scores).any(): # 有一个score为0或者i非法都不可以
                scores.zero_()
                scores[..., 5] = 5e4
            return scores
    if history is None:
        history = []

    print("\n== Input ==\n", query)
    print("\n==History==\n", history)

    if logits_processor is None:
        logits_processor = LogitsProcessorList()
    logits_processor.append(InvalidScoreLogitsProcessor())
    
    # 三个控制生成的结束token对应ID
    eos_token_id = [tokenizer.eos_token_id, tokenizer.get_command("<|user|>"),
                    tokenizer.get_command("<|observation|>")]
    for eos in eos_token_id:
        print(f"\033[32meos_token:{tokenizer.decode(eos)}\033[0m")
    
    # 超参
    gen_kwargs = {"max_new_tokens": max_new_tokens,
                  "do_sample": do_sample,
                  "top_p": top_p,
                  "temperature": temperature,
                  "logits_processor": logits_processor,
                  "repetition_penalty": repetition_penalty,
                  "length_penalty": length_penalty,
                  "num_beams": num_beams,
                  **kwargs
                  }
    # 过去 的 K、V 这块需要理解
    if past_key_values is None:
        inputs = tokenizer.build_chat_input(query, history=history, role=role)
    else:
        inputs = tokenizer.build_chat_input(query, role=role)

    print(f"\033[32m模型最终输入：\n{inputs}\033[0m")

    inputs = inputs.to(self.device) # 将输入数据放到gpu或者cpu上
    # 根据past_key_values来整理一下attention_mask
    if past_key_values is not None:
        past_length = past_key_values[0][0].shape[0] # 获取过去内容的长度
        if self.transformer.pre_seq_len is not None:
            past_length -= self.transformer.pre_seq_len # 是否有pre_seq_len限制，有的话进行调整
        inputs.position_ids += past_length # 当前问题其实位置
        attention_mask = inputs.attention_mask
        attention_mask = torch.cat((attention_mask.new_ones(1, past_length), attention_mask), dim=1) # 将past_key_values对应的位置设置mask不需要再计算了，因为有保存
        inputs['attention_mask'] = attention_mask
    history.append({"role": role, "content": query})
    input_sequence_length = inputs['input_ids'].shape[1]
    # 这里的加号的意思是seq_length规定的是输入输出的总长
    if input_sequence_length + max_new_tokens >= self.config.seq_length:
        yield "Current input sequence length {} plus max_new_tokens {} is too long. The maximum model sequence length is {}. You may adjust the generation parameter to enable longer chat history.".format(
            input_sequence_length, max_new_tokens, self.config.seq_length
        ), history
        return

    if input_sequence_length > self.config.seq_length:
        yield "Current input sequence length {} exceeds maximum model sequence length {}. Unable to generate tokens.".format(
            input_sequence_length, self.config.seq_length
        ), history
        return
    # 推理
    for outputs in self.stream_generate(**inputs, past_key_values=past_key_values,
                                        eos_token_id=eos_token_id, return_past_key_values=return_past_key_values,
                                        **gen_kwargs):
        if return_past_key_values:
            outputs, past_key_values = outputs
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):] # 这行代码结合模型的最终输入那块看，好理解
        response = tokenizer.decode(outputs) # 解码，就是从词表中读取
        if response and response[-1] != "�":
            new_history = history
            if return_past_key_values:
                yield response, new_history, past_key_values
            else:
                yield response, new_history # 流式返回结果和新的history，我看到后续的代码没需要这个history


class HFClient(Client):
    def __init__(self, model_path: str, tokenizer_path: str, pt_checkpoint: str = None):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        
        if pt_checkpoint is not None and os.path.exists(pt_checkpoint):
            print(f'\n❀ ❀ ❀ pt_checkpoint==>{pt_checkpoint}')
            config = AutoConfig.from_pretrained(
                model_path,
                trust_remote_code=True,
                pre_seq_len=PRE_SEQ_LEN
            )
            self.model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                config=config,
                device_map="auto"
            ).eval()
            prefix_state_dict = torch.load(os.path.join(pt_checkpoint, "pytorch_model.bin"))
            new_prefix_state_dict = {}
            for k, v in prefix_state_dict.items():
                if k.startswith("transformer.prefix_encoder."):
                    new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
            print("Loaded from pt checkpoints", new_prefix_state_dict.keys())
            self.model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
        else:
            print(f'\npt_checkpoint==> None')
            self.model = (
                AutoModel.from_pretrained(
                    MODEL_PATH,
                    trust_remote_code=True,
                    device_map="auto"
                ).eval())

    def generate_stream(
            self,
            system: str | None,
            tools: list[dict] | None,
            history: list[Conversation],
            **parameters: Any
    ) -> Iterable[TextGenerationStreamResponse]:
        # 1、 从新维护一个历史记录
        chat_history = [{
            'role': 'system',
            'content': system if not tools else TOOL_PROMPT,
        }]

        if tools:
            chat_history[0]['tools'] = tools

        for conversation in history[:-1]:
            chat_history.append({
                'role': str(conversation.role).removeprefix('<|').removesuffix('|>'),
                'content': conversation.content,
            })
        # 2、获取当前用户问题 和 标签
        query = history[-1].content
        role = str(history[-1].role).removeprefix('<|').removesuffix('|>')
        # 3、推理
        text = ''
        for new_text, _ in stream_chat(
                self.model,
                self.tokenizer,
                query,
                chat_history,
                role, # 角色
                **parameters,
        ):
            word = new_text.removeprefix(text)
            word_stripped = word.strip()
            text = new_text
            yield TextGenerationStreamResponse(
                generated_text=text,
                token=Token(
                    id=0,
                    logprob=0,
                    text=word,
                    special=word_stripped.startswith('<|') and word_stripped.endswith('|>'),
                )
            )
