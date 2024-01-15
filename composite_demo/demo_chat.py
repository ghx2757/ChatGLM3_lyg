import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from client import get_client
from conversation import postprocess_text, preprocess_text, Conversation, Role

import time
import json
import os
MAX_LENGTH = 8192

client = get_client()

# 计算推理时间，并输出到json文件中
def count_time(start, out_path, prompt_text, output_text):
    end = time.time()
    running_time = end-start
    msg = f'prompt_text字符长度：{len(prompt_text)};    output_text字符长度：{len(output_text)}    推理时长：{running_time}'
    with open(out_path, 'a', encoding='utf-8') as f:
        json.dump(msg, f, indent=4, ensure_ascii=False)
        f.write('\n')
    print(msg)
    
# 添加当前会话到历史记录中(这里面是包含标识符的)
def append_conversation(
        conversation: Conversation,
        history: list[Conversation],
        placeholder: DeltaGenerator | None = None,
) -> None:
    history.append(conversation)
    conversation.show(placeholder)

# 主函数
def main(
        prompt_text: str,
        system_prompt: str,
        top_p: float = 0.8,
        temperature: float = 0.95,
        repetition_penalty: float = 1.0,
        max_new_tokens: int = 1024,
        retry: bool = False
):
    # 1、初始化 session_state的缓存
    placeholder = st.empty()
    with placeholder.container():
        # streamlit.session_state存储的是原始的数据，而不是显示的内容，即存储的history是包含标识符的
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

    # 2、是否要清除缓存
    if prompt_text == "" and retry == False:
        print("\n== Clean History ==\n")
        st.session_state.chat_history = []
        return
    
    # 3、将缓存数据存储到history列表，并逐一展示
    history: list[Conversation] = st.session_state.chat_history
    for conversation in history:
        conversation.show()
    # 4、判断 retry 
    if retry:
        print("\n== Retry ==\n")
        last_user_conversation_idx = None
        for idx, conversation in enumerate(history):
            if conversation.role == Role.USER:
                last_user_conversation_idx = idx
        if last_user_conversation_idx is not None:
            prompt_text = history[last_user_conversation_idx].content
            del history[last_user_conversation_idx:]

    # 5、获取用户问题并推理
    if prompt_text:
        prompt_text = prompt_text.strip()
        append_conversation(Conversation(Role.USER, prompt_text), history)
        placeholder = st.empty()
        message_placeholder = placeholder.chat_message(name="assistant", avatar="assistant") # 用于显示输出内容
        markdown_placeholder = message_placeholder.empty() 

        output_text = ''
        start = time.time()
        for response in client.generate_stream(
                system_prompt,
                tools=None,
                history=history,
                do_sample=True,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_sequences=[str(Role.USER)],
                repetition_penalty=repetition_penalty,
        ):
            token = response.token
            if response.token.special: # 如果遇到了终止标识符，则打印最终回复，并判断该标识符是否为正常的'<|user|>'
                print("\n==Output:==\n", output_text)
                match token.text.strip():
                    case '<|user|>':
                        break
                    case _:
                        st.error(f'Unexpected special token: {token.text.strip()}')
                        break
            output_text += response.token.text
            markdown_placeholder.markdown(postprocess_text(output_text + '▌')) # 显示在前端页面，效果是流式的
        
        # 计算推理时间
        # out_path = os.path.join(os.getcwd(), 'timeCount.txt')   
        # count_time(start, out_path, prompt_text, output_text)

        # 显示完并将大模型的回复加到历史记录中
        append_conversation(Conversation(
            Role.ASSISTANT,
            postprocess_text(output_text),
        ), history, markdown_placeholder)