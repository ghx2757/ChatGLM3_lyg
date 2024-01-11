import streamlit as st
import os

# 设置页面配置：
st.set_page_config(
    page_title="元客世界-Chat",# 页面主题
    page_icon=":elephant:",# robot:（一个机器人图标）
    layout='centered',# centered（居中布局）
    initial_sidebar_state='expanded',# 初始侧边栏状态：expanded（展开状态）
)


import demo_chat, demo_ci, demo_tool
from enum import Enum

# DEFAULT_SYSTEM_PROMPT = '''
# You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown.
# '''.strip()
DEFAULT_SYSTEM_PROMPT = '''

'''.strip()

# st.title("✨元客视界🐱‍🏍" )

# st.markdown(
#     "<sub>智谱AI 公开在线技术文档: https://www.luster3ds.com/about/ </sub> \n\n <sub> 更多 ChatGLM3-6B 的使用方法请参考文档。</sub>",
#     unsafe_allow_html=True)


class Mode(str, Enum):
    CHAT, TOOL, CI = '💬 Chat', '🛠️ Tool', '🧑‍💻 Code Interpreter'

system_prompt = st.text_area(
        label="System Prompt (Only for chat mode)",
        height=125,
        value=DEFAULT_SYSTEM_PROMPT,
    )
st.toast(
            f"哔哔哔~欢迎使用 [`元客视界 WebUI`](https://www.luster3ds.com/about/) ! \n\n"
            f"当前运行的模型`ChatGLM3-6b`, 您可以开始提问了."
        )
with st.sidebar: # 设置左边栏    
    st.title("✨元客视界:blue[ Chat]")    
    st.image(
            os.path.join(
                "img",
                "滚雪卡通动图.gif"
            ),
            use_column_width=True
        )
    top_p = st.slider(
        'top_p', 0.0, 1.0, 0.8, step=0.01
    )
    temperature = st.slider(
        'temperature', 0.0, 1.5, 0.95, step=0.01
    )
    repetition_penalty = st.slider(
        'repetition_penalty', 0.0, 2.0, 1.1, step=0.01
    )
    max_new_token = st.slider(
        'Output length', 5, 32000, 256, step=1
    )

    cols = st.columns(2)
    export_btn = cols[0]
    clear_history = cols[1].button("Clear", use_container_width=True)
    retry = export_btn.button("Retry", use_container_width=True)

    

prompt_text = st.chat_input(
    'Chat with 元客世界!',
    key='chat_input',
)

tab = st.radio(
    'Mode',
    [mode.value for mode in Mode],
    horizontal=True,
    label_visibility='hidden',
)

if clear_history or retry:
    prompt_text = ""

match tab:
    case Mode.CHAT:
        demo_chat.main(
            retry=retry,
            top_p=top_p,
            temperature=temperature,
            prompt_text=prompt_text,
            system_prompt=system_prompt,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_token
        )
    case Mode.TOOL:
        demo_tool.main(
            retry=retry,
            top_p=top_p,
            temperature=temperature,
            prompt_text=prompt_text,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_token,
            truncate_length=1024)
    case Mode.CI:
        demo_ci.main(
            retry=retry,
            top_p=top_p,
            temperature=temperature,
            prompt_text=prompt_text,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_token,
            truncate_length=1024)
    case _:
        st.error(f'Unexpected tab: {tab}')
