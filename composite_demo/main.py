from enum import Enum
import streamlit as st
# 设置页面配置：
st.set_page_config(
    page_title="元客世界-Demo",# 页面主题
    page_icon=":elephant:",# robot:（一个机器人图标）
    layout='centered',# centered（居中布局）
    initial_sidebar_state='expanded',# 初始侧边栏状态：expanded（展开状态）
)

import demo_chat, demo_ci, demo_tool

# DEFAULT_SYSTEM_PROMPT = '''
# You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown.
# '''.strip()
DEFAULT_SYSTEM_PROMPT = '''

'''.strip()

st.title("✨元客视界🐱‍🏍")

# Add your custom text here, with smaller font size
st.markdown(
    "<sub>智谱AI 公开在线技术文档: https://lslfd0slxc.feishu.cn/wiki/WvQbwIJ9tiPAxGk8ywDck6yfnof </sub> \n\n <sub> 更多 ChatGLM3-6B 的使用方法请参考文档。</sub>",
    unsafe_allow_html=True)


class Mode(str, Enum):
    CHAT, TOOL, CI = '💬 Chat', '🛠️ Tool', '🧑‍💻 Code Interpreter'


with st.sidebar:
    top_p = st.slider(
        'top_p', 0.0, 1.0, 0.8, step=0.01
    )
    temperature = st.slider(
        'temperature', 0.0, 1.5, 0.95, step=0.01
    )
    repetition_penalty = st.slider(
        'repetition_penalty', 0.0, 2.0, 1.2, step=0.01
    )
    max_new_token = st.slider(
        'Output length', 5, 32000, 256, step=1
    )
    system_prompt = st.text_area(
        label="System Prompt (Only for chat mode)",
        height=500,
        value=DEFAULT_SYSTEM_PROMPT,
    )

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

match tab:
    case Mode.CHAT:
        demo_chat.main(top_p=top_p,
                       temperature=temperature,
                       prompt_text=prompt_text,
                       system_prompt=system_prompt,
                       repetition_penalty=repetition_penalty,
                       max_new_tokens=max_new_token)
    case Mode.TOOL:
        demo_tool.main(top_p=top_p,
                       temperature=temperature,
                       prompt_text=prompt_text,
                       repetition_penalty=repetition_penalty,
                       max_new_tokens=max_new_token,
                       truncate_length=1024)
    case Mode.CI:
        demo_ci.main(top_p=top_p,
                     temperature=temperature,
                     prompt_text=prompt_text,
                     repetition_penalty=repetition_penalty,
                     max_new_tokens=max_new_token,
                     truncate_length=1024)
    case _:
        st.error(f'Unexpected tab: {tab}')
