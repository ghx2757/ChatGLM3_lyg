from enum import Enum
import streamlit as st
# 设置页面配置：
st.set_page_config(
    page_title="ChatGLM3 Demo",# 页面主题
    page_icon=":robot:",# robot:（一个机器人图标）
    layout='centered',# centered（居中布局）
    initial_sidebar_state='expanded',# 初始侧边栏状态：expanded（展开状态）
)

import demo_chat, demo_ci, demo_tool

# DEFAULT_SYSTEM_PROMPT = '''
# You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown.
# '''.strip()
# DEFAULT_SYSTEM_PROMPT = '''
# 你是电商主播李佳琦，你在电商直播间内推销各种化妆品，回答客户对于化妆品的质量、价格、适宜人群、使用季节、功效、制作成份等等方面的问题，客户有可能买也有可能不买，你要认真回答态度友好，要更贴近直播间真实对话场景！
# '''.strip()
# DEFAULT_SYSTEM_PROMPT = '''

# '''.strip()
DEFAULT_SYSTEM_PROMPT = '''
你是电商主播郝邵文，你所在的直播间是阿文小超市，请结合直播间场景回复客户问题！
'''.strip()
# DEFAULT_SYSTEM_PROMPT = '''
# 请使用韩语回答, 不要出现其他语言！
# '''.strip()



st.title("✨元客视界🐱‍🏍")

# Add your custom text here, with smaller font size
st.markdown("<sub>智谱AI 公开在线技术文档: https://lslfd0slxc.feishu.cn/wiki/WvQbwIJ9tiPAxGk8ywDck6yfnof </sub> \n\n <sub> 更多 ChatGLM3-6B 的使用方法请参考文档。</sub>", unsafe_allow_html=True)

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
    system_prompt = st.text_area(
        label="System Prompt (Only for chat mode)",
        height=500,
        value=DEFAULT_SYSTEM_PROMPT,
    )

prompt_text = st.chat_input(
    'Chat with ChatGLM3!',
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
        demo_chat.main(top_p, temperature, system_prompt, prompt_text, repetition_penalty)
    case Mode.TOOL:
        demo_tool.main(top_p, temperature, prompt_text, repetition_penalty)
    case Mode.CI:
        demo_ci.main(top_p, temperature, prompt_text, repetition_penalty)
    case _:
        st.error(f'Unexpected tab: {tab}')
