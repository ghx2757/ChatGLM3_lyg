"""
This code is the tool registration part. By registering the tool, the model can call the tool.
This code provides extended functionality to the model, enabling it to call and interact with a variety of utilities
through defined interfaces.
"""

import copy
import inspect
from pprint import pformat
import traceback
from types import GenericAlias
from typing import get_origin, Annotated
import subprocess

import requests
from datetime import datetime
import cn2an

_TOOL_HOOKS = {}
_TOOL_DESCRIPTIONS = {}

# register_tool 函数可以作为一种装饰器，用于在函数定义之上自动将其注册为系统中的 "工具"，并附上完整的描述、参数和类型注释。这种注册可用于生成帮助文本、强制正确使用或将函数动态链接到命令行界面或图形用户界面。
def register_tool(func: callable):
    tool_name = func.__name__
    tool_description = inspect.getdoc(func).strip() # 获取函数描述，并删除所有前导或尾部空白。
    python_params = inspect.signature(func).parameters # 检索函数的参数
    tool_params = []
    for name, param in python_params.items(): # 遍历这些参数，构建一个参数描述列表 (tool_params)
        annotation = param.annotation # 参数注解
        if annotation is inspect.Parameter.empty: # 检查该参数是否有类型注解
            raise TypeError(f"Parameter `{name}` missing type annotation")
        if get_origin(annotation) != Annotated: # 检查类型注解是否属于特殊类型 typing.Annotated
            raise TypeError(f"Annotation type for `{name}` must be typing.Annotated")

        typ, (description, required) = annotation.__origin__, annotation.__metadata__
        typ: str = str(typ) if isinstance(typ, GenericAlias) else typ.__name__
        if not isinstance(description, str):
            raise TypeError(f"Description for `{name}` must be a string")
        if not isinstance(required, bool):
            raise TypeError(f"Required for `{name}` must be a bool")

        tool_params.append({   # 每个参数的信息都会添加到 tool_params 列表中
            "name": name,
            "description": description,
            "type": typ,
            "required": required
        })
    tool_def = { # 构建一个字典（tool_def）其中包含工具名称、描述和参数信息
        "name": tool_name,
        "description": tool_description,
        "params": tool_params
    }
    print("[registered tool] " + pformat(tool_def))
    _TOOL_HOOKS[tool_name] = func  # 函数 func 注册到两个字典中： _TOOL_HOOKS 和 _TOOL_DESCRIPTIONS
    _TOOL_DESCRIPTIONS[tool_name] = tool_def

    return func


def dispatch_tool(tool_name: str, tool_params: dict) -> str:
    if tool_name not in _TOOL_HOOKS:
        return f"Tool `{tool_name}` not found. Please use a provided tool."
    tool_call = _TOOL_HOOKS[tool_name]
    try:
        ret = tool_call(**tool_params)
    except:
        ret = traceback.format_exc()
    return str(ret)


def get_tools() -> dict:
    return copy.deepcopy(_TOOL_DESCRIPTIONS)


# Tool Definitions

@register_tool
def random_number_generator(
        seed: Annotated[int, 'The random seed used by the generator', True],
        range: Annotated[tuple[int, int], 'The range of the generated numbers', True],
) -> int:
    """
    Generates a random number x, s.t. range[0] <= x < range[1]
    """
    if not isinstance(seed, int):
        raise TypeError("Seed must be an integer")
    if not isinstance(range, tuple):
        raise TypeError("Range must be a tuple")
    if not isinstance(range[0], int) or not isinstance(range[1], int):
        raise TypeError("Range must be a tuple of integers")

    import random
    return random.Random(seed).randint(*range)


@register_tool
def get_weather(
        city_name: Annotated[str, 'The name of the city to be queried', True],
) -> str:
    """
    Get the current weather for `city_name`
    """

    if not isinstance(city_name, str):
        raise TypeError("City name must be a string")

    key_selection = {
        "current_condition": ["temp_C", "FeelsLikeC", "humidity", "weatherDesc", "observation_time"],
    }
    import requests
    try:
        resp = requests.get(f"https://wttr.in/{city_name}?format=j1")
        resp.raise_for_status()
        resp = resp.json()
        ret = {k: {_v: resp[k][0][_v] for _v in v} for k, v in key_selection.items()}
    except:
        import traceback
        ret = "Error encountered while fetching weather data!\n" + traceback.format_exc()

    return str(ret)


# @register_tool
# def get_weather(
#         city_name: Annotated[str, 'The name of the city to be queried', True],
#     ) -> str:
#     """
#     Useful for when you need to answer questions about weather information. Input should be a string of city and country split by ',', both must be in English, the country should be a ISO 3166-1 alpha-2 code, for example 'London,GB'.
#     """ 
#     # OPENWEATHER_API_KEY = "2396a2ca7589f811fb038a2ffa1cc1b4" # wjm
#     OPENWEATHER_API_KEY = "526134873b19da21fac3d25509807d39" # ghx
#     lang = "zh_cn" # 语言
       
#     base_url = f'http://api.openweathermap.org/data/2.5/weather?'
#     url = f"{base_url}q={city_name}&lang={lang}&appid={OPENWEATHER_API_KEY}"
#     res = requests.get(url)
#     weather = res.json()
#     if weather["cod"] != 200:
#         return "没有寻找到有关该城市的天气信息"
#     else:
#         description = weather["weather"][0]["description"] # 天气状况
#         temp = weather["main"]["temp"] - 273.15 # 温度
#         feels_like = int(weather["main"]["feels_like"] - 273.15) # 体感温度
#         humidity = weather["main"]["humidity"] # 湿度
#         wind_speed = weather["wind"]["speed"] # 风速度

#         des = f"该地区当前天气为:{description},体感温度{feels_like}℃,空气湿度{humidity}%,风速{wind_speed}米/秒"
#         return des
    
@register_tool
def get_time(
        no_use: Annotated[str, 'The question of the time to be queried', True],
    ) -> str:
    """
    Useful for when you need to answer questions about Date or Time.
    s""" 
    time_now = datetime.now()
    time_now_str = time_now.strftime('%H时%M分%S秒')
    date_now_str = time_now.strftime('%Y年%m月%d日')
    weekday = str(time_now.weekday() + 1)
    weekday_cn = cn2an.an2cn(weekday)
    if weekday_cn == "七":
        weekday_cn = "日"
    return "今天的日期是:" + date_now_str + ",当前时间为:" + time_now_str + ",今天是星期" + weekday_cn


@register_tool
def get_shell(
        query: Annotated[str, 'The command should run in Linux shell', True],
) -> str:
    """
       Use shell to run command
    """
    if not isinstance(query, str):
        raise TypeError("Command must be a string")
    try:
        result = subprocess.run(query, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return e.stderr


if __name__ == "__main__":
    # print(dispatch_tool("get_shell", {"query": "pwd"}))
    print(get_tools())