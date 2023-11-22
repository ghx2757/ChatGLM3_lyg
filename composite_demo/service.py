import json
from flask import Flask, request, Response
from client import get_client
from conversation import postprocess_text, preprocess_text, Conversation, Role


##
client = get_client()


## 创建服务
app = Flask(__name__)

# 配置
MAX_LENGTH = 8192 # 最大长度
DEFAULT_SYSTEM_PROMPT = '''
你是电商主播李佳琦，你在电商直播间内推销各种化妆品，回答客户对于化妆品的质量、价格、适宜人群、使用季节、功效、制作成份等等方面的问题，客户有可能买也有可能不买，你要认真回答态度友好，要更贴近直播间真实对话场景！
'''.strip()
top_p = 0.80
temperature = 0.95
system_prompt = DEFAULT_SYSTEM_PROMPT
repetition_penalty = 1.2

# 响应返回
response_dict = {
    "code" : int,
    "data" : {
      "talkId": str ,
      "question": str,
      "answer": {
          "segId": int,
          "text" : "",
          "startEnd":str  # 开始:"start", 结束:"end" , 内容:"continue" 
        }   
    },
    "msg" : str  
}

# 历史信息    
history: list[Conversation] = []

# 更新历史信息
def append_conversation(
    conversation: Conversation,
    history: list[Conversation],
) -> None:
    
    history.append(conversation)

# 响应
@app.route('/meet/chat',  methods=['POST'])
def answer():
    data = request.get_json()
    if data["prompt"]:
        prompt_text = data["prompt"]  
        talkId  = data["talkId"]
        
        dict_now = response_dict        
        dict_now["data"]["talkId"] = talkId
        dict_now["data"]["question"] = prompt_text
        
        prompt_text = prompt_text.strip()
        append_conversation(Conversation(Role.USER, prompt_text), history)

        input_text = preprocess_text(
            system_prompt,
            tools=None,
            history=history,
        )
        
        print("=== Input:")
        print(input_text)
        print("=== History:")
        print(history)
        
        # 流
        def generate():
            output_text = ''
            i = 0
            for response in client.generate_stream(
                system_prompt,
                tools=None, 
                history=history,
                do_sample=True,
                max_length=MAX_LENGTH,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty = repetition_penalty,
                stop_sequences=[str(Role.USER)],
            ):
                token = response.token
                if response.token.special:
                    print("=== Output:")
                    print(output_text)
                    match token.text.strip():
                        case '<|user|>':
                            break
                        case _:
                            print(f'Unexpected special token: {token.text.strip()}')
                            break
                output_text += response.token.text
                
                dict_now["data"]["answer"]["segId"] = i
                dict_now["data"]["answer"]["text"] = response.token.text
                if i == 0:            
                    dict_now["data"]["answer"]["startEnd"] = "start"
                else:
                    dict_now["data"]["answer"]["startEnd"] = "continue"
                
                dict_now["code"] = 0
                dict_now["msg"] = "generating"
                yield json.dumps(dict_now, ensure_ascii=False)
                i += 1
                
            
            
            # 更新历史
            append_conversation(Conversation(
                Role.ASSISTANT,
                postprocess_text(output_text),
            ), history)
            # 打印answer
            print(f'\ntalkId:{data["talkId"]}  prompt:{data["prompt"]}')
            print(f'answer:{output_text.strip()}')
            
            
        # 返回流    
        return Response(generate())
    
    else:
        # 返回异常信息
        dict_end = response_dict
        dict_end["code"] = -1
        dict_end["msg"] = "prompt is null"
        return json.dumps(dict_end, ensure_ascii=False)
    


if __name__ == '__main__':
    app.run(host='127.0.0.1',port=8601)