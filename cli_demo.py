import os
import platform
from transformers import AutoTokenizer, AutoModel
model_path = "/home/liuhongyue/chatglm3/finetune_demo/chatglm3-6b"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# model = AutoModel.from_pretrained(model_path, trust_remote_code=True).cuda()
# 多显卡支持，使用下面两行代替上面一行，将num_gpus改为你实际的显卡数量
from utils import load_model_on_gpus
model = load_model_on_gpus(model_path, num_gpus=2)
model = model.eval()
# tokenizer = AutoTokenizer.from_pretrained("/home/liuhongyue/chatglm3/finetune_demo/chatglm3-6b",
#                                           trust_remote_code=True)
# model = AutoModel.from_pretrained("/home/liuhongyue/chatglm3/finetune_demo/chatglm3-6b",
#                                   trust_remote_code=True, device='cuda').half().cuda()
# model = model.eval()

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False

welcome_prompt = "欢迎使用亚信安全漏洞知识库问答应用，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"

def build_prompt(history):
    prompt = welcome_prompt
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM3-6B：{response}"
    return prompt

def main():
    past_key_values, history = None, []
    global stop_stream
    print(welcome_prompt)
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            past_key_values, history = None, []
            os.system(clear_command)
            print(welcome_prompt)
            continue
        print("\nChatGLM：", end="")
        current_length = 0
        for response, history, past_key_values in model.stream_chat(tokenizer, query, history=history,
                                                                    past_key_values=past_key_values,
                                                                    return_past_key_values=True):
            if stop_stream:
                stop_stream = False
                break
            else:
                # print(response)
                print(response[current_length:], end="", flush=True)
                current_length = len(response)
        print("")


if __name__ == "__main__":
    main()