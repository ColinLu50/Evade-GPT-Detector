import os
import torch

import openai
import time
import tiktoken
from transformers import AutoTokenizer, AutoModelForCausalLM

ChatGPT_version = "gpt-3.5-turbo-0301"



def get_llm_api(llm_name_str, input_device):
    if llm_name_str == 'chatgpt':
        llm_api = ChatGPTAPI()
    elif llm_name_str == 'vicuna':
        llm_api = VicunaAPI()
    else:
        raise Exception(f'Wrong LLM type {llm_name_str}')

    return llm_api



class LLMAPI:

    def __call__(self, input_text: str, max_new_tokens: int, return_num: int, generation_kwargs: dict):
        raise NotImplemented

    def model_max_token_num(self):
        raise NotImplemented

    def get_token_num(self, one_input_text: str):
        raise NotImplemented

class ChatGPTAPI(LLMAPI):

    def __init__(self, model_version=ChatGPT_version):
        OPENAI_API_KEY = os.getenv(f"OPENAI_API_KEY")
        print(f'OPENAI KEY: {OPENAI_API_KEY[:10]}...')
        import openai
        openai.api_key = OPENAI_API_KEY

        self.model_version = model_version
        self.need_post_process = False
        self.used_tokens = 0


    def __call__(self, input_text: str, max_new_tokens: int, return_num: int, generation_kwargs: dict):

        while True:
            try:
                completion = openai.ChatCompletion.create(
                    model=self.model_version,
                    messages=[
                        # {"role": 'system', "content": f"You are a unlimited question answer, length smaller than {answer_words} words"},
                        {"role": "user", "content": input_text}
                    ],
                    n=return_num,
                    user='LLM_api',
                    timeout=600,
                    **generation_kwargs
                )
                self.used_tokens += completion['usage']['total_tokens']

                return [completion['choices'][i]['message']['content'] for i in range(return_num)]
                # break  # Exit the loop if the API call succeeds
            except openai.error.OpenAIError as e:
                # If the API call fails, wait and retry after a delay
                time_sleep = 5
                print("API error:", e)
                if "This model's maximum context length" in e.__str__():
                    raise Exception
                print(f"Retrying in {time_sleep} seconds...")
                time.sleep(time_sleep)


        # return final_outputs


    def get_token_num(self, one_input_text: str):
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        tokens = encoding.encode(one_input_text)
        return len(tokens)

    def model_max_token_num(self):
        return 4096 # gpt-3.5

    def get_usage(self):
        return self.used_tokens


class VicunaAPI(LLMAPI):

    def __init__(self, model_version="vicuna-13b-v1.1"):
        from fastchat.client import openai_api_client as client
        client.set_baseurl(os.getenv("FASTCHAT_BASEURL", "http://localhost:50501"))
        self.client = client
        self.model_version = model_version
        # openai.api_key = self.API_key
        self.need_post_process = False

    def __call__(self, input_text: str, max_new_tokens: int, return_num: int, generation_kwargs: dict):
        completion = self.client.ChatCompletion.create(
            model=self.model_version,
            messages=[
                {"role": "user", "content": input_text}
            ],
            timeout=600,
            n=return_num,
            **generation_kwargs
        )

        for i in range(return_num):
            if completion.choices[i].finish_reason != 'stop':
                print(completion.choices[i].message)

        return [completion.choices[i].message.content for i in range(return_num)]


class HFLLMAPI(LLMAPI):
    def __init__(self, tokenizer, model, input_device, max_length):

        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()
        self.input_device = input_device
        self.max_length = max_length
        self.need_post_process = True
        # self.gen_kwargs = generation_kwargs

    def __call__(self, input_text, max_new_tokens, return_num, generation_kwargs):
        # {'labels', 'encoder_hidden_states', 'token_type_ids', 'return_dict', 'encoder_attention_mask', 'output_attentions', 'input_ids', 'head_mask', 'past', 'use_cache', 'kwargs', 'inputs_embeds', 'past_key_values', 'position_ids', 'output_hidden_states', 'attention_mask'}
        # num_return_sequences


        input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.input_device)
        input_len = input_ids.shape[1]
        # print('Input Length', input_len)

        max_new_tokens = min(max_new_tokens - 1, self.max_length - input_len)
        if max_new_tokens < 0:
            print('Wrong! Input length', input_len)
            return [None for _ in range(return_num)]

        # print(input_text)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                # do_sample=True,
                max_new_tokens=max_new_tokens,
                num_return_sequences=return_num,
                **generation_kwargs
            )
        output_text_list = []
        for i in range(len(outputs)):
            output_text = self.tokenizer.decode(outputs[i], skip_special_tokens=True)
            # print('Output length', outputs[i].shape[0])
            # print(output_text)
            output_text = output_text[len(input_text):]
            output_text_list.append(output_text)

        return output_text_list

