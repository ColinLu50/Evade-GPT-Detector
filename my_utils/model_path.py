
local_flag = True

model_path_dict = {
    "Hello-SimpleAI/chatgpt-detector-roberta": "/data/data/hf_model_hub/HC3-chatgpt-detector-roberta",
    "roberta-base-openai-detector": "/data/data/hf_model_hub/roberta-base-openai-detector",
    "gpt2-medium": "/data/data/hf_model_hub/gpt2/medium",
    "t5-large": "/data/data/hf_model_hub/t5/large",
    "distilroberta-base": "/data/data/hf_model_hub/distillroberta"
}

def get_model_path(model_name):

    if local_flag and model_name in model_path_dict:
        return model_path_dict[model_name]
    else:
        raise Exception #TODO

    # return model_name



