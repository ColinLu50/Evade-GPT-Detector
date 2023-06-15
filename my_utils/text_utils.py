import re

def replace_changeline(text):
    return re.sub("\n+", " ", text).strip()

def is_obvious_AI(text):
    # text = text.lower()
    if text[:len("I'm sorry")].lower() == "I'm sorry".lower():
        return True
    ai_str_list = ['AI', 'language model', 'OpenAI']

    return any(substring in text for substring in ai_str_list)

def escape_curly_braces(s):
    return re.sub(r'{(.*?)}', r'(\1)', s)