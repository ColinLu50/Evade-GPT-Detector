from my_utils.text_utils import escape_curly_braces

class PromptConstructor:

    def __init__(self, task_type):
        # In implementation, we refer P1 to AI and P2 to Human to eliminate LLM's bias
        self.text_type = 'writing'
        self.task_type = task_type

        assert self.task_type in {'essay', 'qa', 'rev-gen', 'paraphrase'}

    def prompt_extract_feature(self, human_task_outputs, ai_task_outputs):
        # The prompt in step 1: compare and extract the features from AI and Humans
        cur_prompt = 'Here are the {}s from two people P1 and P2:\n'.format(self.text_type)

        cur_prompt += '\nP1 {}s:\n'.format(self.text_type)
        for i, ai_data in enumerate(ai_task_outputs):
            cur_prompt += f'{ai_data}\n'

        cur_prompt += '\nP2 {}s:\n'.format(self.text_type)
        for i, human_data in enumerate(human_task_outputs):
            cur_prompt += f'{human_data}\n'

        cur_prompt += '\nCompare and give the key distinct feature (specifically vocabulary, sentence structure) of P2\'s {}s (do not show specific examples):'.format(self.text_type)

        return cur_prompt

    def _process_incontext_string(self, incontext_examples):

        new_incontext_examples = []
        for ice in incontext_examples:
            new_ice = []
            for s in ice:
                new_ice.append(escape_curly_braces(s))
            new_incontext_examples.append(new_ice)

        return new_incontext_examples



    def get_final_prompt(self, feature_text, incontext_examples):

        incontext_examples = self._process_incontext_string(incontext_examples)

        if self.task_type == 'essay':
            return self.get_final_prompt_essay(feature_text, incontext_examples)
        elif self.task_type == 'qa':
            return self.get_final_prompt_qa(feature_text, incontext_examples)
        elif self.task_type == 'rev-gen':
            return self.get_final_prompt_revgen(feature_text, incontext_examples)
        elif self.task_type == 'paraphrase':
            return self.get_final_prompt_paraphrase(feature_text, incontext_examples)
        else:
            raise Exception('Wrong task', self.task_type)

    def get_final_prompt_qa(self, feature_text, incontext_examples):
        prompt_str = '''{}\nBased on the description, answer questions in P2 style {}s:\n'''.format(feature_text, self.text_type)

        for x_ic, y_ic in incontext_examples:
            prompt_str += 'Q: {}\nP2: {}\n\n'.format(x_ic, y_ic)

        prompt_str += '''Q: {}\nP2:'''

        return prompt_str

    def get_final_prompt_essay(self, feature_text, incontext_examples):
        prompt_str = '''{}\nBased on the description, complete academic paragraph in P2 style {}s:\n'''.format(feature_text, self.text_type)

        # prompt_str = '''{}\nBased on the description, rewrite this to P2 style writings in {}:\n'''.format(extract_info, self.text_type)

        # append each icd
        for x_ic, y_ic in incontext_examples:
            prompt_str += 'Prompt: {}\nP2: {}\n\n'.format(x_ic, y_ic)

        prompt_str += '''Prompt: {}\nP2:'''

        return prompt_str


    def get_final_prompt_revgen(self, feature_text, incontext_examples):
        prompt_str = '''{}\nBased on the description, write a P2 style review about given object and key words, with specified sentiment:\n'''.format(feature_text)

        # append each icd
        for x_ic, y_ic in incontext_examples:
            prompt_str += '{}\nP2: {}\n\n\n'.format(x_ic, y_ic)

        prompt_str += '''{}\nP2:'''

        return prompt_str

    def get_final_prompt_paraphrase(self, feature_text, incontext_examples):
        prompt_str = '''{}\nBased on the description, rewrite this to P2 style {}:\n'''.format(feature_text,
                                                                                               self.text_type)

        for x_ic, y_ic in incontext_examples:
            prompt_str += 'Origin: {}\nP2: {}\n\n'.format(x_ic, y_ic)

        prompt_str += '''Origin: {}\nP2:'''

        return prompt_str


