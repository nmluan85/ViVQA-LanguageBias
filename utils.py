import re

period_strip = re.compile(r'(?!<=\d)(\.)(?!\d)')
comma_strip = re.compile(r'(\d)(,)(\d)')
punctuation_chars = re.escape(r';/[]"{}()=+\_-><@`,?!.')
punctuation = re.compile(r'([{}])'.format(re.escape(punctuation_chars)))
punctuation_with_a_space = re.compile(r'(?<= )([{0}])|([{0}])(?= )'.format(punctuation_chars))

def process_punctuation(s):
    if punctuation.search(s) is None:
        return s
    s = punctuation_with_a_space.sub('', s)
    if re.search(comma_strip, s) is not None:
        s = s.replace(',', '')
    s = punctuation.sub(' ', s)
    s = period_strip.sub('', s)
    return s.strip()

def get_answer_ids_from_train(df_train, vocab, type):
    color_answers = (
        df_train[df_train["type"] == type]["answer"]
        .astype(str)
        .str.lower()
        .map(process_punctuation)
        .unique()
        .tolist()
    )

    color_ids = [vocab[a] for a in color_answers if a in vocab]
    color_ids = sorted(list(set(color_ids)))
    return color_answers, color_ids


def preprocess_questions(df):
    questions = [question.lower() for question in list(df['question'])]
    return questions

def preprocess_answers(df):
    answers = [process_punctuation(answer.lower()) for answer in list(df['answer'])]
    return answers