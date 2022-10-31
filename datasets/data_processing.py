import pandas as pd
import re
import yaml
import argparse
from hanspell import spell_checker
from transformers import AutoTokenizer


def make_one_chars(sentence):
    # sentence = re.subn(s+'+',s, sentence)[0]
    sentence = re.subn('ㅋ+', 'ㅋ', sentence)[0]
    sentence = re.subn('ㅎ+', 'ㅎ', sentence)[0]
    sentence = re.subn(';+', ';', sentence)[0]
    sentence = re.subn('!+', '!', sentence)[0]
    sentence = re.subn('\?+', '?', sentence)[0]
    sentence = re.subn('ㅠ+', 'ㅠ', sentence)[0]
    sentence = re.subn('~+', '~', sentence)[0]
    
    return sentence

#아직 사용안하는 함수
def remove_char(sentence): #ㅋㅋㅋ 나 ㅎㅎ가 연속으로 나오는거 필요 없을 수도 있으니 아예 제거하는
    # pattern = re.compile('[ㅋ+]')
    sentence = re.sub('[ㅋ+]', '', sentence)
    sentence = re.sub('[ㅎ+]', '', sentence)
    return sentence

def processed_sentences(df):
    test1 = []
    test2 = []
    # i = 0
    for s in df['sentence_1']:
        sen_temp = s
        sen_temp = make_one_chars(sen_temp)
        test1.append(sen_temp)
        # i += 1
    for s in df['sentence_2']:
        sen_temp = s
        sen_temp = make_one_chars(sen_temp)
        test2.append(sen_temp)

    return test1, test2

def haspell_processing(sent1, sent2):
    fixed_test1 = []
    fixed_test2 = []

    for s in sent1:
        try:
            fixed_test1.append(spell_checker.check(s).checked)
        except: #에러 나는 애들은 그대로 문장 추가
            fixed_test1.append(s)
            print("sent_1", s)
    for s in sent2:
        try:
            fixed_test2.append(spell_checker.check(s).checked)
        except: #에러 나는 애들은 그대로 문장 추가
            fixed_test2.append(s)
            print("sent_2", s)
    return fixed_test1, fixed_test2

def stopword_processing(tsv, stopwords, tokenizer):
    new_tsv = tsv
    s1s, s2s = [], []
    for s1 in tsv['sentence_1']:
        sentence_tokens = []
        for word in tokenizer.tokenize(s1):
            tmp_word = word
            if "##" in word:
                tmp_word = word.replace('##', '')
            if tmp_word not in stopwords:
                sentence_tokens.append(word)
        s1s.append(tokenizer.decode(tokenizer.convert_tokens_to_ids(sentence_tokens)))
        
    for s2 in tsv['sentence_2'] :
        sentence_tokens = []
        for word in tokenizer.tokenize(s2):
            tmp_word = word
            if "##" in word:
                tmp_word = word.replace('##', '')
            if tmp_word not in stopwords:
                sentence_tokens.append(word)
        s2s.append(tokenizer.decode(tokenizer.convert_tokens_to_ids(sentence_tokens)))   
        
    new_tsv['sentence_1'] = s1s
    new_tsv['sentence_2'] = s2s
    
    return new_tsv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training SBERT.')
    parser.add_argument("--conf", type=str, default="../sbert_config.yaml", help="config file path(.yaml)")
    args = parser.parse_args()
    with open(args.conf, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    
    train_df = pd.read_csv('../NLP_dataset/train.csv')
    test_df = pd.read_csv('../NLP_dataset/test.csv')
    dev_df = pd.read_csv('../NLP_dataset/dev.csv')
          
    train_tmp = train_df
    test_tmp = test_df
    dev_tmp = dev_df

    train_sent1, train_sent2 = processed_sentences(train_tmp)
    test_sent1, test_sent2 = processed_sentences(test_tmp)
    dev_sent1, dev_sent2 = processed_sentences(dev_tmp)

    train_sent1, train_sent2 = haspell_processing(train_sent1, train_sent2)
    test_sent1, test_sent2 = haspell_processing(test_sent1, test_sent2)
    dev_sent1, dev_sent2 = haspell_processing(dev_sent1, dev_sent2)

    train_tmp['sentence_1'] = train_sent1
    train_tmp['sentence_2'] = train_sent2

    test_tmp['sentence_1'] = test_sent1
    test_tmp['sentence_2'] = test_sent2

    dev_tmp['sentence_1'] = dev_sent1
    dev_tmp['sentence_2'] = dev_sent2

    processed_train = train_tmp.to_csv("../NLP_dataset/han_processed_train.csv",index=False)
    processed_test = test_tmp.to_csv("../NLP_dataset/han_processed_test.csv",index=False)
    processed_dev = dev_tmp.to_csv("../NLP_dataset/han_processed_dev.csv",index=False)

    #read stopwords
    stopwords = []
    f = open('../stopwords_ver2.txt')
    lines = f.readlines()
    for line in lines:
        if '\n' in line:
            stopwords.append(line[:-1])
            
    tokenizer = AutoTokenizer.from_pretrained(config['base_model'])
    
    train_stop_tmp = stopword_processing(train_tmp, stopwords, tokenizer)
    dev_stop_tmp = stopword_processing(dev_tmp, stopwords, tokenizer)
    test_stop_tmp = stopword_processing(test_tmp, stopwords, tokenizer)
    
    train_stop_tmp.to_csv("../NLP_dataset/han_processed_stop_train.csv", index=False)
    dev_stop_tmp.to_csv("../NLP_dataset/han_processed_stop_dev.csv", index=False)
    test_stop_tmp.to_csv("../NLP_dataset/han_processed_stop_test.csv", index=False)
