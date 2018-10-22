### Naive Bayes ###
import csv
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import pandas as pd
import numpy as np
from nltk import pos_tag
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

## 전처리 ## 5572개 data
smsdata = open("d:/data/spam/SMSSpamCollection.txt", "r", encoding='utf-8')
csv_reader = csv.reader(smsdata, delimiter='\t')
smsdata_data = []
smsdata_labels = []

for line in csv_reader:
    smsdata_labels.append(line[0])
    smsdata_data.append(line[1])
    
smsdata.close()

c = {} ## import Counter; c = Counter(smsdata_labels)
for i in smsdata_labels:
    if not i in c:
        c[i] = 1
    else:
        c[i] += 1
#print(c)  # {'ham': 4825, 'spam': 747}

## 자연어 처리 (Nartural Language Processing) ##
test = "Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat..."
def preprocessing(text):
    # 단어 분할하여, punctuation이라면, 공백으로 대체, 나머지 문장은 str로 #
    text2 = " ".join("".join([" " if ch in string.punctuation else ch for ch in text]).split())
    # text2를 token화 (공백과 띄어쓰기 기준), 리스트 저장
    tokens = [word for sent2 in nltk.sent_tokenize(text2) for word in nltk.word_tokenize(sent2)]
    tokens = [word.lower() for word in tokens]
    stopwds = stopwords.words('english') # 대명사&접속어 사전
    tokens = [token for token in tokens if token not in stopwds]
    tokens = [word for word in tokens if len(word) >= 3]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    tagged_corpus = pos_tag(tokens) # pos-tagging -> 품사 dict 적용
    Noun_tags = ['NN', 'NNP', 'NNPS', 'NNS']
    Verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    lemmatizer = WordNetLemmatizer()
    # 표제어와 pos 사이에 불일치를 조정 (태그가 n이나 v라면 n과 v로 변환)
    def prat_lemmatize(token, tag):
        if tag in Noun_tags:
            return lemmatizer.lemmatize(token, 'n')
        elif tag in Verb_tags:
            return lemmatizer.lemmatize(token, 'v')
        else:
            return lemmatizer.lemmatize(token,'n')
        ##
    pre_proc_text = " ".join([prat_lemmatize(token, tag) for token, tag in tagged_corpus])
    return pre_proc_text # string으로 return

smsdata_data_2 = []
for i in smsdata_data:
    smsdata_data_2.append(preprocessing(i))    

trainset_size = int(round(len(smsdata_data_2) * 0.70))
x_train = np.array([''.join(rec) for rec in smsdata_data_2[0: trainset_size]])
y_train = np.array([''.join(rec) for rec in smsdata_labels[0: trainset_size]])
x_test = np.array([''.join(rec) for rec in smsdata_data_2[trainset_size+1: len(smsdata_data_2)]])
y_test = np.array([''.join(rec) for rec in smsdata_labels[trainset_size+1: len(smsdata_labels)]])

## Word2Vec: TF-IDF(term-frequency-inverse document frequency) 가중값 수행 ##
vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2), stop_words='english',
                             max_features=4000, strip_accents='unicode', norm='l2')
x_train_2 = vectorizer.fit_transform(x_train).todense() ## 각 단어의 빈도 차원마다 벡터 수치
x_test_2 = vectorizer.transform(x_test).todense() ## 엄청난 차원 형성 shape(3900, 4000(col))
## sklearn --> 첫 학습 데이터에 fit --> 이후는 transform

## Naive Bayes ##
clf = MultinomialNB().fit(x_train_2, y_train)
ytrain_nb_predicted = clf.predict(x_train_2)
ytest_nb_predicted = clf.predict(x_test_2)

print ("\nNaive Bayes - Train Confusion Matrix\n\n",pd.crosstab(y_train, ytrain_nb_predicted, rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nNaive Bayes- Train accuracy",round(accuracy_score(y_train, ytrain_nb_predicted),3))
print ("\nNaive Bayes  - Train Classification Report\n",classification_report(y_train, ytrain_nb_predicted))

print ("\nNaive Bayes - Test Confusion Matrix\n\n",pd.crosstab(y_test, ytest_nb_predicted,rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nNaive Bayes- Test accuracy",round(accuracy_score(y_test, ytest_nb_predicted),3))
print ("\nNaive Bayes  - Test Classification Report\n",classification_report(y_test, ytest_nb_predicted))

# Test accuracy 0.965 #
## 상위 특징 출력 ## (빈도)
feature_names = vectorizer.get_feature_names()
coefs = clf.coef_ # 다변수
intercept = clf.intercept_ # 절편
coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
n = 10
top_n_coefs = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1): -1])
for (coef_1, fn_1), (coef_2, fn_2) in top_n_coefs:
    print('\t%.4f\t%-15s\t\t%.4f\t%-15s' % (coef_1, fn_1, coef_2, fn_2))


















