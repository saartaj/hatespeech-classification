# hatespeech-classification


Hate Speech Classification Using ML-Algorithms & NLP¶
In [2]:
from zipfile import ZipFile  
import pandas as pd
import nltk
import re
import matplotlib.pyplot as plt
In [3]:
# connecting zipfile with python to read
file = ZipFile('Basic_ML_Model_for_Text_Classification.zip')
In [4]:
# using open function to open zipfile then reading csv using pandas
df = pd.read_csv(file.open('final_dataset_basicmlmodel.csv'))

# taking a glimpse of first 6 rows.
df.head()
Out[4]:
id	label	tweet
0	1	0	@user when a father is dysfunctional and is s...
1	2	0	@user @user thanks for #lyft credit i can't us...
2	3	0	bihday your majesty
3	4	0	#model i love u take with u all the time in ...
4	5	0	factsguide: society now #motivation
In [4]:
# Label summary
print('Class balance check:')
print(df['label'].value_counts()) # counting labels
df.drop('id', inplace=True, axis = 1)
Class balance check:
0    3000
1    2242
Name: label, dtype: int64
In [5]:
# choosen randomly few columns to get a fair ideal about comments made on the platform.

for index, comment in enumerate(df['tweet'][125:135]):
    print(index+125, '. ', comment)
125 .   @user ð d most impoant thing is to #enjoy your life - to be   - itâs all that matters. life is too sho. #pooh4u 
126 .  happy bihday chris evansððððððð a great actor and human ððð³ðð»ð¸ððð #chrisevans   #bihdayâ¦ 
127 .  our heas, thoughts, prayers go out to the more than 50 people who were murdered @ a gay nightclub in #florida   
128 .   @user demoing guitars for new album #newalbum #indie #guitars   #echobelly 
129 .  retweeted lion pro (@user  #tgif #webmareting #seo #community #management   #weekend... 
130 .   â #nzd/usd: targets the 100 week sma at 0.7190   #blog #silver #gold #forex
131 .   @user i've had pretty bad bihday weeks before, but so far this is the worst ever. ð #bihdayweeksucks #bithday27   #tâ¦
132 .  so blessed to have worked with sa's best leading ladiesðð 
133 .  happiest place on eah ð« #disneysmagickingdom #disney #magickingdom #disneyland   #orlandoâ¦ 
134 .  is kinda   to be among humans again.
In [6]:
def clean_frame(text):
    """
    This function will clean the data frame. It will let the go all the alphabates and filterout all other     characters.
    """
    # Anything which will be other than a to z or A to Z and ' will be replaced by whitespace
    text = re.sub(r'[^a-zA-Z\']', ' ', text)  
    # All the unicode characters will be removed
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    # all the letters will be lowered 
    text = text.lower()

    return text
In [9]:
# adding new column called clean text in the data frame. Using lambda function to clean per tweet.S
df['clean_text'] = df['tweet'].apply(lambda x: clean_frame(x))

# taking glimpse to get confirmation of filter
df.loc[1125:1135, :]
Out[9]:
id	label	tweet	clean_text
1125	1212	0	â #aud/usd: failures apparent at key resist...	aud usd failures apparent at key resist...
1126	1213	0	because i am happy! ðð #happiness #minio...	because i am happy happiness minio...
1127	1214	0	when someone is doing the effo to make people ...	when someone is doing the effo to make people ...
1128	1216	0	i'm super hungry but don't feel like cooking.....	i'm super hungry but don't feel like cooking ...
1129	1217	0	new bikini from my amazon list âºï¸ big x to...	new bikini from my amazon list big x to...
1130	1218	0	@user will definitely be reading these! !	user will definitely be reading these
1131	1219	0	my new car should be ready middle of next week...	my new car should be ready middle of next week...
1132	1220	0	happy monday everyone! lets make it a good wee...	happy monday everyone lets make it a good wee...
1133	1221	0	after sex sex video free tube	after sex sex video free tube
1134	1222	0	looks like #knowledge is power - and #happines...	looks like knowledge is power and happines...
1135	1223	0	on our way to the #cmtredcarpet â¨ #tunein ...	on our way to the cmtredcarpet tunein ...
In [10]:
STOP_WORDS = ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'also', 'am', 'an', 'and',
              'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below',
              'between', 'both', 'but', 'by', 'can', "can't", 'cannot', 'com', 'could', "couldn't", 'did',
              "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'else', 'ever',
              'few', 'for', 'from', 'further', 'get', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having',
              'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how',
              "how's", 'however', 'http', 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it',
              "it's", 'its', 'itself', 'just', 'k', "let's", 'like', 'me', 'more', 'most', "mustn't", 'my', 'myself',
              'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'otherwise', 'ought', 'our', 'ours',
              'ourselves', 'out', 'over', 'own', 'r', 'same', 'shall', "shan't", 'she', "she'd", "she'll", "she's",
              'should', "shouldn't", 'since', 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs',
              'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're",
              "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't",
              'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where',
              "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't",
              'www', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves']
In [11]:
def freq_creation(text):
    """
    Function will create frequency for the given corpses. 
    """
    word_list = []

    for tweets in text.split():
        # extending tweets
        word_list.extend(tweets)
        # creating series of words and counting their frequency.
    word_freq = pd.Series(word_list).value_counts()
        # removing disturbance words.
    word_freq = word_freq.drop(STOP_WORDS, errors='ignore')

    return word_freq
In [12]:
def negativity(words):
    for word in words:
        # if any word contains the given value then 1 otherwise 0 will be written.
        if word in ['n', 'not', 'non', 'no'] or re.search(r"\n't", word):
            return 1
        else:
            return 0
In [13]:
def rare_words(words, rare_100):
    for word in words:
        # for rare word encoding will be 1 otherwise 0.
        if word in rare_100:
            return 1
        else:
            return 0
In [14]:
def is_question(words):
    for word in words:
        # for question comment encoding will be 1 otherwise 0.
        if word in ['what', 'when', 'how', 'why', 'who']:
            return 1
        else:
            return 0
In [15]:
# Introducing new columns using lambda as we did see before.
word_freq = freq_creation(df['clean_text'].str)

least_100 = word_freq[-100:]

df['word_count'] = df['clean_text'].str.split().apply(lambda x: len(x))

df['negatives'] = df['clean_text'].str.split().apply(lambda x: negativity(x))

df['question'] = df['clean_text'].str.split().apply(lambda x: is_question(x))

df['word_rare'] = df['clean_text'].str.split().apply(lambda x: rare_words(x, least_100))


df['chr_count'] = df['clean_text'].apply(lambda x: len(x))
Explortion is required

In [16]:
# sklearn for modelling 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
In [17]:
X = df[['chr_count', 'word_rare', 'word_count', 'question', 'negatives']]

y = df['label']

trainx, testx, trainy, testy = train_test_split(X, y, random_state= 123)
In [18]:
# Naive Bayes Classifier modelling & fitting on train data. 
nb = GaussianNB()
nbfit = nb.fit(trainx, trainy)
In [19]:
# predicting for test dataset, using only 
pred = nb.predict(testx)
accuracy_score(testy, pred)
Out[19]:
0.4248665141113654
Concluding remarks: We have attempted to create a model which can classify tweets either if it is a hate speech or not. Although the accuracy score for our model is not upto the mark. But for sure once the data size will increase the model performance will also increase.

In [ ]:
