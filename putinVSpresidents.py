
import nltk
from nltk.corpus import stopwords
import string
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

with open('putin.txt', encoding='UTF-8') as file:
    putinText = file.read()

with open('USAInaugural.txt', encoding='UTF-8') as file:
    presText = file.read()

#PRE-PROCESSING

putinText = putinText.lower()
putTokens = nltk.word_tokenize(putinText)

presText = presText.lower()
presTokens = nltk.word_tokenize(presText)

stops = list(stopwords.words('english'))
punc = string.punctuation
punc = list(punc)
stops.extend(['would', 'must', 'like','applause','every']) 
punc.extend(['’', '–', '--','\'s'])
cleanPutinTokens = [token for token in putTokens if token not in punc and token not in stops]
cleanPresTokens = [token for token in presTokens if token not in punc and token not in stops]

fdPutin = nltk.FreqDist(cleanPutinTokens)
fdPres = nltk.FreqDist(cleanPresTokens)
print(fdPutin.most_common(25))
print(fdPres.most_common(25))

#putin bar graph
top10 = nltk.FreqDist(cleanPutinTokens).most_common(10)
top10 = pd.Series(dict(top10))
fig, ax = plt.subplots(figsize =(6,6))
theplot = sns.barplot(x=top10.values, y = top10.index, ax = ax, orient='h', palette = 'dark:salmon_r')
plt.title('Putin\'s Word Frequency')
plt.savefig('putinWordDist.png')


#Pres bar graph
top10Pres = nltk.FreqDist(cleanPresTokens).most_common(10)
top10Pres = pd.Series(dict(top10Pres))
fig, ax = plt.subplots(figsize =(6,6))
theplot = sns.barplot(x=top10Pres.values, y = top10Pres.index, ax = ax, orient='h', palette='coolwarm')
plt.title('US President\'s Word Frequency')
plt.savefig('presWordDist.png')

putText = nltk.Text(cleanPutinTokens)
presText = nltk.Text(cleanPresTokens)

#Putin dispersion plot
fig = plt.figure()
putText.dispersion_plot(['development', 'freedom','state', 'power', 'russia'])
fig.savefig('PutinDist.png')


#Presidents dispersion plot
fig = plt.figure()
presText.dispersion_plot(['development', 'freedom', 'state','power', 'america'])
fig.savefig('PresDist.png')

