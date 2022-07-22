import nltk
from nltk.corpus import stopwords
import string
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#read in the file
with open('putin2.txt', encoding= 'UTF-8') as text:
    myfile = text.read()

#Tokenizing 
myfile = myfile.lower()
putTokens = nltk.word_tokenize(myfile)

#Defining and removing stops
stops = list(stopwords.words('english'))
punc = string.punctuation
punc = list(punc)
stops.extend(['would', 'must', 'like']) #for when we need to add custom stop words
punc.extend(['’', '–'])
cleanTokens = [token for token in putTokens if token not in stops and token not in punc]

fd = nltk.FreqDist(cleanTokens)
print(fd.most_common(50))

thetext = nltk.Text(cleanTokens)

#bar graph 
top20 = nltk.FreqDist(cleanTokens).most_common(20)
top20 = pd.Series(dict(top20))
fig, ax = plt.subplots(figsize =(10,10))
theplot = sns.barplot(x=top20.index, y = top20.values, ax=ax)
plt.xticks(rotation = 30)
plt.title('Putin\'s favorite words!!!')
plt.savefig('putinWordDist.png')

#dispersion plot 
fig = plt.figure()
thetext.dispersion_plot(['development', 'crimea', 'ukraine', 'sevastopol', 'defence', 'weapons', 'new'])
fig.savefig('nameDist.png')

