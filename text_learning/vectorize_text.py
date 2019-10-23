#!/usr/bin/python

import os
import pickle
import re
import sys



from nltk.stem.snowball import SnowballStemmer
import string
try:
    maketrans = ''.maketrans
except AttributeError:
    # fallback for Python 2
    from string import maketrans

def parseOutText(f):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
        (in Part 2, you will also add stemming capabilities)
        and return a string that contains all the words
        in the email (space-separated) 
        
        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)
        
        """


    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    stemmed=""
    if len(content) > 1:
        ### remove punctuation
        text_string = content[1].translate(maketrans("", "", string.punctuation))
        
        ### project part 2: comment out the line below
#         words = text_string

        ### split the text string into individual words, stem each word,    
        ### and append the stemmed word to words (make sure there's a single
        ### space between each stemmed word)
        ps = SnowballStemmer("english")   
        
        words = text_string.split() 
        for w in words: 
#             print(w, " : ", ps.stem(w)) 
            stemmed= stemmed+ " "+ ps.stem(w)
        stemmed= stemmed.lstrip()
    

    return stemmed

    

# def main():
#     ff = open("./text_learning/test_email.txt", "r")
#     text = parseOutText(ff)
#     print (text)



# if __name__ == '__main__':
#     main()




import os
import pickle
import re
import sys

# sys.path.append( "./tools/" )
# from parse_out_email_text import parseOutText

"""
    Starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification.

    The list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    The actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project. If you have
    not obtained the Enron email corpus, run startup.py in the tools folder.

    The data is stored in lists and packed away in pickle files at the end.
"""


from_sara  = open("./text_learning/from_sara.txt", "r")
from_chris = open("./text_learning/from_chris.txt", "r")

from_data = []
word_data = []

### temp_counter is a way to speed up the development--there are
### thousands of emails from Sara and Chris, so running over all of them
### can take a long time
### temp_counter helps you only look at the first 200 emails in the list so you
### can iterate your modifications quicker
temp_counter = 0


for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    for path in from_person:
        ### only look at first 200 emails when developing
        ### once everything is working, remove this line to run over full dataset
#         temp_counter += 1
#         if temp_counter < 200:
        path = os.path.join('.', path[:-1])
        print (path)
#             email = open(path, "rb")
        email = open(path, "r")
        text = parseOutText(email)
#             print(text)
        text=text.replace("sara","")
        text=text.replace("shackleton","")
        text=text.replace("chris","")
        text=text.replace("germani","")
        word_data.append(text)
        if name == "sara":
            from_data.append(0)
        elif name == "chris":
            from_data.append(1)


        ### use parseOutText to extract the text from the opened email

        ### use str.replace() to remove any instances of the words
        ### ["sara", "shackleton", "chris", "germani"]

        ### append the text to word_data

        ### append a 0 to from_data if email is from Sara, and 1 if email is from Chris


        email.close()

print ("emails processed")
from_sara.close()
from_chris.close()

pickle.dump( word_data, open("your_word_data.pkl", "wb") )
pickle.dump( from_data, open("your_email_authors.pkl", "wb") )

### in Part 4, do TfIdf vectorization here

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(word_data)

print(len(vectorizer.get_feature_names()))
print(vectorizer.get_feature_names()[34597])





