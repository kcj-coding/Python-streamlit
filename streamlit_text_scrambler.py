import streamlit as st
import random
import re

# python -m streamlit run streamlit_text_scrambler.py

st.title("Text Scrambler App")

# want 3 text boxes:
#    1 to take user input
#    1 to display original (user) input text
#    1 to show scrambled text

user_text = st.text_area(label="User text", placeholder="Enter text here...")

letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
numbers = "0123456789"

use_same_mappings = st.checkbox("Use same mappings")

if str(user_text) == "":
    user_text = "Enter text here..."
else:
    user_text = user_text

# functions to define how to scramble text

def phrase_generation(user_text):
    # for each word in text, replace each letter by a random letter from letters
    phrase = []
    for word in user_text.split(" "):
        new_word = []
        for letter in word:
            # check if letter is not a word, if so do not adjust it
            if letter in "[!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]":
                new = letter
                # check if letter is a number, if so pick a random number
            elif letter in numbers:
                #new_letter_num = random.randint(0,len(numbers)-1)
                #new = numbers[new_letter_num]
                new = str(random.randint(0,9))
                
            elif letter in "\n":
                new = letter
            else:
                new_letter_num = random.randint(0,len(letters)-1)
                new = letters[new_letter_num]
                if letter.isupper():
                    new = new.upper()
            new_word.append(new)
            new_phrase = ''.join(new_word)
        phrase.append(new_phrase)
    phrase = ' '.join(phrase)
    return phrase

def phrase_generation_same_mappings(user_text):
    # for each word in text, replace each letter by a random letter from letters
    xyz=[]
    xtras=[]
    phrase = []
    for word in user_text.split(" "):
        new_word = []
        for letter in word:
            if letter in xyz:
                t=[i for i,j in enumerate(xyz) if j==letter][0] # capture first instance
                # use existing match
                xtras.append(xtras[t])
                #t=t[0]
                xyz.append(letter)
                new = xtras[t]
            else:
                xyz.append(letter)
                # check if letter is not a word, if so do not adjust it
                if letter in "[!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]":
                    new = letter
                    xtras.append(new)
                    # check if letter is a number, if so pick a random number
                elif letter in numbers:
                    #new_letter_num = random.randint(0,len(numbers)-1)
                    #new = numbers[new_letter_num]
                    new = str(random.randint(0,9))
                    xtras.append(new)
                    
                elif letter in "\n":
                    new = letter
                    xtras.append(new)
                else:
                    new_letter_num = random.randint(0,len(letters)-1)
                    new = letters[new_letter_num]
                    xtras.append(new)
            if letter.isupper():
                new = new.upper()
                
            new_word.append(new)
            new_phrase = ''.join(new_word)
        phrase.append(new_phrase)
    phrase = ' '.join(phrase)
    return phrase

def capitalize(text):
    punc_filter = re.compile('([.!?;]\s*)')
    split_with_punctuation = punc_filter.split(text)
    for i,j in enumerate(split_with_punctuation):
        if len(j) > 1:
            split_with_punctuation[i] = j[0].upper() + j[1:]
    text = ''.join(split_with_punctuation)
    return text

col1, col2 = st.columns(2)
with col1:
    st.header("Original text")
    #st.text(user_text)
    st.text_area(label="", value=user_text)
with col2:
    st.header("Scrambled text")
    #st.text(phrase_generation(user_text))
    if use_same_mappings:
        st.text_area(label="", value=phrase_generation_same_mappings(user_text))
    else:
        st.text_area(label="", value=phrase_generation(user_text))
    #st.text(capitalize(phrase_generation(user_text)))