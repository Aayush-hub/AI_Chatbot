import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy
import tflearn
import tensorflow
import random
import json
import pickle
from tkinter import *

stemmer = LancasterStemmer()
root = Tk()

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

# If model is saved use this:
try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

# model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
# model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)


def chat():
    
    inp = "You: "+e.get()
    txt.insert(END,"\n"+inp)
    if e.get().lower() == "quit":
        txt.insert(END,"\n"+"Bye, Press the cross button to exit, see you again")
    else:
        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        if results[results_index] > 0.7:

            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']

            txt.insert(END,"\n"+f"(Bot): {random.choice(responses)}")
            e.delete(0, END)
            e.insert(0, "")
        else:
            txt.insert(END,"\n"+"I didn't get that. Please try another question")
            e.delete(0, END)
            e.insert(0, "")

root.resizable(width=True, height=True)
txt = Text(root)
txt.grid(row = 0, column = 0, columnspan = 2)
e = Entry(root,width = 80)
send = Button(root,text = "Enter", command=chat, width = 20, height = 1).grid(row = 1,column=1)
e.grid(row = 1,column=0)
root.title("Chatbot")
root.geometry("350x390")
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)
root.mainloop()
