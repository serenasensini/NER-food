"""

STEP 0 - PRE REQUISITES

"""


# python -m spacy download en_core_web_lg

# Import libraries
import spacy
import random
from spacy.util import minibatch, compounding
from pathlib import Path
from spacy.training import Example

# Load previous model
nlp = spacy.load('en_core_web_lg')

with open('food.txt') as file:
    dataset = file.read()

# Apply model to file
doc = nlp(dataset)
print("Entity:", [(ent.text, ent.label_) for ent in doc.ents])

"""

STEP 1 - TRAIN DATA

"""

# Prepare training data
words = ["ketchup", "pasta", "carrot", "pizza", "garlic", "tomato sauce", "basil", "carbonara", "eggs", "pillow",
         "pancakes", "parmigiana", "eggplant", "zucchini", "pineapple", "risotto", "espresso", "arrosticini", "antipasti",
         "fettuccine", "heavy cream", "bacon", "polenta", "tiramisù", "chocolate", "bucatini", "amatriciana", "applepie",
         "spaghetti", "fiorentina steak", "bistecca", "lemon", "cheesecake", "pecorino", "peperoni",
         "maccherone", "bread", "pastries", "Nutella", "amaro", "donut", "wine", "pear", "pistachio",
         "coca-cola", "fanta", "pepsi", "soup", "watermelon", "croissant", "cappuccino", "cherry", "strawberries",
         "coffee", "jam", "ice cream", "pummarola", "Pastiera", "beer"]

TRAIN_DATA = []

with open('food.txt') as file:
    dataset = file.readlines()
    for sentence in dataset:
        print("#############")
        print("Sentence")
        print(sentence)
        print("-----------")
        sentence = sentence.lower()
        entities = []
        for word in words:
            word = word.lower()
            if word in sentence:
                print("-----------")
                print("Word")
                print(word)
                print("-----------")
                print("Indexes")
                start_index = sentence.index(word)
                print(start_index)
                end_index = start_index + len(word)
                print(end_index)
                print("-----------")
                pos = (start_index, end_index, "FOOD")
                entities.append(pos)
        element = (sentence.rstrip('\n'), {"entities": entities})
        TRAIN_DATA.append(element)
        print("-----------")
        print("Entity")
        print(element)
        print("-----------")
        print("#############")


print(TRAIN_DATA)

"""

STEP 2 - UPDATE MODEL

"""

ner = nlp.get_pipe("ner")

for _, annotations in TRAIN_DATA:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

# Train model
with nlp.disable_pipes(*unaffected_pipes):
    for iteration in range(30):
        print("Iteration #" + str(iteration))

        # Data shuffle for each iteration
        random.shuffle(TRAIN_DATA)
        losses = {}
        batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
        for batch in spacy.util.minibatch(TRAIN_DATA, size=3):
            for text, annotations in batch:
                # Creation of Example object
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                # Update model
                nlp.update([example], losses=losses, drop=0.2)
        print("Losses", losses)


# Save the model
output_dir = Path('/ner/')
nlp.to_disk(output_dir)
print("Saved in folder: ", output_dir)


"""

STEP 3 - TEST THE UPDATED MODEL

"""


# Load updated model
print("Loading model: ", output_dir)
nlp_updated = spacy.load(output_dir)


doc = nlp_updated("I don't like pasta with chocolate!")
print("Entity:", [(ent.text, ent.label_) for ent in doc.ents])

doc = nlp_updated("Don't try to make pizza and pineapple!")
print("Entity:", [(ent.text, ent.label_) for ent in doc.ents])

doc = nlp_updated("I like pasta with salmon and vodka")
print("Entity:", [(ent.text, ent.label_) for ent in doc.ents])

doc = nlp_updated("in carbonara, parmigiano is not used.")
print("Entity:", [(ent.text, ent.label_) for ent in doc.ents])

doc = nlp_updated("I like pasta with mozzarella and tomato sauce")
print("Entity:", [(ent.text, ent.label_) for ent in doc.ents])
