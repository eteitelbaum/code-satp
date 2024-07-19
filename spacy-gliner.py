import spacy
import pandas as pd
from spacy.tokens import Span
import gliner_spacy  # Importing the gliner-spacy package

# Load the CSV file
df = pd.read_csv('locations.csv')
districts = df['district'].unique().tolist()
blocks = df['block'].unique().tolist()
villages = df['village'].unique().tolist()

# Initialize a blank spaCy model
nlp = spacy.blank("en")

# Add the GLiNER component with custom labels
nlp.add_pipe("gliner_spacy", config={"labels": ["district", "block", "village"]})

# Custom NER component
def custom_ner_component(doc):
    entities = []
    for ent in doc.ents:
        entities.append(ent)
    for token in doc:
        if token.text in districts:
            entities.append(Span(doc, token.i, token.i+1, label="district"))
        elif token.text in blocks:
            entities.append(Span(doc, token.i, token.i+1, label="block"))
        elif token.text in villages:
            entities.append(Span(doc, token.i, token.i+1, label="village"))
    doc.ents = entities
    return doc

# Add the custom NER component to the pipeline
nlp.add_pipe(custom_ner_component, after="gliner_spacy")

# Example text
text = "Conflict event reported in the district of Kanpur, block Chakeri, and village Kulgaon."

# Process the text
doc = nlp(text)

# Print identified entities
for ent in doc.ents:
    print(ent.text, ent.label_)
