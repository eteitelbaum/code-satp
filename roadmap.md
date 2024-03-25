## wrangling steps

- read into R using readxl (analysis)
- clean varnames with janitor `clean_names()`
- select
    - !last_name
- remove or merge duplicate event descriptions

## types of variables

- incident identifier
- location variables
    - district, block, village
    - constituency
    - longitude latitude
- date variables
    - date
    - year
- perpetrator
    - security, maoist, unknown, ?
    - number of perpetrators
- action type
    - armed assault, property, hijacking, etc.
    - up to three actions
        - recode according to categories, did this type of action occur?
    - action success
- target type
    - up to two targets
    - additional information on civilian targets
- fatalities and injuries
    - number of fatalities/injuries
    - civilian, security, maoist, govt. official, other armed group, total
- property damage
    - yes/no
    - extent of damage
- total number of hostages
- number of arrests 
    - total, commander, cadre, sympathizer, unknown
- number of surrenders
    - total, commander, cadre, sympathizer, unknown

## types of models

- text classification
- text summarization
- question-answering
- named entity recognition (place names)
- text generation (geocoding)?

- multiple labels
- single labels

## application

1. Assign incident number
    - Use LLM to identify location (below) and then ask ChatGPT for a suitable Python strategy for generating an identifier based on location and date
2. Identify district, block, village
    - **named entity recognition and/or information extraction (NER/IE), perhaps with Hugging Face spaCy**
3. Assign coordinates
    - **connect to API to Google Maps API, Open Street maps or GeoNames look for coordinates of village** 
4. Identify constituency
    - **Use Geopandas and Shapely to merge constituency boundaries and village coordinates**
5. Assign date variable
    - I think this would be given... 
6. Identify perpetrator (Maoist, security, unknown)
    - **single-label classification (SLC)**
7. Identify action type (armed assault, bombing, etc.)
    - **single-label or multi-label classification (SLC/MLC)**
8. Total number of hostages if any
    - **NER/IE**
    - **question answering (QA)**
9. Was the action successful?
    - **likely QA, could also try SLC or sentiment analysis**
10. Identify target type (up to two and additional info on civilian targets)
    - **SLC/MLC**
12. Number of fatalities in different categories
    - civilian, security, maoist, govt. official, other armed group, total
    - **NIE/QA**
13. Property damage (yes/no) and extent of damage (high/med/low)
14. Total number of arrests by category (total, commander, cadre, sympathizer, unknown)
    - **NIE/QA**
15. Total number of surrenders by category (total, commander, cadre, sympathizer, unknown) 
    - **NIE/QA**

## specific models for fine-tuning single-label and multiple-label models

- BERT: developed by Google, commonly used for classification and other tasks (computationally intensive)
- RoBERTa: trained by Facebook, optimized version of BERT (computationally intensive)
- DistilBERT: a lighter version of BERT that retains 95% of the performance but is 40% smaller and 60% faster
- ALBERT: another lighter version of BERT developed by Google
- GPT: can be adapted for classification tasks through techniques like prompt engineering or fine-tuning with a classification layer
- XLNet: a generalized autoregressive pretraining model that outperforms BERT on several benchmarks
- Transformer-XL: A specialized model for datasets with long documents; designed to handle long-term dependencies in text

## technical notes

- Can use these models on Mac with M1 or M2 chip but it is recommended to use Anaconda miniforge. 
- 10k rows should be enough for classification tasks
- Can use data augmentation techniques like paraphrasing sentences, swapping out named entities, or using back-translation to effectively increase your dataset size
- Proper use of regularization techniques can help prevent overfitting, making the model more robust even when the dataset is not very large

## issues

- Look at duplicate incidents... Are they really separate incidents and do they need to be merged?

## thoughts

- could be multiple papers for different elements of coding and different models/methods
- one paper could look at zero-shot or few-shot models versus fine-tuning/transfer learning (low-level versus high-level APIs) to code event types, and also potentially the effectiveness of single- versus multi-label approaches to event classification 
- a second could look at using NIE and QA to do event counts
- a third could be how do geo-location, e.g. identify the place names and coordinates based on the event description and then merge coordinates into a shapefile with constituency boundaries

