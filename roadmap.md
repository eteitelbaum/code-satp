## the pipeline

1. scrape SATP, store in DB
2. code location, longitude and latitude 
3. code perpetrator and event type
4. code casualties
5. code hostages and events/surrenders
6. analyze and visualize

## wrangling steps

- read into R using readxl (analysis)
- clean varnames with janitor `clean_names()`
- select
    - !last_name
- remove or merge duplicate event descriptions

## scraping

The ConfliBERT team did some work on scraping SATP. See their repo here: 

https://github.com/snowood1/SATP/tree/master

It has good code for automating the downloading of event descriptions

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
    + multiple labels
    + single labels
- text summarization
- question-answering
- named entity recognition (place names)
- text generation (geocoding)?


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
    - **IE/QA**
    - multi-output poison regression
13. Property damage (yes/no) and extent of damage (high/med/low)
14. Total number of arrests by category (total, commander, cadre, sympathizer, unknown)
    - **IE/QA**
15. Total number of surrenders by category (total, commander, cadre, sympathizer, unknown) 
    - **IE/QA**

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
- perhaps the focus of a first paper could just be on doing the different types of coding tasks with an LLM that you could run on a laptop, and the issue of how many examples you need to get above a 90% accuracy threshold
- could be interesting to think of LLMs as just another layer of the data generating process (DGP); there is a level of randomness at every layer, starting with whether the incident gets reported to police to whether it gets reported in the media to whether the person summarizing the news accurately reads and summarizes it to how good a job the coders do with it... using an LLM to code only adds a small amount of randomness on top of all of those other layers of randomness... the only real question is whether there is any kind of **systematic bias** in the process... 

## for count models

- multi-output regression model
- use a poisson loss function
- treat like a regression of count data, but predicting multiple categories at once
- zero inflation negative binomial model likely not available out of the box
- decision to do predict one versus multiple categories at once depends on available computing power and how related categories are to one another
- for predicting the exact counts, the model's final layer should output a vector of size nn (the number of categories), and you would typically use a loss function suitable for regression, such as Mean Squared Error (MSE)
- another way to do it would be to treat it as a multi-output classification problem where counts fall within a discrete set of bins or ranges (e.g., 0, 1-5, 6-10, etc.)... each category would have its output layer predicting the bin index for its count. this approach requires more complexity in the model architecture and loss function, as you'd essentially be running nn separate classification tasks simultaneously
- can use DistilBERT for this

## plan of action

1. single-label classification models 
    + perpetrator (Maoist, security, unknown)
    + success of action (yes/no)
    + property damage (yes/no)
2. mulitple-label classification models
    + action type (armed assault, bombing, etc.)
    + target type 
    + civilian target type
    + extent of property damage (high/medium/low)
4. information extraction
    + number of fatalities in different categories (civilian, security, maoist, govt. official, other armed group, total)
    + number of injuries in different categories (civilian, security, maoist, govt. official, other armed group, total)
    + number of arrests in different categories (total, commander, cadre, sympathizer, unknown)
    + number of surrenders in different categories (total, commander, cadre, sympathizer, unknown) 
    + number of hostages  
3. named entity recognition
    + location of incident
4. geocoding locations? 
    - connect to API to Google Maps API, Open Street maps or GeoNames look for coordinates of village
    - use Geopandas and Shapely to merge constituency boundaries and village coordinates

  ## paper options

- Paper 1: Using classification models to identify perpetrator, action type, victim, whether violence occurred, etc. 
- Paper 2: Extracting counts such as deaths, injuries, number of people arrested, surrendered, protesting, etc. What strategies are available and which work best? Regression, BertForQuestionAnswering, NER. NER seems most promising at the moment.
- Paper 3: Extracting location data, presumably with NER, and automating the geocoding (if possible).
- Paper 4: Topic modeling to select articles and text summarization to prep them for coding.
- Paper 5: The whole Megillah, CI/CD pipeline of scraping, data, feeding to a topic model, summarizing the text, auto-coding the events and then visualizing the data.
    