Date: 2024-09-01

## Overview

This repository contains data and code for a data pipeline that scrapes, auto-codes and visualizes data from the South Asia Terrorism Portal (SATP). The auto-coding is done by fine-tuning LLMs with a hand-coded set of approximately 10,000 conflict event descriptions. 

The objective is to fine-tune a series of models that can used to code new events. The events break down into a few key categories: 

1) The "perpetrator" or assailant (who took the action);
2) Action type (what was done);
3) Target (who was affected);
4) Casualties (how many people were killed or injured);
5) Location (where did the event take place);
6) Miscellaneous other information (e.g. property damage, kidnappings, surrenders, etc.)
   
## Data

The original hand-coded data is in the `data/satp-dataset.xslx` file. These have been partially wrangled in R (see `wrangle_sapt.qmd`). The cleaned data is in `data/satp_clean.csv` which is then used to generate CSV files relevant for each model we want to fine-tune, e.g. `perpetrator.csv`, `action_type.csv`, etc.

The data for location and the miscellaneous other categories like kidnappings, surrenders, etc. have not yet been hived off into separate files. 

I am not totally sure what is in all of the CSV files, e.g. `satp_classification.csv` and will need to look at them more closely when I get time. 

## Models

So far, we have been able to fine-tune a series of models using the Hugging Face `transformers` library. Specifically, we have used BERT-based models to code for the "perpetrator" and "action type" categories. 

We also tried fine-tuning a DistilBERT model with a Poisson head for the death count, but this was not very successful. So one next step is to try to fine-tune a Named Entity Recognition (NER) model like GLiNER to get the casualty figures. 

The team has discussed using an NER model to also get the location, but there is also a thought that some of this work could be done through the scraping process or third party API services. 

The model files could probably be cleaned up and combined in some cases (e.g. multiple perpetrator models), but let's discuss this in person before taking any action. 

There was some thought given to using RAG models to do some of the coding. I have not seen a RAG used in this way before, but I do like the idea of including a RAG model in the pipeline. One possibility is to use it to answer questions about the data in the web interface. 


