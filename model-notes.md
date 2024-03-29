## Perpetrator


## Action Type


## Deaths

### Poisson models

To predict the number of fatalities based on textual descriptions of incidents, I initially attempted a multi-output regression approach using a customized Poisson regression model built on top of a DistilBERT transformer architecture. The objective was to predict multiple categories of fatalities from the incident summaries, aiming to leverage the model's understanding of natural language to accurately infer these counts.

The dataset contained highly skewed data, with many incidents resulting in zero fatalities, making it a challenging task for the model. To address this, I experimented with preprocessing techniques, including text tokenization using DistilBertTokenizer and adjusting the training setup to optimize for this skewed distribution.

However, after several training iterations, adjustments to the learning rate, and batch size, the model's predictions did not show significant improvement. The loss remained constant at zero, and the evaluation metrics (MSE, RMSE, MAE) on the validation set indicated that the model was not effectively learning from the training data.

Subsequently, I pivoted to a simpler approach, focusing on predicting just one category of fatalities at a time, hoping to simplify the task for the model. Despite this adjustment and the application of PoissonNLLLoss for a more appropriate loss function given the count data nature of your target variable, the predictions remained far from accurate. The model produced continuous values as predictions, which, even when rounded, did not align well with the true discrete integer labels in the dataset.

Throughout this process, I utilized various tools and techniques, including creating custom datasets for tokenization, adjusting training arguments for the Transformer's Trainer class, and implementing custom callbacks for debugging. Despite these efforts, the challenge of accurately predicting discrete count data from text with this model setup remained unsolved.

### Other potential options

- Question and answer model, prompting whether there were any deaths and how many people died in each category
- Simply classify whether there was a death and who was killed using a multiple-classification model
- I am thinking the best option might be to do a classification model first for whether there were deaths and who the victims were, and in the second stage try to do the counts of only the cases with the deaths. Here are some potential options for count extraction models:
    - regression model (Poisson perhaps, since there won't be zeros maybe it would work)
    - BertForQuestionAnswering (potentially very labor-intensive)
    - named entity recognition could extract the counts as entities (need to explore more)
- Here are some benefits of this approach: 
    = the initial classification model serves as a filter, reducing the possibility of the extraction model being applied to irrelevant text
    - since the extraction model only processes texts identified to contain relevant information, the data it trains on is more consistent, potentially making it easier to learn the task
    - the two-step approach allows us to manage and improve each step independently, which can be helpful for debugging and optimizing the system
    - even if we don't get results for the text extraction, we still have the codings for whether deaths occured


  

