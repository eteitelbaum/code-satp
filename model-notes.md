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

## ConfliBERT paper models

### SATP tasks

Here is ChatGPT's summary of the evaluation tasks the ConfliBERT team performed on the SATP dataset.

1. **Binary Classification**:
   - **Task**: Events were categorized into two groups: those that were terrorist incidents and those that were not terrorist incidents.
   - **Purpose**: The goal here was to distinguish between text narratives that describe terrorist activities and those that do not, essentially filtering relevant terrorist-related content from unrelated information.

2. **Multi-label Classification of Terrorist Events**:
   - **Task**: Within the subset of events identified as terrorist incidents, this task involved classifying events based on the type of terrorist activity, such as bombings, armed assaults, kidnappings, etc.
   - **Purpose**: The aim was to further categorize the terrorist incidents into more specific types of events, providing a finer-grained understanding of the nature of the terrorist activities described in the text. This task acknowledges that a single incident can involve multiple types of terrorist activities (e.g., an incident could be both a bombing and an armed assault).

3. **Multi-label Classification Including Both Terrorist and Non-Terrorist Incidents**:
   - **Task**: This was similar to the second task but was performed on the full dataset, including both terrorist incidents and non-terrorist incidents.
   - **Purpose**: The objective here was to classify the full range of events based on the type of event, including identifying various types of terrorist activities within the broader context of all events covered in the dataset. This approach tests the model’s ability to identify specific event types in a mixed dataset of both relevant and irrelevant narratives.

They get relatively low F1 scores for the last task, which I think may have to do with the fact that the action types they specified were only relevant for the terrorist groups. Maybe doing the multi-label classification with categories that pertain to security forces would improve the model accuracy for the classification when performed on the whole dataset. 

Another possibility is that they trained across different types of conflicts, combining the Maoist insurgency with events in J&K and the Northeast. Perhaps models fine-tuned on specific conflicts would perform better. 

## Augmented data

One idea for a paper could be to walk the reader through the steps to improve the codings performed by the model with a limited dataset using augmented data. We could fine tune the model on the full training set and then see what happens when we drop down to a smaller number of cases (say 100 or 200). Then we could talk about strategies for getting the accuracy up to the level of what we see on the full training set. The same principles should apply to improving the codings for the full training set, so I suppose we could try doing that too. But I think it would be more interesting to look at it from the standpoint of "how low can you go" in terms of examples to get 90% accuracy or above. 

Here are some libraries ChatGPT recommended for augmented/synthetic data; 


1. **NLPAug:** A Python library that provides support for textual augmentations including synonym replacement, back translation, and insertion of contextual word embeddings. It can augment text at the word, character, or sentence level.

2. **TextAttack:** A framework for generating adversarial examples in NLP. It includes capabilities for data augmentation as well, allowing for synonym substitution and other transformations that can enrich your dataset without altering the original labels.

3. **AugLy:** A library developed by Facebook, which supports a wide range of multimedia augmentations including text. It can perform operations like text insertion, deletion, and replacement to generate augmented data.

4. **Snorkel:** A system for generating training data, which uses a programmatic approach. It allows users to write labeling functions to automatically label or augment data, which is particularly useful when scaling up the dataset creation process.

### Augment and train only on cases that are miscoded or both accurately and miscoded cases?

One question that I have is whether you would augment only examples that were miscoded or if you would also included accurately coded cases. Here is what ChatGPT says about it: 

When using data augmentation techniques, especially in the context of improving model performance on misclassified examples, you have a couple of strategic approaches:

1. **Augment Only Misclassified Examples:**
   - This approach focuses specifically on the examples that the model initially misclassified. By augmenting only these examples, you aim to provide the model with additional information and variations of the challenging cases, potentially improving its ability to correctly classify similar examples in the future.
   - This can be particularly useful when you have a limited dataset or when you’ve identified specific patterns, words, or contexts that consistently lead to misclassifications.

2. **Augment the Entire Dataset:**
   - Augmenting the entire dataset, including both correctly classified and misclassified examples, can increase the overall robustness and generalization of the model. This approach ensures that the model is not only exposed to more variations of the difficult cases but also reinforces its understanding of the broader dataset.
   - This strategy can be beneficial for preventing overfitting and improving the model's performance across a wider range of inputs.

**Choosing an Approach:**
- The choice between these strategies depends on your specific goals, the size and diversity of your original dataset, and the nature of the misclassifications. 
- If the dataset is relatively small or if the misclassifications are concentrated in certain identifiable areas, focusing on augmenting only the misclassified examples might be more efficient.
- If the goal is to improve the overall robustness and generalization capability of the model, or if the misclassifications are spread out without a clear pattern, augmenting the entire dataset could be more effective.

In practice, it might be useful to experiment with both approaches to see which one yields better results for your specific task and dataset.

## K-fold cross-validation

Could combine k-fold cross-validation with data augmentation. Start with augmentation then apply k-fold validation.

  

