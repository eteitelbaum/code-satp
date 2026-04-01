# Presentation Notes

## Title Slide

I am Manny Teitelbaum. I am an associate professor of political science at GW. I want to acknowledge my caouthors Mohiddin Bacha Shaik and Shiva Sharma, both of whom could unfortunately not be here today.

## Overview

- In comparative politics and international relations, we spend a long time hand coding event data
- Our idea was to take one dataset that we had spent a lot of time and resources coding to see how well LLMs could do at reproducing our human codings
- What we have is a dataset of 10k events from the SATP that focus on the Maoist insurgency in India
- The dataset is similar to the kids of event datasets that are used in a lot of political science research that include niche labels and a lot of lexical complexity, meaning the reports have lots of complicated ways of describing the same thing
- We decided to focus on LLMs with an encoder-only architecture because a) they designed to understand and classify text and b) because they have significant advantages in terms of cost, speed, privacy and reproducibility compared to commercial decoder-only models like Claude or the GPT series of models 

## Data: South Asia Terrorism Portal (SATP)

- So here is one event from our dataset that illustrates what I mean when I say "niche labels" and "lexical" complexity
- Read part of it
- Here the label that we coded this for is a "surrender" which was important for analyzing the dynamics of this particular conflict because there were a ton of surrenders as the momentum shifted away from the Moaist insurgents toward the government
- In terms of lexical complexity, you can see that there are a lot of India-specific names and terminology and also lots of non-standard phraseology

## Protocols for Human Codings

So then to code these events we followed this protocol where (read from slide)...

## Classification Tasks

For this paper, we had our LLMs attempt three classification tasks from this dataset: who was the perpetrator, what was the action that they took and what was the target of their action? These are very similar classification tasks to what you would find in a confict database like ACLED or GTD with the main difference being the lexical complexity of the event descriptions and that some of the labels are more specific to our research interests, like ""surrender" (which I mentioned earlier) or an attack on a mining company because there was a lot of mining activity in India's Red Corridor. "Maoist" or "surrender" or an attack on a mining company. 

## Models Compared

Here are the models that we compared. Each of them has a slightly different size and architecture. These are mostly general purpose encoder modes but notably we also included the domain specific ConfliBERT model which Javier is going to talk about in a little bit. 

## Research Questions

And so then we wanted to know how well these models would do replacing humans at these coding tasks, how fast they could do it (do they have to use the whole training set or could they do it with a subset of the data), how the various models trade off in terms of speed and accuracy, and how best to deal with rare labels, i.e. labels that don't have many examples to train on. 

## Modeling Strategy

So here is our modeling strategy. We set up perpetrator as a multiclass classification task and action and target as multilabel classification tasks. We used a standard 80-10-10 train-val-test split with stratified sampling across labels.  And then as is standard in the literature we focused on macro F1 as our main evaluation metric for the multiclass task and micro F1 for the multilabel tasks.

## Training Strategy

To assess this part about how long it takes the LLM to get up to speed on the data, we trained each model on progressively larger subsets of the training data, starting with 1/32nd (or roughly 3%) of the training data and doubling the size of the training set each time until we got to the full training set. And for each iteration we used a random sample of the training data.

## Perpetrator Results

These are the results for the perpetrator tasks. As you can see this was relatively easy for all of them. Except for the Electra model, they only needed about a quarter of the training data or 2000 examples to get up to F1 scores of .95 or so. And, with the exception of Electra, there are not really meaningful differences between the models.

## Action Type

Now looking at the action type task, we see a bit more variation in the performance of the models. Although they all end up with pretty high F1 scores, at the 25% mark, we can see that RoBERTA, ConfliBERT and XLNet are all doing pretty well, getting up to F1 scores of .9 or better while BERT, Electra and DistilBERT are lagging behind.

## Model Performance and Test Support by Action Type Label

Now looking at the F1 Scores for the individual labels at 100% of the training data, we can see that there is some varion in how well the models do acros the various action type labels. So for example the model does really well with arrest, surrender and armed assault, but not quite as well with infrastructure attacks or abduction. 

The issue seems to have more to do with lexical complexity than scarcity of data or label imbalance. So for example, surrender has a lot fewer examples than arrest or armed assault but the F1 scores are still really high. Whereas infrastructure and bombing don't have fewer examples than surrender but the F1 scores are lower. 

But overall the performance here is pretty good. We can debate whether an F1 of .85 for abductions is good enough for our research purposes but it is not bad. 

## Target Type

For the target type model, we see a similar pattern to the action type model where RoBERTA, ConfliBERT and XLNet are all doing pretty well at the 25% mark while BERT, Electra and DistilBERT are lagging behind but it is even more pronounced than it was with the action type task and the overall F1 scores are lower than for the action type model.

## Model Performance and Test Support by Target Type Label

When we look at the performance on the individual labels we can see why. Some labels like attacks on government officials, attacks on mining companies and attacks on non-maoist armed groups are very rare, and so the models cannot predict these labels at all. So here is where imablance handling strategies might actually make a difference. 

## Imbalance Handling Strategies

So then to deal with some of these rare labels like attacks on government officials, we tried six different strategies using 100% of the training data with ConfliBERT... (read from slide).

## Imbalance Handling Results

Here we see that these strategies do really well at improving the performance of the rare labels. Just focusing on non-Maoist armed groups, we see that label go from an F1 score of 0 to above .8 for back translation, T5 augmentation and class weights. 

Whether this is high enough to be considered "success" from a social science standpoint is something we can debate but it does illustrated that these strategies can make a big difference for rare labels.

On the other hand mining company does not improve above 50% so maybe for labels like this that are still challenging for the LLM after imbalance handling strategies are applied, we would still want to keep human coders on hand. 

## Precision Recall Tradeoffs

So it is important to note that these strategies do come with tradeoffs in terms of precision and recall. We may want to privelege precision when we want to minimize false positives, for example if we are using the data to predict future events. Or we may want to privelege recall when we want to minimize false negatives, for example if we are using the data to understand the full scope of a conflict.

## Precision-Recall Plots

We can see that most of the strategies tend to trade off precision for recall, but some are more balanced than others. And focal loss, which focuses training on the hardest to classify examples, goes in the opposite direction and is more likely to improve precision than recall.

## Model Performance Versus Speed

So one other thing we looked at was speed and performance tradeoffs of these models. So looking at the action type and target typemodels, XLNET was often the best performing model but it was also the slowest. Whereas DistilBERT was the fastest but also took the most examples to get up to speed. Electra was the worst performing but also the slowest. ConfliBERT, Roberta and BERT were similar in terms of speed and performance on 100% of the training data, but as we saw earlier, Roberta and ConfliBERT got up to speed faster than BERT. So just from a basic horse-race perspective, Roberta and ConfliBERT seem to be the best options to start out with for these kinds of classification tasks.

## Summary of Findings

(read slide)

## Future Work

(read slide)

## Comments? 

Thanks for your attention! This QR code will take you to the slides and the link at the bottom of the slides will take you to our GitHub repo where you can find the code and data to replicate this analysis. We look forward to your comments and questions.

