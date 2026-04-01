## Editorial changes

- No indent after quotes
- Cross-ref to appendices not working
- Something looks messed up about Figure 1. 
  + First, is BERT really outperforming other models at 25%
  + Second, do the CIs match what is in the line chart? 
  + Oh wait, yeah I think it is right... 
- More like 1250 than 1000 examples for encoders to surpass chatgpt


## CPW Workshop

- Thank you 
- Background
- What do you think of these results. Are we close to replacing human coders? 
    - To what extent is it a labor-saving device
    - How would it change your funding applications? 
    - Does it make the continuation of bespoke datasets like this more viable? 
- Is it framed well for a PSC audience
    - Would it fit well in a conflict studies journal? 
    - Where would you send it for review? 
    - Anything that absolutely has to be done before sending it out? 

--- 

***Omar***

1. Contextualize the general background for a political science audience.
    - Perhaps broaden the discussion of the waves
    - Talk about the tradeoff between GPT models and encoders more in a more accessible way
    - Well read generalist versus well-trained undergrads (versus mechanical turk?)
2. Be more declarative about the benefits of encoders for focused, specialized topics 
    - my thought: lexical complexity versus contextual complexity needs to be fleshed out
3. Fixed random seed... Significantly different results for different results.
    - Try cross-validation with different fixed random seeds
4. What if you give decoder models more than 3-5 prompts
5. Human intercoder reliability rate as a benchmark
6. External validity--what about other conflicts/datasets
7. Can we combine all of these things? Discuss.

**Jacob**

1. Assumes that the reader knows the terms...
    - Micro, macro, weighted F1...
    - Maoist insurgency

**Students**
2. cnnnn
    - What is the target audience?
    - Need audience
    - More footnotes
    - More discussion of dataset at the beginning or dataset construction in general
    - Intercoder reliability
    - How did you select the time period
    - Threat to validity by comparing the models (5.1 versus 4.1)

3. Isabella
    - total time that model took
    - maybe say something about the architecture I used and how long it took

4. Janet
    - use more plain English the first time you are introducing
    - is conflict data a bottleneck... (UCDP)
    - missingness and systematic biases are bigger problems
    - e.g. unidentified actors
    - reliability of sources
    - multiple imputation

5. Cynthia
    - Define the terms
    - Hybrid models
    - More of an introduction
    - **Political bias, can machine coders be less biased?**

6. Alicia
    - Intro is very dense
    - Give some more qualitative examples up front; these are things that are hard to code
    - Do specifics matter? 

7. Jordy
    - Combination of human and AI
    - Intercoder reliability 
    - AI adjudicators

