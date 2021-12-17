# Grammatical Error Correction at the Character Level
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shubham-shetty/Reformer_GEC/blob/main/main.ipynb)

## Introduction
Grammatical error correction (GEC) is the task of detection and correction of grammatical errors in ungrammatical text. Grammatical errors include errors such as spelling mistake, incorrect use of articles or prepositions, subject-verb disagreement, or even poor sentence construction. GEC has become an important NLP task, with applications ranging from auto-correct for email or text editors to a language learning aid.

Over the past decade, neural machine translation-based approaches to GEC, which generate the grammatical version of a sentence from its ungrammatical form, have dominated the field. However, their slow inference speed, large data requirement, and poor explainability has led to research toward text-edit approaches, which make modifications to  ungrammatical sentences to produce their grammatical form.

In our work, we model GEC as a two-step problem that involves edits on the character level. The first subtask is deletion, where we fine tune a Reformer, an efficient character-level transformer, to predict whether or not to delete each character. The second step is insertion, where we predict which characters to insert between existing characters to produce a grammatical sentence. These operations can be trained independently, using the longest common subsequence between the ungrammatical and grammatical sentence as an intermediary label. 

This repository contains code to recreate models generated for our project. This project was developed as part of coursework for COMPSCI-685: Advanced Natural Language Processing at UMass Amherst, Fall '21.

## Dataset
Download datasets from following locations:  
1. [CLC FCE Dataset](https://ilexir.co.uk/datasets/index.html)

Prepare dataset once downloaded using following code:  
```.bash
python utils/prepare_clc_fce_data.py <INPUT> --output <OUTPUT_FILE> 
```

Generate labelled data from FCE data by running following code:
```.bash
python utils/process_FCE_data.py <OUTPUT_PATH> --truth <GRAMMATICAL_TEXT_PATH> --actual <ORIGINAL_TEXT_PATH>
```

## Model
![alt text](https://github.com/shubham-shetty/Reformer-GEC/blob/main/docs/GEC_Architecture.png?raw=true)
