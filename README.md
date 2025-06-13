# BERT-E2E-ABSA
Exploiting **BERT** **E**nd-**t**o-**E**nd **A**spect-**B**ased **S**entiment **A**nalysis
<p align="center">
    <img src="architecture.jpg" height="400"/>
</p>

## Objective 
- In this project, we aim to extract pairs (aspect, polarity) from sentences **(focus on sentence-level ABSA not entire document)**
- The input may like this "The food is great but the service is bad." -> Output format = [(food,POS) ; (service,NEG)]
- **We have updated the code by introducing the CNN-GRU layer on top of BERT architecture to get better result**
  
## Architecture
* Pre-trained embedding layer: BERT-Base-Uncased (12-layer, 768-hidden, 12-heads, 110M parameters)
* Task-specific layer: 
  - Linear
  - Recurrent Neural Networks (GRU)
  - Self-Attention Networks (TFM)
  - CNN + GRU 
## Dataset
- Rest14 - semeval2014 task 4

## Quick Start
- !python fast_run.py

## Environment
* Kaggle - GPU-P-100 

## Result
- Micro-f1, precision, recall and macro-f1 were used to evaluate, the ** micro-f1** is the one we prioritize

## Citation
The idea and code is reference from the paper:
```
@inproceedings{li-etal-2019-exploiting,
    title = "Exploiting {BERT} for End-to-End Aspect-based Sentiment Analysis",
    author = "Li, Xin  and
      Bing, Lidong  and
      Zhang, Wenxuan  and
      Lam, Wai",
    booktitle = "Proceedings of the 5th Workshop on Noisy User-generated Text (W-NUT 2019)",
    year = "2019",
    url = "https://www.aclweb.org/anthology/D19-5505",
    pages = "34--41"
}
```
     
