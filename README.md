# EBGCN
We use BERT embeddings (bert-base-multilingual-uncased) instead of the unknown and irreproducible Ma Jing Embeds of Twitter15/16.
Since we cannot reproduce those, and half the tweets are missing, and a lot of her embeds have more tweets in the processed version than in the reported tree,
we change datasets to PHEME. 

(That's the point of this entire github repo)


Keep in mind the original code reports results via batch wise averaged results for F1. <br>
The results are hence inaccurate as compared to overall results.<br>
There is also the issue that their pheme dataset uses 3 class split when pheme is naturally only 4 or 2.<br>
So we don't know if "True rumours" = Rumour + True or Rumour+ False (using the veracity annotations.)<br>
No point investigating either because like Bigcn the original code is incorrect in how it reports results.<br>
forced_parser.py can be run on an output file (see it's description string inside) and it will give you a quick comparison between actual and their reported version. The issue is minor but an issue nonetheless <br>

DO THIS FOR PHEME (after Shaun's NAUGHTY Hijack)
```python main_ebgcn.py --datasetname PHEMEevent 1> PHEME_training_output.txt 2>&1```
```python main_ebgcn.py --datasetname PHEMEsplits 1> PHEME_training_output.txt 2>&1```

else:

```python main_ebgcn.py ```

is sufficient

Note that: ```ebgcn_predict_all.py ``` has an example within for usage on single instances. (requires a tree) Otherwise it is PHEME only, and requires the actual files.bigcn_predict_all.py


The PyTorch implementation for the [paper](https://arxiv.org/pdf/2107.11934.pdf): Towards Propagation Uncertainty: Edge-enhanced Bayesian Graph Convolutional Networks for Rumor Detection
 


```
@inproceedings{DBLP:conf/acl/WeiHZYH20,
  author    = {Lingwei Wei and
               Dou Hu and
               Wei Zhou and
               Zhaojuan Yue and
               Songlin Hu},
  title     = {Towards Propagation Uncertainty: Edge-enhanced Bayesian Graph Convolutional
               Networks for Rumor Detection},
  booktitle = {{ACL/IJCNLP} {(1)}},
  pages     = {3845--3854},
  publisher = {Association for Computational Linguistics},
  year      = {2021}
}
```


## Usage

You need to run the file ```Process/getTwittergraph.py``` first to preprocess the data. 

Then you can run the file ```main_ebgcn.py``` to train the model. 

