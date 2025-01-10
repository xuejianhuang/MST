# Multiview spatio-temporal learning method based on dual dynamic graph convolutional networks
Code for the paper “Multiview Spatio-Temporal Learning with Dual Dynamic Graph Convolutional Networks for Rumor Detection”.

Detecting rumors on social networks is increasingly important due to their rapid dissemination and negative societal impact. The structural characteristics of propagation play a crucial role in rumor detection. However, most current graph neural network-based methods focus on spatial structural features, overlooking the temporal structural features or exploring spatio-temporal features from a single perspective, failing to comprehensively and finely learn representations of dynamic events. Therefore, this article proposes a multiview spatio-temporal feature learning method based on dual dynamic graph convolutional networks. 

<div align="center">
<img src='./model.png' width='80%'>
</div>

* Make sure the following files are present as per the directory structure before running the code：
```
├── data
|   └── pheme
|        ├── all-rnr-annotated-threads (this directory contains the raw files of the PHEME dataset)
|        │    ├── ebola-essien-all-rnr-threads
|        │    ├── charliehebdo-all-rnr-threads
|        |    ├── ......
|        ├── pheme_clean (this directory contains the .csv files processed from the raw files)
|        |    ├── 498235547685756928.csv
|        |    ├── ......
|        ├── pheme_temporal_data （this directory contains .npy files of labels, nodes, and text semantic features）
|        |     ├── label.npy
|        |     ├── propagation_node.npy
|        |     ├── propagation_node_idx.npy
|        |     ├── propagation_root_index.npy
|        |     └── text_embeddings.npy
|        ├── mid2stat.txt
|        ├── mid2text.txt
|        ├── mid2user.txt
|        ├── node2idx_mid.txt
|        └── pheme_id_label.txt
|   └── weibo
|        ├── json (this directory contains the raw files of the weibo dataset)
|        │   ├── 4010312877.json
|        |   ├── ......
|        ├── weibo_clean (this directory contains the .csv files processed from the raw files)
|        │     ├── 4010312877.csv
|        │     ├── ......
|        ├── weibo_temporal_data （this directory contains .npy files of labels, nodes, and text semantic features）
|        │     ├── label.npy
|        │     ├── propagation_node.npy
|        │     ├── propagation_node_idx.npy
|        │     ├── propagation_root_index.npy
|        │     └── text_embeddings.npy
|        ├── mid2stat.txt
|        ├── mid2text.txt
|        ├── mid2user.txt
|        ├── node2idx_mid.txt
|        ├── Weibo.txt
|        └── weibo_id_label.txt
├── logs（Log file directory）
|    ├── pheme_dual_32_2024-05-22-09-42-09.log
|    ├── ......
├── models
|    ├── bert-base-chinese
|    |     ├── config.json
|    |     ├── pytorch_model.bin
|    |     └── vocab.txt
|    ├── bert-base-uncased
|    |     ├── config.json
|    |     ├── pytorch_model.bin
|    |     └── vocab.txt
|    ├── config.py
|    ├── data.py
|    ├── data_process.py
|    ├── layers.py
|    ├── main.py
|    ├── models.py
|    ├── path_zh.py
|    └── util.py   
├── model_saved
├── preprocess
|    ├── getTextEmbedding.py
|    ├── getTwittergraph.py
|    ├── getWeibograph.py
|    ├── pheme_pre.py
|    ├── stop_words.txt
|    └── weibo_pre.py
└── requirement.txt
```
## Datasets
The experiments use two publicly real-world social network rumor datasets: Pheme and Weibo.
* The Raw Pheme dataset can be obtained from https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078 (or https://www.dropbox.com/s/j8x105s60ow997f/all-rnr-annotated-threads.zip?dl=0).
* The raw Weibo dataset can be downloaded from https://www.dropbox.com/s/46r50ctrfa0ur1o/rumdect.zip?dl=0. More information about this dataset is available here 'https://github.com/majingCUHK/Rumor_GAN'.
## Dependencies
* torch_geometric==2.5.2
* torch_scatter==2.1.2
* torch==1.12.1
* scipy==1.5.4
* tqdm==4.63.1
* numpy==1.21.5
* pandas==1.1.5
* visdom==0.2.4
* transformers==4.17.0
* jieba==0.42.1

## Run
* Step 1：Run pheme_pre.py and weibo_pre.py to process the raw files of the two datasets.
* Step 2: Run getTwittergraph.py and getWeibograph.py to generate files such as node2idx_mid.txt, mid2text.txt, mid2user.txt, propagation_node.npy, propagation_node_idx.npy, label.npy, and others.
* Step 3: Run getTextEmbedding.py to generate the text semantic feature matrix file text_embeddings.npy.
* Step 4: Run the command ‘python -m visdom.server’ to start the Visdom server for visualizing the training process.
* Step 5: Run ‘python main.py --dataset weibo --model dual’ to start training, validation, and testing. You can specify the dataset and model by using dataset and model arguments.

## Citation
If you find this project helps your research, please kindly consider citing our project or papers in your publications.
```
@ARTICLE{10836129,
  author={Huang, Xuejian and Ma, Tinghuai and Jin, Wenwen and Rong, Huan and Jia, Li and Yang, Bin and Xie, Xintong},
  journal={IEEE Transactions on Computational Social Systems}, 
  title={Multiview Spatio-Temporal Learning With Dual Dynamic Graph Convolutional Networks for Rumor Detection}, 
  year={2025},
  volume={},
  number={},
  pages={1-11},
  keywords={Feature extraction;Social networking (online);Semantics;Graph convolutional networks;Electronic mail;Data mining;Blogs;Information science;Spatiotemporal phenomena;Noise;Cross-attention;dynamic graph convolutional networks (DGCN);multiview learning;rumor detection;spatio-temporal features},
  doi={10.1109/TCSS.2024.3520105}}
```

## Note
* The default configuration of the code assumes that the number of states in the dynamic graph is 3. Trying other values requires modifying the configuration and some code.
* The hyperparameters of different models may vary, requiring adjustments to the configuration file.
* The data preprocessing process takes some time, and some of the generated files are quite large, making it inconvenient to upload them. We will share the preprocessing results later on.

## Acknowledgements
Thank you to **Mengzhu Sun** (Beijing University of Posts and Telecommunications, China), **Xi Zhang** (Beijing University of Posts and Telecommunications, China), **Jiaqi Zheng** (Beijing University of Posts and Telecommunications, China), **Guixiang Ma** (University of Illinois at Chicago) for providing the base model.










