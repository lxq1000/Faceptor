
# Faceptor: A Generalist Model for Face Perception

Official implementation of **[Faceptor: A Generalist Model for Face Perception](https://arxiv.org/abs/2403.09500)**.


Existing efforts for unified face perception mainly concentrate on representation and training. Our work focuses on unified model structure, achieving improved task extensibility and increased application efficiency by two designs of face generalist models.
<img src="https://github.com/lxq1000/Faceptor/blob/main/pictures/main.png" alt="Image" width="800">
- Overall architecture for the proposed Naive Faceptor
 <img src="https://github.com/lxq1000/Faceptor/blob/main/pictures/naive_faceptor.png" alt="Image" width="500">

- Overall architecture for the proposed Faceptor
 <img src="https://github.com/lxq1000/Faceptor/blob/main/pictures/faceptor.png" alt="Image" width="800">

**<p align="justify"> Abstract:** *With the comprehensive research conducted on various face analysis tasks, there is a growing interest among researchers to develop a unified approach to face perception.
Existing methods mainly discuss unified representation and training, which lack task extensibility and application efficiency. 
To tackle this issue, we focus on the unified model structure, exploring a face generalist model.
As an intuitive design, **Naive Faceptor** enables tasks with the same output shape and granularity to share the structural design of the standardized output head, achieving improved task extensibility.
Furthermore, **Faceptor** is proposed to adopt a well-designed single-encoder dual-decoder architecture, allowing task-specific queries to represent new-coming semantics. 
This design enhances the unification of model structure while improving application efficiency in terms of storage overhead.
Additionally, we introduce Layer-Attention into Faceptor, enabling the model to adaptively select features from optimal layers to perform the desired tasks. 
Through joint training on 13 face perception datasets, Faceptor achieves exceptional performance in facial landmark localization, face parsing, age estimation, expression recognition, binary attribute classification, and face recognition, achieving or surpassing specialized methods in most tasks.
Our training framework can also be applied to auxiliary supervised learning, significantly improving performance in data-sparse tasks such as age estimation and expression recognition.* </p>


## Highlights
- To the best of our knowledge, our work is the first to explore a face generalist model, with unified representation, training, and model structure. Our main focus is on the development of unified model structures.
- With one shared backbone and three types of standardized output heads, **Naive Faceptor** achieves improved task extensibility and increased application efficiency.
- With task-specific queries to deal with new-coming semantics, **Faceptor** further enhances the unification of model structure and employs significantly fewer parameters than Naive Faceptor. 
- The proposed Faceptor demonstrates outstanding performance under both multi-task learning and auxiliary supervised learning settings.

## TODO
- Release model files and inference code.
- Release training and validation code.
- Release data preparation process.

## Dataset
The face analysis tasks included in our experiment and the corresponding datasets used are as follows.
<img src="https://github.com/lxq1000/Faceptor/blob/main/pictures/dataset.png" alt="Image" width="800">

## Evaluation
Here are some test results. For detailed experimental information, please refer to our paper.

- Comparison between Naive Faceptor and Faceptor-Base.

  <img src="https://github.com/lxq1000/Faceptor/blob/main/pictures/compare.png" alt="Image" width="800">

- Comparison with other specialized models for dense prediction tasks (facial landmark localization, face parsing).

  <img src="https://github.com/lxq1000/Faceptor/blob/main/pictures/dense.png" alt="Image" width="800">

- Comparison with other specialized models for attribute prediction tasks (age estimation, expression recognition, binary attribute classification).

  <img src="https://github.com/lxq1000/Faceptor/blob/main/pictures/attribute.png" alt="Image" width="800">

- Comparison for face recognition.

  <img src="https://github.com/lxq1000/Faceptor/blob/main/pictures/identity.png" alt="Image" width="800">






## Citation
If you find Faceptor (or Naive Faceptor) useful for your research, please consider citing us:

```bibtex
@article{qin2024faceptor,
  title={Faceptor: A Generalist Model for Face Perception},
  author={Qin, Lixiong and Wang, Mei and Liu, Xuannan and Zhang, Yuhang and Deng, Wei and Song, Xiaoshuai and Xu, Weiran and Deng, Weihong},
  journal={arXiv preprint arXiv:2403.09500},
  year={2024}
}
@article{qin2023swinface,
  title={Swinface: a multi-task transformer for face recognition, expression recognition, age estimation and attribute estimation},
  author={Qin, Lixiong and Wang, Mei and Deng, Chao and Wang, Ke and Chen, Xi and Hu, Jiani and Deng, Weihong},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2023},
  publisher={IEEE}
}
```

