# Awesome-Personalized-RAG-Agent

![Awesome](https://awesome.re/badge.svg)  [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Papers

### 1. Pre-retrieval
<details><summary><b>1.1 Query Rewriting</b></summary>

<p>

 **Name** | **Title** |              **Personalized presentation**              | **Publication** |                **Paper Link**                | **Code Link**                |
|:---:|:---|:-------------------------------------------------------:|:---------------:|:---:|:--------------------------------------------:|
| Least-to-most Prompting | Least-to-Most Prompting Enables Complex Reasoning in Large Language Models |     Split by sub-query terms in different questions     |    ICLR 2023    | [[Link]](https://openreview.net/forum?id=WZH7099tgfM) |


</p>
</details>

<details><summary><b>1.2 Query Expansion</b></summary>
<p>


</p>
</details>

<details><summary><b>1.3 Other Query-related</b></summary>
<p>


</p>
</details>

### 2. Retrieval
<details><summary><b>2.1 Indexing </b></summary>
<p>
  
 **Name** | **Title** |              **Personalized presentation**              | **Publication** |                **Paper Link**                | **Code Link**                |
|:---:|:---|:-------------------------------------------------------:|:---------------:|:---:|:--------------------------------------------:|
| Least-to-most Prompting | Least-to-Most Prompting Enables Complex Reasoning in Large Language Models |     Split by sub-query terms in different questions     |    ICLR 2023    | [[Link]](https://openreview.net/forum?id=WZH7099tgfM) |


</p>
</details>

<details><summary><b>2.2 Retrieve </b></summary>
<p>

 **Name** | **Title** |              **Personalized presentation**              | **Publication** |                **Paper Link**                | **Code Link**                |
|:---:|:---|:-------------------------------------------------------:|:---------------:|:---:|:--------------------------------------------:|
|            | Optimization Methods for  Personalizing Large Language Models through Retrieval Augmentation | Gradients based on personalized scores | SIGIR 2024         | [[Link]](https://dl.acm.org/doi/pdf/10.1145/3626772.3657783) |                                                              |
| MeMemo     | MeMemo: On-device Retrieval  Augmentation for Private and Personalized Text Generation | 保护隐私数据的工作，使用户能够隐私和高效的检索               | SIGIR 2024 (short) | https://dl.acm.org/doi/pdf/10.1145/3626772.3657662 | https://github.com/poloclub/mememo                           |
| LAPS       | Doing Personal LAPS:  LLM-Augmented Dialogue Construction for Personalized Multi-Session  Conversational Search | 利用LLM在会话历史中生成用户的偏好，并利用人工检验            | SIGIR 2024         | https://dl.acm.org/doi/pdf/10.1145/3626772.3657815 | https://github.com/informagi/laps                            |
|            | Partner Matters! An Empirical  Study on Fusing Personas for Personalized Response Selection in  Retrieval-Based Chatbots | 个性化对话系统                                               | SIGIR 2021         | https://dl.acm.org/doi/pdf/10.1145/3404835.3462858 | https://github.com/JasonForJoy/Personalized-Response-Selection |
| ERRA       | Explainable Recommendation with  Personalized Review Retrieval and Aspect Learning | 使用注意力机制封装个性化信息                                 | ACL 2023           | https://arxiv.org/pdf/2306.12657                   | https://github.com/Complex-data/ERRA                         |
|            | RECAP: Retrieval-Enhanced  Context-Aware Prefix Encoder for Personalized Dialogue Response Generation | 设计了一个分层 transformer  检索器，它可以根据不同的目标用户进行个性化的历史检索 | ACL 2023           | https://arxiv.org/pdf/2306.07206                   | https://github.com/isi-nlp/RECAP                             |
| HEART      | HEART-felt Narratives:     Tracing Empathy and Narrative Style in Personal Stories with LLMs | LLM 在从 HEART 中提取叙事元素，分析写作风格                  | EMNLP 2024         | https://arxiv.org/pdf/2405.17633                   | https://github.com/mitmedialab/heartfelt-narratives-emnlp    |
| OPPU       | Democratizing Large Language  Models via Personalized Parameter-Efficient Fine-tuning | 使用用户的个人行为历史微调 PEFT 模块，个性化的 PEFT  参数封装了行为模式和偏好。 | EMNLP 2024         | https://arxiv.org/pdf/2402.04401                   | https://github.com/TamSiuhin/OPPU                            |
| LAPDOG     | Learning Retrieval Augmentation  for Personalized Dialogue Generation | 提出一个用于检索有用信息以丰富角色的检索器和一个用于生成对话的生成器，用于将额外的上下文信息集成到个性化对话生成中 | EMNLP 2023         | https://arxiv.org/pdf/2406.18847                   | https://github.com/hqsiswiliam/LAPDOG                        |
| UniMP      | Towards Unified Multi-Modal  Personalization: Large Vision-Language Models for Generative Recommendation  and Beyond | 统一的数据格式，以摄取用户历史记录信息,有助于生成多模态输出以满足个人需求 | ICLR 2024          | https://arxiv.org/pdf/2403.10667                   |                                                              |
|            | Personalized Language Generation  via Bayesian Metric Augmented Retrieval | 使检索机制适应用户的偏好                                     | Arxiv              | https://openreview.net/pdf?id=n1LiKueC4F           |                                                              |
|            | Leveraging Similar Users for  Personalized Language Modeling with Limited Data | 利用来自相似用户的数据为新用户构建个性化 LM 的方法           | ACL 2022           | https://aclanthology.org/2022.acl-long.122.pdf     |                                                              |
| UIA        | A Personalized Dense Retrieval  Framework for     Unified Information Access | 使用注意力网络对用户数据加权学习表征，同时对模型进行个性化微调 | SIGIR 2023         | https://dl.acm.org/doi/pdf/10.1145/3539618.3591626 | https://github.com/HansiZeng/UIA                             |
| XPERT      | Personalized Retrieval over  Millions of Items               | 将个性化的用户数据作为输入来生成个性化的用户以及物品表征来进行retrieval | SIGIR 2023         | https://dl.acm.org/doi/pdf/10.1145/3539618.3591749 | https://github.com/personalizedretrieval/xpert               |
| DPSR       | Towards personalized and  semantic retrieval: An end-to-end solution for e-commerce search via  embedding learning | 数据输入不同，把加入了用户不同的数据称为模型的个性化版本     | SIGIR 2020         | https://dl.acm.org/doi/pdf/10.1145/3397271.3401446 |                                                              |
| PersonalTM | PersonalTM: Transformer Memory  for Personalized Retrieval   | 将用户个性化信息作为embedding利用transformer计算相似度来进行retrieval | SIGIR 2023 (short) | https://dl.acm.org/doi/pdf/10.1145/3539618.3592037 |                                                              |
|            | A zero attention model for  personalized product search      | 根据当前查询和用户信息自动确定何时以及如何个性化搜索结果     | CIKM 2019          | https://dl.acm.org/doi/pdf/10.1145/3357384.3357980 |                                                              |
| RTM        | Learning a Fine-Grained  Review-based Transformer Model for Personalized Product Search | 提出基于评论的 transformer 模型来个性化产品搜索  ，能实现更细粒度的匹配、动态用户/项目表示、泛化能力和个性化 | SIGIR 2021         | https://dl.acm.org/doi/pdf/10.1145/3404835.3462911 | https://github.com/kepingbi/ProdSearch                       |

</p>
</details>

<details><summary><b>2.3 Post-Retrieve</b></summary>
<p>



</p>
</details>

### 3. Generation
<details><summary><b>3.1 Generation from Explicit Preference</b></summary>
<p>



</p>
</details>

<details><summary><b>3.2 Generation from Implicit Preference</b></summary>
<p>



</p>
</details>

### 4. Agentic RAG

<details><summary><b> 4.1 Understanding </b></summary>
<p>



</p>
</details>

<details><summary><b> 4.2 Planing and Execution </b></summary>
<p>



</p>
</details>

<details><summary><b> 4.3 Generation </b></summary>
<p>



</p>
</details>


## Datasets and Evaluation

|    **Field**    | **Dataset**  |    **Matrics**  | **Link**   |
|:---------------:|:--------|:-----------:|:-----------:|
| Query Rewriting | SCAN | accuracy | [[Link]](https://openreview.net/forum?id=WZH7099tgfM) |

## Contributing


## Citation
