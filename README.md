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
| Pearl | Pearl: Personalizing large language model writing assistants with generation-calibrated retrievers |     Personalized Indexing     |    ACL 2024    | [[Link]](https://aclanthology.org/2024.customnlp4u-1.16.pdf) | 


</p>
</details>

<details><summary><b>2.2 Retrieve </b></summary>
<p>

 **Name** | **Title** |              **Personalized presentation**              | **Publication** |                **Paper Link**                | **Code Link**                |
|:---:|:---|:-------------------------------------------------------:|:---------------:|:---:|:--------------------------------------------:|
|            | Optimization Methods for  Personalizing Large Language Models through Retrieval Augmentation | Gradients based on personalized scores | SIGIR 2024         | [[Link]](https://dl.acm.org/doi/pdf/10.1145/3626772.3657783) |                                                              |
| MeMemo     | MeMemo: On-device Retrieval  Augmentation for Private and Personalized Text Generation | Privacy Protection              | SIGIR 2024 (short) | [[Link]](https://dl.acm.org/doi/pdf/10.1145/3626772.3657662) | [[Link]](https://github.com/poloclub/mememo)                           |
| LAPS       | Doing Personal LAPS:  LLM-Augmented Dialogue Construction for Personalized Multi-Session  Conversational Search | Personalized Dialogue            | SIGIR 2024         | [[Link]](https://dl.acm.org/doi/pdf/10.1145/3626772.3657815) | [[Link]](https://github.com/informagi/laps)                            |
|            | Partner Matters! An Empirical  Study on Fusing Personas for Personalized Response Selection in  Retrieval-Based Chatbots | Personalized Dialogue                                                | SIGIR 2021         | [[Link]](https://dl.acm.org/doi/pdf/10.1145/3404835.3462858) | [[Link]](https://github.com/JasonForJoy/Personalized-Response-Selection) |
| ERRA       | Explainable Recommendation with  Personalized Review Retrieval and Aspect Learning | Personalized Recommendation                                | ACL 2023           | [[Link]](https://arxiv.org/pdf/2306.12657)                   | [[Link]](https://github.com/Complex-data/ERRA)                         |
|            | RECAP: Retrieval-Enhanced  Context-Aware Prefix Encoder for Personalized Dialogue Response Generation | Personalized Dialogue  | ACL 2023           | [[Link]](https://arxiv.org/pdf/2306.07206)                   | [[Link]](https://github.com/isi-nlp/RECAP)                             |
| HEART      | HEART-felt Narratives:     Tracing Empathy and Narrative Style in Personal Stories with LLMs | Personalized Writing Style                  | EMNLP 2024         | [[Link]](https://arxiv.org/pdf/2405.17633)                   | [[Link]](https://github.com/mitmedialab/heartfelt-narratives-emnlp)    |
| OPPU       | Democratizing Large Language  Models via Personalized Parameter-Efficient Fine-tuning | Personalized Parameter Fine-tuning | EMNLP 2024         | [[Link]](https://arxiv.org/pdf/2402.04401)                   | [[Link]](https://github.com/TamSiuhin/OPPU)                            |
| LAPDOG     | Learning Retrieval Augmentation  for Personalized Dialogue Generation | Personalized Dialogue  | EMNLP 2023         | [[Link]](https://arxiv.org/pdf/2406.18847)                   | [[Link]](https://github.com/hqsiswiliam/LAPDOG)                        |
| UniMP      | Towards Unified Multi-Modal  Personalization: Large Vision-Language Models for Generative Recommendation  and Beyond | Personalized Recommendation | ICLR 2024          | [[Link]](https://arxiv.org/pdf/2403.10667)                   |                                                              |
|            | Personalized Language Generation  via Bayesian Metric Augmented Retrieval | Personalized Retrieval                                     | Arxiv              | [[Link]](https://openreview.net/pdf?id=n1LiKueC4F)           |                                                              |
|            | Leveraging Similar Users for  Personalized Language Modeling with Limited Data | Personalized Retrieval           | ACL 2022           | [[Link]](https://aclanthology.org/2022.acl-long.122.pdf)     |                                                              |
| UIA        | A Personalized Dense Retrieval  Framework for     Unified Information Access | Personalized Retrieval | SIGIR 2023         | [[Link]](https://dl.acm.org/doi/pdf/10.1145/3539618.3591626) | [[Link]](https://github.com/HansiZeng/UIA)                             |
| XPERT      | Personalized Retrieval over  Millions of Items               | Personalized Retrieval| SIGIR 2023         | [[Link]](https://dl.acm.org/doi/pdf/10.1145/3539618.3591749) | [[Link]](https://github.com/personalizedretrieval/xpert)               |
| DPSR       | Towards personalized and  semantic retrieval: An end-to-end solution for e-commerce search via  embedding learning | Personalized Retrieval     | SIGIR 2020         | [[Link]](https://dl.acm.org/doi/pdf/10.1145/3397271.3401446) |                                                              |
| PersonalTM | PersonalTM: Transformer Memory  for Personalized Retrieval   | Personalized Retrieval | SIGIR 2023 (short) | [[Link]](https://dl.acm.org/doi/pdf/10.1145/3539618.3592037) |                                                              |
|            | A zero attention model for  personalized product search      | Personalized Search     | CIKM 2019          | [[Link]](https://dl.acm.org/doi/pdf/10.1145/3357384.3357980) |                                                              |
| RTM        | Learning a Fine-Grained  Review-based Transformer Model for Personalized Product Search | Personalized Search | SIGIR 2021         | [[Link]](https://dl.acm.org/doi/pdf/10.1145/3404835.3462911) | [[Link]](https://github.com/kepingbi/ProdSearch)                       |

</p>
</details>

<details><summary><b>2.3 Post-Retrieve</b></summary>
<p>

 **Name** | **Title** |              **Personalized presentation**              | **Publication** |                **Paper Link**                | **Code Link**                |
|:---:|:---|:-------------------------------------------------------:|:---------------:|:---:|:--------------------------------------------:|
| LLM4Rerank | LLM4Rerank: LLM-based Auto-Reranking Framework for Recommendations|     Personalized Recommendation      |    WWW 2025    | [[Link]](https://arxiv.org/pdf/2406.12433v3) |

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
| Retrieve / Generate | TOPDIAL  | BLEU, F1, Succ | [[Link]](https://github.com/iwangjian/TopDial)|
  |       Retrieve / Generate    | LiveChat  | Recall,MRR |[[Link]]( https://github.com/gaojingsheng/LiveChat) |
  |          Retrieve / Generate     | PersonalityEvd  | ACC,Fluency, Coherence, Plausibility |   [[Link]](https://github.com/Lei-Sun-RUC/PersonalityEvd) |
  |            Retrieve / Generate   | Pchatbot  | BLEU, ROUGE, Dist, MRR | [[Link]](https://github.com/qhjqhj00/SIGIR2021-Pchatbot) |
|    Retrieve / Generate     | DuLemon  | PPL, BLUE, ACC,Precision,Recall,F1 | [[Link]](https://github.com/PaddlePaddle/Research/tree/master/NLP/ACL2022-DuLeMon) |
|    Retrieve / Generate    |  PersonalityEdit  |  ES, DD, Acc, TPEI, PAE|[[Link]](https://github.com/zjunlp/EasyEdit)|


## Contributing


## Citation
