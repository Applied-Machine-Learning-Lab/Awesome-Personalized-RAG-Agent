<a name="readme-top"></a>

<div align="center">
  <img src="./assets/logo.ico" alt="Logo" width="200">
  <h1 align="center">Awesome-Personalized-RAG-Agent</h1>
</div>

<p align="center">
    <a href=""><img alt="Static Badge" src="https://img.shields.io/badge/License-Apache_2.0-blue">
    </a>
    <a href='https://arxiv.org/abs/2504.10147'><img src='https://img.shields.io/badge/arXiv-2504.10147-b31b1b'></a>
</p>


## 📘 Overview

This is the official repository of the paper ["*A Survey of Personalization: From RAG to Agent*"](https://arxiv.org/abs/2504.10147), arXiv.

🎯 Our repository **Awesome-Personalized-RAG-Agent** is a curated collection of papers, resources, benchmarks, and datasets focused on **Personalized Retrieval-Augmented Generation (RAG)** and **personalized agentic RAG system**.

Personalization has become a cornerstone in modern AI systems, enabling customized interactions that reflect individual user preferences, contexts, and goals. Recent research has increasingly explored **RAG frameworks** and their evolution into **agent-based architectures**, aiming to improve user alignment and satisfaction.

This repository systematically categorizes personalization across the three core stages of RAG:
- **Pre-retrieval** (e.g., query rewriting and expansion),
- **Retrieval** (e.g., indexing, personalized reranking),
- **Generation** (e.g., using explicit or implicit user signals).

Beyond traditional RAG pipelines, we extend the scope to **Personalized LLM Agents**—systems enhanced with **agentic functionalities** such as dynamic user modeling, personalized planning, memory integration, and autonomous behavior.

## 🔥 News

<div class="scrollable">
    <ul>
      <li><strong>[2025, Apr 5]</strong>: &nbsp;🚀🚀 Our paper is now available on 
      <a href="https://" target="_blank">arXiv</a>, and the reading list is on 
      <a href="https://" target="_blank">GitHub Repo</a>.
    </ul>
</div>

## 🧭 Table of Contents

- [📃 Papers](#-papers)
  - [1. Pre-retrieval](#1-pre-retrieval)
    - [🔄 1.1 Query Rewriting](#-11-query-rewriting)
    - [➕ 1.2 Query Expansion](#-12-query-expansion)
    - [🛠️ 1.3 Other Query-related](#-13-other-query-related)
  - [2. Retrieval](#2-retrieval)
    - [🗂️ 2.1 Indexing](#-21-indexing)
    - [🔍 2.2 Retrieve](#-22-retrieve)
    - [🧹 2.3 Post-Retrieve](#-23-post-retrieve)
  - [3. Generation](#3-generation)
    - [🎯 3.1 Generation from Explicit Preference](#-31-generation-from-explicit-preference)
    - [🕵️ 3.2 Generation from Implicit Preference](#-32-generation-from-implicit-preference)
  - [4. Agentic RAG](#4-agentic-rag)
    - [🧠 4.1 Understanding](#-41-understanding)
    - [🗺️ 4.2 Planing and Execution](#-42-planing-and-execution)
    - [✍️ 4.3 Generation](#-43-generation)
- [📚 Datasets and Evaluation](#-datasets-and-evaluation)
- [🔗 Related Surveys and Repositories](#-related-surveys-and-repositories)
  - [📚 Surveys](#-surveys)
  - [📁 Repositories](#-repositories)
- [🚀 Contributing](#-contributing)
- [📌 Citation](#-citation)


## 📃  Papers

### 1. Pre-retrieval
#### 🔄 1.1 Query Rewriting

**Name**          | **Title**                                                                                                      | **Personalized presentation**                                                   | **Publication** |                **Paper Link**                | **Code Link**                               |
|:-----------------------:|:---------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------|:----------------|:---|:--------------------------------------------|
| Least-to-most Prompting | Least-to-Most Prompting Enables Complex Reasoning in Large Language Models                                     | Split by sub-query terms in different questions                                 | ICLR 2023       | [[Link]](https://openreview.net/forum?id=WZH7099tgfM) |
|                         | Personalized Search-based Query Rewrite System for Conversational AI                                           | Build a personalized index for each user                                        | ACL 2021        | [[Link]](https://aclanthology.org/2021.nlp4convai-1.17/) |
|      Agent4Ranking      | Agent4Ranking: Semantic Robust Ranking via Personalized Query Rewriting Using Multi-agent LLM                  | Use agents for efficient query rewriting.                                       | arXiv 2023      | [[Link]](https://arxiv.org/pdf/2312.15450) |
| Least-to-most Prompting | Query Rewriting in TaoBao Search                                                                               | A learning enhanced architecture based on “query retrieval−semantic relevance”. | CIKM 2022       | [[Link]](https://dl.acm.org/doi/abs/10.1145/3511808.3557068?casa_token=UdZBGUHNJQYAAAAA:eetXcV5SxHrcP-82xXpYJa2jR1-0eeKgaaRa_raoEQks4q2CwXUP_VseC_3bGE8qM1_dgQYnC32T) |
|         CLE-QR          | Learning to rewrite prompts for personalized text generation                                                   | Multistage framework for personalized rewrites.                                 | WWW 2024        | [[Link]](https://dl.acm.org/doi/abs/10.1145/3589334.3645408) |
|           CGF           | CGF: Constrained Generation Framework for Query Rewriting in Conversational AI                                 | Personalized prompt rewriting by using an LLM agent.                            | ACL 2022        | [[Link]](https://aclanthology.org/2022.emnlp-industry.48.pdf) |
|                         | RL-based Query Rewriting with Distilled LLM for online E-Commerce Systems                                      | Student model to rewrite query.                                                 | arXiv 2025      | [[Link]](https://arxiv.org/pdf/2501.18056) |
|          CoPS           | Cognitive Personalized Search Integrating Large Language Models with an Efficient Memory Mechanism             | Personalized query intent.                                                      | WWW 2024        | [[Link]](https://dl.acm.org/doi/abs/10.1145/3589334.3645482) |
|          BASES          | BASES: Large-scale Web Search User Simulation with Large Language Model based Agents                           | User simulation agent.                                                          | arXiv 2024      | [[Link]](https://arxiv.org/pdf/2402.17505v1) |
|         ERAGent         | ERAGent: Enhancing Retrieval-Augmented Language Models with Improved Accuracy, Efficiency, and Personalization | Collaorative module for query rewrite.                                          | arXiv 2024      | [[Link]](https://arxiv.org/pdf/2405.06683) |
|          PEARL          | PEARL: Personalizing Large Language Model Writing Assistants with Generation-Calibrated Retrievers             | Personlaized LLM for query write.                                               | arXiv 2024      | [[Link]](https://arxiv.org/pdf/2311.09180) |
|           FIG           | Graph Meets LLM: A Novel Approach to Collaborative Filtering for Robust Conversational Understanding           | Graph-based methods with LLMs to query rewrite                                  | arXiv 2023      | [[Link]](https://arxiv.org/pdf/2305.14449) |


#### ➕ 1.2 Query Expansion
| **Name** | **Title**                                                                  |                                                      **Personalized presentation**                                                      | **Publication**                                                           | **Paper Link**                                                                                                                                                        | **Code Link** |
|:-----------------------:|:---------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------|:--------------------------------------------------------------------------|:---|:--------------------------------------------|
|  PNQES   | Personalized Optimal Search in Local Query Expansion                       |                                                Search history, Latent semantic indexing                                                 | ACLCLP 2006                                                               | [[Link]](https://aclanthology.org/O06-1015.pdf)                                                                                                                       |               |
|          | Exploiting social relations for query expansion and result ranking         |                                                Friendship, Tagging                                                                      | ICDE Workshop 2008                                                        | [[Link]](https://ieeexplore.ieee.org/abstract/document/4498369/?casa_token=1kg_6Ae8v2QAAAAA:XYQuADKra_GcrnshEKYT3OFjvujhEOidXjxKa8Tls5AFlkiIetfKejLHv5Nfjxvmn5rMHogn) |               |
| Gossple  | Toward personalized query expansion                                        | Calculating the distance between users, constructing a personalized network that connects users with similar interests. Extracting tags | EuroSys Workshop 2009                                                     | [[Link]](https://dl.acm.org/doi/abs/10.1145/1578002.1578004?casa_token=xrBTIRsHVbgAAAAA:ZQl0ReJup7-48PiKIM_JNZ2ioWKvrIunR_4arW2ULoSDgG8En-oGs6brf0NUzonXb1f8E6kpb7oR) |               |
|          | Social tagging in query expansion: A new way for personalized web search   |                                                          User Interest Tagging                                                          | CSE 2009                                                                  | [[Link]](https://ieeexplore.ieee.org/abstract/document/5283040/?casa_token=29tFHyo36rsAAAAA:cUKGboQJ_6dSOyw2Jt_mc2yak8G2oP3piyLZvjNGbMd0WMLuBwfkIIGT3_HbE3_0-1CMrFYW) |               |
|  SoQuES  | personalized social query expansion using social bookmarking systems       |                                     Extract user tag behavior to build personalized user profiles.                                      | SIGIR 2011                                                                | [[Link]](https://dl.acm.org/doi/abs/10.1145/2009916.2010075?casa_token=y-b4ZLwha4YAAAAA:VK9Fmq-iRQP1jgU-hBODrlIzTbpD88wGCmYpSBUtQMvYnm7dAiMOiIjaYe7NMGX_Vipjfxzgi4bL) |               |
|          | Improving search via personalized query expansion using social media       |                                                                 Tagging                                                                 | Information retrieval 2012                                                | [[Link]](https://link.springer.com/article/10.1007/s10791-012-9191-2)                                                                                                 |               |
|          | Axiomatic term-based personalized query expansion using bookmarking system |                                                       Bookmarking, Social Network                                                       | International Conference on Database and Expert Systems Applications 2016 | [[Link]](https://link.springer.com/chapter/10.1007/978-3-319-44406-2_17)                                                                                              |               |
|  WE-LM   | personalized query expansion utilizing multi-relationalsocial data         |                                                                 Tagging                                                                 | SMAP Workshop 2017                                                        | [[Link]](https://ieeexplore.ieee.org/abstract/document/8022669/)                                                                                                      |               |
|   PSQE   | personalized social query expansion using social annotations               |                                                          User Interest Tagging                                                          | Transactions on Large-Scale Data-and Knowledge-Centered Systems XL 2019   | [[Link]](https://link.springer.com/chapter/10.1007/978-3-662-58664-8_1)                                                                                               |               |
|  PQEWC   | Personalized Query Expansion with Contextual Word Embeddings               |                             Employing topic modeling on user texts and dynamically selecting relevant words                             | TOIS 2023                                                                 | [[Link]](https://dl.acm.org/doi/abs/10.1145/3624988)                                                                                                                  |               |

#### 🛠️ 1.3 Other Query-related
| **Name** | **Title**                                                                                   | **Personalized presentation**                                                                                    | **Publication** |                        **Paper Link**                        | **Code Link** |
|:--------:|:--------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------|:----------------|:---|:--------------------------------------------|
|   PSQE   | PSQE: Personalized Semantic Query Expansion for user-centric query disambiguation           | Leveraging synthetic user profiles built from Wikipedia articles, training word2vec embeddings on these profiles |                 | [[Link]](https://www.researchsquare.com/article/rs-4178030/latest) |               |
| Bobo     | Utilizing user-input contextual terms for query disambiguation                              |  contextual terms                                                                                                |   Coling 2010   |       [[Link]](https://aclanthology.org/C10-2038.pdf)        |               |
|          | Personalized Query Auto-Completion Through a Lightweight Representation of the User Context | Learning embeddings from the user’s recent queries                                                               |   arXiv 2019    |          [[Link]](https://arxiv.org/abs/1905.01386)          |               |


### 2. Retrieval
#### 🗂️ 2.1 Indexing
  
   **Name**     | **Title**                                                                                          | **Personalized presentation**                           | **Publication** |                **Paper Link**                | **Code Link**                                              |
|:-------------:|:---------------------------------------------------------------------------------------------------|:--------------------------------------------------------|:----------------|:---|:-----------------------------------------------------------|
|     Pearl     | Pearl: Personalizing large language model writing assistants with generation-calibrated retrievers | Personalized Indexing                                   | ACL 2024        | [[Link]](https://aclanthology.org/2024.customnlp4u-1.16.pdf) | |
| KG-Retrieve   | KG-Retriever: Efficient Knowledge Indexing for Retrieval-Augmented Large Language Models           | Hierarchical Index Graph                                | arXiv 2024      | [[Link]](https://arxiv.org/pdf/2412.05547) |  |
|    EMG-RAG    | Crafting Personalized Agents through Retrieval-Augmented Generation on Editable Memory Graphs      | Editable Knowledge Graph for Indexing                   | EMNLP 2024      | [[Link]](https://arxiv.org/pdf/2409.19401) | |
|   PGraphRAG   | Personalized Graph-Based Retrieval for Large Language Models                                       | User-centric Knowledge Graphs for Personalized Indexing | arXiv 2025        | [[Link]](https://arxiv.org/pdf/2501.02157) |[[Link]](https://github.com/PGraphRAG-benchmark/PGraphRAG) |


#### 🔍 2.2 Retrieve

 **Name** | **Title**                                                                                                                |                    **Personalized presentation**                    | **Publication** |                **Paper Link**                |                              **Code Link**                               |
|:-----------------------:|:---------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------|:----------------|:---|:--------------------------------------------|
|            | Optimization Methods for  Personalizing Large Language Models through Retrieval Augmentation                             |               Gradients based on personalized scores                | SIGIR 2024         | [[Link]](https://dl.acm.org/doi/pdf/10.1145/3626772.3657783) |                                                                          |
| MeMemo     | MeMemo: On-device Retrieval  Augmentation for Private and Personalized Text Generation                                   |                         Privacy Protection                          | SIGIR 2024 (short) | [[Link]](https://dl.acm.org/doi/pdf/10.1145/3626772.3657662) |               [[Link]](https://github.com/poloclub/mememo)               |
| LAPS       | Doing Personal LAPS:  LLM-Augmented Dialogue Construction for Personalized Multi-Session  Conversational Search          |                        Personalized Dialogue                        | SIGIR 2024         | [[Link]](https://dl.acm.org/doi/pdf/10.1145/3626772.3657815) |               [[Link]](https://github.com/informagi/laps)                |
|            | Partner Matters! An Empirical  Study on Fusing Personas for Personalized Response Selection in  Retrieval-Based Chatbots |                        Personalized Dialogue                        | SIGIR 2021         | [[Link]](https://dl.acm.org/doi/pdf/10.1145/3404835.3462858) | [[Link]](https://github.com/JasonForJoy/Personalized-Response-Selection) |
| ERRA       | Explainable Recommendation with  Personalized Review Retrieval and Aspect Learning                                       |                     Personalized Recommendation                     | ACL 2023           | [[Link]](https://arxiv.org/pdf/2306.12657)                   |              [[Link]](https://github.com/Complex-data/ERRA)              |
|            | RECAP: Retrieval-Enhanced  Context-Aware Prefix Encoder for Personalized Dialogue Response Generation                    |                        Personalized Dialogue                        | ACL 2023           | [[Link]](https://arxiv.org/pdf/2306.07206)                   |                [[Link]](https://github.com/isi-nlp/RECAP)                |
| HEART      | HEART-felt Narratives:     Tracing Empathy and Narrative Style in Personal Stories with LLMs                             |                     Personalized Writing Style                      | EMNLP 2024         | [[Link]](https://arxiv.org/pdf/2405.17633)                   |   [[Link]](https://github.com/mitmedialab/heartfelt-narratives-emnlp)    |
| OPPU       | Democratizing Large Language  Models via Personalized Parameter-Efficient Fine-tuning                                    |                 Personalized Parameter Fine-tuning                  | EMNLP 2024         | [[Link]](https://arxiv.org/pdf/2402.04401)                   |               [[Link]](https://github.com/TamSiuhin/OPPU)                |
| LAPDOG     | Learning Retrieval Augmentation  for Personalized Dialogue Generation                                                    |                        Personalized Dialogue                        | EMNLP 2023         | [[Link]](https://arxiv.org/pdf/2406.18847)                   |             [[Link]](https://github.com/hqsiswiliam/LAPDOG)              |
| UniMP      | Towards Unified Multi-Modal  Personalization: Large Vision-Language Models for Generative Recommendation  and Beyond     |                     Personalized Recommendation                     | ICLR 2024          | [[Link]](https://arxiv.org/pdf/2403.10667)                   |                                                                          |
|            | Personalized Language Generation  via Bayesian Metric Augmented Retrieval                                                |                       Personalized Retrieval                        | arXiv              | [[Link]](https://openreview.net/pdf?id=n1LiKueC4F)           |                                                                          |
|            | Leveraging Similar Users for  Personalized Language Modeling with Limited Data                                           |                       Personalized Retrieval                        | ACL 2022           | [[Link]](https://aclanthology.org/2022.acl-long.122.pdf)     |                                                                          |
| UIA        | A Personalized Dense Retrieval  Framework for     Unified Information Access                                             |                       Personalized Retrieval                        | SIGIR 2023         | [[Link]](https://dl.acm.org/doi/pdf/10.1145/3539618.3591626) |                [[Link]](https://github.com/HansiZeng/UIA)                |
| XPERT      | Personalized Retrieval over  Millions of Items                                                                           |                       Personalized Retrieval                        | SIGIR 2023         | [[Link]](https://dl.acm.org/doi/pdf/10.1145/3539618.3591749) |         [[Link]](https://github.com/personalizedretrieval/xpert)         |
| DPSR       | Towards personalized and  semantic retrieval: An end-to-end solution for e-commerce search via  embedding learning       |                       Personalized Retrieval                        | SIGIR 2020         | [[Link]](https://dl.acm.org/doi/pdf/10.1145/3397271.3401446) |                                                                          |
| PersonalTM | PersonalTM: Transformer Memory  for Personalized Retrieval                                                               |                       Personalized Retrieval                        | SIGIR 2023 (short) | [[Link]](https://dl.acm.org/doi/pdf/10.1145/3539618.3592037) |                                                                          |
|            | A zero attention model for  personalized product search                                                                  |                         Personalized Search                         | CIKM 2019          | [[Link]](https://dl.acm.org/doi/pdf/10.1145/3357384.3357980) |                                                                          |
| RTM        | Learning a Fine-Grained  Review-based Transformer Model for Personalized Product Search                                  |                         Personalized Search                         | SIGIR 2021         | [[Link]](https://dl.acm.org/doi/pdf/10.1145/3404835.3462911) |             [[Link]](https://github.com/kepingbi/ProdSearch)             |

#### 🧹 2.3 Post-Retrieve

**Name**     | **Title**                                                                                              | **Personalized presentation**      | **Publication** |                **Paper Link**                | **Code Link**                                                      |
|:---------------:|:-------------------------------------------------------------------------------------------------------|:-----------------------------------|:----------------|:---|:-------------------------------------------------------------------|
|    PersonaRAG   | PersonaRAG: Enhancing Retrieval-Augmented Generation Systems with User-Centric Agents                  | User-centric Agent for Ranking     | arXiv 2024      | [Link](https://arxiv.org/abs/2407.09394) | [Link](https://github.com/padas-lab-de/ir-rag-sigir24-persona-rag) |
|                 | Improving RAG with LoRA Finetuning for Persona Text Generation                                         | Personalized Re-ranking            |                 | [Link](http://ftp.pdmi.ras.ru/pub/publicat/znsl/v540/p162.pdf) |                                              |
|    UniMS-RAG    | UniMS-RAG: A Unified Multi-source Retrieval-Augmented Generation for Personalized Dialogue Systems     | Personalized Re-ranking            | arXiv 2024      | [Link](https://arxiv.org/pdf/2401.13256) |                                                                    |
|                 | Learning to Rank for Multiple Retrieval-Augmented Models through Iterative Utility Maximization        | Personalized Ranking               | arXiv 2024      | [Link](https://arxiv.org/pdf/2410.09942) | [Link](https://github.com/alirezasalemi7/uRAG)                     |
|                 | Rehearse With User: Personalized Opinion Summarization via Role-Playing Based on Large Language Models | Personalized Opinion Summarization | arXiv 2025      | [Link](https://arxiv.org/pdf/2503.00449) |                                                                    |
| AutoCompressors | Adapting Language Models to Compress Contexts                                                          | Contextual Embedding Compression   | EMNLP 2023      | [Link](https://arxiv.org/pdf/2305.14788) | [Link](https://github.com/princeton-nlp/AutoCompressors)           |
|     FIT-RAG     | FIT-RAG: Black-Box RAG with Factual Information and Token Reduction                                    | Tokens Minization within RAG       | ToIS 2025       | [Link](https://arxiv.org/pdf/2403.14374) |                                                                    |
### 3. Generation

#### 🎯 3.1 Generation from Explicit Preference
  
**Name**       | **Title**                                                                                                |  **Personalized presentation**  |  **Publication**  |                                                                               **Paper Link**                                                                               |                                  **Code Link**                                  |
|:-----------------------:|:---------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------|:----------------|:---|:--------------------------------------------|
|         P2          | Evaluating and inducing personality in pre-trained language models                                       |         Role Playing            | NeurIPS 2023      |                         [[Link]](https://proceedings.neurips.cc/paper_files/paper/2023/file/21f7b745f73ce0d1f9bcea7f40b1388e-Paper-Conference.pdf)                         |[[Link]](https://sites.google.com/view/machinepersonality)                       |
|      OpinionQA      | Whose opinions do language models reflect?                                                               |          Role Playing           |     ICML 2023     |                                                 [[Link]](https://proceedings.mlr.press/v202/santurkar23a/santurkar23a.pdf)                                                 |               [[Link]](https://github.com/tatsu-lab/opinions_qa)                |
| Character Profiling | Evaluating Character Understanding of Large Language Models via Character Profiling from Fictional Works |          Role Playing           |     ICML 2023     |                                                                 [[Link]](https://arxiv.org/pdf/2404.12726)                                                                 |           [[Link]](https://github.com/Joanna0123/character_profiling)           |
|                     | Do LLMs Understand User Preferences? Evaluating LLMs On User Rating Prediction                           |   Personalized Recommendation   |       arXiv       |                                                                 [[Link]](https://arxiv.org/pdf/2305.06474)                                                                 |                                                                                 |
|       Cue-CoT       | Cue-CoT: Chain-of-thought prompting for responding to in-depth dialogue questions with LLMs              |      Personalized Dialogue      |    EMNLP 2023     |                                                        [[Link]](https://aclanthology.org/2023.findings-emnlp.806/)                                                         |                 [[Link]](https://github.com/ruleGreen/Cue-CoT)                  |
|        TICL         | Tuning-Free Personalized Alignment via Trial-Error-Explain In-Context Learning                           |  Personalized Text Generation   |       arXiv       |                                                                 [[Link]](https://arxiv.org/pdf/2502.08972)                                                                 |                 [[Link]](https://github.com/ruleGreen/Cue-CoT)                  |
|         GPG         | Guided Profile Generation Improves Personalization with LLMs                                             |  Personalized Text Generation   |       arXiv       |                                                                 [[Link]](https://arxiv.org/pdf/2409.13093)                                                                 |                                                                                 |
|                     | Integrating Summarization and Retrieval for Enhanced Personalization via Large Language Models           |  Personalized Text Generation   |       arXiv       |                                                                 [[Link]](https://arxiv.org/pdf/2310.20081)                                                                 |                                                                                 |
|     LLMTreeRec      | LLMTreeRec: Unleashing the Power of Large Language Models for Cold-Start Recommendations                 |   Personalized Recommendation   |    COLING 2025    |                                                          [[Link]](https://aclanthology.org/2025.coling-main.59/)                                                           |      [[Link]](https://github.com/Applied-Machine-Learning-Lab/LLMTreeRec)       |
|     Matryoshka      | MATRYOSHKA: Learning To Drive Black-Box LLMS With LLMS                                                   |  Personalized Text Generation   |       arXiv       |                                                                 [[Link]](https://arxiv.org/pdf/2410.20749)                                                                 |                                                                                 |
|                     | Learning to rewrite prompts for personalized text generation                                             |  Personalized Text Generation   |     WWW 2024      |                                                        [[Link]](https://dl.acm.org/doi/pdf/10.1145/3589334.3645408)                                                        |                                                                                 |
|       RecGPT        | RecGPT: Generative Pre-training for Text-based Recommendation                                            |   Personalized Recommendation   |     ACL 2024      |                                                           [[Link]](https://aclanthology.org/2024.acl-short.29/)                                                            |                [[Link]](https://github.com/VinAIResearch/RecGPT)                |
|      PEPLER-D       | Personalized prompt learning for explainable recommendation                                              |   Personalized Recommendation   |     TOIS 2023     |                                                            [[Link]](https://dl.acm.org/doi/pdf/10.1145/3580488)                                                            |                 [[Link]](https://github.com/lileipisces/PEPLER)                 |
|        SGPT         | Unlocking the potential of prompt-tuning in bridging generalized and personalized federated learning     | Personalized Federated Learning |     CVPR 2024     | [[Link]](https://openaccess.thecvf.com/content/CVPR2024/papers/Deng_Unlocking_the_Potential_of_Prompt-Tuning_in_Bridging_Generalized_and_Personalized_CVPR_2024_paper.pdf) |                    [[Link]](https://github.com/ubc-tea/SGPT)                    |
|        PFCL         | Personalized federated continual learning via multi-granularity prompt                                   | Personalized Federated Learning |     KDD 2024      |                                                        [[Link]](https://dl.acm.org/doi/abs/10.1145/3637528.3671948)                                                        |               [[Link]](https://github.com/SkyOfBeginning/FedMGP)                |

#### 🕵️ 3.2 Generation from Implicit Preference

**Name**       | **Title**                                                                                                             |     **Personalized presentation**     |                **Publication**                 |                         **Paper Link**                          |                            **Code Link**                            |
|:-----------------------:|:---------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------|:----------------|:---|:--------------------------------------------|
|       PLoRA       | Personalized LoRA for Human-Centered Text Understanding                                                               |    Personalized Text Understanding    |                   AAAI 2024                    | [[Link]](https://arxiv.org/pdf/2403.06208)                      | [[Link]](https://github.com/yoyo-yun/PLoRA)                         |
|       LM-P        | Personalized Large Language Models                                                                                    |       Personalized Fine-tuning        |          SENTIRE 2024 (ICDM Workshop)          |           [[Link]](https://arxiv.org/pdf/2402.09269)            |         [[Link]](https://github.com/Rikain/llm-finetuning)          |
|       MiLP        | Personalized LLM Response Generation with Parameterized User Memory Injection                                         |     Personalized Text Generation      |                     arXiv                      |           [[Link]](https://arxiv.org/pdf/2404.03565)            |            [[Link]](https://github.com/MatthewKKai/MiLP)            |
|       OPPU        | Democratizing Large Language Models via Personalized Parameter-Efficient Fine-tuning                                  |     Personalized Text Generation      |                   EMNLP 2024                   |   [[Link]](https://aclanthology.org/2024.emnlp-main.372.pdf)    |             [[Link]](https://github.com/TamSiuhin/OPPU)             |
|      PER-PCS      | PERSONALIZED PIECES: Efficient Personalized Large Language Models through Collaborative Efforts                       |     Personalized Text Generation      |                   EMNLP 2024                   |   [[Link]](https://aclanthology.org/2024.emnlp-main.371.pdf)    |           [[Link]](https://github.com/TamSiuhin/Per-Pcs)            |
|    Review-LLM     | Review-LLM: Harnessing Large Language Models for Personalized Review Generation                                       |    Personalized Review Generation     |                     arXiv                      |           [[Link]](https://arxiv.org/pdf/2407.07487)            |                                                                     |
|  UserIdentifier   | UserIdentifier: Implicit User Representations for Simple and Effective Personalized Sentiment Analysis                |    Personalized Text Understanding    |                   NAACL 2022                   |   [[Link]](https://aclanthology.org/2022.naacl-main.252.pdf)    |                                                                     |
|    UserAdapter    | UserAdapter: Few-Shot User Learning in Sentiment Analysis                                                             |    Personalized Text Understanding    |               ACL Fingdings 2021               |  [[Link]](https://aclanthology.org/2021.findings-acl.129.pdf)   |                                                                     |
|       HYDRA       | HYDRA: Model Factorization Framework for Black-Box LLM Personalization                                                | Personalized Reranking and Generation |                  NeurIPS 2024                  |           [[Link]](https://arxiv.org/pdf/2406.02888)            |            [[Link]](https://github.com/night-chen/HYDRA)            |
|     PocketLLM     | PocketLLM: Enabling On-Device Fine-Tuning for Personalized LLMs                                                       |     Personalized Text Generation      |         PrivateNLP 2024 (ACL Workshop)         |   [[Link]](https://aclanthology.org/2024.privatenlp-1.10.pdf)   |                                                                     |
|     CoGenesis     | CoGenesis: A Framework Collaborating Large and Small Language Models for Secure Context-Aware Instruction Following   |     Personalized Text Generation      |                    ACl 2024                    |           [[Link]](https://arxiv.org/pdf/2403.03129)            |         [[Link]](https://github.com/TsinghuaC3I/CoGenesis)          |
|      P-RLHF       | P-RLHF: Personalized Language Modeling from Personalized Human Feedback                                               |     Personalized Text Generation      |                     arXiv                      |           [[Link]](https://arxiv.org/pdf/2402.05133)            |      [[Link]](https://github.com/HumainLab/Personalized_RLHF)       |
|      P-SOUPS      | Personalized Soups: Personalized Large Language Model Alignment via Post-hoc Parameter Merging                        |     Personalized Text Generation      | Adaptive Foundation Models 2024 (NeurIPS 2024) |       [[Link]](https://openreview.net/pdf?id=EMrnoPRvxe)        |             [[Link]](https://github.com/joeljang/RLPHF)             |
|        PAD        | PAD: Personalized Alignment of LLMs at Decoding-Time                                                                  |     Personalized Text Generation      |                   ICLR 2025                    |           [[Link]](https://arxiv.org/pdf/2410.04070)            |           [[Link]](https://github.com/zjuruizhechen/PAD)            |
|      REST-PG      | Reasoning-enhanced self-training for long-form personalized Text Generation                                           |     Personalized Text Generation      |                     arXiv                      |           [[Link]](https://arxiv.org/pdf/2501.04167)            |                                                                     |
|                   | Optimization Methods for Personalizing Large Language Models through Retrieval Augmentation                           | Personalized Retrieval and Generation |                   SIGIR 2024                   |  [[Link]](https://dl.acm.org/doi/pdf/10.1145/3626772.3657783)   |                                                                     |
|   RewriterSlRl    | Learning to Rewrite Prompts for Personalized Text Generation                                                          |     Personalized Text Generation      |                    WWW 2024                    |           [[Link]](https://arxiv.org/pdf/2310.00152)            |                                                                     |
|                   | Reinforcement learning for optimizing rag for domain chatbots                                                         |         Personalized Chatbot          |          RL+LLMs 2024 (AAAI Workshop)          |           [[Link]](https://arxiv.org/pdf/2401.06800)            |                                                                     |


### 4. Agentic RAG

#### 🧠 4.1 Understanding

|   **Name**    | **Title**                                                                                           | **Personalized Presentation**                                            | **Publication**       | **Paper Link**                                                 | **Code Link**                                               |
|:-------------:|:----------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------|:----------------------|:---------------------------------------------------------------|:------------------------------------------------------------|
|               | Penetrative AI: Making LLMs Comprehend the Physical World                                           | User interaction with physical-world data via sensors                    | ACL Findings 2024     | [Link](https://arxiv.org/abs/2310.09605)                       | [Link](https://hkustwands.github.io/penetrative-ai/)        |
|               | Conversational Health Agents: A Personalized LLM-Powered Agent Framework                            | Personalized healthcare support via health data and knowledge base       | arXiv 2024            | [Link](https://arxiv.org/pdf/2310.02374)                       | [Link](https://github.com/Institute4FutureHealth/CHA)       |
|    RoleLLM    | RoleLLM: Benchmarking, eliciting, and enhancing role-playing abilities of large language models     | LLMs role-playing characters                                             | arXiv 2024            | [Link](https://arxiv.org/pdf/2310.00746)                       |                                                             |
| Character-llm | Character-llm: A trainable agent for role-playing                                                   | LLM character simulation with memory/personality                         | EMNLP 2023            | [Link](https://arxiv.org/pdf/2310.10158)                       | [Link](https://github.com/choosewhatulike/trainable-agents) |
|  InCharacter  | Evaluating personality fidelity in role-playing agents through psychological interviews             | Psychological evaluation of role-play agents                             | ACL 2024              | [Link](https://aclanthology.org/2024.acl-long.102/)            | [Link](https://incharacter.github.io/)                      |
|  Socialbench  | Socialbench: Sociality evaluation of role-playing conversational agents                             | Benchmarking social interaction ability in role-playing                  | ACL Findings 2024     | [Link](https://arxiv.org/pdf/2403.13679)                       | [Link](https://github.com/X-PLUG/SocialBench)               |
|    MMrole     | Mmrole: A comprehensive framework for developing and evaluating multimodal role-playing agents      | Developing and Evaluating Multimodal Role-Playing Agents                 | ICLR 2025             | [Link](https://arxiv.org/abs/2408.04203)                       | [Link](https://github.com/YanqiDai/MMRole)                  |
|               | Enhancing role-playing language models with personality-indicative data                             | Leveraging psychological scale questions for agent personality modelling | EMNLP Findings 2024   | [Link](https://arxiv.org/pdf/2406.18921)                       | [Link](https://github.com/alienet1109/RolePersonality)      |                                                                |
|               | Enabling Conversational Interaction with Mobile UI using Large Language Models                      | Mobile UI interaction with LLMs                                          | CHI 2023              | [Link](https://arxiv.org/abs/2209.08655)                       |                                                             |
| CharacterEval | A chinese benchmark for role-playing conversational agent evaluation                                | Chinese dataset for role-playing agent evaluation                        | arXiv 2024            | [Link](https://arxiv.org/abs/2401.01275)                       | [Link](https://github.com/morecry/CharacterEval)            |
|     Neeko     | Neeko: Leveraging dynamic lora for efficient multi-character role-playing agent                     | Efficient multi-character simulation                                     | EMNLP 2024            | [Link](https://arxiv.org/pdf/2402.13717)                       | [Link](https://github.com/weiyifan1023/Neeko)               |


#### 🗺️ 4.2 Planing and Execution

|       **Name**        | **Title**                                                                                                         | **Personalized Presentation**                                        | **Publication**       | **Paper Link**                                                 | **Code Link**                                               |
|:---------------------:|:------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------|:----------------------|:---------------------------------------------------------------|:------------------------------------------------------------|
| EMG-RAG               | Crafting Personalized Agents through Retrieval-Augmented Generation on Editable Memory Graphs                     | Personalized assistant using user memory                             | EMNLP 2024            | [Link](https://arxiv.org/abs/2409.19401)                       |                                                             |
|                       | Generative agents: Interactive simulacra of human behavior                                                        | Simulation of social and individual behaviors in Stanford Town       | UIST 2023             | [Link](https://arxiv.org/abs/2304.03442)                       |                                                             |
|                       | Conversational Health Agents: A Personalized LLM-Powered Agent Framework                                          | Personalized healthcare support via health data and knowledge base   | arXiv 2024            | [Link](https://arxiv.org/pdf/2310.02374)                       | [Link](https://github.com/Institute4FutureHealth/CHA)       |
|                       | User Behavior Simulation with Large Language Model based Agents                                                   | Simulation of user behavior in recommender systems                   | arXiv 2024            | [Link](https://arxiv.org/abs/2306.02552)                       |                                                             |
|                       | Personal Large Language Model Agents: A Case Study on Tailored Travel Planning                                    | Tailored travel based on user preferences                            | EMNLP 2024            | [Link](https://aclanthology.org/2024.emnlp-industry.37/)       |                                                             |
|      PersonalWAB      | Large Language Models Empowered Personalized Web Agents                                                           | User profile + web action optimization                               | WWW 2025 Oral         | [Link](https://arxiv.org/pdf/2410.17236)                       |                                                             |
|                       | Voyager: An Open-Ended Embodied Agent with Large Language Models                                                  | Minecraft agent simulation and skill learning                        | TMLR 2024             | [Link](https://arxiv.org/abs/2305.16291)                       | [Link](https://voyager.minedojo.org/)                       |
|                       | MEMORYLLM: Towards Self-Updatable Large Language Models                                                           | Self-evolution via memory update                                     | ICML 2024             | [Link](https://arxiv.org/pdf/2402.04624)                       |                                                             |
|                       | Bootstrap Your Own Skills: Learning to Solve New Tasks with Large Language Model Guidance                         | Robotic arm learning to manipulate objects                           | CoRL 2023             | [Link](https://arxiv.org/pdf/2310.10021)                       | [Link](https://clvrai.github.io/boss/)                      |
|                       | Enabling Conversational Interaction with Mobile UI using Large Language Models                                    | Mobile UI interaction with LLMs                                      | CHI 2023              | [Link](https://arxiv.org/abs/2209.08655)                       |                                                             |
|                       | Penetrative AI: Making LLMs Comprehend the Physical World                                                         | User interaction with physical-world data via sensors                | ACL Findings 2024     | [Link](https://arxiv.org/abs/2310.09605)                       | [Link](https://hkustwands.github.io/penetrative-ai/)        |
|                       | Language models as zero-shot planners: Extracting actionable knowledge for embodied agents                        | Task planning in virtual environments                                | ICML 2022             | [Link](https://arxiv.org/pdf/2201.07207)                       | [Link](https://wenlong.page/language-planner/)              |
|        MetaGPT        | Metagpt: Meta programming for multi-agent collaborative framework                                                 | Multi-agent collaboration for complex tasks                          | ICLR 2024             | [Link](https://arxiv.org/pdf/2308.00352)                       |                                                             |
|       OKR-Agent       | Agents meet okr: An object and key results driven agent system with hierarchical self-collaboration and self-evaluation | Role-assigned agents solving creative tasks                          | arXiv 2023            | [Link](https://arxiv.org/pdf/2311.16542)                       | [Link](https://okr-agent.github.io/)                        |


#### ✍️ 4.3 Generation 



|      **Name**       | **Title**                                                                                         | **Personalized Presentation**                                      | **Publication**         | **Paper Link**                                               | **Code Link**                                               |
|:-------------------:|:--------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------|:------------------------|:-------------------------------------------------------------|:------------------------------------------------------------|
| Character-llm       | Character-llm: A trainable agent for role-playing                                                 | LLM character simulation with memory/personality                   | EMNLP 2023              | [Link](https://arxiv.org/pdf/2310.10158)                     | [Link](https://github.com/choosewhatulike/trainable-agents) |
|                     | Investigating the Personality Consistency in Quantized Role-Playing Dialogue Agents               | Robustness and consistency in constrained setups                   | EMNLP 2024              | [Link](https://aclanthology.org/2024.emnlp-industry.19/)     |                                                             |
| MMrole              | Mmrole: A comprehensive framework for developing and evaluating multimodal role-playing agents    | Developing and Evaluating Multimodal Role-Playing Agents           | ICLR 2025               | [Link](https://arxiv.org/abs/2408.04203)                     | [Link](https://github.com/YanqiDai/MMRole)                  |
| RoleLLM             | RoleLLM: Benchmarking, eliciting, and enhancing role-playing abilities of large language models   | LLMs role-playing characters                                       | arXiv 2024              | [Link](https://arxiv.org/pdf/2310.00746)                     |                                                             |
|                     | Enhancing role-playing language models with personality-indicative data                           | Personality-focused character simulation                           | EMNLP 2024 Findings     | [Link](https://arxiv.org/pdf/2406.18921)                     | [Link](https://github.com/alienet1109/RolePersonality)      |
| InCharacter         | Evaluating personality fidelity in role-playing agents through psychological interviews           | Psychological evaluation of role-play agents                       | ACL 2024                | [Link](https://aclanthology.org/2024.acl-long.102/)          | [Link](https://incharacter.github.io/)                      |
| Socialbench         | Socialbench: Sociality evaluation of role-playing conversational agents                           | Benchmarking social interaction ability in role-playing            | ACL Findings 2024       | [Link](https://arxiv.org/pdf/2403.13679)                     | [Link](https://github.com/X-PLUG/SocialBench)               |


## 📚 Datasets and Evaluation

|       **Field**        | **Dataset**                       | **Metrics**                                                       | **Link**                                                                          |
|:----------------------:|:----------------------------------|:------------------------------------------------------------------|:----------------------------------------------------------------------------------|
|    Query Rewriting     | SCAN                              | Accuracy                                                          | [Link]()                                                                          |
|    Query Rewriting     | Robust04                          | Accuracy                                                          | [Link]()                                                                          |
|    Query Rewriting     | Avocado Research Email Collection | BLEU, ROUGE                                                       | [Link]()                                                                          |
|    Query Rewriting     | Amazon Review                     | BLEU, ROUGE                                                       | [Link]()                                                                          |
|    Query Rewriting     | Reddit Comments                   | BLEU, ROUGE                                                       | [Link]()                                                                          |
|    Query Rewriting     | Amazon ESCI Dataset               | EM, ROUGE-L, XEntropy                                             | [Link]()                                                                          |
|    Query Rewriting     | AOL                               | MAP, MRR, P@1                                                     | [Link]()                                                                          |
|    Query Rewriting     | WARRIORS                          | MRR, NDCG                                                         | [Link]()                                                                          |
|    Query Rewriting     | AITA WORKSM                       | Macro-F1, BS-F1                                                   | [Link]()                                                                          |
|    Query Rewriting     | PIP                               | PMS, Image-Align, ROUGE                                           | [Link]()                                                                          |
|    Query Expansion     | Personalized Results Re-Ranking   | MAP, MRR, NDCG, RBP                                               | [Link]()                                                                          |
|    Query Expansion     | del.icio.us                       | Precision, MAP, MRR, Recall                                       | [Link]()                                                                          |
|    Query Expansion     | Flickr                            | Precision                                                         | [Link]()                                                                          |
|    Query Expansion     | CiteULike                         | Recall, MAP, MRR                                                  | [Link]()                                                                          |
|    Query Expansion     | LRDP                              | Precision, Recall, F1                                             | [Link]()                                                                          |
|    Query Expansion     | Delicious                         | MAP, MRR                                                          | [Link]()                                                                          |
|    Query Expansion     | Flickr                            | MAP, MRR                                                          | [Link]()                                                                          |
|    Query Expansion     | Bibsonomy                         | MAP, Precision, PQEC, Prof-overlap                                | [Link]()                                                                          |
|  Other Query-related   | Wikipedia                         | Precision, Recall                                                 | [Link]()                                                                          |
| Retrieval / Generation | TOPDIAL                           | BLEU, F1, Success Rate                                            | [Link](https://github.com/iwangjian/TopDial)                                      |
| Retrieval / Generation | LiveChat                          | Recall, MRR                                                       | [Link](https://github.com/gaojingsheng/LiveChat)                                  |
| Retrieval / Generation | PersonalityEvd                    | Accuracy, Fluency, Coherence, Plausibility                        | [Link](https://github.com/Lei-Sun-RUC/PersonalityEvd)                             |
| Retrieval / Generation | Pchatbot                          | BLEU, ROUGE, Distinct, MRR                                        | [Link](https://github.com/qhjqhj00/SIGIR2021-Pchatbot)                            |
| Retrieval / Generation | DuLemon                           | Perplexity, BLEU, Accuracy, Precision, Recall, F1                 | [Link](https://github.com/PaddlePaddle/Research/tree/master/NLP/ACL2022-DuLeMon)  |
| Retrieval / Generation | PersonalityEdit                   | ES, DD, Accuracy, TPEI, PAE                                       | [Link](https://github.com/zjunlp/EasyEdit)                                        |
|       Generation       | LaMP                              | Accuracy, F1, MAE, RMSE, ROUGE                                    | [Link](https://lamp-benchmark.github.io/)                                         |
|       Generation       | LongLaMP                          | Accuracy, F1, MAE, RMSE, ROUGE                                    | [Link](https://longlamp-benchmark.github.io/)                                     |
|       Generation       | PGraphRAG                         | ROUGE, METEOR, MAE, RMSE                                          | [Link](https://github.com/PGraphRAG-benchmark/PGraphRAG)                          |
|       Generation       | AmazonQA Products                 | ROUGE, Persona-F1                                                 | [Link](https://arxiv.org/pdf/1610.08095)                                          |
|       Generation       | Reddit                            | ROUGE, Persona-F1                                                 | [Link](https://aclanthology.org/2022.naacl-main.426.pdf)                          |
|       Generation       | MedicalDialogue                   | ROUGE, Persona-F1                                                 | [Link](https://arxiv.org/pdf/2309.11696)                                          |
|       Generation       | Personalized-gen                  | Mean Success Rate, Median Relative Improvements, Fluency          | [Link](https://github.com/balhafni/personalized-gen)                              |

## 🔗 Related Surveys and Repositories

### 📚 Surveys

#### 🧠 Personalized LLMs
- [When Large Language Models Meet Personalization: Perspectives of Challenges and Opportunities](https://arxiv.org/pdf/2307.16376)
- [Personalization of Large Language Models: A Survey](https://arxiv.org/pdf/2411.00027)
- [A Survey of Personalized Large Language Models: Progress and Future Directions](https://arxiv.org/pdf/2502.11528#page=10.52)

#### 🎭 Personalized Role-Playing
- [From Persona to Personalization: A Survey on Role-Playing Language Agents](https://arxiv.org/pdf/2404.18231)
- [Two Tales of Persona in LLMs: A Survey of Role-Playing and Personalization](https://arxiv.org/pdf/2406.01171)

### 📁 Repositories

- [Awesome Personalized Large Language Models (PLLMs)](https://github.com/JiahongLiu21/Awesome-Personalized-Large-Language-Models)  
  A curated list of resources on personalized large language models.

- [PersonaLLM Survey](https://github.com/MiuLab/PersonaLLM-Survey)  
  Companion repository for the PersonaLLM survey, covering role-playing and personalization.

- [Awesome Personalized LLM](https://github.com/HqWu-HITCS/Awesome-Personalized-LLM)  
  A collection of papers and tools focused on personalized LLM development.



## 🚀 Contributing
We sincerely welcome you to contribute to this repository! 

🙌 Whether you're adding new papers or datasets/benchmarks, fixing bugs, improving the documentation, or suggesting ideas, every bit of help is appreciated.

## 📌 Citation
If you find this repository useful in your research, please consider citing our paper:
```bibtex