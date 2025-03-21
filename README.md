# Awesome-Personalized-RAG-Agent

![Awesome](https://awesome.re/badge.svg)  [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Papers

### 1. Pre-retrieval
<details><summary><b>1.1 Query Rewriting</b></summary>

<p>

 **Name** | **Title** |              **Personalized presentation**              | **Publication** |                **Paper Link**                | **Code Link**                |
|:---:|:---|:-------------------------------------------------------:|:---------------:|:---:|:--------------------------------------------:|
| Least-to-most Prompting | Least-to-Most Prompting Enables Complex Reasoning in Large Language Models |     Split by sub-query terms in different questions     |    ICLR 2023    | [[Link]](https://openreview.net/forum?id=WZH7099tgfM) |
| Personalized Search-based Query Rewrite System for Conversational AI |     Build a personalized index for each user     |    ACL 2021    | [[Link]](https://aclanthology.org/2021.nlp4convai-1.17/) |
| Agent4Ranking: Semantic Robust Ranking via Personalized Query Rewriting Using Multi-agent LLM |      Use agents for efficient query rewriting.     |    Arxiv 2023    | [[Link]](https://arxiv.org/pdf/2312.15450) |
| Query Rewriting in TaoBao Search |      A learning enhanced architecture based on “query retrieval−semantic relevance”.      |    CIKM 2022    | [[Link]](https://dl.acm.org/doi/abs/10.1145/3511808.3557068?casa_token=UdZBGUHNJQYAAAAA:eetXcV5SxHrcP-82xXpYJa2jR1-0eeKgaaRa_raoEQks4q2CwXUP_VseC_3bGE8qM1_dgQYnC32T) |
| Learning to rewrite prompts for personalized text generation |       Multistage framework for personalized rewrites.       |    WWW 2024    | [[Link]](https://dl.acm.org/doi/abs/10.1145/3589334.3645408) |
| CGF: Constrained Generation Framework for Query Rewriting in Conversational AI |       Personalized prompt rewriting by using an LLM agent.       |    ACL 2022    | [[Link]](https://aclanthology.org/2022.emnlp-industry.48.pdf) |
| RL-based Query Rewriting with Distilled LLM for online E-Commerce Systems |       Student model to rewrite query.       |    Arxiv 2025    | [[Link]](https://arxiv.org/pdf/2501.18056) |
| Cognitive Personalized Search Integrating Large Language Models with an Efficient Memory Mechanism |        Personalized query intent.      |    WWW 2024    | [[Link]](https://dl.acm.org/doi/abs/10.1145/3589334.3645482) |
| BASES: Large-scale Web Search User Simulation with Large Language Model based Agents |        User simulation agent.       |    Arxiv 2024    | [[Link]](https://arxiv.org/pdf/2402.17505v1) |
| ERAGent: Enhancing Retrieval-Augmented Language Models with Improved Accuracy, Efficiency, and Personalization |        Collaorative module for query rewrite.       |    Arxiv 2024    | [[Link]](https://arxiv.org/pdf/2405.06683) |
| PEARL: Personalizing Large Language Model Writing Assistants with Generation-Calibrated Retrievers |      Personlaized LLM for query write.       |    Arxiv 2024    | [[Link]](https://arxiv.org/pdf/2311.09180) |
| Graph Meets LLM: A Novel Approach to Collaborative Filtering for Robust Conversational Understanding |      Graph-based methods with LLMs to query rewrite       |    Arxiv 2023    | [[Link]](https://arxiv.org/pdf/2305.14449) |


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

 **Name** | **Title**                                                                                                                |                    **Personalized presentation**                    | **Publication** |                **Paper Link**                |                              **Code Link**                               |
|:---:|:-------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------:|:---------------:|:---:|:------------------------------------------------------------------------:|
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
|            | Personalized Language Generation  via Bayesian Metric Augmented Retrieval                                                |                       Personalized Retrieval                        | Arxiv              | [[Link]](https://openreview.net/pdf?id=n1LiKueC4F)           |                                                                          |
|            | Leveraging Similar Users for  Personalized Language Modeling with Limited Data                                           |                       Personalized Retrieval                        | ACL 2022           | [[Link]](https://aclanthology.org/2022.acl-long.122.pdf)     |                                                                          |
| UIA        | A Personalized Dense Retrieval  Framework for     Unified Information Access                                             |                       Personalized Retrieval                        | SIGIR 2023         | [[Link]](https://dl.acm.org/doi/pdf/10.1145/3539618.3591626) |                [[Link]](https://github.com/HansiZeng/UIA)                |
| XPERT      | Personalized Retrieval over  Millions of Items                                                                           |                       Personalized Retrieval                        | SIGIR 2023         | [[Link]](https://dl.acm.org/doi/pdf/10.1145/3539618.3591749) |         [[Link]](https://github.com/personalizedretrieval/xpert)         |
| DPSR       | Towards personalized and  semantic retrieval: An end-to-end solution for e-commerce search via  embedding learning       |                       Personalized Retrieval                        | SIGIR 2020         | [[Link]](https://dl.acm.org/doi/pdf/10.1145/3397271.3401446) |                                                                          |
| PersonalTM | PersonalTM: Transformer Memory  for Personalized Retrieval                                                               |                       Personalized Retrieval                        | SIGIR 2023 (short) | [[Link]](https://dl.acm.org/doi/pdf/10.1145/3539618.3592037) |                                                                          |
|            | A zero attention model for  personalized product search                                                                  |                         Personalized Search                         | CIKM 2019          | [[Link]](https://dl.acm.org/doi/pdf/10.1145/3357384.3357980) |                                                                          |
| RTM        | Learning a Fine-Grained  Review-based Transformer Model for Personalized Product Search                                  |                         Personalized Search                         | SIGIR 2021         | [[Link]](https://dl.acm.org/doi/pdf/10.1145/3404835.3462911) |             [[Link]](https://github.com/kepingbi/ProdSearch)             |

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
  
**Name**       | **Title**                                                                                                |  **Personalized presentation**  |  **Publication**  |                                                                               **Paper Link**                                                                               |                                  **Code Link**                                  |
|:-------------------:|:---------------------------------------------------------------------------------------------------------|:-------------------------------:|:-----------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------:|
|         P2          | Evaluating and inducing personality in pre-trained language models                                       |         Role Playing            | NeurIPS 2023      |                         [[Link]](https://proceedings.neurips.cc/paper_files/paper/2023/file/21f7b745f73ce0d1f9bcea7f40b1388e-Paper-Conference.pdf)                         |[[Link]](https://sites.google.com/view/machinepersonality)                       |
|      OpinionQA      | Whose opinions do language models reflect?                                                               |          Role Playing           |     ICML 2023     |                                                 [[Link]](https://proceedings.mlr.press/v202/santurkar23a/santurkar23a.pdf)                                                 |               [[Link]](https://github.com/tatsu-lab/opinions_qa)                |
| Character Profiling | Evaluating Character Understanding of Large Language Models via Character Profiling from Fictional Works |          Role Playing           |     ICML 2023     |                                                                 [[Link]](https://arxiv.org/pdf/2404.12726)                                                                 |           [[Link]](https://github.com/Joanna0123/character_profiling)           |
|                     | Do LLMs Understand User Preferences? Evaluating LLMs On User Rating Prediction                           |   Personalized Recommendation   |       Arxiv       |                                                                 [[Link]](https://arxiv.org/pdf/2305.06474)                                                                 |                                                                                 |
|       Cue-CoT       | Cue-CoT: Chain-of-thought prompting for responding to in-depth dialogue questions with LLMs              |      Personalized Dialogue      |    EMNLP 2023     |                                                        [[Link]](https://aclanthology.org/2023.findings-emnlp.806/)                                                         |                 [[Link]](https://github.com/ruleGreen/Cue-CoT)                  |
|        TICL         | Tuning-Free Personalized Alignment via Trial-Error-Explain In-Context Learning                           |  Personalized Text Generation   |       Arxiv       |                                                                 [[Link]](https://arxiv.org/pdf/2502.08972)                                                                 |                 [[Link]](https://github.com/ruleGreen/Cue-CoT)                  |
|         GPG         | Guided Profile Generation Improves Personalization with LLMs                                             |  Personalized Text Generation   |       Arxiv       |                                                                 [[Link]](https://arxiv.org/pdf/2409.13093)                                                                 |                                                                                 |
|                     | Integrating Summarization and Retrieval for Enhanced Personalization via Large Language Models           |  Personalized Text Generation   |       Arxiv       |                                                                 [[Link]](https://arxiv.org/pdf/2310.20081)                                                                 |                                                                                 |
|     LLMTreeRec      | LLMTreeRec: Unleashing the Power of Large Language Models for Cold-Start Recommendations                 |   Personalized Recommendation   |    COLING 2025    |                                                          [[Link]](https://aclanthology.org/2025.coling-main.59/)                                                           |      [[Link]](https://github.com/Applied-Machine-Learning-Lab/LLMTreeRec)       |
|     Matryoshka      | MATRYOSHKA: Learning To Drive Black-Box LLMS With LLMS                                                   |  Personalized Text Generation   |       Arxiv       |                                                                 [[Link]](https://arxiv.org/pdf/2410.20749)                                                                 |                                                                                 |
|                     | Learning to rewrite prompts for personalized text generation                                             |  Personalized Text Generation   |     WWW 2024      |                                                        [[Link]](https://dl.acm.org/doi/pdf/10.1145/3589334.3645408)                                                        |                                                                                 |
|       RecGPT        | RecGPT: Generative Pre-training for Text-based Recommendation                                            |   Personalized Recommendation   |     ACL 2024      |                                                           [[Link]](https://aclanthology.org/2024.acl-short.29/)                                                            |                [[Link]](https://github.com/VinAIResearch/RecGPT)                |
|      PEPLER-D       | Personalized prompt learning for explainable recommendation                                              |   Personalized Recommendation   |     TOIS 2023     |                                                            [[Link]](https://dl.acm.org/doi/pdf/10.1145/3580488)                                                            |                 [[Link]](https://github.com/lileipisces/PEPLER)                 |
|        SGPT         | Unlocking the potential of prompt-tuning in bridging generalized and personalized federated learning     | Personalized Federated Learning |     CVPR 2024     | [[Link]](https://openaccess.thecvf.com/content/CVPR2024/papers/Deng_Unlocking_the_Potential_of_Prompt-Tuning_in_Bridging_Generalized_and_Personalized_CVPR_2024_paper.pdf) |                    [[Link]](https://github.com/ubc-tea/SGPT)                    |
|        PFCL         | Personalized federated continual learning via multi-granularity prompt                                   | Personalized Federated Learning |     KDD 2024      |                                                        [[Link]](https://dl.acm.org/doi/abs/10.1145/3637528.3671948)                                                        |               [[Link]](https://github.com/SkyOfBeginning/FedMGP)                |

</p>
</details>

<details><summary><b>3.2 Generation from Implicit Preference</b></summary>
<p>

  
**Name**       | **Title** |              **Personalized presentation**              |                **Publication**                 |                **Paper Link**                |                                                              **Code Link**                                                               |
|:-----------------:|:---|:-------------------------------------------------------:|:----------------------------------------------:|:---:|:----------------------------------------------------------------------------------------------------------------------------------------:|
|       PLoRA       | Personalized LoRA for Human-Centered Text Understanding                                                             | Personalized Text Understanding       |                   AAAI 2024                    | [[Link]](https://arxiv.org/pdf/2403.06208)                    |  [[Link]](https://github.com/yoyo-yun/PLoRA)                       |
|       LM-P        | Personalized Large Language Models                                                                                  | Personalized Fine-tuning              |          SENTIRE 2024 (ICDM Workshop)          | [[Link]](https://arxiv.org/pdf/2402.09269)                    |  [[Link]](https://github.com/Rikain/llm-finetuning)                |
|       MiLP        | Personalized LLM Response Generation with Parameterized User Memory Injection                                       | Personalized Text Generation          |                     Arxiv                      | [[Link]](https://arxiv.org/pdf/2404.03565)                    |  [[Link]](https://github.com/MatthewKKai/MiLP)                     |
|       OPPU        | Democratizing Large Language Models via Personalized Parameter-Efficient Fine-tuning                                | Personalized Text Generation          |                   EMNLP 2024                   | [[Link]](https://aclanthology.org/2024.emnlp-main.372.pdf)    |  [[Link]](https://github.com/TamSiuhin/OPPU)                       |
|      PER-PCS      | PERSONALIZED PIECES: Efficient Personalized Large Language Models through Collaborative Efforts                     | Personalized Text Generation          |                   EMNLP 2024                   | [[Link]](https://aclanthology.org/2024.emnlp-main.371.pdf)    |  [[Link]](https://github.com/TamSiuhin/Per-Pcs)                    |
|    Review-LLM     | Review-LLM: Harnessing Large Language Models for Personalized Review Generation                                     | Personalized Review Generation        |                     Arxiv                      | [[Link]](https://arxiv.org/pdf/2407.07487)                    |                                                                    |
|  UserIdentifier   | UserIdentifier: Implicit User Representations for Simple and Effective Personalized Sentiment Analysis              | Personalized Text Understanding       |                   NAACL 2022                   | [[Link]](https://aclanthology.org/2022.naacl-main.252.pdf)    |                                                                    |
|    UserAdapter    | UserAdapter: Few-Shot User Learning in Sentiment Analysis                                                           | Personalized Text Understanding       |               ACL Fingdings 2021               | [[Link]](https://aclanthology.org/2021.findings-acl.129.pdf)  |                                                                    |
|       HYDRA       | HYDRA: Model Factorization Framework for Black-Box LLM Personalization                                              | Personalized Reranking and Generation |                  NeurIPS 2024                  | [[Link]](https://arxiv.org/pdf/2406.02888)                    |  [[Link]](https://github.com/night-chen/HYDRA)                     |
|     PocketLLM     | PocketLLM: Enabling On-Device Fine-Tuning for Personalized LLMs                                                     | Personalized Text Generation          |         PrivateNLP 2024 (ACL Workshop)         | [[Link]](https://aclanthology.org/2024.privatenlp-1.10.pdf)   |                                                                    |
|     CoGenesis     | CoGenesis: A Framework Collaborating Large and Small Language Models for Secure Context-Aware Instruction Following | Personalized Text Generation          |                    ACl 2024                    | [[Link]](https://arxiv.org/pdf/2403.03129)                    |  [[Link]](https://github.com/TsinghuaC3I/CoGenesis)                |
|      P-RLHF       | P-RLHF: Personalized Language Modeling from Personalized Human Feedback                                             | Personalized Text Generation          |                     Arxiv                      | [[Link]](https://arxiv.org/pdf/2402.05133)                    |  [[Link]](https://github.com/HumainLab/Personalized_RLHF)          |
|      P-SOUPS      | Personalized Soups: Personalized Large Language Model Alignment via Post-hoc Parameter Merging                      | Personalized Text Generation          | Adaptive Foundation Models 2024 (NeurIPS 2024) | [[Link]](https://openreview.net/pdf?id=EMrnoPRvxe)            |  [[Link]](https://github.com/joeljang/RLPHF)                       |
|        PAD        | PAD: Personalized Alignment of LLMs at Decoding-Time                                                                | Personalized Text Generation          |                   ICLR 2025                    | [[Link]](https://arxiv.org/pdf/2410.04070)                    |  [[Link]](https://github.com/zjuruizhechen/PAD)                    |
|      REST-PG      | Reasoning-enhanced self-training for long-form personalized Text Generation                                         | Personalized Text Generation          |                     Arxiv                      | [[Link]](https://arxiv.org/pdf/2501.04167)                    |                                                                    |
|                   | Optimization Methods for Personalizing Large Language Models through Retrieval Augmentation                         | Personalized Retrieval and Generation |                  SIGIR 2024                    | [[Link]](https://dl.acm.org/doi/pdf/10.1145/3626772.3657783)  |                                                                    |
|   RewriterSlRl    | Learning to Rewrite Prompts for Personalized Text Generation                                                        | Personalized Text Generation          |                   WWW 2024                     | [[Link]](https://arxiv.org/pdf/2310.00152)                    |                                                                    |
|                   | Reinforcement learning for optimizing rag for domain chatbots                                                       | Personalized Chatbot                  |       RL+LLMs 2024 (AAAI Workshop)             | [[Link]](https://arxiv.org/pdf/2401.06800)                    |                                                                    |

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

|        **Field**        |    **Dataset**    |                       **Matrics**                        |                                      **Link**                                      |
|:-----------------------:|:-----------------:|:--------------------------------------------------------:|:----------------------------------------------------------------------------------:|
|     Query Rewriting     |       SCAN        |                         Accuracy                         |               [[Link]](https://openreview.net/forum?id=WZH7099tgfM)                |
| Retrieval / Generation  |      TOPDIAL      |                      BLEU, F1, Succ                      |                   [[Link]](https://github.com/iwangjian/TopDial)                   |
| Retrieval / Generation  |     LiveChat      |                        Recall,MRR                        |                [[Link]]( https://github.com/gaojingsheng/LiveChat)                 |
| Retrieval / Generation  |  PersonalityEvd   |           ACC,Fluency, Coherence, Plausibility           |              [[Link]](https://github.com/Lei-Sun-RUC/PersonalityEvd)               |
| Retrieval / Generation  |     Pchatbot      |                  BLEU, ROUGE, Dist, MRR                  |              [[Link]](https://github.com/qhjqhj00/SIGIR2021-Pchatbot)              |
| Retrieval / Generation  |      DuLemon      |           PPL, BLUE, ACC, Precision, Recall,F1           | [[Link]](https://github.com/PaddlePaddle/Research/tree/master/NLP/ACL2022-DuLeMon) |
| Retrieval / Generation  |  PersonalityEdit  |                  ES, DD, Acc, TPEI, PAE                  |                    [[Link]](https://github.com/zjunlp/EasyEdit)                    |
|       Generation        |       LaMP        |              Accuracy, F1, MAE, RMSE, ROUGE              |                    [[Link]](https://lamp-benchmark.github.io/)                     |
|       Generation        |     LongLaMP      |              Accuracy, F1, MAE, RMSE, ROUGE              |                  [[Link]](https://longlamp-benchmark.github.io/)                   |
|       Generation        |     PGraphRAG     |                 ROUGE, METEOR, MAE, RMSE                 |             [[Link]](https://github.com/PGraphRAG-benchmark/PGraphRAG)             |
|       Generation        | AmazonQA Products |                    ROUGE, Persona F1                     |                     [[Link]](https://arxiv.org/pdf/1610.08095)                     |
|       Generation        |      Reddit       |                    ROUGE, Persona F1                     |             [[Link]](https://aclanthology.org/2022.naacl-main.426.pdf)             |
|       Generation        |  MedicalDialogue  |                    ROUGE, Persona F1                     |                     [[Link]](https://arxiv.org/pdf/2309.11696)                     |
|       Generation        | Personalized-gen  | Mean Success Rate, Median Relative Improvements, Fluency |              [[Link]](https://github.com/balhafni/personalized-gen)                |


[//]: # (acc==Accuracy?)
## Contributing


## Citation
