<h1>Promoting Sustainable Web Agents: Benchmarking and Estimating Energy Consumption through Empirical and Theoretical Analysis</h1>

This work has been accepted at The 40th Annual AAAI Conference on Artificial Intelligence.
Code, Supplementary Material, Evaluation and Results for the paper "[Promoting Sustainable Web Agents: Benchmarking and Estimating Energy Consumption through Empirical and Theoretical Analysis](https://arxiv.org/abs/2511.04481)".

## Abstract
Web agents, like OpenAI's Operator and Google's Project Mariner, are powerful agentic systems pushing the boundaries of Large Language Models (LLM). They can autonomously interact with the internet at the user's behest, such as navigating websites, filling search masks, and comparing price lists.
Though web agent research is thriving, induced sustainability issues remain largely unexplored.
To highlight the urgency of this issue, we provide an initial exploration of the energy and CO<sub>2</sub> cost associated with web agents from both a theoretical &mdash;via estimation&mdash; and an empirical perspective &mdash;by benchmarking. Our results show how different philosophies in web agent creation can severely impact the associated expended energy, and that more energy consumed does not necessarily equate to better results. We highlight a lack of transparency regarding disclosing model parameters and processes used for some web agents as a limiting factor when estimating energy consumption. 
Our work contributes towards a change in thinking of how we evaluate web agents, advocating for dedicated metrics measuring energy consumption in benchmarks.


## Structure
```
WebAgentEnergy
├── agents
│   ├── AutoWebGLM_Agent
│   │   └── ...
│   ├── Mind2Web
│   │   └── ...
│   ├── MultiUI
│   │   └── ...
│   ├── Synapse
│   │   └── ...
│   └── synatra
│       └── ...
├── evaluation
│   ├── data
│   │   └── *.csv
│   └── *.R
└── supplementary_material
    └── *.pdf
```

## Agents

All web agents within the agents directory are individual web agents that can be found here:
- [AutoWebGLM](https://github.com/THUDM/AutoWebGLM)
- [Mind2Web](https://github.com/OSU-NLP-Group/Mind2Web)
- [MultiUI](https://github.com/neulab/multiui)
- [Synapse](https://github.com/ltzheng/Synapse)
- [Synatra](https://github.com/web-arena-x/synatra)

We made small modifications to their code to be able to measure the energy the web agents consume when ran on the Mind2Web benchmark.

### Evaluation
Within the evaluation directory you can find the results of our runs measuring energy consumption for the 5 web agents and the R files analyzing them.

### Supplementary Material
In the Supplementary Material you will find the appendix of our paper and high res versions of all Figures contained within.

## Contact

[Lars Krupp](mailto:lars.krupp@dfki.de)

## Licensing Information


## Citation Information

If you find this dataset useful, please consider citing our paper:

```
@misc{krupp2025promotingsustainablewebagents,
      title={Promoting Sustainable Web Agents: Benchmarking and Estimating Energy Consumption through Empirical and Theoretical Analysis}, 
      author={Lars Krupp and Daniel Geißler and Vishal Banwari and Paul Lukowicz and Jakob Karolus},
      year={2025},
      eprint={2511.04481},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2511.04481}, 
}
```
