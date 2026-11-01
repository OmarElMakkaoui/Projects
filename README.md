# Omar El Makkaoui — projects

Student at CentraleSupélec. My interests are Bayesian statistics, models for
longitudinal and time-to-event data, and the numerical methods that make them
computable. ORCID [0009-0008-7324-2947](https://orcid.org/0009-0008-7324-2947).

## Main work: `rstanjmwiv`

**[marcopalma3/rstanjmwiv](https://github.com/marcopalma3/rstanjmwiv)**, with
Marco Palma (MRC Biostatistics Unit, Cambridge). An R/Stan package for a Bayesian
location-scale joint model of multivariate longitudinal markers and a
time-to-event outcome, in which the hazard depends on each individual's
*within-individual variability* of the markers, not only on their trajectory
(Palma et al., 2025). I am a co-author of version 2.0.0, to which I contributed:

- a **penalised B-spline (P-spline) prior for the baseline hazard**, with a
  difference-penalty matrix and gamma hyperprior on the smoothing precision, in the
  Stan model and data blocks;
- **left truncation** (delayed entry) in the survival likelihood, via a rewrite of
  the event-data preparation (`make_standata_event`);
- **per-patient log-likelihood contributions** in the generated quantities, so that
  fitted models can be compared with PSIS-LOO / WAIC.

## This repository

| Project | What it is | Methods |
|---|---|---|
| [**HPC_Parallel_Metaheuristics**](HPC_Parallel_Metaheuristics) | Tuning the parameters of a simulated HPC code under a budget of expensive simulator calls | hill climbing, simulated annealing, tabu search, Gaussian-process Bayesian optimisation; MPI multi-start (mpi4py); benchmarks against the exhaustively computed global optimum |
| [**Air_Quality_Ile-de-France**](Air_Quality_Ile-de-France) | Meteorological normalisation of NO₂ in Paris, long-term trend, and COVID-19 lockdown effects | generalised additive models (cyclic splines), blocked cross-validation, Theil–Sen / Mann–Kendall trend inference, weather-matched counterfactuals; validated on synthetic data with a planted answer |
| [**RAG_Chatbot_LangChain**](RAG_Chatbot_LangChain) | Research internship at UIR (Rabat): a retrieval-augmented assistant for university programme information | LangChain, embeddings + vector store, conversational retrieval with query rewriting |
| [**TIPE_Classes_Preparatoires**](TIPE_Classes_Preparatoires) | The two TIPE research projects from *classes préparatoires* | reaction–diffusion PDEs and travelling waves (2022); Schelling-type agent models, Metropolis–Hastings (2023) |

Each folder has its own README with the method, the results and how to reproduce them.
