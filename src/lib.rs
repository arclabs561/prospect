//! Hyperparameter optimization and search space exploration.
//!
//! Typed search spaces, trial management, and black-box optimization
//! for machine learning pipelines.
//!
//! # Design sketch
//!
//! ```text
//! // Define a search space
//! let space = SearchSpace::new()
//!     .float("learning_rate", 1e-5, 1e-1, Scale::Log)
//!     .int("hidden_dim", 64, 512, Scale::Linear)
//!     .categorical("optimizer", &["sgd", "adam", "lamb"])
//!     .conditional("momentum", 0.0, 0.99, |t| t["optimizer"] == "sgd");
//!
//! // Run optimization
//! let study = Study::new(space)
//!     .sampler(Sampler::TPE)        // or Random, Grid, Bayesian
//!     .pruner(Pruner::Hyperband)    // early stopping
//!     .direction(Direction::Minimize);
//!
//! study.optimize(100, |trial| {
//!     let lr = trial.suggest_float("learning_rate");
//!     let dim = trial.suggest_int("hidden_dim");
//!     let opt = trial.suggest_categorical("optimizer");
//!
//!     let loss = train_model(lr, dim, opt);
//!     loss  // return objective value
//! });
//!
//! let best = study.best_trial();
//! let pareto = study.pareto_front(); // multi-objective via `pare`
//! ```
//!
//! # Ecosystem integration
//!
//! - [`pare`](https://crates.io/crates/pare): multi-objective Pareto frontier
//!   for studies with multiple objectives
//! - [`kuji`](https://crates.io/crates/kuji): Gumbel-max sampling for
//!   stochastic candidate selection
//! - [`descend`](https://crates.io/crates/descend): optimizers and schedules
//!   as the inner loop of HPO
//! - [`statskit`](https://crates.io/crates/statskit): statistical comparison
//!   of trial results (bootstrap, Wilcoxon, ASO)
//!
//! # Key types (planned)
//!
//! - `SearchSpace`: typed parameter definitions (float, int, categorical, conditional)
//! - `Trial`: a frozen configuration sampled from the space
//! - `Study`: optimization session managing trials + results
//! - `Sampler`: strategy for proposing trials (Random, Grid, TPE, Bayesian/GP)
//! - `Pruner`: early stopping (MedianPruner, Hyperband, SuccessiveHalving)
//! - `Direction`: Minimize or Maximize (multi-objective via pare)
