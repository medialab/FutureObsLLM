# FutureObsLLM
Scripts used to detect entities in FutureObs data.

## Repository structure
   script | description |
 |--------|-------------|
  | `test_ollama_instructor.py` | Testing script to deploy Deepseek model via ollama, using Instructor library. |
 | `ollama_deepseek_extraction_base.py` | Deployment of Deepseek model with ollama for tag extraction and description. |
 | `ollama_deepseek_extraction.py` | Preprocessing of FutureObs data and deployment of Deepseek model with ollama for tag extraction and description. |
 | `triple_extraction.py` | Automatic extraction of location and geographical entities with three NER models (camembert, spacy, GliNER). |
 | `triple_extraction_evaluation.py` | Automatic extraction of entities and evaluation of named entity recognition extractions with ground truth on a sample. |
 | `comparison_models.py` | Evaluation of model specificity by calculating the number of unique entities predicted by each model. |
