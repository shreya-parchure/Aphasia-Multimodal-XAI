# Code associated with Paper: *Predicting Chronic Post-Stroke Aphasia Speech Performance From Clinically Accessible Multimodal Data Using Machine Learning*

**Authors & Affiliations:**  
Shreya Parchure, Apoorva Kelkar, Leslie Vnenchak, Olufunsho Faseyitan, John Medaglia, Denise Y. Harvey, H. Branch Coslett, Roy H. Hamilton

Laboratory for Cognition and Neural Stimulation, University of Pennsylvania

**Background:**  
The ability to predict language performance in persons with aphasia (PWA) following a stroke is crucial for guiding rehabilitation and chronic management. This study presents a novel machine learning approach to predict language outcomes in PWA for interpersonal communicative exchanges using commonly available clinical data. We also employed explainable AI to identify key features influencing individual language impairments. We trained random forest classifiers to predict conversational naming accuracy on 4620 communicative exchanges from 35 chronic PWA. Input features included 3 categories: Clinical characteristics such as Western Aphasia Battery (WAB) scores, lesion volume, and demographics; Linguistic metrics such as semantic demands and number of phonemes; White matter connectivity strengths of 83 regions from diffusion weighted MRI obtained using standard clinical parameters. Models were optimized and evaluated using cross-validation, followed by feature explainability assessment using game theory based methods through Shapley values. We then prospectively tested the best models on both new participants and unseen conversational prompts.

---

## Table of Contents

- [Paper Information](#paper-information)
- [Data and Availability](#data-and-availability)
  - [Primary Datasets](#primary-datasets)
  - [Trained Models and Saved Code Outputs](#trained-models-and-saved-code-outputs)
- [Notebooks Overview](#notebooks-overview)
  - [Notebook 1: Model Training and Comparison of Multimodal Inputs](#notebook-1-model-training-and-comparison-of-multimodal-inputs)
  - [Notebook 2: Feature Explainability](#notebook-2-feature-explainability)
  - [Notebook 3: Prospective Generalization Testing](#notebook-3-prospective-generalization-testing)
- [Contact](#contact)

---

## Paper Information

This repository contains the code associated with the paper *"Predicting Chronic Post-Stroke Aphasia Speech Performance From Clinically Accessible Multimodal Data Using Machine Learning"*. A link to the full paper will be made available after publication.

---

## Data and Availability

### Primary Datasets
The experiments in this paper use input data sourced from intake information of a clinical trial (NCT03651700). De-identified data used to conduct analyses is included in supplemental information of the paper and will be made available here after publication. Neuroimaging files will be available upon written request only, due to privacy reasons. There were 3 types of input data:

- Linguistic Task Difficulties
- Clinical Information related to standardized aphasia testing, stroke severity, and person with aphasia (PWA) demographics
- Network Node strengths computed from DWI MRI volume weighted streamlines, including 83 cortical regions according to Lausanne atlas

### Trained Models and Saved Code Outputs
After publication of the paper, we will also share the best models with fully trained weights. These models can be used directly to further predict from novel data, or can be used with interim checkpoints in code without need to re-train from the start.

---

## Code Overview

This repository includes three main code scripts that correspond to different steps of the research.

### Notebook 1: Model Training and Comparison of Multimodal Inputs

- **Function:**  
 Random forest classifiers were trained using different combinations of the 3 input feature sets and multimodal combinations. Their performance metrics were evaluated and compared to identify the best models.

- **Data Dependencies:**  
  This notebook requires access to the 3 primary datasets as specified in the [Data and Availability](#data-and-availability) section. 

- **File:**  
  *`2025 Final Code for ML Aphasia Paper - Features Comparison.ipynb`*

---

### Notebook 2: Feature Explainability

- **Function:**  
  This notebook is focused on the explainability of the features used by the best machine learning models. It employs SHAP (Shapley Additive Explanations) to interpret the contribution of different features toward model predictions.

- **Data Dependencies:**  
  Like the first notebook, this one also depends on the 3 primary datasets along with saved model weights generated from Notebook 1 as specified in the [Data and Availability](#data-and-availability) section.   

- **File:**  
  *`2025 explainability for ML aphasia paper.ipynb`*

---

### Notebook 3: Prospective Generalization Testing

- **Function:**  
  The third notebook is designed to evaluate the generalization ability of the best classifiers. It tests the models on a prospective, out-of-sample dataset to verify their robustness and practical applicability under real-world conditions.

- **Data Dependencies:**  
  This notebook requires a separate, prospective dataset which is distinct from the training and validation sets used in earlier experiments.  
  *(Prospective data will be made available after paper publication at the location specified in the [Data and Availability](#data-and-availability) section.)*

- **File:**  
  *`2025 Generalization tests Final Code for ML Aphasia Paper.ipynb`*

---

## Contact

For questions or further discussion, please contact Shreya Parchure or open an issue in the repository.
