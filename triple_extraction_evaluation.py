from gliner import GLiNER
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import spacy
from sklearn.metrics import precision_score, recall_score, f1_score

# Load models
gliner_model = GLiNER.from_pretrained("EmergentMethods/gliner_large_news-v2.1")

camembert_model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/camembert-ner")
camembert_tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner")
camembert_pipeline = pipeline("ner", model=camembert_model, tokenizer=camembert_tokenizer, aggregation_strategy="simple")

spacy_model = spacy.load("fr_core_news_lg")

tqdm.pandas()

def extract_gliner(text):
    if not isinstance(text, str):
        return []
    try:
        labels = ["paysage", "ville", "région", "lieu"]
        entities = gliner_model.predict_entities(text, labels)
        return [ent["text"].strip() for ent in entities]
    except Exception as e:
        print(f"[GLiNER] Error with text: {text}\n{e}")
        return []

def extract_camembert(text):
    if not isinstance(text, str):
        return []
    try:
        ner_results = camembert_pipeline(text)
        return [ent["word"] for ent in ner_results if ent["entity_group"] in ["LOC", "PER", "ORG"]]
    except Exception as e:
        print(f"[CamemBERT] Error with text: {text}\n{e}")
        return []

def extract_spacy(text):
    if not isinstance(text, str):
        return []
    try:
        doc = spacy_model(text)
        return [ent.text for ent in doc.ents if ent.label_ in ["LOC", "GPE"]]
    except Exception as e:
        print(f"[SpaCy] Error with text: {text}\n{e}")
        return []

def join_entities(entities):
    return "|".join(sorted(set(entities)))

def to_entity_set(s):
    if pd.isna(s) or not s:
        return set()
    return set([e.strip().lower() for e in s.split('|') if e.strip()])

# metrics
def compute_exact_match_metrics(true_entities_list, predicted_entities_list):
    """
    Compute micro-averaged precision, recall, and f1 over exact matches.
    """
    concatenated_true_labels = []
    concatenated_pred_labels = []

    for true_entities, predicted_entities in zip(true_entities_list, predicted_entities_list):
        all_labels = list(true_entities.union(predicted_entities))
        true_binary_vector = [1 if label in true_entities else 0 for label in all_labels]
        pred_binary_vector = [1 if label in predicted_entities else 0 for label in all_labels]

        concatenated_true_labels.extend(true_binary_vector)
        concatenated_pred_labels.extend(pred_binary_vector)

    precision = precision_score(concatenated_true_labels, concatenated_pred_labels, zero_division=0)
    recall = recall_score(concatenated_true_labels, concatenated_pred_labels, zero_division=0)
    f1 = f1_score(concatenated_true_labels, concatenated_pred_labels, zero_division=0)

    return precision, recall, f1


def compute_partial_match_metrics(true_entities_list, predicted_entities_list):
    """
    Compute precision, recall, F1 considering partial matches between predicted and true entities.
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for true_entities, pred_entities in zip(true_entities_list, predicted_entities_list):
        matched_true = set()
        matched_pred = set()

        for pred_ent in pred_entities:
            pred_ent_lower = pred_ent.lower()
            matched = False
            for true_ent in true_entities:
                true_ent_lower = true_ent.lower()
                if (pred_ent_lower in true_ent_lower) or (true_ent_lower in pred_ent_lower):
                    matched = True
                    matched_true.add(true_ent)
                    matched_pred.add(pred_ent)
                    break
            if matched:
                true_positives += 1
            else:
                false_positives += 1

        unmatched_true = true_entities - matched_true
        false_negatives += len(unmatched_true)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

def process_annotated_sample(
    file_path,
    output_path="data/all_models_evaluated.csv",
    frequency_output_path="data/all_models_frequency_evaluated.csv",
    use_partial_match=False,
):
    df = pd.read_csv(file_path)

    print(f"Total number of rows in the file: {len(df)}")

    # Filter valid rows
    df_filtered = df[df['description'].notna() | df['channel_title'].notna()]
    print(f"Rows after filtering: {len(df_filtered)}")

    # Fill NaNs
    df_filtered['description'] = df_filtered['description'].fillna('')
    df_filtered['channel_title'] = df_filtered['channel_title'].fillna('')

    # gliner
    tqdm.pandas(desc="GLiNER: description")
    gliner_desc = df_filtered['description'].progress_apply(extract_gliner)

    tqdm.pandas(desc="GLiNER: channel_title")
    gliner_title = df_filtered['channel_title'].progress_apply(extract_gliner)

    df_filtered['predictions_gliner'] = (gliner_desc + gliner_title).apply(join_entities)

    # camembert
    tqdm.pandas(desc="CamemBERT: description")
    camembert_desc = df_filtered['description'].progress_apply(extract_camembert)

    tqdm.pandas(desc="CamemBERT: channel_title")
    camembert_title = df_filtered['channel_title'].progress_apply(extract_camembert)

    df_filtered['predictions_camembert'] = (camembert_desc + camembert_title).apply(join_entities)

    # spacy
    tqdm.pandas(desc="SpaCy: description")
    spacy_desc = df_filtered['description'].progress_apply(extract_spacy)

    tqdm.pandas(desc="SpaCy: channel_title")
    spacy_title = df_filtered['channel_title'].progress_apply(extract_spacy)

    df_filtered['predictions_spacy'] = (spacy_desc + spacy_title).apply(join_entities)

    # output
    df_filtered.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    # frequency count
    entity_list = []
    for col in ['predictions_gliner', 'predictions_camembert', 'predictions_spacy']:
        df_filtered[col].dropna().apply(lambda x: entity_list.extend(x.split('|')))

    entity_df = pd.DataFrame(entity_list, columns=['entity'])
    entity_freq = entity_df['entity'].value_counts().reset_index()
    entity_freq.columns = ['entity', 'frequency']
    entity_freq.to_csv(frequency_output_path, index=False)

    print("\nPerformance metrics")

    # Ground truth
    gt_desc = df_filtered['location_desc'].fillna('').apply(to_entity_set)
    gt_title = df_filtered['location_title'].fillna('').apply(to_entity_set)
    gt_total = [a.union(b) for a, b in zip(gt_desc, gt_title)]

    # Prediction
    preds_gliner = df_filtered['predictions_gliner'].fillna('').apply(to_entity_set).tolist()
    preds_camembert = df_filtered['predictions_camembert'].fillna('').apply(to_entity_set).tolist()
    preds_spacy = df_filtered['predictions_spacy'].fillna('').apply(to_entity_set).tolist()

    if use_partial_match:
        metric_fn = compute_partial_match_metrics
        print("PARTIAL MATCH")
    else:
        metric_fn = compute_exact_match_metrics
        print("EXACT MATCH")

    for name, preds in [
        ("GLiNER", preds_gliner),
        ("CamemBERT", preds_camembert),
        ("SpaCy", preds_spacy),
    ]:
        precision, recall, f1 = metric_fn(gt_total, preds)
        print(f"{name} - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

    predictions_dict = {
        "GLiNER": preds_gliner,
        "CamemBERT": preds_camembert,
        "SpaCy": preds_spacy,
    }
    
    combined_preds = []
    for i in range(len(gt_total)):
        combined_set = set()
        for preds_list in predictions_dict.values():
            combined_set |= preds_list[i]
        combined_preds.append(combined_set)

    combined_precision, combined_recall, combined_f1 = metric_fn(gt_total, combined_preds)
    print(f"\nCombined Models - Precision: {combined_precision:.3f}, Recall: {combined_recall:.3f}, F1: {combined_f1:.3f}")

if __name__ == "__main__":
    sample_file = "data/sample_baignade.csv"
    process_annotated_sample(sample_file, use_partial_match=True)



