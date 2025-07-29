from gliner import GLiNER
import pandas as pd
from tqdm import tqdm
import torch
import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# config
MAX_LINES = None  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# models
gliner_news_model = GLiNER.from_pretrained("EmergentMethods/gliner_large_news-v2.1").to(DEVICE)

camembert_model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/camembert-ner").to(DEVICE)
camembert_tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner")
camembert_pipeline = pipeline(
    "ner",
    model=camembert_model,
    tokenizer=camembert_tokenizer,
    aggregation_strategy="simple",
    device=0 if DEVICE == "cuda" else -1,
    batch_size=8
)

spacy_model = spacy.load("fr_core_news_lg")

# chunk gliner input
gliner_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
GLINER_MAX_TOKENS = 280  # because of gliner truncation limit

tqdm.pandas()

# functions
def chunk_text_roberta(text, max_tokens=GLINER_MAX_TOKENS):
    """Chunk text into parts that are ≤ max_tokens tokens when tokenized."""
    tokens = gliner_tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        token_chunk = tokens[i:i + max_tokens]
        text_chunk = gliner_tokenizer.decode(token_chunk, skip_special_tokens=True)
        chunks.append(text_chunk)
    return chunks

def extract_gliner_news(text):
    try:
        labels = ["paysage", "ville", "région", "lieu"]
        chunks = chunk_text_roberta(text)
        all_entities = []
        for chunk in chunks:
            entities = gliner_news_model.predict_entities(chunk, labels)
            all_entities.extend(ent["text"].strip() for ent in entities)
        return list(set(all_entities))
    except Exception as e:
        print(f"[GLiNER-News] Error with text: {text}\n{e}")
        return []

def extract_camembert(text):
    try:
        ner_results = camembert_pipeline(text)
        return [ent["word"] for ent in ner_results if ent["entity_group"] in ["LOC", "ORG"]]
    except Exception as e:
        print(f"[CamemBERT] Error with text: {text}\n{e}")
        return []

def extract_spacy(text):
    try:
        doc = spacy_model(text)
        return [ent.text for ent in doc.ents if ent.label_ in ["LOC", "GPE"]]
    except Exception as e:
        print(f"[SpaCy] Error with text: {text}\n{e}")
        return []

def join_entities(entities):
    return "|".join(sorted(set(entities)))

def process_annotated_sample(file_path, output_path="data/all_models_output.csv", frequency_output_path="data/frequencies.csv"):
    df = pd.read_csv(file_path)
    print(f"Original rows: {len(df)}")

    df_filtered = df[df['text'].notna()].copy()
    df_filtered['text'] = df_filtered['text'].fillna('')

    if MAX_LINES is not None:
        df_filtered = df_filtered.head(MAX_LINES)
        print(f"Processing only first {MAX_LINES} rows for testing")

    tqdm.pandas(desc="GLiNER-News")
    df_filtered['predictions_gliner_news'] = df_filtered['text'].progress_apply(
        lambda text: join_entities(extract_gliner_news(text))
    )

    tqdm.pandas(desc="CamemBERT")
    df_filtered['predictions_camembert'] = df_filtered['text'].progress_apply(
        lambda text: join_entities(extract_camembert(text))
    )

    tqdm.pandas(desc="SpaCy")
    df_filtered['predictions_spacy'] = df_filtered['text'].progress_apply(
        lambda text: join_entities(extract_spacy(text))
    )

    df_filtered.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    entity_list = []
    for col in ['predictions_gliner_news', 'predictions_camembert', 'predictions_spacy']:
        df_filtered[col].dropna().apply(lambda x: entity_list.extend(x.split('|')))

    entity_df = pd.DataFrame(entity_list, columns=['entity'])
    entity_freq = entity_df['entity'].value_counts().reset_index()
    entity_freq.columns = ['entity', 'frequency']
    entity_freq.to_csv(frequency_output_path, index=False)
    print(f"Entity frequencies saved to {frequency_output_path}")

if __name__ == "__main__":
    sample_file = "data/V2_med.csv"
    process_annotated_sample(sample_file)


