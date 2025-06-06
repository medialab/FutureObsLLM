"""
Compare overlap and specificity scores, comparing the predictions of all three NER models used. 
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/all_models_evaluated.csv") 

def to_entity_set(s):
    if pd.isna(s) or not s:
        return set()
    return set(e.strip().lower() for e in s.split('|') if e.strip())

df['gliner_set'] = df['predictions_gliner'].apply(to_entity_set)
df['camembert_set'] = df['predictions_camembert'].apply(to_entity_set)
df['spacy_set'] = df['predictions_spacy'].apply(to_entity_set)

unique_counts = {'gliner': 0, 'camembert': 0, 'spacy': 0}
common_counts = {'gliner': 0, 'camembert': 0, 'spacy': 0}
total_counts = {'gliner': 0, 'camembert': 0, 'spacy': 0}

for _, row in df.iterrows():
    sets = {
        'gliner': row['gliner_set'],
        'camembert': row['camembert_set'],
        'spacy': row['spacy_set']
    }

    # Totals
    for model, ents in sets.items():
        total_counts[model] += len(ents)

    # unique entities
    unique_counts['gliner'] += len(sets['gliner'] - sets['camembert'] - sets['spacy'])
    unique_counts['camembert'] += len(sets['camembert'] - sets['gliner'] - sets['spacy'])
    unique_counts['spacy'] += len(sets['spacy'] - sets['gliner'] - sets['camembert'])

    # entity present in at least two models
    all_entities = sets['gliner'] | sets['camembert'] | sets['spacy']
    for entity in all_entities:
        count = sum(entity in s for s in sets.values())
        if count >= 2:
            for model, s in sets.items():
                if entity in s:
                    common_counts[model] += 1

summary_df = pd.DataFrame({
    'model': ['gliner', 'camembert', 'spacy'],
    'common entities': [common_counts[m] for m in ['gliner', 'camembert', 'spacy']],
    'unique entities': [unique_counts[m] for m in ['gliner', 'camembert', 'spacy']],
    'total entities': [total_counts[m] for m in ['gliner', 'camembert', 'spacy']],
})

summary_df['specificity (%)'] = (summary_df['unique entities'] / summary_df['total entities'] * 100).round(2)

plt.figure(figsize=(9, 6))
sns.set_style("whitegrid")

x = range(len(summary_df))
width = 0.85  
plt.bar(x, summary_df['common entities'], width=width, label='common entities', color='#36a2eb')
plt.bar(x, summary_df['unique entities'], width=width, bottom=summary_df['common entities'], label='unique entities', color='#ff5e57')

plt.title("entity overlap and specificity per model", fontsize=14, weight='bold')
plt.ylabel("number of entities")
plt.xticks(ticks=x, labels=summary_df['model'])
plt.xlabel("model")
plt.legend(title="Entity type")
plt.grid(axis='y', linestyle='--', alpha=0.6)

for i in x:
    plt.text(i, summary_df['common entities'][i] / 2, str(summary_df['common entities'][i]), ha='center', va='center', color='white', fontsize=10)
    plt.text(i, summary_df['common entities'][i] + summary_df['unique entities'][i] / 2, str(summary_df['unique entities'][i]), ha='center', va='center', color='white', fontsize=10)

plt.tight_layout()
plt.savefig("entity_plot.png", dpi=300)
plt.show()

print(summary_df[['model', 'total entities', 'common entities', 'unique entities', 'specificity (%)']])