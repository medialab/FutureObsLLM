"""
Compare overlap and specificity scores, comparing the predictions of all three NER models used.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/all_models_output.csv")

def to_entity_set(s):
    if pd.isna(s) or not s:
        return set()
    return set(e.strip().lower() for e in s.split('|') if e.strip())

df['gliner_set'] = df['predictions_gliner_news'].apply(to_entity_set)
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

    for model, ents in sets.items():
        total_counts[model] += len(ents)

    unique_counts['gliner'] += len(sets['gliner'] - sets['camembert'] - sets['spacy'])
    unique_counts['camembert'] += len(sets['camembert'] - sets['gliner'] - sets['spacy'])
    unique_counts['spacy'] += len(sets['spacy'] - sets['gliner'] - sets['camembert'])

    all_entities = sets['gliner'] | sets['camembert'] | sets['spacy']
    for entity in all_entities:
        count = sum(entity in s for s in sets.values())
        if count >= 2:
            for model, s in sets.items():
                if entity in s:
                    common_counts[model] += 1

summary_df = pd.DataFrame({
    'model': ['gliner', 'camembert', 'spacy'],
    'overlapping entities': [common_counts[m] for m in ['gliner', 'camembert', 'spacy']],
    'unique entities': [unique_counts[m] for m in ['gliner', 'camembert', 'spacy']],
    'total entities': [total_counts[m] for m in ['gliner', 'camembert', 'spacy']],
})

summary_df['common (%)'] = (summary_df['overlapping entities'] / summary_df['total entities'] * 100).round(1)
summary_df['unique (%)'] = (summary_df['unique entities'] / summary_df['total entities'] * 100).round(1)
summary_df['specificity (%)'] = summary_df['unique (%)']

plt.figure(figsize=(9, 6))
sns.set_style("whitegrid")

x = range(len(summary_df))
width = 0.8

plt.bar(x, summary_df['overlapping entities'], width=width, label='overlapping entities', color='#36a2eb')
plt.bar(x, summary_df['unique entities'], width=width, bottom=summary_df['overlapping entities'],
        label='unique entities', color='#ff5e57')

for i in x:
    c_val = summary_df['overlapping entities'][i]
    u_val = summary_df['unique entities'][i]
    u_pct = summary_df['unique (%)'][i]
    plt.text(i, c_val / 2, f"{c_val}", ha='center', va='center', color='white', fontsize=10)
    plt.text(i, c_val + u_val / 2, f"{u_val}", ha='center', va='center', color='white', fontsize=10)

    total_height = c_val + u_val
    plt.text(i, total_height + 5, f"specificity rate: {u_pct}%", ha='center', va='bottom',
             color='black', fontsize=10, fontstyle='italic')

plt.title("entity overlap and specificity per model", fontsize=14, weight='bold')

ax = plt.gca()
ax.set_ylabel("number of entities", labelpad=20)
ax.yaxis.set_label_coords(-0.08, 0)

plt.xlabel("")
plt.text(len(summary_df) - 0.5, -max(summary_df['total entities']) * 0.2, "model", ha='right', fontsize=11)

plt.xticks(ticks=x, labels=summary_df['model'])
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(title="entity type", loc='lower right')

plt.tight_layout(rect=[0, 0.1, 1, 0.95])
plt.savefig("data/entity_plot.png", dpi=300)
plt.show()

# terminal summary
print(summary_df[['model', 'total entities', 'overlapping entities', 'unique entities', 'specificity (%)']])
