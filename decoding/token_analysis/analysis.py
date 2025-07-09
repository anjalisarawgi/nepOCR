import pandas as pd


df = pd.read_csv('decoding/token_analysis/token_errors.csv')
total_pred_tokens = df['num_pred_tokens'].sum()

token_edits = (
    df['error_tokens']
    .dropna()                      
    .str.split('|')                
    .apply(len)                    
    .sum()                         
)

probs = (
    df['error_probs']
    .dropna()
    .str.split('|')
    .explode()
    .astype(float)
)

prob_threshold = 1.0
count_low_prob  = (probs < prob_threshold).sum()
count_high_prob = (probs >= prob_threshold).sum()

assert count_low_prob + count_high_prob == token_edits, (
    f"Exploded entropy count ({count_low_prob + count_high_prob}) "
    f"≠ token_edits ({token_edits})"
)

print(f"Total predicted tokens:        {total_pred_tokens}")
print(f"Total token edits:             {token_edits}")
print(f"Error tokens with prob <{prob_threshold}:  {count_low_prob}")
print(f"Error tokens with prob >={prob_threshold}: {count_high_prob}")

count_0_2_to_0_4 = ((probs >= 0.0) & (probs < 0.2)).sum()
print(f"Error tokens with prob between 0.2 and 0.4: {count_0_2_to_0_4}")