"""
generate_action_embeddings.py - Pre-compute Action Embeddings for Semantic Search

This script generates sentence embeddings for all action descriptions in the
action vocabulary. The embeddings are saved to a pickle file and used by the
ActionSelector at runtime to map LLM intents to robot actions via cosine similarity.

Usage:
    python generate_action_embeddings.py

Output:
    action_embeddings.pkl - Contains:
        - keys: List of action names
        - embeddings: Numpy array of sentence embeddings

Note:
    Run this script whenever you add or modify action descriptions in
    op2_action_vocab.py to update the embedding database.
"""

import pickle
from sentence_transformers import SentenceTransformer
from op2_action_vocab import OP2_ACTIONS, OP2_COMPLEX_ACTIONS

def generate():
    """
    Generate and save embeddings for all action descriptions.
    
    Loads the sentence transformer model, collects descriptions from
    OP2_COMPLEX_ACTIONS, generates embeddings, and saves to pickle file.
    
    The embeddings enable fast semantic similarity search at runtime
    without needing to re-encode action descriptions each time.
    """
    print("Loading Sentence Transformer Model (all-MiniLM-L6-v2)...")
    # This model is tiny (22MB) and fast on CPU
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    keys = []
    descriptions = []
    
    # 2. Collect from Complex Actions
    for key, data in OP2_COMPLEX_ACTIONS.items():
        if "description" in data:
            keys.append(key)
            descriptions.append(data["description"])
            
    print(f"Generating embeddings for {len(keys)} actions...")
    embeddings = model.encode(descriptions)
    
    output_file = "action_embeddings.pkl"
    with open(output_file, "wb") as f:
        pickle.dump({"keys": keys, "embeddings": embeddings}, f)
        
    print(f"Saved embeddings to {output_file}")

if __name__ == "__main__":
    generate()
