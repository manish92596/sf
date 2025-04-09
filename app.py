

import ast
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------
# 1. Initialize Flask and Load Data
# ---------------------------------------------------
app = Flask(__name__)

# Load your CSV containing assessment data and embeddings
df = pd.read_csv("assessments_with_embeddings-2.csv")

# Convert the 'embedding' column from string to a Python list/array
# Example format for each row: "[0.12, 0.45, ...]"
df["embedding"] = df["embedding"].apply(ast.literal_eval)

# Stack embeddings into a 2D numpy array for similarity calculations
all_embeddings = np.vstack(df["embedding"].to_numpy())

# Load a text-embedding model (e.g., all-MiniLM-L6-v2)
model = SentenceTransformer("all-MiniLM-L6-v2")


# ---------------------------------------------------
# 2. Health Check Endpoint
#    GET /health
# ---------------------------------------------------
@app.route("/health", methods=["GET"])
def health_check():
    """
    Returns a simple JSON object to confirm the API is running.
    """
    return jsonify({"status": "healthy"}), 200


# ---------------------------------------------------
# 3. Recommendation Endpoint
#    POST /recommend
# ---------------------------------------------------
@app.route("/recommend", methods=["POST"])
def recommend_assessments():
    """
    Expects a JSON body of the form:
       {
         "query": "some text or job description"
       }

    Returns a JSON response with up to 10 recommended assessments:
       {
         "recommended_assessments": [
           {
             "url": "<string>",
             "adaptive_support": "<Yes/No>",
             "description": "<string>",
             "duration": <integer>,
             "remote_support": "<Yes/No>",
             "test_type": ["list", "of", "strings"]
           }, ...
         ]
       }
    """
    try:
        # Parse the incoming JSON body
        data = request.get_json(force=True)
        user_query = data.get("query", "")
        if not user_query:
            return jsonify({"error": "No 'query' field found in request"}), 400

        # 1. Generate embedding for the user's query
        query_embedding = model.encode([user_query])  # shape: (1, embedding_dim)

        # 2. Compute cosine similarity against all precomputed embeddings
        similarities = cosine_similarity(query_embedding, all_embeddings)[0]  # shape: (n_assessments,)

        # 3. Sort by similarity, descending; pick top 10
        df_copy = df.copy()
        df_copy["similarity"] = similarities
        top_matches = df_copy.sort_values(by="similarity", ascending=False).head(10)

        # 4. Build the response
        recommended = []
        for _, row in top_matches.iterrows():
            # If 'Test type' is a string with comma-separated values, split it into a list
            if isinstance(row.get("Test type"), str):
                test_type_list = [t.strip() for t in row["Test type"].split(",")]
            else:
                # If it's not a string, use an empty list or row["Test type"] if it's already a list
                test_type_list = row.get("Test type", [])
                if not isinstance(test_type_list, list):
                    test_type_list = []

            # Handle NaN in Duration(min); default to 0
            duration_value = row.get("Duration(min)", 0)
            if pd.isna(duration_value):
                duration_value = 0
            duration = int(duration_value)

            recommended.append({
                "url": str(row.get("URL", "")),
                "adaptive_support": str(row.get("Adaptive/IRT", "No")),
                "description": str(row.get("Assessment name", "")),
                "duration": duration,
                "remote_support": str(row.get("Remote Testing Support", "No")),
                "test_type": test_type_list
            })

        # Return the list under the "recommended_assessments" key
        return jsonify({"recommended_assessments": recommended}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------
# 4. Run the Flask App
# ---------------------------------------------------
if __name__ == "__main__":
    # You can modify the host and port as needed
    app.run(host="0.0.0.0", port=5000, debug=True)
