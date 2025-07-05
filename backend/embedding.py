from sentence_transformers import SentenceTransformer

# Set your project path
project_path = "D:/Projects/MultiChatBot"

# List of models to download and save
models = [
    "intfloat/e5-small-v2",
    "BAAI/bge-small-en-v1.5",
    "all-MiniLM-L6-v2"
]

for model_name in models:
    model = SentenceTransformer(model_name)
    local_path = f"{project_path}/hf_models/{model_name.replace('/', '_')}"
    model.save(local_path)
    print(f"Saved {model_name} to {local_path}")