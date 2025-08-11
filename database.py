import chromadb
import uuid

class VectorDatabase:
    def __init__(self, path="./data/chroma_db"):
        """
        Initializes the ChromaDB vector database.
        Args:
            path (str): Path to store the ChromaDB data.
        """
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(name="object_embeddings")

    def add_object(self, embedding, label: str):
        """
        Adds an object embedding and its label to the database.
        Args:
            embedding: The embedding vector of the object.
            label (str): The label of the object.
        """
        # ChromaDB expects embeddings as a list of floats
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.squeeze().tolist()

        self.collection.add(
            embeddings=[embedding],
            metadatas={"label": label},
            ids=[str(uuid.uuid4())]
        )

    def query_object(self, embedding, n_results=1, min_distance=0.5):
        """
        Queries the database for similar objects.
        Args:
            embedding: The embedding vector of the object to query.
            n_results (int): Number of similar results to return.
            min_distance (float): Minimum distance for a match to be considered valid.
        Returns:
            tuple: (label, distance) of the most similar object, or (None, None) if no match.
        """
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.squeeze().tolist()

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results
        )

        if results['ids'] and results['distances']:
            # Assuming we only care about the top result for now
            label = results['metadatas'][0][0]['label']
            distance = results['distances'][0][0]
            if distance <= min_distance:
                return label, distance
        return None, None

if __name__ == '__main__':
    import torch
    # Example usage:
    db = VectorDatabase()

    # Add some dummy objects
    embedding1 = torch.randn(512) # Example embedding
    db.add_object(embedding1, "cat")
    print("Added cat embedding.")

    embedding2 = torch.randn(512) # Example embedding
    db.add_object(embedding2, "dog")
    print("Added dog embedding.")

    # Query for a similar object
    query_embedding = embedding1 + torch.randn(512) * 0.1 # Slightly perturbed version of embedding1
    label, distance = db.query_object(query_embedding)
    if label:
        print(f"Queried object: {label} with distance {distance}")
    else:
        print("No similar object found.")

    # Query for an unknown object
    unknown_embedding = torch.randn(512) # Completely new embedding
    label, distance = db.query_object(unknown_embedding)
    if label:
        print(f"Queried unknown object: {label} with distance {distance}")
    else:
        print("No similar object found for unknown object.")
