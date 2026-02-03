import os
from dotenv import load_dotenv

from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings

from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

from langchain_pinecone import PineconeVectorStore


def main():
    load_dotenv()

    # --- ENV ---
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    if not PINECONE_API_KEY:
        raise ValueError("❌ PINECONE_API_KEY not found. Add it to .env as PINECONE_API_KEY=...")

    # --- SETTINGS (same as your screenshots) ---
    index_name = "medicalbot"
    dimension = 384
    metric = "cosine"
    cloud = "aws"
    region = "us-east-1"

    # --- 1) Load docs + split + embeddings ---
    extracted_data = load_pdf_file(data="data/")  # folder name must match your project: /data
    text_chunks = text_split(extracted_data)
    embeddings = download_hugging_face_embeddings()

    # --- 2) Pinecone client ---
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # --- 3) Create index only if it doesn't exist ---
    existing_indexes = [i["name"] for i in pc.list_indexes()]
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region)
        )
        print(f"✅ Created index: {index_name}")
    else:
        print(f"✅ Index already exists: {index_name}")

    # --- 4) Upsert chunks into Pinecone ---
    docsearch = PineconeVectorStore.from_documents(
        documents=text_chunks,
        embedding=embeddings,
        index_name=index_name,
        # namespace="medical"  # (optional) uncomment if you want a namespace
    )

    print(f"✅ Upsert complete: {index_name}")


if __name__ == "__main__":
    main()
