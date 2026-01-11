
import pandas as pd
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from uuid import uuid4
import time
import os
import json


QDRANT_API_KEY=os.environ.get("QDRANT_API_KEY")
QDRANT_URL=os.environ.get("QDRANT_URL")
QDRANT_COLLECTION_NAME=os.environ.get("QDRANT_COLLECTION_NAME")

GOOGLE_API_KEY=os.environ.get("GOOGLE_API_KEY")

class HealthQDrant:
    def __init__(self, qdrant_url: str, qdrant_api_key: str):
        self.qdrant_client = QdrantClient(
            url=qdrant_url, 
            api_key=qdrant_api_key,
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=QDRANT_COLLECTION_NAME,
            embedding=self.embeddings,
        )

    def return_df_from_excel(file_path: str) -> pd.DataFrame:
        print(f"Reading {file_path}...")
        try:
            df_rag = pd.read_excel(file_path, sheet_name="Info RAG")
        except Exception as e:
            raise ValueError(f"Error reading Excel file: {e}")

        return df_rag


    def create_collection(self, collection_name: str) -> None: 
        if not self.qdrant_client.collection_exists(collection_name):
            try:
                print(f"Creating collection '{collection_name}'...")
                result = self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=768, 
                        distance=models.Distance.COSINE
                    )
                )
                print(f"Collection {collection_name} criada." if result else "Erro durante a criação da collection.")
            except Exception as e:
                raise ValueError(f"Operation failed: {e}")
        else:
            print(f"Collection {collection_name} already exists.")


    def insert_data_into_collection(self, df: pd.DataFrame, column_name: str, collection_name: str):

        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in Excel file. Available columns: {list(df.columns)}")

        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")
        genai.configure(api_key=GOOGLE_API_KEY)

        health_collection = self.qdrant_client.get_collection(collection_name)

        if not health_collection:
            raise ValueError(f"Collection {collection_name} does not exist.")

        if health_collection.points_count > 0 :
            print(f"Collection {collection_name} já foi preenchida.")
            return

        try:
            points = []
            print("Starting processing...") 
            total_rows = len(df)
            batch_size = 30

            for index, row in df.iterrows():
                text = str(row[column_name])
                
                # Skip empty rows
                if not text or text.strip() == "" or text.lower() == "nan":
                    continue
                
                try:
                    # Generate Embedding
                    result = genai.embed_content(
                        model="models/text-embedding-004",
                        content=text,
                        task_type="retrieval_document"
                    )
                    vector = result['embedding']
                    
                    # Prepare Payload
                    # Convert NaN values to None for JSON compatibility
                    # payload = row.where(pd.notnull(row), None).to_dict()
                    payload = {
                        "conteúdo": text
                    }
                    
                    points.append(models.PointStruct(
                        id=index,
                        vector=vector,
                        payload=payload
                    ))
                    
                    # Rate limiting (adjust as needed for your tier)
                    time.sleep(0.05)
                    
                except Exception as e:
                    print(f"Error processing row {index}: {e}")
                    continue

                if len(points) >= batch_size:
                    print(f"Upserting batch ({index + 1}/{total_rows})...")
                    self.qdrant_client.upsert(
                        collection_name=collection_name,
                        points=points
                    )
                    points = []

            # Upload final batch
            if points:
                print(f"Upserting final batch...")
                self.qdrant_client.upsert(
                    collection_name=collection_name,
                    points=points
                )
            
            print("Ingestion complete!")
        except Exception as e:
            raise ValueError(f"Operation failed: {e}")


    def insert_data_into_collection_md(self, df: pd.DataFrame, column_name: str, metadata_column: str, collection_name: str):
        if column_name not in df.columns or metadata_column not in df.columns:
            raise ValueError(f"Columns '{column_name}' or '{metadata_column}' not found in Excel file. Available columns: {list(df.columns)}")

        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")
        genai.configure(api_key=GOOGLE_API_KEY)
        

        health_collection = self.qdrant_client.get_collection(collection_name)
        if health_collection.points_count > 0 :
            print(f"Collection {collection_name} já foi preenchida.")
            return

        print("Starting processing...") 
        total_rows = len(df)
        batch_size = 30
        documents = []

        for index, row in df.iterrows():
            text = str(row[column_name])
            text_metadata = str(row[metadata_column])
            metadata = {} if (not text_metadata or text_metadata.strip() == "" or text_metadata.lower() == "nan") else json.loads(text_metadata)
            
            # Skip empty rows
            if not text or text.strip() == "" or text.lower() == "nan":
                continue
            
            try:
                documents.append(
                    Document(
                        page_content=text,
                        metadata=metadata,
                    )
                )
                # Rate limiting (adjust as needed for your tier)
                time.sleep(0.05)
                
            except Exception as e:
                print(f"Error processing row {index}: {e}")
                continue

            if len(documents) >= batch_size:
                print(f"Upserting batch ({index + 1}/{total_rows})...")
                uuids = [str(uuid4()) for _ in range(len(documents))]
                self.vector_store.add_documents(documents=documents, ids=uuids)
                
                documents = []

        # Upload final batch
        if documents:
            print(f"Inserting final batch...")
            uuids = [str(uuid4()) for _ in range(len(documents))]
            self.vector_store.add_documents(documents=documents, ids=uuids)



    def retrieve_qdrant_point(self, collection_name: str, point_id: int):
        retrieved_points = self.qdrant_client.retrieve(
            collection_name=collection_name,
            ids=[point_id],
            with_payload=True,  # Set to True to include the payload (metadata)
            with_vectors=True   # Set to True to include the vector(s)
        )

        # Access the retrieved point
        if retrieved_points:
            point = retrieved_points[0]
            print(f"Retrieved Point ID: {point.id}")
            print(f"Payload: {point.payload}")
            print(f"Vectors: {point.vector}")
        else:
            print(f"Point with ID {point_id} not found in collection {collection_name}")


