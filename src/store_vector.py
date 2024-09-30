from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma

from concurrent.futures import ThreadPoolExecutor

def split_text(text):
    splitter = CharacterTextSplitter(separator=" ",
                                     chunk_size=1000,
                                     chunk_overlap=50,
                                     length_function=len)

    chunks = splitter.split_text(text)
    return chunks

def get_docs(dataframe):
    chunks_list = dataframe['review_text'].apply(lambda x: split_text(x) if isinstance(x, str) else [])
    docs = [chunk for chunks in chunks_list for chunk in chunks]
    return docs

def get_vectorstore(embedding):
    vectorstore = Chroma(embedding_function=embedding,
                         collection_name="spotify-test",
                         persist_directory="./chroma_db")
    return vectorstore

def populate_vectorstore(embedding, dataframe, batch_size=1000, sample=None):
    texts = get_docs(dataframe)
    if sample:
        texts = texts[:sample]

    vectorstore = Chroma(embedding_function=embedding,
                         collection_name="spotify-test",
                         persist_directory="./chroma_db")
    
    def add_batch_to_store(start_idx):
        batch = texts[start_idx:start_idx + batch_size]
        vectorstore.add_texts(batch)

    with ThreadPoolExecutor(max_workers=12) as executor:
        batch_indices = range(0, len(texts), batch_size)
        executor.map(add_batch_to_store, batch_indices)
    
    return vectorstore


if __name__=="__main__":
    # for initial run
    # populating the vectorstore
    import pandas as pd
    from langchain_openai import OpenAIEmbeddings
    
    import argparse
    import time
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Sample ")
    parser.add_argument('--sample', type=int, help='An integer sample value', required=False)
    args = parser.parse_args()

    start_time = time.time()
    print(f"Vector storing process started")

    data_df = pd.read_csv('data/SPOTIFY_REVIEWS.csv')
    embedding = OpenAIEmbeddings()
    if args.sample:
        vectorstore = populate_vectorstore(embedding, data_df, sample=args.sample)
    else:
        vectorstore = populate_vectorstore(embedding, data_df)
    print(f"Vector storing process ended")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Vector storing process: {elapsed_time:.2f} seconds")
