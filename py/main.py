from glob import glob
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import json
import logging
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


model = SentenceTransformer('all-mpnet-base-v2')

# Configure the logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)


def ocr_all_docs():
    pass


def make_chunks():
    # preserve some metadata here
    txt_files = glob('./docs/txt/*.txt')
    chunk_size = 100
    j = []
    for i, path in enumerate(txt_files):
        destination = './docs/json/{}.json'.format(path.split('/')[-1].split('.')[-2])
        if os.path.exists(destination):
            logger.info('Skipping embedding {}, file already exists'.format(destination))
            continue

        logger.info('Embedding text file {} / {}: {}'.format(i, len(txt_files), path))
        f = open(path, 'r')
        txt = f.read()
        words = txt.split(' ')
        chunk_num = 0
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i+chunk_size])
            j.append({
                'txt': chunk,
                'chunk_num': chunk_num,
                'embedding': model.encode([chunk]).tolist(),
                'file': path
            })

        json.dump(j, open(destination, 'w'), indent=4)


def load_json():
    logger.info('Loading json files')
    paths = glob('./docs/json/*.json')
    dfs = []
    for path in paths:
        try:
            j = json.load(open(path, 'r'))
            dfs.append(pd.DataFrame(j))
        except Exception as e:
            logger.error('Could not load file {}: {}'.format(path, e))
    df = pd.concat(dfs, axis=0)

    return df


def get_embedding_index(df):
    logger.info('Loading index for embeddings')
    # todo: make embedding a 2d array
    index = faiss.IndexFlatL2(df['embedding'].shape[1])
    index.add(df['embedding'])
    return index


def nearest_texts(df, query, index, k):
    xq = model.encode([query])
    _, I = index.search(xq, k)
    nearest_rows = df.iloc[I[0]]
    return nearest_rows


def ask_llm(query, nearest):
    prompt = '{} {}'.format(' '.join(nearest['txt']), query)

    model_name_or_path = "TheBloke/Wizard-Vicuna-7B-Uncensored-GPTQ"
    # To use a different branch, change revision
    # For example: revision="main"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                 device_map="auto",
                                                 trust_remote_code=True,
                                                 revision="main")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    prompt_template=f'''A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {prompt} ASSISTANT:
    
    '''

    logger.info("\n\n*** Generate:")

    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
    logger.info(tokenizer.decode(output[0]))



query = 'What is the role of the mob in JFKs death?'
k = 5
# make_chunks()
df = load_json()
index = get_embedding_index(df)
nearest = nearest_texts(df, query, index, k)
print(nearest)
ask_llm(query, nearest)

