import chromadb
from openai import OpenAI
from config import BASE_URL, API_KEY, LLM_MODEL, EMBEDDING_MODEL

# 初始化大模型客户端
client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY
)

# 初始化本地向量数据库
chroma_client = chromadb.PersistentClient(path="./rag_vector_db")
collection = chroma_client.get_or_create_collection(name="my_knowledge")


def get_embedding(text):
    """简易模拟Embedding（纯标准库，无额外依赖，仅用于跑通流程）"""
    import hashlib
    hash_bytes = hashlib.sha256(text.encode("utf-8")).digest()
    vector = [byte / 255.0 for byte in hash_bytes]
    return vector


def add_document_to_db(file_path):
    """读取本地txt文档，切片，生成向量并存入向量库（离线阶段）"""
    # 1. 读取文档
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 2. 固定长度切片 + 重叠
    chunk_size = 300
    overlap = 50
    chunks = []
    start = 0
    while start < len(content):
        end = start + chunk_size
        chunks.append(content[start:end])
        start = end - overlap

    # 3. 存入向量数据库
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"source": file_path, "chunk_id": i}]
        )

    print(f"文档处理完成，共切分为 {len(chunks)} 个片段")


def search_relevant_chunks(query, top_k=2):
    """根据问题检索最相关文档片段（在线检索阶段）"""
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results["documents"][0]


def ask_rag(query):
    """RAG主流程：检索 → 拼接Prompt → 调用大模型生成回答"""
    relevant_chunks = search_relevant_chunks(query)
    context = "\n---\n".join(relevant_chunks)

    prompt = f"""请严格基于下面的【参考资料】回答用户的问题。
如果参考资料里没有答案，请直接回答“资料中未提及相关内容”，不要编造。

【参考资料】
{context}

【用户问题】
{query}
"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    # 测试入口：读取test.txt构建知识库，然后提问
    add_document_to_db("test.txt")
    question = "什么是RAG？"
    answer = ask_rag(question)
    print("\n回答：", answer)
