from openai import OpenAI
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



# 初始化DeepSeek客户端
#api = ds_api()
client = OpenAI(api_key="", base_url="https://api.deepseek.com")


# 加载本地txt文档并进行分块
def load_and_chunk_documents(directory, chunk_size=500):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                # 简单的分块逻辑，按固定长度分块
                for i in range(0, len(text), chunk_size):
                    documents.append(text[i:i + chunk_size])
    return documents


# 向量化文档并进行检索
def retrieve_relevant_chunks(query, documents, top_k=3):
    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(documents)
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, doc_vectors).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [documents[i] for i in top_indices]


# 请求DeepSeek模型生成回答
def request_deepseek_V3(prompt, context):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个有用的助手"},
            {"role": "user", "content": f"基于以下上下文回答我的问题：\n{context}\n\n问题：{prompt}"},
        ],
        stream=False
    )
    return response.choices[0].message.content


# 主函数
def main():
    # 加载并分块文档
    documents = load_and_chunk_documents("reference_book/刑法")

    # 用户问题
    prompt = input("请输入问题：")

    # 检索相关文档块
    relevant_chunks = retrieve_relevant_chunks(prompt, documents)
    context = "\n".join(relevant_chunks)

    # 生成回答
    result = request_deepseek_V3(prompt, context)
    print(result)


if __name__ == "__main__":
    main()