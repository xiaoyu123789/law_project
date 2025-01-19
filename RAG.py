# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu122
# pip install sentence-transformers nltk openai transformers protobuf==3.20.3


from openai import OpenAI
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
import logging
import torch


# 初始化日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 检查GPU可用性
if torch.cuda.is_available():
    device = torch.device("cuda")
    logging.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    logging.warning("GPU不可用，使用CPU。")

# 下载NLTK的句子分隔器模型
nltk.download('punkt')

# 初始化DeepSeek客户端

client = OpenAI(api_key="", base_url="https://api.deepseek.com")

# 加载Sentence-BERT模型到GPU
try:
    embedder = SentenceTransformer('all-MiniLM-L6-v2').to(device)
except Exception as e:
    logging.error(f"加载Sentence-BERT模型时出错: {e}")
    embedder = None


# 加载本地txt文档并进行分块
def load_and_chunk_documents(directory, max_chunk_tokens=300):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                # 按句子分块
                sentences = sent_tokenize(text)
                chunk = []
                for sentence in sentences:
                    chunk.append(sentence)
                    if len(chunk) >= max_chunk_tokens:
                        documents.append(' '.join(chunk))
                        chunk = []
                if chunk:
                    documents.append(' '.join(chunk))
    return documents


# 向量化文档并进行检索
def retrieve_relevant_chunks(query, documents, top_k=3):
    if embedder is None:
        logging.error("嵌入模型未加载，无法进行检索。")
        return ""
    try:
        # 向量化查询和文档
        query_embedding = embedder.encode([query], convert_to_tensor=True, device=device)
        doc_embeddings = embedder.encode(documents, convert_to_tensor=True, device=device)

        # 计算余弦相似度
        cos_scores = torch.nn.functional.cosine_similarity(query_embedding, doc_embeddings, dim=1)
        print(f"cos_scores device: {cos_scores.device}")  # 打印张量的设备信息
        # 将张量移动到CPU并转换为列表
        cos_scores = cos_scores.squeeze().cpu().tolist()

        # 获取最相关的文档块
        top_indices = np.argsort(cos_scores)[::-1][:top_k]
        relevant_chunks = [documents[i] for i in top_indices]

        # 去重并组合上下文
        unique_chunks = []
        seen = set()
        for chunk in relevant_chunks:
            if chunk not in seen:
                unique_chunks.append(chunk)
                seen.add(chunk)

        context = "\n\n".join(unique_chunks)
        return context
    except Exception as e:
        logging.error(f"检索相关文档块时出错: {e}")
        return ""


# 请求DeepSeek模型生成回答
def request_deepseek_V3(prompt, context):
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个有用的助手"},
                {"role": "user", "content": f"基于以下上下文回答我的问题：\n{context}\n\n问题：{prompt}"},
            ],
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"请求DeepSeek模型时出错: {e}")
        return "抱歉，我无法回答这个问题。"


# 主函数
def main():
    try:
        # 加载并分块文档
        documents = load_and_chunk_documents("C:\\临时文件\\RAG")
        logging.info(f"加载了{len(documents)}个文档块。")

        # 用户问题
        prompt = input("请输入问题：")

        # 检索相关文档块
        context = retrieve_relevant_chunks(prompt, documents)
        logging.info("检索到相关上下文。")

        # 生成回答
        result = request_deepseek_V3(prompt, context)
        print(result)
    except Exception as e:
        logging.error(f"主函数执行出错: {e}")


if __name__ == "__main__":
    main()