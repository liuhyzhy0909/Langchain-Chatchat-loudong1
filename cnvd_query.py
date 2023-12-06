import re
from langchain import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import os
import pickle
import langchain.schema.document
import torch
from transformers import AutoTokenizer, AutoModel, pipeline

os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'


#加载本地知识库文件，并将其切分向量化存入向量数据库
def get_embedding_db(file_path):
    # Document Loaders
    # loader = TextLoader('cnvd-name.txt', encoding='utf8')
    loader = TextLoader(file_path, encoding='utf8')
    documents = loader.load()

    texts_list = documents[0].page_content.split("\n\n")
    texts_list = [x for x in texts_list if len(x) > 0]

    # texts = [t] * len(texts_list)
    texts = [documents[0]] * len(texts_list)

    for i in range(len(texts_list)):
        texts[i].page_content = texts_list[i]

    texts = []
    for text in texts_list:
        new_text = langchain.schema.document.Document(page_content=text)
        texts.append(new_text)

    # select embeddings
    # embeddings = HuggingFaceEmbeddings()
    model_path = "/home/liuhongyue/all-mpnet-base-v2"
    # model_path = "/home/liuhongyue/LaBSE"
    # model_path = "/home/liuhongyue/sbert-base-chinese-nli"
    #
    # 选择嵌入模型并加载本地模型
    embeddings = HuggingFaceEmbeddings(model_name=model_path)
    # create vectorstores
    db = Chroma.from_documents(texts, embeddings)
    return db


def get_id(text):
    pattern = r'CNVD-\d{4}-\d{5}'  # 正则表达式模式
    matches = re.findall(pattern, text)  # 在文本中查找匹配的部分

    ids = []
    for match in matches:
        ids.append(match)
    return ids


def main(query):
    # 加载大模型
    tokenizer = AutoTokenizer.from_pretrained("/home/liuhongyue/chatglm3/finetune_demo/chatglm3-6b",
                                              trust_remote_code=True)
    model = AutoModel.from_pretrained("/home/liuhongyue/chatglm3/finetune_demo/chatglm3-6b",
                                      trust_remote_code=True, device='cuda').half().cuda()
    model = model.eval()
    db = get_embedding_db('cnvd-name.txt')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # docs = [torch.tensor(doc).to(device) for doc in docs]
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2000,
        temperature=0.1,  # temperature是用于控制生成文本的多样性的参数。较高的温度值会增加输出的随机性，而较低的值会使输出更加确定和保守。
        top_p=0.5,
        repetition_penalty=1.15,
        device=device
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    chain = load_qa_chain(llm, chain_type="stuff")

    id_list = get_id(query)
    # 加载字典数据
    with open('id_des_dict.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)

    # 加载字典数据
    with open('name_des_dict.pkl', 'rb') as f:
        loaded_dict_name = pickle.load(f)

    if len(id_list) > 0:
        print("开始查找id")
        l = len(id_list)

        # Retriever
        retriever = db.as_retriever(search_kwargs={"k": l})
        docs = retriever.get_relevant_documents(query)

        for i in range(len(id_list)):
            if id_list[i] in loaded_dict.keys():
                docs[i].page_content = loaded_dict[id_list[i]]
        docs.append(langchain.schema.document.Document(page_content='请用中文回答'))
        llm_response = chain.run(input_documents=docs, question=query)

    else:
        retriever = db.as_retriever(search_kwargs={"score": 0.85})
        docs = retriever.get_relevant_documents(query)
        if len(docs) > 0:
            print("开始查找漏洞名称")
            for doc in docs:
                a = doc.page_content
                a = a.rstrip()
                if a in loaded_dict_name.keys():
                    doc.page_content = loaded_dict_name[a]
            docs.append(langchain.schema.document.Document(page_content='请用中文回答'))

            llm_response = chain.run(input_documents=docs, question=query)
        else:
            print("查询不到id与名称，使用chatGLM3回答")
            llm_response, history = model.chat(tokenizer, query, history=[])
    return llm_response

if __name__ == '__main__':
    # query = "漏洞CNVD-2022-32838和CNVD-2022-33134的危险级别是什么"
    # query = "Laurent Rineau CGAL代码执行漏洞"
    query = "越界写入漏洞是什么"
    # query = "SAP 3D Visual Enterprise Viewer输入验证错误漏洞"
    response = main(query)
    print(response)
