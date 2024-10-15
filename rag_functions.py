
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch
import re
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from openai import OpenAI

# huggingface 모델 사용하기 위해 필요한 개인키
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
githubkey=os.environ["GITHUB_TOKEN"] = "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"

#모든 문서를 백터화할 작은 단위로 나눈다.
def load_docs(files):
    #문서를 저장할 data 디렉토리를 찾고, 존재하지 않으면 만든다.
    if os.path.isdir("./data"):
        os.chdir("./data")
    else:
        os.mkdir("./data")
    
    #streamlit ui를 통해 추가한 문서들을 data 디렉토리로 저장한다. 문서 내에 줄들 사이에 공백이 너무 크면 줄바꿈을 하나로 줄인다.
    for file in files:
        savefile=open(file.name, "w")
        savefile.write(re.sub('\s\s+', '\n', file.read()))

        file.close()
        savefile.close()

    #확장자에 따라 문서를 로드한다.
    documents=[]
    for file in [f for f in os.listdir(os.curdir) if os.path.isfile(os.path.join(os.curdir, f))]:
        _, ext=file.split(".")
        if ext=="pdf":
            loader=PyPDFLoader(file)
        elif ext=="txt":
            loader=TextLoader(file, encoding='utf-8')
        documents += loader.load()
    os.chdir("..")

    # RecursiveCharacterTextSplitter는 chunk 크기보다 separators를 더 중요시해 이상한 지점에 문서를 나누기보다 줄 끝과 단어 사이에
    # 나누기 때문에 선택하게 되었다.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=int(512 / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=["\n제"],
        is_separator_regex=True,
    )

    splits=[]
    for doc in documents:
        splits += text_splitter.split_documents([doc])
    
    return splits

#chroma를 이용해서 나누어진 문서 chunk들을 임베드하고 벡터스토어를 만든다.
def create_vectorstore(splits):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device':('cuda:0' if torch.cuda.is_available() else 'cpu')}, # Pass the model configuration options
        encode_kwargs={'normalize_embeddings': True}, # Pass the encoding options
    )
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

    return vectorstore

#사용자한테 받은 질문으로 필요한 context를 찾은 후 프롬프트를 만들어 준비한 모델에 넣어 대답을 받는다.
def create_rag_chain(vectorstore, question, on):
    # READER_MODEL_NAME = "openchat/openchat_3.5"
    READER_MODEL_NAME = "maywell/Synatra-V0.1-7B-Instruct"
    # 크기와 복잡도를 줄이기 위해 파라미터 무게를 원래보다 더 작은 타입으로 바꾸는 quantization을 한다.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(READER_MODEL_NAME, quantization_config=bnb_config)
        tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)
    else:
        model = AutoModelForCausalLM.from_pretrained(READER_MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)

    #준비한 모델
    llm = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        do_sample=True,
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=500,
    )

    # 사용할 prompt template
    print(f'on is {on}')
    if on:
        prompt_template = """아래의 문맥을 사용하여 질문에 답하십시오.
        만약 답을 모른다면, 모른다고 말하고 답을 지어내지 마십시오.
        최대한 세 문장으로 답하고 가능한 한 간결하게 유지하십시오.
        {context}
        질문: {question}
        유용한 답변:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"],
        )
    else:
        prompt_template = """질문에 답하십시오.
        만약 답을 모른다면, 모른다고 말하고 답을 지어내지 마십시오.
        최대한 세 문장으로 답하고 가능한 한 간결하게 유지하십시오.
        질문: {question}
        유용한 답변:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["question"],
        )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    #template에 정보와 질문을 넣은 후 모델에 던져 대답을 받는다.
    if on:
        prompt_inputs={
            "context": format_docs(vectorstore.as_retriever().invoke(question)),
            "question": question
        }
    else:
        prompt_inputs={
            "question": question
        }

    final_prompt=PROMPT.invoke(prompt_inputs)
    print(final_prompt)
    answer=llm(final_prompt.text)
    true_answer=answer[0]['generated_text'].split("질문:")[0].strip()

    # 평가점수 얻기 위한 테스트용 코드
    reference_str="""1. 이용하려는 토지의 도면
    2. 매립폐기물의 종류ㆍ양 및 복토상태를 적은 서류
    3. 지적도"""

    print("BLEU score: {}".format(sentence_bleu(reference_str.split(), true_answer.split())))
    print("ROUGE score:")
    scorer=rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores=scorer.score(reference_str, true_answer)
    for key in scores:
        print(f'\t{key}: {scores[key]}')

    return true_answer

def create_github_rag_chain(vectorstore, question, on):
    client=OpenAI(api_key=githubkey, base_url="https://models.inference.ai.azure.com")
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    response=client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f"""아래의 문맥을 사용하여 질문에 답하십시오.
                만약 답을 모른다면, 모른다고 말하고 답을 지어내지 마십시오.
                최대한 세 문장으로 답하고 가능한 한 간결하게 유지하십시오.
                {format_docs(vectorstore.as_retriever().invoke(question))}"""
            },
            {
                "role": "user",
                "content": f'{question}'
            }
        ],
        temperature=1.0,
        top_p=1.0,
        max_tokens=1000,
        model="gpt-4o-mini"
    )

    return response.choices[0].message.content