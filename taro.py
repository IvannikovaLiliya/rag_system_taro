from src.document import DocumentProcessor
from src.embeddings import VectorStore
from src.llm2 import VLLMWrapper
from src.llm import LLMManager
from langchain.schema import Document
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from IPython.display import display, Markdown

'''----------------------------------------чтение pdf-файлов----------------------------------------'''
dp = DocumentProcessor()

pdf_files = ['./data/Все_описания_карт.pdf',
             './data/Все_подробности_карт.pdf',
             './data/Все_сочит_карт.pdf']

query = "Что значит карта Солнце в моем раскладе Таро?"

documents = []
for pdf_file in pdf_files:
    loaded_docs = dp.load_pdf(pdf_file)
    documents.extend(loaded_docs)

combined_content = "\n".join([doc.page_content for doc in documents])
combined_document = Document(page_content=combined_content)

chunks = dp.split_documents([combined_document])

'''------------------------- embeddings и получение векторной БД на базе библиотеки FAISS-----------------------'''
vs = VectorStore(embedding_model = "all-MiniLM-L6-v2")

create_vector_db = vs.create_vector_db([doc.page_content for doc in chunks])

'''---------------------------------------- Загрузка LLM ----------------------------------------'''
rag = True
query = "Что значит карта Солнце в моем раскладе Таро?"
model_name="microsoft/Phi-3-mini-4k-instruct"
vllm_model = None
llm_model = None

'''---------------------------------------- Загрузка LLM ----------------------------------------'''
if rag:
  if vllm_model is None: 
    vllm_model = VLLMWrapper(model_name=model_name)

    # Query prompt template
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""Ты - помощник по гаданию на картах Таро. Отвечай на запросы пользователя кратко - не больше 20 слов,
                    используя контекст, если он предоставлен. Если в контексте нет информации для ответа, 
                    скажи, что не можешь помочь с этим запросом.
                    Запрос пользователя: {question}""",
    )

    # Создаем retriever
    retriever = MultiQueryRetriever.from_llm(
        create_vector_db, 
        vllm_model,
        prompt=QUERY_PROMPT
    )

    # RAG prompt template
    template = """Answer the question based ONLY on the following context:
                  {context}
                  Question: {question}
                  """

    prompt = ChatPromptTemplate.from_template(template)

    def limit_context_length(context: str, max_tokens: int = 4000) -> str:
        """
        Ограничивает длину контекста до max_tokens токенов.
        """
        tokens = context.split()
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        return " ".join(tokens)

    chain = (
        {
            "context": retriever | (lambda docs: limit_context_length(docs[0].page_content)),  # Ограничиваем длину контекста
            "question": RunnablePassthrough()
        }
        | prompt
        | vllm_model
        | StrOutputParser()
    )

    def chat_with_pdf(question):
        return display(Markdown(chain.invoke(question)))
    
    chat_with_pdf(query)

  else:

    chat_with_pdf(query)

else:
  if llm_model is None: 
    # случай, когда обращаемся к llm
    llm_model = LLMManager(model_name = model_name)

    display(Markdown(llm_model.get_llm_prompt(query)))

  else:
    
    display(Markdown(llm_model.get_llm_prompt(query)))