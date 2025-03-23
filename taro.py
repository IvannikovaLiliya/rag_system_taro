from src.document import DocumentProcessor
from src.embeddings import VectorStore
from src.llm2 import VLLMWrapper
from langchain.schema import Document
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from IPython.display import display, Markdown

'''---------------------------------------- Функции ----------------------------------------'''
class RagModel():

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

            answer = chain.invoke(question)
            parts = answer.split('\n')
        
            # Если есть хотя бы одна новая строка, берем текст после нее
            if len(parts) > 1:
                answer = parts[1]  # Берем текст после первого '\n'
            else:
                answer = parts[0]  # Если новой строки нет, берем весь текст
            return display(Markdown(answer))

    '''----------------------------------------чтение pdf-файлов----------------------------------------'''

    dp = DocumentProcessor()

    pdf_files = ['./data/Все_описания_карт.pdf',
                './data/Все_подробности_карт.pdf',
                './data/Все_сочит_карт.pdf']

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
    model_name="microsoft/Phi-3-mini-4k-instruct"
    vllm_model = VLLMWrapper(model_name=model_name)

    query = "Что значит карта Солнце в моем раскладе Таро?"

    def rag_luanch(query):
            # Query prompt template
            QUERY_PROMPT = PromptTemplate(
                input_variables=["question"],
                template="""You are an AI language model assistant. Your task is to generate 1
                            version of the answer for user question to retrieve relevant documents from
                            a vector database. Your goal is to help the user overcome some of the limitations of the distance-based
                            similarity search.
                            Original question: {question}""",
                            )

            # Создаем retriever
            retriever = MultiQueryRetriever.from_llm(
                create_vector_db,
                vllm_model,
                prompt=QUERY_PROMPT
            )

            # RAG prompt template
            template = """Ты - помощник по гаданию на картах Таро. Отвечай на вопрос
                        КОРОТКО, 1-2 предложения, используя ТОЛЬКО контекст:{context}
                        Вопрос: {question}
                        """

            prompt = ChatPromptTemplate.from_template(template)

            chain = (
                {
                    "context": retriever | (lambda docs: limit_context_length(docs[0].page_content)),  # Ограничиваем длину контекста
                    "question": RunnablePassthrough()
                }
                | prompt
                | vllm_model
                | StrOutputParser()
            )

            chat_with_pdf(query)