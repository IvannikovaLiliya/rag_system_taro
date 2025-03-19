from src.document import DocumentProcessor
from src.embeddings import VectorStore
from src.llm import LLMManager
from langchain.schema import Document

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
llm = LLMManager()
'''
Менеджер должен крутиться в состоянии while True, пока не будет выбрана опция завершить. 
Иначе при повторной инициализации llm и всех баз выше будет каждый раз долгое ожидание, пока все подгрузится
'''

# случай, когда обращаемся к llm
answer = print(llm.get_llm_prompt(query))

# случай, когда обращаемся к llm+RAG
answer = print(llm.get_rag_prompt(query))
