# 私有数据问答测试
# langchain 可用于构建语言模型应用
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain

import os
# your OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = "sk-xxxx"

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader,SeleniumURLLoader

# Add the data you need
# pdf_loader = DirectoryLoader('./Reports/', glob="**/*.pdf")
# excel_loader = DirectoryLoader('./Reports/', glob="**/*.txt")
word_loader = DirectoryLoader('./before_data/', glob="**/*.docx")
url_1 = "https://shopflyer.cc/2023/06/09/starting-a-shopify-store-in-2023-comprehensive-budget-guide/"
url_2 = "https://shopflyer.cc/2023/05/30/how-to-use-utm-parameters-builder-to-track-everything/"
url_3 = "https://shopflyer.cc/2023/05/29/how-to-set-up-install-the-facebook-pixel-on-shopify-in-2023/"
url_loader = SeleniumURLLoader(urls=[url_1, url_2, url_3])

# load data
loaders = [word_loader, url_loader]
documents = []
for loader in loaders:
    documents.extend(loader.load())

# 
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(documents)

# Chroma 
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)

# init Langchain
qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0), vectorstore.as_retriever())

def handleQuestion(prompt_template, user_message, chat_history) :
    prompt = prompt_template
    print("handleQuestion", prompt, user_message)
    # alt text
    list = []
    for item in chat_history :
        for cItem in item :
            list.append(cItem)
    context = "\n".join(list)
    prompt = prompt.replace("{chat_history}", context)
    prompt = prompt.replace("{question}", user_message)
    return prompt

# gradio page
import gradio as gr
with gr.Blocks() as demo:
    radio = gr.Radio(['default', 'user-defined'], label="template", value="default")
    prompt = gr.Textbox()
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    # history data
    chat_history = []
    
    def user(user_message, radio, prompt):
        print('==>', user_message)
        # handle question
        question = [user_message]
        if (radio == "user-defined") :
            question[0] = handleQuestion(prompt, user_message, chat_history)
        
        response = qa({
            "question": question[0],
            "chat_history": chat_history
            })
        print('==> response', response)

        chat_history.append((user_message, response["answer"]))
        return gr.update(value=""), chat_history
    msg.submit(user, [msg, radio, prompt], [msg, chatbot], queue=False)

    # clear chat_history
    def handleClear() :
        chat_history.clear()
        return []
    clear.click(handleClear, outputs=chatbot)

# Display page
if __name__ == "__main__":
    demo.launch(debug=True)

