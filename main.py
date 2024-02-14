import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
#os.environ["OPENAI_API_KEY"] = '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'

medical_advise_template = """
Can you give me {number} main root causes, prevention and cure for {disease}.
"""

medical_advise_prompt = PromptTemplate(template = medical_advise_template, input_variables = ['number', 'disease'])

gpt3_model = ChatOpenAI(model_name = "gpt-3.5-turbo-0125")  
#gpt4_model = ChatOpenAI(model_name = "gpt-4")

medical_advise_generator = LLMChain(prompt = medical_advise_prompt, llm = gpt3_model)

st.title("Gen AI Medical Advisor")
st.subheader("Learn about root causes, prevention and cure for any disease")

input_disease = st.text_input("Which disease you would like to know about?")

input_number = st.number_input("How many root cause, prevention and cure do you want me to provide?", min_value = 1, max_value = 5, value = 1, step = 1)

if st.button("Generate"):
    medical_advises = medical_advise_generator.run(number = input_number, disease = input_disease)
    st.write(medical_advises)