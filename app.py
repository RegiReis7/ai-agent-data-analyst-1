from langchain_experimental.agents import create_csv_agent
from langchain.llms import bedrock
import boto3
import streamlit as st

bedrock_client = boto3.client(
    service_name='bedrock-runtime', region_name='us-east-1')


def main():

    st.set_page_config(page_title="Ask your CSV")
    st.header("Ask your CSV 📈")

    csv_file = st.file_uploader("Upload a CSV file", type="csv")
    if csv_file is not None:

        agent = create_csv_agent(
            bedrock.Bedrock(model_id="cohere.command-text-v14", client=bedrock_client), csv_file, verbose=True)

        agent.handle_parsing_errors = True

        user_question = st.text_input("Ask a question about your CSV: ")

        if user_question is not None and user_question != "":
            with st.spinner(text="In progress..."):
                st.write(agent.run(user_question))


if __name__ == "__main__":
    main()
