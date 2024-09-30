from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

import argparse
from dotenv import load_dotenv
load_dotenv()

from store_vector import get_vectorstore
from generate import get_response

parser = argparse.ArgumentParser()
parser.add_argument('--th', type=float, help='metrics threshold', required=True)
args = parser.parse_args()

embedding = OpenAIEmbeddings()
llm_model = ChatOpenAI(model_name="gpt-4o", temperature=0)

vectorstore = get_vectorstore(embedding)
retriever = vectorstore.as_retriever()

user_input = [
    "What are the specific features or aspects that users appreciate the most in our application?",
    "In comparison to our application, which music streaming platform are users most likely to compare ours with?",
    "What are the primary reasons users express dissatisfaction with Spotify? ",
    "Can you identify emerging trends or patterns in recent user reviews that may impact our product strategy?"
]
expected_output = [
    "Users frequently praise the intuitive UI design, extensive variety of music choices, and seamless user experience.",
    "Users often draw comparisons with Pandora when discussing our application's features and functionalities.",
    "Common concerns among dissatisfied users include occasional playback issues, difficulties in discovering new music, and a desire for a more personalized recommendation system.",
    "Recent reviews highlight an increasing demand for enhanced social sharing features, indicating a potential opportunity for improving the platform's community engagement."
]
test_cases = []
for i, test in enumerate(zip(user_input, expected_output)):
    response = get_response(llm_model, retriever, user_input)
    test_case = LLMTestCase(
        input=test[0],
        actual_output=response,
        expected_output=test[1]
    )
    test_cases.append(test_case)

def test_rag_metrics(test_cases):    
    
    for i, test_case in enumerate(test_cases):
        answer_relevancy = AnswerRelevancyMetric(threshold=args.th)
        answer_relevancy.measure(test_case)
        if answer_relevancy.score > answer_relevancy.threshold:
            print(f"Test case {i+1} passed!: {test_case.input}")
        else: 
            print(f"Test case {i+1} failed!: {test_case.input}")
        print("Score: ", answer_relevancy.score)
        print("Reason: ", answer_relevancy.reason)

        print("Model output: ", test_case.actual_output)
        print("Expected output: ", test_case.expected_output)
        print("=====================================================")

if __name__=="__main__":
    test_rag_metrics(test_cases)