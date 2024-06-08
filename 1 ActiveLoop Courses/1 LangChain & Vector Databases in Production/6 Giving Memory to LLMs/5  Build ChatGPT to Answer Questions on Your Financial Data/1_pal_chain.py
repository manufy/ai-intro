from langchain_openai import OpenAI
from langchain_experimental.pal_chain.base import PALChain

def main():
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

    pal_chain = PALChain.from_math_prompt(llm, verbose=True)
    question = "Alex has three times the number of balls as Ben. Ben has two more balls than Claire. If Claire has four balls, how many total balls do the three have together?"

    result = pal_chain.run(question)
    print(result)

    def solution():
        """Alex has three times the number of balls as Ben. Ben has two more balls than Claire. If Claire has four balls, how many total balls do the three have together?"""
        claire_balls = 4
        ben_balls = claire_balls + 2
        alex_balls = ben_balls * 3
        total_balls = alex_balls + ben_balls + claire_balls
        result = total_balls
        return result

if __name__ == "__main__":
    main()
