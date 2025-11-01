import time
import google.generativeai as genai
import PyPDF2
from transformers import pipeline
import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from typing import List, Dict
HF_TOKEN="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"
api = "AIzaSyBm_lflyT7_lLL-Q_h0yvzfXiTAWbE8YeQ"  
genai.configure(api_key=api)
model = genai.GenerativeModel('gemini-2.0-flash')

def analyze_physics_text(text):

    rules = [
        "Terms that appear in section headings.",
        "Terms that are defined explicitly in the text.",
        "Terms that are used frequently in the conclusion.",
        "Important Formulas"
    ]

    # 2. Structured prompt
    prompt = f"""
    Analyze the following physics text and identify key concepts based on these rules and present formulas in selectable form avoid latex notation representation:
    {rules}
    and also find and filter out the questions asked in the text along with the answers if provided in the text itself, seperately(the questions without answer seperate, question with answers seperate) and make sure they are not referring to any activity, figure, image, table
    in the format
    Problem: ...
    Solution: ...
    
    Text:
    {text}
    """
    response = model.generate_content(prompt)
    return response.text

class TimeoutException(Exception):
    pass

def process_pdf_with_timeout_and_output(pdf_path, output_file="output.txt", timeout_sec=500):
    all_results = []
    start_time = time.time()
    error_occurred = False

    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_number in range(len(reader.pages)):
                elapsed_time = time.time() - start_time
                if elapsed_time > timeout_sec:
                    raise TimeoutException(f"Function execution exceeded {timeout_sec} seconds!")

                try:
                    page = reader.pages[page_number]
                    page_text = page.extract_text()
                    if page_text.strip():
                        prompt = f"""
                            Return me all the data provided with suitable formatting done, formulas in readable and selectable form ignore the vector notations on formula.
                            Ensure any mathematical expressions readability and selectibility (avoid LaTeX representation, and unnecessary delimiters like '$' unless it improves readability) and don't summarize the text
                            Here is the text:
                            ```
                            {page_text}
                            ```
                            """
                        response = model.generate_content(prompt)
                        page_result = f"--- Page {page_number + 1} ---\n{response.text}\n"
                        all_results.append(page_result)
                    time.sleep(3)  # Simulate a small delay
                except Exception as e:
                    error_occurred = True
                    print(f"Error processing page {page_number + 1}: {e}")

    except TimeoutException as e:
        print(f"Timeout Error: {e}")
        return None, False
    except Exception as e:
        print(f"An error occurred during PDF processing: {e}")
        return None, False

    # Write results to the output file
    try:
        with open(output_file, "w", encoding="utf-8") as outfile:
            outfile.write("\n".join(all_results))
        print(f"Result successfully written to: {output_file}")
    except Exception as e:
        error_occurred = True
        print(f"An error occurred while writing to the file: {e}")

    return "\n".join(all_results), not error_occurred

def generate_n_easy_problems(text: str, n: int) -> List[Dict[str, str]]:
    """
    Generates 'n' easy physics numerical problems with basic solutions
    based on the provided text.
    """
    generated_easy_problems = []
    for i in range(n):
        prompt = f"""
        Generate one easy physics numerical problem with a detailed solution (minimum 4 lines or above), present formulas in selectable form avoid latex notation representation.
        The problem should be based on terms explicitly defined in the text and important formulas from this text: {text}
        Aim for variety in the physical concepts tested within the scope of the text. Consider different scenarios, quantities to be solved for, and ways to apply the fundamental formulas.Could also use to prove a cocept.
        Format your response starting with "Problem:", then "Solution:", and finally "Answer:".
        Ensure each section starts on a new line.
        """
        response = model.generate_content(prompt)
        output_text = response.text.strip()  # Trim leading/trailing whitespace

        easy_match = re.search(r"Problem:\s*(.*?)\s*Solution:\s*(.*?)\s*Answer:\s*(.*?)(?:\n---|$)", output_text, re.DOTALL | re.IGNORECASE)
        if easy_match:
            generated_easy_problems.append({
                "problem": easy_match.group(1).strip(),
                "solution": easy_match.group(2).strip(),
                "answer": easy_match.group(3).strip()
            })
        else:
            print(f"Failed to generate easy problem {i+1}. Raw output:\n{output_text}")
    return generated_easy_problems

def generate_n_hard_problems(text: str, n: int) -> List[Dict[str, str]]:
    """
    Generates 'n' hard physics numerical problems with complex solutions
    based on the provided text, requiring multiple concepts or advanced math.
    """
    generated_hard_problems = []
    for i in range(n):
        prompt = f"""
        Generate one hard physics problem with a detailed solution based on the interplay of [mention two or three related concepts from the text]in a realistic scenario(minimum 7 lines or above), present formulas in selectable form avoid latex notation representation
        The problem should be based on the terms and important formulas from this text: {text}
        This problem should include some advanced mathematical calculations only if possible(like integration, differentiation etc).

        Format your response starting with "Problem:", then "Solution:", and finally "Answer:".
        Ensure each section starts on a new line.
        """
        response = model.generate_content(prompt)
        output_text = response.text.strip()  # Trim leading/trailing whitespace

        hard_match = re.search(r"Problem:\s*(.*?)\s*Solution:\s*(.*?)\s*Answer:\s*(.*?)(?:\n---|$)", output_text, re.DOTALL | re.IGNORECASE)
        if hard_match:
            generated_hard_problems.append({
                "problem": hard_match.group(1).strip(),
                "solution": hard_match.group(2).strip(),
                "answer": hard_match.group(3).strip()
            })
        else:
            print(f"Failed to generate hard problem {i+1}. Raw output:\n{output_text}")
    return generated_hard_problems

def generate_theoretical_qa(text, difficulty, num_results):
    """
    Generates theoretical questions and concise answers based on the provided text
    with a specified difficulty level and number of results using Gemini.

    Args:
        text: The text from which to generate questions and answers (string).
        difficulty: The desired difficulty level (e.g., "easy", "medium", "hard").
        num_results: The number of question-answer pairs to generate (integer).
        api_key: Your Google Gemini API key (string).

    Returns:
        A list of dictionaries, where each dictionary contains a 'question' and 'answer'.
    """
    genai.configure(api_key="AIzaSyBm_lflyT7_lLL-Q_h0yvzfXiTAWbE8YeQ")
    model = genai.GenerativeModel('gemini-2.0-flash')
    qa_pairs = []

    prompt_template = f"""Generate {num_results} theoretical questions and their concise answers (4-5 lines maximum) based on the following text. The questions should be at a '{difficulty}' difficulty level and should focus on underlying principles, concepts, or proofs if applicable. The answers MUST be directly derivable from the provided text.

    Text:
    {{text}}

    Format each question and answer pair as:
    Question: [Your generated question]
    Answer: [Your generated concise answer]

    ---
    """

    prompt = prompt_template.format(text=text)

    response = model.generate_content(prompt)

    if response.parts and hasattr(response.parts[0], 'text'):
        generated_text = response.parts[0].text
        # Use regex to extract question and answer, allowing for multiple 'Answer:'
        for result in generated_text.split("---"):
            if "Question:" in result:  # Check if it's a question block
                question_match = re.search(r"Question:\s*(.+)", result, re.DOTALL)
                answer_match = re.search(r"Answer:\s*(.+?)(?=(?:Question:\s*|$))", result, re.DOTALL)  # Non-greedy, lookahead for next question or end

                if question_match and answer_match:
                    question = question_match.group(1).strip()
                    answer = answer_match.group(1).strip()
                    qa_pairs.append({"question": question, "answer": answer})
    else:
        print(f"Error generating content: {response.prompt_feedback}")

    return qa_pairs

def generate_fill_in_the_blanks(text, num_questions, difficulty_level):
    """
    Generates fill-in-the-blanks questions based on the provided text using Gemini.

    Args:
        text: The text from which to generate questions (string).
        num_questions: The number of fill-in-the-blanks questions to generate (integer).
        api_key: Your Google Gemini API key (string).

    Returns:
        A list of dictionaries, where each dictionary contains:
            'question' (string): The fill-in-the-blanks question with a blank.
            'answer' (string): The correct answer for the blank.
    """

    questions = []

    prompt_template = f"""Generate {num_questions} fill-in-the-blanks questions based on the following text.  Choose key terms or phrases from the text to leave blank.  Provide the correct answer for each blank.
    for this difficulty{difficulty_level}
    Text:
    {{text}}

    Format each question and answer pair as:
    Question: [Your fill-in-the-blanks question, with the blank indicated by  '______']
    Answer: [The correct answer for the blank]

    ---
    """

    prompt = prompt_template.format(text=text)
    response = model.generate_content(prompt)

    if response.parts and hasattr(response.parts[0], 'text'):
        generated_text = response.parts[0].text
        split_results = generated_text.split("---")
        for result in split_results:
            if "Question:" in result and "Answer:" in result:
                try:
                    # Modified to handle multiple "Answer:" occurrences
                    question_part = result.split("Question:")[1].split("Answer:")[0].strip()
                    answer_part = result.split("Answer:")[1].strip()  # Get the part after the first "Answer:"

                    question = question_part
                    answer = answer_part
                    questions.append({"question": question, "answer": answer})
                except IndexError:
                    continue
    else:
        print(f"Error generating content: {response.prompt_feedback}")
    return questions
def generate_questions_refined(context, num_questions=3, temperature=0.8, top_k=60, top_p=0.95):
    question_generator = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl", device=0)
    prompts = [
        f"Generate a question about a key concept or fact in this text: {context}",
        f"Ask a specific question that can be answered directly from this passage not relating to its page number or position in text : {context}",
        f"Create a question that tests understanding of a definition or property mentioned here: {context}",
        f"Formulate a question focusing on 'what' or 'how' related to the main topic's application : {context}",
        f"What is an important question one might ask after reading this: {context}"
    ]
    generated_questions = set() # Use a set to automatically handle duplicates within the generation step
    for prompt in prompts:
        questions = question_generator(
            prompt,
            max_length=64,
            num_return_sequences=num_questions // len(prompts) + 1, # Generate a few per prompt
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True
        )
        for q in questions:
            generated_questions.add(q['generated_text'])
    return list(generated_questions)

def extract_answer(question, context):
    question_answerer = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad", device=0)
    answer = question_answerer(question=question, context=context)
    return answer['answer']

def filter_questions_even_more(questions):
    filtered_questions = []
    blacklist_keywords_strict = ["activity", "Figure", "image", "table", "graph", "equation", "experiment", "procedure", "appendix", "section", "chapter","curriculum","teaching"]
    blacklist_patterns_strict = [r"fig\.?\s*\d+", r"table\.?\s*\d+", r"eq\.?\s*\d+"]
    blacklist_prefixes_strict = ["what is the name of", "what is the title of", "according to figure", "according to table"]

    for question in questions:
        question_lower = question.lower()
        skip = False
        if any(keyword in question_lower for keyword in blacklist_keywords_strict):
            skip = True
        if any(re.search(pattern, question_lower) for pattern in blacklist_patterns_strict):
            skip = True
        if any(question_lower.startswith(prefix) for prefix in blacklist_prefixes_strict):
            skip = True
        if not skip:
            # Basic check for question mark
            if question.endswith("?"):
                filtered_questions.append(question)
    return filtered_questions

def deduplicate_and_select_questions(questions, n=5, similarity_threshold=0.8):
    if not questions:
        return []
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(questions)
    unique_questions = []
    indices_to_keep = []
    for i in range(len(questions)):
        is_duplicate = False
        for kept_index in indices_to_keep:
            similarity = cosine_similarity([embeddings[i]], [embeddings[kept_index]])[0][0]
            if similarity > similarity_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_questions.append(questions[i])
            indices_to_keep.append(i)
    return unique_questions[:n]

def evaluate_question(qa_pair):
    """
    Evaluates a single question-answer pair check if they are meaningful they are and checks student's undertanding, reasoning,remembering of facts and topics dicussed and a question that can be put in a test paper.
    in this it might contains question like this(for eg) which should not be included as it is referring to main idea and not evaluating on that main idea.
    Question: What is the main purpose of this unit of teaching?
    Answer: Explain terminal velocity
    Returns True if the question is good, False otherwise.
    """
    question = qa_pair['question']
    answer = qa_pair['answer']

    # Example evaluation criteria (you can customize these)
    # if not question.endswith("?"):
    #     return False, "Question does not end with a question mark."
    # if not answer or len(answer.split()) > 5: # Check for concise answer
    #     return False, "Answer is too long or missing."
    # if any(keyword in question.lower() for keyword in ["figure", "table", "image"]):
    #     return False, "Question refers to figures or tables."
    # # Add more sophisticated checks using Gemini if needed
    return True, "Good"

def get_gemini_refined_questions(context, existing_questions, num_needed):
    """
    Prompts Gemini to generate more questions and answers based on the context
    and the style of the existing (good) questions.
    """
    good_example_questions = [qa['question'] for qa in existing_questions if evaluate_question(qa)[0]]
    if not good_example_questions:
        example_prompt = "Generate a question about a key concept with a one or two-word answer."
    else:
        example_prompt = "Generate a question similar in style to these examples, requiring a one or two-word answer: " + ", ".join(good_example_questions[:3])

    prompt = f"""Based on the following text: "{context}" and following this style: {example_prompt}, generate {num_needed} more distinct questions and their concise (one or two word) answers."""

    response = model.generate_content(prompt)
    new_qa_pairs = []
    if hasattr(response, 'text'):
        # You'll need to parse the Gemini's output to extract questions and answers
        # This parsing logic will depend on how Gemini formats its response.
        # It might involve splitting by newlines and looking for patterns like "Q: ... A: ..."
        print("Gemini Response:\n", response.text)
        # Placeholder for parsing logic
        lines = response.text.split('\n')
        for i in range(0, len(lines) - 1, 2):
            if lines[i].startswith("Q:") and lines[i+1].startswith("A:"):
                question = lines[i][3:].strip()
                answer = lines[i+1][3:].strip()
                new_qa_pairs.append({"question": question, "answer": answer})

    return new_qa_pairs

def generate_from_here (processed_text,n):
    processed_text1 = processed_text
    num_total_questions=n
    # num_total_questions = int(input("Enter the total number of questions to generate: "))
    num_total_desired=num_total_questions
    if processed_text1:
        section_length = 4096
        overlap = 1024
        sections = [processed_text1[i:i + section_length] for i in range(0, len(processed_text1) - section_length + 1, section_length - overlap)]
        if not sections:
            sections = [processed_text1]

        potential_questions = []
        questions_per_section = 4 # Adjust as needed
        for i, section in enumerate(sections):
            print(f"Generating initial questions from section {i+1}/{len(sections)}...")
            questions = generate_questions_refined(section, num_questions=questions_per_section)
            potential_questions.extend(questions)

        print("Deduplicating and selecting top questions...")
        selected_questions = deduplicate_and_select_questions(potential_questions, n=num_total_questions * 3, similarity_threshold=0.75) # Generate even more initially

        print("Filtering out irrelevant questions...")
        filtered_questions = filter_questions_even_more(selected_questions)

        filtered_qa_pairs = []
        good_questions = []
        bad_questions = []
        num_generated = 0
        for question in filtered_questions:
            if num_generated < num_total_questions:
                print(f"Extracting answer for question: {question}")
                answer = extract_answer(question, processed_text1)
                filtered_qa_pairs.append({"question": question, "answer": answer})
                num_generated += 1
            else:
                break

        if filtered_qa_pairs:

            # print(f"\nGenerated {len(filtered_qa_pairs)} Questions and Answers:")
            # for i, qa_pair in enumerate(filtered_qa_pairs):
            #     print(f"{i+1}. Question: {qa_pair['question']}")
            #     print(f"  Answer: {qa_pair['answer']}")
            #     print("-" * 30)
            for qa in filtered_qa_pairs:
               is_good, feedback = evaluate_question(qa)
               if is_good:
                  good_questions.append(qa)
               else:
                  bad_questions.append(qa)
                  print(f"Question '{qa['question']}' considered not good: {feedback}")

            num_needed = num_total_desired - len(good_questions)

            if num_needed > 0:
              print(f"\nNeed to generate {num_needed} more questions using Gemini...")
              # Assuming you still have access to the 'processed_text1' context
              additional_qa_pairs = get_gemini_refined_questions(processed_text1, good_questions, num_needed)
              good_questions.extend(additional_qa_pairs)

            print(f"\nFinal set of {len(good_questions)} questions and answers:")
            for i, qa_pair in enumerate(good_questions):
              print(f"{i+1}. Question: {qa_pair['question']}")
              print(f"   Answer: {qa_pair['answer']}")
              print("-" * 30)
        else:
              print("No suitable questions and answers could be generated.")

    else:
        print("No text to process.")


if __name__ == "__main__":
    pdf_path = r"C:\Users\kriti\OneDrive\Pictures/physics_test1.pdf"  # Or your actual path
    output_file_path = "final_output.txt"
    processed_text, success = process_pdf_with_timeout_and_output(pdf_path, output_file_path)

    if success:
        print("PDF processing and output file creation successful.")
        # You can now work with the 'processed_text' variable
        # print(processed_text)
        astro=analyze_physics_text(processed_text)
        print(astro)
    else:
        print("PDF processing encountered errors or timed out.")
    
    # astro=analyze_physics_text(processed_text)
    # print(astro)

    num_easy = 2
    easy_problems = generate_n_easy_problems(astro, num_easy)
    if easy_problems:
        print(f"\n--- Generated {num_easy} Easy Problems ---")
        for i, problem_data in enumerate(easy_problems):
            print(f"--- Easy Problem {i+1} ---")
            print("Problem:", problem_data["problem"])
            print("Solution:\n", problem_data["solution"])
            print("Answer:", problem_data["answer"])
            print("-" * 30)

    num_hard = 2
    hard_problems = generate_n_hard_problems(astro, num_hard)
    if hard_problems:
        print(f"\n--- Generated {num_hard} Hard Problems ---")
        for i, problem_data in enumerate(hard_problems):
            print(f"--- Hard Problem {i+1} ---")
            print("Problem:", problem_data["problem"])
            print("Solution:\n", problem_data["solution"])
            print("Answer:", problem_data["answer"])
            print("-" * 30)

    provided_text = processed_text
    difficulty_level = "medium"
    number_of_questions = 2

    generated_qas = generate_theoretical_qa(provided_text, difficulty_level, number_of_questions)

    if generated_qas:
        for qa in generated_qas:
            print("Question:", qa['question'])
            print("Answer:", qa['answer'])
            print("---")
    else:
        print("No question-answer pairs were generated.")

    # generate_from_here (processed_text,10)