"""
Builds prompts by formatting template prompts
using given input variables
"""
def build_prompt(prompt, input_vars: dict):
    return prompt.format_map(input_vars)

### MMLU PROMPTS ###
"""
MMLU Baseline prompt template
"""
mmlu_baseline_prompt_template = """
Answer the following multiple-choice question about chemistry or physics or biology
by selecting the correct option: 'A', 'B', 'C', or 'D'. Only give the
correct option as the answer without reasoning.\n
Question:\n
{question}\n
{formatted_options}\n
Answer:
"""

"""
MMLU Baseline prompt template 2
"""
mmlu_baseline_prompt_template_2 = """
Answer the following question by providing only the letter (A, B, C, or D)
that corresponds to the correct option.\n
Question:\n
{question}\n
{formatted_options}\n
Answer:
"""

"""
MMLU Few-shot prompt intro
"""
mmlu_few_shot_prompt_intro = """
Below are some questions about {subject} with their corresponding answers.\n
"""

"""
MMLU Few-shot prompt template
"""
mmlu_few_shot_prompt_template = """
Now answer the following multiple-choice question by selecting the
correct option: 'A', 'B', 'C', or 'D'. Only give the correct option as the answer without reasoning.\n
Question:\n
{question}\n
{formatted_options}\n
"""

"""
MMLU Few-shot & CoT prompt template
"""
mmlu_cot_few_shot_prompt_template = """
Below are some questions about high school chemistry, physics or biology with their corresponding answers.
All questions are solved step-by-step with reasoning, and then the final answer is given among provided
multiple-choices.\n

Question 1:\n
A fruit fly population has a gene with two alleles, A1 and A2. 
Tests show that 70 percentage of the gametes produced in the population contain 
the A1 allele. If the population is in Hardy-Weinberg equilibrium, what proportion 
of the flies carry both A1 and A2?	\n

Options:\n
A) 0.7
B) 0.49
C) 0.21
D) 0.42

Solution:\n
Step 1: Understand allele frequencies in the population. Given that 70 percentage of gametes 
have the A1 allele, the frequency of allele A1 (p) is 0.7.\n
Step 2: Calculate the frequency of the A2 allele. Since the total must equal 1, 
the frequency of allele A2 (q) is 1 - p = 1 - 0.7 = 0.3.\n
Step 3: Apply the Hardy-Weinberg principle, which states that the genotype frequencies 
in a population are given by \( p^2 \) for AA, \( 2pq \) for A1A2, and \( q^2 \) for A2A2, assuming the population is in equilibrium.\n
Step 4: Calculate the proportion of individuals with genotype A1A2. Using the Hardy-Weinberg 
equation for heterozygotes: \( 2pq = 2 \times 0.7 \times 0.3 = 0.42 \).\n
Answer: D. The proportion of flies that carry both A1 and A2 alleles is 0.42, corresponding to option D.\n

Question 2:\n
Which of the following does NOT take place in the small intestine?	\n

Options: \n
A) Pancreatic lipase breaks down fats to fatty acids and glycerol.
B) Pepsin breaks down proteins to amino acids.
C) Pancreatic amylase breaks down carbohydrates into simple sugars.
D) Bile emulsifies fats into smaller fat particles.

Step 1: Understand the context. The question is about identifying which process 
does not occur in the small intestine, focusing on different digestive enzymes and their functions.\n

Step 2: Review the functions of the enzymes listed in the options:\n

Option A: Pancreatic lipase is responsible for breaking down fats into fatty acids and glycerol, which occurs in the small intestine.\n
Option B: Pepsin is an enzyme that breaks down proteins, but it operates in the stomach, not the small intestine.\n
Option C: Pancreatic amylase breaks down carbohydrates into simple sugars, and this action occurs in the small intestine.\n
Option D: Bile emulsifies fats, which helps in their digestion in the small intestine.\n

Step 3: Identify the correct answer based on the function and location of activity of these substances.\n

Answer: B. Option B "Pepsin breaks down proteins to amino acids," is the correct answer as it 
does NOT take place in the small intestine. Pepsin functions in the stomach.\n

Question 3:\n
Consider the Lewis structures for the following molecules: CO2, CO32-, NO2-, 
and NO3-. Which molecule would have the smallest bond angle between terminal atoms?\n

Options: \n
A) CO2
B) CO32-
C) NO2-
D) NO3-

Step 1: Understand the concept of bond angles in molecular geometry. Bond angles 
are influenced by the electron domain geometry around the central atom in a molecule.\n

Step 2: Review the electron domain geometry for each molecule:\n

CO2: Linear molecule with no lone pairs on the central atom, typically 180° bond angles.\n
CO32-: Trigonal planar with no lone pairs on the central carbon, 120° bond angles.\n
NO2-: Bent shape due to one lone pair on the nitrogen, smaller bond angles (typically less than 120°).\n
NO3-: Trigonal planar with no lone pairs on the nitrogen, 120° bond angles.\n

Step 3: Determine which molecule has the smallest bond angle:\n

Among the given molecules, NO2-, with its bent shape due to the presence of a lone pair, likely results in the smallest bond angle.\n
Answer: C. The molecule with the smallest bond angle between terminal atoms is NO2- (Option C).\n

###Question 5:\n
Which is the easiest way to burn a silver coin?	\n

Options: \n
A) Hold the silver coin with crucible tongs, and heat strongly in the flame of a Bunsen burner.
B) Use the method in (A), but use an oxyacetylene torch to reach a higher temperature.
C) Grind the silver coin into very small, dust-sized particles, and spray the particles into a Bunsen burner flame.
D) Dissolve the silver coin in acid, precipitate the hydroxide, and heat in a Bunsen burner flame to make the oxide.

Step 1: Assess the options and understand the physics of heating silver. 
Silver melts at approximately 962 °C, a temperature that a standard Bunsen burner cannot reach.\n

Step 2: Review each method:\n

Option A: Using a Bunsen burner may not provide sufficient heat.\n
Option B: An oxyacetylene torch could reach the necessary temperature, 
but the question specifies the "easiest" method.\n
Option C: Grinding the silver into dust-sized particles increases the surface area, 
making it more reactive and easier to burn. Fine powders can burn or react more readily because 
they can achieve higher temperatures quickly.\n
Option D: This process is complicated, involving multiple chemical processes and still 
requires intense heat for burning.\n

Step 3: Evaluate based on ease and practicality. Grinding into fine particles and 
igniting provides a straightforward path to achieving combustion without needing extremely 
high temperatures over a large bulk material.\n

Answer: C. The easiest way to burn a silver coin, given these options, is C) 
Grind the silver coin into very small, dust-sized particles, and spray the particles into a Bunsen burner flame.\n

Question 6:\n
A lightweight toy car crashes head-on into a heavier toy truck. Which of the 
following statements is true as a result of the collision?\n
I. The car will experience a greater impulse than the truck. 
II. The car will experience a 
greater change in momentum than the truck.
III. The magnitude of the acceleration experienced by the car will be greater than that experienced by the truck.

Options: \n
A) I and II only
B) II only
C) III only
D) II and III only

Step 1: Analyze the principles of impulse and momentum in collisions. 
Recall that impulse equals the change in momentum, and for a closed system (no external forces), 
the impulses experienced by both objects are equal and opposite.\n

Step 2: Review each statement:\n

Statement I: The car and the truck experience equal and opposite impulses due to 
Newton's third law. Thus, the impulse is not greater for either.\n
Statement II: Since impulse equals change in momentum and impulses are equal, the 
changes in momentum are also equal and opposite. Therefore, this statement is also incorrect.\n
Statement III: The magnitude of acceleration experienced by each object is inversely 
proportional to their masses. Given that the car is lighter than the truck, the car will 
indeed experience a greater magnitude of acceleration.\n

Step 3: Determine the correct answer based on the analysis.\n

Answer: C. The correct answer is C) III only, meaning that the magnitude of the 
acceleration experienced by the car is greater than that experienced by the truck, 
while the other statements are incorrect due to the principles of conservation of momentum and Newton's third law.\n

Now solve the following multiple-choice new question by following a step-by-step
solution like in above examples. After solving the question, provide your final answer 
by selecting the correct option: 'A', 'B', 'C', or 'D'.
Only give the correct option as your answer.\n
Question:\n
{question}\n
{formatted_options}\n
Answer : 
"""

"""
MMLU answer verifier prompt template
"""
mmlu_verifier_template = """
return f"You are the wise answer verifier who is specialized in high school chemistry, biology and physics problems.\
You will be provided a problem in of these three fields, the real answer for that problem, and the \
predicted answer from a generation model. You should understand the problem and validate the correctness of the\
generated answer in the context of the provided chemistry, biology or physics problem and the real answer.\
You should not solve the problem by yourself, you only job is to act as a verifier. \
You should only extract the model answer from the generated model response and compare it with the real answer. \
Questions are multiple-choice questions with four options that are A, B, C or D. \
The real answer will be 0 or 1 or 2 or 3. These correspond to A or B or C or D, respectively.  \
Model generated answer can be in various formats, including plain text or LaTeX-formatted text. \
Your job is to extract the model-generated answer from the given response text 
and verify it with the real answer. \
For example, if the model-generated answer is A and the real answer is 0, then the answer will be correct.\
If the model-generated answer is C and the real answer is 3, then the answer will be incorrect. It should have been D. \
Your output are limited to 'correct' or 'incorrect'. You should only response 'correct' or 'incorrect' after verifying \
the answer.\n
Real answer: {real_answer}\n
Model-generated answer: {model_answer}\n
Your output: 
"""


### MATH PROMPTS ###
"""
MATH Baseline prompt template
"""
math_baseline_prompt_template = """
Below is a math problem. Solve the problem and give an answer.
Problem:\n
{problem}\n
Answer:
"""

"""
MATH Few-shot prompt intro
"""
math_few_shot_prompt_intro = """
Below are some math problems with their step-by-step solutions and answers.\n
"""

"""
MATH Few-shot prompt template
"""
math_few_shot_prompt_template = """
Now answer the following math problem following a similar step-by-step solution.
Keep your solution less then five steps.
Give your final answer in the end.\n
Problem:\n
{problem}\n
Answer:
"""

"""
MATH answer verifier prompt template
"""
math_verifier_template = """
You are the wise answer verifier who is specialized in mathematics.\
You will be provided a math problem, the real answer of this problem, and the \
predicted answer from a generation model. You should understand the problem and validate the correctness of the\
model-generated answer in the context of the provided math problem and the real answer.\
You should not solve the problem by yourself, you only job is to act as a verifier.\
Your logic and reasoning should be rigorous and intelligent.\
The model-generated answer can potentially be in various formats, including plain text, LaTeX-formatted text, or multiple-choice options. \
These options may involve single or multiple selections, a numeric value, or a numerical value accompanied by units.\
Both the 'Real Answer' and the 'Model-generated Answer' may correspond to any of these response types.\
Exact string matching is not required; what matters is that the mathematical meaning or the options are consistent. \
In the case of multiple-choice questions, different orders are also acceptable.\
Your output are limited to 'correct' or 'incorrect'. You should only response 'correct' or 'incorrect' after verifying \
the answer. \nReal answer: {real_answer}\nModel-generated answer: {model_answer}\nYour output: 
"""

