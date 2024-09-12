import prompts

""" Calls Mistral via HF API and returns the model-generated answer """
def get_model_response(model, prompt, max_tokens):
    messages = [{"role": "user", "content": prompt}]
    response = model.chat_completion(messages, max_tokens=max_tokens)
    answer = response.choices[0].message.content
    return answer 

""" Returns the model-generated answer using the loaded model"""
def get_model_response_loaded(model, tokenizer, prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

""" Calls GPT-4 via OpenAI API and uses it as an evaluator for Mistral answers """
def call_gpt4_verifier(client, template_prompt, model_answer, real_answer):
    verifier_prompt = prompts.build_prompt(
            prompt=template_prompt,
            input_vars={"model_answer": model_answer, "real_answer": real_answer}
            )
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": verifier_prompt}],
        stream=True,
        )
    gpt_answer = []
    for chunk in stream:
        gpt_answer.append(chunk.choices[0].delta.content)
    return gpt_answer[1]

