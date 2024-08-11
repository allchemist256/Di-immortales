from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"  # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B-Instruct", torch_dtype="auto", device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")

# Start the loop
while True:
    # Get the prompt from the user
    prompt = input("Enter a prompt: ")

    # Prepare the messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    # Apply the chat template and prepare the model inputs
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Tokenize the input and generate the attention mask
    model_inputs = tokenizer([text], return_tensors="pt", padding=True).to(device)
    attention_mask = model_inputs["attention_mask"]

    # Generate the response with the attention mask
    generated_ids = model.generate(
        model_inputs.input_ids, attention_mask=attention_mask, max_new_tokens=512
    )

    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode and print the response
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)

    # Optional: Break the loop if the user wants to quit
    if prompt.lower() in ["exit", "quit"]:
        break
