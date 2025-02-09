from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
# Load the model and tokenizer
model_name = "huihui-ai/Qwen2.5-1.5B-Instruct-abliterated"
model_path = '/code/model_pre/qwen_obti/pre_model_obi/models--huihui-ai--Qwen2.5-1.5B-Instruct-abliterated/snapshots/60801bee812f39cde0fe5bafc7afca56e3c87eef'
config = {
  "architectures": [
    "Qwen2ForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "hidden_act": "silu",
  "hidden_size": 1536,
  "initializer_range": 0.02,
  "intermediate_size": 8960,
  "max_position_embeddings": 32768,
  "max_window_layers": 21,
  "model_type": "qwen2",
  "num_attention_heads": 12,
  "num_hidden_layers": 28,
  "num_key_value_heads": 2,
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000.0,
  "sliding_window": 32768,
  "tie_word_embeddings": True,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.43.1",
  "use_cache": True,
  "use_sliding_window": True,
  "vocab_size": 151936
}
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    # model_name,
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_path,use_fast=False)

# Initialize conversation context
initial_messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."}
]
messages = initial_messages.copy()  # Copy the initial conversation context
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# Enter conversation loop
while True:
    # Get user input
    user_input = input("User: ").strip()  # Strip leading and trailing spaces

    # If the user types '/exit', end the conversation
    if user_input.lower() == "/exit":
        print("Exiting chat.")
        break

    # If the user types '/clean', reset the conversation context
    if user_input.lower() == "/clean":
        messages = initial_messages.copy()  # Reset conversation context
        print("Chat history cleared. Starting a new conversation.")
        continue

    # If input is empty, prompt the user and continue
    if not user_input:
        print("Input cannot be empty. Please enter something.")
        continue

    # Add user input to the conversation
    messages.append({"role": "user", "content": user_input})

    # Build the chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize input and prepare it for the model
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Generate a response from the model
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=8192
    )

    # Extract model output, removing special tokens
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Add the model's response to the conversation
    messages.append({"role": "assistant", "content": response})

    # Print the model's response
    print(f"Qwen: {response}")
