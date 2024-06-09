
from transformers import AutoModelForCausalLM, AutoTokenizer

# OPT GPU  = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b", load_in_8bit=True)
OPT = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b", cache_dir="cache")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b", cache_dir="cache")   

inp = "The quick brown fox jumps over the lazy dog"
inp_tokenized = tokenizer(inp, return_tensors="pt")
print(inp_tokenized['input_ids'].size())
print(inp_tokenized)

print(OPT.model)

embedded_input = OPT.model.decoder.embed_tokens(inp_tokenized['input_ids'])
print("Layer:\t", OPT.model.decoder.embed_tokens)
print("Size:\t", embedded_input.size())
print("Output:\t", embedded_input)

embed_pos_input = OPT.model.decoder.embed_positions(inp_tokenized['attention_mask'])
print("Layer:\t", OPT.model.decoder.embed_positions)
print("Size:\t", embed_pos_input.size())
print("Output:\t", embed_pos_input)


embed_position_input = embedded_input + embed_pos_input
hidden_states, _, _ = OPT.model.decoder.layers[0].self_attn(embed_position_input)
print("Layer:\t", OPT.model.decoder.layers[0].self_attn)
print("Size:\t", hidden_states.size())
print("Output:\t", hidden_states)