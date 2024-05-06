from transformers import AutoTokenizer

# Download and load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir="./custom_cache")


# The code snippet above will grab the tokenizer and load the dictionary,
# so you can simply use the tokenizer variable to encode/decode your text.
# But before we do that, let's take a look at what the vocabulary contains.

print( tokenizer.vocab )

# each entry is a pair of token and ID. Ġ, preceding certain tokens represents a space

# convert a sentence into tokens and IDs:

token_ids = tokenizer.encode("This is a sample text to test the tokenizer.")

print( "Tokens:   ", tokenizer.convert_ids_to_tokens( token_ids ) )
print( "Token IDs:", token_ids )

# The .encode() method can convert any given text into a numerical representation,
# a list of integers.
# To further investigate the process, we can use the .convert_ids_to_tokens() function 
# that shows the extracted tokens. As an example, you can observe that the word "tokenizer" 
# has been split into a combination of "token" + "izer" tokens.

# Tokens:    ['This', 'Ġis', 'Ġa', 'Ġsample', 'Ġtext', 'Ġto', 'Ġtest', 'Ġthe', 'Ġtoken', 'izer', '.']
# Token IDs: [1212, 318, 257, 6291, 2420, 284, 1332, 262, 11241, 7509, 13]
