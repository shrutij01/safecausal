from nnsight import LanguageModel

llm = LanguageModel("meta-llama/Meta-Llama-3.1-8B", device_map="auto")

with llm.trace("The Eiffel Tower is in the city of", remote=True):
    llm.model.layers[7].output[0][:] = 4
    output = llm.output.save()

output_logits = output["logits"]
print("Model Output Logits: ", output_logits[0])

# decode the final model output from output logits
max_probs, tokens = output_logits[0].max(dim=-1)
word = [llm.tokenizer.decode(tokens.cpu()[-1])]
print("Model Output: ", word[0])
