from nnsight import LanguageModel
import debug_tools as dbg
from transformers import AutoTokenizer


def main():
    llm = LanguageModel("meta-llama/Meta-Llama-3.1-8B", device_map="auto")
    print(llm)

    # with llm.trace("The Eiffel Tower is in the city of", remote=True):
    #     llm.model.layers[7].output[0][:] = 4
    #     output = llm.output.save()

    # output_logits = output["logits"]
    # print("Model Output Logits: ", output_logits[0])

    # # decode the final model output from output logits
    # max_probs, tokens = output_logits[0].max(dim=-1)
    # word = [llm.tokenizer.decode(tokens.cpu()[-1])]
    # print("Model Output: ", word[0])

    model = LanguageModel("meta-llama/Llama-2-7b-hf", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    inputs = tokenizer("The universe is", return_tensors="pt")

    with model.trace() as tracer:
        output = model(**inputs)
    print("Model Output: ", output)


if __name__ == "__main__":
    with dbg.debug_on_exception():
        main()
