from nnsight import LanguageModel
import debug_tools as dbg
from transformers import AutoTokenizer


def main():
    llm = LanguageModel("meta-llama/Meta-Llama-3.1-8B", device_map="auto")
    print(llm)

    with llm.trace("The Eiffel Tower is in the city of"):
        llm.model.layers[7].output[0][:] = 4
        output = llm.output.save()

    output_logits = output["logits"]
    print("Model Output Logits: ", output_logits[0])

    # decode the final model output from output logits
    max_probs, tokens = output_logits[0].max(dim=-1)
    word = [llm.tokenizer.decode(tokens.cpu()[-1])]
    print("Model Output: ", word[0])


if __name__ == "__main__":
    with dbg.debug_on_exception():
        main()
