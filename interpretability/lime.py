import requests
import math
from lime.lime_text import LimeTextExplainer

# URL of your local Ollama HTTP endpoint
OLLAMA_URL = "http://127.0.0.1:11434/v1/completions"
MODEL_NAME = "your‑ollama‑model"    # e.g. "vicuna‑7b‑q4"

def predict_hiring_ollama(texts):
    """
    Given a list of candidate profiles, return [[P(Yes),P(No),P(Maybe)], ...]
    by querying Ollama with logprobs=3 and temperature=0.
    """
    headers = {"Content-Type": "application/json"}
    all_probs = []
    for txt in texts:
        prompt = (
            "You are a hiring assistant. Decide: Yes, No, or Maybe.\n"
            f"Candidate profile:\n\"{txt}\"\n"
            "Decision:"
        )
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "max_tokens": 1,
            "logprobs": 3,
            "temperature": 0.0
        }
        r = requests.post(OLLAMA_URL, json=payload, headers=headers)
        r.raise_for_status()
        choice = r.json()["choices"][0]
        top_lp = choice["logprobs"]["top_logprobs"][0]
        # pull out each class (allowing for leading space)
        yes_lp   = top_lp.get(" Yes",   top_lp.get("Yes",   -1e9))
        no_lp    = top_lp.get(" No",    top_lp.get("No",    -1e9))
        maybe_lp = top_lp.get(" Maybe", top_lp.get("Maybe", -1e9))
        # softmax
        m = max(yes_lp, no_lp, maybe_lp)
        e_yes, e_no, e_maybe = map(lambda lp: math.exp(lp - m),
                                  (yes_lp, no_lp, maybe_lp))
        s = e_yes + e_no + e_maybe
        all_probs.append([e_yes/s, e_no/s, e_maybe/s])

    return all_probs


# --- now hook into LIME exactly as before ---

explainer = LimeTextExplainer(class_names=["Yes","No","Maybe"])

candidate = (
    "5 years of backend Java experience, led a team of 3, "
    "familiar with AWS and Terraform, seeking a DevOps role."
)

exp = explainer.explain_instance(
    candidate,
    predict_hiring_ollama,
    num_features=6,
    top_labels=1
)

label = exp.top_labels[0]
print("Decision:", explainer.class_names[label])
print("Contributing words:", exp.as_list(label=label))

# In Jupyter you can also do:
# exp.show_in_notebook(label=label)
