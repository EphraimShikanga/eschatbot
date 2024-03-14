# import spacy

# nlp = spacy.load("en_core_web_sm")


def complex_s():
    # d = nlp("This is a complex sentence.")
    word_count = len("d")
    # has_subordinate_clause = any(
    #     token.dep_ == "mark" for token in d
    # )  # Check for subordinating conjunctions
    return word_count < 20


print(complex_s())
