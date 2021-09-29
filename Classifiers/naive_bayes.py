from Classifiers.nb_Operations import *
# NOME_VOC = "vocabolario.txt"


def nb_init():

    # Crea il vocabolario
    vocabolario = {}
    train_corpus = analizza_corpus("20news-bydate-train", vocabolario)
    # test_corpus = analizza_corpus("20news-bydate-test", vocabolario)
    print(f"Corpus con {len(train_corpus)} classi e {len(vocabolario)} parole")

    # Salva il vocabolario
    # with open(NOME_VOC, "w") as file:
    #    for p in vocabolario:
    #        file.write(p + "\n")

    training_set = {}   # training_set[classe] = [documenti in quella classe]
    prob_classi = {}    # {classe: P(classe) }
    prob_parole = {}    # {parola,classe: P(parola|classe), ...}

    training(train_corpus, vocabolario, prob_classi, prob_parole, training_set)

    print("\nFase di test: classifica i documenti che non sono nel training set.")
    risultato = testing(train_corpus, vocabolario, prob_classi, prob_parole, training_set)

    print(f"\nRisultato totale della classificazione: {risultato}% di risposte esatte")
    input("Premere un tasto per terminare...")
