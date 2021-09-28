from Classifiers.nb_Operations import *



# NOME_VOC = "vocabolario.txt"


def nb_init():

    # Crea il vocabolario
    vocabolario = {}
    corpus = analizza_corpus("20_newsgroups", vocabolario)

    print(f"Corpus con {len(corpus)} classi e {len(vocabolario)} parole")

    # Salva il vocabolario
    # with open(NOME_VOC, "w") as file:
    #    for p in vocabolario:
    #        file.write(p + "\n")

    n = int(input("Percentuale di documenti nel training set: "))
    training_set = {}   # training_set[classe] = [documenti in quella classe]
    prob_classi = {}    # {classe: P(classe) }
    prob_parole = {}    # {parola,classe: P(parola|classe), ...}
    print(f"\nTraining set: {n}%")
    training(corpus, vocabolario, n, prob_classi, prob_parole, training_set)

    print("\nFase di test: classifica i documenti che non sono nel training set.")
    risultato = testing(corpus, vocabolario, prob_classi, prob_parole, training_set)

    print(f"\nRisultato totale della classificazione: {risultato}% di risposte esatte")
    input("Premere un tasto per terminare...")
