from Classifiers.stopwords import STOPWORDS
import os
import re

PATHSEP = os.path.sep


def analizza_testo(doc):
    """Tokenizza il testo contenuto in un documento e restituisce una lista che contiene i token estratti
        meno eventuali stopwords."""
    lista_parole = []
    for parola in re.findall(r"[a-zA-Z]+", doc):
        if parola not in STOPWORDS:
            lista_parole.append(parola)
    return lista_parole


def analizza_corpus(dir_path, vocabolario):         # nome non mi convince
    """Data una dir_path che contenga solo cartelle,
        ciascuna delle quali rappresenta una classe ed ognuna contiene documenti
        relativi alla classe descritta da tale cartella.
        Scandisce tutti gli elementi ed inserisce le parole nel dizionario
        vocabolario. Restituisce il corpus contenuto nella directory, cioè un
        dizionario le cui chiavi sono le sottocartelle (le classi) e i cui valori
        sono dizionari le cui chiavi sono i nomi dei file contenuti nelle
        sottocartelle e le liste di parole che sono state estratte da essi."""
    print("\nCarica nel vocabolario tutte le parole di tutti i documenti")
    corpus = {}
    for root, dirs, files in os.walk(dir_path):
        for d in dirs:
            corpus[d] = {}
            # Ora apre tutti i file nella classe
            print(d)
            for sroot, sdirs, sfiles in os.walk(dir_path + PATHSEP + d):
                for f in sfiles:
                    with open(dir_path + PATHSEP + d + PATHSEP + f, "r", encoding="latin-1") as file:
                        lista_parole = analizza_testo(file.read().lower())
                        corpus[d][f] = lista_parole
                        for t in lista_parole:
                            if t in vocabolario:
                                vocabolario[t] += 1
                            else:
                                vocabolario[t] = 1
    pulizia_vocabolario(vocabolario)
    return corpus


def pulizia_vocabolario(vocabolario):
    """Questa funzione effettua la rimozione di parole che compaiono al massimo 3 volte in tutto il corpus
     per migliorare le prestazioni del sistema."""
    da_eliminare = []
    for p in vocabolario:
        if vocabolario[p] <= 3:
            da_eliminare.append(p)
    for p in da_eliminare:
        del vocabolario[p]


def training(corpus, vocabolario, n_training, prob_classi, prob_parole, training_set):
    """Costruisce le matrici con le probabilita' a priori P(c) e a posteriori
        P(p|c) di ciascuna classe c del corpus dei documenti e di ciascuna parola nella
        lista delle classi e nel vocabolario. Usa i primi n_training file
        nel dizionario corpus[c] dove c e' il nome di una classe. Pone i
        risultati nei dizionari prob_classi e prob_parole, mentre in training_set
        pone per ciascuna chiave (classe) la lista dei documenti che ha usato per
        addestrare il classificatore."""
    lunghezza_vocabolario = len(vocabolario)
    n_classi = len(corpus)
    n_doc = n_classi * n_training   # numero di doc di training set totali
    for c in corpus:
        print("Classe", c, end=" ")
        # produce la lista di tutte le parole (con ripetizione) di tutti i
        # documenti della classe c: la ottiene concatenando le parole di tutti
        # i documenti
        lista_parole = []
        training_set[c] = []
        for testo in corpus[c]:
            training_set[c].append(testo)
            lista_parole.extend(corpus[c][testo])
            # Training set = primi n_training documenti.
            if len(training_set[c]) >= n_training:
                break

        # calcolo probabilità condizionata P(p|c) usando la correzione di laplace per evitare che il
        # prodotto venga annullato da parole non presenti nella classe
        print("contiene", len(lista_parole), "parole totali (anche ripetute)")
        denominatore = len(lista_parole) + lunghezza_vocabolario
        for p in vocabolario:
            prob_parole[p, c] = ((lista_parole.count(p) + 1) / denominatore) * lunghezza_vocabolario

        # Calcolo delle probabilità P(c) relative alle classi
        prob_classi[c] = len(training_set[c]) / n_doc


def testing(corpus, vocabolario, prob_classi, prob_parole, training_set):
    """Classifica gli esempi di test del corpus
        sulla base delle probabilita' fornite. Il dizionario
        training_set contiene come chiavi le classi e come valori le liste di
        file che NON vanno utilizzati per il test ???. Restituisce
        la percentuale di risposte esatte."""
    risposte_esatte = 0
    risposte_sbagliate = 0
    for c in corpus:
        print("Classe", c, end=" ")
        e = risposte_esatte
        s = risposte_sbagliate
        for testo in corpus[c]:
            if testo in training_set[c]:
                continue
            # Ora prova a classificare il documento
            # print("Documento", nome, end="")
            parole = corpus[c][testo]
            # Cerca la classe che massimizza P(classe)P(token|classe)
            p_max = -1
            class_max = ""
            for cc in prob_classi:
                p = prob_classi[cc]
                for t in parole:
                    if t in vocabolario:
                        p *= prob_parole[t, cc]
                if p > p_max:
                    p_max = p
                    class_max = cc
            # La classe scelta è class_max
            # print(class_max, "[", p_max*100, "%]?", c)
            if class_max == c:
                risposte_esatte += 1
            else:
                risposte_sbagliate += 1
        print(str(100*(risposte_esatte - e)/(risposte_esatte - e + risposte_sbagliate - s)) + "% di risposte esatte")
    return risposte_esatte / (risposte_esatte + risposte_sbagliate) * 100
