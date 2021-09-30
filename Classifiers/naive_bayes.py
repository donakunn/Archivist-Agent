import os
import re
import numpy as np
from Classifiers.stopwords import STOPWORDS

PATHSEP = os.path.sep
# NOME_VOC = "vocabolario.txt"


class NaiveBayesClassifier:

    def __init__(self):
        self.vocabolario = {}
        # self.training_set = {}  # training_set[classe] = [documenti in quella classe]
        self.prob_classi = {}  # {classe: P(classe) }
        self.prob_parole = {}  # {parola,classe: P(parola|classe), ...}

    def save_classifier_attributes(self):
        np.save('vocabolario.npy', self.vocabolario)
        np.save('prob_classi.npy', self.prob_classi)
        np.save('prob_parole.npy', self.prob_parole)

    def load_classifier_attributes(self):
        self.vocabolario = np.load('vocabolario.npy', allow_pickle= True).item()
        self.prob_classi = np.load('prob_classi.npy', allow_pickle=True).item()
        self.prob_parole = np.load('prob_parole.npy', allow_pickle=True).item()

    def nb_init(self):

        if not os.path.exists('./vocabolario.npy'):
            train_corpus = self.costruisci_corpus("20news-bydate-train", True)
            print(f"Corpus con {len(train_corpus)} classi e {len(self.vocabolario)} parole diverse")

        # Salva il vocabolario
        # with open(NOME_VOC, "w") as file:
        #    for p in vocabolario:
        #        file.write(p + "\n")

            self.training(train_corpus)
        else:
            self.load_classifier_attributes()

        test_corpus = self.costruisci_corpus("20news-bydate-test", False)
        print("\nFase di test: classifica i documenti che non sono nel training set.")
        risultato = self.testing(test_corpus)

        print(f"\nRisultato totale della classificazione: {risultato}% di risposte esatte")
        input("Premere un tasto per terminare...")

    def analizza_testo(self, doc):
        """Tokenizza il testo contenuto in un documento e restituisce una lista che contiene i token estratti
            meno eventuali stopwords."""
        lista_parole = []
        for parola in re.findall(r"[a-zA-Z]+", doc):
            if parola not in STOPWORDS:
                lista_parole.append(parola)
        return lista_parole

    def costruisci_corpus(self, dir_path, is_training_set):  # nome non mi convince
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
                            lista_parole = self.analizza_testo(file.read().lower())
                            corpus[d][f] = lista_parole
                            if is_training_set:
                                for t in lista_parole:
                                    if t in self.vocabolario:
                                        self.vocabolario[t] += 1
                                    else:
                                        self.vocabolario[t] = 1

                if is_training_set:
                    self.pulizia_vocabolario()
        return corpus

    def pulizia_vocabolario(self):
        """Questa funzione effettua la rimozione di parole che compaiono al massimo 3 volte in tutto il corpus
         per migliorare le prestazioni del sistema."""
        da_eliminare = []
        for p in self.vocabolario:
            if self.vocabolario[p] <= 3:
                da_eliminare.append(p)
        for p in da_eliminare:
            del self.vocabolario[p]

    def training(self, corpus):
        """Costruisce le matrici con le probabilita' a priori P(c) e a posteriori
            P(p|c) di ciascuna classe c del corpus dei documenti e di ciascuna parola nella
            lista delle classi e nel vocabolario. Usa i primi n_training file
            nel dizionario corpus[c] dove c e' il nome di una classe. Pone i
            risultati nei dizionari prob_classi e prob_parole, mentre in training_set
            pone per ciascuna chiave (classe) la lista dei documenti che ha usato per
            addestrare il classificatore."""
        lunghezza_vocabolario = len(self.vocabolario)
        n_documenti_totali = 0

        for c in corpus:
            print("Classe", c, end=" ")
            # produce la lista di tutte le parole (con ripetizione) di tutti i
            # documenti della classe c: la ottiene concatenando le parole di tutti
            # i documenti
            lista_parole = []
            n_documenti_totali += len(corpus[c])  # numero di doc di training set totali
            for testo in corpus[c]:
                lista_parole.extend(corpus[c][testo])

            # calcolo probabilità condizionata P(p|c) usando la correzione di laplace per evitare che il
            # prodotto venga annullato da parole non presenti nella classe
            print("contiene", len(lista_parole), "parole totali (anche ripetute)")
            denominatore = len(lista_parole) + lunghezza_vocabolario
            for p in self.vocabolario:
                self.prob_parole[p, c] = ((lista_parole.count(p) + 1) / denominatore) * lunghezza_vocabolario

            # Calcolo delle probabilità P(c) relative alle classi
            self.prob_classi[c] = len(corpus[c].keys()) / n_documenti_totali
        self.save_classifier_attributes()

    def classifica(self, testo_doc):
        # Cerca la classe che massimizza P(classe)P(token|classe)
        p_max = -1
        class_max = ""
        for cc in self.prob_classi:
            p = self.prob_classi[cc]
            for t in testo_doc:
                if t in self.vocabolario:
                    p *= self.prob_parole[t, cc]
            if p > p_max:
                p_max = p
                class_max = cc
        return class_max

    def testing(self, corpus):
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
                # Ora prova a classificare il documento
                # print("Documento", nome, end="")
                class_max = self.classifica(testo)
                # La classe scelta è class_max
                # print(class_max, "[", p_max*100, "%]?", c)
                if class_max == c:
                    risposte_esatte += 1
                else:
                    risposte_sbagliate += 1
            print(str(100 * (risposte_esatte - e) / (
                        risposte_esatte - e + risposte_sbagliate - s)) + "% di risposte esatte")
        return risposte_esatte / (risposte_esatte + risposte_sbagliate) * 100
