import os
import re
import numpy as np
from Classifiers.stopwords import STOPWORDS

PATHSEP = os.path.sep


class NaiveBayesClassifier:

    def __init__(self):
        self.vocabolario = {}
        self.prob_classi = {}  # {classe: P(classe) }
        self.prob_parole = {}  # {parola,classe: P(parola|classe), ...}

    def save_classifier_attributes(self):
        np.save('vocabolario.npy', self.vocabolario)
        np.save('prob_classi.npy', self.prob_classi)
        np.save('prob_parole.npy', self.prob_parole)

    def load_classifier_attributes(self):
        self.vocabolario = np.load('vocabolario.npy', allow_pickle=True).item()
        self.prob_classi = np.load('prob_classi.npy', allow_pickle=True).item()
        self.prob_parole = np.load('prob_parole.npy', allow_pickle=True).item()

    def nb_init(self):
        if not os.path.exists('./vocabolario.npy') or not os.path.exists('./prob_classi.npy')\
                or not os.path.exists('./prob_parole.npy'):
            train_corpus = self.costruisci_corpus("20news-bydate-train", True)
            print(f"Corpus con {len(train_corpus)} classi e {len(self.vocabolario)} parole diverse")
            self.training(train_corpus)
        else:
            self.load_classifier_attributes()

        test_corpus = self.costruisci_corpus("20news-bydate-test", False)
        print("\nFase di test: classifica i documenti che non sono nel training set.")
        risultato = self.testing(test_corpus)

        print("\nRisultato totale della classificazione: {0:6.2f}% di risposte esatte".format(risultato))
        input("Premere un tasto per continuare...")

    def analizza_testo(self, doc):
        lista_parole = []
        for parola in re.findall(r"[a-zA-Z]+", doc):
            if parola not in STOPWORDS:
                lista_parole.append(parola)
        return lista_parole

    def costruisci_corpus(self, dir_path, is_training_set):
        print("\nCorpus in costruzione..")
        corpus = {}
        for root, dirs, files in os.walk(dir_path):
            for d in dirs:
                corpus[d] = {}
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
        da_eliminare = []
        for p in self.vocabolario:
            if self.vocabolario[p] <= 3:
                da_eliminare.append(p)
        for p in da_eliminare:
            del self.vocabolario[p]

    def training(self, corpus):
        lunghezza_vocabolario = len(self.vocabolario)
        n_documenti_totali = 0
        for c in corpus:
            print("Classe", c, end=" ")
            lista_parole = []
            n_documenti_totali += len(corpus[c])  # numero di doc di training set totali
            for testo in corpus[c]:
                lista_parole.extend(corpus[c][testo])
            print("contiene", len(lista_parole), "parole totali (anche ripetute)")
            denominatore = len(lista_parole) + lunghezza_vocabolario
            for p in self.vocabolario:
                self.prob_parole[p, c] = np.log(((lista_parole.count(p) + 1) / denominatore))
        for c in corpus:
            self.prob_classi[c] = np.log(len(corpus[c].keys()) / n_documenti_totali)
        self.save_classifier_attributes()

    def classifica(self, testo_doc):
        p_max = -10000000000
        class_max = ""
        for cc in self.prob_classi:
            p = self.prob_classi[cc]
            for t in testo_doc:
                if t in self.vocabolario:
                    p += self.prob_parole[t, cc]
            if p > p_max:
                p_max = p
                class_max = cc
        return class_max

    def testing(self, corpus):
        tot_esatte = 0
        tot_errate = 0
        for c in corpus:
            risposte_esatte = 0
            risposte_sbagliate = 0
            print("Classe", c, end=" ")
            for documento in corpus[c]:
                class_max = self.classifica(corpus[c][documento])
                if class_max == c:
                    risposte_esatte += 1
                else:
                    risposte_sbagliate += 1
            tot_esatte += risposte_esatte
            tot_errate += risposte_sbagliate
            print("{0:6.2f}% di risposte esatte".format((100 * risposte_esatte / (
                        risposte_esatte + risposte_sbagliate))))
        return tot_esatte / (tot_esatte + tot_errate) * 100
