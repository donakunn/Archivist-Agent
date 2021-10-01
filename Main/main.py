import Classifiers.naive_bayes


def start_menu():
    bayes_classifiers = Classifiers.naive_bayes.NaiveBayesClassifier()
    # bayes_classifiers.load_classifier_attributes()
    choice = int(input("Premi 1 per Addestrare e testare il classificatore, 2 per classificare un documento: "))
    if choice == 1:
        bayes_classifiers.nb_init()
    else:
        nomefile = str(input("Inserisci il nome del file da classificare: "))
        nomefile += '.txt'
        with open(nomefile, "r", encoding="latin-1") as file:
            lista_parole = bayes_classifiers.analizza_testo(file.read().lower())
            classmax = bayes_classifiers.classifica(lista_parole)
            print("Il documento è stato classificato come ", classmax)


if __name__ == '__main__':
    start_menu()

