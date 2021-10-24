import Classifiers.naive_bayes
import Search.ArchivePathSearcher


def start_menu():
    bayes_classifiers = Classifiers.naive_bayes.NaiveBayesClassifier()
    path_searcher = Search.ArchivePathSearcher.ArchivePathSearcher()
    while True:
        try:
            choice = int(input("Premi 1 per Addestrare e testare il classificatore, 2 per classificare un documento"
                               ", 0 per uscire: "))
            if choice == 1:
                bayes_classifiers.nb_init()
            elif choice == 2:
                bayes_classifiers.load_classifier_attributes()
                nomefile = str(input("Inserisci il nome del file da classificare: "))
                nomefile += '.txt'
                try:
                    with open(nomefile, "r", encoding="latin-1") as file:
                        lista_parole = bayes_classifiers.analizza_testo(file.read().lower())
                        classmax = bayes_classifiers.classifica(lista_parole)
                        print("Il documento è stato classificato come ", classmax)
                        path_searcher.archive_document(classmax, nomefile)
                except OSError:
                    print('Impossibile aprire il file specificato')

            elif choice == 0:
                break
        except Exception:
            print('Scelta non valida')


if __name__ == '__main__':
    start_menu()

