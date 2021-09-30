import Classifiers.naive_bayes


def start_menu():
    choice = 1  # int(input("Premi 1 per Classificatore Bayesiano, 2 per Classificatore k-Means: "))
    if choice == 1:
        bayes_classifiers = Classifiers.naive_bayes.NaiveBayesClassifier()
        bayes_classifiers.nb_init()
    else:
        print("2")


if __name__ == '__main__':
    start_menu()

