from Classifiers.naive_bayes import nb_init


def start_menu():
    choice = 1  # int(input("Premi 1 per Classificatore Bayesiano, 2 per Classificatore k-Means: "))
    if choice == 1:
        nb_init()
    else:
        print("2")


if __name__ == '__main__':
    start_menu()

