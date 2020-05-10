Bonjour, 
Nous vous proposons deux versions de model.py.
La première, 'model.py', est celle qui fonctionne sur codalab. Elle ne contient pas de main et ne comporte que notre classe model à proprement parler.
La seconde, 'model_2.py', est une version approfondis, dédiée à être utilisée sur un terminal. Elle comporte donc un main, qui permettra un affichage plus lisisble de nos résultats.
Afin de fonctionner, 'model_2.py' a besoin des dossiers suivants : .ingestion_program
								   .scoring_program
								   .public_data
Vous devrez donc les placer dans le même répertoire que model_2.py.
Si vous souhaitez utiliser le modèle depuis un terminal, vous pouvez indiquer l'adresse de votre ensemble de données comme ceci : 

$ python model.py adresse

A default, le programme utilisera public_data.

Merci d'avance,
L'Equipe GREEN

PS: il est normal de retrouver les fichiers 'public_data', 'ingestion_program' et 'scoring_program' dans cette soumission, car model.py en a besoin pour certaines de ses fonctions et notamment
dans le 'main'.

