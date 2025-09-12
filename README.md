Pour exécuter le code :

1. Télécharger les documents de mon GitHub et dézipper dans le répertoire de votre choix.

2. Modifier éventuellement le fichier "Requetes.csv" pour indiquer les requêtes à tester

3. Lancer le docker : dans l'invite de commandes :

cd /d "D:\Documents\Exemple Docker"    -->    remplacer par l'emplacement du répertoire de travail

docker build -t mon_image .

docker run --name mon_conteneur -it mon_image

4. Copier-coller le fichier généré en-dehors du conteneur :

docker cp mon_conteneur:/app/Reponses.csv .

Pour supprimer le conteneur plus tard :

docker rm -f mon_conteneur
