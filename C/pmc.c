#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// On fixe le nombre de couches a 3 : entree + 1 couche cachee + sortie
#define NB_COUCHES   3
// Taille maximale d'une couche ( 1024 neurones max par couche)
#define MAX_NEURONES 1024

// Taille de chaque couche, ex : taille[0]=2, taille[1]=4, taille[2]=1
int taille[NB_COUCHES];

// Les poids entre les neurones
// poids[i][j][k] = poids entre le neurone k (couche i-1) et le neurone j (couche i)
double poids[NB_COUCHES][MAX_NEURONES][MAX_NEURONES];

// Le biais de chaque neurone (un biais par neurone)
double biais[NB_COUCHES][MAX_NEURONES];

// La valeur de sortie de chaque neurone apres l'activation
double sortie[NB_COUCHES][MAX_NEURONES];

// L'erreur locale de chaque neurone, utilisee pendant la backpropagation
double delta[NB_COUCHES][MAX_NEURONES];


// La fonction sigmoid ecrase n'importe quelle valeur entre 0 et 1
// C'est la fonction d'activation qu'on applique a chaque neurone
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}


// Initialise le reseau avec des poids aleatoires entre -1 et 1
// nb_entrees  : nombre de neurones dans la couche d'entree
// nb_cachees  : nombre de neurones dans la couche cachee
// nb_sorties  : nombre de neurones dans la couche de sortie
void py_init(int nb_entrees, int nb_cachees, int nb_sorties) {
    srand(time(NULL)); // initialise le generateur aleatoire

    taille[0] = nb_entrees;
    taille[1] = nb_cachees;
    taille[2] = nb_sorties;

    // Pour chaque couche (on commence a 1 car la couche 0 n'a pas de poids)
    for (int i = 1; i < NB_COUCHES; i++)
        for (int j = 0; j < taille[i]; j++) {
            // Biais aleatoire entre -1 et 1
            biais[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
            // Poids aleatoires entre -1 et 1
            for (int k = 0; k < taille[i-1]; k++)
                poids[i][j][k] = ((double)rand() / RAND_MAX) * 2 - 1;
        }
}


// Le forward pass : on fait passer les donnees d'entree a travers le reseau
// pour calculer la sortie du reseau
void forward(double *entrees) {
    // Couche 0 : on copie simplement les valeurs d'entree
    for (int j = 0; j < taille[0]; j++)
        sortie[0][j] = entrees[j];

    // Pour chaque couche suivante, on calcule la sortie de chaque neurone
    for (int i = 1; i < NB_COUCHES; i++)
        for (int j = 0; j < taille[i]; j++) {
            // On commence par le biais
            double s = biais[i][j];
            // On ajoute la somme des poids * sorties de la couche precedente
            for (int k = 0; k < taille[i-1]; k++)
                s += poids[i][j][k] * sortie[i-1][k];
            // On applique sigmoid pour obtenir la sortie finale du neurone
            sortie[i][j] = sigmoid(s);
        }
}


// La backpropagation : on calcule l'erreur et on corrige les poids
void backprop(double *cibles, double alpha) {
    int last = NB_COUCHES - 1; // indice de la derniere couche

    // Etape 1 : calcul de l'erreur pour la couche de sortie
    for (int j = 0; j < taille[last]; j++) {
        double s = sortie[last][j];
        delta[last][j] = (cibles[j] - s) * s * (1 - s);
    }

    // Etape 2 : on remonte l'erreur vers les couches cachees (backpropagation)
    for (int i = last - 1; i >= 1; i--)
        for (int j = 0; j < taille[i]; j++) {
            // On somme les erreurs venant de la couche suivante
            double e = 0;
            for (int k = 0; k < taille[i+1]; k++)
                e += poids[i+1][k][j] * delta[i+1][k];
            double s = sortie[i][j];
            delta[i][j] = e * s * (1 - s);
        }

    // Etape 3 : on met a jour les poids et biais avec les erreurs calculees
    for (int i = 1; i < NB_COUCHES; i++)
        for (int j = 0; j < taille[i]; j++) {
            biais[i][j] += alpha * delta[i][j];
            for (int k = 0; k < taille[i-1]; k++)
                poids[i][j][k] += alpha * delta[i][j] * sortie[i-1][k];
        }
}


// Entraine le reseau sur tous les exemples pendant un certain nombre d'epochs
// X       : toutes les entrees mises bout a bout dans un tableau 1D
// Y       : toutes les cibles mises bout a bout dans un tableau 1D
// nb_ex   : nombre d'exemples
// nb_in   : nombre d'entrees par exemple
// nb_out  : nombre de sorties par exemple
// epochs  : nombre de fois qu'on repasse sur tous les exemples
// alpha   : taux d'apprentissage
// pertes  : tableau rempli avec l'erreur moyenne a chaque epoch (pour la courbe)
void py_train(double *X, double *Y, int nb_ex, int nb_in, int nb_out,
              int epochs, double alpha, double *pertes) {

    for (int ep = 0; ep < epochs; ep++) {
        double perte = 0;

        // On passe sur chaque exemple
        for (int ex = 0; ex < nb_ex; ex++) {
            // X + ex * nb_in : pointeur vers l'exemple numero ex
            forward(X + ex * nb_in);
            backprop(Y + ex * nb_out, alpha);

            // On calcule l'erreur quadratique pour cet exemple
            for (int j = 0; j < nb_out; j++) {
                double d = Y[ex * nb_out + j] - sortie[NB_COUCHES-1][j];
                perte += d * d;
            }
        }

        // On sauvegarde la perte moyenne de cet epoch
        pertes[ep] = perte / nb_ex;
    }
}


// Fait une prediction pour une entree donnee
// entrees    : les valeurs d'entree
// out        : tableau dans lequel on ecrit la sortie
// nb_out     : nombre de sorties
void py_predict(double *entrees, double *out, int nb_out) {
    forward(entrees);
    for (int j = 0; j < nb_out; j++)
        out[j] = sortie[NB_COUCHES-1][j];
}


// Sauvegarde tous les poids et biais dans un fichier texte
void py_sauvegarder(const char *f) {
    FILE *fp = fopen(f, "w");
    for (int i = 1; i < NB_COUCHES; i++)
        for (int j = 0; j < taille[i]; j++) {
            fprintf(fp, "%f\n", biais[i][j]);
            for (int k = 0; k < taille[i-1]; k++)
                fprintf(fp, "%f\n", poids[i][j][k]);
        }
    fclose(fp);
}


// Charge les poids et biais depuis un fichier texte
void py_charger(const char *f) {
    FILE *fp = fopen(f, "r");
    for (int i = 1; i < NB_COUCHES; i++)
        for (int j = 0; j < taille[i]; j++) {
            fscanf(fp, "%lf", &biais[i][j]);
            for (int k = 0; k < taille[i-1]; k++)
                fscanf(fp, "%lf", &poids[i][j][k]);
        }
    fclose(fp);
}