#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

extern "C" {

// On fixe le nombre de couches a 3 : entree + 1 couche cachee + sortie
#define NB_COUCHES   3
// Taille maximale d'une couche (3072 = entree RGB 32x32x3)
#define MAX_NEURONES 3072

int taille[NB_COUCHES];
double poids[NB_COUCHES][MAX_NEURONES][MAX_NEURONES];
double biais[NB_COUCHES][MAX_NEURONES];
double sortie[NB_COUCHES][MAX_NEURONES];
double delta[NB_COUCHES][MAX_NEURONES];

// 0 = classification (sortie tanh, cibles -1/+1), 1 = regression (sortie lineaire, cibles reelles)
int sortie_lineaire = 0;

void py_init_commun(int nb_entrees, int nb_cachees, int nb_sorties) {
    //srand(time(NULL));
    srand(42);
    taille[0] = nb_entrees;
    taille[1] = nb_cachees;
    taille[2] = nb_sorties;
    for (int i = 1; i < NB_COUCHES; i++)
        for (int j = 0; j < taille[i]; j++) {
            biais[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
            for (int k = 0; k < taille[i-1]; k++)
                poids[i][j][k] = ((double)rand() / RAND_MAX) * 2 - 1;
        }
}

EXPORT void py_init(int nb_entrees, int nb_cachees, int nb_sorties) {
    sortie_lineaire = 0;
    py_init_commun(nb_entrees, nb_cachees, nb_sorties);
}

EXPORT void py_init_regression(int nb_entrees, int nb_cachees, int nb_sorties) {
    sortie_lineaire = 1;
    py_init_commun(nb_entrees, nb_cachees, nb_sorties);
}

void forward(double *entrees) {
    int last = NB_COUCHES - 1;
    for (int j = 0; j < taille[0]; j++)
        sortie[0][j] = entrees[j];
    for (int i = 1; i < NB_COUCHES; i++)
        for (int j = 0; j < taille[i]; j++) {
            double s = biais[i][j];
            for (int k = 0; k < taille[i-1]; k++)
                s += poids[i][j][k] * sortie[i-1][k];
            sortie[i][j] = (i == last && sortie_lineaire) ? s : tanh(s);
        }
}

void backprop(double *cibles, double alpha) {
    int last = NB_COUCHES - 1;
    for (int j = 0; j < taille[last]; j++) {
        double s = sortie[last][j];
        double derivee = sortie_lineaire ? 1.0 : (1 - s * s);
        delta[last][j] = (cibles[j] - s) * derivee;
    }
    for (int i = last - 1; i >= 1; i--)
        for (int j = 0; j < taille[i]; j++) {
            double e = 0;
            for (int k = 0; k < taille[i+1]; k++)
                e += poids[i+1][k][j] * delta[i+1][k];
            double s = sortie[i][j];
            delta[i][j] = e * (1 - s * s);
        }
    for (int i = 1; i < NB_COUCHES; i++)
        for (int j = 0; j < taille[i]; j++) {
            biais[i][j] += alpha * delta[i][j];
            for (int k = 0; k < taille[i-1]; k++)
                poids[i][j][k] += alpha * delta[i][j] * sortie[i-1][k];
        }
}

// Entrainement sur toutes les epochs d'un coup (retourne les pertes)
EXPORT void py_train(double *X, double *Y, int nb_ex, int nb_in, int nb_out,
              int epochs, double alpha, double *pertes) {
    for (int ep = 0; ep < epochs; ep++) {
        double perte = 0;
        for (int ex = 0; ex < nb_ex; ex++) {
            forward(X + ex * nb_in);
            backprop(Y + ex * nb_out, alpha);
            for (int j = 0; j < nb_out; j++) {
                double d = Y[ex * nb_out + j] - sortie[NB_COUCHES-1][j];
                perte += d * d;
            }
        }
        pertes[ep] = perte / nb_ex;
    }
}

// Entrainement sur une seule image — comme train_one_linear de mon pote
EXPORT void py_train_one(double *entree, double *cible, double alpha) {
    forward(entree);
    backprop(cible, alpha);
}

EXPORT void py_predict(double *entrees, double *out, int nb_out) {
    forward(entrees);
    for (int j = 0; j < nb_out; j++)
        out[j] = sortie[NB_COUCHES-1][j];
}

// 5- Sauvegarde du modele dans un fichier texte
EXPORT void py_sauvegarder(const char* path) {
    FILE* f = fopen(path, "w");
    if (!f) return;

    fprintf(f, "%d %d %d %d\n", taille[0], taille[1], taille[2], sortie_lineaire);

    for (int i = 1; i < NB_COUCHES; i++)
        for (int j = 0; j < taille[i]; j++)
            fprintf(f, "%.17g\n", biais[i][j]);

    for (int i = 1; i < NB_COUCHES; i++)
        for (int j = 0; j < taille[i]; j++)
            for (int k = 0; k < taille[i-1]; k++)
                fprintf(f, "%.17g ", poids[i][j][k]);

    fclose(f);
}

// 6- Chargement du modele depuis un fichier texte
EXPORT void py_charger(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) return;

    sortie_lineaire = 0;
    fscanf(f, "%d %d %d %d", &taille[0], &taille[1], &taille[2], &sortie_lineaire);

    for (int i = 1; i < NB_COUCHES; i++)
        for (int j = 0; j < taille[i]; j++)
            fscanf(f, "%lf", &biais[i][j]);

    for (int i = 1; i < NB_COUCHES; i++)
        for (int j = 0; j < taille[i]; j++)
            for (int k = 0; k < taille[i-1]; k++)
                fscanf(f, "%lf", &poids[i][j][k]);

    fclose(f);
}

} // extern "C"