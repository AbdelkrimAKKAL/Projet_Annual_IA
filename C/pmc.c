#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

// On fixe le nombre de couches a 3 : entree + 1 couche cachee + sortie
#define NB_COUCHES   3
// Taille maximale d'une couche ( 1024 neurones max par couche)
#define MAX_NEURONES 1024

int taille[NB_COUCHES];
double poids[NB_COUCHES][MAX_NEURONES][MAX_NEURONES];
double biais[NB_COUCHES][MAX_NEURONES];
double sortie[NB_COUCHES][MAX_NEURONES];
double delta[NB_COUCHES][MAX_NEURONES];

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

EXPORT void py_init(int nb_entrees, int nb_cachees, int nb_sorties) {
    srand(time(NULL));
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

void forward(double *entrees) {
    for (int j = 0; j < taille[0]; j++)
        sortie[0][j] = entrees[j];
    for (int i = 1; i < NB_COUCHES; i++)
        for (int j = 0; j < taille[i]; j++) {
            double s = biais[i][j];
            for (int k = 0; k < taille[i-1]; k++)
                s += poids[i][j][k] * sortie[i-1][k];
            sortie[i][j] = sigmoid(s);
        }
}

void backprop(double *cibles, double alpha) {
    int last = NB_COUCHES - 1;
    for (int j = 0; j < taille[last]; j++) {
        double s = sortie[last][j];
        delta[last][j] = (cibles[j] - s) * s * (1 - s);
    }
    for (int i = last - 1; i >= 1; i--)
        for (int j = 0; j < taille[i]; j++) {
            double e = 0;
            for (int k = 0; k < taille[i+1]; k++)
                e += poids[i+1][k][j] * delta[i+1][k];
            double s = sortie[i][j];
            delta[i][j] = e * s * (1 - s);
        }
    for (int i = 1; i < NB_COUCHES; i++)
        for (int j = 0; j < taille[i]; j++) {
            biais[i][j] += alpha * delta[i][j];
            for (int k = 0; k < taille[i-1]; k++)
                poids[i][j][k] += alpha * delta[i][j] * sortie[i-1][k];
        }
}

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

EXPORT void py_predict(double *entrees, double *out, int nb_out) {
    forward(entrees);
    for (int j = 0; j < nb_out; j++)
        out[j] = sortie[NB_COUCHES-1][j];
}

EXPORT void py_sauvegarder(const char *f) {
    FILE *fp = fopen(f, "w");
    for (int i = 1; i < NB_COUCHES; i++)
        for (int j = 0; j < taille[i]; j++) {
            fprintf(fp, "%f\n", biais[i][j]);
            for (int k = 0; k < taille[i-1]; k++)
                fprintf(fp, "%f\n", poids[i][j][k]);
        }
    fclose(fp);
}

EXPORT void py_charger(const char *f) {
    FILE *fp = fopen(f, "r");
    for (int i = 1; i < NB_COUCHES; i++)
        for (int j = 0; j < taille[i]; j++) {
            fscanf(fp, "%lf", &biais[i][j]);
            for (int k = 0; k < taille[i-1]; k++)
                fscanf(fp, "%lf", &poids[i][j][k]);
        }
    fclose(fp);
}