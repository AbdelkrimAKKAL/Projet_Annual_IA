#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// Déclaration d'exportation pour créer un .dll (Windows)
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif


// Un SVM binaire est entierement decrit par ses vecteurs de support
// (necessaire pour pouvoir utiliser un noyau non lineaire : on ne peut
// plus resumer le modele par un simple vecteur de poids W)
typedef struct {
    int nb_sv;
    double** sv_features;  // [nb_sv][input_size]
    double* sv_alpha;       // [nb_sv]
    double* sv_y;            // [nb_sv] (+1 / -1)
    double w0;
} ClassModel;

typedef struct {
    ClassModel* classes;   // [output_size], un SVM binaire par classe (one vs all)

    int input_size;
    int output_size;

    int kernel_type;   // 0 = lineaire, 1 = RBF 
    double gamma;       // parametre du noyau RBF
    double C;      // parametre de marge souple

} SVM;

// 1- Creation du model
EXPORT SVM* create_svm_model(int input_size, int output_size, int kernel_type, double gamma, double C){
    SVM* m = (SVM*)malloc(sizeof(SVM));

    m->input_size = input_size;
    m->output_size = output_size;
    m->kernel_type = kernel_type;
    m->gamma = gamma;
    m->C = C;

    m->classes = (ClassModel*)calloc(output_size, sizeof(ClassModel));

    return m;

}

// 2- Destruction du model
EXPORT void destroy_svm_model(SVM* m){
    for (int c = 0; c < m->output_size; c++){
        ClassModel* cm = &m->classes[c];
        for (int i = 0; i < cm->nb_sv; i++){
            free(cm->sv_features[i]);
        }
        free(cm->sv_features);
        free(cm->sv_alpha);
        free(cm->sv_y);
    }
    free(m->classes);
    free(m);
}

// produit scalaire entre deux lists
double dot_product(double* x1, double* x2, int nbr_features){
    double produit = 0.0;
    for (int i = 0; i < nbr_features; i++){
        produit += x1[i] * x2[i];
    }
    return produit;
}

// noyau (kernel trick) : lineaire (produit scalaire) ou RBF/gaussien
// RBF(x1,x2) = exp(-gamma * ||x1-x2||^2)
double kernel(double* x1, double* x2, int nbr_features, int kernel_type, double gamma){
    if (kernel_type == 0){
        return dot_product(x1, x2, nbr_features);
    }

    double dist2 = 0.0;
    for (int i = 0; i < nbr_features; i++){
        double diff = x1[i] - x2[i];
        dist2 += diff * diff;
    }
    return exp(-gamma * dist2);
}

// 3- Entrainement SVM binaire
// features : representation 1D d'un tableau 2D
EXPORT void train_one_svm(SVM* m, double* features, double* ybin, int nbr_ex, int classe){

    // trouver les vecteurs de support
    // Minimiser :   (1/2) * alpha^T * [ matrice ] * alpha   +   [-1 ... -1] * alpha
    // sous contraintes : 0 <= alpha <= C (marge souple) et somme(y*alpha) = 0
    // on calcule la pente pour voir la direction de l'erreur (descente de gradient)

    // 1. construire la matrice Q, avec le noyau choisi (lineaire ou RBF)
    double** bigM = malloc(nbr_ex * sizeof(double*));
    for (int i=0; i<nbr_ex; i++){
        bigM[i] = malloc(nbr_ex * sizeof(double));
        for(int j=0; j<nbr_ex; j++){
            double* x1 = features + i*m->input_size;
            double* x2 = features + j*m->input_size;
            bigM[i][j] = ybin[i] * ybin[j] * kernel(x1, x2, m->input_size, m->kernel_type, m->gamma);
        }
    }

    //2. trouver les alphas
    double* alpha = calloc(nbr_ex, sizeof(double));
    double lr = 0.01;
    int iteration = 1000;
    // gradient = Q.alpha - 1 => (N*N)x(N*1) = N*1
    double* g = calloc(nbr_ex,sizeof(double));

    for(int x =0; x<iteration; x++){
        for(int i = 0; i < nbr_ex; i++){
            double somme =0.0;
            for(int j=0; j< nbr_ex; j++){
                somme += bigM[i][j] * alpha[j];
            }
            g[i] = somme -1;
        }

        // alpha = alpha - lr*g
        for (int i=0; i<nbr_ex; i++){
            alpha[i] = alpha[i] - lr * g[i];
        }

        // projection 1 : 0 <= alpha <= C (marge souple : C borne la tolerance aux erreurs)
        for (int i=0; i<nbr_ex; i++){
            if (alpha[i] < 0) alpha[i] = 0.0;
            if (alpha[i] > m->C) alpha[i] = m->C;
        }


        // projection 2 : recentrage somme(ybin*alpha)=0
        double V = 0.0;
        for (int i=0; i<nbr_ex; i++){
            V += ybin[i] * alpha[i];
        }
        for (int i=0; i<nbr_ex; i++){
            alpha[i] = alpha[i] - (V / nbr_ex) * ybin[i];
        }
    }


    //3. ne garder que les vecteurs de support (alpha > seuil)
    int nbr_sv = 0;
    for (int i=0; i<nbr_ex; i++){
        if (alpha[i] > 1e-6) nbr_sv++;
    }
    if (nbr_sv == 0) nbr_sv = 1; // securite, ne devrait pas arriver

    ClassModel* cm = &m->classes[classe];
    cm->nb_sv = nbr_sv;
    cm->sv_features = malloc(nbr_sv * sizeof(double*));
    cm->sv_alpha = malloc(nbr_sv * sizeof(double));
    cm->sv_y = malloc(nbr_sv * sizeof(double));

    int ct = 0;
    for (int i=0; i<nbr_ex; i++){
        if (alpha[i] > 1e-6){
            cm->sv_features[ct] = malloc(m->input_size * sizeof(double));
            for (int k=0; k<m->input_size; k++){
                cm->sv_features[ct][k] = features[i*m->input_size + k];
            }
            cm->sv_alpha[ct] = alpha[i];
            cm->sv_y[ct] = ybin[i];
            ct++;
        }
    }

    //4. Calcule de w0 a partir d'un vecteur de support : w0 = 1/y_sv - somme_k(alpha_k * y_k * K(x_k, x_sv))
    double* x_sv = cm->sv_features[0];
    double y_sv = cm->sv_y[0];
    double somme = 0.0;
    for (int k=0; k<cm->nb_sv; k++){
        somme += cm->sv_alpha[k] * cm->sv_y[k] * kernel(cm->sv_features[k], x_sv, m->input_size, m->kernel_type, m->gamma);
    }
    cm->w0 = 1.0/y_sv - somme;

    // netoyage
    free(alpha);
    free(g);
    for (int i = 0; i < nbr_ex; i++) free(bigM[i]);
    free(bigM);
}


//4- Entrainement de output_size SVM (one vs all), classe = 0, 1 ou 2
EXPORT void train_svm(SVM* m, double* features, double* labels, int nbr_ex){

    double* ybin = malloc(nbr_ex * sizeof(double));

    for (int classe=0; classe< m->output_size; classe++){
        for(int j=0; j<nbr_ex; j++){
            if (labels[j] == classe){
                ybin[j] = 1.0;
            }else{
                ybin[j] = -1.0;
            }
        }

        // entrainer le svm de cette classe vs reste
        train_one_svm(m, features, ybin, nbr_ex, classe);
    }
    free(ybin);
}


// 6- Prédiction : f(x) = w0 + somme_k( alpha_k * y_k * K(x_k, x) )
EXPORT int predict_svm(SVM* m, double* image){
    double max_score = -1e300;
    int max_index = 0;

    for (int c = 0; c < m->output_size; c++){
        ClassModel* cm = &m->classes[c];
        double score = cm->w0;
        for (int k = 0; k < cm->nb_sv; k++){
            score += cm->sv_alpha[k] * cm->sv_y[k] * kernel(cm->sv_features[k], image, m->input_size, m->kernel_type, m->gamma);
        }
        if (score > max_score){
            max_score = score;
            max_index = c;
        }
    }
    return max_index;
}


// 7- Sauvegarde du modele dans un fichier texte (vecteurs de support + alphas + w0 par classe)
EXPORT void save_svm(SVM* m, const char* path){
    FILE* f = fopen(path, "w");
    if (!f) return;

    fprintf(f, "%d %d %d %.17g %.17g\n", m->input_size, m->output_size, m->kernel_type, m->gamma, m->C);

    for (int c = 0; c < m->output_size; c++){
        ClassModel* cm = &m->classes[c];
        fprintf(f, "%d %.17g\n", cm->nb_sv, cm->w0);
        for (int i = 0; i < cm->nb_sv; i++){
            fprintf(f, "%.17g %.17g ", cm->sv_alpha[i], cm->sv_y[i]);
            for (int k = 0; k < m->input_size; k++){
                fprintf(f, "%.17g ", cm->sv_features[i][k]);
            }
            fprintf(f, "\n");
        }
    }

    fclose(f);
}

// 8- Chargement du modele depuis un fichier texte
EXPORT SVM* load_svm(const char* path){
    FILE* f = fopen(path, "r");
    if (!f) return NULL;

    int input_size, output_size, kernel_type;
    double gamma, C;
    fscanf(f, "%d %d %d %lf %lf", &input_size, &output_size, &kernel_type, &gamma, &C);

    SVM* m = create_svm_model(input_size, output_size, kernel_type, gamma, C);

    for (int c = 0; c < output_size; c++){
        ClassModel* cm = &m->classes[c];
        fscanf(f, "%d %lf", &cm->nb_sv, &cm->w0);

        cm->sv_features = malloc(cm->nb_sv * sizeof(double*));
        cm->sv_alpha = malloc(cm->nb_sv * sizeof(double));
        cm->sv_y = malloc(cm->nb_sv * sizeof(double));

        for (int i = 0; i < cm->nb_sv; i++){
            fscanf(f, "%lf %lf", &cm->sv_alpha[i], &cm->sv_y[i]);
            cm->sv_features[i] = malloc(input_size * sizeof(double));
            for (int k = 0; k < input_size; k++){
                fscanf(f, "%lf", &cm->sv_features[i][k]);
            }
        }
    }

    fclose(f);
    return m;
}
