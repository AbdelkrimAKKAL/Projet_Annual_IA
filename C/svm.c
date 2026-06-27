#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// Déclaration d'exportation pour créer un .dll (Windows)
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif


typedef struct {
    double** weights;
    double* w0;

    int input_size;
    int output_size;

} SVM;

// 1- Creation du model
EXPORT SVM* create_svm_model(int input_size, int output_size){
    SVM* m = (SVM*)malloc(sizeof(SVM));

    m->input_size = input_size;
    m->output_size = output_size;

    m->weights = (double**)malloc(output_size * sizeof(double*));
    for (int i =0; i< output_size; i++){
        m->weights[i] = (double*)calloc(input_size , sizeof(double));
    }

    m->w0 = (double*)calloc(output_size, sizeof(double));
    
    return m;

} 

// 2- Destruction du model
EXPORT void destroy_svm_model(SVM* m){
    for (int i=0; i<m->output_size; i++){
        free(m->weights[i]);
    }
    free(m->weights);
    free(m->w0);

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

// 3- Entrainement SVM binaire
// features : representation 1D d'un tableau 2D
EXPORT void train_one_svm(SVM*m, double* features, double* ybin, int nbr_ex, int classe){

    // trouver les vecteurs de support
    // Minimiser :   (1/2) * alpha^T * [ matrice ] * alpha   +   [-1 ... -1] * alpha
    // on calcule la pente pour voir la direction de l'erreur (descente de gradient)

    // 1. construire la matrice Q
    double** bigM = malloc(nbr_ex * sizeof(double*));
    for (int i=0; i<nbr_ex; i++){
        bigM[i] = malloc(nbr_ex * sizeof(double));
        for(int j=0; j<nbr_ex; j++){
            double* x1 = features + i*m->input_size;
            double* x2 = features + j*m->input_size;
            bigM[i][j] = ybin[i] * ybin[j] * dot_product(x1, x2, m->input_size);
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

        // projection 1 : alpha >= 0               
        for (int i=0; i<nbr_ex; i++){
            if (alpha[i] < 0) alpha[i] = 0.0;
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
       

    //3. Calcule des poids W : m->weights
    // W[k] = somme sur n de ( alpha[n] * ybin[n] * features[n][k] )
    for (int k=0; k<m->input_size; k++){          
        double somme = 0.0;
        for(int n=0; n<nbr_ex; n++){              
            somme += alpha[n] * ybin[n] * features[n*m->input_size + k];
        }
        m->weights[classe][k] = somme;            
    }

    //4. Calcule des w0 : w0 = 1/ybin[n_sv] - ( W . X[n_sv] )
    int nbr_sv = 0;
    for (int i=0; i<nbr_ex; i++){
        if (alpha[i] > 0) nbr_sv++;          
    }

    // les indices des images ou alpha > 0
    int* index_sv = malloc(nbr_sv * sizeof(int));
    int ct = 0;
    for (int i=0; i<nbr_ex; i++){
        if(alpha[i] > 1e-6){
            index_sv[ct] = i;
            ct++;
        }
    }

    // on prend le premier vecteur support pour calculer w0
    int n_sv = index_sv[0];
    double* X_sv = features + n_sv * m->input_size;
    double prod = 0.0;
    for(int j=0; j<m->input_size; j++){           // produit scalaire W . X_sv
        prod += m->weights[classe][j] * X_sv[j];
    }
    m->w0[classe] = 1.0/ybin[n_sv] - prod;        

    // netoyage
    free(index_sv);                             
    free(alpha);                                  
    free(g);                                      
    for (int i = 0; i < nbr_ex; i++) free(bigM[i]);
    free(bigM);
}


//4- Entrainement de 3 SVM (one vs all), classe = 0, 1 ou 2
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


// 6- Prédiction
EXPORT int predict_svm(SVM* m, double* image){
    double max_score = -99999.0;
    int max_index = 0;
    double* score = malloc(m->output_size * sizeof(double));

    for (int c =0; c < m->output_size; c++){
        score[c] = m->w0[c] + dot_product(m->weights[c], image, m->input_size);
    }

    for (int j=0; j<m->output_size; j++){
        if(score[j] > max_score){
            max_score = score[j];
            max_index = j;
        }
    }
    free(score);
    return max_index;
}


// 7- Sauvegarde du modele dans un fichier texte
EXPORT void save_svm(SVM* m, const char* path){
    FILE* f = fopen(path, "w");
    if (!f) return;

    fprintf(f, "%d %d\n", m->input_size, m->output_size);

    for (int i = 0; i < m->output_size; i++) {
        fprintf(f, "%.17g\n", m->w0[i]);
    }

    for (int i = 0; i < m->output_size; i++) {
        for (int j = 0; j < m->input_size; j++) {
            fprintf(f, "%.17g ", m->weights[i][j]);
        }
        fprintf(f, "\n");
    }

    fclose(f);
}

// 8- Chargement du modele depuis un fichier texte
EXPORT SVM* load_svm(const char* path){
    FILE* f = fopen(path, "r");
    if (!f) return NULL;

    int input_size, output_size;
    fscanf(f, "%d %d", &input_size, &output_size);

    SVM* m = create_svm_model(input_size, output_size);

    for (int i = 0; i < output_size; i++) {
        fscanf(f, "%lf", &m->w0[i]);
    }

    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < input_size; j++) {
            fscanf(f, "%lf", &m->weights[i][j]);
        }
    }

    fclose(f);
    return m;
}