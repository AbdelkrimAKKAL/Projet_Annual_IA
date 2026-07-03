#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <Eigen/Dense>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

// extern "C" : empeche le name mangling C++ pour que ctypes retrouve
// les fonctions par leur nom exact (create_rbf_model, train_rbf, ...)
extern "C" {

// Structure RBF
typedef struct {
    double** centers;
    double** weights;

    int      input_size;
    int      n_centers;
    int      output_size;
    double   gamma;
} RBFModel;


// 1- Creation du modele
EXPORT RBFModel* create_rbf_model(int input_size, int n_centers,
                                  int output_size, double gamma) {
    RBFModel* m = (RBFModel*)malloc(sizeof(RBFModel));
    m->input_size  = input_size;
    m->n_centers   = n_centers;
    m->output_size = output_size;
    m->gamma       = gamma;

    m->centers = (double**)malloc(n_centers * sizeof(double*));
    for (int j = 0; j < n_centers; j++)
        m->centers[j] = (double*)calloc(input_size, sizeof(double));

    m->weights = (double**)malloc(output_size * sizeof(double*));
    for (int i = 0; i < output_size; i++)
        m->weights[i] = (double*)calloc(n_centers, sizeof(double));

    return m;

}

// 2- Destruction
EXPORT void destroy_rbf_model(RBFModel* m) {
    if (m) {
        for (int j = 0; j < m->n_centers; j++) free(m->centers[j]);
        free(m->centers);
        for (int i = 0; i < m->output_size; i++) free(m->weights[i]);
        free(m->weights);

        free(m);
    }
}

// distance au carre entre deux vecteurs
static double distance2(double* a, double* b, int n) {
    double s = 0;
    for (int k = 0; k < n; k++) {
        double d = a[k] - b[k];
        s += d * d;
    }
    return s;
}

// Inversion d'une matrice carree n x n, via Eigen.
static void inverser_matrice(double* A, int n) {
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat(A, n, n);
    mat = mat.inverse();
}

// 3- k-means
EXPORT void kmeans_fit(RBFModel* m, double* X, int nb_ex, int nb_iter) {
    int K = m->n_centers, D = m->input_size;
    int* groupe = (int*)malloc(nb_ex * sizeof(int));

    // Vue 2D de la liste des images [1 .. 1024 1025 ... 2048..]
    double** mat = (double**)malloc(nb_ex * sizeof(double*));
    for (int i = 0; i < nb_ex; i++){
        mat[i] = X + i * D;
    }


    // Initialisation: les K premiers centres
    for (int j = 0; j < K; j++)
        for (int k = 0; k < D; k++)
            m->centers[j][k] = mat[j][k];

    for (int it = 0; it < nb_iter; it++) {
        // 1- assignation : chaque exemple -> centre le plus proche
        for (int i = 0; i < nb_ex; i++) {
            double best = 1e18;
            int    bj   = 0;
            for (int j = 0; j < K; j++) {
                double d = distance2(mat[i], m->centers[j], D);
                if (d < best){
                     best = d; bj = j;
                }
            }
            groupe[i] = bj;
        }
        // 2- mise a jour: chaque centre = moyenne de son groupe
        for (int j = 0; j < K; j++) {
            int cnt = 0;
            for (int k = 0; k < D; k++) m->centers[j][k] = 0;
            for (int i = 0; i < nb_ex; i++) if (groupe[i] == j) {
                cnt++;
                for (int k = 0; k < D; k++) m->centers[j][k] += mat[i][k];
            }
            if (cnt > 0)
                for (int k = 0; k < D; k++) m->centers[j][k] /= cnt;
        }
    }

    free(mat);      // free uniquement les pointeurs
    free(groupe);
}

// 4- Entrainement de la couche de sortie : W = (phiT phi)^-1 phiT Y
EXPORT void train_rbf(RBFModel* m, double* X, int* labels, int nb_ex) {
    int K = m->n_centers, C = m->output_size, D = m->input_size;

    // phi : nb_ex x K
    double** phi = (double**)malloc(nb_ex * sizeof(double*));
    for (int i = 0; i < nb_ex; i++) {
        phi[i] = (double*)malloc(K * sizeof(double));
        for (int j = 0; j < K; j++)
            phi[i][j] = exp(-m->gamma * distance2(X + i * D, m->centers[j], D));
    }

    // A = phiT phi (K x K)   et   B = phiT Y (K x C), Y one-hot
    // (matrices a plat, format ligne par ligne)
    double* A = (double*)calloc((size_t)K * K, sizeof(double));
    double* B = (double*)calloc((size_t)K * C, sizeof(double));
    int cntA = 0;
    int cntB = 0;
    for (int a = 0; a < K; a++) {
        for (int b = 0; b < K; b++) {
            double s = 0;
            for (int i = 0; i < nb_ex; i++) s += phi[i][a] * phi[i][b];
            A[cntA] = s;
            cntA++;
        }
        for (int c = 0; c < C; c++) {
            double s = 0;
            for (int i = 0; i < nb_ex; i++) {
                double y = (labels[i] == c) ? 1.0 : 0.0;
                s += phi[i][a] * y;
            }
            B[cntB] = s;
            cntB++;
        }
    }

    // Calcul de l'inverse : A = (phiT phi)^-1 
    inverser_matrice(A, K);

    // W = A^-1 . B   (K x K . K x C -> K x C)
    // weights[classe][centre] = W[centre][classe]
    for (int j = 0; j < K; j++) {          // ligne = centre
        for (int c = 0; c < C; c++) {      // colonne = classe
            double s = 0;
            for (int t = 0; t < K; t++)
                s += A[j * K + t] * B[t * C + c];
            m->weights[c][j] = s;
        }
    }


    for (int i = 0; i < nb_ex; i++) free(phi[i]);
    free(phi); free(A); free(B);
}




// 5- Entrainement complet
EXPORT void fit_rbf(RBFModel* m, double* X, int* labels, int nb_ex,
                    int kmeans_iter) {
    kmeans_fit(m, X, nb_ex, kmeans_iter);
    train_rbf(m, X, labels, nb_ex);
}

//6- Prediction
EXPORT int predict_rbf(RBFModel* m, double* features) {
    int K = m->n_centers, C = m->output_size, D = m->input_size;
    double* bosse = (double*)malloc(K * sizeof(double));

    // les bosses (distance entre l'images et les centres)
    for (int j = 0; j < K; j++){
        bosse[j] = exp(-m->gamma * distance2(features, m->centers[j], D));
    }

    int    top_index = 0;
    double max_score = -999999.0;
    for (int i = 0; i < C; i++) {
        double score = 0;
        for (int j = 0; j < K; j++)
            score += m->weights[i][j] * bosse[j];

        // Argmax
        if (i == 0 || score > max_score) {
            max_score = score;
            top_index = i;
        }
    }
    free(bosse);
    return top_index;
}

// 7- Sauvegarde du modele dans un fichier texte
EXPORT void save_rbf_model(RBFModel* m, const char* path) {
    FILE* f = fopen(path, "w");
    if (!f) return;

    fprintf(f, "%d %d %d %.17g\n", m->input_size, m->n_centers, m->output_size, m->gamma);

    for (int j = 0; j < m->n_centers; j++) {
        for (int k = 0; k < m->input_size; k++) {
            fprintf(f, "%.17g ", m->centers[j][k]);
        }
        fprintf(f, "\n");
    }

    for (int i = 0; i < m->output_size; i++) {
        for (int j = 0; j < m->n_centers; j++) {
            fprintf(f, "%.17g ", m->weights[i][j]);
        }
        fprintf(f, "\n");
    }

    fclose(f);
}

// 8- Chargement du modele depuis un fichier texte
EXPORT RBFModel* load_rbf_model(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) return NULL;

    int input_size, n_centers, output_size;
    double gamma;
    fscanf(f, "%d %d %d %lf", &input_size, &n_centers, &output_size, &gamma);

    RBFModel* m = create_rbf_model(input_size, n_centers, output_size, gamma);

    for (int j = 0; j < n_centers; j++) {
        for (int k = 0; k < input_size; k++) {
            fscanf(f, "%lf", &m->centers[j][k]);
        }
    }

    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < n_centers; j++) {
            fscanf(f, "%lf", &m->weights[i][j]);
        }
    }

    fclose(f);
    return m;
}

} // extern "C"
