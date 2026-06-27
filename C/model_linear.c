#include <stdlib.h>
#include <stdio.h>
#include <time.h>

// Déclaration d'exportation pour créer un .dll (Windows)
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

// Structure du modèle linéaire 
typedef struct {
    double** weights;
    double* biases;
    int input_size;
    int output_size;
} LinearModel;


// 1- Création du modèle 
EXPORT LinearModel* create_linear_model(int input_size, int output_size) {
    LinearModel* model = (LinearModel*)malloc(sizeof(LinearModel));
    model->input_size = input_size;
    model->output_size = output_size;
    
    model->biases = (double*)malloc(output_size * sizeof(double));
    model->weights = (double**)malloc(output_size * sizeof(double*));
    
    // Initialisation aléatoire (seed fixe pour pouvoir comparer les runs)
    srand(42);
    for (int i = 0; i < output_size; i++) {
        model->biases[i] = 0.0;
        model->weights[i] = (double*)malloc(input_size * sizeof(double));
        for (int j = 0; j < input_size; j++) {
            // Poids initiaux petits  
            model->weights[i][j] = ((double)rand() / RAND_MAX) * 0.002 - 0.001; 
        }
    }
    return model;
}

// 2- Destruction propre de la mémoire 
EXPORT void destroy_linear_model(LinearModel* model) {
    if (model) {
        for (int i = 0; i < model->output_size; i++) {
            free(model->weights[i]);
        }
        free(model->weights);
        free(model->biases);
        free(model);
    }
}

// 3- L'Entraînement sur une seule image 
EXPORT void train_one_linear(LinearModel* m, double* features, int label, double learning_rate) {
    double* scores = (double*)malloc(m->output_size * sizeof(double));
    
    // Prediction temporaire 
    for (int i = 0; i < m->output_size; i++) {
        scores[i] = m->biases[i];
        for (int j = 0; j < m->input_size; j++) {
            scores[i] += m->weights[i][j] * features[j];
        }
    }
    
    // Calcul de l'erreur et mise à jour  
    for (int i = 0; i < m->output_size; i++) {
        double target = (i == label) ? 1.0 : 0.0;
        double error = target - scores[i];

        // Si l'erreur est positive on augmente les poids => Score Eleve
        // Si l'erreur est negative on diminue les poids => Score Bas
        
        m->biases[i] += learning_rate * error;
        for (int j = 0; j < m->input_size; j++) {
            m->weights[i][j] += learning_rate * error * features[j];
        }
    }
    
    free(scores);
}

// 4- Prediction
EXPORT int predict_linear(LinearModel* m, double* features) {
    int top_index = 0;
    double max_score = -999999999.0;

    for (int i = 0; i < m->output_size; i++) {
        double score = m->biases[i];
        for (int j = 0; j < m->input_size; j++) {
            score += m->weights[i][j] * features[j];
        }

        // Argmax
        if (i == 0 || score > max_score) {
            max_score = score;
            top_index = i;
        }
    }
    return top_index;
}

// 5- Sauvegarde du modele dans un fichier texte
EXPORT void save_linear_model(LinearModel* m, const char* path) {
    FILE* f = fopen(path, "w");
    if (!f) return;

    fprintf(f, "%d %d\n", m->input_size, m->output_size);

    for (int i = 0; i < m->output_size; i++) {
        fprintf(f, "%.17g\n", m->biases[i]);
    }

    for (int i = 0; i < m->output_size; i++) {
        for (int j = 0; j < m->input_size; j++) {
            fprintf(f, "%.17g ", m->weights[i][j]);
        }
        fprintf(f, "\n");
    }

    fclose(f);
}

// 6- Chargement du modele depuis un fichier texte
EXPORT LinearModel* load_linear_model(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) return NULL;

    int input_size, output_size;
    fscanf(f, "%d %d", &input_size, &output_size);

    LinearModel* m = create_linear_model(input_size, output_size);

    for (int i = 0; i < output_size; i++) {
        fscanf(f, "%lf", &m->biases[i]);
    }

    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < input_size; j++) {
            fscanf(f, "%lf", &m->weights[i][j]);
        }
    }

    fclose(f);
    return m;
}