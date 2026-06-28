document.getElementById("btn").addEventListener("click", async () => {
    const imageFile = document.getElementById("image").files[0];
    const modele = document.getElementById("modele").value;

    if (!imageFile) {
        alert("Sélectionne une image d'abord.");
        return;
    }

    const formData = new FormData();
    formData.append("image", imageFile);
    formData.append("modele", modele);

    try {
        const response = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        const resultat = document.getElementById("resultat");
        const apercu = document.getElementById("apercu");
        const prediction = document.getElementById("prediction");

        apercu.src = URL.createObjectURL(imageFile);
        prediction.textContent = data.erreur
            ? `Erreur : ${data.erreur}`
            : `Prédiction (${data.modele}) : ${data.prediction}`;

        resultat.classList.remove("hidden");

    } catch (e) {
        alert("Erreur de connexion au serveur.");
    }
});