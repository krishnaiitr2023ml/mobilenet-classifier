const form = document.getElementById("uploadForm");
const fileInput = document.getElementById("imageFile");
const preview = document.getElementById("preview");
const previewWrap = document.getElementById("previewWrap");
const result = document.getElementById("result");
const statusBox = document.getElementById("status");
const button = document.getElementById("submitButton");


// Show image preview
fileInput.addEventListener("change", () => {

    const file = fileInput.files[0];

    if (!file) {
        previewWrap.classList.add("hidden");
        return;
    }

    // Client-side size check: 10 MB
    if (file.size > 10 * 1024 * 1024) {

        statusBox.textContent =
            "Please select an image smaller than 10 MB.";

        fileInput.value = "";

        previewWrap.classList.add("hidden");

        return;
    }

    statusBox.textContent = "";

    const reader = new FileReader();

    reader.onload = (event) => {

        preview.src = event.target.result;

        previewWrap.classList.remove("hidden");
    };

    reader.readAsDataURL(file);
});


// Send image to FastAPI
form.addEventListener("submit", async (event) => {

    event.preventDefault();

    const file = fileInput.files[0];

    if (!file) {

        statusBox.textContent =
            "Please select an image first.";

        return;
    }


    const formData = new FormData();

    formData.append(
        "file",
        file
    );


    button.disabled = true;

    button.textContent =
        "Classifying...";

    statusBox.textContent = "";

    result.innerHTML = "";

    result.classList.add("hidden");


    try {

        const response = await fetch(
            "/api/predict",
            {
                method: "POST",
                body: formData
            }
        );


        const contentType =
            response.headers.get("content-type") || "";


        if (!contentType.includes("application/json")) {

            throw new Error(
                "Server returned an invalid response."
            );
        }


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Prediction failed."
            );
        }


        let html = `
            <h2>Prediction Result</h2>
        `;


        data.predictions.forEach((item) => {

            html += `
                <div class="prediction">

                    <span class="class-name">
                        ${item.class}
                    </span>

                    <span class="confidence">
                        ${item.confidence}%
                    </span>

                </div>
            `;
        });


        result.innerHTML = html;

        result.classList.remove("hidden");

    }

    catch (error) {

        console.error(error);

        statusBox.textContent =
            error.message;

    }

    finally {

        button.disabled = false;

        button.textContent =
            "Classify Image";
    }
});
