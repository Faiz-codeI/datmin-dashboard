/* ================================
   OPTIONAL BUTTON DEMO
================================ */
document.addEventListener("DOMContentLoaded", () => {
    const btn = document.getElementById("btn");
    if (btn) {
        btn.addEventListener("click", () => {
            const output = document.getElementById("output");
            if (output) output.textContent = "Keren! Kamu berhasil membuat JavaScript berjalan 🎉";
        });
    }
});


/* ================================
   LOGIN
================================ */
function login() {
    const email = document.getElementById("email")?.value.trim();
    const password = document.getElementById("password")?.value.trim();

    if (!email || !password) {
        document.getElementById("login-result").textContent = "Email dan password harus diisi!";
        return;
    }

    fetch("http://localhost:5000/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password })
    })
    .then(res => res.json())
    .then(data => {
        const result = document.getElementById("login-result");
        result.textContent = data.message;

        if (data.success) {
            alert("Login berhasil!");
            window.location.href = "web.html";
        }
    })
    .catch(err => {
        console.error("Error:", err);
        document.getElementById("login-result").textContent = "Terjadi kesalahan koneksi ke server.";
    });
}

