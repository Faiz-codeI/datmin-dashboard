const express = require("express");
const cors = require("cors");

const app = express();

app.use(cors());
app.use(express.json());

// Admin login sederhana
const ADMIN_EMAIL = "faiz@gmail.com"; // ganti
const ADMIN_PASSWORD = "12345";       // ganti

app.post("/login", (req, res) => {
    const { email, password } = req.body;

    if (email === ADMIN_EMAIL && password === ADMIN_PASSWORD) {
        return res.json({ success: true, message: "Login berhasil!" });
    }

    return res.json({ success: false, message: "Email atau password salah." });
});

app.listen(5000, () => {
    console.log("Backend Nebeng.in berjalan di port 5000");
});
