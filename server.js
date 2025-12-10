const express = require("express");
const cors = require("cors");
const path = require("path");

const app = express();

app.use(cors());
app.use(express.json());

// penting → serve folder Public
app.use(express.static(path.join(__dirname, "Public")));

// akun admin
const ADMIN_EMAIL = "admin@nebeng.in";
const ADMIN_PASS = "123456";

app.post("/login", (req, res) => {
    const { email, password } = req.body;

    if (email === ADMIN_EMAIL && password === ADMIN_PASS) {
        return res.json({
            success: true,
            message: "Login berhasil"
        });
    }

    res.json({
        success: false,
        message: "Email atau password salah"
    });
});

let drivers = [];
let passengers = [];
let requests = [];
let requestId = 1;

app.get("/drivers", (req, res) => res.json(drivers));
app.get("/passengers", (req, res) => res.json(passengers));
app.get("/requests", (req, res) => res.json(requests));

app.post("/drivers", (req, res) => {
    drivers.push(req.body);
    res.json({ success: true });
});

app.post("/passengers", (req, res) => {
    passengers.push(req.body);
    res.json({ success: true });
});

app.post("/requests", (req, res) => {
    requests.push({ id: requestId++, status: "pending", ...req.body });
    res.json({ success: true });
});

app.put("/requests/:id/approve", (req, res) => {
    const reqId = Number(req.params.id);
    const reqItem = requests.find(r => r.id === reqId);
    if (reqItem) reqItem.status = "approved";
    res.json({ success: true });
});

app.put("/requests/:id/reject", (req, res) => {
    const reqId = Number(req.params.id);
    const reqItem = requests.find(r => r.id === reqId);
    if (reqItem) reqItem.status = "rejected";
    res.json({ success: true });
});


app.listen(5000, () => {
    console.log("Server berjalan di http://localhost:5000");
});


