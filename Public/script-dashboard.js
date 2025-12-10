/* =======================================================
   SIDEBAR TOGGLE
======================================================= */
document.addEventListener("DOMContentLoaded", () => {
    const menuBtn = document.getElementById("menu-btn");
    const sidebar = document.getElementById("sidebar");
    const wrapper = document.getElementById("content");

    if (menuBtn && sidebar && wrapper) {
        menuBtn.addEventListener("click", () => {
            sidebar.classList.toggle("closed");
            wrapper.classList.toggle("shifted");
            feather.replace();
        });
    }
});

/* =======================================================
   LOAD DATA FROM SERVER
======================================================= */
async function loadData() {
    try {
        const drivers = await fetch("/drivers").then(res => res.json());
        const passengers = await fetch("/passengers").then(res => res.json());
        const requests = await fetch("/requests").then(res => res.json());

        // UPDATE STAT CARDS
        document.getElementById("driver-count").textContent = drivers.length;
        document.getElementById("passenger-count").textContent = passengers.length;
        document.getElementById("request-count").textContent = requests.length;

        let active = drivers.length + passengers.length;
        document.getElementById("active-count").textContent = active;

        // DRIVER TABLE
        const driverBody = document.querySelector("#table-driver tbody");
        driverBody.innerHTML = "";
        drivers.forEach(d => {
            driverBody.innerHTML += `
                <tr>
                    <td>${d.nama}</td>
                    <td>${d.dari}</td>
                    <td>${d.tujuan}</td>
                    <td>${d.jam}</td>
                    <td>${d.kursi}</td>
                </tr>`;
        });

        // PASSENGER TABLE
        const passengerBody = document.querySelector("#table-passenger tbody");
        passengerBody.innerHTML = "";
        passengers.forEach(p => {
            passengerBody.innerHTML += `
                <tr>
                    <td>${p.nama}</td>
                    <td>${p.dari}</td>
                    <td>${p.tujuan}</td>
                    <td>${p.jam}</td>
                </tr>`;
        });

        // REQUEST TABLE
        const requestBody = document.querySelector("#table-request tbody");
        requestBody.innerHTML = "";
        requests.forEach(r => {
            requestBody.innerHTML += `
                <tr>
                    <td>${r.nama}</td>
                    <td>${r.dari}</td>
                    <td>${r.tujuan}</td>
                    <td>${r.status}</td>
                    <td>
                        <button class="approve" onclick="approve(${r.id})">Approve</button>
                        <button class="reject" onclick="reject(${r.id})">Reject</button>
                    </td>
                </tr>`;
        });

        feather.replace();
    } catch (err) {
        console.error("Error load data:", err);
    }
}

loadData();


/* =======================================================
   REQUEST APPROVE / REJECT
======================================================= */
async function approve(id) {
    await fetch(`/requests/${id}/approve`, { method: "PUT" });
    loadData();
}

async function reject(id) {
    await fetch(`/requests/${id}/reject`, { method: "PUT" });
    loadData();
}


/* =======================================================
   PROFILE MODAL
======================================================= */
document.addEventListener("DOMContentLoaded", () => {
    const openBtn = document.getElementById("open-profile");
    const modal = document.getElementById("profile-modal");
    const closeBtn = document.getElementById("close-profile");

    if (openBtn && modal) {
        openBtn.addEventListener("click", (e) => {
            e.preventDefault();
            modal.classList.remove("hidden");
        });
    }

    if (closeBtn) {
        closeBtn.addEventListener("click", () => {
            modal.classList.add("hidden");
        });
    }

    // klik area gelap buat nutup
    modal.addEventListener("click", (e) => {
        if (e.target === modal) modal.classList.add("hidden");
    });
});


/* =======================================================
   LOGOUT
======================================================= */
function logout() {
    alert("Logout berhasil!");
    window.location.href = "web.html";
}
