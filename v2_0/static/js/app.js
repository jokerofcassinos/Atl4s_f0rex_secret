// Animation started at the end of file


// --- Socket Events ---
const socket = io();

socket.on('connect', () => {
    document.getElementById('conn-status').innerText = 'ONLINE';
    document.getElementById('conn-status').style.color = '#34A853';
    log("Connected to Neural Core.");
});

socket.on('disconnect', () => {
    document.getElementById('conn-status').innerText = 'OFFLINE';
    document.getElementById('conn-status').style.color = '#EA4335';
    log("Connection Lost.", "warning");
});

socket.on('ACCOUNT', (data) => {
    document.getElementById('equity').innerText = `$${data.equity.toFixed(2)}`;
    document.getElementById('balance').innerText = `$${data.balance.toFixed(2)}`;
    document.getElementById('profit').innerText = `$${data.profit.toFixed(2)}`;
});

socket.on('LOG', (data) => {
    log(data.message, data.level);
});

socket.on('ANALYSIS', (data) => {
    // data: { regime, modules: {...}, hurst, entropy }
    document.getElementById('regime').innerText = data.regime;
    if (data.hurst) document.getElementById('hurst').innerText = data.hurst.toFixed(2);
    if (data.entropy) document.getElementById('entropy').innerText = data.entropy.toFixed(2);

    // Update Module List
    const list = document.getElementById('module-list');
    list.innerHTML = ''; // Clear

    if (data.modules) {
        for (const [name, score] of Object.entries(data.modules)) {
            // Create Bar
            const item = document.createElement('div');
            item.className = 'module-item';

            const nameSpan = document.createElement('span');
            nameSpan.className = 'mod-name';
            nameSpan.innerText = name.toUpperCase();

            const bar = document.createElement('div');
            bar.className = 'mod-bar';

            const fill = document.createElement('div');
            fill.className = 'fill';
            fill.style.width = `${Math.min(100, Math.abs(score))}%`;

            bar.appendChild(fill);
            item.appendChild(nameSpan);
            item.appendChild(bar);
            list.appendChild(item);
        }
    }
});

// --- Helper Functions ---

function log(msg, type = 'system') {
    const term = document.getElementById('terminal');
    const line = document.createElement('div');
    line.className = `log-line ${type}`;
    line.innerText = `[${new Date().toLocaleTimeString()}] ${msg}`;
    term.appendChild(line);
    term.scrollTop = term.scrollHeight;
}

function sendCommand(cmd) {
    fetch('/command', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command: cmd })
    })
        .then(res => res.json())
        .then(data => log(`Command Sent: ${data.command}`))
        .catch(err => log(`Command Failed: ${err}`, 'warning'));
}

// Tab Switching
// Tab Switching
function switchTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
    // Deactivate all pills
    document.querySelectorAll('.nav-pill').forEach(el => el.classList.remove('active'));

    // Show target tab
    const targetTab = document.getElementById('tab-' + tabName);
    if (targetTab) targetTab.classList.add('active');

    // Activate target pill
    const indices = { 'home': 0, 'system': 1, 'neural': 2, 'logs': 3 };
    const pills = document.querySelectorAll('.nav-pill');
    if (pills[indices[tabName]]) {
        pills[indices[tabName]].classList.add('active');
    }

    // Switch Background
    const bgContainer = document.getElementById('bg-container');
    if (bgContainer) {
        bgContainer.className = 'bg-container ' + tabName;
    }
}

// --- Particle Animation (The Matrix/Neural Effect) ---
const canvas = document.getElementById('particle-canvas');
const ctx = canvas.getContext('2d');

let width, height;
let particles = [];
const particleCount = 100;
const connectionDistance = 100;

function resize() {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
}
window.addEventListener('resize', resize);
resize();

class Particle {
    constructor() {
        this.x = Math.random() * width;
        this.y = Math.random() * height;
        this.vx = (Math.random() - 0.5) * 0.5;
        this.vy = (Math.random() - 0.5) * 0.5;
        this.size = Math.random() * 2;
    }

    update() {
        this.x += this.vx;
        this.y += this.vy;

        if (this.x < 0 || this.x > width) this.vx *= -1;
        if (this.y < 0 || this.y > height) this.vy *= -1;
    }

    draw() {
        ctx.fillStyle = 'rgba(52, 168, 83, 0.5)'; // Google Green
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fill();
    }
}

// Initialize Particles
for (let i = 0; i < particleCount; i++) {
    particles.push(new Particle());
}

// Mouse Interaction
let mouse = { x: null, y: null };
window.addEventListener('mousemove', (e) => {
    mouse.x = e.x;
    mouse.y = e.y;
});

function animate() {
    ctx.clearRect(0, 0, width, height);

    particles.forEach(p => {
        p.update();
        p.draw();

        // Connect particles
        particles.forEach(p2 => {
            const dx = p.x - p2.x;
            const dy = p.y - p2.y;
            const dist = Math.sqrt(dx * dx + dy * dy);

            if (dist < connectionDistance) {
                ctx.strokeStyle = `rgba(52, 168, 83, ${1 - dist / connectionDistance})`;
                ctx.lineWidth = 0.5;
                ctx.beginPath();
                ctx.moveTo(p.x, p.y);
                ctx.lineTo(p2.x, p2.y);
                ctx.stroke();
            }
        });

        // Connect to mouse
        if (mouse.x != null) {
            const dx = p.x - mouse.x;
            const dy = p.y - mouse.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < 150) {
                ctx.strokeStyle = `rgba(52, 168, 83, ${1 - dist / 150})`;
                ctx.lineWidth = 0.8;
                ctx.beginPath();
                ctx.moveTo(p.x, p.y);
                ctx.lineTo(mouse.x, mouse.y);
                ctx.stroke();
            }
        }
    });

    requestAnimationFrame(animate);
}

// Start Animation
animate();
