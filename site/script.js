document.addEventListener('DOMContentLoaded', () => {
    console.log('Atl4s-Forex Interface Loaded');

    /* --- NAVIGATION --- */
    const navPills = document.querySelectorAll('.nav-pill');
    const views = document.querySelectorAll('.view-section');

    navPills.forEach(pill => {
        pill.addEventListener('click', () => {
            const targetId = pill.getAttribute('data-target');

            // Update Nav State
            navPills.forEach(p => p.classList.remove('active'));
            pill.classList.add('active');

            // Update View State
            views.forEach(view => {
                view.classList.remove('active');
                if (view.id === `${targetId}-view`) {
                    // Small delay for fade effect if needed, but CSS handles opacity
                    view.classList.add('active');
                }
            });
        });
    });

    /* --- SYSTEM VIEW: CIRCULAR PROGRESS --- */
    // Simulating values for the circular charts
    const stats = [
        { selector: '.c-1', value: 75, color: '#fff' }, // Equity
        { selector: '.c-2', value: 60, color: '#fff' }, // Balance
        { selector: '.c-3', value: 85, color: '#fff' }, // Profit
        { selector: '.c-4', value: 40, color: '#fff' }  // Win Rate
    ];

    stats.forEach(stat => {
        const el = document.querySelector(stat.selector);
        if (el) {
            // Using conic-gradient for the progress ring
            // transparent center is handled by mask or inner div, but simple border-radius works too
            // Wait, border-radius makes it a circle. Conic gradient fills it.
            // To make it a "ring", we need a mask or an inner circle.
            // Let's use `background` with conic-gradient and `mask` or `::before` pseudo element.

            el.style.background = `conic-gradient(${stat.color} ${stat.value}%, rgba(255,255,255,0.1) ${stat.value}% 100%)`;

            // To make it a ring, we can use a mask or just a centered black circle (if bg is solid)
            // But our bg is transparent/glass. So we need a mask-image.
            el.style.maskImage = 'radial-gradient(transparent 55%, black 56%)';
            el.style.webkitMaskImage = 'radial-gradient(transparent 55%, black 56%)';
        }
    });

    /* --- LOGS SIMULATION --- */
    const logsContent = document.querySelector('.logs-content');
    if (logsContent) {
        // Auto scroll to bottom
        logsContent.scrollTop = logsContent.scrollHeight;
    }
});
