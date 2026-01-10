// script.js
document.addEventListener('DOMContentLoaded', function() {
    // Inizializza Animate On Scroll
    AOS.init({
        duration: 800,
        easing: 'ease-in-out',
        once: true
    });

    // Effetto trasparenza navbar allo scorrimento
    const navbar = document.querySelector('.navbar');
    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            navbar.classList.add('shadow');
        } else {
            navbar.classList.remove('shadow');
        }
    });
});