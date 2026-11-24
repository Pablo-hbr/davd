/* Portada con efecto parallax.
 *
 * La portada (#portada) mide varias pantallas de alto y su escena queda fija mientras se hace
 * scroll. Según el avance del scroll (0 → 1):
 *   - se mueven las capas del fondo, el texto y las figuras a distinta velocidad (solo con
 *     transform, que el navegador anima sin repintar la página);
 *   - se «encienden» figuras de personas hasta llegar a data-objetivo (las que tuvieron un
 *     evento coronario en el estudio) y el contador sube a la vez;
 *   - al final aparece la llamada a la calculadora;
 *   - mientras dura la animación, la navegación de la cabecera se oculta (clase "en-portada" en <html>)
 *     y aparece al terminarla. En las secciones sin portada se ve siempre.
 * Con «reducir movimiento» activado en el sistema se muestra directamente el estado final.
 */
(function () {
  "use strict";

  var reducido = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var orden = null; // orden (disperso, pero siempre el mismo) en que se encienden las figuras
  var pendiente = false;

  function ordenDisperso(n) {
    var indices = [];
    for (var i = 0; i < n; i++) indices.push(i);
    var semilla = 7;
    for (var j = n - 1; j > 0; j--) {
      semilla = (semilla * 9301 + 49297) % 233280;
      var k = Math.floor((semilla / 233280) * (j + 1));
      var tmp = indices[j]; indices[j] = indices[k]; indices[k] = tmp;
    }
    return indices;
  }

  function actualizar() {
    pendiente = false;
    var portada = document.getElementById("portada");
    if (!portada || portada.offsetParent === null) { // no existe o su sección está oculta
      document.documentElement.classList.remove("en-portada");
      return;
    }

    var figuras = portada.querySelectorAll(".persona-figura");
    if (!orden || orden.length !== figuras.length) orden = ordenDisperso(figuras.length);

    var recorrido = portada.offsetHeight - window.innerHeight;
    var p = reducido ? 1 : Math.min(1, Math.max(0, -portada.getBoundingClientRect().top / Math.max(recorrido, 1)));
    if (portada._p !== p) {
      portada._p = p;
      mover(portada, p);
    }

    var objetivo = parseInt(portada.dataset.objetivo || "0", 10);
    var avance = Math.min(1, Math.max(0, (p - 0.08) / 0.6));
    var encendidas = Math.round(avance * objetivo);
    if (portada._encendidas !== encendidas) {
      portada._encendidas = encendidas;
      for (var i = 0; i < orden.length; i++) {
        figuras[orden[i]].classList.toggle("activa", i < encendidas);
      }
      var contador = portada.querySelector(".portada-contador");
      if (contador) contador.textContent = encendidas;
    }
    portada.classList.toggle("empezada", p > 0.03);
    portada.classList.toggle("final", p > 0.7);
    document.documentElement.classList.toggle("en-portada", p < 0.98);
  }

  // Parallax: cada capa se desplaza a una velocidad distinta según el avance p (0 → 1).
  var CAPAS = [
    [".burbuja-1", function (p, h, w) { return "translate3d(0," + (p * 0.28 * h) + "px,0)"; }],
    [".burbuja-2", function (p, h, w) { return "translate3d(0," + (-p * 0.42 * h) + "px,0) scale(" + (1 + p * 0.25) + ")"; }],
    [".portada-texto", function (p, h, w) { return "translate3d(0," + (-p * 0.04 * h) + "px,0)"; }],
    [".portada-personas", function (p, h, w) { return "scale(" + (0.94 + p * 0.06) + ")"; }],
  ];

  function mover(portada, p) {
    var h = window.innerHeight, w = window.innerWidth;
    for (var i = 0; i < CAPAS.length; i++) {
      var el = portada.querySelector(CAPAS[i][0]);
      if (el) el.style.transform = CAPAS[i][1](p, h, w);
    }
  }

  function programar() {
    if (!pendiente) {
      pendiente = true;
      window.requestAnimationFrame(actualizar);
    }
  }

  window.addEventListener("scroll", programar, { passive: true });
  window.addEventListener("resize", programar);
  // Dash pinta el layout después de cargar este fichero (y al cambiar de sección): se revisa a menudo.
  setInterval(actualizar, 400);
})();
