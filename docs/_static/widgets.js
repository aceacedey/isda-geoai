// GeoAI on Helsinki hexagons: two small widgets that need no server.
//  .frames   a slider over pictures that were drawn beforehand (docs/_static/frames/<name>/00.png, 01.png, ...)
//  .linelab  a line y = a + b x on a few made-up points, with sliders for a and b and the error of the line
(function () {
  "use strict";
  var DATA = window.GEOAI || { frames: {}, toy: {} };

  function pad(i) { return (i < 10 ? "0" : "") + i; }

  function buildFrames(box) {
    var name = box.dataset.name;
    var n = DATA.frames[name] || 0;
    if (!n) { box.textContent = "The pictures of this slider are missing (" + name + ")."; return; }
    var labels = (box.dataset.labels || "").split("|").filter(Boolean);
    var base = "_static/frames/" + name + "/";
    var bar = document.createElement("div"); bar.className = "bar";
    var label = document.createElement("label");
    var range = document.createElement("input");
    range.type = "range"; range.min = 0; range.max = n - 1; range.step = 1; range.value = 0;
    var play = document.createElement("button"); play.type = "button"; play.textContent = "Play";
    var img = document.createElement("img"); img.alt = name;
    bar.appendChild(label); bar.appendChild(range); bar.appendChild(play);
    box.appendChild(bar); box.appendChild(img);
    for (var k = 0; k < n; k++) { new Image().src = base + pad(k) + ".png"; }        // load all pictures at once, so that the slider is smooth
    function show(i) {
      i = Math.max(0, Math.min(n - 1, i));
      range.value = i;
      img.src = base + pad(i) + ".png";
      label.textContent = labels[i] || ((box.dataset.prefix || "step") + " " + (i + (+box.dataset.first || 0)));
    }
    var timer = null;
    range.addEventListener("input", function () { show(+range.value); });
    play.addEventListener("click", function () {
      if (timer) { clearInterval(timer); timer = null; play.textContent = "Play"; return; }
      if (+range.value >= n - 1) { show(0); }
      play.textContent = "Stop";
      timer = setInterval(function () {
        if (+range.value >= n - 1) { clearInterval(timer); timer = null; play.textContent = "Play"; return; }
        show(+range.value + 1);
      }, 1100);
    });
    show(0);
  }

  function buildLine(box) {
    var xs = DATA.toy.x || [], ys = DATA.toy.y || [];
    if (!xs.length) { box.textContent = "The points of this picture are missing."; return; }
    var W = 640, H = 380, L = 56, R = 16, T = 16, B = 44;
    var sx = function (x) { return L + (x / 10) * (W - L - R); };
    var sy = function (y) { return H - B - (Math.max(-5, Math.min(110, y)) / 105) * (H - T - B); };
    var NS = "http://www.w3.org/2000/svg";
    function el(tag, attrs, text) {
      var e = document.createElementNS(NS, tag);
      for (var k in attrs) { e.setAttribute(k, attrs[k]); }
      if (text !== undefined) { e.textContent = text; }
      return e;
    }
    function slider(name, min, max, step, value) {
      var bar = document.createElement("div"); bar.className = "bar";
      var label = document.createElement("label");
      var range = document.createElement("input");
      range.type = "range"; range.min = min; range.max = max; range.step = step; range.value = value;
      bar.appendChild(label); bar.appendChild(range); box.appendChild(bar);
      return { range: range, label: label, name: name };
    }
    var A = slider("a", 0, 60, 1, 40), Bs = slider("b", 0, 12, 0.5, 0);
    var bar = document.createElement("div"); bar.className = "bar";
    var best = document.createElement("button"); best.type = "button"; best.textContent = "Show the best line";
    bar.appendChild(best); box.appendChild(bar);
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img" });
    box.appendChild(svg);
    var readout = document.createElement("div"); readout.className = "readout"; box.appendChild(readout);

    svg.appendChild(el("line", { x1: L, y1: H - B, x2: W - R, y2: H - B, stroke: "#bbb" }));
    svg.appendChild(el("line", { x1: L, y1: T, x2: L, y2: H - B, stroke: "#bbb" }));
    for (var t = 0; t <= 10; t += 2) { svg.appendChild(el("text", { x: sx(t), y: H - B + 18, "text-anchor": "middle", "font-size": 12, fill: "#555" }, t)); }
    for (var v = 0; v <= 100; v += 20) { svg.appendChild(el("text", { x: L - 8, y: sy(v) + 4, "text-anchor": "end", "font-size": 12, fill: "#555" }, v)); }
    svg.appendChild(el("text", { x: (L + W - R) / 2, y: H - 8, "text-anchor": "middle", "font-size": 13, fill: "#1a1a1a" }, "x"));
    svg.appendChild(el("text", { x: 16, y: (T + H - B) / 2, "text-anchor": "middle", "font-size": 13, fill: "#1a1a1a" }, "y"));
    var strokes = xs.map(function () { var s = el("line", { stroke: "#bb16a3", "stroke-width": 1.6, opacity: 0.7 }); svg.appendChild(s); return s; });
    var line = el("line", { stroke: "#bb16a3", "stroke-width": 3.4 }); svg.appendChild(line);
    xs.forEach(function (x, i) { svg.appendChild(el("circle", { cx: sx(x), cy: sy(ys[i]), r: 5.5, fill: "#0b0b0b" })); });

    function draw() {
      var a = +A.range.value, b = +Bs.range.value, sum = 0;
      A.label.textContent = "a = " + a; Bs.label.textContent = "b = " + b;
      line.setAttribute("x1", sx(0)); line.setAttribute("y1", sy(a));
      line.setAttribute("x2", sx(10)); line.setAttribute("y2", sy(a + 10 * b));
      xs.forEach(function (x, i) {
        var p = a + b * x; sum += (p - ys[i]) * (p - ys[i]);
        strokes[i].setAttribute("x1", sx(x)); strokes[i].setAttribute("x2", sx(x));
        strokes[i].setAttribute("y1", sy(ys[i])); strokes[i].setAttribute("y2", sy(p));
      });
      readout.innerHTML = "y = <b>" + a + "</b> + <b>" + b + "</b> &times; x &nbsp;&nbsp; typical error <b>" + Math.sqrt(sum / xs.length).toFixed(1) + "</b>";
    }
    A.range.addEventListener("input", draw); Bs.range.addEventListener("input", draw);
    best.addEventListener("click", function () {
      var n = xs.length, mx = 0, my = 0, sxy = 0, sxx = 0;
      xs.forEach(function (x, i) { mx += x / n; my += ys[i] / n; });
      xs.forEach(function (x, i) { sxy += (x - mx) * (ys[i] - my); sxx += (x - mx) * (x - mx); });
      var b = sxy / sxx, a = my - b * mx;
      Bs.range.value = Math.round(b * 2) / 2; A.range.value = Math.round(my - (Math.round(b * 2) / 2) * mx);
      draw();
    });
    draw();
  }

  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll("div.frames").forEach(buildFrames);
    document.querySelectorAll("div.linelab").forEach(buildLine);
  });
})();
