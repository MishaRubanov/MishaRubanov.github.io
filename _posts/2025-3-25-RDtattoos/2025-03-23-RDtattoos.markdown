---
layout: post
title:  "Reaction-Diffusion Tattoos"
date:   2025-03-23 12:00:45 -0600
categories: jekyll update
---
For the last few months, I've wanted to write a basic Turing pattern simulator as a tattoo - representing the beauty of these reaction diffusion patterns in life. I would like this to be a dot pattern as well.


<!-- Embed the tattoo.html file -->
<iframe src="/tattoo.html" width="100%" height="600" title="Tattoo Content"></iframe>

<!-- Add the slider -->
<div>
  <label for="slider">Rate this post:</label>
  <input type="range" id="slider" name="slider" min="1" max="10" value="5" oninput="updateSliderValue(this.value)">
  <span id="sliderValue">5</span>
</div>

<!-- Add JavaScript to handle the slider value -->
<script>
function updateSliderValue(value) {
  document.getElementById('sliderValue').innerText = value;
}
</script>